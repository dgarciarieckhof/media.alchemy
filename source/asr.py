import gc
import time
import torch
import triton
import logging
import torchaudio
import numpy as np
import pandas as pd
import ctc_segmentation
import multiprocessing as mp
import triton.language as tl
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel, BatchedInferencePipeline
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModelForCTC

# Set up a basic logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================
# Triton kernel
@triton.autotune(
    configs=[
        triton.Config(
            kwargs=dict(
                BLOCK_SIZE_ROWS=BLOCK_SIZE_ROWS,
                num_stages=num_stages,
            ),
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for BLOCK_SIZE_ROWS in (16, 32, 64, 128)
        for num_stages in (2, 3, 4)
        for num_warps in (2, 4, 8)
    ],
    key=['N_COLS'],
)
@triton.heuristics(
    values=dict(
        BLOCK_SIZE_COLS=lambda args: triton.next_power_of_2(args['N_COLS'])
    )
)
@triton.jit
def softmax_kernel(
    input_ptr: tl.tensor,
    output_ptr: tl.tensor,
    input_row_stride: int,
    output_row_stride: int,
    n_rows: int,
    N_COLS: tl.constexpr,
    BLOCK_SIZE_ROWS: tl.constexpr,
    BLOCK_SIZE_COLS: tl.constexpr,
    num_stages: tl.constexpr
):
    input_ptr = tl.make_block_ptr(
        base=input_ptr,
        shape=(n_rows, N_COLS),
        strides=(input_row_stride, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_ROWS, BLOCK_SIZE_COLS),
        order=(1, 0),
    )

    output_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(n_rows, N_COLS),
        strides=(output_row_stride, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_ROWS, BLOCK_SIZE_COLS),
        order=(1, 0),
    )

    cols_mask = tl.arange(0, BLOCK_SIZE_COLS) < N_COLS

    row_idx = tl.program_id(0) * BLOCK_SIZE_ROWS
    in_tile_ptr = tl.advance(input_ptr, (row_idx, 0))
    row = tl.load(pointer=in_tile_ptr, boundary_check=(0, 1))

    # Subtract maximum for numerical stability
    row_minus_max = row - tl.max(row, axis=1, keep_dims=True)
    row_minus_max = tl.where(cols_mask, row_minus_max, -float('inf'))

    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=1, keep_dims=True)
    softmax_output = numerator / denominator

    out_tile_ptr = tl.advance(output_ptr, (row_idx, 0))
    tl.store(out_tile_ptr, softmax_output, boundary_check=(0, 1))


def softmax(x: torch.Tensor):
    x_orig_shape = x.shape
    x = x.view(-1, x_orig_shape[-1])
    n_rows, n_cols = x.shape

    y = torch.empty_like(x, memory_format=torch.contiguous_format)

    grid = lambda args: (
        triton.cdiv(n_rows, args['BLOCK_SIZE_ROWS']),
        1,
        1
    )

    softmax_kernel[grid](
        input_ptr=x,
        output_ptr=y,
        input_row_stride=x.stride(0),
        output_row_stride=y.stride(0),
        n_rows=n_rows,
        N_COLS=n_cols,
    )
    return y.view(*x_orig_shape)

# ==================
# ASR
def transcribe(model_size, device, audio_file, options):
    """
    Transcribe an audio file using a Whisper model and return a DataFrame.

    :param model_size: The size or variant of the Whisper model.
    :param device: The device to run the model on (e.g., 'cpu', 'cuda').
    :param audio_file: Path to the audio file to be transcribed.
    :param options: A dictionary of transcription options.
    :return: A pandas DataFrame containing word-level transcription details.
    """
    # Instantiate the model
    model = WhisperModel(model_size, device, compute_type='float16', download_root='model/')

    # Wrap the model in batched inference pipeline
    model = BatchedInferencePipeline(model=model)

    # Perform the transcription
    segments, _ = model.transcribe(audio_file, **options)

    # Collect all word-level data in rows
    rows = []
    append = rows.append

    for segment in segments:
        seg_id = segment.id
        seg_seek = segment.seek
        seg_start = segment.start
        seg_end = segment.end

        for word in segment.words:
            append((
                seg_id, seg_seek, seg_start, seg_end, word.start, word.end, word.word, word.probability,
            ))

    # Create a DataFrame from the collected rows
    transcript = pd.DataFrame(rows, columns=[ 'id', 'seek', 'segment_start', 'segment_end', 'word_start', 'word_end', 'word', 'prob'])

    # Clean up references and trigger garbage collection
    del model, rows, segments
    gc.collect()
    torch.cuda.empty_cache()

    return transcript

def get_transcription(model_size, device, audio_file, options):
    """
    Retrieve the transcription from a single-process pool
    and log how long the transcription takes.

    :param model_size: The size or variant of the Whisper model.
    :param device: The device to run the model on.
    :param audio_file: Path to the audio file to be transcribed.
    :param options: A dictionary of transcription options.
    :return: A pandas DataFrame containing the transcription.
    """
    start_time = time.time()  # Record the start time

    with mp.Pool(processes=1) as pool:
        # Asynchronously apply the transcribe function
        result = pool.apply_async(transcribe, (model_size, device, audio_file, options))
        # Block until result is ready
        transcript = result.get()

    end_time = time.time()    # Record the end time
    elapsed = end_time - start_time

    # Log or print the execution time
    logger.info("Transcription took %.2f seconds", elapsed)

    return transcript

# ==================
# CTC Forced Alignments
def ctc_alignment(audio_file, transcript, model_name, device="cuda"):
    """
    Perform forced alignment using CTC segmentation.

    :param audio_file: Path to the audio file to be transcribed.
    :transcript: A pandas DataFrame containing the transcription.
    :param model_name: Model name in Hugging Face.
    :param device: The device to run the model on.
    :return: A pandas DataFrame containing the transcription.
    """
    # load model
    model = AutoModelForCTC.from_pretrained(model_name, cache_dir='./model/')
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='./model/')
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, cache_dir='./model/')
    model.to(torch.device(device))
    
    # read audiofiles
    logging.info("Loading audiofile!")
    waveform, sample_rate = torchaudio.load(audio_file)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    target_sample_rate = 16000
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
        sample_rate = target_sample_rate
    waveform = waveform.squeeze(0) 

    input_values = feature_extractor(waveform, sampling_rate = sample_rate, return_tensors="pt").input_values
    input_values = input_values.to(torch.device(device))

    # Run inference
    logger.info("Running inference!")
    model.half()
    torch.cuda.synchronize()
    with torch.no_grad():
        logits = model(input_values.half()).logits[0]
    
    del model, input_values
    gc.collect()
    torch.cuda.empty_cache()

    logger.info("Inference Done!")
    torch.cuda.synchronize()

    logger.info("Softmax!")
    probs = softmax(logits.to(torch.float32))

    del logits
    gc.collect()
    torch.cuda.empty_cache()

    # Tokenize transcripts
    logger.info("Getting tokens!")
    vocab = tokenizer.get_vocab()
    inv_vocab = {v:k for k,v in vocab.items()}
    unk_id = vocab["<unk>"]

    trt = transcript['word'].to_list()

    tokens = []
    for word in trt:
        assert len(word) > 0
        tok_ids = tokenizer(word.replace("\n"," ").upper())['input_ids']
        tok_ids = np.array(tok_ids,dtype=np.int32)
        tokens.append(tok_ids[tok_ids != unk_id])

    # Align
    logger.info("Aligning!")
    char_list = [inv_vocab[i] for i in range(len(inv_vocab))]
    config = ctc_segmentation.CtcSegmentationParameters(char_list=char_list)
    config.index_duration = waveform.shape[0] / probs.size()[0] / sample_rate

    ground_truth_mat, utt_begin_indices = ctc_segmentation.prepare_token_list(config, tokens)
    timings, char_probs, state_list = ctc_segmentation.ctc_segmentation(config, probs.cpu().numpy(), ground_truth_mat)
    segments = ctc_segmentation.determine_utterance_segments(config, utt_begin_indices, char_probs, timings, trt)

    transcript_aligned = pd.DataFrame([{"word" : t, "word_start" : p[0], "word_end" : p[1], "conf" : p[2]} for t,p in zip(trt, segments)])
    transcript_f = pd.concat([transcript[['id','seek','prob']], transcript_aligned], axis=1)

    return transcript_f

def ctc_aligned_transcript(audio_file, transcript, model_name, device="cuda"):
    """
    Perform forced alignment using CTC segmentation.
    """
    start_time = time.time()  # Record the start time

    with mp.Pool(processes=1) as pool:
        # Asynchronously apply the transcribe function
        result = pool.apply_async(ctc_alignment, (audio_file, transcript, model_name, device))
        # Block until result is ready
        transcript = result.get()

    end_time = time.time()    # Record the end time
    elapsed = end_time - start_time

    # Log or print the execution time
    logger.info("Alignment took %.2f seconds", elapsed)

    return transcript

# ==================
# Diarize
def most_frequent(List):
    return max(set(List), key=List.count)

def speaker_diarization(model_diar, HF_TOKEN, audio_file, transcript, device="cuda"):
    """
    Perform diarization on an audio file and merge the results with a transcript DataFrame.
    
    Parameters:
        model_diar (str): The name or path of the diarization model.
        HF_TOKEN (str): The Hugging Face authentication token.
        audio_file (str): Path to the audio file to be processed.
        transcript (pd.DataFrame): The transcript DataFrame containing word-level timestamps.
        device (str): The device to use for processing ('cuda' or 'cpu').

    Returns:
        pd.DataFrame: A DataFrame with merged diarization and transcript data.
    """
    # Load model and move to device
    logging.info("Loading Pipeline")
    pipeline = Pipeline.from_pretrained(model_diar, use_auth_token=HF_TOKEN, cache_dir='./model/')
    pipeline = pipeline.to(torch.device(device))
    logging.info("Pipeline loaded!")

    # Perform diarization
    logging.info("Running diarization...")
    torch.cuda.synchronize()
    diarize_segments = pipeline(audio_file)
    logging.info("Diarization done!")
    torch.cuda.synchronize()

    # Convert to DataFrame
    logging.info("Wrangling...")
    diarize_segments = pd.DataFrame(
        diarize_segments.itertracks(yield_label=True), 
        columns=['segment', 'label', 'speaker']
    )
    diarize_segments['start'] = diarize_segments['segment'].apply(lambda x: x.start)
    diarize_segments['end'] = diarize_segments['segment'].apply(lambda x: x.end)

    # Clean up memory
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()

    # Convert timestamp columns to float
    transcript["word_start"] = transcript["word_start"].astype(float)
    transcript["word_end"] = transcript["word_end"].astype(float)
    diarize_segments["start"] = diarize_segments["start"].astype(float)
    diarize_segments["end"] = diarize_segments["end"].astype(float)

    # Merge transcript with diarization results
    logging.info("Merging diarization with transcript...")
    transcript = pd.merge_asof(
        left=transcript, 
        right=diarize_segments[["speaker", "start", "end"]], 
        left_on=['word_start'], 
        right_on=["start"]
    )

    # Fill missing speaker labels
    transcript['speaker'] = transcript['speaker'].fillna("UNK")

    # Aggregate sentence and most frequent speaker
    transcript = (
        transcript
        .groupby(['id', 'seek'])
        .agg(
            sentence=('word', lambda x: "".join(list(x)).strip()), 
            speaker=('speaker', lambda x: most_frequent(list(x))),
            start=('word_start', lambda x: list(x)[0]),
            end=('word_start', lambda x: list(x)[-1]),
        )
        .reset_index()
    )

    return transcript

def get_speakers(model_diar, HF_TOKEN, audio_file, transcript, device="cuda"):
    """
    Perform diarization on an audio file and merge the results with a transcript DataFrame.
    """
    start_time = time.time()  # Record the start time

    with mp.Pool(processes=1) as pool:
        # Asynchronously apply the transcribe function
        result = pool.apply_async(speaker_diarization, (model_diar, HF_TOKEN, audio_file, transcript, device))
        # Block until result is ready
        transcript = result.get()

    end_time = time.time()    # Record the end time
    elapsed = end_time - start_time

    # Log or print the execution time
    logger.info("Diarization took %.2f seconds", elapsed)

    return transcript