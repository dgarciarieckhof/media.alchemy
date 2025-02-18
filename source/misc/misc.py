import os
import tempfile
from dotenv import load_dotenv
from source.asr import get_transcription, ctc_aligned_transcript, get_speakers
from source.utils import download_youtube_video, convert_video_to_audio, load_yaml, get_youtube_video_metadata

load_dotenv()

# ==================
# Load env/conf
HF_TOKEN = os.environ.get("HF_TOKEN")
models = load_yaml("conf/config.yaml")['model']
options = load_yaml("conf/config.yaml")['whisper']

# Set up
device = "cuda"
resolution = "360p"
videoid = "sI5b655zPCs"

model_size = models['whisper']
model_name = models['ctc']
model_diar = models['diarization']

# ==================
# Download video and extract audio
temp_dir = tempfile.mkdtemp()
video_file = download_youtube_video(videoid, resolution, output_dir=temp_dir, format="mp4")
audio_file = convert_video_to_audio(video_file, output_dir=temp_dir, sample_rate=16000, format="wav")

# ==================
# Test ASR
transcript = get_transcription(model_size, device, audio_file, options)
transcript_a = ctc_aligned_transcript(audio_file, transcript, model_name, device)
transcript_f = get_speakers(model_diar, HF_TOKEN, audio_file, transcript_a, device)

# Get metadata
metadata = get_youtube_video_metadata(videoid)
metadata['description']

# ==================
# Knowledge extraction