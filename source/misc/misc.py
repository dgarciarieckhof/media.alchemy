import os
import tempfile
from dotenv import load_dotenv
from source.asr import get_transcription, ctc_aligned_transcript, get_speakers
from source.utils import download_youtube_video, convert_video_to_audio, load_yaml, get_youtube_video_metadata, convert_to_script_with_speaker

load_dotenv()

# ==================
# Load env/conf
HF_TOKEN = os.environ.get("HF_TOKEN")
models = load_yaml("conf/config.yaml")['model']
options = load_yaml("conf/config.yaml")['whisper']

# Set up
device = "cuda"
resolution = "720p"
videoid = "0mnjKLOJOz8"

model_size = models['whisper']
model_name = models['ctc']
model_diar = models['diarization']

# ==================
# Download video and extract audio
temp_dir = tempfile.mkdtemp()
video_file = download_youtube_video(videoid, resolution, output_dir="data/raw/video", format="mp4")
audio_file = convert_video_to_audio(video_file, output_dir="data/raw/audio", sample_rate=16000, format="wav")

# ==================
# Test ASR
audio_file = "data/raw/audio/Session 6 - How To Build Continual Learning Systems.wav"

transcript = get_transcription(model_size, device, audio_file, options)
transcript_a = ctc_aligned_transcript(audio_file, transcript, model_name, device)
transcript_f = get_speakers(model_diar, HF_TOKEN, audio_file, transcript, device)

filename = audio_file.split("/")[-1].split(".wav")[0]
script = convert_to_script_with_speaker(transcript_f)

with open(f"data/raw/text/{filename}.txt", "w", encoding="utf-8") as file:
    file.write(script)

# ==================
# Knowledge extraction