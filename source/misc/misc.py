import re
import os
import json
import tempfile
from openai import OpenAI
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate
from source.utils import load_yaml, save_json, read_txt_file
from langchain_experimental.graph_transformers import LLMGraphTransformer
from source.asr import get_transcription, ctc_aligned_transcript, get_speakers
from source.utils import download_youtube_video, convert_video_to_audio, get_youtube_video_metadata, convert_to_script_with_speaker 

load_dotenv()

# ==================
# Load env/conf
HF_TOKEN = os.environ.get("HF_TOKEN")
OPENROUTER = os.environ.get("OPENROUTER_API_KEY")
COHERE = os.environ.get("COHERE_API_KEY")
llm_conf = load_yaml("conf/config.yaml")['llm']
models = load_yaml("conf/config.yaml")['model']
options = load_yaml("conf/config.yaml")['whisper']

# Set up
device = "cuda"
resolution = "320p"
videoid = "yWgCxBdpvq8"

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
transcript = get_transcription(model_size, device, audio_file, options)
transcript_a = ctc_aligned_transcript(audio_file, transcript, model_name, device)
transcript_f = get_speakers(model_diar, HF_TOKEN, audio_file, transcript, device)

filename = audio_file.split("/")[-1].split(".wav")[0]
script = convert_to_script_with_speaker(transcript_f)

# ==================
# Knowledge extraction
text_file = "data/raw/text/Session 1 - How To Start (Almost) Any Project.txt"
script = read_txt_file(text_file)
