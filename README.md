# Media Alchemy
Media Alchemy is a hands-on toolkit for exploring and building Automatic Speech Recognition (ASR) systems. It provides practical tools to experiment with CTC alignment, speech-to-text conversion, speaker diarization, and LLM-enhanced interactions. This repository helps organize my learning about these models and methods.

## Features
- **Speech-to-Text (STT)** – Convert spoken language into text using pre-trained or fine-tuned ASR models.
- **CTC Alignment** – Explore Connectionist Temporal Classification (CTC) for aligning text and speech without pre-segmented data.
- **Speaker Diarization** – Identify and segment speakers in multi-speaker recordings.
- **LLM Interactions** – Use Large Language Models (LLMs) for context-aware transcription, summarization, and Q&A from transcribed audio.
- **Integrations** – Use simple interfaces to integrate ASR into applications or interact via the command line.

## Installation  
```sh
git clone https://github.com/dgarciarieckhof/media-alchemy.git
cd media-alchemy

pip install uv
uv sync
```

## Integrations
- **ColumbusAI**: A Gradio app for YouTube exploration that allows you to interact with videos, download them, generate reports, and chat with them.  
  👉 Check out the [demo](https://www.youtube.com/watch?v=WyYV0tILzu0).

## Misc
- Using cuda 12.4 and cuDNN9.
- You will need to set up a Hugging Face token, and Open router api key or similar.
- [Workaround: Isolate CTranslate2 process using multiprocessing](https://github.com/m-bain/whisperX/issues/1027).