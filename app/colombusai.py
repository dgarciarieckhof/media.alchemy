import os
import tempfile
import gradio as gr
from openai import OpenAI
from langchain_openai import ChatOpenAI
from source.asr import get_transcription
from source.llm import gen_llm_description, gen_llm_report
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from source.utils import download_youtube_video, convert_video_to_audio, get_youtube_video_metadata
from source.utils import load_yaml, format_duration, format_published_date, load_thumbnail, convert_to_script, markdown_to_pdf

# ============
# Set up
HF_TOKEN = os.environ.get("HF_TOKEN")
OPENROUTER = os.environ.get("OPENROUTER_API_KEY")
models = load_yaml("conf/config.yaml")['model']
llm_conf = load_yaml("conf/config.yaml")['llm']
options = load_yaml("conf/config.yaml")['whisper']

temp_dir = tempfile.mkdtemp()

client = OpenAI(
    api_key=OPENROUTER,
    base_url = llm_conf['base_url']
    )

model = ChatOpenAI(
    model=llm_conf['gemini'],
    api_key=OPENROUTER,
    base_url=llm_conf['base_url'],
    max_completion_tokens=250,
    temperature=0.1,
    )

# ============
# Utils
def get_video_info(videoid):
    """get metadata related to the video"""
    metadata = get_youtube_video_metadata(videoid)
    title, source, channel_url, duration, views, published, description, image = (
        metadata['title'], metadata['channel'], metadata['channel_url'], format_duration(metadata['duration']), 
        metadata['views'], format_published_date(metadata['upload_date']), metadata['description'], metadata['thumbnail'])
    image = load_thumbnail(image)

    if description=="":
        summary = f"""
        <h1>{title}</h1>
        <h2>Video Information</h2>
        <ul>
        <li><strong>Source:</strong> <a href="{channel_url}" target="_blank">{source}</a></li>
        <li><strong>Duration:</strong> {duration}</li>
        <li><strong>Views:</strong> {views:,.0f}</li>
        <li><strong>Published:</strong> {published}</li>
        </ul>
        <h2>Description</h2>
        <p>No description provided.</p>
        """
        return image, summary
    
    description = gen_llm_description(client, title, description, llm_conf['gemini'])
    summary = f"""
    <h1>{title}</h1>
    <h2>Video Information</h2>
    <ul>
    <li><strong>Source:</strong> <a href="{channel_url}" target="_blank">{source}</a></li>
    <li><strong>Duration:</strong> {duration}</li>
    <li><strong>Views:</strong> {views:,.0f}</li>
    <li><strong>Published:</strong> {published}</li>
    </ul>
    <h2>Description</h2>
    <p>{description}</p>
    """
    return image, summary

def get_video(videoid, resolution):
    """get the video"""
    video_output = download_youtube_video(videoid, resolution, output_dir=temp_dir)
    return video_output

def start_chat(video_output):
    """get documents to start the chat"""
    audio_output = convert_video_to_audio(video_output, output_dir=temp_dir, sample_rate=16000, format="wav")
    transcript = get_transcription(models['whisper'], "cuda", audio_output, options)
    transcript = transcript.groupby(["id","segment_start","segment_end"]).agg(text=("word", lambda x: " ".join(x))).reset_index(drop=False)
    transcript['text'] = transcript['text'].str.replace(r"\s+", " ", regex=True).str.strip()
    transcript = convert_to_script(transcript)
    # write transcript
    output_text = f"{temp_dir}/script.txt"
    with open(f"{temp_dir}/script.txt", "w", encoding="utf-8") as file:
        file.write(transcript)
    # create report
    report = gen_llm_report(client, transcript, llm_conf['gemini'])
    output_report = markdown_to_pdf(report, output_dir=temp_dir)
    return output_text, output_report

def chat_with_script(message, history, text_output):
    """Handles chat using OpenAI with a script as context."""
    with open(text_output, "r", encoding="utf-8") as file:
        document = file.read()
    
    system_prompt = (
        "You are Colombus, a smart AI assistant that interacts with users based on the document provided. "
        "Use the following document as context when answering queries:\n\n"
        f"{document}\n\n"
    )
    
    history_langchain_format = [SystemMessage(content=system_prompt)]

    for msg in history:
        if msg['role'] == "user":
            history_langchain_format.append(HumanMessage(content=msg['content']))
        elif msg['role'] == "assistant":
            history_langchain_format.append(AIMessage(content=msg['content']))

    history_langchain_format.append(HumanMessage(content=message))
    gpt_response = model.invoke(history_langchain_format)
    return gpt_response.content

# ============
# Gradio
with gr.Blocks(css="body { background-color: #121212; color: white; }") as app:
    gr.Markdown("# 🌎 ColombusAI: YouTube explorer", elem_id="title")

    with gr.Row():
        youtube_id = gr.Textbox(label="YouTube ID", placeholder="Enter YouTube Video ID")
        resolution = gr.Dropdown(["144p", "360p", "480p", "720p", "1080p"], value="360p", label="Resolution")

    with gr.Row():
        get_video_info_btn = gr.Button("📘 Get Video Info")
        get_video_btn = gr.Button("📺 Get Video")
        get_chat_btn = gr.Button("🤖 Start chat")

    with gr.Row():
        with gr.Column():
            gr.Markdown("## ℹ️ Info about the video")
            thumbnail_display = gr.Image(label="Thumbnail")
            video_info_display = gr.HTML()
            video_output = gr.File(label="Downloaded Video")
            report_output = gr.File(label="Downloaded Report")
            text_output = gr.File(label="Downloaded text", visible=False)
            audio_output = gr.File(label="Downloaded Audio", visible=False)

        with gr.Column():
            gr.Markdown("## 💬 Chat about the video")
            chat_interface = gr.ChatInterface(
                fn=chat_with_script,
                chatbot=gr.Chatbot(height=700),
                additional_inputs=[text_output], 
                type="messages"
            )       

    # Button Click Events
    get_video_info_btn.click(get_video_info, inputs=youtube_id, outputs=[thumbnail_display, video_info_display])
    get_video_btn.click(get_video, inputs=[youtube_id,resolution], outputs=video_output)
    get_chat_btn.click(start_chat, inputs=video_output, outputs=[text_output, report_output])

app.launch()