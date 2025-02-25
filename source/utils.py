import os
import json
import yaml
import yt_dlp
import ffmpeg
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# ============
# Load/save
def load_yaml(file_path):
    """Load YAML file and return its contents as a dictionary."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)
    
def load_json(file_path):
    """Load JSON file and return its contents as a dictionary."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)
    

# ============
# Utils
def format_published_date(date_str):
    """Format string date"""
    try:
        return datetime.strptime(date_str, "%Y%m%d").strftime("%B %d, %Y")
    except ValueError:
        return "Unknown date"

def format_duration(duration):
    """Format seconds into string HH:MM:SS"""
    try:
        duration = int(duration)
        hours, remainder = divmod(duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"
    except ValueError:
        return "Unknown duration"

def load_thumbnail(url):
    """Load image"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"Error loading thumbnail: {e}")
        return np.zeros((180, 320))
    
def convert_to_script(df):
    """Convert a DataFrame into a movie script format with timestamps."""
    script = ""
    for _, row in df.iterrows():
        start_time = format_duration(row['segment_start'])
        end_time = format_duration(row['segment_end'])
        script += f"[{start_time} - {end_time}]\n{row['text']}\n\n"
    return script

def convert_to_script_with_speaker(transcript_df):
    """Convert a DataFrame into a movie script format with timestamps and speaker."""
    script = ""
    
    for _, row in transcript_df.iterrows():
        speaker = row['speaker']
        start_time = format_duration(row['start'])
        end_time = format_duration(row['end'])
        sentence = row['sentence']
        
        script += f"[{start_time} - {end_time}] {speaker}: {sentence}\n\n"
    
    return script

def markdown_to_pdf(report, output_dir, output_filename="report.pdf"):
    """Convert Markdown text to a well-styled PDF using ReportLab."""
    output_filename = os.path.join(output_dir, output_filename)
    doc = SimpleDocTemplate(output_filename, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Title'],
        fontSize=18,
        spaceAfter=12,
        textColor=colors.darkblue
    )
    
    section_style = ParagraphStyle(
        'SectionStyle',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=10,
        textColor=colors.darkred
    )

    subsection_style = ParagraphStyle(
        'SubsectionStyle',
        parent=styles['Heading3'],
        fontSize=12,
        spaceAfter=8,
        textColor=colors.black,
        fontName="Helvetica-Bold"
    )

    bullet_style = ParagraphStyle(
        'BulletStyle',
        parent=styles['BodyText'],
        leftIndent=20,
        bulletFontSize=12,
    )
    
    body_style = styles['BodyText']
    
    elements = [Paragraph("Transcript Report", title_style)]
    
    lines = report.split("\n")
    for line in lines:
        if line.startswith("# "):
            elements.append(Spacer(1, 12))
            elements.append(Paragraph(line[2:], title_style))
        elif line.startswith("## "):
            elements.append(Spacer(1, 10))
            elements.append(Paragraph(line[3:], section_style))
        elif line.startswith("### "):
            elements.append(Spacer(1, 8))
            elements.append(Paragraph(line[4:], subsection_style))
        elif line.startswith("* "):
            elements.append(Paragraph(f"• {line[2:]}", bullet_style))
        elif line.strip():
            elements.append(Paragraph(line, body_style))
    doc.build(elements)
    return output_filename

# ============
# Youtube
def get_youtube_video_metadata(video_id):
    """Fetch metadata for a given YouTube video ID."""
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        'format': 'best',
        'noplaylist': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(f'https://www.youtube.com/watch?v={video_id}', download=False)

    metadata = {
        "title": info.get("title"),
        "channel": info.get("uploader"),
        "channel_url": info.get("channel_url"),
        "thumbnail": info.get("thumbnail"),
        "views": info.get("view_count"),
        "upload_date": info.get("upload_date"),  # Format: YYYYMMDD
        "duration": info.get("duration"),  # In seconds
        "description": info.get("description"),
        "tags": info.get("tags"),
        "like_count": info.get("like_count"),
        "dislike_count": info.get("dislike_count"),
        "categories": info.get("categories"),
    }

    return metadata
    
def download_youtube_video(video_id, resolution, output_dir, format='mp4', threads=24):
    """Download a YouTube video with the specified resolution and format."""
    os.makedirs(output_dir, exist_ok=True)
    
    output_template = os.path.join(output_dir, '%(title)s.%(ext)s')

    ydl_opts = {
        'format': f'bestvideo[height<={resolution}]+bestaudio/best',
        'outtmpl': output_template,
        'merge_output_format': format,
        'n_threads': threads,
        'concurrent_fragments': threads,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(f'https://www.youtube.com/watch?v={video_id}', download=True)
        output_file_path = os.path.join(output_dir, f"{info_dict['title']}.{format}")

    return output_file_path

def convert_video_to_audio(video_file_path, output_dir, threads=16, format='wav', sample_rate=16000):
    """Convert a video file to an audio file with the specified format and sample rate using ffmpeg-python."""
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(video_file_path))[0] + f'.{format}')
    
    (
        ffmpeg
        .input(video_file_path)
        .output(output_file, format=format, ar=sample_rate, acodec='pcm_s16le' if format == 'wav' else 'libmp3lame')
        .run(overwrite_output=True)
    )
    
    return output_file

