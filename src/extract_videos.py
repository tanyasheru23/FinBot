import os
import time
from config import VIDEO_URLS_FILE, TRANSCRIPTS_DIR

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
import yt_dlp
import whisper

def get_youtube_id(url):
    import re
    match = re.search(r'(?:v=|youtu.be/)([A-Za-z0-9_-]{11})', url)
    return match.group(1) if match else None

def fetch_transcript_api(video_id, lang_pref=['en']):
    for lang in lang_pref + ['en']:
        try:
            transcript = YouTubeTranscriptApi().fetch(video_id, languages=[lang])
            return '\n'.join([t.text for t in transcript])
        except (TranscriptsDisabled, NoTranscriptFound):
            continue
    return None

def download_audio_ytdlp(url, out_path):
    ydl_opts = {
        'format': 'mp3/bestaudio/best',
        'outtmpl': out_path,
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def fetch_transcript_whisper(audio_path, lang="en"):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, language=lang)
    return result["text"]

def extract_all_transcripts(video_urls_file, transcripts_dir):
    os.makedirs(transcripts_dir, exist_ok=True)
    with open(video_urls_file, "r") as f:
        urls = [line.strip() for line in f if line.strip()]
    for url in urls:
        vid_id = get_youtube_id(url)
        transcript_path = os.path.join(transcripts_dir, f"{vid_id}.txt")
        if os.path.exists(transcript_path):
            continue  # Already processed
        print(f"Processing: {url}")
        # Try official API first
        transcript = fetch_transcript_api(vid_id)
        if transcript is None:
            print("No API transcript—using Whisper...")
            audio_file = os.path.join(transcripts_dir, f"{vid_id}.mp3")
            download_audio_ytdlp(url, audio_file)
            transcript = fetch_transcript_whisper(audio_file)
            os.remove(audio_file)  # Clean up
        with open(transcript_path, "w", encoding="utf-8") as out_f:
            out_f.write(transcript)
        print(f"Saved transcript for {vid_id}")
        time.sleep(1)  # Be nice to servers

if __name__ == "__main__":
    extract_all_transcripts(VIDEO_URLS_FILE, TRANSCRIPTS_DIR)
