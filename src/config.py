import os

# Core project directories (relative to project root)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

DATA_DIR = os.path.join(BASE_DIR, "data")
PDFS_DIR = os.path.join(DATA_DIR, "pdfs")
VIDEOS_DIR = os.path.join(DATA_DIR, "videos")

TRANSCRIPTS_DIR = os.path.join(VIDEOS_DIR, "transcripts")
IMG_DIR = os.path.join(BASE_DIR, "images")

CHROMA_DIR = os.path.join(BASE_DIR, "db", "chroma")

# Path to video URLs file
VIDEO_URLS_FILE = os.path.join(VIDEOS_DIR, "video_urls.txt")

# API keys, etc.
OPENAI_API_KEY = "your-api-key"  # Optional: use dotenv for security!
