# FinBot

# **BabyFinBOT Beginner Financial Literacy – README**

## **Project Overview**
**BabyFinBOT** is a multimodal intelligent assistant designed to support beginner financial analytics and literacy.
It leverages YouTube videos that are particularly helpful and beginner-friendly, selected books, and classroom material.
The bot combines advanced semantic search across text (documents, transcripts, books, and classroom notes) and visual statistics (charts, graphs, tables) extracted from PDFs and YouTube videos to make financial education practical and accessible.

***

## **Features Built So Far**
- **Automated extraction of text and images from classroom PDFs**
    - Text chunks and slides indexed for search
- **Lecture video transcript extraction**
    - Converts classroom videos to transcripts ready for embedding
- **Multimodal Embedding Pipeline**
    - **Text embeddings:** [SentenceTransformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`) for documents and transcripts
    - **Image embeddings:** [CLIP](https://github.com/openai/CLIP) (ViT-B-32) for graphs, tables, charts, maps, and other informative visuals
    - **Intelligent filtering:** Only statistical images are embedded; doodles, icons, faces, logos, and irrelevant visuals are filtered with CLIP zero-shot classification
- **Efficient vector database storage**
    - Powered by [ChromaDB](https://docs.trychroma.com/) with separate collections for text and images (`text_chunks`, `image_chunks`)
    - All embeddings and metadata are persistently saved and ready for retrieval
- **Configurable extraction**
    - Size, dimension, and label-based filtering for high-quality data

***

## **Current Directory Structure**
```
BabyFinBOT/
├── README.md
├── src/
│   ├── build_index.py
│   ├── extract_pdfs.py
│   ├── embed_images.py
│   ├── embed_text.py
│   ├── extract_videos.py
├── requirements.txt
├── data/
│   ├── pdfs/
│   └── videos/
│       ├── video_urls.txt - has urls of yt videos
│       └── transcripts/
├── images/ - are extracted from pdfs, can add own extra images as well.
├── db/
│   └── chroma/
```

***

## **How to Run (Setup Instructions)**

1. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2. **Extract video transcripts (if using videos):**
    ```bash
    python src/extract_videos.py
    ```
    - This will process all classroom videos and store transcripts in `data/videos/transcripts/`.

3. **Extract and embed text and images into ChromaDB:**
    ```bash
    python src/build_index.py
    ```
    - This script indexes all text (from PDFs and transcripts) and images (graphs, charts, etc.).
    - Only relevant visuals are embedded; unwanted images are filtered.

4. **Custom filtering:**
    - Update labels in `embed_images.py` as needed to improve classification for your dataset.

***

## **Next Steps**
- Implement retrieval and query handling (user-facing Q&A, chat interface, analytics dashboard)
- Integrate multimodal search (text/image)
- Extend analytics and reporting features

***

## **Technical Highlights**
- **Hugging Face SentenceTransformers** for semantic text and transcript embedding
- **OpenAI CLIP (open-clip)** for intelligent selection and embedding of graphical/statistical content
- **ChromaDB** for scalable, persistent vector database and retrieval
- **Automated filtering** to remove irrelevant images (logos, icons, faces, doodles)

