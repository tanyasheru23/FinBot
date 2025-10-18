import os
from config import DATA_DIR, IMG_DIR, TRANSCRIPTS_DIR, CHROMA_DIR, PDFS_DIR
from extract_pdfs import batch_extract
from embed_text import embed_text_chunk
from embed_image import embed_image

def load_video_transcripts(transcripts_dir):
    video_chunks = []
    for fname in os.listdir(transcripts_dir):
        if fname.endswith('.txt'):
            path = os.path.join(transcripts_dir, fname)
            vid_id = os.path.splitext(fname)[0]
            with open(path, "r", encoding="utf-8") as f:
                transcript = f.read()
            # Chunk if very long
            chunk_size = 1000
            transcript_chunks = [transcript[i:i+chunk_size] for i in range(0, len(transcript), chunk_size)]
            for i, chunk in enumerate(transcript_chunks):
                video_chunks.append({
                    "text": chunk,
                    "source": vid_id,
                    "page_num": i,
                    "chunk_type": "video"
                })
    return video_chunks

def main():
    chunks = batch_extract(PDFS_DIR, IMG_DIR)
    video_chunks = load_video_transcripts(TRANSCRIPTS_DIR)
    all_chunks = chunks + video_chunks
    
    # Initialize ChromaDB
    import chromadb
    from chromadb.config import Settings
    # client = chromadb.Client(Settings(
    #     persist_directory=CHROMA_DIR,
    #     chroma_db_impl="duckdb+parquet"
    # ))
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    text_collection = client.get_or_create_collection(name="text_chunks", embedding_function=None)
    image_collection = client.get_or_create_collection(name="image_chunks", embedding_function=None)


    for chunk in all_chunks:
        if chunk.get("text") and len(chunk["text"].strip()) > 0:
            text_emb = embed_text_chunk(chunk["text"])
            meta = {
                "source": chunk.get("source"),
                "page_num": chunk.get("page_num"),
                "chunk_type": chunk.get("chunk_type", "text")
            }
            text_collection.add(
                ids=[f"{meta['source']}_{meta['chunk_type']}_{meta['page_num']}"],   # create a unique ID!
                embeddings=[text_emb],
                metadatas=[meta],
                documents=[chunk["text"]]
            )
            
        for img_path in chunk.get("images", []):
            try:
                img_emb = embed_image(img_path)
                meta = {
                    "source": chunk["source"],
                    "page_num": chunk["page_num"],
                    "chunk_type": "image",
                    "image_path": img_path
                }
                image_collection.add(
                    ids=[f"{meta['source']}_img_{meta['page_num']}_{os.path.basename(img_path)}"],
                    embeddings=[img_emb],
                    metadatas=[meta],
                    documents=[""]
                )
                
            except Exception as e:
                print(f"Image {img_path} embedding failed: {e}")

    print("DB population complete.")

if __name__ == "__main__":
    main()
