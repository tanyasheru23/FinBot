# from langchain_community.embeddings import OpenAIEmbeddings
# from config import OPENAI_API_KEY

# def embed_text_chunk(text):
#     embedder = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
#     return embedder.embed_query(text)

# Alternative using SentenceTransformers
from sentence_transformers import SentenceTransformer

# Load a popular, efficient free model (downloads automatically)
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text_chunk(text):
    # Returns a list/array. For chromadb, use .tolist() if needed
    return model.encode(text).tolist()
