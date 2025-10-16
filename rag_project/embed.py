# embed.py
import os
import json
import boto3
from dotenv import load_dotenv
from pypdf import PdfReader  # pip install pypdf if missing
import chromadb
from chromadb.config import Settings

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")  # Add this to .env
LLM_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")  # For reference

bedrock_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
# Chroma client (persistent for local dev; adjust path if needed)
client = chromadb.PersistentClient(path="./chroma_db")  # Creates ./chroma_db dir
collection = client.get_or_create_collection(name="pdf_docs")

# ----------------------
# Embedding function
# ----------------------
def get_embedding(text: str) -> list:
    """
    Get embedding from Bedrock (Titan Embeddings).
    Returns list of floats.
    """
    if not text.strip():
        return [0.0] * 1536  # Dummy zero-vector for empty text (adjust dim if using different model)

    body = json.dumps({
        "inputText": text
    })
    try:
        resp = bedrock_client.invoke_model(
            modelId=EMBEDDING_MODEL_ID,
            body=body,
            contentType="application/json",
            accept="application/json"
        )
        # <-- FIXED: Always json.loads() the body—don't skip this!
        response_body = json.loads(resp["body"].read().decode("utf-8"))
        return response_body["embedding"]  # List of floats
    except Exception as e:
        print(f"❌ Embedding error: {e}")
        raise

# ----------------------
# Build / update Chroma index
# ----------------------

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from PDF using pypdf (reliable for most PDFs).
    """
    try:
        reader = PdfReader(file_path)
        text = ""
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:  # Skip empty pages
                text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
        return text.strip()
    except Exception as e:
        print(f"❌ PDF extraction error for {file_path}: {e}")
        raise

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Simple overlapping chunking.
    """
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

def build_or_update_index(file_paths: list[str]):
    """
    Build/update Chroma index from PDF files.
    Extracts text → chunks → embeds → adds to collection.
    """
    for file_path in file_paths:
        print(f"Processing {file_path}...")
        # Extract text
        full_text = extract_text_from_pdf(file_path)
        
        # Chunk
        chunks = chunk_text(full_text)
        if not chunks:
            print(f"No text extracted from {file_path}")
            continue
        
        # Embed chunks
        embeddings = []
        for i, chunk in enumerate(chunks):
            emb = get_embedding(chunk)  # <-- This now works—no string index error
            embeddings.append(emb)
        
        # Metadatas & IDs
        metadatas = [{"file": os.path.basename(file_path), "chunk_id": i} for i in range(len(chunks))]
        ids = [f"{os.path.basename(file_path)}_chunk_{i}" for i in range(len(chunks))]
        
        # Add to collection (upserts if ID exists)
        collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Added {len(chunks)} chunks from {file_path}")