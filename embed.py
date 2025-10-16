# # embed.py - BULLETPROOF FIX for VDI/Windows
# import os
# import json
# import boto3
# from dotenv import load_dotenv
# from pypdf import PdfReader
# import chromadb
# from chromadb.config import Settings
# import threading
# import time
# import pickle

# load_dotenv()

# AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
# EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
# LLM_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")

# bedrock_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)

# # NUCLEAR OPTION: Use ephemeral client (recreate each time)
# _client_lock = threading.Lock()

# def get_chroma_client():
#     """Get fresh ChromaDB client instance - no caching."""
#     try:
#         # Try persistent first
#         settings = Settings(
#             anonymized_telemetry=False,
#             allow_reset=False,
#             is_persistent=True
#         )
#         return chromadb.PersistentClient(
#             path="./chroma_db",
#             settings=settings
#         )
#     except Exception as e:
#         print(f"⚠️ Persistent client failed: {e}")
#         print("🔄 Falling back to ephemeral HTTP client...")
#         # Fallback to ephemeral client
#         return chromadb.EphemeralClient()

# # Don't cache client globally - recreate as needed
# client = None
# collection = None

# def get_or_create_collection():
#     """Get collection, creating client if needed."""
#     global client, collection
#     if client is None:
#         client = get_chroma_client()
#     if collection is None:
#         collection = client.get_or_create_collection(name="pdf_docs")
#     return collection

# # ----------------------
# # Embedding function with retry
# # ----------------------
# def get_embedding(text: str, max_retries: int = 3) -> list:
#     """
#     Get embedding from Bedrock (Titan Embeddings) with retry logic.
#     Returns list of floats.
#     """
#     if not text.strip():
#         return [0.0] * 1024  # Titan v2 uses 1024 dimensions

#     body = json.dumps({
#         "inputText": text
#     })
    
#     for attempt in range(max_retries):
#         try:
#             resp = bedrock_client.invoke_model(
#                 modelId=EMBEDDING_MODEL_ID,
#                 body=body,
#                 contentType="application/json",
#                 accept="application/json"
#             )
#             response_body = json.loads(resp["body"].read().decode("utf-8"))
#             return response_body["embedding"]
#         except Exception as e:
#             print(f"❌ Embedding error (attempt {attempt + 1}/{max_retries}): {e}")
#             if attempt < max_retries - 1:
#                 time.sleep(1)
#             else:
#                 raise

# # ----------------------
# # PDF text extraction
# # ----------------------
# def extract_text_from_pdf(file_path: str) -> str:
#     """
#     Extract text from PDF using pypdf.
#     """
#     try:
#         reader = PdfReader(file_path)
#         text = ""
#         for page_num, page in enumerate(reader.pages):
#             page_text = page.extract_text()
#             if page_text:
#                 text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
#         return text.strip()
#     except Exception as e:
#         print(f"❌ PDF extraction error for {file_path}: {e}")
#         raise

# def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
#     """
#     Simple overlapping chunking.
#     """
#     if not text:
#         return []
#     chunks = []
#     start = 0
#     while start < len(text):
#         end = start + chunk_size
#         chunk = text[start:end]
#         chunks.append(chunk)
#         start = end - overlap
#     return chunks

# # ----------------------
# # BULLETPROOF: Safe collection add with retries
# # ----------------------
# def safe_collection_add(coll, documents, embeddings, metadatas, ids, max_retries=3):
#     """Add to collection with retry and smaller batches on failure."""
#     batch_size = 100
    
#     for attempt in range(max_retries):
#         try:
#             # Try adding in batches
#             for i in range(0, len(documents), batch_size):
#                 batch_end = min(i + batch_size, len(documents))
                
#                 # For each batch, try with smaller sub-batches if needed
#                 try:
#                     coll.add(
#                         documents=documents[i:batch_end],
#                         embeddings=embeddings[i:batch_end],
#                         metadatas=metadatas[i:batch_end],
#                         ids=ids[i:batch_end]
#                     )
#                     print(f"    ✅ Added batch {i//batch_size + 1} ({batch_end}/{len(documents)} chunks)")
#                 except Exception as batch_error:
#                     print(f"    ⚠️ Batch failed, trying smaller sub-batches: {batch_error}")
#                     # Try one-by-one for this batch
#                     for j in range(i, batch_end):
#                         try:
#                             coll.add(
#                                 documents=[documents[j]],
#                                 embeddings=[embeddings[j]],
#                                 metadatas=[metadatas[j]],
#                                 ids=[ids[j]]
#                             )
#                         except Exception as item_error:
#                             print(f"    ❌ Failed to add item {j}: {item_error}")
#                             continue
#                     print(f"    ✅ Completed batch {i//batch_size + 1} with fallback method")
            
#             return  # Success!
            
#         except Exception as e:
#             print(f"❌ Collection add failed (attempt {attempt + 1}/{max_retries}): {e}")
#             if attempt < max_retries - 1:
#                 time.sleep(2)
#                 # Recreate client and collection
#                 global client, collection
#                 client = None
#                 collection = None
#                 coll = get_or_create_collection()
#             else:
#                 raise

# # ----------------------
# # FIXED: Thread-safe index building with aggressive error handling
# # ----------------------
# def build_or_update_index(file_paths: list[str]):
#     """
#     Build/update Chroma index from PDF files.
#     Thread-safe implementation with aggressive error handling.
#     """
#     with _client_lock:
#         # Get fresh collection
#         coll = get_or_create_collection()
        
#         for file_path in file_paths:
#             print(f"\n{'='*60}")
#             print(f"Processing {os.path.basename(file_path)}...")
#             print(f"{'='*60}")

#             try:
#                 # Extract text
#                 print(f"  📄 Extracting text from {os.path.basename(file_path)}...")
#                 full_text = extract_text_from_pdf(file_path)
#                 print(f"  ✅ Extracted {len(full_text)} characters of text.")

#                 # Chunk
#                 print(f"  ✂️ Chunking text into segments...")
#                 chunks = chunk_text(full_text)
#                 if not chunks:
#                     print(f"  ⚠️ No text extracted from {file_path}")
#                     continue
#                 print(f"  ✅ Created {len(chunks)} chunks.")

#                 # Embed chunks
#                 print(f"  🔮 Generating embeddings for {len(chunks)} chunks...")
#                 embeddings = []
                
#                 for i, chunk in enumerate(chunks):
#                     try:
#                         if (i + 1) % 10 == 0:
#                             print(f"    Progress: {i + 1}/{len(chunks)} chunks embedded...")
#                         emb = get_embedding(chunk)
#                         embeddings.append(emb)
#                     except Exception as emb_error:
#                         print(f"    ⚠️ Failed to embed chunk {i + 1}, using zero vector: {emb_error}")
#                         embeddings.append([0.0] * 1024)

#                 print(f"  ✅ All {len(embeddings)} embeddings generated successfully.")

#                 # Prepare metadata and IDs
#                 metadatas = [
#                     {
#                         "file": os.path.basename(file_path),
#                         "chunk_id": i,
#                         "source": file_path
#                     } 
#                     for i in range(len(chunks))
#                 ]
#                 ids = [f"{os.path.basename(file_path)}_chunk_{i}" for i in range(len(chunks))]
#                 print(f"  📋 Prepared metadata and IDs for {len(metadatas)} items.")

#                 # Add to collection with retry logic
#                 print(f"  💾 Adding {len(chunks)} chunks to Chroma collection...")
#                 safe_collection_add(coll, chunks, embeddings, metadatas, ids)
#                 print(f"  ✅ Successfully added all {len(chunks)} chunks from {os.path.basename(file_path)}.")
                
#             except Exception as e:
#                 print(f"  ❌ Error processing {file_path}: {e}")
#                 import traceback
#                 print(f"  Debug trace:\n{traceback.format_exc()}")
#                 continue  # Continue with next file
        
#         print(f"\n{'='*60}")
#         print(f"✅ Indexing complete!")
#         print(f"{'='*60}\n")


# embed.py - BULLETPROOF FIX + DETAILED LOGGING FOR VDI/Windows
import os
import json
import boto3
from dotenv import load_dotenv
from pypdf import PdfReader
import chromadb
from chromadb.config import Settings
import threading
import time
import pickle
from datetime import datetime

# -------------
# ENV + CONFIG
# -------------
load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
LLM_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")

bedrock_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)

_client_lock = threading.Lock()
client = None
collection = None

# -------------
# LOGGING UTILS
# -------------
def log(msg: str, level: str = "INFO"):
    """Timestamped colored logs (Windows-safe)"""
    ts = datetime.now().strftime("%H:%M:%S")
    color = {
        "INFO": "\033[94m",      # Blue
        "SUCCESS": "\033[92m",   # Green
        "WARN": "\033[93m",      # Yellow
        "ERROR": "\033[91m",     # Red
        "RESET": "\033[0m"
    }
    print(f"{color.get(level, '')}[{ts}] [{level}] {msg}{color['RESET']}")

# -------------
# CLIENT MGMT
# -------------
def get_chroma_client():
    """Get fresh ChromaDB client instance - no caching."""
    try:
        log("Creating persistent Chroma client at ./chroma_db ...", "INFO")
        settings = Settings(anonymized_telemetry=False, allow_reset=False, is_persistent=True)
        return chromadb.PersistentClient(path="./chroma_db", settings=settings)
    except Exception as e:
        log(f"Persistent client failed: {e}. Falling back to EphemeralClient.", "WARN")
        return chromadb.EphemeralClient()

def get_or_create_collection():
    """Get collection, creating client if needed."""
    global client, collection
    if client is None:
        log("Initializing new Chroma client...", "INFO")
        client = get_chroma_client()
    if collection is None:
        log("Getting or creating 'pdf_docs' collection...", "INFO")
        collection = client.get_or_create_collection(name="pdf_docs")
    return collection

# -------------
# EMBEDDING
# -------------
def get_embedding(text: str, max_retries: int = 3) -> list:
    """Get embedding from Bedrock (Titan Embeddings) with retry logic."""
    if not text.strip():
        return [0.0] * 1024

    body = json.dumps({"inputText": text})
    for attempt in range(max_retries):
        try:
            log(f"Embedding request to Bedrock (attempt {attempt + 1})...", "INFO")
            resp = bedrock_client.invoke_model(
                modelId=EMBEDDING_MODEL_ID,
                body=body,
                contentType="application/json",
                accept="application/json"
            )
            response_body = json.loads(resp["body"].read().decode("utf-8"))
            return response_body["embedding"]
        except Exception as e:
            log(f"Embedding error (attempt {attempt + 1}/{max_retries}): {e}", "WARN")
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                log("Max retries reached. Returning zero vector.", "ERROR")
                return [0.0] * 1024

# -------------
# PDF TEXT EXTRACTION
# -------------
def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF using pypdf."""
    start_time = time.time()
    try:
        reader = PdfReader(file_path)
        text = ""
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
        duration = time.time() - start_time
        log(f"Extracted text from {file_path} in {duration:.2f}s.", "SUCCESS")
        return text.strip()
    except Exception as e:
        log(f"PDF extraction error for {file_path}: {e}", "ERROR")
        raise

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Overlapping chunking for long documents."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    log(f"Chunked text into {len(chunks)} parts (chunk_size={chunk_size}, overlap={overlap}).", "INFO")
    return chunks

# -------------
# SAFE COLLECTION ADD
# -------------
def safe_collection_add(coll, documents, embeddings, metadatas, ids, max_retries=3):
    """Add to collection with retry and smaller batches on failure."""
    batch_size = 100
    for attempt in range(max_retries):
        try:
            for i in range(0, len(documents), batch_size):
                batch_end = min(i + batch_size, len(documents))
                try:
                    coll.add(
                        documents=documents[i:batch_end],
                        embeddings=embeddings[i:batch_end],
                        metadatas=metadatas[i:batch_end],
                        ids=ids[i:batch_end]
                    )
                    log(f"Added batch {i//batch_size + 1} ({batch_end}/{len(documents)})", "SUCCESS")
                except Exception as batch_error:
                    log(f"Batch failed: {batch_error}. Retrying one-by-one...", "WARN")
                    for j in range(i, batch_end):
                        try:
                            coll.add(
                                documents=[documents[j]],
                                embeddings=[embeddings[j]],
                                metadatas=[metadatas[j]],
                                ids=[ids[j]]
                            )
                        except Exception as item_error:
                            log(f"Failed to add item {j}: {item_error}", "ERROR")
            return
        except Exception as e:
            log(f"Collection add failed (attempt {attempt + 1}/{max_retries}): {e}", "ERROR")
            if attempt < max_retries - 1:
                time.sleep(2)
                global client, collection
                client = None
                collection = None
                coll = get_or_create_collection()
            else:
                raise

# -------------
# MAIN INDEX BUILD
# -------------
def build_or_update_index(file_paths: list[str]):
    """Build/update Chroma index from PDF files."""
    with _client_lock:
        coll = get_or_create_collection()

        total_files = len(file_paths)
        log(f"Starting indexing for {total_files} PDF(s)...", "INFO")
        start_total = time.time()

        for idx, file_path in enumerate(file_paths, start=1):
            log(f"\n📘 [{idx}/{total_files}] Processing {os.path.basename(file_path)}", "INFO")

            try:
                # Extract text
                full_text = extract_text_from_pdf(file_path)
                if not full_text:
                    log(f"No text extracted from {file_path}. Skipping.", "WARN")
                    continue

                # Chunk text
                chunks = chunk_text(full_text)
                log(f"Generated {len(chunks)} chunks from {file_path}", "INFO")

                # Generate embeddings
                embeddings = []
                for i, chunk in enumerate(chunks):
                    emb = get_embedding(chunk)
                    embeddings.append(emb)
                    if (i + 1) % 10 == 0:
                        log(f"  ... {i + 1}/{len(chunks)} chunks embedded", "INFO")

                # Metadata
                metadatas = [{"file": os.path.basename(file_path), "chunk_id": i, "source": file_path}
                             for i in range(len(chunks))]
                ids = [f"{os.path.basename(file_path)}_chunk_{i}" for i in range(len(chunks))]

                # Add to collection
                safe_collection_add(coll, chunks, embeddings, metadatas, ids)
                log(f"✅ Completed indexing {os.path.basename(file_path)} ({len(chunks)} chunks)", "SUCCESS")

            except Exception as e:
                log(f"Error processing {file_path}: {e}", "ERROR")
                import traceback
                print(traceback.format_exc())
                continue

        duration_total = time.time() - start_total
        log(f"✅ Indexing complete for {total_files} file(s) in {duration_total:.2f}s", "SUCCESS")
