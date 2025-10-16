# embed.py
import os
import pickle
import faiss
import numpy as np
from utils import extract_text_from_pdf, chunk_text
import boto3
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ----------------------
# Configuration
# ----------------------
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
# CRITICAL: Use correct Titan V2 model ID
BEDROCK_EMBED_MODEL = "amazon.titan-embed-text-v2:0"

FAISS_INDEX_DIR = "faiss_index"
FAISS_INDEX_FILE = os.path.join(FAISS_INDEX_DIR, "index.faiss")
METADATA_FILE = os.path.join(FAISS_INDEX_DIR, "metadata.pkl")

os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

# ----------------------
# AWS Bedrock client
# ----------------------
bedrock_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)

# ----------------------
# Embedding function
# ----------------------
def get_embedding(text: str) -> np.ndarray:
    """
    Calls Bedrock Titan V2 embedding model and returns a float32 numpy array.
    
    Amazon Titan Text Embeddings V2 specifications:
    - Model ID: amazon.titan-embed-text-v2:0
    - Max input tokens: 8,192
    - Max input characters: 50,000
    - Output vector size: 1,024 (default), 512, or 256
    - Supports 100+ languages
    """
    # Truncate text if too long (Titan V2 max: 50,000 chars)
    max_chars = 50000
    if len(text) > max_chars:
        print(f"Warning: Text truncated from {len(text)} to {max_chars} characters")
        text = text[:max_chars]
    
    # Titan V2 request format
    payload = {
        "inputText": text,
        "dimensions": 1024,  # Output dimension (1024 is default)
        "normalize": True    # Normalize embeddings for cosine similarity
    }
    
    try:
        resp = bedrock_client.invoke_model(
            modelId=BEDROCK_EMBED_MODEL,
            body=json.dumps(payload),
            contentType="application/json"
        )
        
        # Read and parse response
        resp_body = resp['body'].read().decode("utf-8")
        resp_json = json.loads(resp_body)
        
        # Extract embedding from response
        embedding = np.array(resp_json["embedding"], dtype="float32")
        
        # Verify dimension
        if embedding.shape[0] != 1024:
            raise ValueError(f"Expected 1024 dimensions, got {embedding.shape[0]}")
        
        return embedding
        
    except Exception as e:
        print(f"❌ Error getting embedding: {e}")
        print(f"   Model ID: {BEDROCK_EMBED_MODEL}")
        print(f"   Region: {AWS_REGION}")
        print(f"   Text length: {len(text)} characters")
        raise

# ----------------------
# FAISS index builder / updater
# ----------------------
def build_or_update_index(pdf_paths):
    """
    Build or update FAISS index from a list of local PDF file paths.
    Uses Amazon Titan Text Embeddings V2 (1024 dimensions)
    """
    # Load existing index and metadata if available
    if os.path.exists(FAISS_INDEX_FILE):
        index = faiss.read_index(FAISS_INDEX_FILE)
        with open(METADATA_FILE, "rb") as f:
            metadata = pickle.load(f)
        print(f"✅ Loaded existing index with {len(metadata)} chunks")
    else:
        # Titan V2 embedding dimension is 1024
        index = faiss.IndexFlatL2(1024)
        metadata = []
        print("✅ Created new FAISS index (1024 dimensions)")

    # Process each PDF
    for pdf_path in pdf_paths:
        filename = os.path.basename(pdf_path)

        # Skip already processed files
        if any(m["file"] == filename for m in metadata):
            print(f"⏭️  {filename} already processed. Skipping.")
            continue

        print(f"\n📄 Processing: {filename}")
        print(f"   Extracting text...")
        text = extract_text_from_pdf(pdf_path)
        
        if not text.strip():
            print(f"⚠️  Warning: No text extracted from {filename}")
            continue
        
        print(f"   Text length: {len(text)} characters")
        chunks = chunk_text(text, chunk_size=500)
        print(f"   Created {len(chunks)} chunks (500 words each)")
        print(f"   Creating embeddings...")
        
        successful_chunks = 0
        for idx, chunk in enumerate(chunks):
            try:
                emb = get_embedding(chunk).reshape(1, -1)
                index.add(emb)
                metadata.append({"file": filename, "text": chunk})
                successful_chunks += 1
                
                # Progress indicator every 10 chunks
                if (idx + 1) % 10 == 0:
                    print(f"   ✓ Processed {idx + 1}/{len(chunks)} chunks")
                    
            except Exception as e:
                print(f"   ❌ Error processing chunk {idx}: {e}")
                continue

        print(f"✅ Completed {filename}: {successful_chunks}/{len(chunks)} chunks embedded")

    # Save updated index and metadata
    print(f"\n💾 Saving index and metadata...")
    faiss.write_index(index, FAISS_INDEX_FILE)
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadata, f)

    print(f"✅ FAISS index updated successfully!")
    print(f"   Total chunks in index: {len(metadata)}")
    print(f"   Unique documents: {len(set(m['file'] for m in metadata))}")
    print(f"   Index saved to: {FAISS_INDEX_FILE}")


# ----------------------
# Test function
# ----------------------
if __name__ == "__main__":
    print("Testing Titan V2 Embedding Model...")
    print(f"Model ID: {BEDROCK_EMBED_MODEL}")
    print(f"Region: {AWS_REGION}")
    
    try:
        test_text = "This is a test sentence for embedding generation."
        print(f"\nTest text: '{test_text}'")
        
        embedding = get_embedding(test_text)
        print(f"✅ Embedding generated successfully!")
        print(f"   Dimension: {embedding.shape[0]}")
        print(f"   Type: {embedding.dtype}")
        print(f"   Sample values: {embedding[:5]}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check if Bedrock is available in your region")
        print("2. Verify model access: aws bedrock list-foundation-models --region us-west-2")
        print("3. Ensure IAM permissions for bedrock:InvokeModel")
        print("4. Confirm model ID: amazon.titan-embed-text-v2:0")
