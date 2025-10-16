# query.py
import pickle
import faiss
import numpy as np
from embed import get_embedding, FAISS_INDEX_FILE, METADATA_FILE
import os
import boto3
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")
bedrock_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)

def load_index():
    """Load FAISS index and metadata"""
    if not os.path.exists(FAISS_INDEX_FILE):
        raise FileNotFoundError("FAISS index not found. Please upload and process PDFs first.")
    
    index = faiss.read_index(FAISS_INDEX_FILE)
    with open(METADATA_FILE, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

def query_rag_system(query, top_k=5):
    """Query the RAG system and get answer from Claude"""
    index, metadata = load_index()
    query_emb = get_embedding(query)
    D, I = index.search(np.array([query_emb]), top_k)
    
    # Gather context from top-k results
    context = "\n\n".join([f"[Source: {metadata[i]['file']}]\n{metadata[i]['text']}" for i in I[0]])
    
    # Build prompt for Claude
    prompt = f"""Answer the following query based on the provided context. If the context doesn't contain relevant information, say so.

Context:
{context}

Query: {query}

Answer:"""
    
    # Prepare request body for Claude
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2000,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    
    # Call Bedrock
    try:
        resp = bedrock_client.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            body=json.dumps(request_body),
            contentType="application/json"
        )
        
        # Parse response
        response_body = json.loads(resp['body'].read().decode("utf-8"))
        answer = response_body['content'][0]['text']
        
        return answer
    except Exception as e:
        print(f"Error querying Claude: {e}")
        raise

if __name__ == "__main__":
    query = input("Enter your query: ")
    try:
        answer = query_rag_system(query)
        print("\nAnswer:\n", answer)
    except Exception as e:
        print(f"Error: {e}")
