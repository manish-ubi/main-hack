# # query.py - FIXED VERSION
# import os
# import json
# import boto3
# from dotenv import load_dotenv

# # ----------------------
# # Load environment variables
# # ----------------------
# load_dotenv()
# AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
# BEDROCK_MODEL_ID = os.getenv(
#     "BEDROCK_MODEL_ID",
#     "anthropic.claude-3-sonnet-20240229-v1:0"
# )

# bedrock_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)

# def query_rag_system(query: str, top_k: int = 5) -> str:
#     """
#     Query the RAG system with thread-safe client handling.
#     """
#     try:
#         # Import here to avoid circular import and ensure fresh client
#         from embed import get_embedding, get_or_create_collection
        
#         # Get fresh collection
#         collection = get_or_create_collection()
        
#         # Get query embedding
#         query_embedding = get_embedding(query)
        
#         # Search in ChromaDB
#         results = collection.query(
#             query_embeddings=[query_embedding],
#             n_results=top_k,
#             include=["documents", "metadatas", "distances"]
#         )
        
#         if not results["documents"] or not results["documents"][0]:
#             return "No relevant documents found."
        
#         # Build context from results
#         context_parts = []
#         for i, (doc, metadata) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
#             source = metadata.get("file", "unknown")
#             chunk_id = metadata.get("chunk_id", "?")
#             context_parts.append(f"[Source: {source}, Chunk: {chunk_id}]\n{doc}\n")
        
#         context = "\n---\n".join(context_parts)
        
#         # Build prompt
#         prompt = f"""You are a helpful assistant. Answer the question based on the following context.

# Context:
# {context}

# Question: {query}

# Answer the question based only on the context provided above. If the context doesn't contain relevant information, say so."""

#         # Call Bedrock
#         request_body = {
#             "anthropic_version": "bedrock-2023-05-31",
#             "max_tokens": 2000,
#             "temperature": 0.7,
#             "messages": [
#                 {"role": "user", "content": prompt}
#             ]
#         }
        
#         response = bedrock_client.invoke_model(
#             modelId=BEDROCK_MODEL_ID,
#             body=json.dumps(request_body),
#             contentType="application/json"
#         )
        
#         response_body = json.loads(response['body'].read().decode("utf-8"))
#         answer = response_body['content'][0]['text']
        
#         return answer
        
#     except Exception as e:
#         import traceback
#         error_details = traceback.format_exc()
#         return f"Error querying RAG system: {e}\n\nDebug info:\n{error_details}"


# # ----------------------
# # CLI Test
# # ----------------------
# if __name__ == "__main__":
#     query = input("Enter your query: ")
#     try:
#         answer = query_rag_system(query)
#         print("\nAnswer:\n", answer)
#     except Exception as e:
#         print(f"Error: {e}")




# query.py - FIXED + DETAILED LOGGING VERSION
import os
import json
import boto3
import time
from dotenv import load_dotenv
from datetime import datetime

# ----------------------
# Load environment variables
# ----------------------
load_dotenv()
AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
BEDROCK_MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID",
    "anthropic.claude-3-sonnet-20240229-v1:0"
)

bedrock_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)

# ----------------------
# LOGGING UTILS
# ----------------------
def log(msg: str, level: str = "INFO"):
    """Timestamped colored logs (Windows-safe)."""
    ts = datetime.now().strftime("%H:%M:%S")
    color = {
        "INFO": "\033[94m",      # Blue
        "SUCCESS": "\033[92m",   # Green
        "WARN": "\033[93m",      # Yellow
        "ERROR": "\033[91m",     # Red
        "RESET": "\033[0m"
    }
    print(f"{color.get(level, '')}[{ts}] [{level}] {msg}{color['RESET']}")

# ----------------------
# RAG QUERY FUNCTION
# ----------------------
def query_rag_system(query: str, top_k: int = 5) -> str:
    """
    Query the RAG system with detailed logs and error handling.
    """
    start_time = time.time()
    log(f"Starting RAG query: \"{query}\" (top_k={top_k})", "INFO")

    try:
        # Import here to avoid circular imports
        from embed import get_embedding, get_or_create_collection

        # Step 1: Get collection
        log("🔍 Getting or creating Chroma collection...", "INFO")
        collection = get_or_create_collection()

        # Step 2: Get query embedding
        log("🔮 Generating query embedding via Bedrock Titan...", "INFO")
        t0 = time.time()
        query_embedding = get_embedding(query)
        log(f"✅ Query embedding generated in {time.time() - t0:.2f}s", "SUCCESS")

        # Step 3: Query Chroma
        log("📚 Querying Chroma for top similar chunks...", "INFO")
        t1 = time.time()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        log(f"✅ Chroma query completed in {time.time() - t1:.2f}s", "SUCCESS")

        # Step 4: Handle no results
        if not results["documents"] or not results["documents"][0]:
            log("⚠️ No relevant documents found in the collection.", "WARN")
            return "No relevant documents found."

        # Step 5: Build context
        log(f"🧩 Building context from top {len(results['documents'][0])} chunks...", "INFO")
        context_parts = []
        for i, (doc, metadata) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
            source = metadata.get("file", "unknown")
            chunk_id = metadata.get("chunk_id", "?")
            dist = results["distances"][0][i] if "distances" in results else None
            log(f"  → Chunk {i+1}: Source={source}, ID={chunk_id}, Distance={dist}", "INFO")
            context_parts.append(f"[Source: {source}, Chunk: {chunk_id}]\n{doc}\n")

        context = "\n---\n".join(context_parts)

        # Optional: preview first few chars
        preview = context[:300].replace("\n", " ")
        log(f"🧠 Context preview: {preview}...", "INFO")

        # Step 6: Prepare Bedrock prompt
        prompt = f"""You are a helpful assistant. Answer the question based on the following context.

Context:
{context}

Question: {query}

Answer the question based only on the context provided above. 
If the context doesn't contain relevant information, say so."""

        # Step 7: Invoke Bedrock model
        log(f"💬 Calling Bedrock model ({BEDROCK_MODEL_ID})...", "INFO")
        t2 = time.time()
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2000,
            "temperature": 0.7,
            "messages": [{"role": "user", "content": prompt}]
        }

        response = bedrock_client.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            body=json.dumps(request_body),
            contentType="application/json"
        )

        response_body = json.loads(response["body"].read().decode("utf-8"))
        answer = response_body["content"][0]["text"]
        log(f"✅ Bedrock response received in {time.time() - t2:.2f}s", "SUCCESS")

        total_time = time.time() - start_time
        log(f"🏁 Query completed successfully in {total_time:.2f}s", "SUCCESS")

        return answer

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        log(f"❌ Error during RAG query: {e}", "ERROR")
        log(f"Debug Trace:\n{error_details}", "ERROR")
        return f"Error querying RAG system: {e}\n\nDebug info:\n{error_details}"

# ----------------------
# CLI TEST
# ----------------------
if __name__ == "__main__":
    query = input("Enter your query: ")
    try:
        answer = query_rag_system(query)
        print("\n\n==============================")
        print("🧠 FINAL ANSWER:")
        print("==============================")
        print(answer)
    except Exception as e:
        log(f"Fatal error in CLI mode: {e}", "ERROR")
