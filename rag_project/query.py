# # query.py
# import os
# import json
# import boto3
# from embed import get_embedding, client
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

# # ----------------------
# # Chroma collection
# # ----------------------
# collection = client.get_collection("pdf_docs")


# def query_rag_system(query: str, top_k: int = 5) -> str:
#     """
#     Query the RAG system and get answer from Bedrock LLM (Claude).
#     Returns the LLM-generated answer.
#     """
#     # ----------------------
#     # 1️⃣ Generate embedding for the query
#     # ----------------------
#     query_emb = get_embedding(query) # ensure it's a list of floats

#     # ----------------------
#     # 2️⃣ Retrieve top-k relevant documents from Chroma
#     # ----------------------
#     results = collection.query(
#         query_embeddings=[query_emb],
#         n_results=top_k,
#         include=["documents", "metadatas"]
#     )

#     docs = results.get("documents", [[]])[0]      # list of retrieved document texts
#     metadatas = results.get("metadatas", [{}])[0]  # list of metadata dicts

#     # ----------------------
#     # 3️⃣ Build context for the LLM
#     # ----------------------
#     context = "\n\n".join(
#         f"[Source: {m.get('file', 'unknown')}, chunk: {m.get('chunk_id', '?')}]\n{d}"
#         for d, m in zip(docs, metadatas)
#     )

#     prompt = f"""Answer the following query based on the provided context. 
# If the context doesn't contain relevant information, respond honestly that you don't know.

# Context:
# {context}

# Query: {query}

# Answer:"""

#     # ----------------------
#     # 4️⃣ Call Bedrock LLM
#     # ----------------------
#     request_body = {
#         "anthropic_version": "bedrock-2023-05-31",
#         "max_tokens": 2000,
#         "messages": [
#             {"role": "user", "content": prompt}
#         ]
#     }

#     try:
#         resp = bedrock_client.invoke_model(
#             modelId=BEDROCK_MODEL_ID,
#             body=json.dumps(request_body),
#             contentType="application/json"
#         )

#         response_body = json.loads(resp['body'].read().decode("utf-8"))
#         answer = response_body['content'][0]['text']
#         return answer

#     except Exception as e:
#         print(f"❌ Error querying Bedrock: {e}")
#         raise


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



# query.py (updated)
import os
import json
import boto3
from embed import get_embedding, client
from dotenv import load_dotenv

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
# Chroma collection
# ----------------------
collection = client.get_collection("pdf_docs")


def query_rag_system(query: str, top_k: int = 5) -> str:
    """
    Query the RAG system and get answer from Bedrock LLM (Claude).
    Returns the LLM-generated answer.
    """
    print(f"🔍 [QUERY LOG] Starting RAG query: '{query}' (top_k={top_k})")
    
    # ----------------------
    # 1️⃣ Generate embedding for the query
    # ----------------------
    print("🔍 [QUERY LOG] Generating embedding for query...")
    query_emb = get_embedding(query)  # ensure it's a list of floats
    print("✅ [QUERY LOG] Embedding generated successfully.")

    # ----------------------
    # 2️⃣ Retrieve top-k relevant documents from Chroma
    # ----------------------
    print(f"🔍 [QUERY LOG] Retrieving top-{top_k} documents from Chroma...")
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k,
        include=["documents", "metadatas"]
    )
    print(f"✅ [QUERY LOG] Retrieved {len(results.get('documents', [[]])[0])} documents.")

    docs = results.get("documents", [[]])[0]      # list of retrieved document texts
    metadatas = results.get("metadatas", [{}])[0]  # list of metadata dicts

    # ----------------------
    # 3️⃣ Build context for the LLM
    # ----------------------
    print("🔍 [QUERY LOG] Building context for LLM...")
    context = "\n\n".join(
        f"[Source: {m.get('file', 'unknown')}, chunk: {m.get('chunk_id', '?')}]\n{d}"
        for d, m in zip(docs, metadatas)
    )
    print(f"✅ [QUERY LOG] Context built ({len(context)} chars).")

    prompt = f"""Answer the following query based on the provided context. 
If the context doesn't contain relevant information, respond honestly that you don't know.

Context:
{context}

Query: {query}

Answer:"""
    print("✅ [QUERY LOG] Prompt prepared.")

    # ----------------------
    # 4️⃣ Call Bedrock LLM
    # ----------------------
    print("🔍 [QUERY LOG] Calling Bedrock LLM (Claude)...")
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2000,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    try:
        resp = bedrock_client.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            body=json.dumps(request_body),
            contentType="application/json"
        )

        response_body = json.loads(resp['body'].read().decode("utf-8"))
        answer = response_body['content'][0]['text']
        print("✅ [QUERY LOG] LLM response received successfully.")
        print(f"📝 [QUERY LOG] Answer length: {len(answer)} chars")
        return answer

    except Exception as e:
        print(f"❌ [QUERY LOG] Error querying Bedrock: {e}")
        raise


# ----------------------
# CLI Test
# ----------------------
if __name__ == "__main__":
    query = input("Enter your query: ")
    try:
        answer = query_rag_system(query)
        print("\nAnswer:\n", answer)
    except Exception as e:
        print(f"Error: {e}")