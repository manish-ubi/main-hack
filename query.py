
# query.py - RAG Query with DynamoDB Cache Integration (Bedrock Titan Embeddings)
import os
import json
import boto3
from dotenv import load_dotenv
from embed import get_or_create_collection, get_embedding, log
from cache_dynamodb import (
    get_cached_answer,
    put_cached_answer,
    create_query_hash
)

load_dotenv()

# AWS Bedrock Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
LLM_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")
ENABLE_CACHE = os.getenv("ENABLE_CACHE", "true").lower() == "true"

bedrock_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)


def query_rag_system(query: str, top_k: int = 3) -> str:
    """
    Query the RAG system with DynamoDB caching using Bedrock Titan embeddings.
    
    Args:
        query: User's question
        top_k: Number of top documents to retrieve
        
    Returns:
        Answer string from the LLM
    """
    # Check cache first (if enabled)
    if ENABLE_CACHE:
        query_hash = create_query_hash(query)
        log(f"Checking cache for query: {query[:50]}...", "INFO")
        
        cached_result = get_cached_answer(query_hash)
        if cached_result:
            answer = cached_result.get("answer", "")
            log(f"Returning cached answer (hash: {query_hash[:8]}...)", "SUCCESS")
            return answer
        
        log("Cache miss - proceeding with RAG retrieval", "INFO")
    
    # Generate query embedding using Bedrock Titan (from embed.py)
    try:
        log("Generating query embedding with Bedrock Titan...", "INFO")
        query_embedding = get_embedding(query)
        
        # Get collection and retrieve relevant documents
        collection = get_or_create_collection()
        log(f"Querying collection with top_k={top_k}", "INFO")
        
        # Query using the Bedrock embedding (NOT query_texts)
        results = collection.query(
            query_embeddings=[query_embedding],  # Use Bedrock embedding, not text
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Extract retrieved documents
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        
        if not documents:
            log("No relevant documents found in collection", "WARN")
            return "I couldn't find any relevant information to answer your question."
        
        # Build context from retrieved documents
        context_parts = []
        for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
            source = meta.get("file", "unknown") if isinstance(meta, dict) else "unknown"
            chunk_id = meta.get("chunk_id", i) if isinstance(meta, dict) else i
            context_parts.append(f"[Document {i+1} - {source} (chunk {chunk_id}, distance: {dist:.3f})]:\n{doc}\n")
        
        context = "\n".join(context_parts)
        log(f"Retrieved {len(documents)} documents for context", "SUCCESS")
        
        # Build prompt for LLM
        prompt = f"""You are a helpful assistant. Answer the user's question based on the provided context.

Context from documents:
{context}

User Question: {query}

Instructions:
- Answer directly and concisely based on the context provided
- If the context doesn't contain relevant information, say so
- Cite the document sources when possible
- Be factual and accurate

Answer:"""
        
        # Call Bedrock LLM
        log(f"Invoking Bedrock model: {LLM_MODEL_ID}", "INFO")
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "temperature": 0.3,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        response = bedrock_client.invoke_model(
            modelId=LLM_MODEL_ID,
            body=json.dumps(request_body),
            contentType="application/json"
        )
        
        response_body = json.loads(response['body'].read().decode("utf-8"))
        answer = response_body['content'][0]['text'].strip()
        
        log(f"Generated answer (length: {len(answer)} chars)", "SUCCESS")
        
        # Cache the result (if enabled)
        if ENABLE_CACHE:
            log("Caching the answer for future queries", "INFO")
            retrieved_docs_summary = {
                "count": len(documents),
                "sources": [m.get("file", "unknown") for m in metadatas if isinstance(m, dict)],
                "avg_distance": sum(distances) / len(distances) if distances else 0
            }
            put_cached_answer(
                user_query=query,
                answer=answer,
                retrieved_docs=retrieved_docs_summary
            )
        
        return answer
        
    except FileNotFoundError:
        log("Collection not found - no documents indexed yet", "ERROR")
        raise
    except Exception as e:
        log(f"Error during RAG query: {e}", "ERROR")
        import traceback
        print(traceback.format_exc())
        raise


def query_rag_with_metadata(query: str, top_k: int = 3) -> dict:
    """
    Query the RAG system and return detailed metadata.
    
    Args:
        query: User's question
        top_k: Number of top documents to retrieve
        
    Returns:
        Dictionary with answer, sources, and cache status
    """
    query_hash = create_query_hash(query)
    
    # Check cache
    if ENABLE_CACHE:
        cached_result = get_cached_answer(query_hash)
        if cached_result:
            return {
                "answer": cached_result.get("answer", ""),
                "cached": True,
                "query_hash": query_hash,
                "sources": cached_result.get("retrieved_docs", {}).get("sources", []) if isinstance(cached_result.get("retrieved_docs"), dict) else [],
                "access_count": cached_result.get("access_count", 0)
            }
    
    # Proceed with RAG
    answer = query_rag_system(query, top_k)
    
    # Get metadata about sources
    try:
        query_embedding = get_embedding(query)
        collection = get_or_create_collection()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "distances"]
        )
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        sources = [m.get("file", "unknown") for m in metadatas if isinstance(m, dict)]
        avg_distance = sum(distances) / len(distances) if distances else 0
    except Exception as e:
        log(f"Error getting metadata: {e}", "WARN")
        sources = []
        avg_distance = 0
    
    return {
        "answer": answer,
        "cached": False,
        "query_hash": query_hash,
        "sources": sources,
        "access_count": 0,
        "avg_distance": avg_distance
    }