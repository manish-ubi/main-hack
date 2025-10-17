import os
import time
import hashlib
from typing import Any, Dict, Optional
from datetime import datetime

import boto3
from boto3.dynamodb.conditions import Attr

# ================================
# CONFIGURATION
# ================================
DEFAULT_TTL_SECONDS = 60 * 60 * 24  # 1 day
CACHE_CLEANUP_INTERVAL = 60 * 60    # 1 hour

_CACHED_TABLE = None
_dynamodb_resource = None


# ================================
# LOGGING UTILS (Consistent with embed.py)
# ================================
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


# ================================
# DYNAMODB RESOURCE INITIALIZER
# ================================
def get_dynamodb_resource():
    """Get or initialize DynamoDB resource."""
    global _dynamodb_resource
    if _dynamodb_resource is not None:
        return _dynamodb_resource
    try:
        region = os.getenv("AWS_REGION", "ap-south-1")
        log(f"Initializing DynamoDB resource in {region}", "INFO")
        _dynamodb_resource = boto3.resource("dynamodb", region_name=region)
        return _dynamodb_resource
    except Exception as e:
        log(f"DynamoDB resource initialization failed: {e}", "ERROR")
        return None


# ================================
# DYNAMODB TABLE INITIALIZER
# ================================
def _table():
    """Get or initialize DynamoDB table reference."""
    global _CACHED_TABLE
    if _CACHED_TABLE is not None:
        return _CACHED_TABLE
    try:
        tbl_name = os.getenv("DDB_TABLE", "agentic_qa_cache")
        log(f"Using DynamoDB table: {tbl_name}", "INFO")
        db = get_dynamodb_resource()
        if db is None:
            return None
        _CACHED_TABLE = db.Table(tbl_name)
        return _CACHED_TABLE
    except Exception as e:
        log(f"DDB table unavailable: {e}", "WARN")
        return None


# ================================
# HASH HELPER
# ================================
def create_query_hash(user_query: str) -> str:
    """
    Create a stable SHA256 hash from a user query.
    Ensures consistent normalization.
    """
    if not user_query:
        return ""
    normalized = user_query.strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


# ================================
# CACHE READ
# ================================
def get_cached_answer(query_hash: str) -> Optional[Dict[str, Any]]:
    """Retrieve cached item by its query_hash."""
    table = _table()
    if table is None:
        return None

    try:
        resp = table.get_item(Key={"query_hash": query_hash})
        item = resp.get("Item")

        if item:
            log(f"Cache hit for hash: {query_hash[:8]}...", "SUCCESS")
            # Update access stats asynchronously
            try:
                current_time = int(time.time())
                table.update_item(
                    Key={"query_hash": query_hash},
                    UpdateExpression="SET access_count = if_not_exists(access_count, :zero) + :inc, last_accessed = :now",
                    ExpressionAttributeValues={":inc": 1, ":zero": 0, ":now": current_time},
                )
            except Exception as update_err:
                log(f"Failed to update access count: {update_err}", "WARN")
        else:
            log(f"Cache miss for hash: {query_hash[:8]}...", "INFO")

        return item
    except Exception as e:
        log(f"DDB get_item failed: {e}", "WARN")
        return None


# ================================
# CACHE WRITE
# ================================
def put_cached_answer(
    user_query: str,
    answer: str,
    retrieved_docs: Optional[Any] = None,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> None:
    """Insert a new cache record with query_hash and user_query."""
    table = _table()
    if table is None:
        return

    try:
        query_hash = create_query_hash(user_query)
        now = int(time.time())
        expire_at = now + ttl_seconds

        item = {
            "query_hash": query_hash,   # Primary key
            "user_query": user_query,   # Original query text
            "answer": answer,
            "ttl": expire_at,
            "created_at": now,
            "access_count": 0,
            "last_accessed": now,
        }
        
        # Add retrieved_docs if provided
        if retrieved_docs is not None:
            item["retrieved_docs"] = str(retrieved_docs)  # Convert to string for DDB

        table.put_item(Item=item)
        log(f"Cache stored for query '{user_query[:50]}...' (hash={query_hash[:8]}...)", "SUCCESS")
    except Exception as e:
        log(f"DDB put_item failed: {e}", "ERROR")


# ================================
# INVALIDATE CACHE
# ================================
def invalidate_cache(query_hash: Optional[str] = None, pattern: Optional[str] = None) -> int:
    """
    Invalidate cache entries.
    - If query_hash provided: deletes specific entry.
    - If pattern provided: deletes entries whose hash or user_query contains the pattern.
    - If neither: deletes all entries.
    Returns number of entries deleted.
    """
    table = _table()
    if table is None:
        return 0

    try:
        if query_hash:
            table.delete_item(Key={"query_hash": query_hash})
            log(f"Invalidated cache entry: {query_hash[:8]}...", "SUCCESS")
            return 1

        # Otherwise, scan and delete
        scan_kwargs = {}
        if pattern:
            scan_kwargs["FilterExpression"] = (
                Attr("query_hash").contains(pattern) | Attr("user_query").contains(pattern)
            )

        deleted_count = 0
        response = table.scan(**scan_kwargs)

        for item in response.get("Items", []):
            table.delete_item(Key={"query_hash": item["query_hash"]})
            deleted_count += 1

        while "LastEvaluatedKey" in response:
            scan_kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]
            response = table.scan(**scan_kwargs)
            for item in response.get("Items", []):
                table.delete_item(Key={"query_hash": item["query_hash"]})
                deleted_count += 1

        log(f"Invalidated {deleted_count} cache entries", "SUCCESS")
        return deleted_count
    except Exception as e:
        log(f"Cache invalidation failed: {e}", "ERROR")
        return 0


# ================================
# CLEANUP EXPIRED ENTRIES
# ================================
def cleanup_expired_cache() -> int:
    """Remove expired cache entries. Returns number removed."""
    table = _table()
    if table is None:
        return 0

    try:
        now = int(time.time())
        response = table.scan(FilterExpression=Attr("ttl").lt(now))
        deleted_count = 0

        for item in response.get("Items", []):
            table.delete_item(Key={"query_hash": item["query_hash"]})
            deleted_count += 1

        while "LastEvaluatedKey" in response:
            response = table.scan(
                FilterExpression=Attr("ttl").lt(now),
                ExclusiveStartKey=response["LastEvaluatedKey"],
            )
            for item in response.get("Items", []):
                table.delete_item(Key={"query_hash": item["query_hash"]})
                deleted_count += 1

        if deleted_count > 0:
            log(f"Cleaned up {deleted_count} expired cache entries", "SUCCESS")
        return deleted_count
    except Exception as e:
        log(f"Cache cleanup failed: {e}", "ERROR")
        return 0


# ================================
# CACHE STATS
# ================================
def get_cache_stats() -> Dict[str, Any]:
    """Compute basic statistics about cache table."""
    table = _table()
    if table is None:
        return {"error": "DynamoDB table unavailable"}

    try:
        now = int(time.time())
        response = table.scan()
        items = response.get("Items", [])

        # Continue scanning if there are more items
        while "LastEvaluatedKey" in response:
            response = table.scan(ExclusiveStartKey=response["LastEvaluatedKey"])
            items.extend(response.get("Items", []))

        access_counts = [i.get("access_count", 0) for i in items]
        created_times = [i.get("created_at", 0) for i in items]

        stats = {
            "total_entries": len(items),
            "avg_access_count": sum(access_counts) / len(access_counts) if access_counts else 0,
            "max_access_count": max(access_counts) if access_counts else 0,
            "oldest_entry_age_hours": (now - min(created_times)) / 3600 if created_times else 0,
            "newest_entry_age_hours": (now - max(created_times)) / 3600 if created_times else 0,
        }
        return stats
    except Exception as e:
        log(f"Failed to get cache stats: {e}", "ERROR")
        return {"error": str(e)}


# ================================
# UPDATE ACCESS COUNT
# ================================
def update_access_stats(query_hash: str) -> None:
    """Increment access count for a specific cached item."""
    table = _table()
    if table is None:
        return

    try:
        now = int(time.time())
        table.update_item(
            Key={"query_hash": query_hash},
            UpdateExpression="SET access_count = if_not_exists(access_count, :zero) + :inc, last_accessed = :now",
            ExpressionAttributeValues={":inc": 1, ":zero": 0, ":now": now},
        )
        log(f"Updated access stats for {query_hash[:8]}...", "INFO")
    except Exception as e:
        log(f"Failed to update access stats: {e}", "WARN")