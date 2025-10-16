# main.py (enhanced with CSV SQL integration)
import os
import streamlit as st
import time
from datetime import datetime
import json
import re
import sqlparse
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import duckdb
import boto3
from embed import build_or_update_index, client, bedrock_client, get_embedding
from upload import upload_single, upload_batch
from query import query_rag_system
from dotenv import load_dotenv

load_dotenv()

# Set page config
st.set_page_config(
    page_title="Gilead Agentic QA (AWS) - Enhanced",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Gilead Agentic QA (AWS) - Enhanced")

# Initialize session state
if 'query_cache' not in st.session_state:
    st.session_state.query_cache = {}  # {cache_key: {'answer': str, 'access_count': int, 'timestamp': float}}

if 'feedback' not in st.session_state:
    st.session_state.feedback = []  # List of {'query': str, 'answer': str, 'feedback': str, 'timestamp': float}

if 'csv_handler' not in st.session_state:
    st.session_state.csv_handler = None

if 'csv_query_history' not in st.session_state:
    st.session_state.csv_query_history = []

# Custom function to compute cache stats
@st.cache_data(ttl=60)  # Cache stats for 1 min
def compute_cache_stats():
    if not st.session_state.query_cache:
        return {
            'total_entries': 0,
            'avg_access_count': 0.0,
            'max_access_count': 0,
            'oldest_entry_age_hours': 0.0,
            'newest_entry_age_hours': 0.0
        }
    
    entries = list(st.session_state.query_cache.values())
    total_entries = len(entries)
    access_counts = [e['access_count'] for e in entries]
    avg_access = sum(access_counts) / total_entries if total_entries > 0 else 0.0
    max_access = max(access_counts) if access_counts else 0
    
    timestamps = [e['timestamp'] for e in entries]
    now = time.time()
    oldest_age = (now - min(timestamps)) / 3600 if timestamps else 0.0
    newest_age = (now - max(timestamps)) / 3600 if timestamps else 0.0
    
    return {
        'total_entries': total_entries,
        'avg_access_count': round(avg_access, 1),
        'max_access_count': max_access,
        'oldest_entry_age_hours': round(oldest_age, 1),
        'newest_entry_age_hours': round(newest_age, 1)
    }

# CSVSqlHandler class (adapted for current codebase)
class CSVSqlHandler:
    """Handler for CSV file loading and natural language to SQL conversion."""
    
    def __init__(self, workspace_dir: str = "csv_workspace"):
        self.workspace_dir = workspace_dir
        self.db_path = os.path.join(workspace_dir, "workspace.duckdb")
        self.conn = None
        self._ensure_workspace()
        self._connect_db()
    
    def _ensure_workspace(self):
        """Ensure workspace directory exists."""
        os.makedirs(self.workspace_dir, exist_ok=True)
        st.info(f"Workspace directory: {self.workspace_dir}")
    
    def _connect_db(self):
        """Connect to DuckDB database."""
        try:
            self.conn = duckdb.connect(self.db_path)
            st.success("Connected to DuckDB")
        except Exception as e:
            st.error(f"Failed to connect to DuckDB: {e}")
            # Simplified retry logic
            time.sleep(1)
            try:
                self.conn = duckdb.connect(self.db_path)
                st.success("Connected to DuckDB after retry")
            except Exception:
                # Fallback to in-memory for simplicity
                self.conn = duckdb.connect(":memory:")
                self.db_path = ":memory:"
                st.warning("Using in-memory DuckDB due to connection issues")
    
    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
    
    def load_csv_files(self, csv_files: List[str]) -> Dict[str, Any]:
        """
        Load CSV files into DuckDB tables.
        Returns information about loaded tables.
        """
        loaded_tables = []
        errors = []
        
        for csv_file in csv_files:
            try:
                # Extract table name from filename
                table_name = os.path.splitext(os.path.basename(csv_file))[0]
                # Sanitize table name
                table_name = re.sub(r'[^a-zA-Z0-9_]', '_', table_name)
                if not table_name or table_name[0].isdigit():
                    table_name = f"table_{table_name}"
                
                # Load CSV into DuckDB
                self.conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_csv_auto(?)", [csv_file])
                
                # Get table info
                table_info = self.get_table_info(table_name)
                loaded_tables.append({
                    "table_name": table_name,
                    "file_path": csv_file,
                    "row_count": table_info["row_count"],
                    "columns": table_info["columns"]
                })
                
                st.info(f"Loaded table '{table_name}' with {table_info['row_count']} rows")
                
            except Exception as e:
                error_msg = f"Failed to load {csv_file}: {e}"
                st.error(error_msg)
                errors.append(error_msg)
        
        return {
            "loaded_tables": loaded_tables,
            "errors": errors,
            "success_count": len(loaded_tables),
            "error_count": len(errors)
        }
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get information about a table."""
        try:
            # Get row count
            result = self.conn.execute(f"SELECT COUNT(*) as count FROM {table_name}").fetchone()
            row_count = result[0] if result else 0
            
            # Get column information
            result = self.conn.execute(f"DESCRIBE {table_name}").fetchall()
            columns = [{"name": row[0], "type": row[1]} for row in result]
            
            return {
                "row_count": row_count,
                "columns": columns
            }
        except Exception as e:
            st.warning(f"Failed to get table info for {table_name}: {e}")
            return {"row_count": 0, "columns": []}
    
    def get_all_tables(self) -> List[Dict[str, Any]]:
        """Get information about all tables in the database."""
        try:
            result = self.conn.execute("SHOW TABLES").fetchall()
            tables = []
            for row in result:
                table_name = row[0]
                table_info = self.get_table_info(table_name)
                tables.append({
                    "table_name": table_name,
                    **table_info
                })
            return tables
        except Exception as e:
            st.warning(f"Failed to get tables: {e}")
            return []
    
    def natural_language_to_sql(self, question: str, table_context: Optional[str] = None) -> str:
        """
        Convert natural language question to SQL query using Bedrock.
        """
        # Get table information for context
        tables_info = self.get_all_tables()
        if not tables_info:
            raise ValueError("No tables available for querying")
        
        # Build context about available tables
        context_parts = []
        for table in tables_info:
            columns_str = ", ".join([f"{col['name']} ({col['type']})" for col in table['columns']])
            context_parts.append(f"Table '{table['table_name']}': {columns_str} ({table['row_count']} rows)")
        
        tables_context = "\n".join(context_parts)
        
        # Add specific table context if provided
        if table_context:
            tables_context = f"Focus on table: {table_context}\n\nAll tables:\n{tables_context}"
        
        prompt = f"""You are a SQL expert for DuckDB. Convert the user's natural language question to a valid DuckDB SQL query.

Available tables and their schemas:
{tables_context}

User question: {question}

Requirements:
1. Ensure you understand the requirement of the user in the SQL format
2. Generate ONLY a valid DuckDB SQL query
3. Use proper table and column names as shown above
4. Include appropriate WHERE clauses, JOINs, and aggregations as needed
5. Use LIMIT clause if the result might be large
6. Do not include any explanations or markdown formatting 
7. Ensure the query is safe and doesn't contain any dangerous operations

SQL Query:"""
        
        # Use existing bedrock_client and model
        AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
        BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")
        
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 500,
            "temperature": 0.1,
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
            sql_query = response_body['content'][0]['text']
            
            # Clean up the response
            sql_query = sql_query.strip()
            if sql_query.startswith("```sql"):
                sql_query = sql_query[6:]
            if sql_query.endswith("```"):
                sql_query = sql_query[:-3]
            sql_query = sql_query.strip()
            
            st.info(f"Generated SQL: {sql_query}")
            return sql_query
            
        except Exception as e:
            st.error(f"Failed to generate SQL: {e}")
            raise
    
    def validate_sql(self, sql_query: str) -> Tuple[bool, Optional[str]]:
        """
        Validate SQL query for safety and syntax.
        Returns (is_valid, error_message).
        """
        try:
            # Parse SQL to check syntax
            parsed = sqlparse.parse(sql_query)
            if not parsed:
                return False, "Empty SQL query"
            
            # Check for dangerous operations
            dangerous_keywords = [
                "DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE", "TRUNCATE",
                "EXEC", "EXECUTE", "CALL", "GRANT", "REVOKE", "REMOVE"
            ]
            
            upper_sql = sql_query.upper()
            for keyword in dangerous_keywords:
                if keyword in upper_sql:
                    return False, f"Dangerous operation detected: {keyword}"
            
            # Try to explain the query (DuckDB's EXPLAIN will validate syntax)
            try:
                self.conn.execute(f"EXPLAIN {sql_query}")
                return True, None
            except Exception as e:
                return False, f"SQL syntax error: {e}"
                
        except Exception as e:
            return False, f"Validation error: {e}"
    
    def execute_sql(self, sql_query: str) -> Tuple[bool, Optional[pd.DataFrame], Optional[str]]:
        """
        Execute SQL query and return results.
        Returns (success, dataframe, error_message).
        """
        try:
            # Validate first
            is_valid, error_msg = self.validate_sql(sql_query)
            if not is_valid:
                return False, None, error_msg
            
            # Execute query
            result = self.conn.execute(sql_query)
            df = result.df()
            
            st.info(f"SQL executed successfully, returned {len(df)} rows")
            return True, df, None
            
        except Exception as e:
            error_msg = f"SQL execution failed: {e}"
            st.error(error_msg)
            return False, None, error_msg
    
    def get_table_sample(self, table_name: str, limit: int = 5) -> Optional[pd.DataFrame]:
        """Get a sample of data from a table."""
        try:
            result = self.conn.execute(f"SELECT * FROM {table_name} LIMIT {limit}")
            return result.df()
        except Exception as e:
            st.warning(f"Failed to get sample from {table_name}: {e}")
            return None
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            st.info("Database connection closed")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📄 PDF Q&A", "📊 CSV SQL", "📈 Analytics", "⚙️ Cache Management"])

with tab1:
    # PDF Q&A Tab
    st.header("📄 PDF Document Q&A")
    
    # Mode selection
    col1, col2 = st.columns([3, 1])
    with col1:
        mode = st.radio("Select Mode:", ["Index New Documents", "Query Existing Documents"], index=0, horizontal=True)
    with col2:
        st.empty()  # Spacer
    
    if mode == "Index New Documents":
        # Ingest PDFs section
        st.subheader("🚀 Ingest PDFs")
        
        col_a, col_b = st.columns([3, 1])
        with col_a:
            uploaded_files = st.file_uploader("Upload one or more PDFs", type="pdf", accept_multiple_files=True, help="Drag and drop files here (Limit 200MB per file PDF)")
        with col_b:
            st.info("👆 Browse files")
        
        # Or local directory
        directory_path = st.text_input("Or provide a local directory path of PDFs")
        
        if uploaded_files or directory_path:
            if st.button("🚀 Run Ingest", type="primary"):
                uploaded_files_local = []
                if uploaded_files:
                    for file in uploaded_files:
                        if file.size > 200 * 1024 * 1024:  # 200MB limit
                            st.error(f"❌ {file.name} exceeds 200MB limit!")
                            continue
                        local_path = os.path.join(os.getcwd(), file.name)
                        with open(local_path, "wb") as f:
                            f.write(file.getbuffer())
                        upload_single(local_path)
                        uploaded_files_local.append(local_path)
                        st.success(f"✅ Uploaded {file.name}")
                
                if directory_path and os.path.exists(directory_path):
                    upload_batch(directory_path)
                    for f in os.listdir(directory_path):
                        if f.lower().endswith(".pdf"):
                            uploaded_files_local.append(os.path.join(directory_path, f))
                    st.success(f"✅ Batch uploaded from {directory_path}")
                
                if uploaded_files_local:
                    # Indexing process display
                    with st.container():
                        st.markdown("""
                        <div style="background-color: #E3F2FD; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #2196F3;">
                            <h4 style="margin-top: 0;">📋 Indexing Process:</h4>
                            <ol>
                                <li>✅ Upload PDFs to S3</li>
                                <li>🔄 Extract text with Textract</li>
                                <li>🔗 Sync with Knowledge Base</li>
                                <li>🎉 Ready for querying</li>
                            </ol>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.info("🔄 Extracting text and creating embeddings...")
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        build_or_update_index(uploaded_files_local)
                        progress_bar.progress(100)
                        status_text.success("✅ All embeddings built and index updated!")
                    except Exception as e:
                        st.error(f"❌ Error processing PDFs: {e}")
    
    elif mode == "Query Existing Documents":
        # Query section
        st.subheader("🔍 Query Documents")
        query = st.text_input("Enter your query", placeholder="What would you like to know?")
        top_k = st.slider("Number of results", 1, 10, 5)
        
        col_query, col_search = st.columns([4, 1])
        with col_query:
            st.empty()
        with col_search:
            if st.button("🔎 Search", type="primary"):
                pass  # Button in main flow below
        
        if query.strip() != "":
            # Caching logic
            cache_key = query.lower().strip() + "_" + str(top_k)
            cache_entry = st.session_state.query_cache.get(cache_key)
            
            if cache_entry:
                # Cache hit - increment access
                cache_entry['access_count'] += 1
                st.info("📦 Using cached response...")
                st.markdown("### 📝 Answer:")
                st.write(cache_entry['answer'])
            else:
                try:
                    with st.spinner("Querying RAG system..."):
                        answer = query_rag_system(query, top_k)
                    # Store in cache
                    st.session_state.query_cache[cache_key] = {
                        'answer': answer,
                        'access_count': 1,
                        'timestamp': time.time()
                    }
                    st.markdown("### 📝 Answer:")
                    st.write(answer)
                except FileNotFoundError:
                    st.error("❌ No documents processed yet.")
                except Exception as e:
                    st.error(f"❌ Error querying system: {e}")
        
        # Feedback
        if 'answer' in locals():
            col_thumbs_up, col_thumbs_down, col_text = st.columns([1, 1, 3])
            with col_thumbs_up:
                if st.button("👍", key="thumbs_up"):
                    st.session_state.feedback.append({
                        'query': query,
                        'answer': answer,
                        'feedback': 'positive',
                        'timestamp': time.time()
                    })
                    st.success("Thanks for the feedback! 👍")
            with col_thumbs_down:
                if st.button("👎", key="thumbs_down"):
                    feedback_text = st.text_input("What can we improve?", key="feedback_text")
                    st.session_state.feedback.append({
                        'query': query,
                        'answer': answer,
                        'feedback': 'negative',
                        'timestamp': time.time(),
                        'details': feedback_text
                    })
                    st.warning("Thanks for the feedback! 👎 We'll improve.")
            with col_text:
                st.empty()

with tab2:
    # CSV SQL Tab
    st.header("📊 CSV SQL")
    
    # Initialize CSV handler if not exists
    if st.session_state.csv_handler is None:
        st.session_state.csv_handler = CSVSqlHandler()
    
    csv_handler = st.session_state.csv_handler
    
    # File upload section
    st.subheader("📁 Load CSV Files")
    csv_files = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)
    
    if csv_files and st.button("📥 Load CSVs", type="primary"):
        # Save uploaded files
        workspace = os.path.join(os.getcwd(), "csv_workspace")
        os.makedirs(workspace, exist_ok=True)
        
        file_paths = []
        for f in csv_files:
            csv_path = os.path.join(workspace, f.name)
            with open(csv_path, "wb") as w:
                w.write(f.read())
            file_paths.append(csv_path)
        
        # Load into database
        with st.spinner("Loading CSV files..."):
            try:
                result = csv_handler.load_csv_files(file_paths)
                
                if result["success_count"] > 0:
                    st.success(f"✅ Loaded {result['success_count']} tables successfully!")
                    for table in result["loaded_tables"]:
                        st.info(f"📊 {table['table_name']}: {table['row_count']} rows, {len(table['columns'])} columns")
                
                if result["errors"]:
                    st.warning(f"⚠️ {len(result['errors'])} errors occurred")
                    for error in result["errors"]:
                        st.error(error)
                        
            except Exception as e:
                st.error(f"❌ Loading failed: {e}")
    
    # Show loaded tables
    tables = csv_handler.get_all_tables()
    if tables:
        st.subheader("📋 Loaded Tables")
        for table in tables:
            with st.expander(f"📊 {table['table_name']} ({table['row_count']} rows)"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Columns:**")
                    for col in table['columns']:
                        st.write(f"- {col['name']} ({col['type']})")
                with col2:
                    # Show sample data
                    sample = csv_handler.get_table_sample(table['table_name'])
                    if sample is not None:
                        st.write("**Sample Data:**")
                        st.dataframe(sample, use_container_width=True)
    
    # Natural Language to SQL
    st.subheader("🤖 Natural Language to SQL")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        nl_question = st.text_input(
            "Ask a question about your data:", 
            placeholder="What are the top 5 products by sales?"
        )
    
    with col2:
        selected_table = st.selectbox(
            "Focus on table (optional):", 
            ["All tables"] + [t['table_name'] for t in tables]
        ) if tables else st.selectbox("Focus on table (optional):", ["All tables"])
    
    if st.button("🔍 Generate SQL", type="primary") and nl_question.strip():
        with st.spinner("Generating SQL..."):
            try:
                table_context = selected_table if selected_table != "All tables" else None
                sql_query = csv_handler.natural_language_to_sql(nl_question, table_context)
                
                st.subheader("📝 Generated SQL")
                st.code(sql_query, language="sql")
                
                # Validate SQL
                is_valid, error_msg = csv_handler.validate_sql(sql_query)
                if is_valid:
                    st.success("✅ SQL is valid and safe")
                    
                    # Execute button
                    if st.button("▶️ Execute SQL", type="primary"):
                        with st.spinner("Executing query..."):
                            success, df, exec_error = csv_handler.execute_sql(sql_query)
                            if success:
                                st.subheader("📊 Results")
                                st.dataframe(df, use_container_width=True)
                                # Show summary stats
                                if len(df) > 0:
                                    st.info(f"📈 Returned {len(df)} rows, {len(df.columns)} columns")
                                # Store in CSV query history
                                st.session_state.csv_query_history.append({
                                    "question": nl_question,
                                    "sql": sql_query,
                                    "rows_returned": len(df),
                                    "timestamp": time.time()
                                })
                            else:
                                st.error(f"❌ Execution failed: {exec_error}")
                else:
                    st.error(f"❌ SQL validation failed: {error_msg}")
                
            except Exception as e:
                st.error(f"❌ Generation failed: {e}")
    
    # Manual SQL execution
    st.subheader("✏️ Manual SQL Execution")
    manual_sql = st.text_area(
        "Write your SQL query:", 
        value="SELECT * FROM information_schema.tables LIMIT 10;",
        height=100
    )
    
    if st.button("▶️ Execute Manual SQL", type="secondary") and manual_sql.strip():
        with st.spinner("Executing..."):
            success, df, error_msg = csv_handler.execute_sql(manual_sql)
            if success:
                st.dataframe(df, use_container_width=True)
            else:
                st.error(f"❌ {error_msg}")

with tab3:
    # Analytics Tab
    st.header("📈 Analytics & Insights")
    
    # Cache Statistics
    st.subheader("💾 Cache Statistics")
    stats = compute_cache_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Entries", stats['total_entries'])
    col2.metric("Avg Access Count", stats['avg_access_count'])
    col3.metric("Max Access Count", stats['max_access_count'])
    col4.metric("Oldest Entry (hrs)", stats['oldest_entry_age_hours'])
    
    # PDF Feedback Statistics
    st.subheader("👍👎 PDF Feedback Statistics")
    if st.session_state.feedback:
        positive = sum(1 for f in st.session_state.feedback if f['feedback'] == 'positive')
        negative = sum(1 for f in st.session_state.feedback if f['feedback'] == 'negative')
        total = len(st.session_state.feedback)
        col_fb1, col_fb2 = st.columns(2)
        col_fb1.metric("Positive Feedback", positive)
        col_fb2.metric("Negative Feedback", negative)
        st.caption(f"Total Feedback: {total}")
    else:
        st.info("📭 PDF Feedback statistics unavailable")
    
    # CSV Query History (if any)
    if st.session_state.csv_query_history:
        st.subheader("📊 CSV Query History")
        history_df = pd.DataFrame(st.session_state.csv_query_history)
        st.dataframe(history_df, use_container_width=True)

with tab4:
    # Cache Management Tab
    st.header("⚙️ Cache Management")
    
    # Cache Operations
    st.subheader("🧹 Cache Operations")
    col_op1, col_op2 = st.columns(2)
    with col_op1:
        if st.button("🗑️ Clear All Cache"):
            st.session_state.query_cache = {}
            st.success("✅ All cache cleared!")
            st.rerun()
    with col_op2:
        if st.button("💰 Clean Expired Entries"):
            # Simple: clear entries older than 24 hours
            now = time.time()
            expired_keys = [k for k, v in st.session_state.query_cache.items() if (now - v['timestamp']) > 86400]
            for key in expired_keys:
                del st.session_state.query_cache[key]
            st.success(f"✅ Cleaned {len(expired_keys)} expired entries!")
            st.rerun()
    
    # Clear by Pattern
    pattern = st.text_input("By pattern (optional):", placeholder="Enter pattern to match")
    if st.button("🎯 Clear by Pattern"):
        if pattern:
            matched_keys = [k for k in st.session_state.query_cache if pattern.lower() in k.lower()]
            for key in matched_keys:
                del st.session_state.query_cache[key]
            st.success(f"✅ Cleared {len(matched_keys)} entries matching '{pattern}'!")
        else:
            st.warning("⚠️ Enter a pattern first.")
        st.rerun()
    
    # Cache Details
    st.subheader("📊 Cache Details")
    if st.session_state.query_cache:
        # Compute and display as JSON-like
        details = {
            "total_entries": stats['total_entries'],
            "avg_access_count": f"Decimal({stats['avg_access_count']})",
            "max_access_count": f"Decimal({stats['max_access_count']})",
            "oldest_entry_age_hours": f"Decimal({stats['oldest_entry_age_hours']})",
            "newest_entry_age_hours": f"Decimal({stats['newest_entry_age_hours']})"
        }
        st.json(details)
    else:
        st.info("📭 No cache entries available")
    
    # Feedback Management
    st.subheader("👍👎 Feedback Management")
    if st.session_state.feedback:
        recent_feedback = [f for f in st.session_state.feedback if (time.time() - f['timestamp']) < 86400]  # Last 24h
        if recent_feedback:
            for fb in recent_feedback[-5:]:  # Last 5
                with st.expander(f"Query: {fb['query'][:50]}... | Feedback: {fb['feedback'].upper()} | {datetime.fromtimestamp(fb['timestamp']).strftime('%Y-%m-%d %H:%M')}"):
                    st.write(f"**Answer:** {fb['answer'][:200]}...")
                    if 'details' in fb:
                        st.write(f"**Details:** {fb['details']}")
        else:
            st.info("📭 No recent feedback available")
        
        if st.button("🧹 Clean Old Feedback"):
            now = time.time()
            old_feedback = [f for f in st.session_state.feedback if (now - f['timestamp']) > 86400 * 7]  # Older than 7 days
            st.session_state.feedback = [f for f in st.session_state.feedback if f not in old_feedback]
            st.success(f"✅ Cleaned {len(old_feedback)} old feedback entries!")
            st.rerun()
    else:
        st.info("📭 No recent feedback available")

# Sidebar for Index Stats (shared)
with st.sidebar:
    st.divider()
    st.subheader("📊 Index Stats")
    try:
        collection = client.get_collection("pdf_docs")
        results = collection.get(include=["documents", "metadatas"])
        total_chunks = len(results["ids"])
        unique_files = set(m.get("file", "unknown") for m in results["metadatas"] if isinstance(m, dict))
        
        st.metric("Total Chunks", total_chunks)
        st.metric("Documents", len(unique_files))
        
        with st.expander("View Documents"):
            for doc in sorted(unique_files):
                st.text(f"📄 {doc}")
    except Exception as e:
        st.error(f"❌ Could not load stats: {e}")