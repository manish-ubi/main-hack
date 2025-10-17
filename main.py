
# main.py - Streamlit App with DynamoDB Cache Integration
import os
import streamlit as st
import time
from datetime import datetime
import warnings 
warnings.filterwarnings("ignore")
import json
import re
import sqlparse
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import duckdb
import boto3
import gc
import decimal

# Import from your modules
from embed import build_or_update_index, get_or_create_collection
from query import query_rag_system, query_rag_with_metadata
from cache_dynamodb import (
    get_cache_stats, 
    invalidate_cache, 
    cleanup_expired_cache, 
    create_query_hash,
    get_cached_answer
)
from dotenv import load_dotenv

load_dotenv()

# Set page config
st.set_page_config(
    page_title="Gilead Agentic QA (AWS) - Enhanced with Cache",
    page_icon="🧬",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196F3;
    }
    .cache-hit {
        color: #4CAF50;
        font-weight: bold;
    }
    .cache-miss {
        color: #FF9800;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧬 Gilead Agentic QA (AWS) - Enhanced with Cache")



# helper to convert Decimal -> int/float and sanitize nested structures
def _decimal_to_primitive(obj):
    """
    Convert decimal.Decimal instances (or nested dict/list structures containing them)
    into int or float so streamlit and formatting won't crash.
    """
    if isinstance(obj, decimal.Decimal):
        # convert whole-number decimals to int, else to float
        try:
            if obj == obj.to_integral_value():
                return int(obj)
            else:
                return float(obj)
        except Exception:
            # fallback
            try:
                return float(obj)
            except Exception:
                return str(obj)
    elif isinstance(obj, dict):
        return {k: _decimal_to_primitive(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_decimal_to_primitive(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_decimal_to_primitive(v) for v in obj)
    else:
        return obj


# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Cache toggle
    enable_cache = st.toggle("Enable DynamoDB Cache", value=True)
    if enable_cache:
        os.environ["ENABLE_CACHE"] = "true"
        ddb_table = st.text_input("DynamoDB Table Name", value=os.getenv("DDB_TABLE", "gilead_qa_cache"))
        os.environ["DDB_TABLE"] = ddb_table
    else:
        os.environ["ENABLE_CACHE"] = "false"
    
    # AWS Configuration
    aws_region = st.text_input("AWS Region", value=os.getenv("AWS_REGION", "ap-south-1"))
    os.environ["AWS_REGION"] = aws_region
    
    bedrock_model = st.selectbox(
        "Bedrock Model",
        ["anthropic.claude-3-sonnet-20240229-v1:0", "anthropic.claude-3-haiku-20240307-v1:0"],
        index=0
    )
    os.environ["BEDROCK_MODEL_ID"] = bedrock_model
    
    st.divider()
    
    # Index Stats
    st.subheader("📊 Index Stats")
    try:
        collection = get_or_create_collection()
        results = collection.get(include=["documents", "metadatas"])
        total_chunks = len(results["ids"])
        unique_files = set(m.get("file", "unknown") for m in results["metadatas"] if isinstance(m, dict))
        st.metric("Total Chunks", total_chunks)
        st.metric("Documents", len(unique_files))
        with st.expander("View Documents"):
            for doc in sorted(unique_files):
                st.text(f"📄 {doc}")
    except Exception as e:
        st.warning(f"Could not load stats: {e}")
    
    st.divider()
    
    # Cache Stats (if enabled)
    if enable_cache:
        st.subheader("💾 Cache Stats")
        cache_stats = _decimal_to_primitive(get_cache_stats())
        if "error" not in cache_stats:
            st.metric("Cached Entries", cache_stats.get("total_entries", 0))
            avg_access = cache_stats.get("avg_access_count", 0)
            st.metric("Avg Access Count", f"{avg_access:.1f}")
        else:
            st.info("Cache unavailable")

# Initialize session state
if 'feedback' not in st.session_state:
    st.session_state.feedback = []
if 'csv_handler' not in st.session_state:
    st.session_state.csv_handler = None
if 'csv_query_history' not in st.session_state:
    st.session_state.csv_query_history = []
if 'query_history' not in st.session_state:
    st.session_state.query_history = []



# CSVSqlHandler class (same as before)
class CSVSqlHandler:
    """Handler for CSV file loading and natural language to SQL conversion."""

    def __init__(self, workspace_dir: str = "csv_workspace"):
        self.workspace_dir = workspace_dir
        self.db_path = os.path.join(workspace_dir, "workspace.duckdb")
        self.conn = None
        self._ensure_workspace()
        self._connect_db()

    def _ensure_workspace(self):
        os.makedirs(self.workspace_dir, exist_ok=True)

    def _connect_db(self):
        try:
            self.conn = duckdb.connect(self.db_path)
        except Exception as e:
            st.warning(f"Using in-memory DuckDB: {e}")
            self.conn = duckdb.connect(":memory:")
            self.db_path = ":memory:"

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def load_csv_files(self, csv_files: List[str]) -> Dict[str, Any]:
        loaded_tables = []
        errors = []
        for csv_file in csv_files:
            try:
                table_name = os.path.splitext(os.path.basename(csv_file))[0]
                table_name = re.sub(r'[^a-zA-Z0-9_]', '_', table_name)
                if not table_name or table_name[0].isdigit():
                    table_name = f"table_{table_name}"
                
                self.conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_csv_auto(?)", [csv_file])
                table_info = self.get_table_info(table_name)
                loaded_tables.append({
                    "table_name": table_name,
                    "file_path": csv_file,
                    "row_count": table_info["row_count"],
                    "columns": table_info["columns"]
                })
                
            except Exception as e:
                error_msg = f"Failed to load {csv_file}: {e}"
                errors.append(error_msg)

        return {
            "loaded_tables": loaded_tables,
            "errors": errors,
            "success_count": len(loaded_tables),
            "error_count": len(errors)
        }

    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        try:
            result = self.conn.execute(f"SELECT COUNT(*) as count FROM {table_name}").fetchone()
            row_count = result[0] if result else 0
            result = self.conn.execute(f"DESCRIBE {table_name}").fetchall()
            columns = [{"name": row[0], "type": row[1]} for row in result]
            return {"row_count": row_count, "columns": columns}
        except:
            return {"row_count": 0, "columns": []}

    def get_all_tables(self) -> List[Dict[str, Any]]:
        try:
            result = self.conn.execute("SHOW TABLES").fetchall()
            tables = []
            for row in result:
                table_name = row[0]
                table_info = self.get_table_info(table_name)
                tables.append({"table_name": table_name, **table_info})
            return tables
        except:
            return []

    def natural_language_to_sql(self, question: str, table_context: Optional[str] = None) -> str:
        tables_info = self.get_all_tables()
        if not tables_info:
            raise ValueError("No tables available for querying")
        
        context_parts = []
        for table in tables_info:
            columns_str = ", ".join([f"{col['name']} ({col['type']})" for col in table['columns']])
            context_parts.append(f"Table '{table['table_name']}': {columns_str} ({table['row_count']} rows)")

        tables_context = "\n".join(context_parts)
        if table_context:
            tables_context = f"Focus on table: {table_context}\n\nAll tables:\n{tables_context}"

        prompt = f"""You are a SQL expert for DuckDB. Convert the user's natural language question to a valid DuckDB SQL query.

Available tables and their schemas:
{tables_context}

User question: {question}

Requirements:
- Generate ONLY a valid DuckDB SQL query
- Use proper table and column names as shown above
- Include appropriate WHERE clauses, JOINs, and aggregations as needed
- Use LIMIT clause if the result might be large
- Do not include any explanations or markdown formatting
- Ensure the query is safe and doesn't contain any dangerous operations

SQL Query:"""

        AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
        BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")
        bedrock_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
        
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 500,
            "temperature": 0.1,
            "messages": [{"role": "user", "content": prompt}]
        }

        try:
            resp = bedrock_client.invoke_model(
                modelId=BEDROCK_MODEL_ID,
                body=json.dumps(request_body),
                contentType="application/json"
            )
            response_body = json.loads(resp['body'].read().decode("utf-8"))
            sql_query = response_body['content'][0]['text'].strip()
            
            if sql_query.startswith("```sql"):
                sql_query = sql_query[6:]
            if sql_query.endswith("```"):
                sql_query = sql_query[:-3]
            sql_query = sql_query.strip()
            
            return sql_query
        except Exception as e:
            raise

    def validate_sql(self, sql_query: str) -> Tuple[bool, Optional[str]]:
        try:
            parsed = sqlparse.parse(sql_query)
            if not parsed:
                return False, "Empty SQL query"
            
            # DML transactions 
            dangerous_keywords = [
                "DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE", "TRUNCATE",
                "EXEC", "EXECUTE", "CALL", "GRANT", "REVOKE", "REMOVE"
            ]
            
            upper_sql = sql_query.upper()
            for keyword in dangerous_keywords:
                if keyword in upper_sql:
                    return False, f"Dangerous operation detected: {keyword}"
            
            try:
                self.conn.execute(f"EXPLAIN {sql_query}")
                return True, None
            except Exception as e:
                return False, f"SQL syntax error: {e}"
        except Exception as e:
            return False, f"Validation error: {e}"

    def execute_sql(self, sql_query: str) -> Tuple[bool, Optional[pd.DataFrame], Optional[str]]:
        try:
            is_valid, error_msg = self.validate_sql(sql_query)
            if not is_valid:
                return False, None, error_msg
            
            result = self.conn.execute(sql_query)
            df = result.df()
            gc.collect()
            return True, df, None
        except Exception as e:
            error_msg = f"SQL execution failed: {e}"
            return False, None, error_msg

    def get_table_sample(self, table_name: str, limit: int = 5) -> Optional[pd.DataFrame]:
        try:
            result = self.conn.execute(f"SELECT * FROM {table_name} LIMIT {limit}")
            return result.df()
        except:
            return None

    def close(self):
        if self.conn:
            self.conn.close()


# Main Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📄 PDF Q&A", "📊 CSV SQL", "📈 Analytics", "⚙️ Cache Management"])

# ===========================================
# TAB 1: PDF Q&A
# ===========================================
with tab1:
    st.header("📄 PDF Document Q&A")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        mode = st.radio("Select Mode:", ["Index New Documents", "Query Existing Documents"], index=0, horizontal=True)
    with col2:
        st.empty()

    if mode == "Index New Documents":
        st.subheader("📤 Ingest PDFs (Local Processing)")
        
        col_a, col_b = st.columns([3, 1])
        with col_a:
            uploaded_files = st.file_uploader(
                "Upload one or more PDFs", 
                type="pdf", 
                accept_multiple_files=True,
                help="Drag and drop files here (Limit 50MB per file)"
            )
        with col_b:
            st.info("💡 Browse files")

        directory_path = st.text_input("Or provide a local directory path of PDFs")

        if uploaded_files or directory_path:
            if st.button("🚀 Run Ingest", type="primary"):
                uploaded_files_local = []
                
                if uploaded_files:
                    for file in uploaded_files:
                        if file.size > 50 * 1024 * 1024:  # 50MB limit
                            st.error(f"❌ {file.name} exceeds 50MB limit!")
                            continue
                        local_path = os.path.join(os.getcwd(), file.name)
                        with open(local_path, "wb") as f:
                            f.write(file.getbuffer())
                        uploaded_files_local.append(local_path)
                        st.success(f"✅ Prepared {file.name} locally")
                
                if directory_path and os.path.exists(directory_path):
                    for f in os.listdir(directory_path):
                        if f.lower().endswith(".pdf"):
                            uploaded_files_local.append(os.path.join(directory_path, f))
                    st.success(f"✅ Prepared {len(uploaded_files_local)} files from {directory_path}")
                
                if uploaded_files_local:
                    with st.container():
                        st.markdown("""
                        <div style="background-color: #E3F2FD; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #2196F3;">
                            <h4 style="margin-top: 0;">🔄 Indexing Process:</h4>
                            <ol>
                                <li>📄 Extract text locally</li>
                                <li>🧠 Create embeddings</li>
                                <li>💾 Store in memory DB</li>
                                <li>✅ Ready for querying</li>
                            </ol>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.info("⏳ Extracting text and creating embeddings...")
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        build_or_update_index(uploaded_files_local)
                        progress_bar.progress(100)
                        status_text.success("✅ All embeddings built and index updated!")
                        
                        # # Clean up local files
                        # for path in uploaded_files_local:
                        #     if os.path.exists(path):
                        #         os.remove(path)
                    except Exception as e:
                        st.error(f"❌ Error processing PDFs: {e}")

    elif mode == "Query Existing Documents":
        st.subheader("❓ Query Documents")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input("Enter your query", placeholder="What would you like to know?")
        with col2:
            top_k = st.slider("Top results", 1, 5, 3)

        if st.button("🔍 Search", type="primary") and query.strip():
            start_time = time.time()
            
            try:
                with st.spinner("🤔 Thinking..."):
                    # Use enhanced query with metadata
                    result_raw = query_rag_with_metadata(query, top_k)
                    result = _decimal_to_primitive(result_raw)
                    
                response_time = time.time() - start_time
                answer = result["answer"]
                cached = result["cached"]
                query_hash = result["query_hash"]
                
                # Display cache status
                if cached:
                    st.markdown(f'<p class="cache-hit">💾 Cache Hit - Instant Response!</p>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<p class="cache-miss">🔄 Cache Miss - Generated New Response</p>', unsafe_allow_html=True)
                
                # Display answer
                st.markdown("### 💬 Answer:")
                st.write(answer)
                
                # Display metadata
                col_meta1, col_meta2, col_meta3 = st.columns(3)
                with col_meta1:
                    st.metric("Response Time", f"{response_time:.2f}s")
                with col_meta2:
                    st.metric("Cache Status", "Hit" if cached else "Miss")
                with col_meta3:
                    st.metric("Access Count", result.get("access_count", 0))
                
                # Show sources
                if result.get("sources"):
                    with st.expander("📚 View Sources"):
                        for source in result["sources"]:
                            st.write(f"- 📄 {source}")
                
                # Store in query history
                st.session_state.query_history.append({
                    "query": query,
                    "answer": answer,
                    "cached": cached,
                    "response_time": response_time,
                    "timestamp": time.time(),
                    "query_hash": query_hash
                })
                
                # Feedback section
                st.divider()
                col_thumbs_up, col_thumbs_down, col_text = st.columns([1, 1, 3])
                
                with col_thumbs_up:
                    if st.button("👍 Like", key=f"thumbs_up_{query_hash}"):
                        st.session_state.feedback.append({
                            'query': query,
                            'answer': answer,
                            'feedback': 'positive',
                            'timestamp': time.time(),
                            'query_hash': query_hash
                        })
                        st.success("✅ Thanks for the feedback!")
                
                with col_thumbs_down:
                    if st.button("👎 Dislike", key=f"thumbs_down_{query_hash}"):
                        feedback_text = st.text_input("What can we improve?", key=f"feedback_text_{query_hash}")
                        st.session_state.feedback.append({
                            'query': query,
                            'answer': answer,
                            'feedback': 'negative',
                            'timestamp': time.time(),
                            'details': feedback_text,
                            'query_hash': query_hash
                        })
                        st.warning("⚠️ Thanks for the feedback! We'll improve.")
                
                with col_text:
                    st.empty()
                    
            except FileNotFoundError:
                st.error("❌ No documents processed yet. Please index documents first.")
            except Exception as e:
                st.error(f"❌ Error querying system: {e}")

# ===========================================
# TAB 2: CSV SQL
# ===========================================
with tab2:
    st.header("📊 CSV SQL Analysis")
    
    # Initialize CSV handler
    if st.session_state.csv_handler is None:
        st.session_state.csv_handler = CSVSqlHandler()
    
    csv_handler = st.session_state.csv_handler
    
    # File upload section
    st.subheader("📁 Load CSV Files")
    csv_files = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)
    
    if csv_files and st.button("📥 Load CSVs", type="primary"):
        workspace = os.path.join(os.getcwd(), "csv_workspace")
        os.makedirs(workspace, exist_ok=True)
        file_paths = []
        
        for f in csv_files:
            csv_path = os.path.join(workspace, f.name)
            with open(csv_path, "wb") as w:
                w.write(f.read())
            file_paths.append(csv_path)
        
        with st.spinner("⏳ Loading CSV files..."):
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
        with st.spinner("⏳ Generating SQL..."):
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
                        with st.spinner("⏳ Executing query..."):
                            success, df, exec_error = csv_handler.execute_sql(sql_query)
                            if success:
                                st.subheader("📊 Results")
                                st.dataframe(df, use_container_width=True)
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
        with st.spinner("⏳ Executing..."):
            success, df, error_msg = csv_handler.execute_sql(manual_sql)
            if success:
                st.dataframe(df, use_container_width=True)
            else:
                st.error(f"❌ {error_msg}")

# ===========================================
# TAB 3: Analytics
# ===========================================
with tab3:
    st.header("📈 Analytics & Insights")
    
    # Cache statistics
    if enable_cache:
        st.subheader("💾 Cache Statistics")
        cache_stats = _decimal_to_primitive(get_cache_stats())
        
        if "error" not in cache_stats:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Entries", cache_stats.get("total_entries", 0))
            with col2:
                avg_access = cache_stats.get("avg_access_count", 0)
                try:
                    avg_access_f = float(avg_access)
                except Exception:
                    avg_access_f = 0.0
                st.metric("Avg Access Count", f"{avg_access_f:.1f}")

            with col3:
                max_access = cache_stats.get("max_access_count", 0)
                if isinstance(max_access, decimal.Decimal):
                    max_access = int(max_access)  # optional now because of sanitizer
                st.metric("Max Access Count", max_access)
            with col4:
                age = cache_stats.get("oldest_entry_age_hours", 0)
                st.metric("Oldest Entry (hrs)", f"{age:.1f}" if age else "N/A")
        else:
            st.warning("Cache statistics unavailable")
    
    # Query history
    if st.session_state.query_history:
        st.subheader("📊 Query History")
        history_df = pd.DataFrame(st.session_state.query_history)
        
        # Show cache hit rate
        cached_count = sum(1 for q in st.session_state.query_history if q.get("cached", False))
        total_count = len(st.session_state.query_history)
        cache_hit_rate = cached_count / total_count if total_count > 0 else 0
        
        col_metric1, col_metric2, col_metric3 = st.columns(3)
        with col_metric1:
            st.metric("Total Queries", total_count)
        with col_metric2:
            st.metric("Cache Hit Rate", f"{cache_hit_rate:.1%}")
        with col_metric3:
            avg_response = history_df["response_time"].mean()
            st.metric("Avg Response Time", f"{avg_response:.2f}s")
        
        # Response time visualization
        st.subheader("⏱️ Response Time Trend")
        chart_data = pd.DataFrame({
            "Query": range(len(history_df)),
            "Response Time (s)": history_df["response_time"],
            "Cached": history_df["cached"].map({True: "Hit", False: "Miss"})
        })
        
        import plotly.express as px
        fig = px.line(chart_data, x="Query", y="Response Time (s)", color="Cached",
                     title="Response Time Over Queries")
        st.plotly_chart(fig, use_container_width=True)
        
        # Show recent queries
        st.subheader("🕒 Recent Queries")
        for i, query in enumerate(reversed(st.session_state.query_history[-5:])):
            cache_badge = "💾 Cached" if query.get("cached", False) else "🔄 Fresh"
            with st.expander(f"Query {total_count - i}: {query['query'][:50]}... | {cache_badge}"):
                st.write(f"**Answer:** {query['answer'][:200]}...")
                st.write(f"**Response Time:** {query.get('response_time', 0):.2f}s")
                st.write(f"**Timestamp:** {datetime.fromtimestamp(query['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Feedback statistics
    st.subheader("👍👎 Feedback Statistics")
    if st.session_state.feedback:
        positive = sum(1 for f in st.session_state.feedback if f['feedback'] == 'positive')
        negative = sum(1 for f in st.session_state.feedback if f['feedback'] == 'negative')
        total_fb = len(st.session_state.feedback)
        
        col_fb1, col_fb2, col_fb3 = st.columns(3)
        with col_fb1:
            st.metric("Total Feedback", total_fb)
        with col_fb2:
            st.metric("Positive", f"{positive} ({positive/total_fb*100:.0f}%)")
        with col_fb3:
            st.metric("Negative", f"{negative} ({negative/total_fb*100:.0f}%)")
    else:
        st.info("No feedback data available yet")

# ===========================================
# TAB 4: Cache Management
# ===========================================
with tab4:
    st.header("⚙️ Cache Management")
    
    if not enable_cache:
        st.warning("⚠️ Cache is currently disabled. Enable it in the sidebar to use cache management features.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🧹 Cache Operations")
            
            if st.button("🗑️ Clear All Cache", type="secondary"):
                with st.spinner("⏳ Clearing cache..."):
                    deleted = invalidate_cache()
                    st.success(f"✅ Deleted {deleted} cache entries")
                    st.rerun()
            
            if st.button("🧽 Clean Expired Entries"):
                with st.spinner("⏳ Cleaning..."):
                    cleaned = cleanup_expired_cache()
                    st.success(f"✅ Cleaned {cleaned} expired entries")
                    st.rerun()
            
            pattern = st.text_input("Clear by pattern (optional):", placeholder="Enter pattern to match")
            if st.button("🎯 Clear by Pattern") and pattern:
                with st.spinner("⏳ Clearing by pattern..."):
                    deleted = invalidate_cache(pattern=pattern)
                    st.success(f"✅ Deleted {deleted} entries matching pattern")
                    st.rerun()
        
        with col2:
            st.subheader("📊 Cache Details")
            cache_stats = _decimal_to_primitive(get_cache_stats())
            if "error" not in cache_stats:
                st.json(cache_stats)
            else:
                st.error("Cache details unavailable")
        
        # Recent cached queries
        st.divider()
        st.subheader("📋 Recent Cached Queries")
        

        if st.session_state.query_history:
            cached_queries = [q for q in st.session_state.query_history if q.get("cached", False)]
            
            if cached_queries:
                for i, query in enumerate(cached_queries[-10:]):  # ensure unique keys
                    query_hash = query.get("query_hash") or f"nohash_{i}"
                    query_preview = query.get("query", "")[:60]
                    cached_at = datetime.fromtimestamp(query.get("timestamp", 0)).strftime('%Y-%m-%d %H:%M:%S')
                    response_time = query.get("response_time", 0.0)

                    with st.expander(f"🔹 {query_preview}..."):
                        st.markdown(f"**Query Hash:** `{query_hash[:16]}...`")
                        st.markdown(f"**Cached At:** {cached_at}")
                        st.markdown(f"**Response Time:** `{response_time:.2f}s`")

                        # unique key using both index + hash
                        invalidate_key = f"invalidate_{i}_{query_hash}"

                        if st.button("🗑️ Invalidate", key=invalidate_key):
                            invalidate_cache(query_hash=query_hash)
                            st.success(f"✅ Cache entry invalidated for `{query_hash[:8]}...`")
                            st.rerun()
            else:
                st.info("No cached queries yet.")
        else:
            st.info("No query history available.")
