# # # main.py
# # import os
# # import streamlit as st
# # # from embed import build_or_update_index
# # from embed import build_or_update_index
# # from embed import client


# # from upload import upload_single, upload_batch
# # from query import query_rag_system
# # from dotenv import load_dotenv

# # load_dotenv()
# # st.set_page_config(page_title="📄 RAG System", layout="wide")
# # st.title("📄 RAG System with AWS Bedrock & S3")

# # # ----------------------
# # # Sidebar: Upload PDFs
# # # ----------------------
# # st.sidebar.header("📤 Upload PDFs")
# # upload_mode = st.sidebar.radio("Mode", ["Single File", "Batch Folder"])
# # uploaded_files_local = []

# # log_container = st.sidebar.empty()
# # progress_bar = st.sidebar.progress(0)

# # def log(msg, step=None, total=None):
# #     log_container.text(msg)
# #     if step and total:
# #         progress_bar.progress(min(int((step/total)*100), 100))

# # # ----------------------
# # # Upload PDFs
# # # ----------------------
# # if upload_mode == "Single File":
# #     file = st.sidebar.file_uploader("Upload PDF", type="pdf")
# #     if file and st.sidebar.button("Upload & Process"):
# #         local_path = os.path.join(os.getcwd(), file.name)
# #         with open(local_path, "wb") as f:
# #             f.write(file.getbuffer())
# #         upload_single(local_path)
# #         uploaded_files_local.append(local_path)
# #         st.sidebar.success(f"✅ Uploaded {file.name}")

# # elif upload_mode == "Batch Folder":
# #     folder = st.sidebar.text_input("Folder Path")
# #     if folder and st.sidebar.button("Upload Batch"):
# #         if not os.path.exists(folder):
# #             st.sidebar.error("Folder does not exist!")
# #         else:
# #             upload_batch(folder)
# #             for f in os.listdir(folder):
# #                 if f.lower().endswith(".pdf"):
# #                     uploaded_files_local.append(os.path.join(folder, f))
# #             st.sidebar.success(f"✅ Batch uploaded from {folder}")

# # # ----------------------
# # # Build Index Immediately
# # # ----------------------
# # if uploaded_files_local:
# #     st.info("🔄 Extracting text and creating embeddings...")
# #     try:
# #         build_or_update_index(uploaded_files_local)
# #         st.success("✅ All embeddings built and index updated!")
# #         uploaded_files_local = []
# #         progress_bar.progress(100)
# #     except Exception as e:
# #         st.error(f"❌ Error processing PDFs: {e}")

# # # ----------------------
# # # Query RAG system
# # # ----------------------
# # st.header("🔍 Query Documents")
# # query = st.text_input("Enter your query", placeholder="What would you like to know?")
# # top_k = st.slider("Number of results", 1, 10, 5)

# # if st.button("🔎 Search"):
# #     if query.strip() == "":
# #         st.warning("⚠️ Enter a query first!")
# #     else:
# #         try:
# #             with st.spinner("Querying RAG system..."):
# #                 answer = query_rag_system(query, top_k)
# #             st.write("### 📝 Answer:")
# #             st.write(answer)
# #         except FileNotFoundError:
# #             st.error("❌ No documents processed yet.")
# #         except Exception as e:
# #             st.error(f"❌ Error querying system: {e}")

# # # ----------------------
# # # ----------------------
# # # Show index stats
# # # ----------------------
# # with st.sidebar:
# #     st.divider()
# #     st.subheader("📊 Index Stats")

# #     # Get Chroma collection
# #     collection = client.get_collection("pdf_docs")
# #     results = collection.get(include=["documents", "metadatas"])  # documents + metadata

# #     total_chunks = len(results["ids"])

# #     # Ensure metadata is dict and has "file"
# #     unique_files = set()
# #     for m in results["metadatas"]:
# #         if isinstance(m, dict) and "file" in m:
# #             unique_files.add(m["file"])

# #     st.metric("Total Chunks", total_chunks)
# #     st.metric("Documents", len(unique_files))

# #     with st.expander("View Documents"):
# #         for doc in sorted(unique_files):
# #             st.text(f"📄 {doc}")


# # main.py (updated)
# import os
# import streamlit as st
# from embed import build_or_update_index, client
# from upload import upload_single, upload_batch
# from query import query_rag_system
# from dotenv import load_dotenv

# load_dotenv()
# st.set_page_config(page_title="📄 RAG System", layout="wide")
# st.title("📄 RAG System with AWS Bedrock & S3")

# # Initialize session state for caching if not exists
# if 'query_cache' not in st.session_state:
#     st.session_state.query_cache = {}

# # ----------------------
# # Sidebar: Upload PDFs
# # ----------------------
# st.sidebar.header("📤 Upload PDFs")
# upload_mode = st.sidebar.radio("Mode", ["Single File", "Batch Folder"])
# uploaded_files_local = []

# log_container = st.sidebar.empty()
# progress_bar = st.sidebar.progress(0)

# def log(msg, step=None, total=None):
#     log_container.text(msg)
#     if step and total:
#         progress_bar.progress(min(int((step/total)*100), 100))

# # ----------------------
# # Upload PDFs
# # ----------------------
# if upload_mode == "Single File":
#     file = st.sidebar.file_uploader("Upload PDF", type="pdf")
#     if file and st.sidebar.button("Upload & Process"):
#         local_path = os.path.join(os.getcwd(), file.name)
#         with open(local_path, "wb") as f:
#             f.write(file.getbuffer())
#         upload_single(local_path)
#         uploaded_files_local.append(local_path)
#         st.sidebar.success(f"✅ Uploaded {file.name}")

# elif upload_mode == "Batch Folder":
#     folder = st.sidebar.text_input("Folder Path")
#     if folder and st.sidebar.button("Upload Batch"):
#         if not os.path.exists(folder):
#             st.sidebar.error("Folder does not exist!")
#         else:
#             upload_batch(folder)
#             for f in os.listdir(folder):
#                 if f.lower().endswith(".pdf"):
#                     uploaded_files_local.append(os.path.join(folder, f))
#             st.sidebar.success(f"✅ Batch uploaded from {folder}")

# # ----------------------
# # Build Index Immediately
# # ----------------------
# if uploaded_files_local:
#     st.info("🔄 Extracting text and creating embeddings...")
#     try:
#         build_or_update_index(uploaded_files_local)
#         st.success("✅ All embeddings built and index updated!")
#         uploaded_files_local = []
#         progress_bar.progress(100)
#     except Exception as e:
#         st.error(f"❌ Error processing PDFs: {e}")

# # ----------------------
# # Query RAG system
# # ----------------------
# st.header("🔍 Query Documents")
# query = st.text_input("Enter your query", placeholder="What would you like to know?")
# top_k = st.slider("Number of results", 1, 10, 5)

# if st.button("🔎 Search"):
#     if query.strip() == "":
#         st.warning("⚠️ Enter a query first!")
#     else:
#         # Caching logic: case-insensitive query + top_k
#         cache_key = query.lower().strip() + "_" + str(top_k)
#         if cache_key in st.session_state.query_cache:
#             st.info("📦 Using cached response...")
#             st.write("### 📝 Answer:")
#             st.write(st.session_state.query_cache[cache_key])
#         else:
#             try:
#                 with st.spinner("Querying RAG system..."):
#                     answer = query_rag_system(query, top_k)
#                 st.session_state.query_cache[cache_key] = answer
#                 st.write("### 📝 Answer:")
#                 st.write(answer)
#             except FileNotFoundError:
#                 st.error("❌ No documents processed yet.")
#             except Exception as e:
#                 st.error(f"❌ Error querying system: {e}")

# # ----------------------
# # Show index stats
# # ----------------------
# with st.sidebar:
#     st.divider()
#     st.subheader("📊 Index Stats")

#     # Get Chroma collection
#     collection = client.get_collection("pdf_docs")
#     results = collection.get(include=["documents", "metadatas"])  # documents + metadata

#     total_chunks = len(results["ids"])

#     # Ensure metadata is dict and has "file"
#     unique_files = set()
#     for m in results["metadatas"]:
#         if isinstance(m, dict) and "file" in m:
#             unique_files.add(m["file"])

#     st.metric("Total Chunks", total_chunks)
#     st.metric("Documents", len(unique_files))

#     with st.expander("View Documents"):
#         for doc in sorted(unique_files):
#             st.text(f"📄 {doc}")



# main.py (enhanced with tabs, cache management, analytics, and UI improvements)
import os
import streamlit as st
import time
from datetime import datetime
from embed import build_or_update_index, client
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
    # CSV SQL Tab - Placeholder
    st.header("📊 CSV SQL")
    st.info("🔄 CSV SQL functionality coming soon. Upload CSVs and query with natural language SQL.")

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
    
    # Feedback Statistics
    st.subheader("👍👎 Feedback Statistics")
    if st.session_state.feedback:
        positive = sum(1 for f in st.session_state.feedback if f['feedback'] == 'positive')
        negative = sum(1 for f in st.session_state.feedback if f['feedback'] == 'negative')
        total = len(st.session_state.feedback)
        col_fb1, col_fb2 = st.columns(2)
        col_fb1.metric("Positive Feedback", positive)
        col_fb2.metric("Negative Feedback", negative)
        st.caption(f"Total Feedback: {total}")
    else:
        st.info("📭 Feedback statistics unavailable")

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