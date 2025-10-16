# # main.py
# import os
# import streamlit as st
# # from embed import build_or_update_index
# from embed import build_or_update_index
# from embed import client


# from upload import upload_single, upload_batch
# from query import query_rag_system
# from dotenv import load_dotenv

# load_dotenv()
# st.set_page_config(page_title="📄 RAG System", layout="wide")
# st.title("📄 RAG System with AWS Bedrock & S3")

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
#         try:
#             with st.spinner("Querying RAG system..."):
#                 answer = query_rag_system(query, top_k)
#             st.write("### 📝 Answer:")
#             st.write(answer)
#         except FileNotFoundError:
#             st.error("❌ No documents processed yet.")
#         except Exception as e:
#             st.error(f"❌ Error querying system: {e}")

# # ----------------------
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
# main.py (updated)
import os
import streamlit as st
from embed import build_or_update_index, client
from upload import upload_single, upload_batch
from query import query_rag_system
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="📄 RAG System", layout="wide")
st.title("📄 RAG System with AWS Bedrock & S3")

# Initialize session state for caching if not exists
if 'query_cache' not in st.session_state:
    st.session_state.query_cache = {}

# ----------------------
# Sidebar: Upload PDFs
# ----------------------
st.sidebar.header("📤 Upload PDFs")
upload_mode = st.sidebar.radio("Mode", ["Single File", "Batch Folder"])
uploaded_files_local = []

log_container = st.sidebar.empty()
progress_bar = st.sidebar.progress(0)

def log(msg, step=None, total=None):
    log_container.text(msg)
    if step and total:
        progress_bar.progress(min(int((step/total)*100), 100))

# ----------------------
# Upload PDFs
# ----------------------
if upload_mode == "Single File":
    file = st.sidebar.file_uploader("Upload PDF", type="pdf")
    if file and st.sidebar.button("Upload & Process"):
        local_path = os.path.join(os.getcwd(), file.name)
        with open(local_path, "wb") as f:
            f.write(file.getbuffer())
        upload_single(local_path)
        uploaded_files_local.append(local_path)
        st.sidebar.success(f"✅ Uploaded {file.name}")

elif upload_mode == "Batch Folder":
    folder = st.sidebar.text_input("Folder Path")
    if folder and st.sidebar.button("Upload Batch"):
        if not os.path.exists(folder):
            st.sidebar.error("Folder does not exist!")
        else:
            upload_batch(folder)
            for f in os.listdir(folder):
                if f.lower().endswith(".pdf"):
                    uploaded_files_local.append(os.path.join(folder, f))
            st.sidebar.success(f"✅ Batch uploaded from {folder}")

# ----------------------
# Build Index Immediately
# ----------------------
if uploaded_files_local:
    st.info("🔄 Extracting text and creating embeddings...")
    try:
        build_or_update_index(uploaded_files_local)
        st.success("✅ All embeddings built and index updated!")
        uploaded_files_local = []
        progress_bar.progress(100)
    except Exception as e:
        st.error(f"❌ Error processing PDFs: {e}")

# ----------------------
# Query RAG system
# ----------------------
st.header("🔍 Query Documents")
query = st.text_input("Enter your query", placeholder="What would you like to know?")
top_k = st.slider("Number of results", 1, 10, 5)

if st.button("🔎 Search"):
    if query.strip() == "":
        st.warning("⚠️ Enter a query first!")
    else:
        # Caching logic: case-insensitive query + top_k
        cache_key = query.lower().strip() + "_" + str(top_k)
        if cache_key in st.session_state.query_cache:
            st.info("📦 Using cached response...")
            st.write("### 📝 Answer:")
            st.write(st.session_state.query_cache[cache_key])
        else:
            try:
                with st.spinner("Querying RAG system..."):
                    answer = query_rag_system(query, top_k)
                st.session_state.query_cache[cache_key] = answer
                st.write("### 📝 Answer:")
                st.write(answer)
            except FileNotFoundError:
                st.error("❌ No documents processed yet.")
            except Exception as e:
                st.error(f"❌ Error querying system: {e}")

# ----------------------
# Show index stats
# ----------------------
with st.sidebar:
    st.divider()
    st.subheader("📊 Index Stats")

    # Get Chroma collection
    collection = client.get_collection("pdf_docs")
    results = collection.get(include=["documents", "metadatas"])  # documents + metadata

    total_chunks = len(results["ids"])

    # Ensure metadata is dict and has "file"
    unique_files = set()
    for m in results["metadatas"]:
        if isinstance(m, dict) and "file" in m:
            unique_files.add(m["file"])

    st.metric("Total Chunks", total_chunks)
    st.metric("Documents", len(unique_files))

    with st.expander("View Documents"):
        for doc in sorted(unique_files):
            st.text(f"📄 {doc}")