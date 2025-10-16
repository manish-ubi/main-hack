# main.py
import os
import streamlit as st
from query import query_rag_system
from upload import upload_single, upload_batch
from embed import build_or_update_index
from utils import extract_text_from_pdf
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(page_title="📄 RAG System", layout="wide")
st.title("📄 RAG System with AWS Bedrock & S3")

# ----------------------
# Sidebar: Upload PDFs
# ----------------------
st.sidebar.header("📤 Upload PDFs")
upload_mode = st.sidebar.radio("Mode", ["Single File", "Batch Folder"])

uploaded_files_local = []

log_container = st.sidebar.empty()  # container for live logs
progress_bar = st.sidebar.progress(0)

def log(message, step=None, total_steps=None):
    """
    Utility to show log messages and update progress bar
    """
    log_container.text(message)
    if step is not None and total_steps is not None:
        progress_bar.progress(min(int((step / total_steps) * 100), 100))

# ----------------------
# Upload PDFs
# ----------------------
if upload_mode == "Single File":
    file = st.sidebar.file_uploader("Upload PDF", type="pdf")
    if file and st.sidebar.button("Upload & Process"):
        try:
            local_path = os.path.join(os.getcwd(), file.name)
            with open(local_path, "wb") as f:
                f.write(file.getbuffer())

            log("Uploading to S3...")
            upload_single(local_path)
            uploaded_files_local.append(local_path)
            st.sidebar.success(f"✅ Uploaded {file.name}")
        except Exception as e:
            st.sidebar.error(f"Error uploading: {e}")

elif upload_mode == "Batch Folder":
    folder = st.sidebar.text_input("Folder Path")
    if folder and st.sidebar.button("Upload Batch"):
        try:
            if not os.path.exists(folder):
                st.sidebar.error("Folder does not exist!")
            else:
                log("Starting batch upload...")
                upload_batch(folder)
                # Collect all PDFs in folder
                for f in os.listdir(folder):
                    if f.lower().endswith(".pdf"):
                        uploaded_files_local.append(os.path.join(folder, f))
                st.sidebar.success(f"✅ Batch uploaded from {folder}")
        except Exception as e:
            st.sidebar.error(f"Error in batch upload: {e}")

# ----------------------
# Process PDFs & Build Index Immediately
# ----------------------
if uploaded_files_local:
    st.info("🔄 Extracting text and creating embeddings...")

    total_files = len(uploaded_files_local)
    try:
        for i, pdf_path in enumerate(uploaded_files_local, 1):
            log(f"Processing {os.path.basename(pdf_path)} ({i}/{total_files})", i, total_files)
            build_or_update_index([pdf_path])  # build/update FAISS index per file

        st.success("✅ All embeddings built and index updated!")
        uploaded_files_local = []  # Clear the list
        progress_bar.progress(100)
    except Exception as e:
        st.error(f"Error processing PDFs: {e}")

# ----------------------
# Query RAG system
# ----------------------
st.header("🔍 Query Documents")
query = st.text_input("Enter your query", placeholder="What would you like to know?")
top_k = st.slider("Number of results to consider", min_value=1, max_value=10, value=5)

if st.button("🔎 Search", type="primary"):
    if query.strip() == "":
        st.warning("⚠️ Please enter a query first!")
    else:
        try:
            with st.spinner("Querying RAG system..."):
                answer = query_rag_system(query, top_k=top_k)
            st.write("### 📝 Answer:")
            st.write(answer)
        except FileNotFoundError:
            st.error("❌ No documents have been processed yet. Please upload PDFs first.")
        except Exception as e:
            st.error(f"❌ Error querying system: {e}")

# ----------------------
# Show index stats
# ----------------------
with st.sidebar:
    st.divider()
    st.subheader("📊 Index Stats")
    
    try:
        import pickle
        from embed import METADATA_FILE
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, "rb") as f:
                metadata = pickle.load(f)
            st.metric("Total Chunks", len(metadata))
            unique_files = set(m["file"] for m in metadata)
            st.metric("Documents", len(unique_files))
            
            with st.expander("View Documents"):
                for doc in sorted(unique_files):
                    st.text(f"📄 {doc}")
        else:
            st.info("No index created yet")
    except Exception as e:
        st.error(f"Error loading stats: {e}")
