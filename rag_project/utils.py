# utils.py
import boto3
import os
from PyPDF2 import PdfReader
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
S3_BUCKET = os.getenv("S3_BUCKET")

s3_client = boto3.client("s3", region_name=AWS_REGION)

def upload_file_to_s3(file_path, s3_key):
    """Upload a file to S3"""
    try:
        s3_client.upload_file(file_path, S3_BUCKET, s3_key)
        print(f"Uploaded {file_path} to s3://{S3_BUCKET}/{s3_key}")
    except Exception as e:
        print(f"Error uploading to S3: {e}")
        raise

def list_files_in_s3(prefix=""):
    """List all files in S3 with given prefix"""
    try:
        resp = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
        return [obj['Key'] for obj in resp.get('Contents', [])]
    except Exception as e:
        print(f"Error listing S3 files: {e}")
        return []

def download_file_from_s3(s3_key, local_path=None):
    """Download a file from S3"""
    if not local_path:
        local_path = os.path.join(tempfile.gettempdir(), os.path.basename(s3_key))
    try:
        s3_client.download_file(S3_BUCKET, s3_key, local_path)
        print(f"Downloaded {s3_key} to {local_path}")
        return local_path
    except Exception as e:
        print(f"Error downloading from S3: {e}")
        raise

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file"""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def chunk_text(text, chunk_size=500):
    """Chunk text by words (~500 words default)"""
    words = text.split()
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return [chunk for chunk in chunks if chunk.strip()]  # Remove empty chunks
