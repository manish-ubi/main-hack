# upload.py
import os
from utils import upload_file_to_s3, list_files_in_s3
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def upload_single(file_path):
    """Upload a single PDF file to S3"""
    filename = os.path.basename(file_path)
    s3_key = f"raw/{filename}"
    upload_file_to_s3(file_path, s3_key)
    print(f"✅ Uploaded {filename} to raw/")

def upload_batch(folder_path):
    """Upload all PDFs in a folder to S3"""
    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} does not exist")
        return
    
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        print("No PDF files found in folder")
        return
    
    print(f"Found {len(pdf_files)} PDF files")
    
    for filename in pdf_files:
        file_path = os.path.join(folder_path, filename)
        try:
            upload_single(file_path)
        except Exception as e:
            print(f"Error uploading {filename}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Single PDF file to upload")
    parser.add_argument("--folder", help="Folder path for batch upload")
    args = parser.parse_args()

    if args.file:
        upload_single(args.file)
    elif args.folder:
        upload_batch(args.folder)
    else:
        print("Provide --file or --folder")
        print("Example: python upload.py --file document.pdf")
        print("Example: python upload.py --folder ./pdfs")
