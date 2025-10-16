# extract_text.py
import os
from utils import list_files_in_s3, download_file_from_s3, extract_text_from_pdf, upload_file_to_s3
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def process_new_pdfs():
    """Process new PDFs from S3 raw/ folder and save text to processed/"""
    raw_files = list_files_in_s3("raw/")
    processed_files = list_files_in_s3("processed/")
    
    processed_names = [os.path.basename(f) for f in processed_files]

    for s3_key in raw_files:
        filename = os.path.basename(s3_key)
        txt_filename = filename.replace(".pdf", ".txt")
        
        if txt_filename in processed_names:
            continue  # Skip already processed
        
        try:
            print(f"Processing {filename}...")
            local_pdf = download_file_from_s3(s3_key)
            text = extract_text_from_pdf(local_pdf)
            
            local_txt = local_pdf.replace(".pdf", ".txt")
            with open(local_txt, "w", encoding="utf-8") as f:
                f.write(text)
            
            upload_file_to_s3(local_txt, f"processed/{txt_filename}")
            print(f"✅ Processed {filename}")
            
            # Clean up local files
            os.remove(local_pdf)
            os.remove(local_txt)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    process_new_pdfs()
