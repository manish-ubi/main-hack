# watcher.py
import time
import os
from extract_text import process_new_pdfs
from embed import build_or_update_index
from utils import list_files_in_s3, download_file_from_s3
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

CHECK_INTERVAL = 60  # seconds, adjust as needed

def get_new_pdfs_from_s3():
    """Download new PDFs from S3 raw/ folder that haven't been processed yet"""
    raw_files = list_files_in_s3("raw/")
    processed_files = list_files_in_s3("processed/")
    
    processed_names = [os.path.basename(f).replace(".txt", ".pdf") for f in processed_files]
    new_pdfs = []
    
    for s3_key in raw_files:
        filename = os.path.basename(s3_key)
        if filename not in processed_names and filename.lower().endswith(".pdf"):
            try:
                local_pdf = download_file_from_s3(s3_key)
                new_pdfs.append(local_pdf)
                print(f"Downloaded new PDF: {filename}")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
    
    return new_pdfs

def continuous_sync():
    """Continuously watch S3 for new PDFs and process them"""
    print("Starting continuous S3 watcher for new PDFs...")
    print(f"Checking every {CHECK_INTERVAL} seconds")
    
    while True:
        try:
            print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Checking for new files...")
            
            # Process text extraction
            process_new_pdfs()
            
            # Get new PDFs and update embeddings
            new_pdfs = get_new_pdfs_from_s3()
            
            if new_pdfs:
                print(f"Found {len(new_pdfs)} new PDFs. Building embeddings...")
                build_or_update_index(new_pdfs)
                print("✅ Embeddings updated successfully")
                
                # Clean up downloaded files
                for pdf in new_pdfs:
                    try:
                        os.remove(pdf)
                    except:
                        pass
            else:
                print("No new files found")
                
        except Exception as e:
            print(f"❌ Error during sync: {e}")
            import traceback
            traceback.print_exc()
        
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    continuous_sync()
