# ingest_policies.py
import logging
from PyPDF2 import PdfReader
from rag_handler import (
    embeddings, 
    pc, 
    text_splitter,
    extract_section
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

index = pc.Index("edu-staff-policies-v1")  # Your existing index

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file using PyPDF2."""
    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {str(e)}")
        raise

def chunk_and_upload(file_path: str, source: str = "edu-staff-policies-v1"):
    """Process policy documents and upload to Pinecone"""
    try:
        logger.info(f"Processing file: {file_path}")
        
        # 1. Extract text from PDF (using PyPDF2)
        text = extract_text_from_pdf(file_path)
        logger.info(f"Extracted text (approx. {len(text)//1024} KB)")
        
        # 2. Split into chunks
        chunks = text_splitter.split_text(text)
        logger.info(f"Generated {len(chunks)} chunks")
        
        # 3. Prepare vectors for Pinecone
        vectors = []
        for i, chunk in enumerate(chunks):
            section = extract_section(chunk)
            if i % 2 == 0:  # Log every 20th chunk
                logger.debug(f"Processing chunk {i}: Section {section[:30]}...")
            
            vectors.append({
                "id": f"{source}-{i}",
                "values": embeddings.embed_documents([chunk])[0],
                "metadata": {
                    "text": chunk,
                    "source": source,
                    "section": section
                }
            })
        
        # 4. Upload to Pinecone in batches
        logger.info(f"Uploading {len(vectors)} vectors to Pinecone")
        for i in range(0, len(vectors), 100):
            batch = vectors[i:i+100]
            index.upsert(vectors=batch)
            logger.info(f"Uploaded batch {i//100 + 1} ({len(batch)} vectors)")
        
        logger.info("Upload completed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Failed to process {file_path}: {str(e)}")
        return False

if __name__ == "__main__":
    chunk_and_upload("./policies/staff_policy_v1.pdf", "edu-staff-policies-v1")