import os
import glob
from pypdf import PdfReader
import docx
import chromadb
from chromadb.utils import embedding_functions

from config import DATA_DIR, DB_DIR, COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP

def extract_pdf_pages(file_path: str) -> list[dict]:
    extracted_data = []
    file_name = os.path.basename(file_path)
    try:
        reader = PdfReader(file_path)
        for index, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                clean_text = " ".join(text.split())
                extracted_data.append({
                    "text": clean_text,
                    "metadata": {"source": file_name, "page": index + 1}
                })
    except Exception as e:
        print(f"Error reading PDF {file_name}: {e}")
    return extracted_data

def extract_docx_pages(file_path: str) -> list[dict]:
    extracted_data = []
    file_name = os.path.basename(file_path)
    try:
        doc = docx.Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text.strip())
        combined_text = " ".join(full_text)
        if combined_text:
            extracted_data.append({
                "text": combined_text,
                "metadata": {"source": file_name, "page": 1}
            })
    except Exception as e:
        print(f"Error reading DOCX {file_name}: {e}")
    return extracted_data

def chunk_extracted_pages(pages: list[dict], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[dict]:
    chunks = []
    for page in pages:
        text = page["text"]
        metadata = page["metadata"]
        start = 0
        text_length = len(text)
        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk_text = text[start:end]
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": metadata["source"],
                    "page": metadata["page"],
                    "chunk_range": f"{start}-{end}"
                }
            })
            start += (chunk_size - chunk_overlap)
    return chunks

def build_vector_database():
    all_pages = []
    pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    docx_files = glob.glob(os.path.join(DATA_DIR, "*.docx"))
    
    print("Scanning data directory...")
    for file in pdf_files:
        all_pages.extend(extract_pdf_pages(file))
    for file in docx_files:
        all_pages.extend(extract_docx_pages(file))
        
    if not all_pages:
        print("No documents found in data/ folder.")
        return

    chunks = chunk_extracted_pages(all_pages)
    print(f"Processing {len(chunks)} chunks using local default embedding function...")
    
    documents = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [f"id_{i}" for i in range(len(chunks))]

    client = chromadb.PersistentClient(path=str(DB_DIR))
    
    # Use Chroma's local built-in default embedding library
    default_ef = embedding_functions.DefaultEmbeddingFunction()
    
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=default_ef
    )
    
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )
    print(f"Success! Locally indexed {len(chunks)} document chunks without API requests.")

if __name__ == "__main__":
    build_vector_database()