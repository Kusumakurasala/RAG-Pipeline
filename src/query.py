import os
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv

from config import DB_DIR, LLM_MODEL, COLLECTION_NAME, TOP_K_RESULTS

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

def retrieve_and_answer(user_query: str) -> tuple[str, list]:
    # 1. Generate query embedding using explicit model path mapping
    response = genai.embed_content(
        model="text-embedding-004",  # Fixed direct string query embedding
        content=user_query,
        task_type="retrieval_query"
    )
    query_vector = response['embedding']
    
    # 2. Search Database
    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=TOP_K_RESULTS
        )
    except Exception:
        return "The local knowledge vector database hasn't been built. Please run python src/ingest.py first!", []
        
    if not results or not results['documents'] or len(results['documents'][0]) == 0:
        return "No relevant information matching your question was found in the files.", []
        
    # 3. Construct Context Prompt
    retrieved_docs = results['documents'][0]
    sources = results['metadatas'][0]
    
    context_str = "\n---\n".join(retrieved_docs)
    
    system_prompt = (
        "You are an expert document assistant. Answer the user's question accurately "
        "using ONLY the provided text segment context down below. If the answer cannot be found "
        "in the context, politely state that you do not know.\n\n"
        f"Context Information:\n{context_str}"
    )
    
    # 4. Request Gemini Answer
    model = genai.GenerativeModel(LLM_MODEL)
    ai_response = model.generate_content(
        contents=[system_prompt, f"User Question: {user_query}"]
    )
    
    # Bundle matching references
    references = []
    for doc, src in zip(retrieved_docs, sources):
        references.append({
            "text": doc,
            "source": src.get("source", "Unknown"),
            "page": src.get("page", "N/A")
        })
        
    return ai_response.text, references