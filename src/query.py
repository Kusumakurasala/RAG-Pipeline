import os
import chromadb
import google.generativeai as genai
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

from config import DB_DIR, LLM_MODEL, COLLECTION_NAME, TOP_K_RESULTS

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

def retrieve_and_answer(user_query: str) -> tuple[str, list]:
    client = chromadb.PersistentClient(path=str(DB_DIR))
    default_ef = embedding_functions.DefaultEmbeddingFunction()
    
    try:
        collection = client.get_collection(
            name=COLLECTION_NAME, 
            embedding_function=default_ef
        )
        # Search Database using the local text query mapping
        results = collection.query(
            query_texts=[user_query],
            n_results=TOP_K_RESULTS
        )
    except Exception as e:
        return f"Database error or missing indexing: {str(e)}", []
        
    if not results or not results['documents'] or len(results['documents'][0]) == 0:
        return "No relevant information matching your question was found in the files.", []
        
    retrieved_docs = results['documents'][0]
    sources = results['metadatas'][0]
    
    context_str = "\n---\n".join(retrieved_docs)
    
    system_prompt = (
        "You are an expert document assistant. Answer the user's question accurately "
        "using ONLY the provided text segment context down below. If the answer cannot be found "
        "in the context, politely state that you do not know.\n\n"
        f"Context Information:\n{context_str}"
    )
    
    # Request Gemini for text generation answer only
    model = genai.GenerativeModel(LLM_MODEL)
    ai_response = model.generate_content(
        contents=[system_prompt, f"User Question: {user_query}"]
    )
    
    references = []
    for doc, src in zip(retrieved_docs, sources):
        references.append({
            "text": doc,
            "source": src.get("source", "Unknown"),
            "page": src.get("page", "N/A")
        })
        
    return ai_response.text, references