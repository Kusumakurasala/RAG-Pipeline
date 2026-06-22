import streamlit as st
from query import retrieve_and_answer

st.set_page_config(page_title="Document QA Bot", layout="wide")

st.title("📄 Smart Document Retrieval-Augmented Generation Bot")
st.write("Ask questions and get answers directly extracted from your local library knowledge base.")

user_query = st.text_input("Enter your document question here:", placeholder="e.g., What are the main deliverables?")

if user_query:
    with st.spinner("Analyzing knowledge tracks and generating response..."):
        answer, references = retrieve_and_answer(user_query)
        
    st.subheader("💡 Answer")
    st.write(answer)
    
    if references:
        st.subheader("📚 Verified Source References")
        for idx, ref in enumerate(references):
            with st.expander(f"Reference #{idx+1} — File: {ref['source']} (Page {ref['page']})"):
                st.write(f"*{ref['text']}*")