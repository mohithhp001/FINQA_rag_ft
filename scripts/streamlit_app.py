import streamlit as st
from financial_rag import FinancialRAG

st.title("Financial RAG Demo")

if "rag" not in st.session_state:
    st.session_state.rag = FinancialRAG(
        [
            "Apple reported record revenue of $90B in Q1 2024",
            "Tesla's earnings grew 20% year over year",
        ]
    )

# Document input area
current_docs = "\n".join(st.session_state.rag.documents)
docs_text = st.text_area("Documents (one per line)", value=current_docs, height=150)
if st.button("Update documents"):
    docs = [d.strip() for d in docs_text.splitlines() if d.strip()]
    st.session_state.rag = FinancialRAG(docs, cross_encoder=st.session_state.rag.cross_encoder)

query = st.text_input("Ask a question")
if st.button("Get answer") and query:
    answer = st.session_state.rag.answer_hierarchical(query)
    st.write(answer)
