import streamlit as st, json, pathlib
from src.config import Config
from src.rag.utils import load_pickle
from src.rag.retrieve import hybrid_search, gather_context
from src.cli.ask import generate_answer

st.set_page_config(page_title='FINQA RAG vs FT', layout='wide')
st.title('FINQA: RAG vs Fine-Tuning')

q = st.text_input('Ask a financial question:', 'What was consolidated PAT in FY2024-25?')
mode = st.radio('Mode', ['RAG','FT'], horizontal=True)
ft_path = st.text_input('FT model path (for FT mode)', 'models/ft-flan-t5-small')

if st.button('Ask'):
    store_path = pathlib.Path(Config.index_dir) / 'store.pkl'
    if not store_path.exists():
        st.error('Index not found. Build with: python -m src.rag.build_index'); st.stop()
    store = load_pickle(store_path)
    ranked = hybrid_search(q, store)
    ctxs = gather_context(ranked, store)
    ans = generate_answer(q, ctxs, mode=mode.lower(), ft_path=ft_path if mode=='FT' else None)
    st.subheader('Answer')
    st.write(ans)
    st.subheader('Evidence')
    st.json([{"id": seg["id"], "title": seg.get("title",""), "source": seg["source"]} for seg,_ in ctxs])
