import streamlit as st
import subprocess, json, time

st.set_page_config(page_title="FINQA RAG vs FT", layout="centered")
st.title("FINQA: RAG vs Fine-tuning")
st.caption("HDFC Bank FY2023–24 & FY2024–25 | Demo UI")

mode = st.selectbox("Mode", ["RAG", "RAG + Re-rank", "FT", "FT-MoE"])
index_dir = st.selectbox("Index", ["indexes", "indexes_100", "indexes_400"])
q = st.text_input("Ask a question", "What were HDFC Bank’s total deposits as on March 31, 2025?")
ft_path = st.text_input("FT model path (for FT mode)", "models/ft-flan-t5-small")
ft_num = st.text_input("FT-MoE numeric expert", "models/ft-flan-t5-small-exp-num")
ft_nar = st.text_input("FT-MoE narrative expert", "models/ft-flan-t5-small-exp-nar")

if st.button("Ask"):
    t0 = time.time()
    try:
        if mode in ["RAG","RAG + Re-rank"]:
            cmd = ["python","-m","src.cli.ask","--q",q,"--k","12","--index",index_dir]
            if mode.endswith("Re-rank"):
                cmd.append("--rerank")
        elif mode == "FT":
            # You can adapt this to call an FT-only inference script if you prefer.
            cmd = ["python","-m","src.cli.ask","--q",q,"--k","12","--index",index_dir]
        else:  # FT-MoE
            cmd = ["python","-m","src.cli.ask_ft_moe","--q",q,"--num",ft_num,"--nar",ft_nar]

        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        resp = json.loads(out)
        elapsed = time.time() - t0

        st.json(resp)
        st.write(f"Elapsed: {elapsed:.2f}s")
        if "evidence" in resp and resp["evidence"]:
            st.subheader("Evidence")
            for e in resp["evidence"]:
                st.write(f"- **{e.get('title','')}** — {e.get('id','')}")
    except subprocess.CalledProcessError as e:
        st.error(f"Error: {e.output}")
