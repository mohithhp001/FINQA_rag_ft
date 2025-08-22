import os, sys, json, subprocess, time
from pathlib import Path
import streamlit as st

# Silence tokenizers warning on macOS forks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------- paths ----------
APP_FILE = Path(__file__).resolve()
REPO_ROOT = APP_FILE.parents[1]          # repo root (…/FINQA_rag_ft)
PYTHON = sys.executable                   # use the current venv interpreter

# ---------- helpers ----------
def list_index_dirs():
    dirs = []
    for p in REPO_ROOT.iterdir():
        if p.is_dir() and p.name.startswith("indexes") and (p / "store.pkl").exists():
            dirs.append(p.name)
    return sorted(dirs) or ["indexes"]

def parse_last_json_block(text: str):
    # Find last {...} block in stdout and parse; raise if none found
    s = text.rfind("{")
    e = text.rfind("}")
    if s >= 0 and e >= s:
        return json.loads(text[s:e+1])
    raise json.JSONDecodeError("No JSON object found", text, 0)

def run_json(cmd):
    """Run a CLI that prints JSON. Returns (obj, raw, elapsed)."""
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(REPO_ROOT),      # <-- run from repo root, fixes imports & paths
        env=os.environ.copy(),
    )
    dt = time.time() - t0
    raw = proc.stdout or ""
    if proc.returncode != 0:
        # Try to salvage JSON even on non-zero; else raise
        try:
            obj = parse_last_json_block(raw)
            return obj, raw, dt
        except Exception:
            raise RuntimeError(raw.strip())
    # parse JSON (direct or salvage last block)
    try:
        obj = json.loads(raw)
    except Exception:
        obj = parse_last_json_block(raw)
    return obj, raw, dt

def pretty_evidence(ev_list):
    if not ev_list:
        st.write("*(no evidence)*")
        return
    for e in ev_list[:5]:
        tid = e.get("id","")
        title = e.get("title","")
        src = e.get("source","")
        st.markdown(f"- **{title}** — `{tid}`  \n  _{src}_")

def code_block(cmd_list):
    # Quote items with spaces for display only
    shown = " ".join(c if " " not in c else f"\"{c}\"" for c in cmd_list)
    st.code(shown, language="bash")

# ---------- UI ----------
st.set_page_config(page_title="FINQA: RAG vs FT", layout="centered")
st.title("FINQA — RAG vs Fine-Tuning (HDFC Bank)")
st.caption("Compare RAG (with/without re-rank) vs FT-MoE. Evidence is shown for RAG.")

with st.sidebar:
    st.subheader("Settings")
    mode = st.selectbox("Mode", ["RAG", "RAG + Re-rank", "FT-MoE"])
    index_dirs = list_index_dirs()
    default_idx = index_dirs.index("indexes_400") if "indexes_400" in index_dirs else 0
    index = st.selectbox("Index directory", index_dirs, default_idx)
    k = st.slider("Top-K (dense + sparse)", 4, 24, 12, 2)
    use_rerank = st.checkbox("Use cross-encoder re-ranker", value=(mode == "RAG + Re-rank"))
    extractive_only = st.checkbox("Extractive-only (RAG)", value=True,
                                  help="For numeric facts, extractive RAG is most reliable.")

    st.divider()
    st.markdown("**FT-MoE model paths**")
    num_model = st.text_input("Numeric expert", "models/ft-flan-t5-small-exp-num")
    nar_model = st.text_input("Narrative expert", "models/ft-flan-t5-small-exp-nar")

examples = [
    "What were total deposits as at March 31, 2025 (standalone)?",
    "What was HDFC Bank's consolidated PAT in FY2024-25?",
    "Report CASA ratio trend FY2023-24 to FY2024-25.",
]
q = st.text_input("Ask a question", examples[0])

with st.expander("Examples"):
    cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        if cols[i].button(ex, key=f"ex{i}"):
            st.session_state["q"] = ex
            st.rerun()

if st.button("Ask"):
    if not q.strip():
        st.warning("Please enter a question.")
        st.stop()

    # Build the command
    if mode.startswith("RAG"):
        cmd = [PYTHON, "-m", "src.cli.ask", "--q", q, "--k", str(k), "--index", index]
        if extractive_only:
            cmd.append("--extractive_only")
        if use_rerank:
            cmd.append("--rerank")
    else:  # FT-MoE
        cmd = [
            PYTHON, "-m", "src.cli.ask_ft_moe",
            "--q", q, "--num", num_model, "--nar", nar_model,
            "--index", index, "--k", str(k)
        ]
        if use_rerank:
            cmd.append("--rerank")

    st.write("**Command:**")
    code_block(cmd)

    try:
        with st.spinner("Running…"):
            resp, raw, elapsed = run_json(cmd)

        st.subheader("Answer")
        st.markdown(f"### {resp.get('answer','(no answer)')}")

        m1, m2, m3 = st.columns(3)
        m1.metric("Mode", mode)
        m2.metric("Latency (s)", f"{elapsed:.2f}")
        m3.metric("Top-K", k)

        if mode == "FT-MoE":
            st.write("**Expert:**", resp.get("expert",""))
            if resp.get("fallback_used", False):
                st.info("Fallback used → FT output invalid, returned RAG-extractive.")
            raw_ft = resp.get("ft_raw","")
            if raw_ft:
                st.caption(f"FT raw: `{raw_ft}`")

        if "evidence" in resp:
            st.subheader("Evidence (RAG)")
            pretty_evidence(resp.get("evidence", []))

        with st.expander("Raw JSON"):
            st.code(json.dumps(resp, indent=2), language="json")
        with st.expander("Raw CLI output"):
            st.code(raw, language="bash")

    except Exception as e:
        st.error("Command failed.")
        with st.expander("Error details"):
            st.code(str(e), language="bash")

st.divider()
st.caption("Tip: Use **indexes_400** + **re-rank** + **extractive-only** for tabular (Balance Sheet) facts.")
