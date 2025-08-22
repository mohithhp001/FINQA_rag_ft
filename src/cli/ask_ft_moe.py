import argparse, json, time, pathlib, pickle, re, subprocess
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from ..ft_moe.gate import pick_expert
from ..rag.retrieve import hybrid_search, gather_context
from ..config import Config
import re, os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # silence fork/parallelism warning

def _has_header_unit(text):
    if re.search(r"(₹|rs\.?|inr)\s*(in)?\s*(crore|cr)\b", text, flags=re.I): return "crore"
    if re.search(r"(₹|rs\.?|inr)\s*(in)?\s*(million|mn)\b", text, flags=re.I): return "million"
    if re.search(r"(₹|rs\.?|inr)\s*(in)?\s*(billion|bn)\b", text, flags=re.I): return "billion"
    return None

def _neighbor_unit_hint(seg, store, back=3):
    try:
        fid, idx = seg["id"].split(":"); idx = int(idx)
    except Exception:
        return None
    ids = {s.get("id"): s for s in store["segs"]}
    for step in range(1, back+1):
        prev = ids.get(f"{fid}:{idx-step}")
        if prev:
            u = _has_header_unit(prev.get("text",""))
            if u: return u
    return None


# -------------------- helpers --------------------
def _load_store(index_dir):
    p = pathlib.Path(index_dir or Config.index_dir) / "store.pkl"
    with open(p, "rb") as f:
        return pickle.load(f)

def _fewshot_block():
    return (
        "Follow the rules strictly:\n"
        "• Use the context to answer.\n"
        "• Output ONLY the final value and unit (e.g., 3,910,199 crore or 38.2 %). No extra words.\n"
        "• If unknown from context, output exactly: Not found in context.\n\n"
        "Examples:\n"
        "Q: What were total deposits as at March 31, 2024?\n"
        "A: 2,000,000 crore\n"
        "Q: What was the CASA ratio in FY2024-25?\n"
        "A: 38.2 %\n\n"
    )

def _build_prompt(q, ctxs):
    prompt = _fewshot_block()
    for i,(seg,score) in enumerate(ctxs[:8], 1):
        prompt += f"[Context {i}] {seg['text'][:1000]}\n"
    prompt += f"\nQuestion: {q}\nFinal answer:"
    return prompt

def _postprocess_numeric(ans: str):
    # Accept "number + optional unit"; reject "per share" etc.
    if not ans:
        return None
    if "per share" in ans.lower():
        return None
    m = re.search(r"([\d,]+(?:\.\d+)?)(?:\s*(lakh\s+crore|crore|cr|billion|million|lakh|bn|mn|%))?$",
                  ans.strip(), flags=re.I)
    if not m:
        return None
    val = m.group(1)
    unit = (m.group(2) or "").strip().lower()
    unit = {"cr":"crore","bn":"billion","mn":"million"}.get(unit, unit)
    try:
        vv = float(val.replace(",",""))
    except Exception:
        return None
    if "." in val:
        val_fmt = f"{vv:,.2f}"
    else:
        val_fmt = f"{int(vv):,}"
    return (val_fmt + (" " + unit if unit and unit != "%" else unit)).strip()

def _rag_fallback(question, index_dir, k, rerank):
    cmd = ["python","-m","src.cli.ask","--q",question,"--k",str(k),"--index",index_dir,"--extractive_only"]
    if rerank:
        cmd.append("--rerank")
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    return json.loads(out)

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--q', required=True)
    ap.add_argument('--num', required=True, help='Path to numeric expert model')
    ap.add_argument('--nar', required=True, help='Path to narrative expert model')
    ap.add_argument('--index', default='indexes_400', help='Index dir to pull context from')
    ap.add_argument('--k', type=int, default=12)
    ap.add_argument('--rerank', action='store_true')
    args = ap.parse_args()

    expert = pick_expert(args.q)
    model_path = args.num if expert == "num" else args.nar

    # Retrieval (shared for FT and fallback)
    store = _load_store(args.index)
    ranked = hybrid_search(args.q, store, top_k_dense=args.k, top_k_sparse=args.k)
    ctxs = gather_context(ranked, store, max_chars=5000)

    # Optional re-rank
    if args.rerank:
        try:
            from ..rag.rerank import Reranker
            rr = Reranker()
            scores = rr.rerank(args.q, [s["text"] for s,_ in ctxs])
            order = sorted(range(len(ctxs)), key=lambda i: -scores[i])
            ctxs = [ctxs[i] for i in order]
        except Exception:
            pass

    # FT with context
    tok = AutoTokenizer.from_pretrained(model_path)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    prompt = _build_prompt(args.q, ctxs)

    t0 = time.time()
    inp = tok(prompt, return_tensors="pt", truncation=True)
    out = mdl.generate(**inp, max_new_tokens=48)
    ft_raw = tok.decode(out[0], skip_special_tokens=True).strip()
    elapsed = round(time.time()-t0, 3)

    # Validate FT output
    normalized = _postprocess_numeric(ft_raw)

    # Fallback to RAG extractive if FT looks wrong
    fallback_used = False
    evidence = []
    if not normalized:
        fallback_used = True
        rag = _rag_fallback(args.q, args.index, args.k, args.rerank)
        normalized = rag.get("answer", "Not found in context.")
        evidence = rag.get("evidence", [])
    # If fallback answer is just a number, try to append unit from the evidence segment or its neighbor
    if fallback_used and normalized and re.fullmatch(r"[\d,]+(?:\.\d+)?", normalized):
        unit_hint = None
        try:
            if evidence:
                evid_id = evidence[0].get("id")
                seg = next((s for s in store["segs"] if s.get("id") == evid_id), None)
                if seg:
                    unit_hint = _has_header_unit(seg.get("text", "")) or _neighbor_unit_hint(seg, store)
        except Exception:
            unit_hint = None
        if unit_hint:
            normalized = f"{normalized} {unit_hint}"
    print(json.dumps({
        "question": args.q,
        "answer": normalized or "Not found in context.",
        "ft_raw": ft_raw,
        "expert": expert,
        "model": model_path,
        "fallback_used": fallback_used,
        "evidence": evidence[:3],
        "time_s": elapsed
    }, indent=2))

if __name__ == "__main__":
    main()
