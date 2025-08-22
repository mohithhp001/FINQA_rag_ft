import argparse, pathlib, json, sys, warnings, re, time, pickle
warnings.filterwarnings('ignore')

# Config fallback
try:
    from ..config import Config
except Exception:
    class Config:
        index_dir = "indexes"
        gen_model = "google/flan-t5-small"

# use your repo's retrieval helpers
from ..rag.retrieve import hybrid_search, gather_context
import re

def _has_header_unit(text):
    if re.search(r"(₹|rs\.?|inr)\s*(in)?\s*(crore|cr)\b", text, flags=re.I): return "crore"
    if re.search(r"(₹|rs\.?|inr)\s*(in)?\s*(million|mn)\b", text, flags=re.I): return "million"
    if re.search(r"(₹|rs\.?|inr)\s*(in)?\s*(billion|bn)\b", text, flags=re.I): return "billion"
    return None

def _neighbor_unit_hint(seg, store):
    # seg["id"] looks like "HDFC_IAR_FY25.jsonl:802"
    try:
        fid, idx = seg["id"].split(":")
        idx = int(idx)
    except Exception:
        return None
    prev_id = f"{fid}:{idx-1}"
    for s in store["segs"]:
        if s.get("id") == prev_id:
            return _has_header_unit(s.get("text",""))
    return None


def reorder_candidates_by_title(cands, metric):
    if metric == "DEPOSITS":
        def rank(seg):
            title = (seg.get("title") or "").upper()
            # BALANCE SHEET first, then MANAGEMENT DISCUSSION, etc.
            if "BALANCE SHEET" in title:
                return (0, title)
            return (1, title)
        return sorted(cands, key=rank)
    return cands

# ------------------------- guardrail -------------------------
FINANCE_HINTS = {"deposit","advance","pat","profit","income","expense","eps","casa","gnpa","nnpa","assets","liabilities","ratio","margin","capital","car","crar","roa","roe","dividend","branch","employee","cash flow","balance sheet"}
def input_guardrail(q: str) -> bool:
    ql = q.lower()
    return any(w in ql for w in FINANCE_HINTS)

# ------------------------- store loader -------------------------
def load_store(index_dir=None):
    p = pathlib.Path(index_dir or Config.index_dir) / "store.pkl"
    if not p.exists():
        print(f"Index not found at {p}. Build with: python -m src.rag.build_index --segments <dir> --out {index_dir or Config.index_dir}")
        sys.exit(1)
    with open(p, "rb") as f:
        return pickle.load(f)

# ------------------------- metric helpers -------------------------
METRIC_SYNONYMS = {
    "PAT": [r"profit\s+after\s+tax", r"net\s+profit", r"\bPAT\b"],
    "DEPOSITS": [r"\btotal\s+deposits\b", r"\bdeposits\b"],
    "ADVANCES": [r"\badvances\b", r"\bloans\b(?! and advances)"],
    "CASA": [r"\bCASA\s*ratio\b"],
    "GNPA": [r"\bgross\s+npas?\b", r"\bGNPA\b"],
    "NNPA": [r"\bnet\s+npas?\b", r"\bNNPA\b"],
    "CRAR": [r"\bCRAR\b", r"\bCAR\b", r"capital\s+adequacy"],
    "EPS": [r"\bEPS\b", r"earnings\s+per\s+share"],
    "DEPOSITS": [r"\btotal\s+deposits\b", r"\bcustomer\s+deposits\b", r"\bdeposits\b"],
    "ADVANCES": [r"\badvances\b", r"\bloans\b(?! and advances)", r"\bcustomer\s+advances\b"],
    "CASA": [r"\bCASA\s*ratio\b", r"\bcasa\b"],
    
}
def detect_metric(q: str):
    ql = q.lower()
    if "pat" in ql or "profit after tax" in ql or "net profit" in ql: return "PAT"
    if "deposit" in ql: return "DEPOSITS"
    if "advance" in ql or "loan" in ql: return "ADVANCES"
    if "casa" in ql: return "CASA"
    if "gnpa" in ql: return "GNPA"
    if "nnpa" in ql: return "NNPA"
    if "crar" in ql or "car" in ql or "capital adequacy" in ql: return "CRAR"
    if "eps" in ql: return "EPS"
    return None

def year_tags(q: str):
    ql = q.lower()
    tags = set()
    def add_year(y):
        y = int(y)
        tags.update({
            f"fy{y-1}-{str(y)[-2:]}", f"fy {y-1}-{str(y)[-2:]}", f"fy{str(y)[-2:]}", f"fy {str(y)[-2:]}",
            f"{y-1}-{y}", f"{y-1}–{y}",
            f"march 31, {y}", f"31 march {y}",
            f"for the year ended march 31, {y}",
            f"as at march 31, {y}",
            f"year ended 31 march {y}",
        })
    if any(t in ql for t in ["2024-25","2024–25","fy2024-25","fy24-25","fy25","2025"]): add_year(2025)
    if any(t in ql for t in ["2023-24","2023–24","fy2023-24","fy23-24","fy24","2024"]): add_year(2024)
    return list(tags)

def normalize_unit(u: str):
    if not u: return ""
    u = u.lower().strip()
    return {"cr":"crore","mn":"million","bn":"billion"}.get(u, u)

def pretty_num(val: str, unit: str):
    unit = normalize_unit(unit or "")
    try:
        if "." in val: v = f"{float(val):,.2f}"
        else: v = f"{int(float(val)):,.0f}"
        return (v + (" " + unit if unit and unit != "%" else unit)).strip()
    except Exception:
        return (val + (" " + unit if unit else "")).strip()

def build_patterns(metric: str):
    syns = METRIC_SYNONYMS.get(metric, [])
    amount_with_unit = r"(?:₹|rs\.?|inr)\s*([\d,]+(?:\.\d+)?)\s*(lakh\s+crore|crore|cr|billion|million|lakh|bn|mn)"
    percent_only   = r"(\d+(?:\.\d+)?)\s*%"
    if metric in ["PAT","DEPOSITS","ADVANCES"]:
        return [
            rf"(?is)(?:{'|'.join(syns)})[\s\S]{{0,240}}{amount_with_unit}",
            rf"(?is)(?:{'|'.join(syns)})[\s\S]{{0,240}}([\d,]+(?:\.\d+)?)(?!\s*%)",
            rf"(?is)\b([\d,]+(?:\.\d+)?)\b(?!\s*%)\s*[\s\S]{{0,120}}(?:{'|'.join(syns)})"
        ]
    if metric == "EPS":
        return [rf"(?is)(?:{'|'.join(syns)})[\s\S]{{0,200}}(?:₹|rs\.?|inr)?\s*([\d,]+(?:\.\d+)?)"]
    if metric == "CASA":
        return [rf"(?is)\bCASA\s*ratio[\s\S]{{0,100}}{percent_only}"]
    if metric in ["GNPA","NNPA","CRAR"]:
        return [rf"(?is)(?:{'|'.join(syns)})[\s\S]{{0,140}}{percent_only}"]
    return [rf"(?is)(?:{'|'.join(syns)})[\s\S]{{0,140}}{percent_only}"]

def has_header_unit(text):
    if re.search(r"(₹|rs\.?|inr)\s*(in)?\s*(crore|cr)\b", text, flags=re.I): return "crore"
    if re.search(r"(₹|rs\.?|inr)\s*(in)?\s*(million|mn)\b", text, flags=re.I): return "million"
    if re.search(r"(₹|rs\.?|inr)\s*(in)?\s*(billion|bn)\b", text, flags=re.I): return "billion"
    return None

def search_text_for_metric(text: str, metric: str, neighbor_unit=None):
    # Skip matches near "Schedule X"
    def schedule_nearby(m):
        start = m.start()
        window = text[max(0, start-40): start+40]
        return re.search(r"\bSchedule\s*\d+\b", window, flags=re.I) is not None

    header_unit = _has_header_unit(text)
    for pat in build_patterns(metric):
        m = re.search(pat, text, flags=re.I)
        if not m:
            continue
        if schedule_nearby(m):
            continue

        groups = [g for g in m.groups() if g]
        val = next((g for g in groups if re.match(r"^[\d,]+(\.\d+)?$", g)), None)
        unit = ""
        for g in groups:
            if isinstance(g, str) and g.lower() in ["lakh crore","crore","cr","billion","million","lakh","bn","mn","%"]:
                unit = g; break

        # For money metrics, reject tiny integers like "8" unless a unit exists
        if metric in ["PAT","DEPOSITS","ADVANCES","EPS"]:
            if not unit and not header_unit and not neighbor_unit:
                if val and len(val.replace(",","").split(".")[0]) < 3:
                    continue

        if (not unit):
            unit = header_unit or neighbor_unit

        if val:
            return pretty_num(val.replace(",",""), unit or "")
    return None
# ------------------------- extraction -------------------------

def extract_from_store(question: str, store):
    metric = detect_metric(question)
    if not metric: return None, None
    tags = [t.lower() for t in year_tags(question)]
    candidates = []
    for seg in store["segs"]:
        t = seg["text"]; tl = t.lower()
        if tags and not any(tag in tl for tag in tags): continue
        if any(re.search(s, t, flags=re.I) for s in METRIC_SYNONYMS[metric]):
            candidates.append(seg)
    candidates = reorder_candidates_by_title(candidates, metric)  # keep if you already added this
    for seg in candidates:
        ans = search_text_for_metric(seg["text"], metric, neighbor_unit=_neighbor_unit_hint(seg, store))
        if ans: return ans, seg
    return None, candidates[0] if candidates else (None, None)

def extract_from_contexts(question: str, contexts, store=None):
    metric = detect_metric(question)
    if not metric: return None, None
    tags = [t.lower() for t in year_tags(question)]
    candidates = []
    if tags:
        for seg,score in contexts:
            if any(t in seg["text"].lower() for t in tags):
                candidates.append((seg,score))
    if not candidates: candidates = contexts
    for seg,score in candidates:
        ans = search_text_for_metric(seg["text"], metric, neighbor_unit=_neighbor_unit_hint(seg, store) if store else None)
        if ans: return ans, seg
    return None, None

# ------------------------- (optional) reranker -------------------------
def maybe_rerank(query, segs, use_rerank=False):
    if not use_rerank:
        return segs
    try:
        from ..rag.rerank import Reranker
        rr = Reranker()
        texts = [s["text"] for s,_ in segs]
        scores = rr.rerank(query, texts)
        order = sorted(range(len(texts)), key=lambda i: -scores[i])
        return [segs[i] for i in order]
    except Exception:
        return segs

# ------------------------- generator fallback -------------------------
def generate_with_model(question, contexts, model_name):
    prompt = "Answer ONLY with the final value and unit (no explanations). If unknown, reply: Not found in context.\n"
    for i,(seg,score) in enumerate(contexts, 1):
        prompt += f"\n[Context {i}] (score={score:.2f})\n{seg['text'][:1200]}\n"
    prompt += f"\nQuestion: {question}\nFinal answer:"
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        inp = tok(prompt, return_tensors="pt", truncation=True)
        out = model.generate(**inp, max_new_tokens=64)
        return tok.decode(out[0], skip_special_tokens=True).strip()
    except Exception:
        return "Not found in context."

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--q', required=True)
    ap.add_argument('--k', type=int, default=12)
    ap.add_argument('--index', default=None, help="Override index dir (e.g., indexes_100 or indexes_400)")
    ap.add_argument('--rerank', action='store_true', help='Re-rank with cross-encoder (advanced RAG)')
    ap.add_argument('--extractive_only', action='store_true', help='Force regex extraction without generator')
    args = ap.parse_args()

    if not input_guardrail(args.q):
        print(json.dumps({"question": args.q, "answer": "Query out of scope for this app.", "evidence": []}, indent=2))
        return

    store = load_store(args.index)
    t0 = time.time()
    ranked = hybrid_search(args.q, store, top_k_dense=args.k, top_k_sparse=args.k)
    ctxs = gather_context(ranked, store, max_chars=5000)
    ctxs = maybe_rerank(args.q, ctxs, use_rerank=args.rerank)

    ans, seg_used = extract_from_store(args.q, store)
    used_global = ans is not None
    if not ans:
        ans, seg_used = extract_from_contexts(args.q, ctxs)

    if not ans:
        final = "Not found in context." if args.extractive_only else generate_with_model(args.q, ctxs, Config.gen_model)
    else:
        final = ans

    evidence = []
    if used_global and seg_used:
        evidence = [{"id": seg_used.get("id",""), "title": seg_used.get("title",""), "source": seg_used.get("source","")}]
    else:
        evidence = [{"id": seg["id"], "title": seg.get("title",""), "source": seg["source"]} for seg,_ in ctxs[:3]]

    elapsed = round(time.time()-t0, 3)
    print(json.dumps({"question": args.q, "answer": final, "evidence": evidence, "reranked": args.rerank, "time_s": elapsed}, indent=2))

if __name__ == "__main__":
    main()
