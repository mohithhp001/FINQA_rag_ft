import argparse, pathlib, json, sys, warnings, re
warnings.filterwarnings('ignore')
from ..config import Config
from ..rag.utils import load_pickle, normalize_text
from ..rag.retrieve import hybrid_search, gather_context

def load_store():
    p = pathlib.Path(Config.index_dir) / "store.pkl"
    if not p.exists():
        print("Index not found. Build with: python -m src.rag.build_index"); sys.exit(1)
    return load_pickle(p)

# ------------------------- heuristics -------------------------

METRIC_SYNONYMS = {
    "PAT": [r"profit\s+after\s+tax", r"net\s+profit", r"\bPAT\b"],
    "DEPOSITS": [r"\btotal\s+deposits\b", r"\bdeposits\b"],
    "ADVANCES": [r"\badvances\b", r"\bloans\b(?! and advances)"],
    "CASA": [r"\bCASA\b", r"current\s+account.*savings\s+account"],
    "GNPA": [r"\bgross\s+npas?\b", r"\bGNPA\b"],
    "NNPA": [r"\bnet\s+npas?\b", r"\bNNPA\b"],
    "CRAR": [r"\bCRAR\b", r"\bCAR\b", r"capital\s+adequacy"],
    "EPS": [r"\bEPS\b", r"earnings\s+per\s+share"],
}

AMOUNT = r"(?:₹|rs\.?|inr)?\s*([\d,]+(?:\.\d+)?)\s*(lakh\s+crore|crore|cr|billion|million|lakh|mn|bn|%)?"
AMOUNT_OR_PERCENT = AMOUNT

def detect_metric(q: str):
    ql = q.lower()
    if any(k in ql for k in ["pat","profit after tax","net profit"]): return "PAT"
    if "deposit" in ql: return "DEPOSITS"
    if "advance" in ql or "loan" in ql: return "ADVANCES"
    if "casa" in ql: return "CASA"
    if "gnpa" in ql: return "GNPA"
    if "nnpa" in ql: return "NNPA"
    if any(k in ql for k in ["crar","car","capital adequacy"]): return "CRAR"
    if "eps" in ql: return "EPS"
    return None

def year_tags(q: str):
    ql = q.lower()
    tags = []
    def add_year(y):
        y = str(y)
        tags.extend([
            f"fy{int(y)-1}-{y[-2:]}", f"fy {int(y)-1}-{y[-2:]}", f"fy{y[-2:]}", f"fy {y[-2:]}",
            f"2024-25" if y=="2025" else f"{int(y)-1}-{y}",
            f"march 31, {y}", f"31 march {y}", f"for the year ended march 31, {y}",
            f"as at march 31, {y}", f"year ended 31 march {y}"
        ])

    if any(t in ql for t in ["2024-25","fy2024-25","fy24-25","fy25","2025"]):
        add_year(2025)
    if any(t in ql for t in ["2023-24","fy2023-24","fy23-24","fy24","2024"]):
        add_year(2024)

    # lower-case & unique
    return list({t.lower() for t in tags})


def normalize_unit(u: str):
    if not u: return ""
    u = u.lower().strip()
    return {"cr":"crore","mn":"million","bn":"billion"}.get(u, u)

def pretty_num(val: str, unit: str):
    unit = normalize_unit(unit or "")
    try:
        # keep decimals if present; otherwise int with commas
        if "." in val:
            v = f"{float(val):,.2f}"
        else:
            v = f"{int(float(val)):,.0f}"
        return (v + (" " + unit if unit and unit != "%" else unit)).strip()
    except Exception:
        return (val + (" " + unit if unit else "")).strip()

def build_patterns(metric: str):
    syns = METRIC_SYNONYMS.get(metric, [])
    # amount with explicit units (avoid random %s)
    amount_pat = r"(?:₹|rs\.?|inr)\s*([\d,]+(?:\.\d+)?)\s*(lakh\s+crore|crore|cr|billion|million|lakh|bn|mn)"
    percent_pat = r"(\d+(?:\.\d+)?)\s*%"

    # NB: (?s) = DOTALL; allows matches across newlines
    if metric in ["PAT", "DEPOSITS", "ADVANCES"]:
        # allow up to ~240 chars between label and value, across newlines
        return [rf"(?is)(?:{'|'.join(syns)})[\s\S]{{0,240}}{amount_pat}"]

    if metric == "EPS":
        # EPS can be plain number or rupee-prefixed
        return [rf"(?is)(?:{'|'.join(syns)})[\s\S]{{0,200}}(?:₹|rs\.?|inr)?\s*([\d,]+(?:\.\d+)?)"]

    if metric == "CASA":
        # insist on the exact phrase 'CASA ratio' near a %
        return [
            rf"(?is)\bCASA\s*ratio[\s\S]{{0,80}}{percent_pat}",
            rf"(?is)(?:{'|'.join(syns)})[\s\S]{{0,120}}{percent_pat}",
        ]

    if metric in ["GNPA","NNPA","CRAR"]:
        return [rf"(?is)(?:{'|'.join(syns)})[\s\S]{{0,140}}{percent_pat}"]

    return [rf"(?is)(?:{'|'.join(syns)})[\s\S]{{0,140}}{percent_pat}"]



def search_text_for_metric(text: str, metric: str):
    for pat in build_patterns(metric):
        m = re.search(pat, text, flags=re.I)
        if m:
            groups = [g for g in m.groups() if g]
            # first numeric-looking group
            val = next((g for g in groups if re.match(r"^[\d,]+(\.\d+)?$", g)), None)
            unit = ""
            for g in groups:
                if isinstance(g, str) and g.lower() in ["lakh crore","crore","cr","billion","million","lakh","mn","bn","%"]:
                    unit = g; break
            if val:
                return pretty_num(val.replace(",",""), unit)
    return None

# ------------------------- extractors -------------------------

def extract_from_contexts(question: str, contexts):
    metric = detect_metric(question)
    if not metric: return None, None
    tags = year_tags(question)
    # filter contexts by year hint if present
    candidates = []
    if tags:
        for seg,score in contexts:
            if any(t in seg["text"].lower() for t in tags):
                candidates.append((seg,score))
    if not candidates:
        candidates = contexts
    for seg,score in candidates:
        ans = search_text_for_metric(seg["text"], metric)
        if ans:
            return ans, seg
    return None, None

def extract_from_store(question: str, store):
    metric = detect_metric(question)
    if not metric: return None, None
    tags = year_tags(question)

    # strict pass: only segments that contain the target year's tokens
    strict_hit = None
    if tags:
        for seg in store["segs"]:
            t = seg["text"]; tl = t.lower()
            if any(tag in tl for tag in tags) and any(re.search(s, t, flags=re.I) for s in METRIC_SYNONYMS[metric]):
                ans = search_text_for_metric(t, metric)
                if ans: return ans, seg
        # If the question names a year and strict search failed, DO NOT fall back to other years:
        return None, None

    # no year specified → relaxed pass ok
    for seg in store["segs"]:
        t = seg["text"]
        if any(re.search(s, t, flags=re.I) for s in METRIC_SYNONYMS[metric]):
            ans = search_text_for_metric(t, metric)
            if ans: return ans, seg
    return None, None


# ------------------------- generator fallback -------------------------

def generate_with_model(question, contexts, mode="rag", ft_path=None):
    prompt = "Answer ONLY with the final value and unit (no explanations). If unknown, reply: Not found in context.\n"
    for i,(seg,score) in enumerate(contexts, 1):
        prompt += f"\n[Context {i}] (score={score:.2f})\n{seg['text'][:1200]}\n"
    prompt += f"\nQuestion: {question}\nFinal answer:"
    if mode == "ft":
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            tok = AutoTokenizer.from_pretrained(ft_path or "google/flan-t5-small")
            model = AutoModelForSeq2SeqLM.from_pretrained(ft_path or "google/flan-t5-small")
            inp = tok("question: " + question, return_tensors="pt", truncation=True)
            out = model.generate(**inp, max_new_tokens=64)
            return tok.decode(out[0], skip_special_tokens=True).strip()
        except Exception:
            pass
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tok = AutoTokenizer.from_pretrained(Config.gen_model)
        model = AutoModelForSeq2SeqLM.from_pretrained(Config.gen_model)
        inp = tok(prompt, return_tensors="pt", truncation=True)
        out = model.generate(**inp, max_new_tokens=64)
        return tok.decode(out[0], skip_special_tokens=True).strip()
    except Exception:
        # last resort: show first sentence mentioning the metric
        blob = " ".join(seg['text'] for seg,_ in contexts)
        m = re.search(r"([A-Z].{0,120}?(?:profit|deposit|advance|casa|gnpa|nnpa|crar|eps)[^.]*\.)", blob, flags=re.I)
        return m.group(1).strip() if m else "Not found in context."

# ------------------------- cli -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--q', required=True)
    ap.add_argument('--k', type=int, default=12)
    ap.add_argument('--mode', choices=['rag','ft'], default='rag')
    ap.add_argument('--ft_path', default=None)
    ap.add_argument('--extractive_only', action='store_true', help='Force regex extraction without generator')
    args = ap.parse_args()

    store = load_store()
    ranked = hybrid_search(args.q, store, top_k_dense=args.k, top_k_sparse=args.k)
    # gather more context chars to catch split tables
    ctxs = gather_context(ranked, store, max_chars=5000)

    # 1) global (whole corpus) deterministic extraction
    ans, seg_used = extract_from_store(args.q, store)
    used_global = False
    if ans:
        used_global = True
    else:
        # 2) context-only deterministic extraction
        ans, seg_used = extract_from_contexts(args.q, ctxs)

    if not ans:
        if args.extractive_only:
            final = "Not found in context."
        else:
            final = generate_with_model(args.q, ctxs, mode=args.mode, ft_path=args.ft_path)
    else:
        final = ans

    # evidence
    evidence = []
    if used_global and seg_used:
        evidence = [{"id": seg_used.get("id",""), "title": seg_used.get("title",""), "source": seg_used.get("source","")}]
    else:
        evidence = [{"id": seg["id"], "title": seg.get("title",""), "source": seg["source"]} for seg,_ in ctxs[:3]]

    print(json.dumps({"question": args.q, "answer": final, "evidence": evidence}, indent=2))

if __name__ == "__main__":
    main()
