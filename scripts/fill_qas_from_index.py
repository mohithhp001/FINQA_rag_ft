
"""
fill_qas_from_index.py
Reads a gold QA file and fills answers deterministically using your built index.
Usage:
  python scripts/fill_qas_from_index.py --index indexes/store.pkl --yaml data/qas/gold.yaml --out data/qa/qa_pairs_filled.csv
  python scripts/fill_qas_from_index.py --index indexes/store.pkl --csv data/qa/qa_pairs.csv --out data/qa/qa_pairs_filled.csv
"""
import argparse, json, re, pathlib, csv, sys, pickle

def load_store(p):
    with open(p, "rb") as f: return pickle.load(f)

METRIC_SYNONYMS = {
    "pat": [r"profit\s+after\s+tax", r"net\s+profit", r"\bPAT\b"],
    "total deposits": [r"\btotal\s+deposits\b", r"\bdeposits\b"],
    "total advances": [r"\badvances\b", r"\bloans\b(?! and advances)"],
    "casa ratio": [r"\bCASA\s*ratio\b"],
    "gross npa ratio": [r"\bgross\s+npas?\b", r"\bGNPA\b"],
    "net npa ratio": [r"\bnet\s+npas?\b", r"\bNNPA\b"],
    "provision coverage ratio": [r"\bprovision\s+coverage\s+ratio\b", r"\bPCR\b"],
    "cost-to-income ratio": [r"cost[-\s]*to[-\s]*income"],
    "net interest margin": [r"\bNIM\b", r"net\s+interest\s+margin"],
    "capital adequacy ratio": [r"\bCRAR\b", r"\bCAR\b", r"capital\s+adequacy"],
    "tier 1 ratio": [r"\btier\s*1\b"],
    "cet1 ratio": [r"\bCET\s*1\b", r"\bCET1\b"],
    "eps (basic)": [r"\bEPS\b", r"earnings\s+per\s+share"],
    "roa": [r"\broa\b", r"return\s+on\s+assets"],
    "roe": [r"\broe\b", r"return\s+on\s+equity", r"return\s+on\s+average\s+net\s+worth"],
    "total income": [r"\btotal\s+income\b"],
    "net interest income": [r"\bNII\b", r"net\s+interest\s+income"],
    "other income": [r"\bother\s+income\b"],
    "operating profit / ppop": [r"\boperating\s+profit\b", r"\bPPOP\b"],
    "total assets": [r"\btotal\s+assets\b"],
    "shareholders’ funds": [r"shareholders[’']?\s+funds", r"net\s+worth"],
    "casa deposits (₹)": [r"\bCASA\s+deposits\b"],
    "time deposits (₹)": [r"\btime\s+deposits\b", r"\bterm\s+deposits\b"],
    "credit-to-deposit ratio": [r"\bcredit[-\s]*to[-\s]*deposit\b", r"\bCD\s*ratio\b"],
    "employees": [r"\bemployees\b", r"\bstaff\b"],
    "branches": [r"\bbranches\b"],
    "dividend per share": [r"\bdividend\s+per\s+share\b", r"\bDPS\b"],
    "book value per share": [r"\bbook\s+value\s+per\s+share\b", r"\bBVPS\b"],
    "casa ratio (avg)": [r"\baverage\s+CASA\s*ratio\b"],
    "casa share of deposits": [r"CASA\s+deposits\s+accounted\s+for", r"CASA\s+share"]
}

def norm(s): return s.lower().strip()

def year_tags_from_str(s: str):
    s = s.lower()
    tags = set()
    if any(t in s for t in ["2024-25","2024–25","fy2024-25","fy24-25","fy25","2025"]):
        tags.update(["2024-25","2024–25","fy24-25","fy 24-25","fy2024-25","fy25","fy 25","march 31, 2025","31 march 2025","for the year ended march 31, 2025","as at march 31, 2025"])
    if any(t in s for t in ["2023-24","2023–24","fy2023-24","fy23-24","fy24","2024"]):
        tags.update(["2023-24","2023–24","fy23-24","fy 23-24","fy2023-24","fy24","fy 24","march 31, 2024","31 march 2024","for the year ended march 31, 2024","as at march 31, 2024"])
    return [t.lower() for t in tags]

def has_header_unit(text):
    if re.search(r"(₹|rs\.?|inr)\s*(in)?\s*(crore|cr)\b", text, flags=re.I): return "crore"
    if re.search(r"(₹|rs\.?|inr)\s*(in)?\s*(million|mn)\b", text, flags=re.I): return "million"
    if re.search(r"(₹|rs\.?|inr)\s*(in)?\s*(billion|bn)\b", text, flags=re.I): return "billion"
    return None

def build_patterns(metric_key: str):
    amount_with_unit = r"(?:₹|rs\.?|inr)\s*([\d,]+(?:\.\d+)?)\s*(lakh\s+crore|crore|cr|billion|million|lakh|bn|mn)"
    percent_only   = r"(\d+(?:\.\d+)?)\s*%"
    syns = METRIC_SYNONYMS.get(metric_key, [re.escape(metric_key)])
    # amounts
    if any(k in metric_key for k in ["total deposits","total advances","pat","total income","net interest income","other income","operating profit / ppop","total assets","shareholders’ funds","casa deposits (₹)","time deposits (₹)","eps (basic)","dividend per share","book value per share"]):
        pat1 = rf"(?is)(?:{'|'.join(syns)})[\s\S]{{0,260}}{amount_with_unit}"
        pat2 = rf"(?is)(?:{'|'.join(syns)})[\s\S]{{0,260}}([\d,]+(?:\.\d+)?)(?!\s*%)"
        pat3 = rf"(?is)\b([\d,]+(?:\.\d+)?)\b(?!\s*%)\s*[\s\S]{{0,120}}(?:{'|'.join(syns)})"
        return [pat1, pat2, pat3]
    # ratios
    return [rf"(?is)(?:{'|'.join(syns)})[\s\S]{{0,160}}{percent_only}"]

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

def try_extract(text: str, metric_key: str, unit_hint: str=None):
    header_unit = has_header_unit(text)
    for pat in build_patterns(metric_key):
        m = re.search(pat, text, flags=re.I)
        if not m: continue
        groups = [g for g in m.groups() if g]
        val = next((g for g in groups if re.match(r"^[\d,]+(\.\d+)?$", g)), None)
        unit = ""
        for g in groups:
            if isinstance(g, str) and g.lower() in ["lakh crore","crore","cr","billion","million","lakh","bn","mn","%"]:
                unit = g; break
        if (not unit) and header_unit:
            unit = header_unit
        if val:
            return pretty_num(val.replace(",", ""), unit)
    return None

def extract_for_item(store, item):
    metric_key = norm(item.get("metric",""))
    unit_hint = norm(item.get("unit_hint","")) if item.get("unit_hint") else None
    ytags = year_tags_from_str(item.get("year",""))
    # strict year filter
    candidates = []
    for seg in store["segs"]:
        t = seg["text"]
        tl = t.lower()
        if ytags and not any(tag in tl for tag in ytags):
            continue
        syns = METRIC_SYNONYMS.get(metric_key, [metric_key])
        if any(re.search(s, t, flags=re.I) for s in syns):
            candidates.append(seg)
    if not candidates:
        return None, None
    for seg in candidates:
        ans = try_extract(seg["text"], metric_key, unit_hint)
        if ans: return ans, seg
    return None, candidates[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="indexes/store.pkl")
    ap.add_argument("--yaml", default="data/qas/gold.yaml")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--out", default="data/qa/qa_pairs_filled.csv")
    args = ap.parse_args()

    store = load_store(args.index)
    # load items
    items = []
    if args.csv:
        import pandas as pd
        df = pd.read_csv(args.csv)
        items = df.to_dict("records")
    else:
        # minimal YAML loader to avoid external deps
        # expects the exact structure we generate (simple key: value)
        cur = None
        for line in open(args.yaml, "r", encoding="utf-8"):
            line=line.rstrip("\n")
            if line.startswith("- id:"):
                if cur: items.append(cur)
                cur = {"id": line.split(":",1)[1].strip()}
            elif line.strip().startswith(("question:","answer:","year:","metric:","basis:","unit_hint:","section_hint:")) and cur is not None:
                k,v = line.strip().split(":",1)
                cur[k.strip()] = v.strip()
        if cur: items.append(cur)

    rows = []
    for it in items:
        ans, seg = extract_for_item(store, it)
        rows.append({
            "id": it.get("id",""),
            "year": it.get("year",""),
            "metric": it.get("metric",""),
            "question": it.get("question",""),
            "basis": it.get("basis",""),
            "unit_hint": it.get("unit_hint",""),
            "section_hint": it.get("section_hint",""),
            "answer": ans or "",
            "evidence_id": seg.get("id","") if seg else "",
            "evidence_source": seg.get("source","") if seg else "",
            "evidence_title": seg.get("title","") if seg else ""
        })

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {args.out} with {len(rows)} rows.")

if __name__ == "__main__":
    main()
