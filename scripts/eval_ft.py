#!/usr/bin/env python3
import argparse, csv, json, subprocess, time, pandas as pd, pathlib

def run(cmd):
    t0 = time.time()
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    return json.loads(out), round(time.time()-t0, 3)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qs", required=True, help="CSV with question,answer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--index", default="indexes_400")
    ap.add_argument("--num", default="models/ft-flan-t5-small-exp-num")
    ap.add_argument("--nar", default="models/ft-flan-t5-small-exp-nar")
    args = ap.parse_args()

    df = pd.read_csv(args.qs)
    if "question" not in df.columns: raise ValueError("CSV must have a 'question' column")
    if "answer" not in df.columns: df["answer"] = ""
    qs = df.sample(min(args.n, len(df)), random_state=42).to_dict("records")

    rows = []
    for r in qs:
        q = str(r["question"])
        gold = str(r.get("answer","")).strip()

        modes = [
            ("rag",      ["python","-m","src.cli.ask","--q",q,"--k","12","--index",args.index,"--extractive_only"]),
            ("rag_rr",   ["python","-m","src.cli.ask","--q",q,"--k","12","--index",args.index,"--extractive_only","--rerank"]),
            ("ft_moe",   ["python","-m","src.cli.ask_ft_moe","--q",q,"--num",args.num,"--nar",args.nar,"--index",args.index,"--k","12","--rerank"]),
        ]

        for name, cmd in modes:
            try:
                resp, dt = run(cmd)
                ans = resp.get("answer","")
                ok = (gold and ans and gold.lower().strip()==gold.lower().strip())
                rows.append({
                    "question": q, "gold": gold, "mode": name,
                    "answer": ans, "correct": int(ok),
                    "time_s": dt,
                    "fallback_used": resp.get("fallback_used","")
                })
            except subprocess.CalledProcessError as e:
                rows.append({
                    "question": q, "gold": gold, "mode": name,
                    "answer": "", "correct": 0, "time_s": -1,
                    "error": e.output[:2000]
                })

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {args.out} ({len(rows)} rows).")

if __name__ == "__main__":
    main()
