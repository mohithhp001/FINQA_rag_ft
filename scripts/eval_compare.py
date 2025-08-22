#!/usr/bin/env python3
"""
Run a small evaluation across modes and log a comparison CSV.
Usage:
  python scripts/eval_compare.py --qs data/qa/qa_pairs_filled_train.csv --out reports/compare.csv --n 10 --index indexes_100
"""
import argparse, csv, subprocess, json, time, pandas as pd

def run_cmd(cmd):
    t0 = time.time()
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    dt = time.time()-t0
    return json.loads(out), dt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--qs', required=True, help='CSV with at least question,answer')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--index', default='indexes')
    args = ap.parse_args()

    df = pd.read_csv(args.qs)
    if 'question' not in df.columns: raise ValueError("CSV must have question column")
    if 'answer' not in df.columns: df['answer'] = ""

    qs = df.sample(min(args.n, len(df)), random_state=42).to_dict("records")

    rows = []
    for row in qs:
        q = str(row['question'])
        gold = str(row.get('answer',"")).strip()

        modes = [
            ("rag", ["python","-m","src.cli.ask","--q",q,"--k","12","--index",args.index]),
            ("rag_rr", ["python","-m","src.cli.ask","--q",q,"--k","12","--index",args.index,"--rerank"]),
        ]
        for name, cmd in modes:
            try:
                resp, dt = run_cmd(cmd)
                ans = resp.get("answer","")
                ok = (gold and ans and gold.lower().strip()==ans.lower().strip())
                rows.append({"question": q, "gold": gold, "mode": name, "answer": ans, "correct": int(ok), "time_s": round(dt,3)})
            except Exception as e:
                rows.append({"question": q, "gold": gold, "mode": name, "answer": "", "correct": 0, "time_s": -1, "error": str(e)})

    # write csv
    import pathlib
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {args.out} ({len(rows)} rows).")

if __name__ == "__main__":
    main()
