#!/usr/bin/env python3
"""
Re-chunk existing segment texts to produce alternate chunk sizes (e.g., 100 vs 400 words).
Usage:
  python scripts/rechunk.py --inp data/segments --out data/segments_100 --chunk_words 100 --stride_words 25

Assumptions:
  - --inp contains a manifest.json and one or more *.jsonl files with fields {id, source, title, text}.
  - This script groups segments by their source file (jsonl) and re-slices text by words.
  - Output is written as *.jsonl files + a manifest.json under --out.
"""
import argparse, json, pathlib, re
from typing import List
try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

class Reranker:
    def __init__(self, name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        if CrossEncoder is None:
            raise ImportError("sentence-transformers not installed. Run: pip install -U sentence-transformers")
        self.model = CrossEncoder(name)

    def rerank(self, query: str, candidates: List[str]) -> List[float]:
        pairs = [(query, c) for c in candidates]
        scores = self.model.predict(pairs)
        return [float(s) for s in scores]


def load_segments(seg_dir):
    seg_dir = pathlib.Path(seg_dir)
    if not seg_dir.exists():
        raise FileNotFoundError(f"{seg_dir} not found")
    jsonls = sorted(seg_dir.glob("*.jsonl")) or sorted(seg_dir.glob("**/*.jsonl"))
    if not jsonls:
        raise FileNotFoundError(f"No .jsonl segments found under {seg_dir}")
    data_by_file = {}
    for jp in jsonls:
        items = []
        with open(jp, "r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if not line: continue
                try:
                    items.append(json.loads(line))
                except Exception:
                    pass
        data_by_file[jp.name] = items
    return data_by_file

def chunk_words(text, size, stride):
    words = re.split(r"\s+", text.strip())
    out, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i:i+size])
        if chunk.strip():
            out.append(chunk)
        if i + size >= len(words):
            break
        i += max(1, size - stride)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True, help="Input segments dir (with manifest.json and *.jsonl)")
    ap.add_argument("--out", required=True, help="Output dir for re-chunked segments")
    ap.add_argument("--chunk_words", type=int, default=100)
    ap.add_argument("--stride_words", type=int, default=25)
    args = ap.parse_args()

    inp = pathlib.Path(args.inp)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data_by_file = load_segments(inp)

    manifest = {"files": []}
    for fname, segs in data_by_file.items():
        # merge all texts to keep ordering; preserve rough title/source
        all_text = []
        title = segs[0].get("title","") if segs else ""
        source = segs[0].get("source","") if segs else ""
        for s in segs:
            t = s.get("text","")
            if t: all_text.append(t)
        merged = "\n\n".join(all_text)
        chunks = chunk_words(merged, size=args.chunk_words, stride=args.stride_words)

        op = out / fname
        with open(op, "w", encoding="utf-8") as f:
            for i, ch in enumerate(chunks):
                rec = {
                    "id": f"{fname}:{i}",
                    "title": title,
                    "source": source if source else f"{fname.replace('.jsonl','')}",
                    "text": ch
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        manifest["files"].append({"file": fname, "segments": len(chunks)})

    with open(out / "manifest.json", "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)
    print(f"Re-chunked {len(manifest['files'])} files -> {out} (chunk={args.chunk_words}, stride={args.stride_words})")

if __name__ == "__main__":
    main()
