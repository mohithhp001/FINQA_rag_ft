import os, json, pickle, numpy as np, pathlib, re

def load_segments(segments_dir):
    segs = []
    for p in pathlib.Path(segments_dir).glob("*.jsonl"):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if 'text' in obj and obj['text'].strip():
                        obj['id'] = f"{p.name}:{len(segs)}"
                        segs.append(obj)
                except Exception:
                    continue
    return segs

def normalize_text(t):
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
