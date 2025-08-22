import argparse, pathlib, numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from .utils import load_segments, save_pickle, normalize_text
from ..config import Config
from sklearn.neighbors import NearestNeighbors

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--segments', default=Config.segments_dir)
    ap.add_argument('--out', default=Config.index_dir)
    args = ap.parse_args()

    segs = load_segments(args.segments)
    if not segs:
        print(f"No segments found in {args.segments}."); return

    texts = [normalize_text(s['text']) for s in segs]
    # Dense embeddings
    model = SentenceTransformer(Config.embed_model)
    emb = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

    # KNN index
    knn = NearestNeighbors(n_neighbors=min(50, len(texts)), metric='cosine')
    knn.fit(emb)

    # Sparse BM25
    tokenized = [t.split() for t in texts]
    bm25 = BM25Okapi(tokenized)

    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    save_pickle({'segs': segs, 'emb': emb, 'bm25': bm25, 'tokenized': tokenized, 'knn': knn}, out / "store.pkl")
    print(f"Indexed {len(segs)} segments -> {out/'store.pkl'}")

if __name__ == "__main__":
    main()
