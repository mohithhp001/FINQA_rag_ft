from .utils import load_pickle, normalize_text
from ..config import Config
import numpy as np

def hybrid_search(query, store, top_k_dense=Config.top_k_dense, top_k_sparse=Config.top_k_sparse):
    # Dense
    q_emb = store['emb'][0:1] * 0  # placeholder shape
    # Compute embedding via stored model? We didn't store the model; we only need neighbor distances via knn.kneighbors on embedding computed externally.
    # For simplicity we require caller to pass precomputed q_emb; but to keep API simple, we compute dynamically using same model name.
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(Config.embed_model)
    q_emb = model.encode([normalize_text(query)], convert_to_numpy=True, normalize_embeddings=True)

    # Dense scores (cosine distance -> similarity)
    dists, idxs = store['knn'].kneighbors(q_emb, n_neighbors=min(top_k_dense, len(store['segs'])))
    dense_hits = [(int(i), 1 - float(d)) for i, d in zip(idxs[0], dists[0])]

    # Sparse BM25
    toks = normalize_text(query).split()
    bm25_scores = store['bm25'].get_scores(toks)
    sparse_hits = sorted([(i, float(s)) for i, s in enumerate(bm25_scores)], key=lambda x: x[1], reverse=True)[:top_k_sparse]

    # Normalize and fuse
    if dense_hits:
        maxd = max(s for _, s in dense_hits) or 1.0
        dense_hits = [(i, s/maxd) for i, s in dense_hits]
    if sparse_hits:
        maxs = max(s for _, s in sparse_hits) or 1.0
        sparse_hits = [(i, s/maxs) for i, s in sparse_hits]

    # Weighted sum
    weight_dense, weight_sparse = 0.55, 0.45
    agg = {}
    for i, s in dense_hits + sparse_hits:
        agg[i] = agg.get(i, 0.0) + (s * (weight_dense if (i, s) in dense_hits else weight_sparse))
    ranked = sorted(agg.items(), key=lambda x: x[1], reverse=True)
    return ranked

def gather_context(indices, store, max_chars=1800):
    ctxs = []
    total = 0
    for i, score in indices:
        seg = store['segs'][i]
        text = seg['text']
        ctxs.append((seg, score))
        total += len(text)
        if total >= max_chars: break
    return ctxs
