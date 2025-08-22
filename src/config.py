from dataclasses import dataclass

@dataclass
class Config:
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    gen_model: str = "google/flan-t5-small"
    index_dir: str = "indexes"
    segments_dir: str = "data/segments"
    qa_path: str = "data/qa/qa_pairs.csv"
    top_k_dense: int = 6
    top_k_sparse: int = 6
    max_input_tokens: int = 1024
