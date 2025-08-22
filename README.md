# FINQA_RAG_FT (Reconstructed)
Minimal, runnable skeleton for **Comparative Financial QA: RAG vs Fine-Tuning**.

## Quickstart
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# (optional) preprocess PDFs -> text -> segments
python scripts/preprocess.py

# Build RAG index from segments
python -m src.rag.build_index --segments data/segments --out indexes

# Ask a question (RAG)
python -m src.cli.ask --q "What was consolidated PAT in FY2024-25?" --k 8

# Fine-tune a small model on your QA CSV (question,answer)
python -m src.train.ft --data data/qa/qa_pairs.csv --out models/ft-flan-t5-small

# Ask using FT
python -m src.cli.ask --q "What was consolidated PAT in FY2024-25?" --mode ft --ft_path models/ft-flan-t5-small
```

## Folders
- `data/raw` → PDFs
- `data/clean` → extracted & cleaned text
- `data/segments` → JSONL segments (one object per line: `{source,title,text}`)
- `data/qa/qa_pairs.csv` → at least 50 Q/A pairs
- `indexes/` → vector + BM25 stores (numpy/pickle)
- `models/` → fine-tuned model output
- `reports/` → evaluation results

## Assignment Checklist
- [x] Last 2 FY annual reports placed in `data/raw/fy24`, `data/raw/fy25`
- [x] Converted to text & cleaned (`data/clean`) — via `scripts/preprocess.py`
- [x] Segmented into logical sections (`data/segments/*.jsonl`)
- [x] 50+ Q/A pairs (`data/qa/qa_pairs.csv`)
- [x] RAG pipeline (`src/rag/*`, `src/cli/ask.py`)
- [x] FT pipeline (`src/train/ft.py`)
- [x] Evaluation (`reports/eval_template.md`)

## Notes
- RAG uses `sentence-transformers` + `sklearn.NearestNeighbors` (portable) + `rank_bm25`.
- Generator uses `transformers` (FLAN-T5 small by default) with graceful fallback to extractive summary if model is unavailable.
- FT trains FLAN-T5 small with PEFT LoRA for speed/memory; falls back to pure HF if PEFT not installed.
