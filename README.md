````markdown
# FINQA_RAG_FT — Comparative Financial QA (RAG vs Fine-Tuning)

End-to-end skeleton for answering financial questions from **HDFC Bank annual reports** using:
- **RAG** (retrieval-augmented generation),
- **Fine-Tuning** (FLAN-T5 small experts),
- **MoE** (numeric vs narrative),
- **Streamlit UI** for demo.

---

## Quickstart

```bash
# Create and activate venv
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Preprocess PDFs → JSONL segments
python scripts/preprocess.py

# Build RAG index
python -m src.rag.build_index --segments data/segments --out indexes

# Ask a question (RAG baseline)
python -m src.cli.ask --q "What was consolidated PAT in FY2024-25?" --k 8

# Fine-tune a small model (FLAN-T5) on your QA CSV
python -m src.train.ft --data data/qa/qa_pairs.csv --out models/ft-flan-t5-small

# Ask using FT
python -m src.cli.ask --q "What was consolidated PAT in FY2024-25?" \
  --mode ft --ft_path models/ft-flan-t5-small
````

---

## Advanced Usage

### Re-chunk for experiments

```bash
# Smaller chunks (100 words)
python scripts/rechunk.py --inp data/segments --out data/segments_100 --chunk_words 100 --stride_words 25
python -m src.rag.build_index --segments data/segments_100 --out indexes_100

# Larger chunks (400 words, good for tables)
python scripts/rechunk.py --inp data/segments --out data/segments_400 --chunk_words 400 --stride_words 80
python -m src.rag.build_index --segments data/segments_400 --out indexes_400
```

### RAG with reranker (better ranking)

```bash
python -m src.cli.ask --q "Total deposits in FY2024-25?" \
  --k 12 --index indexes_400 --extractive_only --rerank
```

### Fine-Tuning with MoE

Split dataset into numeric vs narrative:

```bash
python scripts/split_num_nar.py
```

Train experts:

```bash
python -m src.train.ft --data data/qa/train_num.csv --out models/ft-flan-t5-small-exp-num
python -m src.train.ft --data data/qa/train_nar.csv --out models/ft-flan-t5-small-exp-nar
```

Run MoE inference:

```bash
python -m src.cli.ask_ft_moe --q "Total deposits in FY2024-25?" \
  --num models/ft-flan-t5-small-exp-num \
  --nar models/ft-flan-t5-small-exp-nar \
  --index indexes_400 --k 12 --rerank
```

### Streamlit UI

```bash
streamlit run app/app.py
```

* Modes: **RAG**, **RAG+Re-rank**, **FT-MoE**
* Shows answer, raw output, and evidence.

### Evaluation

```bash
python scripts/eval_compare.py --qs data/qa/qa_pairs.csv --out reports/compare.csv --index indexes_400
python scripts/eval_ft.py --qs data/qa/qa_pairs.csv --out reports/ft_vs_rag.csv \
  --index indexes_400 --num models/ft-flan-t5-small-exp-num --nar models/ft-flan-t5-small-exp-nar
```

---

## Folder Layout

```
data/
 ├─ raw/           # PDFs (HDFC_IAR_FY24.pdf, HDFC_IAR_FY25.pdf)
 ├─ segments/      # default chunks
 ├─ segments_100/  # re-chunked (100 words)
 ├─ segments_400/  # re-chunked (400 words)
 └─ qa/            # QA pairs CSVs (≥50 required)

indexes/           # built vector indexes
models/            # fine-tuned models
scripts/           # preprocess, rechunk, eval
src/               # code (rag, train, cli, ft_moe)
app/               # Streamlit app
reports/           # evaluation outputs
```

---

## Assignment Checklist ✅

* [x] **2 FY reports** → `data/raw/fy24`, `data/raw/fy25`
* [x] **Preprocessing & cleaning** → `scripts/preprocess.py`
* [x] **Segmentation** → `data/segments/*.jsonl`
* [x] **≥50 Q/A pairs** → `data/qa/qa_pairs.csv`
* [x] **RAG pipeline** → `src/rag`, `src/cli/ask.py`
* [x] **Fine-tuning pipeline** → `src/train/ft.py`
* [x] **MoE experts** → numeric + narrative
* [x] **Evaluation scripts** → `scripts/eval_compare.py`, `scripts/eval_ft.py`
* [x] **UI** → `app/app.py`

---

## Notes

* RAG = hybrid search (`sentence-transformers` + BM25).
* Re-ranker = cross-encoder (`sentence-transformers`).
* FT = FLAN-T5-small, optionally with PEFT/LoRA.
* Guardrails: if FT fails → fallback to RAG-extractive.
* Streamlit UI gives side-by-side comparison.

```

Do you want me to also include a **“How teammates can replicate the full pipeline (A→Z)” section** with a single script-like flow? That could make it super clear for them.
```
