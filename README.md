# FINQA: Comparative Financial QA System

This project implements a coursework-style system for answering questions over
financial statements using two approaches:

1. **Retrieval‑Augmented Generation (RAG)**
2. **Fine‑Tuned Language Model (FT)**

The repository starts with a minimal RAG baseline and will evolve to include
fine‑tuning, unified evaluation, and reproducibility tooling.

## Repository Structure

```
configs/    # configuration files
data/       # placeholder for financial statements and QA pairs
src/        # core Python packages
scripts/    # command line interfaces and demos
eval/       # evaluation reports and notebooks
tests/      # unit tests
```

## Installation & Testing

```bash
make install  # install dependencies
make test     # run the test suite
```

## Minimal Usage Example

```python
from finqa import FinancialRAG

docs = [
    "Apple reported record revenue of $90B in Q1 2024",
    "Tesla's earnings grew 20% year over year",
]

rag = FinancialRAG(docs)
print(rag.answer("How much revenue did Apple report?"))
```

## Roadmap

The project will be extended in stages:

1. **RAG baseline** – loaders, embedders, vector and sparse indexes, CLI.
2. **Fine‑tuning baseline** – dataset loader, trainer, evaluation utilities.
3. **Unified evaluation** – accuracy and latency metrics comparing RAG and FT.
4. **Reproducibility** – configuration files, seed control, Dockerfile.

See `assignment_2_comparative_financial_qa_system_rag_vs_fine_tuning_CLEAN.txt`
for the full assignment specification.
