.PHONY: install test lint

install:
pip install -r requirements.txt

lint:
flake8 src tests || true

test:
pytest -q
