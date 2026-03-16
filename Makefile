.PHONY: dev test lint format check clean install install-dev

dev:
	uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check .

format:
	uv run black .

check: lint format test

clean:
	rm -rf /tmp/whisperapy/*
	find . -type d -name __pycache__ -exec rm -rf {} +

install:
	uv sync

install-dev:
	uv sync --extra dev
