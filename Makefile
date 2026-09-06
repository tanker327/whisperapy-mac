.PHONY: dev test lint format typecheck check clean install install-dev

HOST ?= 0.0.0.0
PORT ?= 8000
TEMP_DIR ?= /tmp/whisperapy

dev:
	uv run uvicorn app.main:app --reload --host $(HOST) --port $(PORT)

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run pyright

check: lint typecheck test

clean:
	rm -rf "$(TEMP_DIR)"/* htmlcov .coverage .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +

install:
	uv sync

install-dev:
	uv sync --extra dev
