# =============================================================================
# AET-SE Makefile
# Adverse Event Triage & Signal Detection Engine
# =============================================================================

.PHONY: setup ingest-faers ingest-reviews run-pipeline run-app run-eval test \
        lint clean pull-models check-prereqs help

PYTHON := .venv/bin/python
PIP := .venv/bin/pip
STREAMLIT := .venv/bin/streamlit
PYTEST := .venv/bin/pytest

# Default target
help:  ## Show this help message
	@echo "AET-SE — Adverse Event Triage & Signal Detection Engine"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---- Setup ----

setup: .venv check-prereqs  ## Create venv, install deps, copy .env, pull models
	@echo "✅ Setup complete. Activate with: source .venv/bin/activate"

.venv:
	python3 -m venv .venv
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	@if [ ! -f .env ]; then cp .env.example .env; echo "📄 Created .env from template"; fi
	@mkdir -p data/raw/faers data/raw/reviews data/raw/rxnorm
	@mkdir -p data/processed data/duckdb data/chroma_db
	@mkdir -p data/cache/llm_extractions data/cache/embeddings
	@mkdir -p data/ground_truth data/eval_results
	@mkdir -p logs

check-prereqs:  ## Verify Ollama, Python, GPU are available
	@echo "🔍 Checking prerequisites..."
	@command -v ollama >/dev/null 2>&1 || { echo "❌ Ollama not installed. See: https://ollama.com/download"; exit 1; }
	@python3 -c "import sys; assert sys.version_info >= (3, 12), 'Python 3.12+ required'" || exit 1
	@echo "  ✅ Python $$(python3 --version)"
	@echo "  ✅ Ollama $$(ollama --version 2>/dev/null)"
	@nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null && echo "  ✅ GPU detected" || echo "  ⚠️  No GPU — will use CPU (slower)"
	@echo "✅ All prerequisites met."

pull-models:  ## Pull required Ollama models
	@echo "📥 Pulling Ollama models (this may take a while)..."
	ollama pull llama3.1:8b-instruct-q4_K_M
	@echo "✅ Primary model ready."
	@echo "💡 Optional fallback: ollama pull phi3:mini"

# ---- Data Ingestion ----

ingest-faers:  ## Ingest and normalize FAERS quarterly data into DuckDB
	$(PYTHON) -m aetse.data.ingest_faers

ingest-reviews:  ## Ingest and filter Kaggle drug reviews
	$(PYTHON) -m aetse.data.ingest_reviews

ingest-rxnorm:  ## Load RxNorm RxTerms for drug name normalization
	$(PYTHON) -m aetse.data.ingest_rxnorm

ingest-all: ingest-faers ingest-rxnorm ingest-reviews  ## Run all ingestion steps in order

# ---- Pipeline ----

run-pipeline:  ## Run the LangGraph multi-agent pipeline
	$(PYTHON) -m aetse.pipeline.runner

run-pipeline-sample:  ## Run pipeline on 10 sample reviews (quick test)
	$(PYTHON) -m aetse.pipeline.runner --sample 10

# ---- Dashboard ----

run-app:  ## Launch the Streamlit dashboard
	$(STREAMLIT) run src/aetse/dashboard/app.py --server.port 8501

# ---- Evaluation ----

run-eval:  ## Run evaluation framework (P/R/F1, signal validation)
	$(PYTHON) -m aetse.evaluation.run_eval

# ---- Testing ----

test:  ## Run all tests with pytest
	$(PYTEST) tests/ -v --tb=short --cov=src/aetse --cov-report=term-missing

test-unit:  ## Run unit tests only (fast, no LLM)
	$(PYTEST) tests/unit/ -v --tb=short

test-integration:  ## Run integration tests (requires Ollama running)
	$(PYTEST) tests/integration/ -v --tb=short -m integration

# ---- Code Quality ----

lint:  ## Run ruff linter and formatter check
	.venv/bin/ruff check src/ tests/
	.venv/bin/ruff format --check src/ tests/

format:  ## Auto-format code with ruff
	.venv/bin/ruff format src/ tests/
	.venv/bin/ruff check --fix src/ tests/

# ---- Cleanup ----

clean:  ## Remove caches, temp files, and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage htmlcov/

clean-data:  ## Remove all processed data (keeps raw downloads)
	rm -rf data/processed/* data/duckdb/* data/chroma_db/*
	rm -rf data/cache/llm_extractions/* data/cache/embeddings/*
	@echo "⚠️  Cleaned processed data. Raw data preserved in data/raw/"

clean-all: clean clean-data  ## Full clean including processed data
	rm -rf .venv/
	@echo "⚠️  Removed virtual environment. Run 'make setup' to recreate."
