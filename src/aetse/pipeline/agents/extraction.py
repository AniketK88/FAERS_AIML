"""Extraction Agent — LLM-based adverse event entity extraction.

Uses Llama 3.1 8B via Ollama to extract:
- Drug names
- Adverse reactions
- Severity classification

From free-text drug reviews. Includes:
- JSON parsing with regex fallback
- Retry with exponential backoff via tenacity
- Disk-based caching keyed by xxhash of input text
"""
