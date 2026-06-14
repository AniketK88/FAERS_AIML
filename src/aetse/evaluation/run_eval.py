"""Evaluation runner — computes P/R/F1 and validates signals.

This module handles:
- Comparing LLM extractions against ground truth labels
- Computing entity-level precision, recall, F1 for drugs and reactions
- Validating PRR/ROR signals against known positive controls
- Generating evaluation reports as JSON

Run with: python -m aetse.evaluation.run_eval
"""
