"""LangGraph state machine definition and graph construction.

This module defines:
- The StateGraph with 4 primary nodes: extract, validate, map_terms, compute_signals
- Conditional edges for retry routing and human-review flagging
- The compiled graph object for use by the pipeline runner

Graph topology:
    START → extract → validate →[confidence >= 0.75]→ map_terms → compute_signals → END
                         ↓                    ↑
                  [confidence < 0.75]──→ extract (retry, max 2)
                         ↓
                  [retries exhausted]──→ flag_human → END
"""
