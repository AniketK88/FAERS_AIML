"""PRR/ROR Statistics Agent — disproportionality signal detection.

Computes pharmacovigilance signal scores from FAERS data using:
- Proportional Reporting Ratio (PRR)
- Reporting Odds Ratio (ROR)
- Chi-squared test with Yates correction

Implements safety guards:
- Minimum case count thresholds
- Masking bias warnings
- Parameterized DuckDB queries (no SQL injection)
"""
