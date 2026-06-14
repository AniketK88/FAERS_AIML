# AET-SE Data Directory
## What Goes Where

This directory holds all data for the AET-SE pipeline. **Raw data is gitignored** —
you must download it yourself using the instructions below.

---

## Directory Layout

```
data/
├── raw/                          # Original downloads — NEVER modify these
│   ├── faers/                    # FDA FAERS quarterly CSV zips
│   │   ├── faers_ascii_2024Q3/   # Extracted: DEMO, DRUG, REAC, OUTC, etc.
│   │   └── faers_ascii_2024Q4/
│   ├── reviews/                  # Kaggle Drug Reviews dataset
│   │   └── drugsComTrain_raw.tsv # or drugsCom_raw.tsv
│   └── rxnorm/                   # RxNorm RxTerms vocabulary
│       └── RxTerms202406.txt     # Tab-separated RxTerms file
│
├── processed/                    # Cleaned, filtered intermediate files
│   ├── reviews_filtered.parquet  # ~30K AE-relevant reviews
│   └── rxnorm_lookup.parquet     # Normalized RxNorm drug→generic mapping
│
├── duckdb/                       # DuckDB database files
│   └── faers.duckdb              # Main FAERS + signals database
│
├── chroma_db/                    # ChromaDB persistent vector store
│   └── (auto-generated)         # MedDRA PT embeddings
│
├── cache/                        # LLM and embedding caches
│   ├── llm_extractions/          # JSON files keyed by hash(review_text)
│   └── embeddings/               # Cached sentence embeddings
│
├── ground_truth/                 # Manual labels for evaluation
│   ├── labeled_reviews.json      # 50 manually labeled reviews
│   └── labeling_guide.md         # Instructions for labelers
│
└── eval_results/                 # Evaluation output files
    ├── extraction_metrics.json
    └── signal_validation.json
```

---

## Download Instructions

### 1. FAERS Data (FDA — free, no account needed)

Download the **two most recent quarterly ASCII ZIP files** from:
https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html

For this project, download:
- `faers_ascii_2024Q3.zip`
- `faers_ascii_2024Q4.zip`

```bash
# From project root
cd data/raw/faers/
wget https://fis.fda.gov/content/Exports/faers_ascii_2024Q3.zip
wget https://fis.fda.gov/content/Exports/faers_ascii_2024Q4.zip
unzip faers_ascii_2024Q3.zip -d faers_ascii_2024Q3
unzip faers_ascii_2024Q4.zip -d faers_ascii_2024Q4
```

Each extracted folder contains these key files:
| File      | Contents                                 |
|-----------|------------------------------------------|
| DEMO*.txt | Demographics (caseid, age, sex, country)  |
| DRUG*.txt | Drug information (names, roles)           |
| REAC*.txt | Reactions (MedDRA Preferred Terms)        |
| OUTC*.txt | Outcomes (death, hospitalization, etc.)   |

### 2. Kaggle Drug Reviews

Dataset: https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018

```bash
# Requires Kaggle API key (see .env.example for KAGGLE_USERNAME, KAGGLE_KEY)
pip install kaggle
kaggle datasets download -d jessicali9530/kuc-hackathon-winter-2018 -p data/raw/reviews/
cd data/raw/reviews/
unzip kuc-hackathon-winter-2018.zip
```

### 3. RxNorm RxTerms (NLM — free, no account needed)

Download from: https://data.nlm.nih.gov/umls/sourcereleasedocs/current/RXNORM/

Specifically, download the RxTerms file:
https://lhncbc.nlm.nih.gov/RxNav/applications/RxTermsRelease.html

```bash
cd data/raw/rxnorm/
wget https://data.nlm.nih.gov/umls/kss/rxterms/RxTerms202406.zip
unzip RxTerms202406.zip
```

---

## Gitignore Rules

The following patterns are gitignored (add to your .gitignore):
```
data/raw/
data/processed/
data/duckdb/
data/chroma_db/
data/cache/
data/eval_results/
```

Only `data/ground_truth/` is version-controlled (small, curated labels).
