"""Day 5 — Drug Reviews Ingestion + scispaCy NER Baseline.

Loads Kaggle UCI Drug Review Dataset, filters to 11 target drugs,
applies text quality filters, stores in DuckDB, runs scispaCy NER
on AE-mention reviews, and generates a manual labeling template.

Steps:
    A. Load CSV files with Polars
    B. Filter to target drugs (case-insensitive)
    C. Text quality filter (strip HTML, word_count > 20)
    D. Store in DuckDB drug_reviews table
    E. Run scispaCy NER on first 500 AE-mention reviews
    F. Generate manual labeling template (50 rows)

Usage:
    python -m aetse.data.ingest_reviews

All file I/O uses Polars (not Pandas) per project rules.
DuckDB queries use parameterized values.
"""

from __future__ import annotations

import gc
import hashlib
import re
import time
from pathlib import Path
from typing import Any

import polars as pl

from aetse.config.settings import settings
from aetse.utils.db import get_duckdb_connection
from aetse.utils.logging import logger


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_DRUGS: list[str] = [
    "ibuprofen",
    "naproxen",
    "aspirin",
    "diclofenac",
    "celecoxib",
    "rofecoxib",
    "amlodipine",
    "lisinopril",
    "metformin",
    "atorvastatin",
    "metoprolol",
]

AE_KEYWORDS: list[str] = [
    "side effect",
    "adverse",
    "reaction",
    "allerg",
    "nausea",
    "vomit",
    "dizziness",
    "headache",
    "pain",
    "bleeding",
    "rash",
    "swelling",
    "fatigue",
    "diarrhea",
    "constipation",
]

# HTML entities to strip
HTML_PATTERNS: list[tuple[str, str]] = [
    (r"<br\s*/?>", " "),
    (r"&amp;", "&"),
    (r"&quot;", '"'),
    (r"&#039;", "'"),
    (r"&lt;", "<"),
    (r"&gt;", ">"),
    (r"<[^>]+>", " "),  # Catch remaining HTML tags
]

REVIEWS_DIR: Path = settings.project_root / "data" / "raw" / "reviews"
GROUND_TRUTH_DIR: Path = settings.project_root / "data" / "ground_truth"


# ---------------------------------------------------------------------------
# Step A: Load CSV files
# ---------------------------------------------------------------------------

def _generate_review_id(drug_name: str, review_text: str) -> str:
    """Generate deterministic review_id from drug name and review prefix."""
    key = f"{drug_name}{review_text[:50]}"
    return hashlib.sha256(key.encode("utf-8", errors="replace")).hexdigest()[:12]


def load_review_files() -> pl.DataFrame:
    """Load train + test CSV files and combine.

    Returns:
        Polars DataFrame with all reviews and source column.
    """
    logger.info("Step A: Loading review CSV files...")

    frames: list[pl.DataFrame] = []

    for filename, source_label in [
        ("drugsComTrain_raw.csv", "train"),
        ("drugsComTest_raw.csv", "test"),
        ("drugsComTrain_raw.tsv", "train"),
        ("drugsComTest_raw.tsv", "test"),
    ]:
        path = REVIEWS_DIR / filename
        if not path.exists():
            continue

        # Detect separator
        sep = "\t" if filename.endswith(".tsv") else ","

        logger.info(f"  Reading {filename} ({path.stat().st_size / 1024 / 1024:.1f} MB)")

        try:
            df = pl.read_csv(
                path,
                separator=sep,
                has_header=True,
                infer_schema_length=10000,
                encoding="utf8-lossy",
                ignore_errors=True,
                truncate_ragged_lines=True,
            )
        except Exception as e:
            logger.warning(f"  Failed to read {filename}: {e}")
            continue

        # Normalize column names
        df = df.rename({col: col.lower().strip() for col in df.columns})

        # Add source column
        df = df.with_columns(pl.lit(source_label).alias("source"))

        logger.info(f"  → {df.height:,} rows, columns: {df.columns}")
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No review files found in {REVIEWS_DIR}")

    combined = pl.concat(frames, how="diagonal_relaxed")
    logger.info(f"  Combined: {combined.height:,} reviews")
    return combined


# ---------------------------------------------------------------------------
# Step B: Filter to target drugs
# ---------------------------------------------------------------------------

def filter_target_drugs(df: pl.DataFrame) -> pl.DataFrame:
    """Filter to reviews mentioning any of the 11 target drugs.

    Args:
        df: Full reviews DataFrame.

    Returns:
        Filtered DataFrame with drug_norm column added.
    """
    logger.info("Step B: Filtering to target drugs...")

    # Ensure drugname column exists
    drug_col = None
    for candidate in ["drugname", "drug_name", "drugName"]:
        if candidate in df.columns:
            drug_col = candidate
            break

    if drug_col is None:
        # Try case-insensitive match
        lower_cols = {c.lower(): c for c in df.columns}
        if "drugname" in lower_cols:
            drug_col = lower_cols["drugname"]
        else:
            raise ValueError(f"No drug name column found in: {df.columns}")

    # Build case-insensitive filter
    drug_lower = pl.col(drug_col).cast(pl.Utf8).str.to_lowercase()
    conditions = [drug_lower.str.contains(drug) for drug in TARGET_DRUGS]
    combined_filter = conditions[0]
    for c in conditions[1:]:
        combined_filter = combined_filter | c

    filtered = df.filter(combined_filter)

    # Add drug_norm column: which target drug matched
    def _match_drug(name: str | None) -> str | None:
        if name is None:
            return None
        name_lower = name.lower()
        for drug in TARGET_DRUGS:
            if drug in name_lower:
                return drug
        return None

    filtered = filtered.with_columns(
        pl.col(drug_col)
        .cast(pl.Utf8)
        .map_elements(_match_drug, return_dtype=pl.Utf8)
        .alias("drug_norm")
    )

    # Drop any rows where drug_norm is still null (shouldn't happen)
    filtered = filtered.filter(pl.col("drug_norm").is_not_null())

    logger.info(f"  → {filtered.height:,} reviews for target drugs")

    # Log distribution
    dist = filtered.group_by("drug_norm").len().sort("len", descending=True)
    for row in dist.iter_rows():
        logger.info(f"    {row[0]:<15} {row[1]:>6,}")

    return filtered


# ---------------------------------------------------------------------------
# Step C: Text quality filter
# ---------------------------------------------------------------------------

def _strip_html(text: str) -> str:
    """Strip HTML entities and tags from text."""
    if not text:
        return ""
    for pattern, replacement in HTML_PATTERNS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove surrounding quotes that Kaggle dataset uses
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()
    return text


def apply_text_filters(df: pl.DataFrame) -> pl.DataFrame:
    """Apply text quality filters.

    - Strip HTML
    - Require word_count > 20
    - Remove null/empty reviews

    Args:
        df: Filtered reviews DataFrame.

    Returns:
        Quality-filtered DataFrame with clean review_text and word_count.
    """
    logger.info("Step C: Applying text quality filters...")

    # Find review column
    review_col = None
    for candidate in ["review", "review_text", "reviewtext"]:
        if candidate in df.columns:
            review_col = candidate
            break

    if review_col is None:
        raise ValueError(f"No review text column found in: {df.columns}")

    # Strip HTML and compute word count
    df = df.with_columns(
        pl.col(review_col)
        .cast(pl.Utf8)
        .map_elements(_strip_html, return_dtype=pl.Utf8)
        .alias("review_text_clean")
    )

    df = df.with_columns(
        pl.col("review_text_clean")
        .str.split(" ")
        .list.len()
        .alias("word_count")
    )

    before = df.height
    df = df.filter(
        pl.col("review_text_clean").is_not_null()
        & (pl.col("review_text_clean").str.len_chars() > 0)
        & (pl.col("word_count") > 20)
    )

    logger.info(f"  → {before:,} → {df.height:,} after quality filter ({before - df.height:,} removed)")

    return df


# ---------------------------------------------------------------------------
# Step D: DuckDB storage
# ---------------------------------------------------------------------------

def _has_ae_mention(text: str) -> bool:
    """Check if text mentions any adverse event keyword."""
    if not text:
        return False
    text_lower = text.lower()
    return any(kw in text_lower for kw in AE_KEYWORDS)


def store_reviews_in_duckdb(df: pl.DataFrame) -> int:
    """Store reviews in DuckDB drug_reviews table.

    Args:
        df: Quality-filtered reviews DataFrame.

    Returns:
        Number of rows inserted.
    """
    logger.info("Step D: Storing reviews in DuckDB...")

    # Find the right column names
    drug_col = None
    for candidate in ["drugname", "drug_name"]:
        if candidate in df.columns:
            drug_col = candidate
            break

    condition_col = None
    for candidate in ["condition"]:
        if candidate in df.columns:
            condition_col = candidate
            break

    rating_col = None
    for candidate in ["rating"]:
        if candidate in df.columns:
            rating_col = candidate
            break

    # Generate review_ids and ae_mention flag
    df = df.with_columns(
        pl.struct([drug_col, "review_text_clean"])
        .map_elements(
            lambda row: _generate_review_id(
                str(row[drug_col] or ""),
                str(row["review_text_clean"] or ""),
            ),
            return_dtype=pl.Utf8,
        )
        .alias("review_id")
    )

    df = df.with_columns(
        pl.col("review_text_clean")
        .map_elements(_has_ae_mention, return_dtype=pl.Boolean)
        .alias("has_ae_mention")
    )

    # Build final DataFrame for insertion
    insert_df = df.select([
        pl.col("review_id"),
        pl.col(drug_col).cast(pl.Utf8).alias("drug_name"),
        pl.col("drug_norm"),
        pl.col(condition_col).cast(pl.Utf8).alias("condition") if condition_col else pl.lit(None).alias("condition"),
        pl.col("review_text_clean").alias("review_text"),
        pl.col(rating_col).cast(pl.Int32).alias("rating") if rating_col else pl.lit(None).cast(pl.Int32).alias("rating"),
        pl.col("word_count").cast(pl.Int32),
        pl.col("source"),
        pl.col("has_ae_mention"),
    ])

    # Deduplicate by review_id (same review text → same ID)
    before_dedup = insert_df.height
    insert_df = insert_df.unique(subset=["review_id"], keep="first")
    logger.info(f"  Deduplication: {before_dedup:,} → {insert_df.height:,}")

    with get_duckdb_connection() as conn:
        # Create table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS drug_reviews (
                review_id       VARCHAR PRIMARY KEY,
                drug_name       VARCHAR,
                drug_norm       VARCHAR,
                condition       VARCHAR,
                review_text     VARCHAR,
                rating          INTEGER,
                word_count      INTEGER,
                source          VARCHAR,
                has_ae_mention  BOOLEAN,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Clear and re-insert (idempotent)
        conn.execute("DELETE FROM drug_reviews")

        # Register and insert
        conn.register("reviews_insert", insert_df.to_arrow())
        conn.execute("""
            INSERT INTO drug_reviews
                (review_id, drug_name, drug_norm, condition,
                 review_text, rating, word_count, source, has_ae_mention)
            SELECT review_id, drug_name, drug_norm, condition,
                   review_text, rating, word_count, source, has_ae_mention
            FROM reviews_insert
        """)

        count = conn.execute("SELECT COUNT(*) FROM drug_reviews").fetchone()[0]
        ae_count = conn.execute(
            "SELECT COUNT(*) FROM drug_reviews WHERE has_ae_mention"
        ).fetchone()[0]

    logger.info(f"  → Inserted {count:,} reviews ({ae_count:,} with AE mentions)")
    return count


# ---------------------------------------------------------------------------
# Step E: scispaCy NER baseline
# ---------------------------------------------------------------------------

def run_scispacy_ner(n_reviews: int = 500, batch_size: int = 50) -> int:
    """Run scispaCy NER on first N AE-mention reviews.

    Args:
        n_reviews: Number of reviews to process.
        batch_size: spaCy pipe batch size.

    Returns:
        Number of entities extracted.
    """
    logger.info(f"Step E: Running scispaCy NER on {n_reviews} reviews...")

    try:
        import spacy
        nlp = spacy.load("en_ner_bc5cdr_md")
        logger.info(f"  Model loaded: en_ner_bc5cdr_md")
    except OSError as e:
        logger.error(f"  scispaCy model not available: {e}")
        logger.error("  Install with: pip install scispacy && pip install <model_url>")
        return 0

    # Load reviews ordered by review_id (deterministic)
    with get_duckdb_connection(read_only=True) as conn:
        reviews = conn.execute("""
            SELECT review_id, review_text
            FROM drug_reviews
            WHERE has_ae_mention = TRUE
            ORDER BY review_id
            LIMIT ?
        """, [n_reviews]).fetchall()

    logger.info(f"  → {len(reviews)} AE-mention reviews loaded")

    if not reviews:
        logger.warning("  No AE-mention reviews found — skipping NER")
        return 0

    # Run NER
    review_ids = [r[0] for r in reviews]
    texts = [r[1] for r in reviews]

    entities: list[dict[str, Any]] = []
    processed = 0

    for doc, review_id in zip(nlp.pipe(texts, batch_size=batch_size), review_ids):
        for ent in doc.ents:
            entities.append({
                "review_id": review_id,
                "entity_text": ent.text,
                "entity_label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char,
            })
        processed += 1

        if processed % 100 == 0:
            logger.info(f"  Progress: {processed}/{len(reviews)} reviews, {len(entities)} entities")
            gc.collect()

    logger.info(f"  → {len(entities)} entities from {processed} reviews")

    # Store in DuckDB
    if entities:
        ner_df = pl.DataFrame(entities)

        with get_duckdb_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scispacy_ner_results (
                    review_id       VARCHAR,
                    entity_text     VARCHAR,
                    entity_label    VARCHAR,
                    start_char      INTEGER,
                    end_char        INTEGER,
                    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("DELETE FROM scispacy_ner_results")
            conn.register("ner_insert", ner_df.to_arrow())
            conn.execute("""
                INSERT INTO scispacy_ner_results
                    (review_id, entity_text, entity_label, start_char, end_char)
                SELECT review_id, entity_text, entity_label, start_char, end_char
                FROM ner_insert
            """)

            count = conn.execute("SELECT COUNT(*) FROM scispacy_ner_results").fetchone()[0]

        logger.info(f"  → Stored {count:,} NER entities in scispacy_ner_results")
    else:
        logger.warning("  No entities extracted")

    gc.collect()
    return len(entities)


# ---------------------------------------------------------------------------
# Step F: Manual labeling template
# ---------------------------------------------------------------------------

def generate_labeling_template(n_rows: int = 50) -> Path:
    """Generate CSV template for manual labeling.

    Args:
        n_rows: Number of reviews to include.

    Returns:
        Path to generated CSV file.
    """
    logger.info(f"Step F: Generating labeling template ({n_rows} rows)...")

    with get_duckdb_connection(read_only=True) as conn:
        rows = conn.execute("""
            SELECT review_id, drug_norm,
                   SUBSTRING(review_text, 1, 200) as review_text_preview
            FROM drug_reviews
            WHERE has_ae_mention = TRUE
            ORDER BY review_id
            LIMIT ?
        """, [n_rows]).fetchall()

    # Build template DataFrame
    records = []
    for row in rows:
        records.append({
            "review_id": row[0],
            "drug_norm": row[1],
            "review_text_preview": row[2],
            "drugs_gt": "",
            "reactions_gt": "",
            "severity_gt": "",
            "notes": "",
        })

    template_df = pl.DataFrame(records)

    GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    output_path = GROUND_TRUTH_DIR / "labeling_template.csv"
    template_df.write_csv(output_path)

    logger.info(f"  → Wrote {template_df.height} rows to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def run_verification_queries() -> None:
    """Run Day 5 verification queries."""
    with get_duckdb_connection(read_only=True) as conn:
        logger.info("=" * 60)
        logger.info("Day 5 Verification")
        logger.info("=" * 60)

        # 1. Review distribution
        logger.info("\n1. Review distribution by drug:")
        rows = conn.execute("""
            SELECT drug_norm, COUNT(*) as total_reviews,
                   SUM(CASE WHEN has_ae_mention THEN 1 ELSE 0 END) as ae_reviews
            FROM drug_reviews
            GROUP BY drug_norm
            ORDER BY total_reviews DESC
        """).fetchall()
        logger.info(f"   {'Drug':<15} {'Total':>8} {'AE':>8}")
        for r in rows:
            logger.info(f"   {r[0]:<15} {r[1]:>8,} {r[2]:>8,}")

        # 2. NER coverage
        logger.info("\n2. NER coverage:")
        try:
            rows = conn.execute("""
                SELECT entity_label,
                       COUNT(*) as total_entities,
                       COUNT(DISTINCT review_id) as reviews_covered
                FROM scispacy_ner_results
                GROUP BY entity_label
            """).fetchall()
            for r in rows:
                logger.info(f"   {r[0]:<12} {r[1]:>6,} entities, {r[2]:>4,} reviews")
        except Exception:
            logger.info("   (scispacy_ner_results not yet populated)")

        # 3. Sample NER output
        logger.info("\n3. Sample NER output:")
        try:
            rows = conn.execute("""
                SELECT r.drug_norm,
                       SUBSTRING(r.review_text, 1, 80) as text_preview,
                       n.entity_text, n.entity_label
                FROM drug_reviews r
                JOIN scispacy_ner_results n ON r.review_id = n.review_id
                LIMIT 5
            """).fetchall()
            for r in rows:
                logger.info(
                    f"   [{r[0]}] \"{r[1]}...\" → {r[2]} ({r[3]})"
                )
        except Exception:
            logger.info("   (no NER results to show)")

    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_day5_pipeline() -> None:
    """Run the complete Day 5 pipeline."""
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("AET-SE — Day 5: Drug Reviews Ingestion + scispaCy NER")
    logger.info("=" * 60)

    # Step A: Load
    df = load_review_files()

    # Step B: Filter to target drugs
    df = filter_target_drugs(df)

    # Step C: Text quality filter
    df = apply_text_filters(df)

    # Step D: Store in DuckDB
    store_reviews_in_duckdb(df)

    # Free memory before NER
    del df
    gc.collect()

    # Step E: scispaCy NER
    run_scispacy_ner(n_reviews=500, batch_size=50)

    # Step F: Labeling template
    generate_labeling_template(n_rows=50)

    # Verify
    run_verification_queries()

    elapsed = time.time() - start_time
    logger.info(f"\nDay 5 pipeline completed in {elapsed:.1f}s")
    logger.info("✅ Day 5 — Drug Reviews Ingestion + scispaCy NER complete.")


if __name__ == "__main__":
    run_day5_pipeline()
