"""RxNorm RxTerms ingestion for drug name normalization.

This module handles:
- Auto-downloading the RxTerms ZIP from NIH if not already present
- Extracting and parsing the TSV file into a Polars DataFrame
- Building a generic drug name lookup (brand → generic)
- Caching the parsed lookup as Parquet for fast re-loads
- Providing the lookup dict: {display_name_upper: (rxcui, generic_name)}

Usage:
    python -m aetse.data.ingest_rxnorm

All file I/O uses Polars (not Pandas) per project rules.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import polars as pl

from aetse.config.settings import settings
from aetse.utils.logging import logger


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Primary download URL — update in .env if this version 404s
RXTERMS_URL: str = "https://data.nlm.nih.gov/umls/kss/rxterms/RxTerms202406.zip"

# Fallback: user can override via environment variable
RXTERMS_URL_ENV_KEY: str = "RXTERMS_URL"

# Where raw downloads go
RXNORM_RAW_DIR: Path = settings.data.rxnorm_data_dir

# Cached parquet output
RXNORM_LOOKUP_PATH: Path = settings.project_root / "data" / "processed" / "rxnorm_lookup.parquet"

# Expected TSV filename inside the ZIP
RXTERMS_TSV_PATTERN: str = "RxTerms"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _get_download_url() -> str:
    """Get the RxTerms download URL, checking .env override first."""
    import os
    return os.environ.get(RXTERMS_URL_ENV_KEY, RXTERMS_URL)


def download_rxterms(target_dir: Path | None = None) -> Path:
    """Download the RxTerms ZIP from NIH if not already present.

    Args:
        target_dir: Directory to save the ZIP file. Defaults to data/raw/rxnorm/

    Returns:
        Path to the downloaded (or existing) ZIP file.

    Raises:
        RuntimeError: If download fails after retries.
    """
    import httpx

    if target_dir is None:
        target_dir = RXNORM_RAW_DIR

    target_dir.mkdir(parents=True, exist_ok=True)

    url = _get_download_url()
    filename = url.split("/")[-1]
    zip_path = target_dir / filename

    # Check for any existing RxTerms ZIP
    existing_zips = list(target_dir.glob("RxTerms*.zip"))
    if existing_zips:
        logger.info(f"RxTerms ZIP already exists: {existing_zips[0]}")
        return existing_zips[0]

    logger.info(f"Downloading RxTerms from {url}")
    logger.info(f"  → Target: {zip_path}")

    try:
        with httpx.Client(timeout=120, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()

            zip_path.write_bytes(response.content)
            logger.info(
                f"  → Downloaded {len(response.content) / 1024 / 1024:.1f} MB"
            )
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            logger.error(
                f"RxTerms URL returned 404: {url}\n"
                f"  Check for current version at:\n"
                f"  https://lhncbc.nlm.nih.gov/RxNav/applications/RxTermsRelease.html\n"
                f"  Then set RXTERMS_URL in your .env file."
            )
        raise RuntimeError(f"Failed to download RxTerms: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to download RxTerms: {e}") from e

    return zip_path


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------

def parse_rxterms_zip(zip_path: Path) -> pl.DataFrame:
    """Extract and parse the RxTerms TSV file from a ZIP archive.

    The RxTerms file contains columns like:
    RXCUI, GENERIC_RXCUI, TTY, FULL_NAME, RXN_DOSE_FORM,
    FULL_GENERIC_NAME, BRAND_NAME, DISPLAY_NAME, ROUTE, etc.

    Args:
        zip_path: Path to the RxTerms ZIP file.

    Returns:
        Polars DataFrame with parsed RxTerms data.
    """
    logger.info(f"Parsing RxTerms ZIP: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Find the TSV/txt file inside the ZIP
        tsv_files = [
            f for f in zf.namelist()
            if RXTERMS_TSV_PATTERN in f and (f.endswith(".txt") or f.endswith(".tsv"))
        ]

        if not tsv_files:
            # Try any file that looks like data
            tsv_files = [
                f for f in zf.namelist()
                if not f.endswith("/") and not f.startswith("__MACOSX")
            ]

        if not tsv_files:
            raise FileNotFoundError(f"No data file found in {zip_path}")

        tsv_name = tsv_files[0]
        logger.info(f"  → Reading: {tsv_name}")

        with zf.open(tsv_name) as f:
            raw_bytes = f.read()

    # Parse as pipe-delimited (RxTerms uses | as separator)
    # Try pipe first, then tab
    text = raw_bytes.decode("utf-8", errors="replace")

    # Detect separator from first line
    first_line = text.split("\n")[0]
    if "|" in first_line:
        separator = "|"
    elif "\t" in first_line:
        separator = "\t"
    else:
        separator = "|"  # default

    logger.info(f"  → Detected separator: {'pipe' if separator == '|' else 'tab'}")

    df = pl.read_csv(
        io.StringIO(text),
        separator=separator,
        has_header=True,
        infer_schema_length=5000,
        ignore_errors=True,
        truncate_ragged_lines=True,
    )

    # Normalize column names
    df = df.rename({col: col.strip().upper() for col in df.columns})

    logger.info(f"  → {df.height:,} rows, columns: {df.columns}")
    return df


# ---------------------------------------------------------------------------
# Build lookup
# ---------------------------------------------------------------------------

def build_lookup(df: pl.DataFrame) -> pl.DataFrame:
    """Build a normalized lookup table from parsed RxTerms data.

    Creates a mapping of display_name → (rxcui, generic_name) for
    all drugs in the RxTerms dataset.

    Args:
        df: Parsed RxTerms DataFrame.

    Returns:
        Polars DataFrame with columns: [display_name, rxcui, generic_name]
    """
    # Identify key columns (names vary slightly between RxTerms versions)
    col_map: dict[str, str | None] = {
        "rxcui": None,
        "generic_name": None,
        "display_name": None,
        "brand_name": None,
    }

    for col in df.columns:
        col_upper = col.upper()
        if col_upper in ("RXCUI", "GENERIC_RXCUI"):
            if col_map["rxcui"] is None:
                col_map["rxcui"] = col
        if col_upper in ("FULL_GENERIC_NAME", "GENERIC_NAME"):
            col_map["generic_name"] = col
        if col_upper in ("DISPLAY_NAME", "FULL_NAME"):
            if col_map["display_name"] is None:
                col_map["display_name"] = col
        if col_upper in ("BRAND_NAME",):
            col_map["brand_name"] = col

    logger.info(f"  Column mapping: {col_map}")

    # We need at minimum rxcui and a name column
    if col_map["rxcui"] is None:
        raise ValueError(f"Could not find RXCUI column in RxTerms. Columns: {df.columns}")

    # Build display names from all available name columns
    name_cols = [
        v for k, v in col_map.items()
        if k in ("display_name", "generic_name", "brand_name") and v is not None
    ]

    if not name_cols:
        raise ValueError(f"No name columns found in RxTerms. Columns: {df.columns}")

    # For each name column, create entries in the lookup
    lookup_frames: list[pl.DataFrame] = []

    rxcui_col = col_map["rxcui"]
    generic_col = col_map["generic_name"]

    for name_col in name_cols:
        frame = df.select([
            pl.col(name_col).cast(pl.Utf8).str.strip_chars().str.to_uppercase().alias("display_name"),
            pl.col(rxcui_col).cast(pl.Utf8).alias("rxcui"),
            (
                pl.col(generic_col).cast(pl.Utf8).str.strip_chars().str.to_lowercase()
                if generic_col else pl.col(name_col).cast(pl.Utf8).str.strip_chars().str.to_lowercase()
            ).alias("generic_name"),
        ]).filter(
            pl.col("display_name").is_not_null() & (pl.col("display_name") != "")
        )
        lookup_frames.append(frame)

    lookup = pl.concat(lookup_frames, how="vertical_relaxed")

    # Deduplicate on display_name, keeping first
    lookup = lookup.unique(subset=["display_name"], keep="first")

    logger.info(f"  → Lookup table: {lookup.height:,} unique entries")
    return lookup


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def save_lookup_parquet(lookup: pl.DataFrame, path: Path | None = None) -> Path:
    """Cache the lookup table as Parquet for fast re-loads.

    Args:
        lookup: Lookup DataFrame to save.
        path: Output path. Defaults to data/processed/rxnorm_lookup.parquet

    Returns:
        Path to the saved Parquet file.
    """
    if path is None:
        path = RXNORM_LOOKUP_PATH

    path.parent.mkdir(parents=True, exist_ok=True)
    lookup.write_parquet(path)
    logger.info(f"  → Saved lookup to {path} ({path.stat().st_size / 1024:.0f} KB)")
    return path


def load_lookup_parquet(path: Path | None = None) -> pl.DataFrame | None:
    """Load the cached lookup table if it exists.

    Args:
        path: Path to the Parquet file. Defaults to data/processed/rxnorm_lookup.parquet

    Returns:
        Polars DataFrame or None if cache doesn't exist.
    """
    if path is None:
        path = RXNORM_LOOKUP_PATH

    if path.exists():
        lookup = pl.read_parquet(path)
        logger.info(f"Loaded cached RxNorm lookup: {lookup.height:,} entries from {path}")
        return lookup

    return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_rxnorm_ingestion() -> Path:
    """Run the full RxNorm ingestion pipeline.

    Steps:
    1. Check for cached parquet — if exists, return early
    2. Download RxTerms ZIP if not present
    3. Parse the TSV file
    4. Build lookup table
    5. Cache as Parquet

    Returns:
        Path to the lookup Parquet file.
    """
    logger.info("=" * 60)
    logger.info("AET-SE — RxNorm RxTerms Ingestion")
    logger.info("=" * 60)

    # Check cache first
    cached = load_lookup_parquet()
    if cached is not None:
        logger.info("Using cached RxNorm lookup — skipping download")
        return RXNORM_LOOKUP_PATH

    # Download
    zip_path = download_rxterms()

    # Parse
    df = parse_rxterms_zip(zip_path)

    # Build lookup
    lookup = build_lookup(df)

    # Cache
    output_path = save_lookup_parquet(lookup)

    logger.info("✅ RxNorm ingestion complete")
    return output_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_rxnorm_ingestion()
