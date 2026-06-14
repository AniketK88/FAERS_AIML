"""Unit tests for configuration and settings."""

from pathlib import Path

from aetse.config.settings import Settings, PROJECT_ROOT


class TestSettings:
    """Tests for the Settings configuration."""

    def test_project_root_exists(self) -> None:
        """Test that PROJECT_ROOT points to a valid directory."""
        assert PROJECT_ROOT.exists()
        assert PROJECT_ROOT.is_dir()

    def test_default_settings_load(self) -> None:
        """Test that Settings can be instantiated with defaults."""
        s = Settings()
        assert s.ollama.primary_model == "llama3.1:8b-instruct-q4_K_M"
        assert s.ollama.max_retries == 3
        assert s.processing.confidence_threshold == 0.75

    def test_duckdb_path(self) -> None:
        """Test that DuckDB path resolves correctly."""
        s = Settings()
        assert str(s.duckdb.path).endswith("faers.duckdb")
        assert isinstance(s.duckdb.path, Path)

    def test_ensure_directories(self, tmp_path: Path) -> None:
        """Test that ensure_directories creates required dirs."""
        s = Settings()
        # Just test that the method doesn't raise
        s.ensure_directories()
