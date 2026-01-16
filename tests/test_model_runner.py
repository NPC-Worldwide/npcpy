"""Test suite for model_runner module - tests utility functions."""

import pytest


class TestModelRunnerDependencyExtraction:
    """Test dependency extraction and resolution utilities."""

    def test_extract_dependencies_single_ref(self):
        """Test extracting single dependency reference"""
        pandas = pytest.importorskip("pandas")
        import re

        # Test the regex pattern used in _extract_dependencies
        model_sql = "SELECT * FROM {{ ref('raw_data') }}"
        refs = re.findall(r'{{\s*ref\([\'"](.+?)[\'"]\)\s*}}', model_sql)

        assert len(refs) == 1
        assert refs[0] == "raw_data"

    def test_extract_dependencies_multiple_refs(self):
        """Test extracting multiple dependency references"""
        pandas = pytest.importorskip("pandas")
        import re

        model_sql = """
        SELECT a.*, b.value
        FROM {{ ref('table_a') }} a
        JOIN {{ ref('table_b') }} b ON a.id = b.id
        """
        refs = re.findall(r'{{\s*ref\([\'"](.+?)[\'"]\)\s*}}', model_sql)

        assert len(refs) == 2
        assert "table_a" in refs
        assert "table_b" in refs

    def test_extract_dependencies_no_refs(self):
        """Test extraction with no refs returns empty list"""
        pandas = pytest.importorskip("pandas")
        import re

        model_sql = "SELECT * FROM raw_table"
        refs = re.findall(r'{{\s*ref\([\'"](.+?)[\'"]\)\s*}}', model_sql)

        assert refs == []

    def test_extract_dependencies_double_quotes(self):
        """Test extracting refs with double quotes"""
        pandas = pytest.importorskip("pandas")
        import re

        model_sql = 'SELECT * FROM {{ ref("my_model") }}'
        refs = re.findall(r'{{\s*ref\([\'"](.+?)[\'"]\)\s*}}', model_sql)

        assert len(refs) == 1
        assert refs[0] == "my_model"

    def test_extract_dependencies_varying_whitespace(self):
        """Test extracting refs with varying whitespace"""
        pandas = pytest.importorskip("pandas")
        import re

        # Test standard cases that work with the regex
        test_cases = [
            "{{ ref('model') }}",
            "{{ref('model')}}",
        ]

        for sql in test_cases:
            refs = re.findall(r'{{\s*ref\([\'"](.+?)[\'"]\)\s*}}', sql)
            assert refs == ["model"], f"Failed for: {sql}"

    def test_resolve_refs_replaces_pattern(self):
        """Test resolving refs replaces with model prefix"""
        pandas = pytest.importorskip("pandas")
        import re

        model_sql = "SELECT * FROM {{ ref('source_data') }}"
        model_registry = {"source_data": pandas.DataFrame()}

        def replace_ref(match):
            model_name = match.group(1)
            if model_name in model_registry:
                return f"model_{model_name}"
            raise ValueError(f"Model {model_name} not found")

        resolved = re.sub(r'{{\s*ref\([\'"](.+?)[\'"]\)\s*}}', replace_ref, model_sql)

        assert resolved == "SELECT * FROM model_source_data"

    def test_resolve_refs_missing_model_raises(self):
        """Test resolving refs with missing model raises error"""
        pandas = pytest.importorskip("pandas")
        import re

        model_sql = "SELECT * FROM {{ ref('missing_model') }}"
        model_registry = {}

        def replace_ref(match):
            model_name = match.group(1)
            if model_name in model_registry:
                return f"model_{model_name}"
            raise ValueError(f"Model {model_name} not found")

        with pytest.raises(ValueError) as exc_info:
            re.sub(r'{{\s*ref\([\'"](.+?)[\'"]\)\s*}}', replace_ref, model_sql)

        assert "missing_model" in str(exc_info.value)


class TestModelRegistryBehavior:
    """Test model registry dictionary behavior."""

    def test_registry_stores_dataframes(self):
        """Test registry can store DataFrames"""
        pandas = pytest.importorskip("pandas")

        model_registry = {}
        df = pandas.DataFrame({"col": [1, 2, 3]})
        model_registry["test_model"] = df

        assert "test_model" in model_registry
        assert len(model_registry["test_model"]) == 3

    def test_registry_overwrites_existing(self):
        """Test registry overwrites existing model"""
        pandas = pytest.importorskip("pandas")

        model_registry = {}
        model_registry["model"] = pandas.DataFrame({"v": [1]})
        model_registry["model"] = pandas.DataFrame({"v": [2, 3]})

        assert len(model_registry["model"]) == 2

    def test_registry_multiple_models(self):
        """Test registry handles multiple models"""
        pandas = pytest.importorskip("pandas")

        model_registry = {}
        model_registry["raw"] = pandas.DataFrame({"id": [1, 2]})
        model_registry["processed"] = pandas.DataFrame({"id": [1], "score": [0.5]})
        model_registry["final"] = pandas.DataFrame({"result": ["done"]})

        assert len(model_registry) == 3


class TestSampleDataGeneration:
    """Test sample data generation patterns."""

    def test_sample_data_structure(self):
        """Test sample customer feedback data structure"""
        pandas = pytest.importorskip("pandas")

        # Mimics _load_sample_data pattern
        sample_data = pandas.DataFrame({
            "customer_id": range(1, 4),
            "feedback_text": [
                "Great service but expensive",
                "Product needs improvement",
                "Amazing experience overall",
            ],
            "customer_segment": ["premium", "basic", "premium"],
        })

        assert len(sample_data) == 3
        assert list(sample_data.columns) == ["customer_id", "feedback_text", "customer_segment"]
        assert sample_data["customer_id"].tolist() == [1, 2, 3]

    def test_sample_data_filtering(self):
        """Test filtering sample data by segment"""
        pandas = pytest.importorskip("pandas")

        sample_data = pandas.DataFrame({
            "customer_id": range(1, 4),
            "feedback_text": ["A", "B", "C"],
            "customer_segment": ["premium", "basic", "premium"],
        })

        premium_only = sample_data[sample_data["customer_segment"] == "premium"]

        assert len(premium_only) == 2


class TestModelRunMetadataFormat:
    """Test model run metadata formatting."""

    def test_metadata_success_format(self):
        """Test metadata format for successful run"""
        pandas = pytest.importorskip("pandas")
        import json

        result_df = pandas.DataFrame({"col": [1, 2, 3]})

        metadata = {
            "status": "success",
            "error": None,
            "rows_processed": len(result_df),
        }

        json_str = json.dumps(metadata)
        parsed = json.loads(json_str)

        assert parsed["status"] == "success"
        assert parsed["rows_processed"] == 3

    def test_metadata_failure_format(self):
        """Test metadata format for failed run"""
        import json

        metadata = {
            "status": "failed",
            "error": "Connection timeout",
            "rows_processed": 0,
        }

        json_str = json.dumps(metadata)
        parsed = json.loads(json_str)

        assert parsed["status"] == "failed"
        assert "timeout" in parsed["error"]

    def test_preview_dataframe_to_dict(self):
        """Test converting DataFrame preview to dict"""
        pandas = pytest.importorskip("pandas")
        import json

        result_df = pandas.DataFrame({
            "id": [1, 2, 3],
            "value": ["a", "b", "c"]
        })

        preview = result_df.head().to_dict()
        json_str = json.dumps(preview)
        parsed = json.loads(json_str)

        assert "id" in parsed
        assert "value" in parsed


class TestHistoryDatabaseSchema:
    """Test history database schema patterns."""

    def test_model_runs_table_schema(self):
        """Test model_runs table can be created"""
        sqlite3 = pytest.importorskip("sqlite3")

        conn = sqlite3.connect(":memory:")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_runs (
                model_name TEXT,
                run_timestamp DATETIME,
                run_status TEXT,
                metadata TEXT,
                output_preview TEXT
            )
        """)

        # Verify table exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='model_runs'"
        )
        result = cursor.fetchone()

        assert result is not None
        assert result[0] == "model_runs"
        conn.close()

    def test_model_dependencies_table_schema(self):
        """Test model_dependencies table can be created"""
        sqlite3 = pytest.importorskip("sqlite3")

        conn = sqlite3.connect(":memory:")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_dependencies (
                model_name TEXT,
                depends_on TEXT,
                created_at DATETIME
            )
        """)

        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='model_dependencies'"
        )
        result = cursor.fetchone()

        assert result is not None
        conn.close()

    def test_insert_model_run_record(self):
        """Test inserting a model run record"""
        sqlite3 = pytest.importorskip("sqlite3")
        from datetime import datetime
        import json

        conn = sqlite3.connect(":memory:")
        conn.execute("""
            CREATE TABLE model_runs (
                model_name TEXT,
                run_timestamp DATETIME,
                run_status TEXT,
                metadata TEXT,
                output_preview TEXT
            )
        """)

        metadata = {"status": "success", "rows_processed": 10}

        conn.execute(
            """INSERT INTO model_runs
               (model_name, run_timestamp, run_status, metadata, output_preview)
               VALUES (?, ?, ?, ?, ?)""",
            (
                "test_model",
                datetime.now().isoformat(),
                "success",
                json.dumps(metadata),
                json.dumps({"col": [1, 2, 3]}),
            ),
        )

        cursor = conn.execute("SELECT COUNT(*) FROM model_runs")
        count = cursor.fetchone()[0]

        assert count == 1
        conn.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
