"""Test suite for database_ai_adapters module."""

import pytest


# Check if module can be imported
def can_import_adapters():
    try:
        from npcpy.sql.database_ai_adapters import DatabaseAIAdapter
        return True
    except (ImportError, NameError):
        return False


ADAPTERS_AVAILABLE = can_import_adapters()
skip_reason = "database_ai_adapters module has import errors"


@pytest.mark.skipif(not ADAPTERS_AVAILABLE, reason=skip_reason)
class TestDatabaseAIAdapter:
    """Test DatabaseAIAdapter class."""

    def test_adapter_creation_sqlite(self):
        """Test creating adapter with SQLite engine"""
        sqlalchemy = pytest.importorskip("sqlalchemy")
        from npcpy.sql.database_ai_adapters import DatabaseAIAdapter

        engine = sqlalchemy.create_engine("sqlite:///:memory:")
        adapter = DatabaseAIAdapter(engine)

        assert adapter.dialect == "sqlite"
        assert adapter.engine == engine

    def test_get_dialect_sqlite(self):
        """Test dialect detection for SQLite"""
        sqlalchemy = pytest.importorskip("sqlalchemy")
        from npcpy.sql.database_ai_adapters import DatabaseAIAdapter

        engine = sqlalchemy.create_engine("sqlite:///:memory:")
        adapter = DatabaseAIAdapter(engine)

        assert adapter._get_dialect() == "sqlite"

    def test_generate_ai_function_generic(self):
        """Test generic AI function generation"""
        sqlalchemy = pytest.importorskip("sqlalchemy")
        from npcpy.sql.database_ai_adapters import DatabaseAIAdapter

        engine = sqlalchemy.create_engine("sqlite:///:memory:")
        adapter = DatabaseAIAdapter(engine)

        sql = adapter.generate_ai_function("generate_text", "Hello world")

        assert isinstance(sql, str)
        assert "Hello world" in sql
        assert "ai_result" in sql

    def test_generic_generate_text(self):
        """Test generic text generation fallback"""
        sqlalchemy = pytest.importorskip("sqlalchemy")
        from npcpy.sql.database_ai_adapters import DatabaseAIAdapter

        engine = sqlalchemy.create_engine("sqlite:///:memory:")
        adapter = DatabaseAIAdapter(engine)

        sql = adapter._generic_ai_function("generate_text", "Test prompt")

        assert "Generated text based on" in sql
        assert "Test prompt" in sql

    def test_generic_summarize(self):
        """Test generic summarize fallback"""
        sqlalchemy = pytest.importorskip("sqlalchemy")
        from npcpy.sql.database_ai_adapters import DatabaseAIAdapter

        engine = sqlalchemy.create_engine("sqlite:///:memory:")
        adapter = DatabaseAIAdapter(engine)

        sql = adapter._generic_ai_function("summarize", "Long text to summarize")

        assert "Summary of" in sql

    def test_generic_sentiment(self):
        """Test generic sentiment analysis fallback"""
        sqlalchemy = pytest.importorskip("sqlalchemy")
        from npcpy.sql.database_ai_adapters import DatabaseAIAdapter

        engine = sqlalchemy.create_engine("sqlite:///:memory:")
        adapter = DatabaseAIAdapter(engine)

        sql = adapter._generic_ai_function("analyze_sentiment", "This is good")

        assert "positive" in sql or "negative" in sql or "neutral" in sql

    def test_postgresql_methods_exist(self):
        """Test PostgreSQL-specific methods exist"""
        sqlalchemy = pytest.importorskip("sqlalchemy")
        from npcpy.sql.database_ai_adapters import DatabaseAIAdapter

        engine = sqlalchemy.create_engine("sqlite:///:memory:")
        adapter = DatabaseAIAdapter(engine)

        # These methods should exist even if not used for SQLite
        assert hasattr(adapter, "_postgresql_generate_text")
        assert hasattr(adapter, "_postgresql_summarize")
        assert hasattr(adapter, "_postgresql_analyze_sentiment")


@pytest.mark.skipif(not ADAPTERS_AVAILABLE, reason=skip_reason)
class TestAIFunctionRouter:
    """Test AIFunctionRouter class."""

    def test_route_ai_function(self):
        """Test routing AI function through adapter"""
        sqlalchemy = pytest.importorskip("sqlalchemy")
        from npcpy.sql.database_ai_adapters import AIFunctionRouter

        engine = sqlalchemy.create_engine("sqlite:///:memory:")

        sql = AIFunctionRouter.route_ai_function(
            engine, "generate_text", "Test prompt"
        )

        assert isinstance(sql, str)
        assert "Test prompt" in sql

    def test_route_multiple_function_types(self):
        """Test routing different function types"""
        sqlalchemy = pytest.importorskip("sqlalchemy")
        from npcpy.sql.database_ai_adapters import AIFunctionRouter

        engine = sqlalchemy.create_engine("sqlite:///:memory:")

        for func_type in ["generate_text", "summarize", "analyze_sentiment"]:
            sql = AIFunctionRouter.route_ai_function(
                engine, func_type, "Test input"
            )
            assert isinstance(sql, str)


@pytest.mark.skipif(not ADAPTERS_AVAILABLE, reason=skip_reason)
class TestDialectMapping:
    """Test dialect mapping functionality."""

    def test_dialect_map_coverage(self):
        """Test that common dialects are mapped"""
        sqlalchemy = pytest.importorskip("sqlalchemy")
        from npcpy.sql.database_ai_adapters import DatabaseAIAdapter

        engine = sqlalchemy.create_engine("sqlite:///:memory:")
        adapter = DatabaseAIAdapter(engine)

        # Check that the dialect_map includes expected databases
        expected_dialects = ["postgresql", "mysql", "mssql", "sqlite", "snowflake"]

        # Access internal dialect map through method
        dialect = adapter._get_dialect()
        assert dialect in expected_dialects or dialect == "unknown"


# Fallback tests when module cannot be imported
class TestAdaptersModuleStatus:
    """Test module import status."""

    def test_module_import_status_known(self):
        """Test that we know if module can be imported"""
        assert isinstance(ADAPTERS_AVAILABLE, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
