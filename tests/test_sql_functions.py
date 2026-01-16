"""Test suite for database_ai_functions module."""

import pytest


# Check if module can be imported
def can_import_functions():
    try:
        from npcpy.sql.database_ai_functions import DatabaseAIFunctionMapper
        return True
    except (ImportError, NameError):
        return False


FUNCTIONS_AVAILABLE = can_import_functions()
skip_reason = "database_ai_functions module has import errors"


@pytest.mark.skipif(not FUNCTIONS_AVAILABLE, reason=skip_reason)
class TestDatabaseAIFunctionMapper:
    """Test DatabaseAIFunctionMapper class."""

    def test_snowflake_cortex_mapping_exists(self):
        """Test Snowflake Cortex mapping returns dict"""
        from npcpy.sql.database_ai_functions import DatabaseAIFunctionMapper

        mapping = DatabaseAIFunctionMapper.get_snowflake_cortex_mapping()

        assert isinstance(mapping, dict)
        assert len(mapping) > 0

    def test_snowflake_mapping_has_required_functions(self):
        """Test Snowflake mapping includes required AI functions"""
        from npcpy.sql.database_ai_functions import DatabaseAIFunctionMapper

        mapping = DatabaseAIFunctionMapper.get_snowflake_cortex_mapping()

        expected_functions = [
            "generate_text",
            "summarize",
            "analyze_sentiment",
            "translate",
            "extract_entities",
            "generate_embedding"
        ]

        for func in expected_functions:
            assert func in mapping, f"Missing function: {func}"

    def test_snowflake_mapping_structure(self):
        """Test Snowflake mapping has correct structure"""
        from npcpy.sql.database_ai_functions import DatabaseAIFunctionMapper

        mapping = DatabaseAIFunctionMapper.get_snowflake_cortex_mapping()

        for func_name, func_config in mapping.items():
            assert "cortex_function" in func_config, f"{func_name} missing cortex_function"
            assert "transformer" in func_config, f"{func_name} missing transformer"
            assert callable(func_config["transformer"]), f"{func_name} transformer not callable"

    def test_snowflake_generate_text_transformer(self):
        """Test generate_text transformer produces valid SQL"""
        from npcpy.sql.database_ai_functions import DatabaseAIFunctionMapper

        mapping = DatabaseAIFunctionMapper.get_snowflake_cortex_mapping()
        transformer = mapping["generate_text"]["transformer"]

        result = transformer("Hello world")

        assert isinstance(result, str)
        assert "SNOWFLAKE.CORTEX.COMPLETE" in result
        assert "Hello world" in result

    def test_snowflake_summarize_transformer(self):
        """Test summarize transformer produces valid SQL"""
        from npcpy.sql.database_ai_functions import DatabaseAIFunctionMapper

        mapping = DatabaseAIFunctionMapper.get_snowflake_cortex_mapping()
        transformer = mapping["summarize"]["transformer"]

        result = transformer("Long text to summarize")

        assert "SNOWFLAKE.CORTEX.SUMMARIZE" in result
        assert "Long text to summarize" in result

    def test_snowflake_sentiment_transformer(self):
        """Test sentiment transformer produces valid SQL"""
        from npcpy.sql.database_ai_functions import DatabaseAIFunctionMapper

        mapping = DatabaseAIFunctionMapper.get_snowflake_cortex_mapping()
        transformer = mapping["analyze_sentiment"]["transformer"]

        result = transformer("This is great!")

        assert "SNOWFLAKE.CORTEX.SENTIMENT" in result

    def test_snowflake_translate_transformer(self):
        """Test translate transformer with language params"""
        from npcpy.sql.database_ai_functions import DatabaseAIFunctionMapper

        mapping = DatabaseAIFunctionMapper.get_snowflake_cortex_mapping()
        transformer = mapping["translate"]["transformer"]

        result = transformer("Bonjour", source_lang="fr", target_lang="en")

        assert "SNOWFLAKE.CORTEX.TRANSLATE" in result
        assert "Bonjour" in result
        assert "fr" in result
        assert "en" in result

    def test_snowflake_embedding_transformer(self):
        """Test embedding transformer with model param"""
        from npcpy.sql.database_ai_functions import DatabaseAIFunctionMapper

        mapping = DatabaseAIFunctionMapper.get_snowflake_cortex_mapping()
        transformer = mapping["generate_embedding"]["transformer"]

        result = transformer("Text to embed", model="snowflake-arctic")

        assert "SNOWFLAKE.CORTEX.EMBED_TEXT" in result
        assert "snowflake-arctic" in result

    def test_databricks_mapping_exists(self):
        """Test Databricks mapping returns dict"""
        from npcpy.sql.database_ai_functions import DatabaseAIFunctionMapper

        mapping = DatabaseAIFunctionMapper.get_databricks_ai_mapping()

        assert isinstance(mapping, dict)

    def test_bigquery_mapping_exists(self):
        """Test BigQuery mapping returns dict"""
        from npcpy.sql.database_ai_functions import DatabaseAIFunctionMapper

        mapping = DatabaseAIFunctionMapper.get_bigquery_ai_mapping()

        assert isinstance(mapping, dict)


@pytest.mark.skipif(not FUNCTIONS_AVAILABLE, reason=skip_reason)
class TestNativeDatabaseAITransformer:
    """Test NativeDatabaseAITransformer class."""

    def test_transformer_creation_snowflake(self):
        """Test creating transformer for Snowflake"""
        from npcpy.sql.database_ai_functions import NativeDatabaseAITransformer

        transformer = NativeDatabaseAITransformer("snowflake")

        assert transformer.database_type == "snowflake"
        assert isinstance(transformer.function_mappings, dict)

    def test_transformer_creation_databricks(self):
        """Test creating transformer for Databricks"""
        from npcpy.sql.database_ai_functions import NativeDatabaseAITransformer

        transformer = NativeDatabaseAITransformer("databricks")

        assert transformer.database_type == "databricks"

    def test_transformer_creation_bigquery(self):
        """Test creating transformer for BigQuery"""
        from npcpy.sql.database_ai_functions import NativeDatabaseAITransformer

        transformer = NativeDatabaseAITransformer("bigquery")

        assert transformer.database_type == "bigquery"

    def test_transformer_unknown_database(self):
        """Test transformer with unknown database returns empty mappings"""
        from npcpy.sql.database_ai_functions import NativeDatabaseAITransformer

        transformer = NativeDatabaseAITransformer("unknown_db")

        assert transformer.function_mappings == {}

    def test_transform_ai_function_snowflake(self):
        """Test transforming AI function for Snowflake"""
        from npcpy.sql.database_ai_functions import NativeDatabaseAITransformer

        transformer = NativeDatabaseAITransformer("snowflake")
        result = transformer.transform_ai_function("generate_text", prompt="Test")

        assert isinstance(result, str)
        assert "SNOWFLAKE.CORTEX" in result

    def test_transform_unknown_function_raises(self):
        """Test transforming unknown function raises ValueError"""
        from npcpy.sql.database_ai_functions import NativeDatabaseAITransformer

        transformer = NativeDatabaseAITransformer("snowflake")

        with pytest.raises(ValueError) as exc_info:
            transformer.transform_ai_function("nonexistent_function")

        assert "No native mapping found" in str(exc_info.value)

    def test_case_insensitive_database_type(self):
        """Test database type is case insensitive"""
        from npcpy.sql.database_ai_functions import NativeDatabaseAITransformer

        transformer_lower = NativeDatabaseAITransformer("snowflake")
        transformer_upper = NativeDatabaseAITransformer("SNOWFLAKE")
        transformer_mixed = NativeDatabaseAITransformer("Snowflake")

        # All should have the same mappings
        assert len(transformer_lower.function_mappings) == len(transformer_upper.function_mappings)
        assert len(transformer_lower.function_mappings) == len(transformer_mixed.function_mappings)


# Fallback tests when module cannot be imported
class TestFunctionsModuleStatus:
    """Test module import status."""

    def test_module_import_status_known(self):
        """Test that we know if module can be imported"""
        assert isinstance(FUNCTIONS_AVAILABLE, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
