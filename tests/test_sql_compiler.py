"""Test suite for sql_model_compiler module."""

import os
import tempfile
import pytest


class TestSQLModelCompiler:
    """Test SQLModelCompiler class."""

    def test_compiler_initialization(self):
        """Test SQLModelCompiler initializes correctly"""
        sqlalchemy = pytest.importorskip("sqlalchemy")
        pandas = pytest.importorskip("pandas")
        from npcpy.sql.sql_model_compiler import SQLModelCompiler

        with tempfile.TemporaryDirectory() as tmpdir:
            compiler = SQLModelCompiler(tmpdir, engine_type='sqlite')

            assert compiler.models_dir == tmpdir
            assert compiler.engine_type == 'sqlite'
            assert isinstance(compiler.models, dict)

    def test_compiler_engine_type_lowercase(self):
        """Test engine type is converted to lowercase"""
        sqlalchemy = pytest.importorskip("sqlalchemy")
        pandas = pytest.importorskip("pandas")
        from npcpy.sql.sql_model_compiler import SQLModelCompiler

        with tempfile.TemporaryDirectory() as tmpdir:
            compiler = SQLModelCompiler(tmpdir, engine_type='SNOWFLAKE')

            assert compiler.engine_type == 'snowflake'

    def test_discover_models_empty_dir(self):
        """Test model discovery with empty directory"""
        sqlalchemy = pytest.importorskip("sqlalchemy")
        pandas = pytest.importorskip("pandas")
        from npcpy.sql.sql_model_compiler import SQLModelCompiler

        with tempfile.TemporaryDirectory() as tmpdir:
            compiler = SQLModelCompiler(tmpdir)

            assert len(compiler.models) == 0

    def test_discover_models_with_sql_files(self):
        """Test model discovery finds SQL files"""
        sqlalchemy = pytest.importorskip("sqlalchemy")
        pandas = pytest.importorskip("pandas")
        from npcpy.sql.sql_model_compiler import SQLModelCompiler

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test SQL files
            sql1_path = os.path.join(tmpdir, "model1.sql")
            sql2_path = os.path.join(tmpdir, "model2.sql")

            with open(sql1_path, 'w') as f:
                f.write("SELECT 1 AS value")

            with open(sql2_path, 'w') as f:
                f.write("SELECT 2 AS value")

            compiler = SQLModelCompiler(tmpdir)

            assert len(compiler.models) == 2
            assert "model1" in compiler.models
            assert "model2" in compiler.models

    def test_discover_models_ignores_non_sql(self):
        """Test model discovery ignores non-SQL files"""
        sqlalchemy = pytest.importorskip("sqlalchemy")
        pandas = pytest.importorskip("pandas")
        from npcpy.sql.sql_model_compiler import SQLModelCompiler

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create SQL and non-SQL files
            with open(os.path.join(tmpdir, "model.sql"), 'w') as f:
                f.write("SELECT 1")

            with open(os.path.join(tmpdir, "readme.txt"), 'w') as f:
                f.write("Not SQL")

            with open(os.path.join(tmpdir, "config.yaml"), 'w') as f:
                f.write("key: value")

            compiler = SQLModelCompiler(tmpdir)

            assert len(compiler.models) == 1
            assert "model" in compiler.models

    def test_model_content_loaded(self):
        """Test model content is loaded correctly"""
        sqlalchemy = pytest.importorskip("sqlalchemy")
        pandas = pytest.importorskip("pandas")
        from npcpy.sql.sql_model_compiler import SQLModelCompiler

        with tempfile.TemporaryDirectory() as tmpdir:
            sql_content = "SELECT id, name FROM users WHERE active = 1"
            with open(os.path.join(tmpdir, "active_users.sql"), 'w') as f:
                f.write(sql_content)

            compiler = SQLModelCompiler(tmpdir)

            assert compiler.models["active_users"]["content"] == sql_content
            assert compiler.models["active_users"]["name"] == "active_users"

    def test_compile_model_sqlite(self):
        """Test model compilation for SQLite (no transformation)"""
        sqlalchemy = pytest.importorskip("sqlalchemy")
        pandas = pytest.importorskip("pandas")
        from npcpy.sql.sql_model_compiler import SQLModelCompiler

        with tempfile.TemporaryDirectory() as tmpdir:
            sql_content = "SELECT * FROM {{table}}"
            with open(os.path.join(tmpdir, "test.sql"), 'w') as f:
                f.write(sql_content)

            compiler = SQLModelCompiler(tmpdir, engine_type='sqlite')
            compiled = compiler._compile_model("test")

            # SQLite doesn't transform the content
            assert compiled == sql_content

    def test_compile_model_snowflake(self):
        """Test model compilation for Snowflake"""
        sqlalchemy = pytest.importorskip("sqlalchemy")
        pandas = pytest.importorskip("pandas")
        from npcpy.sql.sql_model_compiler import SQLModelCompiler

        with tempfile.TemporaryDirectory() as tmpdir:
            sql_content = "SELECT {{COMPLETE('prompt')}}"
            with open(os.path.join(tmpdir, "test.sql"), 'w') as f:
                f.write(sql_content)

            compiler = SQLModelCompiler(tmpdir, engine_type='snowflake')
            compiled = compiler._compile_model("test")

            assert "SNOWFLAKE." in compiled

    def test_compile_model_bigquery(self):
        """Test model compilation for BigQuery"""
        sqlalchemy = pytest.importorskip("sqlalchemy")
        pandas = pytest.importorskip("pandas")
        from npcpy.sql.sql_model_compiler import SQLModelCompiler

        with tempfile.TemporaryDirectory() as tmpdir:
            sql_content = "SELECT {{GENERATE_TEXT('prompt')}}"
            with open(os.path.join(tmpdir, "test.sql"), 'w') as f:
                f.write(sql_content)

            compiler = SQLModelCompiler(tmpdir, engine_type='bigquery')
            compiled = compiler._compile_model("test")

            assert "ML." in compiled

    def test_execute_model_not_found(self):
        """Test execute_model raises error for missing model"""
        sqlalchemy = pytest.importorskip("sqlalchemy")
        pandas = pytest.importorskip("pandas")
        from npcpy.sql.sql_model_compiler import SQLModelCompiler

        with tempfile.TemporaryDirectory() as tmpdir:
            compiler = SQLModelCompiler(tmpdir)

            with pytest.raises(ValueError) as exc_info:
                compiler.execute_model("nonexistent")

            assert "not found" in str(exc_info.value)

    def test_execute_model_simple_query(self):
        """Test executing a simple SQL model"""
        sqlalchemy = pytest.importorskip("sqlalchemy")
        pandas = pytest.importorskip("pandas")
        from npcpy.sql.sql_model_compiler import SQLModelCompiler

        with tempfile.TemporaryDirectory() as tmpdir:
            sql_content = "SELECT 1 AS col1, 'test' AS col2"
            with open(os.path.join(tmpdir, "simple.sql"), 'w') as f:
                f.write(sql_content)

            compiler = SQLModelCompiler(tmpdir)
            result = compiler.execute_model("simple")

            assert len(result) == 1
            assert result.iloc[0]["col1"] == 1
            assert result.iloc[0]["col2"] == "test"

    def test_execute_model_with_seed_data(self):
        """Test executing model with seed data"""
        sqlalchemy = pytest.importorskip("sqlalchemy")
        pandas = pytest.importorskip("pandas")
        import pandas as pd
        from npcpy.sql.sql_model_compiler import SQLModelCompiler

        with tempfile.TemporaryDirectory() as tmpdir:
            sql_content = "SELECT * FROM test_table WHERE value > 5"
            with open(os.path.join(tmpdir, "filtered.sql"), 'w') as f:
                f.write(sql_content)

            seed_data = {
                "test_table": pd.DataFrame({
                    "id": [1, 2, 3],
                    "value": [3, 7, 10]
                })
            }

            compiler = SQLModelCompiler(tmpdir)
            result = compiler.execute_model("filtered", seed_data=seed_data)

            assert len(result) == 2
            assert all(result["value"] > 5)

    def test_execute_model_with_engine(self):
        """Test executing model with SQLAlchemy engine"""
        sqlalchemy = pytest.importorskip("sqlalchemy")
        pandas = pytest.importorskip("pandas")
        import pandas as pd
        from npcpy.sql.sql_model_compiler import SQLModelCompiler

        with tempfile.TemporaryDirectory() as tmpdir:
            sql_content = "SELECT * FROM users"
            with open(os.path.join(tmpdir, "users.sql"), 'w') as f:
                f.write(sql_content)

            engine = sqlalchemy.create_engine("sqlite:///:memory:")
            seed_data = {
                "users": pd.DataFrame({
                    "id": [1, 2],
                    "name": ["Alice", "Bob"]
                })
            }

            compiler = SQLModelCompiler(tmpdir, engine=engine)
            result = compiler.execute_model("users", seed_data=seed_data)

            assert len(result) == 2

    def test_run_all_models(self):
        """Test running all discovered models"""
        sqlalchemy = pytest.importorskip("sqlalchemy")
        pandas = pytest.importorskip("pandas")
        from npcpy.sql.sql_model_compiler import SQLModelCompiler

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "model1.sql"), 'w') as f:
                f.write("SELECT 1 AS result")

            with open(os.path.join(tmpdir, "model2.sql"), 'w') as f:
                f.write("SELECT 2 AS result")

            compiler = SQLModelCompiler(tmpdir)
            results = compiler.run_all_models()

            assert len(results) == 2
            assert "model1" in results
            assert "model2" in results


class TestCreateModelCompiler:
    """Test create_model_compiler factory function."""

    def test_create_compiler_sqlite_default(self):
        """Test creating compiler with SQLite (default)"""
        sqlalchemy = pytest.importorskip("sqlalchemy")
        pandas = pytest.importorskip("pandas")
        from npcpy.sql.sql_model_compiler import create_model_compiler

        with tempfile.TemporaryDirectory() as tmpdir:
            compiler = create_model_compiler(tmpdir)

            assert compiler.engine_type == 'sqlite'
            assert compiler.engine is not None

    def test_create_compiler_invalid_snowflake_params(self):
        """Test creating Snowflake compiler with missing params raises error"""
        sqlalchemy = pytest.importorskip("sqlalchemy")
        pandas = pytest.importorskip("pandas")
        from npcpy.sql.sql_model_compiler import create_model_compiler

        with tempfile.TemporaryDirectory() as tmpdir:
            # Should raise when trying to create snowflake engine without params
            with pytest.raises((KeyError, TypeError, ModuleNotFoundError)):
                create_model_compiler(tmpdir, engine_type='snowflake', connection_params={})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
