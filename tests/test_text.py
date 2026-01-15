"""Test suite for text processing functionality."""

import os
import tempfile
import shutil
import pytest


class TestLoadAllFiles:
    """Test load_all_files functionality."""

    def test_load_all_files(self):
        """Test loading all files from directory"""
        from npcpy.data.text import load_all_files

        temp_dir = tempfile.mkdtemp()

        try:
            files_to_create = {
                "test1.py": "def hello():\n    print('Hello from Python')",
                "test2.txt": "This is a text file with some content.",
                "test3.md": "# Test Markdown\nSome content here.",
                "test4.js": "function greet() { console.log('Hello from JS'); }",
                "ignore.log": "This file should be ignored"
            }

            for filename, content in files_to_create.items():
                file_path = os.path.join(temp_dir, filename)
                with open(file_path, "w") as f:
                    f.write(content)

            text_data = load_all_files(temp_dir, depth=1)

            assert isinstance(text_data, dict)
            assert len(text_data) >= 4

            py_files = [path for path in text_data.keys() if path.endswith('.py')]
            assert len(py_files) >= 1

        finally:
            shutil.rmtree(temp_dir)

    def test_load_all_files_custom_extensions(self):
        """Test loading files with custom extensions"""
        from npcpy.data.text import load_all_files

        temp_dir = tempfile.mkdtemp()

        try:
            files = {
                "data.csv": "name,age\nJohn,25",
                "config.json": '{"setting": "value"}',
                "readme.txt": "This is a readme file",
                "script.py": "print('hello')",
                "ignore.tmp": "temporary file"
            }

            for filename, content in files.items():
                file_path = os.path.join(temp_dir, filename)
                with open(file_path, "w") as f:
                    f.write(content)

            text_data = load_all_files(
                temp_dir,
                extensions=[".csv", ".json"],
                depth=1
            )

            assert isinstance(text_data, dict)
            assert len(text_data) == 2

            csv_files = [path for path in text_data.keys() if path.endswith('.csv')]
            json_files = [path for path in text_data.keys() if path.endswith('.json')]

            assert len(csv_files) == 1
            assert len(json_files) == 1

        finally:
            shutil.rmtree(temp_dir)

    def test_load_all_files_with_subdirectories(self):
        """Test loading files from subdirectories"""
        from npcpy.data.text import load_all_files

        temp_dir = tempfile.mkdtemp()

        try:
            subdir = os.path.join(temp_dir, "subdir")
            os.makedirs(subdir)

            with open(os.path.join(temp_dir, "main.txt"), "w") as f:
                f.write("Main directory file")

            with open(os.path.join(subdir, "sub.txt"), "w") as f:
                f.write("Subdirectory file")

            text_data = load_all_files(temp_dir, depth=2)

            assert isinstance(text_data, dict)
            assert len(text_data) >= 2

            main_files = [path for path in text_data.keys() if "main.txt" in path]
            sub_files = [path for path in text_data.keys() if "sub.txt" in path]

            assert len(main_files) == 1
            assert len(sub_files) == 1

        finally:
            shutil.rmtree(temp_dir)

    def test_load_all_files_depth_limit(self):
        """Test depth limiting in load_all_files"""
        from npcpy.data.text import load_all_files

        temp_dir = tempfile.mkdtemp()

        try:
            level1 = os.path.join(temp_dir, "level1")
            level2 = os.path.join(level1, "level2")
            os.makedirs(level2)

            with open(os.path.join(temp_dir, "root.txt"), "w") as f:
                f.write("Root level file")

            with open(os.path.join(level1, "level1.txt"), "w") as f:
                f.write("Level 1 file")

            with open(os.path.join(level2, "level2.txt"), "w") as f:
                f.write("Level 2 file")

            text_data = load_all_files(temp_dir, depth=1)

            level2_files = [path for path in text_data.keys() if "level2.txt" in path]
            assert len(level2_files) == 0

        finally:
            shutil.rmtree(temp_dir)

    def test_load_all_files_empty_directory(self):
        """Test load_all_files with empty directory"""
        from npcpy.data.text import load_all_files

        temp_dir = tempfile.mkdtemp()

        try:
            text_data = load_all_files(temp_dir)

            assert isinstance(text_data, dict)
            assert len(text_data) == 0

        finally:
            shutil.rmtree(temp_dir)


@pytest.mark.slow
class TestRAGSearch:
    """Test RAG search functionality - requires sentence_transformers (slow, loads model)."""

    def test_rag_search_with_string(self):
        """Test RAG search with string input"""
        pytest.importorskip("sentence_transformers")
        from npcpy.data.text import rag_search

        text_data = """
        Python is a programming language. It is used for web development.
        Machine learning is popular with Python. Data science uses Python libraries.
        JavaScript is used for web development. HTML and CSS are markup languages.
        """

        results = rag_search(
            query="Python programming",
            text_data=text_data,
            similarity_threshold=0.1
        )

        assert isinstance(results, list)

    def test_rag_search_with_dict(self):
        """Test RAG search with dictionary input"""
        pytest.importorskip("sentence_transformers")
        from npcpy.data.text import rag_search

        text_data = {
            "doc1.txt": "Python is a versatile programming language used in many fields.",
            "doc2.txt": "Web development often uses JavaScript and HTML technologies.",
            "doc3.txt": "Machine learning algorithms are implemented in Python frameworks."
        }

        results = rag_search(
            query="machine learning Python",
            text_data=text_data,
            similarity_threshold=0.1
        )

        assert isinstance(results, list)

    def test_rag_search_high_threshold(self):
        """Test RAG search with high similarity threshold"""
        pytest.importorskip("sentence_transformers")
        from npcpy.data.text import rag_search

        text_data = "The quick brown fox jumps over the lazy dog. Python is great for programming."

        results = rag_search(
            query="artificial intelligence deep learning",
            text_data=text_data,
            similarity_threshold=0.8
        )

        assert isinstance(results, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
