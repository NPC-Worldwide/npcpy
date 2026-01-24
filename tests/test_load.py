"""Test suite for data loading functionality."""

import os
import tempfile
import shutil
import pytest
import pandas as pd


class TestLoadCSV:
    """Test CSV file loading."""

    def test_load_csv(self):
        """Test CSV file loading"""
        from npcpy.data.load import load_csv

        temp_dir = tempfile.mkdtemp()
        csv_file = os.path.join(temp_dir, "test.csv")

        try:
            with open(csv_file, "w") as f:
                f.write("name,age,city\n")
                f.write("John,25,NYC\n")
                f.write("Jane,30,LA\n")

            df = load_csv(csv_file)

            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            assert "name" in df.columns
            assert df.iloc[0]["name"] == "John"

        finally:
            shutil.rmtree(temp_dir)


class TestLoadJSON:
    """Test JSON file loading."""

    def test_load_json(self):
        """Test JSON file loading"""
        from npcpy.data.load import load_json
        import json

        temp_dir = tempfile.mkdtemp()
        json_file = os.path.join(temp_dir, "test.json")

        try:
            data = [
                {"name": "Alice", "score": 95},
                {"name": "Bob", "score": 87}
            ]
            with open(json_file, "w") as f:
                json.dump(data, f)

            data_out = load_json(json_file)

            assert isinstance(data_out, list)
            assert data_out[0]["name"] == "Alice"

        finally:
            shutil.rmtree(temp_dir)


class TestLoadTxt:
    """Test text file loading."""

    def test_load_txt(self):
        """Test text file loading"""
        from npcpy.data.load import load_txt

        temp_dir = tempfile.mkdtemp()
        txt_file = os.path.join(temp_dir, "test.txt")

        try:
            test_content = "This is a test document.\nIt has multiple lines.\nAnd some content."
            with open(txt_file, "w") as f:
                f.write(test_content)

            text = load_txt(txt_file)

            assert isinstance(text, str)
            assert test_content in text

        finally:
            shutil.rmtree(temp_dir)


class TestLoadFileContents:
    """Test load_file_contents functionality."""

    def test_load_file_contents_txt(self):
        """Test load_file_contents with text file"""
        from npcpy.data.load import load_file_contents

        temp_dir = tempfile.mkdtemp()
        txt_file = os.path.join(temp_dir, "test.txt")

        try:
            content = "This is a test. " * 100
            with open(txt_file, "w") as f:
                f.write(content)

            chunks = load_file_contents(txt_file, chunk_size=100)

            assert isinstance(chunks, list)
            assert len(chunks) > 1
            assert all(isinstance(chunk, str) for chunk in chunks)

        finally:
            shutil.rmtree(temp_dir)

    def test_load_file_contents_csv(self):
        """Test load_file_contents with CSV file"""
        from npcpy.data.load import load_file_contents

        temp_dir = tempfile.mkdtemp()
        csv_file = os.path.join(temp_dir, "test.csv")

        try:
            with open(csv_file, "w") as f:
                f.write("id,name,value\n")
                for i in range(50):
                    f.write(f"{i},item_{i},{i*10}\n")

            chunks = load_file_contents(csv_file, chunk_size=200)

            assert isinstance(chunks, list)
            assert len(chunks) >= 1
            assert "item_0" in chunks[0]

        finally:
            shutil.rmtree(temp_dir)

    def test_load_file_contents_json(self):
        """Test load_file_contents with JSON file"""
        from npcpy.data.load import load_file_contents
        import json

        temp_dir = tempfile.mkdtemp()
        json_file = os.path.join(temp_dir, "test.json")

        try:
            data = {"items": [{"id": i, "name": f"item_{i}"} for i in range(20)]}
            with open(json_file, "w") as f:
                json.dump(data, f, indent=2)

            chunks = load_file_contents(json_file, chunk_size=100)

            assert isinstance(chunks, list)
            assert len(chunks) >= 1
            assert "item_0" in chunks[0]

        finally:
            shutil.rmtree(temp_dir)

    def test_load_file_contents_unsupported(self):
        """Test load_file_contents with unsupported file type"""
        from npcpy.data.load import load_file_contents

        temp_dir = tempfile.mkdtemp()
        unknown_file = os.path.join(temp_dir, "test.unknown")

        try:
            with open(unknown_file, "w") as f:
                f.write("unknown content")

            chunks = load_file_contents(unknown_file)

            assert isinstance(chunks, list)
            assert len(chunks) == 1
            assert "Unsupported file format" in chunks[0]

        finally:
            shutil.rmtree(temp_dir)

    def test_load_file_contents_error_handling(self):
        """Test load_file_contents error handling"""
        from npcpy.data.load import load_file_contents

        chunks = load_file_contents("/nonexistent/file.txt")

        assert isinstance(chunks, list)
        assert len(chunks) == 1
        assert "Error loading file" in chunks[0]


class TestExtensionMap:
    """Test extension map functionality."""

    def test_extension_map(self):
        """Test that extension_map is working correctly"""
        from npcpy.data.load import extension_map

        assert "CSV" in extension_map
        assert extension_map["CSV"] == "documents"
        assert extension_map["PNG"] == "images"
        assert extension_map["MP4"] == "videos"
        assert extension_map["MP3"] == "audio"


class TestLoadExcel:
    """Test Excel file loading - requires openpyxl."""

    def test_load_excel(self):
        """Test Excel file loading"""
        pytest.importorskip("openpyxl")
        from npcpy.data.load import load_excel

        temp_dir = tempfile.mkdtemp()
        excel_file = os.path.join(temp_dir, "test.xlsx")

        try:
            test_data = pd.DataFrame({
                "name": ["Alice", "Bob", "Charlie"],
                "age": [25, 30, 35],
                "salary": [50000, 60000, 70000]
            })
            test_data.to_excel(excel_file, index=False)

            df = load_excel(excel_file)

            assert isinstance(df, pd.DataFrame)
            assert len(df) == 3
            assert "name" in df.columns
            assert df.iloc[0]["name"] == "Alice"

        finally:
            shutil.rmtree(temp_dir)


class TestLoadPDF:
    """Test PDF file loading - requires PyMuPDF."""

    def test_load_pdf(self):
        """Test PDF file loading"""
        fitz = pytest.importorskip("fitz")
        from npcpy.data.load import load_pdf

        temp_dir = tempfile.mkdtemp()
        try:
            pdf_path = os.path.join(temp_dir, "test.pdf")
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((72, 72), "Hello PDF world!")
            doc.save(pdf_path)
            doc.close()

            text = load_pdf(pdf_path)
            assert isinstance(text, str)
            assert "Hello PDF world" in text
        finally:
            shutil.rmtree(temp_dir)


@pytest.mark.slow
class TestLoadAudio:
    """Test audio file loading - requires faster_whisper (slow, loads model)."""

    def test_load_audio_wav(self, tmp_path):
        """Ensure audio loader returns text (transcript or fallback) for wav files."""
        pytest.importorskip("faster_whisper")
        import wave
        import struct
        from npcpy.data.load import load_audio

        wav_path = tmp_path / "beep.wav"

        with wave.open(str(wav_path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(struct.pack("<h", 0) * 16000)  # 1 second of silence

        result = load_audio(str(wav_path))
        assert isinstance(result, str)
        assert len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
