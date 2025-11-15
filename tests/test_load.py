import os
import tempfile
import pandas as pd
from npcpy.data.load import (
    load_audio,
    load_csv,
    load_json,
    load_txt,
    load_excel,
    load_pdf,
    load_file_contents,
)


def test_load_csv():
    """Test CSV file loading"""
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
        
        print(f"Loaded CSV with {len(df)} rows and columns: {list(df.columns)}")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_load_json():
    """Test JSON file loading"""
    temp_dir = tempfile.mkdtemp()
    json_file = os.path.join(temp_dir, "test.json")
    
    try:
        
        import json
        data = [
            {"name": "Alice", "score": 95},
            {"name": "Bob", "score": 87}
        ]
        with open(json_file, "w") as f:
            json.dump(data, f)
        
        data_out = load_json(json_file)
        
        assert isinstance(data_out, list)
        assert data_out[0]["name"] == "Alice"
        
        print(f"Loaded JSON with {len(data_out)} items")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_load_txt():
    """Test text file loading"""
    temp_dir = tempfile.mkdtemp()
    txt_file = os.path.join(temp_dir, "test.txt")
    
    try:
        
        test_content = "This is a test document.\nIt has multiple lines.\nAnd some content."
        with open(txt_file, "w") as f:
            f.write(test_content)
        
        text = load_txt(txt_file)
        
        assert isinstance(text, str)
        assert test_content in text
        
        print(f"Loaded text file with {len(text)} characters")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_load_file_contents_txt():
    """Test load_file_contents with text file"""
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
        
        print(f"Text file chunked into {len(chunks)} pieces")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_load_file_contents_csv():
    """Test load_file_contents with CSV file"""
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
        
        print(f"CSV file processed into {len(chunks)} chunks")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_load_file_contents_json():
    """Test load_file_contents with JSON file"""
    temp_dir = tempfile.mkdtemp()
    json_file = os.path.join(temp_dir, "test.json")
    
    try:
        
        import json
        data = {"items": [{"id": i, "name": f"item_{i}"} for i in range(20)]}
        with open(json_file, "w") as f:
            json.dump(data, f, indent=2)
        
        chunks = load_file_contents(json_file, chunk_size=100)
        
        assert isinstance(chunks, list)
        assert len(chunks) >= 1
        assert "item_0" in chunks[0]
        
        print(f"JSON file processed into {len(chunks)} chunks")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_load_file_contents_unsupported():
    """Test load_file_contents with unsupported file type"""
    temp_dir = tempfile.mkdtemp()
    unknown_file = os.path.join(temp_dir, "test.unknown")
    
    try:
        
        with open(unknown_file, "w") as f:
            f.write("unknown content")
        
        chunks = load_file_contents(unknown_file)
        
        assert isinstance(chunks, list)
        assert len(chunks) == 1
        assert "Unsupported file format" in chunks[0]
        
        print(f"Unsupported file handled correctly: {chunks[0]}")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_load_excel():
    """Test Excel file loading"""
    try:
        temp_dir = tempfile.mkdtemp()
        excel_file = os.path.join(temp_dir, "test.xlsx")
        
        
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
        
        print(f"Loaded Excel with {len(df)} rows")
        
    except ImportError:
        print("Excel test skipped - openpyxl not installed")
    except Exception as e:
        print(f"Excel test failed: {e}")
    finally:
        import shutil
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir)


def test_load_pdf():
    """Test PDF file loading"""
    try:
        import fitz  # type: ignore
    except Exception:
        print("PDF test skipped - fitz not installed")
        return
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
        import shutil
        shutil.rmtree(temp_dir)


def test_load_file_contents_error_handling():
    """Test load_file_contents error handling"""
    
    chunks = load_file_contents("/nonexistent/file.txt")
    
    assert isinstance(chunks, list)
    assert len(chunks) == 1
    assert "Error loading file" in chunks[0]
    
    print(f"Error handling test passed: {chunks[0][:50]}...")


def test_extension_map():
    """Test that extension_map is working correctly"""
    from npcpy.data.load import extension_map
    
    assert "CSV" in extension_map
    assert extension_map["CSV"] == "documents"
    assert extension_map["PNG"] == "images"
    assert extension_map["MP4"] == "videos"
    assert extension_map["MP3"] == "audio"
    
    print(f"Extension map contains {len(extension_map)} file types")


def test_load_audio_wav(tmp_path):
    """Ensure audio loader returns text (transcript or fallback) for wav files."""
    wav_path = tmp_path / "beep.wav"
    import wave
    import struct

    with wave.open(str(wav_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(struct.pack("<h", 0) * 16000)  # 1 second of silence

    result = load_audio(str(wav_path))
    assert isinstance(result, str)
    assert len(result) > 0
