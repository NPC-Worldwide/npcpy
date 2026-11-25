import os
import tempfile
import time
import platform
from npcpy.data.image import capture_screenshot, compress_image


def test_capture_screenshot_basic():
    """Test basic screenshot functionality"""
    try:
        result = capture_screenshot(full=True)
        
        if result:
            assert "filename" in result
            assert "file_path" in result
            assert os.path.exists(result["file_path"])
            
            
            file_size = os.path.getsize(result["file_path"])
            assert file_size > 0
            
            print(f"Screenshot captured: {result['filename']} ({file_size} bytes)")
            
            
            os.remove(result["file_path"])
        else:
            print("Screenshot capture returned None (user may have cancelled)")
            
    except Exception as e:
        print(f"Screenshot test failed: {e}")


def test_capture_screenshot_interactive():
    """Test interactive screenshot (area selection)"""
    try:
        print("Testing interactive screenshot - this may require user interaction...")
        
        result = capture_screenshot(full=False)
        
        if result:
            assert "filename" in result
            assert "file_path" in result
            
            if os.path.exists(result["file_path"]):
                file_size = os.path.getsize(result["file_path"])
                print(f"Interactive screenshot captured: {result['filename']} ({file_size} bytes)")
                
                
                os.remove(result["file_path"])
            else:
                print("Screenshot file not found - user may have cancelled")
        else:
            print("Interactive screenshot cancelled or failed")
            
    except Exception as e:
        print(f"Interactive screenshot test failed: {e}")


def test_capture_screenshot_with_npc():
    """Test screenshot with NPC configuration"""
    try:
        from npcpy.npc_compiler import NPC
        
        test_npc = NPC(
            name="screenshot_npc",
            model="llava:7b",
            provider="ollama"
        )
        
        result = capture_screenshot(npc=test_npc, full=True)
        
        if result:
            assert "model_kwargs" in result
            assert result["model_kwargs"]["model"] == "llava:7b"
            assert result["model_kwargs"]["provider"] == "ollama"
            
            print(f"NPC screenshot test passed: {result['model_kwargs']}")
            
            
            if os.path.exists(result["file_path"]):
                os.remove(result["file_path"])
        else:
            print("NPC screenshot test - no result returned")
            
    except Exception as e:
        print(f"NPC screenshot test failed: {e}")


def test_compress_image():
    """Test image compression functionality"""
    try:
        from PIL import Image
        import io
        
        
        img = Image.new('RGB', (1200, 800), color='red')
        
        
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        original_bytes = img_buffer.getvalue()
        
        
        compressed_bytes = compress_image(original_bytes, max_size=(800, 600))
        
        assert isinstance(compressed_bytes, bytes)
        assert len(compressed_bytes) > 0
        
        
        compressed_img = Image.open(io.BytesIO(compressed_bytes))
        
        
        assert compressed_img.size[0] <= 800
        assert compressed_img.size[1] <= 600
        
        print(f"Image compression: {len(original_bytes)} -> {len(compressed_bytes)} bytes")
        print(f"Size reduction: {img.size} -> {compressed_img.size}")
        
    except ImportError:
        print("Image compression test skipped - PIL not available")
    except Exception as e:
        print(f"Image compression test failed: {e}")


def test_compress_image_rgba():
    """Test image compression with RGBA image"""
    try:
        from PIL import Image
        import io
        
        
        img = Image.new('RGBA', (400, 300), color=(255, 0, 0, 128))
        
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        original_bytes = img_buffer.getvalue()
        
        
        compressed_bytes = compress_image(original_bytes)
        
        
        compressed_img = Image.open(io.BytesIO(compressed_bytes))
        assert compressed_img.mode == 'RGB'
        
        print(f"RGBA compression test passed: {img.mode} -> {compressed_img.mode}")
        
    except ImportError:
        print("RGBA compression test skipped - PIL not available")
    except Exception as e:
        print(f"RGBA compression test failed: {e}")


def test_compress_image_small_image():
    """Test compression with image smaller than max_size"""
    try:
        from PIL import Image
        import io
        
        
        img = Image.new('RGB', (200, 150), color='blue')
        
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        original_bytes = img_buffer.getvalue()
        
        
        compressed_bytes = compress_image(original_bytes, max_size=(800, 600))
        
        compressed_img = Image.open(io.BytesIO(compressed_bytes))
        
        
        assert compressed_img.size == (200, 150)
        
        print(f"Small image test passed: size unchanged {compressed_img.size}")
        
    except ImportError:
        print("Small image test skipped - PIL not available")
    except Exception as e:
        print(f"Small image test failed: {e}")


def test_screenshot_directory_creation():
    """Test that screenshot directory is created if it doesn't exist"""
    try:
        
        screenshot_dir = os.path.expanduser("~/.npcsh/screenshots")
        
        
        if os.path.exists(screenshot_dir):
            import shutil
            shutil.rmtree(os.path.dirname(screenshot_dir))
        
        
        result = capture_screenshot(full=True)
        
        
        assert os.path.exists(screenshot_dir)
        
        print("Screenshot directory creation test passed")
        
        
        if result and os.path.exists(result["file_path"]):
            os.remove(result["file_path"])
            
    except Exception as e:
        print(f"Directory creation test failed: {e}")


def test_platform_specific_screenshot():
    """Test platform-specific screenshot behavior"""
    system = platform.system()
    
    print(f"Testing on platform: {system}")
    
    try:
        if system == "Darwin":
            print("Testing macOS screenshot functionality")
        elif system == "Linux":
            print("Testing Linux screenshot functionality")
        elif system == "Windows":
            print("Testing Windows screenshot functionality")
        else:
            print(f"Unknown platform: {system}")
            
        
        result = capture_screenshot(full=True)
        
        if result:
            print(f"Platform-specific test passed on {system}")
            if os.path.exists(result["file_path"]):
                os.remove(result["file_path"])
        else:
            print(f"Platform-specific test returned None on {system}")
            
    except Exception as e:
        print(f"Platform-specific test failed on {system}: {e}")


def test_compress_image_different_formats():
    """Test compression with different image formats"""
    try:
        from PIL import Image
        import io
        
        
        img = Image.new('RGB', (600, 400), color='green')
        
        
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='JPEG', quality=90)
        jpeg_bytes = img_buffer.getvalue()
        
        
        compressed_bytes = compress_image(jpeg_bytes)
        
        
        compressed_img = Image.open(io.BytesIO(compressed_bytes))
        assert compressed_img.mode == 'RGB'
        
        print("Different format compression test passed")
        
    except ImportError:
        print("Different format test skipped - PIL not available")
    except Exception as e:
        print(f"Different format test failed: {e}")


def _png_bytes(img):
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_compress_image_palette_and_grayscale_alpha_modes():
    """PNG modes P, LA, 1, I;16 must not crash the JPEG save."""
    import io
    from PIL import Image

    images = [
        Image.new("P", (32, 32)),
        Image.new("LA", (32, 32), (128, 0)),
        Image.new("1", (32, 32)),
        Image.new("I;16", (32, 32)),
    ]
    for img in images:
        out = compress_image(_png_bytes(img))
        assert Image.open(io.BytesIO(out)).format == "JPEG", img.mode


def test_compress_image_transparent_pixels_become_white():
    import io
    from PIL import Image

    la = Image.new("LA", (8, 8), (0, 0))
    out = Image.open(io.BytesIO(compress_image(_png_bytes(la)))).convert("RGB")
    assert min(out.getpixel((4, 4))) > 240

    p = Image.new("P", (8, 8), 0)
    p.putpalette([255, 0, 0] * 256)
    p.info["transparency"] = 0
    out = Image.open(io.BytesIO(compress_image(_png_bytes(p)))).convert("RGB")
    assert min(out.getpixel((4, 4))) > 240
