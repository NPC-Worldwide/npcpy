
import time
import platform
import subprocess
import tempfile
from typing import Dict, Any
import os
import io
from PIL import Image

def _windows_snip_to_file(file_path: str) -> bool:
    """Helper function to trigger Windows snipping and save to file."""
    try:
        
        import win32clipboard
        from PIL import ImageGrab
        from ctypes import windll

        
        windll.user32.keybd_event(0x5B, 0, 0, 0)  
        windll.user32.keybd_event(0x10, 0, 0, 0)  
        windll.user32.keybd_event(0x53, 0, 0, 0)  
        windll.user32.keybd_event(0x53, 0, 0x0002, 0)  
        windll.user32.keybd_event(0x10, 0, 0x0002, 0)  
        windll.user32.keybd_event(0x5B, 0, 0x0002, 0)  

        
        print("Please select an area to capture...")
        time.sleep(1)  

        
        max_wait = 30  
        start_time = time.time()

        while time.time() - start_time < max_wait:
            try:
                image = ImageGrab.grabclipboard()
                if image:
                    image.save(file_path, "PNG")
                    return True
            except Exception:
                pass
            time.sleep(0.5)

        return False

    except ImportError:
        print("Required packages not found. Please install: pip install pywin32 Pillow")
        return False

def capture_screenshot(path: str = None, full=False) -> Dict[str, Any]:
    """Capture a screenshot and return image data. If path is given, also save to disk."""
    system = platform.system()
    img = None
    if system.lower() == "darwin":
        tmp = os.path.join(tempfile.gettempdir(), f"sc_{time.time()}.png")
        args = ["screencapture", tmp]
        if not full:
            args.insert(1, "-i")
        subprocess.run(args, capture_output=True)
        if os.path.exists(tmp):
            img = Image.open(tmp)
            if not path:
                os.remove(tmp)
    elif system == "Linux":
        tmp = os.path.join(tempfile.gettempdir(), f"sc_{time.time()}.png")
        took = False
        for cmd, args in [
            ("grim", [tmp] if full else ["-g", "$(slurp)", tmp]),
            ("scrot", [tmp] if full else ["-s", tmp]),
            ("import", ["-window", "root", tmp] if full else [tmp]),
            ("gnome-screenshot", ["-f", tmp] if full else ["-a", "-f", tmp]),
        ]:
            if subprocess.run(["which", cmd], capture_output=True).returncode == 0:
                subprocess.run([cmd] + args, capture_output=True, timeout=10, shell=(cmd == "grim" and not full))
                if os.path.exists(tmp):
                    took = True
                    break
        if took:
            img = Image.open(tmp)
            if not path:
                os.remove(tmp)
        else:
            print("No supported screenshot tool found. Install scrot, grim, or imagemagick.")
            return None
    elif system == "Windows":
        try:
            import win32gui, win32ui, win32con
            hdesktop = win32gui.GetDesktopWindow()
            desktop_dc = win32gui.GetWindowDC(hdesktop)
            img_dc = win32ui.CreateDCFromHandle(desktop_dc)
            mem_dc = img_dc.CreateCompatibleDC()
            width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
            height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
            screenshot = win32ui.CreateBitmap()
            screenshot.CreateCompatibleBitmap(img_dc, width, height)
            mem_dc.SelectObject(screenshot)
            mem_dc.BitBlt((0, 0), (width, height), img_dc, (0, 0), win32con.SRCCOPY)
            bmpinfo = screenshot.GetInfo()
            bmpstr = screenshot.GetBitmapBits(True)
            img = Image.frombuffer('RGB', (bmpinfo['bmWidth'], bmpinfo['bmHeight']), bmpstr, 'raw', 'BGRX', 0, 1)
            mem_dc.DeleteDC()
            win32gui.DeleteObject(screenshot.GetHandle())
        except ImportError:
            print("Required packages not found. Please install: pip install pywin32 Pillow")
            return None
    else:
        print(f"Unsupported operating system: {system}")
        return None

    if img is None:
        print("Screenshot capture failed or was cancelled.")
        return None

    if path:
        os.makedirs(os.path.dirname(os.path.expanduser(path)), exist_ok=True)
        img.save(path, "PNG")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return {"image": img, "bytes": buf.getvalue(), "path": path}

def compress_image(image_bytes, max_size=(800, 600)):
    
    buffer = io.BytesIO(image_bytes)
    img = Image.open(buffer)

    
    img.load()

    
    if img.mode == "RGBA":
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background

    
    if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
        img.thumbnail(max_size)

    
    out_buffer = io.BytesIO()
    img.save(out_buffer, format="JPEG", quality=95, optimize=False)
    return out_buffer.getvalue()

