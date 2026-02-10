
import time
import platform
import subprocess
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

def capture_screenshot( full=False) -> Dict[str, str]:
    """
    Function Description:
        This function captures a screenshot of the current screen and saves it to a file.
    Args:
        npc: The NPC object representing the current NPC.
        full: Boolean to determine if full screen capture is needed. Default to true.
        path: Optional path to save the screenshot. Must not use placeholders. Relative paths preferred if the user specifies they want a specific path, otherwise default to None.
    Returns:
        A dictionary containing the filename, file path, and model kwargs.
    """
    

    directory = os.path.expanduser("~/.npcsh/screenshots")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"screenshot_{timestamp}.png"

    file_path = os.path.join(directory, filename)
    os.makedirs(directory, exist_ok=True)

    

    system = platform.system()

    model_kwargs = {}

    if full:
        
        if system.lower() == "darwin":
            
            subprocess.run(["screencapture", file_path], capture_output=True)
            
        elif system == "Linux":
            _took = False
            for _cmd, _args in [
                ("grim", [file_path]),
                ("scrot", [file_path]),
                ("import", ["-window", "root", file_path]),
                ("gnome-screenshot", ["-f", file_path]),
            ]:
                if subprocess.run(["which", _cmd], capture_output=True).returncode == 0:
                    subprocess.run([_cmd] + _args, capture_output=True, timeout=10)
                    if os.path.exists(file_path):
                        _took = True
                        break
            if not _took:
                print("No supported screenshot tool found. Install scrot, grim, or imagemagick.")

        elif system == "Windows":
            
            try:
                import win32gui
                import win32ui
                import win32con
                from PIL import Image

                
                width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
                height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)

                
                hdesktop = win32gui.GetDesktopWindow()
                desktop_dc = win32gui.GetWindowDC(hdesktop)
                img_dc = win32ui.CreateDCFromHandle(desktop_dc)
                mem_dc = img_dc.CreateCompatibleDC()

                
                screenshot = win32ui.CreateBitmap()
                screenshot.CreateCompatibleBitmap(img_dc, width, height)
                mem_dc.SelectObject(screenshot)
                mem_dc.BitBlt((0, 0), (width, height), img_dc, (0, 0), win32con.SRCCOPY)

                
                screenshot.SaveBitmapFile(mem_dc, file_path)

                
                mem_dc.DeleteDC()
                win32gui.DeleteObject(screenshot.GetHandle())

            except ImportError:
                print(
                    "Required packages not found. Please install: pip install pywin32"
                )
                return None
        else:
            print(f"Unsupported operating system: {system}")
            return None
    else:
        if system == "Darwin":
            subprocess.run(["screencapture", "-i", file_path])
        elif system == "Linux":
            if (
                subprocess.run(
                    ["which", "gnome-screenshot"], capture_output=True
                ).returncode
                == 0
            ):
                subprocess.Popen(["gnome-screenshot", "-a", "-f", file_path])
                while not os.path.exists(file_path):
                    time.sleep(0.5)
            elif (
                subprocess.run(["which", "scrot"], capture_output=True).returncode == 0
            ):
                subprocess.Popen(["scrot", "-s", file_path])
                while not os.path.exists(file_path):
                    time.sleep(0.5)
            else:
                print(
                    "No supported screenshot jinx found. Please install gnome-screenshot or scrot."
                )
                return None
        elif system == "Windows":
            success = _windows_snip_to_file(file_path)
            if not success:
                print("Screenshot capture failed or timed out.")
                return None
        else:
            print(f"Unsupported operating system: {system}")
            return None

    
    if os.path.exists(file_path):
        print(f"Screenshot saved to: {file_path}")
        return {
            "filename": filename,
            "file_path": file_path,
            "model_kwargs": model_kwargs,
        }
    else:
        print("Screenshot capture failed or was cancelled.")
        return None

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

