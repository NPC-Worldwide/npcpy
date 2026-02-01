

action_space = {
        "click": {"x": "int (0-100)", "y": "int (0-100)", "button": "left|right|middle (optional)"},
        "type": {"text": "string"},
        "key": {"keys": "list of key names or a single key string"},
        "scroll": {"direction": "up|down", "amount": "int (optional)"},
        "wait": {"duration": "float seconds"},
        "shell": {"command": "string"},
        "hotkey": {"keys": "list of keys to press simultaneously"},
        "drag": {"start_x": "int (0-100)", "start_y": "int (0-100)", "end_x": "int (0-100)", "end_y": "int (0-100)"},
        "quit": {"description": "Use this action when the main goal is 100% complete."},
    }
try:
    import pyautogui
except:
    print('could not import pyautogui')
import time
import subprocess

def perform_action(action):
    action_type = action.get("type")
    print(f"Action received: {action}")

    try:
        width, height = pyautogui.size()

        if action_type == "click":
            x = max(0, min(100, action.get("x", 50)))
            y = max(0, min(100, action.get("y", 50)))
            x_pixel = int(x * width / 100)
            y_pixel = int(y * height / 100)
            print(f"Moving to ({x_pixel}, {y_pixel}) then clicking.")
            pyautogui.moveTo(x_pixel, y_pixel, duration=0.2)
            pyautogui.click()
            return {"status": "success", "output": f"Clicked at ({x}, {y})."}

        elif action_type in ("shell", "bash"):
            command = action.get("command", "")
            print(f"Executing shell command: {command}")
            subprocess.Popen(command, shell=True)
            return {"status": "success", "output": f"Launched '{command}'."}
        
        elif action_type == "type":
            text_to_type = action.get("text", "")
            print(f"Typing text: {text_to_type}")
            import platform as _plat
            _typed = False
            if _plat.system() == "Linux":
                # xdotool types reliably on X11 (handles focus, modifiers, speed)
                try:
                    r = subprocess.run(
                        ["xdotool", "type", "--clearmodifiers", "--delay", "12", "--", text_to_type],
                        capture_output=True, timeout=15
                    )
                    if r.returncode == 0:
                        _typed = True
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    pass
            if not _typed:
                # Fallback: clipboard paste (works cross-platform, handles all chars)
                try:
                    import pyperclip
                    pyperclip.copy(text_to_type)
                    _mod = "command" if _plat.system() == "Darwin" else "ctrl"
                    time.sleep(0.1)
                    pyautogui.hotkey(_mod, "v")
                    _typed = True
                except Exception:
                    pass
            if not _typed:
                # Last resort: character-by-character with interval
                pyautogui.typewrite(text_to_type, interval=0.03)
            return {"status": "success", "output": f"Typed '{text_to_type}'."}

        elif action_type == "key":
            keys_to_press = action.get("keys", [])
            if isinstance(keys_to_press, str):
                keys_to_press = [keys_to_press]
            print(f"Pressing key(s): {keys_to_press}")
            for k in keys_to_press:
                pyautogui.press(k.lower())
            return {"status": "success", "output": "Pressed key(s)."}

        elif action_type == "hotkey":
            keys = action.get("keys", [])
            print(f"Pressing hotkey: {keys}")
            pyautogui.hotkey(*keys)
            return {"status": "success", "output": "Pressed hotkey."}

        elif action_type == "wait":
            duration = action.get("duration", 1)
            print(f"Waiting for {duration} seconds")
            time.sleep(duration)
            return {"status": "success", "output": f"Waited for {duration}s."}
        
        else:
            return {"status": "error", "message": f"Unknown or malformed action: {action}"}

    except Exception as e:
        return {"status": "error", "message": str(e)}