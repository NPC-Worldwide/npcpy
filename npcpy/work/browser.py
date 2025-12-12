"""
Global browser session storage for selenium automation.
"""

_sessions = {}

def get_sessions():
    return _sessions

def get_current_driver():
    current = _sessions.get('current')
    if current and current in _sessions:
        return _sessions[current]
    return None

def set_driver(session_id, driver):
    _sessions[session_id] = driver
    _sessions['current'] = session_id

def close_current():
    current = _sessions.get('current')
    if current and current in _sessions:
        try:
            _sessions[current].quit()
        except:
            pass
        del _sessions[current]
        _sessions['current'] = None
        return True
    return False
