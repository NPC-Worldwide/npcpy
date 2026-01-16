"""Test suite for browser session storage module."""

import pytest


class TestBrowserSessions:
    """Test browser session storage functions."""

    def test_get_sessions_returns_dict(self):
        """Test get_sessions returns a dictionary"""
        from npcpy.work import browser

        sessions = browser.get_sessions()

        assert isinstance(sessions, dict)

    def test_get_current_driver_none_initially(self):
        """Test get_current_driver returns None when no session set"""
        from npcpy.work import browser

        # Clear any existing sessions
        browser._sessions.clear()

        driver = browser.get_current_driver()

        assert driver is None

    def test_set_driver_stores_session(self):
        """Test set_driver stores the driver"""
        from npcpy.work import browser

        # Clear any existing sessions
        browser._sessions.clear()

        # Create a mock driver object
        class MockDriver:
            pass

        mock_driver = MockDriver()

        browser.set_driver("session-1", mock_driver)

        assert "session-1" in browser._sessions
        assert browser._sessions["session-1"] is mock_driver
        assert browser._sessions["current"] == "session-1"

    def test_set_driver_updates_current(self):
        """Test set_driver updates current session"""
        from npcpy.work import browser

        browser._sessions.clear()

        class MockDriver:
            pass

        browser.set_driver("session-1", MockDriver())
        assert browser._sessions["current"] == "session-1"

        browser.set_driver("session-2", MockDriver())
        assert browser._sessions["current"] == "session-2"

    def test_get_current_driver_returns_active_session(self):
        """Test get_current_driver returns the current driver"""
        from npcpy.work import browser

        browser._sessions.clear()

        class MockDriver:
            def __init__(self, name):
                self.name = name

        driver1 = MockDriver("driver1")
        driver2 = MockDriver("driver2")

        browser.set_driver("session-1", driver1)
        browser.set_driver("session-2", driver2)

        current = browser.get_current_driver()

        assert current is driver2
        assert current.name == "driver2"

    def test_close_current_removes_session(self):
        """Test close_current removes the current session"""
        from npcpy.work import browser

        browser._sessions.clear()

        class MockDriver:
            def __init__(self):
                self.quit_called = False

            def quit(self):
                self.quit_called = True

        mock_driver = MockDriver()
        browser.set_driver("session-1", mock_driver)

        result = browser.close_current()

        assert result is True
        assert "session-1" not in browser._sessions
        assert browser._sessions.get("current") is None

    def test_close_current_calls_quit(self):
        """Test close_current calls quit on driver"""
        from npcpy.work import browser

        browser._sessions.clear()

        quit_called = {"value": False}

        class MockDriver:
            def quit(self):
                quit_called["value"] = True

        browser.set_driver("session-1", MockDriver())
        browser.close_current()

        assert quit_called["value"] is True

    def test_close_current_handles_quit_error(self):
        """Test close_current handles quit errors gracefully"""
        from npcpy.work import browser

        browser._sessions.clear()

        class MockDriver:
            def quit(self):
                raise Exception("Browser already closed")

        browser.set_driver("session-1", MockDriver())

        # Should not raise, should return True
        result = browser.close_current()

        assert result is True

    def test_close_current_no_session(self):
        """Test close_current returns False when no session"""
        from npcpy.work import browser

        browser._sessions.clear()

        result = browser.close_current()

        assert result is False

    def test_multiple_sessions_stored(self):
        """Test multiple sessions can be stored"""
        from npcpy.work import browser

        browser._sessions.clear()

        class MockDriver:
            pass

        browser.set_driver("session-1", MockDriver())
        browser.set_driver("session-2", MockDriver())
        browser.set_driver("session-3", MockDriver())

        sessions = browser.get_sessions()

        assert "session-1" in sessions
        assert "session-2" in sessions
        assert "session-3" in sessions
        assert sessions["current"] == "session-3"

    def test_get_current_driver_invalid_current(self):
        """Test get_current_driver handles invalid current reference"""
        from npcpy.work import browser

        browser._sessions.clear()

        # Set current to non-existent session
        browser._sessions["current"] = "nonexistent"

        driver = browser.get_current_driver()

        assert driver is None


class TestBrowserModuleImports:
    """Test browser module imports correctly."""

    def test_module_imports(self):
        """Test browser module can be imported"""
        from npcpy.work import browser

        assert hasattr(browser, "get_sessions")
        assert hasattr(browser, "get_current_driver")
        assert hasattr(browser, "set_driver")
        assert hasattr(browser, "close_current")

    def test_sessions_is_module_level(self):
        """Test _sessions is module-level dict"""
        from npcpy.work import browser

        assert hasattr(browser, "_sessions")
        assert isinstance(browser._sessions, dict)


class TestSessionIsolation:
    """Test session isolation between tests."""

    def test_sessions_can_be_cleared(self):
        """Test sessions dict can be cleared"""
        from npcpy.work import browser

        browser._sessions["test"] = "value"
        browser._sessions.clear()

        assert len(browser._sessions) == 0

    def test_sessions_persist_in_module(self):
        """Test sessions persist across function calls"""
        from npcpy.work import browser

        browser._sessions.clear()

        class MockDriver:
            pass

        browser.set_driver("persistent", MockDriver())

        # Get sessions in separate call
        sessions = browser.get_sessions()

        assert "persistent" in sessions


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
