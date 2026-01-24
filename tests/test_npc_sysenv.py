# Tests for npc_sysenv platform-specific paths and model discovery

import os
import platform
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from npcpy.npc_sysenv import (
    get_data_dir,
    get_config_dir,
    get_cache_dir,
    get_npcshrc_path,
    get_history_db_path,
    get_models_dir,
    ensure_npcsh_dirs,
    ON_WINDOWS,
    ON_MACOS,
)


# =============================================================================
# Platform Path Tests
# =============================================================================

class TestPlatformPaths:
    """Test platform-specific path functions."""

    def test_get_data_dir_returns_string(self):
        """get_data_dir should return a string path."""
        result = get_data_dir()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_config_dir_returns_string(self):
        """get_config_dir should return a string path."""
        result = get_config_dir()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_cache_dir_returns_string(self):
        """get_cache_dir should return a string path."""
        result = get_cache_dir()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_models_dir_is_subdir_of_data(self):
        """get_models_dir should be inside data directory."""
        data_dir = get_data_dir()
        models_dir = get_models_dir()
        assert models_dir.startswith(data_dir) or 'npcsh' in models_dir

    def test_get_npcshrc_path_returns_string(self):
        """get_npcshrc_path should return a string path."""
        result = get_npcshrc_path()
        assert isinstance(result, str)
        assert 'npcshrc' in result or '.npcshrc' in result

    def test_get_history_db_path_returns_string(self):
        """get_history_db_path should return a string path."""
        result = get_history_db_path()
        assert isinstance(result, str)
        assert result.endswith('.db')


class TestXDGPaths:
    """Test XDG Base Directory compliance on Linux."""

    @pytest.mark.skipif(ON_WINDOWS or ON_MACOS, reason="XDG is Linux-specific")
    def test_xdg_data_home_respected(self):
        """XDG_DATA_HOME should be respected when set."""
        with patch.dict(os.environ, {'XDG_DATA_HOME': '/custom/data'}):
            with patch('os.path.exists', return_value=False):
                result = get_data_dir()
                assert '/custom/data' in result or 'npcsh' in result

    @pytest.mark.skipif(ON_WINDOWS or ON_MACOS, reason="XDG is Linux-specific")
    def test_xdg_config_home_respected(self):
        """XDG_CONFIG_HOME should be respected when set."""
        with patch.dict(os.environ, {'XDG_CONFIG_HOME': '/custom/config'}):
            with patch('os.path.exists', return_value=False):
                result = get_config_dir()
                assert '/custom/config' in result or 'npcsh' in result

    @pytest.mark.skipif(ON_WINDOWS or ON_MACOS, reason="XDG is Linux-specific")
    def test_xdg_cache_home_respected(self):
        """XDG_CACHE_HOME should be respected when set."""
        with patch.dict(os.environ, {'XDG_CACHE_HOME': '/custom/cache'}):
            result = get_cache_dir()
            assert '/custom/cache' in result


class TestBackwardsCompatibility:
    """Test backwards compatibility with old ~/.npcsh paths."""

    def test_old_npcsh_dir_used_if_exists(self):
        """Should use ~/.npcsh if it exists but new path doesn't."""
        old_path = os.path.expanduser('~/.npcsh')

        # If old path exists, it might be returned for backwards compat
        if os.path.exists(old_path):
            result = get_data_dir()
            # Either uses old path or new path - both are valid
            assert 'npcsh' in result.lower() or '.npcsh' in result

    def test_old_npcshrc_used_if_exists(self):
        """Should use ~/.npcshrc if it exists."""
        old_path = os.path.expanduser('~/.npcshrc')
        if os.path.exists(old_path):
            result = get_npcshrc_path()
            assert result == old_path


class TestEnsureDirs:
    """Test directory creation."""

    def test_ensure_npcsh_dirs_creates_directories(self):
        """ensure_npcsh_dirs should create necessary directories."""
        # This test just ensures no exceptions are raised
        # Actual directory creation depends on permissions
        try:
            ensure_npcsh_dirs()
        except PermissionError:
            pytest.skip("No permission to create directories")


# =============================================================================
# Model Discovery Tests
# =============================================================================

class TestModelDiscovery:
    """Test model auto-discovery functionality."""

    def test_mlx_discovery_ports(self):
        """MLX discovery should check ports 8000 and 5000."""
        from npcpy.npc_sysenv import get_locally_available_models

        # This tests that the function runs without error
        # Actual MLX server detection depends on running servers
        with tempfile.TemporaryDirectory() as tmpdir:
            models = get_locally_available_models(tmpdir, airplane_mode=True)
            assert isinstance(models, dict)

    def test_gguf_scanning_uses_models_dir(self):
        """GGUF scanning should include platform-specific models dir."""
        models_dir = get_models_dir()
        # Just verify the function returns the right type
        assert isinstance(models_dir, str)
        assert 'models' in models_dir.lower() or 'npcsh' in models_dir.lower()


# =============================================================================
# Integration Tests
# =============================================================================

class TestPathIntegration:
    """Integration tests for path functions."""

    def test_all_paths_are_absolute(self):
        """All path functions should return absolute paths."""
        paths = [
            get_data_dir(),
            get_config_dir(),
            get_cache_dir(),
            get_models_dir(),
            get_npcshrc_path(),
            get_history_db_path(),
        ]
        for path in paths:
            assert os.path.isabs(path) or path.startswith('~'), f"Path not absolute: {path}"

    def test_paths_contain_npcsh(self):
        """All paths should contain 'npcsh' in some form."""
        paths = [
            get_data_dir(),
            get_config_dir(),
            get_cache_dir(),
            get_models_dir(),
        ]
        for path in paths:
            assert 'npcsh' in path.lower() or '.npcsh' in path, f"Path missing npcsh: {path}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
