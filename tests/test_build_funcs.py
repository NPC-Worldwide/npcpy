"""Test suite for build_funcs module."""

import os
import tempfile
import shutil
import pytest
import yaml


class TestGetTeamName:
    """Test get_team_name function."""

    def test_get_team_name_from_ctx_file(self):
        """Test extracting team name from .ctx file"""
        from npcpy.build_funcs import get_team_name

        temp_dir = tempfile.mkdtemp()
        try:
            # Create a .ctx file with team name
            ctx_content = {"name": "My Awesome Team", "version": "1.0"}
            ctx_path = os.path.join(temp_dir, "team.ctx")
            with open(ctx_path, "w") as f:
                yaml.dump(ctx_content, f)

            name = get_team_name(temp_dir)
            assert name == "My Awesome Team"
        finally:
            shutil.rmtree(temp_dir)

    def test_get_team_name_from_folder(self):
        """Test getting team name from folder when no .ctx file"""
        from npcpy.build_funcs import get_team_name

        temp_dir = tempfile.mkdtemp()
        try:
            # Create a subfolder to test
            team_dir = os.path.join(temp_dir, "my_team")
            os.makedirs(team_dir)

            name = get_team_name(team_dir)
            assert name == "my_team"
        finally:
            shutil.rmtree(temp_dir)

    def test_get_team_name_npc_team_fallback(self):
        """Test fallback when folder is named npc_team"""
        from npcpy.build_funcs import get_team_name

        temp_dir = tempfile.mkdtemp()
        try:
            npc_team_dir = os.path.join(temp_dir, "npc_team")
            os.makedirs(npc_team_dir)

            name = get_team_name(npc_team_dir)
            # Should use parent folder name
            assert name == os.path.basename(temp_dir)
        finally:
            shutil.rmtree(temp_dir)


class TestBuildDockerfile:
    """Test build_dockerfile function."""

    def test_build_dockerfile_default(self):
        """Test Dockerfile generation with defaults"""
        from npcpy.build_funcs import build_dockerfile

        temp_dir = tempfile.mkdtemp()
        try:
            team_dir = os.path.join(temp_dir, "test_team")
            os.makedirs(team_dir)

            config = {"team_path": team_dir}
            dockerfile = build_dockerfile(config)

            assert "FROM python:3.11-slim" in dockerfile
            assert "pip install --no-cache-dir npcsh" in dockerfile
            assert "EXPOSE 5337" in dockerfile
            assert "test_team" in dockerfile
        finally:
            shutil.rmtree(temp_dir)

    def test_build_dockerfile_custom_port(self):
        """Test Dockerfile with custom port"""
        from npcpy.build_funcs import build_dockerfile

        temp_dir = tempfile.mkdtemp()
        try:
            team_dir = os.path.join(temp_dir, "my_team")
            os.makedirs(team_dir)

            config = {"team_path": team_dir, "port": 8080}
            dockerfile = build_dockerfile(config)

            assert "EXPOSE 8080" in dockerfile
            assert '"8080"' in dockerfile
        finally:
            shutil.rmtree(temp_dir)


class TestBuildDockerCompose:
    """Test build_docker_compose function."""

    def test_build_docker_compose_creates_files(self):
        """Test docker-compose build creates all expected files"""
        from npcpy.build_funcs import build_docker_compose

        temp_dir = tempfile.mkdtemp()
        try:
            team_dir = os.path.join(temp_dir, "test_team")
            os.makedirs(team_dir)
            # Create a dummy file in team dir
            with open(os.path.join(team_dir, "test.npc"), "w") as f:
                f.write("name: test\n")

            output_dir = os.path.join(temp_dir, "build")
            config = {"team_path": team_dir, "output_dir": output_dir}

            result = build_docker_compose(config)

            # Check files were created
            assert os.path.exists(os.path.join(output_dir, "Dockerfile"))
            assert os.path.exists(os.path.join(output_dir, "docker-compose.yml"))
            assert os.path.exists(os.path.join(output_dir, ".env.example"))
            assert os.path.exists(os.path.join(output_dir, "test_team"))

            # Check output message
            assert "output" in result
            assert "Docker deployment created" in result["output"]
        finally:
            shutil.rmtree(temp_dir)

    def test_build_docker_compose_with_cors(self):
        """Test docker-compose with CORS origins"""
        from npcpy.build_funcs import build_docker_compose

        temp_dir = tempfile.mkdtemp()
        try:
            team_dir = os.path.join(temp_dir, "test_team")
            os.makedirs(team_dir)

            output_dir = os.path.join(temp_dir, "build")
            config = {
                "team_path": team_dir,
                "output_dir": output_dir,
                "cors_origins": ["http://localhost:3000", "https://example.com"],
            }

            build_docker_compose(config)

            # Check docker-compose.yml contains CORS
            compose_path = os.path.join(output_dir, "docker-compose.yml")
            with open(compose_path, "r") as f:
                content = f.read()

            assert "CORS_ORIGINS" in content
            assert "localhost:3000" in content
        finally:
            shutil.rmtree(temp_dir)


class TestBuildFlaskServer:
    """Test build_flask_server function."""

    def test_build_flask_server_creates_script(self):
        """Test Flask server script generation"""
        from npcpy.build_funcs import build_flask_server

        temp_dir = tempfile.mkdtemp()
        try:
            team_dir = os.path.join(temp_dir, "test_team")
            os.makedirs(team_dir)

            output_dir = os.path.join(temp_dir, "build")
            config = {"team_path": team_dir, "output_dir": output_dir, "port": 5000}

            result = build_flask_server(config)

            # Check server.py was created
            server_path = os.path.join(output_dir, "server.py")
            assert os.path.exists(server_path)

            # Check content
            with open(server_path, "r") as f:
                content = f.read()

            assert "from npcpy.serve import start_flask_server" in content
            assert "port=5000" in content

            # Check output message
            assert "Flask server created" in result["output"]
        finally:
            shutil.rmtree(temp_dir)


class TestBuildCliExecutable:
    """Test build_cli_executable function."""

    def test_build_cli_executable_creates_scripts(self):
        """Test CLI script generation for NPCs"""
        from npcpy.build_funcs import build_cli_executable

        temp_dir = tempfile.mkdtemp()
        try:
            team_dir = os.path.join(temp_dir, "test_team")
            os.makedirs(team_dir)

            # Create some .npc files
            for name in ["assistant", "coder", "reviewer"]:
                npc_path = os.path.join(team_dir, f"{name}.npc")
                with open(npc_path, "w") as f:
                    f.write(f"name: {name}\n")

            output_dir = os.path.join(temp_dir, "build")
            config = {"team_path": team_dir, "output_dir": output_dir}

            result = build_cli_executable(config)

            # Check scripts were created
            assert os.path.exists(os.path.join(output_dir, "assistant"))
            assert os.path.exists(os.path.join(output_dir, "coder"))
            assert os.path.exists(os.path.join(output_dir, "reviewer"))

            # Check output message
            assert "CLI scripts created" in result["output"]
            assert "assistant" in result["output"]
        finally:
            shutil.rmtree(temp_dir)


class TestBuildStaticSite:
    """Test build_static_site function."""

    def test_build_static_site_creates_html(self):
        """Test static site HTML generation"""
        from npcpy.build_funcs import build_static_site

        temp_dir = tempfile.mkdtemp()
        try:
            team_dir = os.path.join(temp_dir, "test_team")
            os.makedirs(team_dir)

            # Create .ctx file with team name
            ctx_content = {"name": "Documentation Team"}
            with open(os.path.join(team_dir, "team.ctx"), "w") as f:
                yaml.dump(ctx_content, f)

            # Create .npc files
            npc_content = {"name": "writer", "primary_directive": "Write documentation"}
            with open(os.path.join(team_dir, "writer.npc"), "w") as f:
                yaml.dump(npc_content, f)

            output_dir = os.path.join(temp_dir, "build")
            config = {"team_path": team_dir, "output_dir": output_dir}

            result = build_static_site(config)

            # Check index.html was created
            html_path = os.path.join(output_dir, "index.html")
            assert os.path.exists(html_path)

            # Check content
            with open(html_path, "r") as f:
                content = f.read()

            assert "Documentation Team" in content
            assert "writer" in content
            assert "Write documentation" in content
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
