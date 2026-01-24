"""Test suite for data_models module - Pydantic model definitions."""

import pytest
from pydantic import ValidationError


class TestNPCModel:
    """Test NPC_Model Pydantic model."""

    def test_npc_model_valid(self):
        """Test creating valid NPC_Model"""
        from npcpy.data.data_models import NPC_Model

        npc = NPC_Model(
            name="assistant",
            primary_directive="Help users with tasks",
            model="llama3.2",
            provider="ollama",
            api_url="http://localhost:11434",
            jinxs=["search", "summarize"]
        )

        assert npc.name == "assistant"
        assert npc.primary_directive == "Help users with tasks"
        assert npc.model == "llama3.2"
        assert npc.provider == "ollama"
        assert len(npc.jinxs) == 2

    def test_npc_model_empty_jinxs(self):
        """Test NPC_Model with empty jinxs list"""
        from npcpy.data.data_models import NPC_Model

        npc = NPC_Model(
            name="basic",
            primary_directive="Basic assistant",
            model="gpt-4",
            provider="openai",
            api_url="https://api.openai.com",
            jinxs=[]
        )

        assert npc.jinxs == []

    def test_npc_model_missing_field(self):
        """Test NPC_Model validation with missing required field"""
        from npcpy.data.data_models import NPC_Model

        with pytest.raises(ValidationError):
            NPC_Model(
                name="incomplete",
                primary_directive="Missing fields"
                # Missing model, provider, api_url, jinxs
            )

    def test_npc_model_serialization(self):
        """Test NPC_Model JSON serialization"""
        from npcpy.data.data_models import NPC_Model

        npc = NPC_Model(
            name="test",
            primary_directive="Test NPC",
            model="test-model",
            provider="test-provider",
            api_url="http://test.com",
            jinxs=["jinx1"]
        )

        # Test dict conversion
        data = npc.model_dump()
        assert isinstance(data, dict)
        assert data["name"] == "test"

        # Test JSON conversion
        json_str = npc.model_dump_json()
        assert isinstance(json_str, str)
        assert "test" in json_str


class TestJinxModel:
    """Test Jinx_Model Pydantic model."""

    def test_jinx_model_valid(self):
        """Test creating valid Jinx_Model"""
        from npcpy.data.data_models import Jinx_Model

        jinx = Jinx_Model(
            jinx_name="search_web",
            description="Search the web for information",
            steps=[
                {"engine": "python", "code": "result = search(query)"},
                {"engine": "llm", "code": "Summarize: {result}"}
            ]
        )

        assert jinx.jinx_name == "search_web"
        assert len(jinx.steps) == 2

    def test_jinx_model_empty_steps(self):
        """Test Jinx_Model with empty steps"""
        from npcpy.data.data_models import Jinx_Model

        jinx = Jinx_Model(
            jinx_name="empty",
            description="Empty jinx",
            steps=[]
        )

        assert jinx.steps == []

    def test_jinx_model_serialization(self):
        """Test Jinx_Model serialization"""
        from npcpy.data.data_models import Jinx_Model

        jinx = Jinx_Model(
            jinx_name="test",
            description="Test description",
            steps=[{"engine": "python", "code": "pass"}]
        )

        data = jinx.model_dump()
        assert data["jinx_name"] == "test"
        assert len(data["steps"]) == 1


class TestJinxStepModel:
    """Test JinxStep_Model Pydantic model."""

    def test_jinx_step_model_valid(self):
        """Test creating valid JinxStep_Model"""
        from npcpy.data.data_models import JinxStep_Model

        step = JinxStep_Model(
            engine="python",
            code="print('hello')"
        )

        assert step.engine == "python"
        assert step.code == "print('hello')"

    def test_jinx_step_model_llm_engine(self):
        """Test JinxStep_Model with LLM engine"""
        from npcpy.data.data_models import JinxStep_Model

        step = JinxStep_Model(
            engine="llm",
            code="Analyze this text: {input}"
        )

        assert step.engine == "llm"


class TestContextModel:
    """Test Context_Model Pydantic model."""

    def test_context_model_valid(self):
        """Test creating valid Context_Model"""
        from npcpy.data.data_models import Context_Model

        ctx = Context_Model(
            databases=["postgres://localhost/db1", "sqlite:///local.db"],
            files=["config.yaml", "data.json"],
            vars=[{"name": "API_KEY", "value": "secret"}]
        )

        assert len(ctx.databases) == 2
        assert len(ctx.files) == 2
        assert len(ctx.vars) == 1

    def test_context_model_empty(self):
        """Test Context_Model with empty lists"""
        from npcpy.data.data_models import Context_Model

        ctx = Context_Model(
            databases=[],
            files=[],
            vars=[]
        )

        assert ctx.databases == []
        assert ctx.files == []
        assert ctx.vars == []


class TestPipelineModel:
    """Test Pipeline_Model Pydantic model."""

    def test_pipeline_model_valid(self):
        """Test creating valid Pipeline_Model"""
        from npcpy.data.data_models import Pipeline_Model

        # Pipeline_Model.steps is List[Dict[str, str]] - all values must be strings
        pipeline = Pipeline_Model(
            steps=[
                {"jinx": "fetch_data", "args": "url"},
                {"jinx": "process", "args": ""},
                {"jinx": "output", "args": "format"}
            ]
        )

        assert len(pipeline.steps) == 3

    def test_pipeline_model_single_step(self):
        """Test Pipeline_Model with single step"""
        from npcpy.data.data_models import Pipeline_Model

        pipeline = Pipeline_Model(
            steps=[{"jinx": "single", "args": ""}]
        )

        assert len(pipeline.steps) == 1


class TestPipelineStepModel:
    """Test PipelineStep_Model Pydantic model."""

    def test_pipeline_step_model_valid(self):
        """Test creating valid PipelineStep_Model"""
        from npcpy.data.data_models import PipelineStep_Model

        step = PipelineStep_Model(
            jinx="analyze",
            args=["input.txt", "--verbose"],
            model="llama3.2",
            provider="ollama",
            task="analysis",
            npc="analyst"
        )

        assert step.jinx == "analyze"
        assert len(step.args) == 2
        assert step.model == "llama3.2"
        assert step.npc == "analyst"

    def test_pipeline_step_model_empty_args(self):
        """Test PipelineStep_Model with empty args"""
        from npcpy.data.data_models import PipelineStep_Model

        step = PipelineStep_Model(
            jinx="simple",
            args=[],
            model="gpt-4",
            provider="openai",
            task="simple_task",
            npc="worker"
        )

        assert step.args == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
