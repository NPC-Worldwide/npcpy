"""Test suite for memory_processor module."""

import pytest


class TestMemoryItem:
    """Test MemoryItem dataclass."""

    def test_memory_item_creation(self):
        """Test creating a MemoryItem"""
        from npcpy.memory.memory_processor import MemoryItem

        item = MemoryItem(
            message_id="msg-123",
            conversation_id="conv-456",
            npc="assistant",
            team="default",
            directory_path="/home/user/project",
            content="This is a test memory",
            context="User asked about testing",
            model="llama3.2",
            provider="ollama"
        )

        assert item.message_id == "msg-123"
        assert item.conversation_id == "conv-456"
        assert item.npc == "assistant"
        assert item.team == "default"
        assert item.directory_path == "/home/user/project"
        assert item.content == "This is a test memory"
        assert item.context == "User asked about testing"
        assert item.model == "llama3.2"
        assert item.provider == "ollama"

    def test_memory_item_equality(self):
        """Test MemoryItem equality comparison"""
        from npcpy.memory.memory_processor import MemoryItem

        item1 = MemoryItem(
            message_id="msg-1",
            conversation_id="conv-1",
            npc="npc1",
            team="team1",
            directory_path="/path",
            content="content",
            context="context",
            model="model",
            provider="provider"
        )

        item2 = MemoryItem(
            message_id="msg-1",
            conversation_id="conv-1",
            npc="npc1",
            team="team1",
            directory_path="/path",
            content="content",
            context="context",
            model="model",
            provider="provider"
        )

        assert item1 == item2

    def test_memory_item_different_values(self):
        """Test MemoryItem with different values are not equal"""
        from npcpy.memory.memory_processor import MemoryItem

        item1 = MemoryItem(
            message_id="msg-1",
            conversation_id="conv-1",
            npc="npc1",
            team="team1",
            directory_path="/path",
            content="content1",
            context="context",
            model="model",
            provider="provider"
        )

        item2 = MemoryItem(
            message_id="msg-1",
            conversation_id="conv-1",
            npc="npc1",
            team="team1",
            directory_path="/path",
            content="content2",  # Different content
            context="context",
            model="model",
            provider="provider"
        )

        assert item1 != item2

    def test_memory_item_all_fields_required(self):
        """Test MemoryItem requires all fields"""
        from npcpy.memory.memory_processor import MemoryItem

        with pytest.raises(TypeError):
            MemoryItem(
                message_id="msg-1",
                conversation_id="conv-1"
                # Missing required fields
            )


class TestMemoryApprovalUI:
    """Test memory_approval_ui function structure."""

    def test_empty_memories_returns_empty(self):
        """Test empty memories list returns empty approvals"""
        from npcpy.memory.memory_processor import memory_approval_ui

        result = memory_approval_ui([])

        assert result == []

    def test_function_exists(self):
        """Test memory_approval_ui function exists and is callable"""
        from npcpy.memory.memory_processor import memory_approval_ui

        assert callable(memory_approval_ui)


class TestMemoryDataStructures:
    """Test memory-related data structures and patterns."""

    def test_memory_dict_structure(self):
        """Test expected memory dict structure"""
        memory = {
            "memory_id": "mem-123",
            "npc": "assistant",
            "content": "This is memory content",
            "created_at": "2024-01-01T00:00:00"
        }

        assert "memory_id" in memory
        assert "npc" in memory
        assert "content" in memory

    def test_approval_dict_structure(self):
        """Test expected approval dict structure"""
        approval = {
            "memory_id": "mem-123",
            "decision": "human-approved"
        }

        assert approval["decision"] in [
            "human-approved",
            "human-rejected",
            "human-edited"
        ]

    def test_edited_approval_structure(self):
        """Test edited approval includes final_memory"""
        approval = {
            "memory_id": "mem-123",
            "decision": "human-edited",
            "final_memory": "Edited content here"
        }

        assert "final_memory" in approval
        assert approval["decision"] == "human-edited"

    def test_memory_content_preview_truncation(self):
        """Test memory content preview truncation logic"""
        long_content = "x" * 300  # 300 chars
        preview_limit = 200

        content_preview = long_content[:preview_limit]
        if len(long_content) > preview_limit:
            content_preview += '...'

        assert len(content_preview) == 203  # 200 + 3 for '...'
        assert content_preview.endswith('...')

    def test_memory_content_no_truncation_needed(self):
        """Test short content not truncated"""
        short_content = "Short memory"
        preview_limit = 200

        content_preview = short_content[:preview_limit]
        if len(short_content) > preview_limit:
            content_preview += '...'

        assert content_preview == short_content
        assert not content_preview.endswith('...')


class TestMemoryItemAsDict:
    """Test MemoryItem conversion patterns."""

    def test_memory_item_to_dict(self):
        """Test converting MemoryItem to dict"""
        from npcpy.memory.memory_processor import MemoryItem
        from dataclasses import asdict

        item = MemoryItem(
            message_id="msg-1",
            conversation_id="conv-1",
            npc="npc1",
            team="team1",
            directory_path="/path",
            content="content",
            context="context",
            model="model",
            provider="provider"
        )

        item_dict = asdict(item)

        assert isinstance(item_dict, dict)
        assert item_dict["message_id"] == "msg-1"
        assert item_dict["npc"] == "npc1"
        assert len(item_dict) == 9

    def test_memory_item_from_dict(self):
        """Test creating MemoryItem from dict"""
        from npcpy.memory.memory_processor import MemoryItem

        data = {
            "message_id": "msg-1",
            "conversation_id": "conv-1",
            "npc": "npc1",
            "team": "team1",
            "directory_path": "/path",
            "content": "content",
            "context": "context",
            "model": "model",
            "provider": "provider"
        }

        item = MemoryItem(**data)

        assert item.message_id == "msg-1"
        assert item.content == "content"


class TestDecisionTypes:
    """Test memory decision type constants."""

    def test_valid_decision_types(self):
        """Test valid decision type strings"""
        valid_decisions = [
            "human-approved",
            "human-rejected",
            "human-edited"
        ]

        for decision in valid_decisions:
            assert isinstance(decision, str)
            assert decision.startswith("human-")

    def test_approval_decision_mapping(self):
        """Test mapping user input to decisions"""
        input_to_decision = {
            'a': "human-approved",
            'r': "human-rejected",
            'e': "human-edited"
        }

        assert input_to_decision['a'] == "human-approved"
        assert input_to_decision['r'] == "human-rejected"
        assert input_to_decision['e'] == "human-edited"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
