# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
import warnings
from typing import Any
from unittest.mock import MagicMock, mock_open, patch

import pytest

from autogen.agentchat.conversable_agent import ConversableAgent
from autogen.agentchat.group.guardrails import LLMGuardrail, RegexGuardrail
from autogen.agentchat.group.safeguard import SafeguardManager, apply_safeguards
from autogen.llm_config.config import LLMConfig

# Suppress Google protobuf warnings at the module level
warnings.filterwarnings("ignore", message=".*MessageMapContainer.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*ScalarMapContainer.*", category=DeprecationWarning)

# Pytest marker to suppress warnings for this entire test module
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


class TestSafeguardManager:
    """Test SafeguardManager core functionality."""

    def test_valid_manifest_initialization(self) -> None:
        """Test SafeguardManager with valid manifest."""
        manifest = {
            "inter_agent_safeguards": {
                "agent_transitions": [
                    {
                        "message_src": "agent1",
                        "message_dst": "agent2",
                        "check_method": "regex",
                        "pattern": r"\btest\b",
                        "violation_response": "block",
                    }
                ]
            }
        }

        manager = SafeguardManager(manifest=manifest)
        assert len(manager.inter_agent_rules) >= 0

    def test_manifest_file_loading(self) -> None:
        """Test loading manifest from file."""
        manifest_content: dict[str, Any] = {"inter_agent_safeguards": {"agent_transitions": []}}

        with patch("builtins.open", mock_open(read_data=json.dumps(manifest_content))):
            manager = SafeguardManager(manifest="/fake/path/manifest.json")
            assert manager.manifest == manifest_content

    def test_missing_required_fields(self) -> None:
        """Test validation fails when required fields are missing."""
        invalid_manifest = {"inter_agent_safeguards": {"agent_transitions": [{"message_src": "agent1"}]}}

        with pytest.raises(ValueError, match="missing required field"):
            SafeguardManager(manifest=invalid_manifest)

    def test_invalid_check_method(self) -> None:
        """Test validation fails with invalid check_method."""
        invalid_manifest = {
            "inter_agent_safeguards": {
                "agent_transitions": [
                    {"message_src": "agent1", "message_dst": "agent2", "check_method": "invalid_method"}
                ]
            }
        }

        with pytest.raises(ValueError, match="invalid check_method"):
            SafeguardManager(manifest=invalid_manifest)


class TestInvalidManifests:
    """Test invalid manifest validation."""

    def test_missing_check_method(self) -> None:
        """Test validation fails when check_method is missing."""
        invalid_manifest = {
            "inter_agent_safeguards": {
                "agent_transitions": [
                    {
                        "message_src": "agent1",
                        "message_dst": "agent2",
                        "pattern": r"\btest\b",
                        "violation_response": "block",
                    }
                ]
            }
        }

        with pytest.raises(ValueError, match="missing required field: check_method"):
            SafeguardManager(manifest=invalid_manifest)

    def test_missing_action_fields(self) -> None:
        """Test validation fails when action fields are missing."""
        invalid_manifest = {
            "inter_agent_safeguards": {
                "agent_transitions": [
                    {"message_src": "agent1", "message_dst": "agent2", "check_method": "regex", "pattern": r"\btest\b"}
                ]
            }
        }

        with pytest.raises(ValueError, match="missing required field: violation_response or action"):
            SafeguardManager(manifest=invalid_manifest)

    def test_invalid_check_method_pattern(self) -> None:
        """Test validation fails with old 'pattern' check_method."""
        invalid_manifest = {
            "agent_environment_safeguards": {
                "tool_interaction": [
                    {
                        "message_source": "agent1",
                        "message_destination": "tool1",
                        "check_method": "pattern",  # Should be 'regex' now
                        "pattern": r"\btest\b",
                        "action": "block",
                    }
                ]
            }
        }

        with pytest.raises(ValueError, match="invalid check_method: pattern"):
            SafeguardManager(manifest=invalid_manifest)

    def test_tool_interaction_missing_action(self) -> None:
        """Test tool_interaction fails without action field."""
        invalid_manifest = {
            "agent_environment_safeguards": {
                "tool_interaction": [
                    {
                        "message_source": "agent1",
                        "message_destination": "tool1",
                        "check_method": "regex",
                        "pattern": r"\btest\b",
                        # Missing action field
                    }
                ]
            }
        }

        with pytest.raises(ValueError, match="missing required field: violation_response or action"):
            SafeguardManager(manifest=invalid_manifest)

    def test_llm_interaction_missing_action(self) -> None:
        """Test llm_interaction fails without action field."""
        invalid_manifest = {
            "agent_environment_safeguards": {
                "llm_interaction": [
                    {
                        "pattern": r"\btest\b"
                        # Missing action field
                    }
                ]
            }
        }

        with pytest.raises(ValueError, match="missing required field: action"):
            SafeguardManager(manifest=invalid_manifest)

    def test_llm_check_method_missing_required_fields(self) -> None:
        """Test LLM check_method fails without required fields."""
        invalid_manifest = {
            "inter_agent_safeguards": {
                "agent_transitions": [
                    {
                        "message_src": "agent1",
                        "message_dst": "agent2",
                        "check_method": "llm",
                        "violation_response": "block",
                        # Missing custom_prompt or disallow_item
                    }
                ]
            }
        }

        with pytest.raises(ValueError, match="must have either 'custom_prompt' or 'disallow_item'"):
            SafeguardManager(manifest=invalid_manifest)

    def test_regex_check_method_missing_pattern(self) -> None:
        """Test regex check_method fails without pattern."""
        invalid_manifest = {
            "inter_agent_safeguards": {
                "agent_transitions": [
                    {
                        "message_src": "agent1",
                        "message_dst": "agent2",
                        "check_method": "regex",
                        "violation_response": "block",
                        # Missing pattern field
                    }
                ]
            }
        }

        with pytest.raises(ValueError, match="must have 'pattern'"):
            SafeguardManager(manifest=invalid_manifest)

    def test_invalid_regex_pattern(self) -> None:
        """Test validation fails with invalid regex pattern."""
        invalid_manifest = {
            "inter_agent_safeguards": {
                "agent_transitions": [
                    {
                        "message_src": "agent1",
                        "message_dst": "agent2",
                        "check_method": "regex",
                        "pattern": "[invalid regex(",  # Invalid regex
                        "violation_response": "block",
                    }
                ]
            }
        }

        with pytest.raises(ValueError, match="invalid regex pattern"):
            SafeguardManager(manifest=invalid_manifest)

    def test_tool_interaction_llm_missing_message_fields(self) -> None:
        """Test LLM tool interaction fails without message_source/message_destination."""
        invalid_manifest = {
            "agent_environment_safeguards": {
                "tool_interaction": [
                    {
                        "check_method": "llm",
                        "custom_prompt": "Check this",
                        "action": "block",
                        # Missing message_source/message_destination
                    }
                ]
            }
        }

        with pytest.raises(ValueError, match="must have 'message_source' and 'message_destination'"):
            SafeguardManager(manifest=invalid_manifest)

    def test_completely_invalid_tool_interaction_format(self) -> None:
        """Test tool interaction with completely wrong format fails early."""
        invalid_manifest = {
            "agent_environment_safeguards": {
                "tool_interaction": [
                    {
                        "some_random_field": "value",
                        "another_field": "value2",
                        # Missing check_method, action, any valid fields
                    }
                ]
            }
        }

        with pytest.raises(ValueError, match="missing required field: check_method"):
            SafeguardManager(manifest=invalid_manifest)


class TestSafeguardChecks:
    """Test safeguard checking functionality."""

    @pytest.fixture
    def regex_manager(self) -> SafeguardManager:
        """SafeguardManager with regex rule."""
        manifest = {
            "inter_agent_safeguards": {
                "agent_transitions": [
                    {
                        "message_src": "agent1",
                        "message_dst": "agent2",
                        "check_method": "regex",
                        "pattern": r"\bpassword\b",
                        "violation_response": "block",
                    }
                ]
            }
        }
        return SafeguardManager(manifest=manifest)

    def test_regex_block_violation(self, regex_manager: SafeguardManager) -> None:
        """Test regex rule blocks violating message."""
        result = regex_manager._check_inter_agent_communication("agent1", "agent2", "Please share your password")

        assert isinstance(result, dict)
        assert "blocked" in result.get("content", "").lower() or "password" not in result.get("content", "")

    def test_regex_pass_safe_message(self, regex_manager: SafeguardManager) -> None:
        """Test regex rule passes safe message."""
        message = "Hello, how are you?"
        result = regex_manager._check_inter_agent_communication("agent1", "agent2", message)

        assert result == message

    def test_llm_guardrail_creation(self) -> None:
        """Test LLM guardrail is created correctly."""
        manifest = {
            "inter_agent_safeguards": {
                "agent_transitions": [
                    {
                        "message_src": "agent1",
                        "message_dst": "agent2",
                        "check_method": "llm",
                        "custom_prompt": "Check content",
                        "violation_response": "block",
                    }
                ]
            }
        }

        mock_llm_config: LLMConfig = LLMConfig(model="test-model")
        with patch("autogen.agentchat.group.guardrails.OpenAIWrapper"):
            manager = SafeguardManager(manifest=manifest, safeguard_llm_config=mock_llm_config)

        assert len(manager.inter_agent_rules) == 1
        assert isinstance(manager.inter_agent_rules[0]["guardrail"], LLMGuardrail)

    def test_regex_guardrail_creation(self) -> None:
        """Test regex guardrail is created correctly."""
        manifest = {
            "inter_agent_safeguards": {
                "agent_transitions": [
                    {
                        "message_src": "agent1",
                        "message_dst": "agent2",
                        "check_method": "regex",
                        "pattern": r"\btest\b",
                        "violation_response": "mask",
                    }
                ]
            }
        }

        manager = SafeguardManager(manifest=manifest)
        assert len(manager.inter_agent_rules) == 1
        assert isinstance(manager.inter_agent_rules[0]["guardrail"], RegexGuardrail)


class TestApplySafeguards:
    """Test apply_safeguards integration."""

    @pytest.fixture
    def mock_agent(self) -> ConversableAgent:
        """Mock ConversableAgent."""
        agent = MagicMock()
        agent.name = "test_agent"
        agent.hook_lists = {
            "process_message_before_send": [],
            "process_tool_input": [],
            "process_tool_output": [],
            "process_llm_input": [],
            "process_llm_output": [],
            "process_human_input": [],
        }
        return agent

    def test_apply_safeguards_to_agents(self, mock_agent: Any) -> None:
        """Test applying safeguards to agents."""
        manifest = {
            "inter_agent_safeguards": {
                "agent_transitions": [
                    {
                        "message_src": "test_agent",
                        "message_dst": "*",
                        "check_method": "regex",
                        "pattern": r"\btest\b",
                        "violation_response": "block",
                    }
                ]
            }
        }

        safeguard_manager = apply_safeguards(agents=[mock_agent], manifest=manifest)

        assert isinstance(safeguard_manager, SafeguardManager)
        assert len(mock_agent.hook_lists["process_message_before_send"]) > 0

    def test_apply_safeguards_no_targets(self) -> None:
        """Test apply_safeguards fails without targets."""
        manifest: dict[str, Any] = {"inter_agent_safeguards": {}}

        with pytest.raises(ValueError, match="Either agents or groupchat_manager must be provided"):
            apply_safeguards(manifest=manifest)

    def test_invalid_agent_names(self, mock_agent: Any) -> None:
        """Test validation fails with invalid agent names."""
        manifest = {
            "inter_agent_safeguards": {
                "agent_transitions": [
                    {
                        "message_src": "unknown_agent",
                        "message_dst": "test_agent",
                        "check_method": "regex",
                        "pattern": r"test",
                        "violation_response": "block",
                    }
                ]
            }
        }

        with pytest.raises(ValueError, match="Agent name validation failed"):
            apply_safeguards(agents=[mock_agent], manifest=manifest)
