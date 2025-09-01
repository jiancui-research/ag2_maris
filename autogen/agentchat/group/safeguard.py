# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from termcolor import colored

from .guardrails import LLMGuardrail, RegexGuardrail
from .targets.transition_target import TransitionTarget

if TYPE_CHECKING:
    from ...llm_config import LLMConfig
    from ..agent import Agent
    from ..conversable_agent import ConversableAgent
    from ..groupchat import GroupChatManager


class SafeguardManager:
    """Main safeguard manager"""

    def __init__(
        self,
        manifest: dict[str, Any] | str,
        llm_config: LLMConfig | None = None,
        safeguard_llm_config: LLMConfig | None = None,
        mask_llm_config: LLMConfig | None = None,
    ):
        """Initialize the safeguard manager.

        Args:
            manifest: Safeguard manifest dict or path to JSON file
            llm_config: Default LLM configuration
            safeguard_llm_config: LLM configuration for safeguard checks
            mask_llm_config: LLM configuration for masking
        """
        self.manifest = self._load_manifest(manifest)
        self.llm_config = llm_config
        self.safeguard_llm_config = safeguard_llm_config or llm_config
        self.mask_llm_config = mask_llm_config or llm_config

        # Validate manifest format before proceeding
        self._validate_manifest()

        # Create mask agent for content masking
        if self.mask_llm_config:
            from ..conversable_agent import ConversableAgent

            self.mask_agent = ConversableAgent(
                name="mask_agent",
                system_message="You are a agent responsible for masking sensitive information.",
                llm_config=self.mask_llm_config,
                human_input_mode="NEVER",
                max_consecutive_auto_reply=1,
            )

        # Parse safeguard rules
        self.inter_agent_rules = self._parse_inter_agent_rules()
        self.environment_rules = self._parse_environment_rules()

        print(
            colored(
                f"✅ Loaded {len(self.inter_agent_rules)} inter-agent and {len(self.environment_rules)} environment safeguard rules",
                "green",
            )
        )

    def _load_manifest(self, manifest: dict[str, Any] | str) -> dict[str, Any]:
        """Load manifest from file or use provided dict."""
        if isinstance(manifest, str):
            with open(manifest) as f:
                result: dict[str, Any] = json.load(f)
                return result
        return manifest

    def _validate_manifest(self) -> None:
        """Validate manifest format and syntax."""
        if not isinstance(self.manifest, dict):
            raise ValueError("Manifest must be a dictionary")

        # Validate inter-agent safeguards
        if "inter_agent_safeguards" in self.manifest:
            self._validate_inter_agent_safeguards()

        # Validate environment safeguards
        if "agent_environment_safeguards" in self.manifest:
            self._validate_environment_safeguards()

    def _validate_inter_agent_safeguards(self) -> None:
        """Validate inter-agent safeguards section."""
        inter_agent = self.manifest["inter_agent_safeguards"]
        if not isinstance(inter_agent, dict):
            raise ValueError("inter_agent_safeguards must be a dictionary")

        # Validate agent_transitions
        if "agent_transitions" in inter_agent:
            if not isinstance(inter_agent["agent_transitions"], list):
                raise ValueError("agent_transitions must be a list")

            for i, rule in enumerate(inter_agent["agent_transitions"]):
                if not isinstance(rule, dict):
                    raise ValueError(f"agent_transitions[{i}] must be a dictionary")

                # Required fields
                required_fields = ["message_src", "message_dst"]
                for field in required_fields:
                    if field not in rule:
                        raise ValueError(f"agent_transitions[{i}] missing required field: {field}")

                # Check method validation - no default, must be explicit
                if "check_method" not in rule:
                    raise ValueError(f"agent_transitions[{i}] missing required field: check_method")
                check_method = rule["check_method"]
                if check_method not in ["llm", "regex"]:
                    raise ValueError(
                        f"agent_transitions[{i}] invalid check_method: {check_method}. Must be 'llm' or 'regex'"
                    )

                # LLM-specific validation
                if check_method == "llm":
                    if "custom_prompt" not in rule and "disallow_item" not in rule:
                        raise ValueError(
                            f"agent_transitions[{i}] with check_method 'llm' must have either 'custom_prompt' or 'disallow_item'"
                        )
                    if "disallow_item" in rule and not isinstance(rule["disallow_item"], list):
                        raise ValueError(f"agent_transitions[{i}] disallow_item must be a list")

                # Regex-specific validation
                if check_method == "regex":
                    if "pattern" not in rule:
                        raise ValueError(f"agent_transitions[{i}] with check_method 'regex' must have 'pattern'")
                    if not isinstance(rule["pattern"], str):
                        raise ValueError(f"agent_transitions[{i}] pattern must be a string")

                    # Test regex pattern validity
                    try:
                        re.compile(rule["pattern"])
                    except re.error as e:
                        raise ValueError(f"agent_transitions[{i}] invalid regex pattern '{rule['pattern']}': {e}")

                # Validate action - no default, must be explicit
                if "violation_response" not in rule and "action" not in rule:
                    raise ValueError(f"agent_transitions[{i}] missing required field: violation_response or action")
                action = rule.get("violation_response", rule.get("action"))
                if action not in ["block", "mask", "warning"]:
                    raise ValueError(
                        f"agent_transitions[{i}] invalid action: {action}. Must be 'block', 'mask', or 'warning'"
                    )

        # Validate groupchat_message_check
        if "groupchat_message_check" in inter_agent:
            rule = inter_agent["groupchat_message_check"]
            if not isinstance(rule, dict):
                raise ValueError("groupchat_message_check must be a dictionary")
            if "disallow_item" in rule and not isinstance(rule["disallow_item"], list):
                raise ValueError("groupchat_message_check disallow_item must be a list")

    def _validate_environment_safeguards(self) -> None:
        """Validate environment safeguards section."""
        env_rules = self.manifest["agent_environment_safeguards"]
        if not isinstance(env_rules, dict):
            raise ValueError("agent_environment_safeguards must be a dictionary")

        # Validate tool_interaction rules
        if "tool_interaction" in env_rules:
            if not isinstance(env_rules["tool_interaction"], list):
                raise ValueError("tool_interaction must be a list")

            for i, rule in enumerate(env_rules["tool_interaction"]):
                if not isinstance(rule, dict):
                    raise ValueError(f"tool_interaction[{i}] must be a dictionary")

                # Check method validation - no default, must be explicit
                if "check_method" not in rule:
                    raise ValueError(f"tool_interaction[{i}] missing required field: check_method")
                check_method = rule["check_method"]
                if check_method not in ["llm", "regex"]:
                    raise ValueError(
                        f"tool_interaction[{i}] invalid check_method: {check_method}. Must be 'llm' or 'regex'"
                    )

                # Validate action - no default, must be explicit
                if "violation_response" not in rule and "action" not in rule:
                    raise ValueError(f"tool_interaction[{i}] missing required field: violation_response or action")
                action = rule.get("violation_response", rule.get("action"))
                if action not in ["block", "mask", "warning"]:
                    raise ValueError(
                        f"tool_interaction[{i}] invalid action: {action}. Must be 'block', 'mask', or 'warning'"
                    )

                if check_method == "llm":
                    # Only support message_source/message_destination format
                    if "message_source" not in rule or "message_destination" not in rule:
                        raise ValueError(
                            f"tool_interaction[{i}] with check_method 'llm' must have 'message_source' and 'message_destination'"
                        )
                    if "custom_prompt" not in rule and "disallow_item" not in rule:
                        raise ValueError(
                            f"tool_interaction[{i}] with check_method 'llm' must have either 'custom_prompt' or 'disallow_item'"
                        )

                elif "pattern" in rule:
                    if not isinstance(rule["pattern"], str):
                        raise ValueError(f"tool_interaction[{i}] pattern must be a string")
                    # Test regex pattern validity
                    try:
                        re.compile(rule["pattern"])
                    except re.error as e:
                        raise ValueError(f"tool_interaction[{i}] invalid regex pattern '{rule['pattern']}': {e}")

        # Validate llm_interaction rules
        if "llm_interaction" in env_rules:
            if not isinstance(env_rules["llm_interaction"], list):
                raise ValueError("llm_interaction must be a list")

            for i, rule in enumerate(env_rules["llm_interaction"]):
                if not isinstance(rule, dict):
                    raise ValueError(f"llm_interaction[{i}] must be a dictionary")

                # Validate action - no default, must be explicit
                if "action" not in rule:
                    raise ValueError(f"llm_interaction[{i}] missing required field: action")
                action = rule["action"]
                if action not in ["block", "mask", "warning"]:
                    raise ValueError(
                        f"llm_interaction[{i}] invalid action: {action}. Must be 'block', 'mask', or 'warning'"
                    )

                if "pattern" in rule:
                    if not isinstance(rule["pattern"], str):
                        raise ValueError(f"llm_interaction[{i}] pattern must be a string")
                    # Test regex pattern validity
                    try:
                        re.compile(rule["pattern"])
                    except re.error as e:
                        raise ValueError(f"llm_interaction[{i}] invalid regex pattern '{rule['pattern']}': {e}")

        # Validate user_interaction rules
        if "user_interaction" in env_rules:
            if not isinstance(env_rules["user_interaction"], list):
                raise ValueError("user_interaction must be a list")

            for i, rule in enumerate(env_rules["user_interaction"]):
                if not isinstance(rule, dict):
                    raise ValueError(f"user_interaction[{i}] must be a dictionary")
                if "agent" not in rule:
                    raise ValueError(f"user_interaction[{i}] missing required field: agent")
                if "user_input" in rule:
                    user_input = rule["user_input"]
                    if not isinstance(user_input, dict):
                        raise ValueError(f"user_interaction[{i}] user_input must be a dictionary")
                    if "disallow_item" in user_input and not isinstance(user_input["disallow_item"], list):
                        raise ValueError(f"user_interaction[{i}] user_input disallow_item must be a list")

    def _validate_agent_names(self, agent_names: list[str]) -> None:
        """Validate that agent names referenced in manifest actually exist."""
        available_agents = set(agent_names)

        # Check inter-agent safeguards
        if "inter_agent_safeguards" in self.manifest:
            inter_agent = self.manifest["inter_agent_safeguards"]

            # Check agent_transitions
            for i, rule in enumerate(inter_agent.get("agent_transitions", [])):
                src_agent = rule.get("message_src")
                dst_agent = rule.get("message_dst")

                # Skip wildcard patterns
                if src_agent != "*" and src_agent not in available_agents:
                    raise ValueError(
                        f"agent_transitions[{i}] references unknown source agent: '{src_agent}'. Available agents: {sorted(available_agents)}"
                    )

                if dst_agent != "*" and dst_agent not in available_agents:
                    raise ValueError(
                        f"agent_transitions[{i}] references unknown destination agent: '{dst_agent}'. Available agents: {sorted(available_agents)}"
                    )

        # Check environment safeguards
        if "agent_environment_safeguards" in self.manifest:
            env_rules = self.manifest["agent_environment_safeguards"]

            # Check tool_interaction rules - only support message_src/message_dst format
            for i, rule in enumerate(env_rules.get("tool_interaction", [])):
                # Only validate message_src/message_dst format
                if (
                    "message_src" in rule
                    and "message_dst" in rule
                    or "message_source" in rule
                    and "message_destination" in rule
                ):
                    # Skip detailed validation since we can't distinguish agent vs tool names
                    pass
                elif "pattern" in rule and "message_source" not in rule and "message_src" not in rule:
                    # Simple pattern rules are allowed
                    pass
                else:
                    raise ValueError(
                        f"tool_interaction[{i}] must use either (message_src, message_dst), (message_source, message_destination), or pattern-only format"
                    )

            # Check llm_interaction rules
            for i, rule in enumerate(env_rules.get("llm_interaction", [])):
                # New format
                if "message_source" in rule and "message_destination" in rule:
                    src = rule["message_source"]
                    dst = rule["message_destination"]

                    # Check agent references (LLM interactions have agent <-> llm)
                    if src != "llm" and src.lower() != "llm" and src not in available_agents:
                        raise ValueError(
                            f"llm_interaction[{i}] references unknown agent: '{src}'. Available agents: {sorted(available_agents)}"
                        )
                    if dst != "llm" and dst.lower() != "llm" and dst not in available_agents:
                        raise ValueError(
                            f"llm_interaction[{i}] references unknown agent: '{dst}'. Available agents: {sorted(available_agents)}"
                        )

                # Legacy format
                elif "message_src" in rule:
                    agent_name = rule["message_src"]
                    if agent_name not in available_agents:
                        raise ValueError(
                            f"llm_interaction[{i}] references unknown agent: '{agent_name}'. Available agents: {sorted(available_agents)}"
                        )

                elif "agent_name" in rule:
                    agent_name = rule["agent_name"]
                    if agent_name not in available_agents:
                        raise ValueError(
                            f"llm_interaction[{i}] references unknown agent: '{agent_name}'. Available agents: {sorted(available_agents)}"
                        )

            # Check user_interaction rules
            for i, rule in enumerate(env_rules.get("user_interaction", [])):
                agent_name = rule.get("agent")
                if agent_name and agent_name not in available_agents:
                    raise ValueError(
                        f"user_interaction[{i}] references unknown agent: '{agent_name}'. Available agents: {sorted(available_agents)}"
                    )

    def _validate_tool_names(self, tool_names: list[str]) -> None:  # pylint: disable=unused-argument
        """Validate that tool names referenced in manifest actually exist."""
        # Skip tool name validation since message_src/message_dst format
        # doesn't explicitly separate tool names and we can't reliably distinguish them
        pass

    def _parse_inter_agent_rules(self) -> list[dict[str, Any]]:
        """Parse inter-agent safeguard rules from manifest."""
        rules = []
        inter_agent = self.manifest.get("inter_agent_safeguards", {})

        # Agent transitions
        for rule in inter_agent.get("agent_transitions", []):
            # Create guardrail based on check_method
            check_method = rule.get("check_method", "regex")  # Default to regex for backward compatibility
            guardrail: LLMGuardrail | RegexGuardrail | None = None
            action = rule.get("violation_response", rule.get("action", "block"))  # Support both field names

            if check_method == "llm":
                if not self.safeguard_llm_config:
                    raise ValueError(
                        f"safeguard_llm_config is required for LLM-based guardrail: {rule['message_src']} -> {rule['message_dst']}"
                    )

                # Handle different LLM check types
                if "custom_prompt" in rule:
                    # Custom prompt for LLM guardrail
                    condition = rule["custom_prompt"]
                    print(
                        colored(
                            f"📝 Using custom prompt for LLM guardrail: {rule['message_src']} -> {rule['message_dst']}",
                            "blue",
                        )
                    )
                elif "disallow_item" in rule:
                    # Disallow items for LLM guardrail
                    condition = f"Check if this content contains any of these disallowed categories: {', '.join(rule['disallow_item'])}"
                    print(colored(f"🚫 Using disallow items for LLM guardrail: {rule['disallow_item']}", "blue"))
                else:
                    raise ValueError(
                        f"Either custom_prompt or disallow_item must be provided for LLM guardrail: {rule['message_src']} -> {rule['message_dst']}"
                    )

                # Create LLM guardrail - handle dict config
                llm_config = self.safeguard_llm_config
                if isinstance(llm_config, dict):

                    class SimpleLLMConfig:
                        def __init__(self, config_dict):
                            self._config = config_dict

                        def deepcopy(self):
                            import copy

                            return SimpleLLMConfig(copy.deepcopy(self._config))

                        def model_dump(self):
                            return self._config

                        def __setattr__(self, name, value):
                            if name.startswith("_"):
                                super().__setattr__(name, value)
                            else:
                                self._config[name] = value

                    llm_config = SimpleLLMConfig(llm_config)

                guardrail = LLMGuardrail(
                    name=f"llm_guard_{rule['message_src']}_{rule['message_dst']}",
                    condition=condition,
                    target=TransitionTarget(),
                    llm_config=llm_config,
                    activation_message=rule.get("activation_message", "LLM detected violation"),
                )

            elif check_method == "regex":
                if "pattern" in rule:
                    # Regex pattern guardrail
                    guardrail = RegexGuardrail(
                        name=f"regex_guard_{rule['message_src']}_{rule['message_dst']}",
                        condition=rule["pattern"],
                        target=TransitionTarget(),
                        activation_message=rule.get("activation_message", "Regex pattern matched"),
                    )
                    print(colored(f"🔍 Using regex pattern for guardrail: {rule['pattern']}", "blue"))

            # Add rule with guardrail
            parsed_rule = {
                "type": "agent_transition",
                "source": rule["message_src"],
                "target": rule["message_dst"],
                "action": action,
                "guardrail": guardrail,
                "activation_message": rule.get("activation_message", "Content blocked by safeguard"),
            }

            # Keep legacy fields for backward compatibility
            if "disallow_item" in rule:
                parsed_rule["disallow"] = rule["disallow_item"]
            if "pattern" in rule:
                parsed_rule["pattern"] = rule["pattern"]
            if "custom_prompt" in rule:
                parsed_rule["custom_prompt"] = rule["custom_prompt"]

            rules.append(parsed_rule)

        # Groupchat message check
        if "groupchat_message_check" in inter_agent:
            rule = inter_agent["groupchat_message_check"]
            rules.append({
                "type": "groupchat_message",
                "source": "*",
                "target": "*",
                "action": rule.get("pet_action", "block"),
                "disallow": rule.get("disallow_item", []),
            })

        return rules

    def _parse_environment_rules(self) -> list[dict[str, Any]]:
        """Parse agent-environment safeguard rules from manifest."""
        rules = []
        env_rules = self.manifest.get("agent_environment_safeguards", {})

        # Tool interaction rules - handle regex and LLM
        for rule in env_rules.get("tool_interaction", []):
            check_method = rule.get("check_method", "regex")  # Default to regex for backward compatibility
            action = rule.get("violation_response", rule.get("action", "block"))

            if check_method == "llm":
                # LLM-based tool interaction rule - requires message_source/message_destination
                if "message_source" not in rule or "message_destination" not in rule:
                    raise ValueError(
                        "tool_interaction with check_method 'llm' must have 'message_source' and 'message_destination'"
                    )

                parsed_rule = {
                    "type": "tool_interaction",
                    "message_source": rule["message_source"],
                    "message_destination": rule["message_destination"],
                    "check_method": "llm",
                    "action": action,
                    "message": rule.get("activation_message", rule.get("message", "LLM blocked tool output")),
                }

                # Add LLM-specific parameters
                if "custom_prompt" in rule:
                    parsed_rule["custom_prompt"] = rule["custom_prompt"]
                    print(
                        colored(
                            f"📝 Using custom prompt for tool interaction LLM: {rule['message_source']} -> {rule['message_destination']}",
                            "blue",
                        )
                    )
                elif "disallow_item" in rule:
                    parsed_rule["disallow"] = rule["disallow_item"]
                    print(colored(f"🚫 Using disallow items for tool interaction LLM: {rule['disallow_item']}", "blue"))

                rules.append(parsed_rule)

            elif check_method == "regex" and "pattern" in rule:
                # Regex pattern-based rule
                if "message_source" in rule and "message_destination" in rule:
                    # Format with message_source, message_destination, pattern, action, message
                    rules.append({
                        "type": "tool_interaction",
                        "message_source": rule["message_source"],
                        "message_destination": rule["message_destination"],
                        "pattern": rule["pattern"],
                        "action": action,
                        "message": rule.get("message", "Content blocked by safeguard"),
                    })
                else:
                    # Simple format with just pattern matching
                    rules.append({
                        "type": "tool_interaction",
                        "pattern": rule["pattern"],
                        "action": action,
                        "message": rule.get("message", "Content blocked by safeguard"),
                    })
            else:
                raise ValueError(
                    "tool_interaction rule must have check_method 'llm' or 'regex' with appropriate parameters"
                )

        # LLM interaction rules
        for rule in env_rules.get("llm_interaction", []):
            if "pattern" in rule:
                # Check if it has message_source/message_destination
                if "message_source" in rule and "message_destination" in rule:
                    # Format with message_source, message_destination, pattern, action, message
                    rules.append({
                        "type": "llm_interaction",
                        "message_source": rule["message_source"],
                        "message_destination": rule["message_destination"],
                        "pattern": rule["pattern"],
                        "action": rule["action"],
                        "message": rule.get("message", "Content blocked by safeguard"),
                    })
                else:
                    # Simple format with just pattern matching
                    rules.append({
                        "type": "llm_interaction",
                        "pattern": rule["pattern"],
                        "action": rule["action"],
                        "message": rule.get("message", "Content blocked by safeguard"),
                    })
            else:
                raise ValueError(
                    "llm_interaction rule must have 'pattern' field and optionally message_source/message_destination"
                )

        # User interaction rules - only support message_source/message_destination format
        for rule in env_rules.get("user_interaction", []):
            if "message_source" in rule and "message_destination" in rule:
                rules.append({
                    "type": "user_interaction",
                    "message_source": rule["message_source"],
                    "message_destination": rule["message_destination"],
                    "action": rule.get("action", "block"),
                    "disallow": rule.get("disallow_item", []),
                })
            else:
                raise ValueError("user_interaction rule must have 'message_source' and 'message_destination' fields")

        return rules

    def create_agent_hooks(self, agent_name: str) -> dict[str, Callable[..., Any]]:
        """Create hook functions for a specific agent, only for rule types that exist."""
        hooks = {}

        # Check if we have any tool interaction rules that apply to this agent
        agent_tool_rules = [
            rule
            for rule in self.environment_rules
            if rule["type"] == "tool_interaction"
            and (
                rule.get("message_destination") == agent_name
                or rule.get("message_source") == agent_name
                or rule.get("agent_name") == agent_name
                or "message_destination" not in rule
            )
        ]  # Simple pattern rules apply to all

        if agent_tool_rules:

            def tool_input_hook(tool_input: dict[str, Any]) -> dict[str, Any] | None:
                result = self._check_tool_interaction(agent_name, tool_input, "input")
                return result if result is not None else tool_input

            def tool_output_hook(tool_input: dict[str, Any]) -> dict[str, Any] | None:
                result = self._check_tool_interaction(agent_name, tool_input, "output")
                return result if result is not None else tool_input

            hooks["process_tool_input"] = tool_input_hook
            hooks["process_tool_output"] = tool_output_hook

        # Check if we have any LLM interaction rules that apply to this agent
        agent_llm_rules = [
            rule
            for rule in self.environment_rules
            if rule["type"] == "llm_interaction"
            and (
                rule.get("message_destination") == agent_name
                or rule.get("message_source") == agent_name
                or rule.get("agent_name") == agent_name
                or "message_destination" not in rule
            )
        ]  # Simple pattern rules apply to all

        if agent_llm_rules:

            def llm_input_hook(tool_input: dict[str, Any]) -> dict[str, Any] | None:
                # Extract messages from the data structure if needed
                messages = tool_input if isinstance(tool_input, list) else tool_input.get("messages", tool_input)
                result = self._check_llm_interaction(agent_name, messages, "input")
                if isinstance(result, list) and isinstance(tool_input, dict) and "messages" in tool_input:
                    return {**tool_input, "messages": result}
                elif isinstance(result, dict):
                    return result
                elif result is not None and not isinstance(result, dict):
                    # Convert string or other types to dict format
                    return {"content": str(result), "role": "function"}
                elif result is not None and isinstance(result, dict) and result != tool_input:
                    # Return the modified dict result
                    return result
                return tool_input

            def llm_output_hook(tool_input: dict[str, Any]) -> dict[str, Any] | None:
                result = self._check_llm_interaction(agent_name, tool_input, "output")
                if isinstance(result, dict):
                    return result
                elif result is not None and not isinstance(result, dict):
                    # Convert string or other types to dict format
                    return {"content": str(result), "role": "function"}
                elif result is not None and isinstance(result, dict) and result != tool_input:
                    # Return the modified dict result
                    return result
                return tool_input

            hooks["process_llm_input"] = llm_input_hook
            hooks["process_llm_output"] = llm_output_hook

        # Check if we have any user interaction rules that apply to this agent
        agent_user_rules = [
            rule
            for rule in self.environment_rules
            if rule["type"] == "user_interaction" and rule.get("message_destination") == agent_name
        ]

        if agent_user_rules:

            def human_input_hook(tool_input: dict[str, Any]) -> dict[str, Any] | None:
                # Extract human input from data structure
                human_input = tool_input.get("content", str(tool_input))
                result = self._check_user_interaction(agent_name, human_input)
                if result != human_input and isinstance(tool_input, dict):
                    return {**tool_input, "content": result}
                return tool_input if result == human_input else {"content": result}

            hooks["process_human_input"] = human_input_hook

        # Check if we have any inter-agent rules that apply to this agent
        agent_inter_rules = [
            rule
            for rule in self.inter_agent_rules
            if (
                rule.get("source") == agent_name
                or rule.get("target") == agent_name
                or rule.get("source") == "*"
                or rule.get("target") == "*"
            )
        ]

        if agent_inter_rules:

            def message_before_send_hook(
                sender: Any, message: dict[str, Any] | str, recipient: Any, silent: Any = None
            ) -> dict[str, Any] | str:
                _ = silent  # Unused parameter
                result = self._check_inter_agent_communication(sender.name, recipient.name, message)
                return result

            hooks["process_message_before_send"] = message_before_send_hook  # type: ignore[assignment]

        return hooks

    def _check_safeguard_condition(
        self, content: str, disallow_items: list[str] | None = None, custom_prompt: str | None = None
    ) -> tuple[bool, str]:
        """Check if content violates safeguard conditions using LLMGuardrail."""
        if not self.safeguard_llm_config:
            raise ValueError("safeguard_llm_config is required for LLM-based safeguard checks")

        # Determine condition based on available parameters
        if custom_prompt:
            condition = custom_prompt
        elif disallow_items:
            condition = (
                f"Check if this content contains any of these disallowed categories: {', '.join(disallow_items)}"
            )
        else:
            raise ValueError("Either custom_prompt or disallow_items must be provided")

        # Create LLM guardrail for checking
        # Handle dict config by wrapping it
        llm_config = self.safeguard_llm_config
        if isinstance(llm_config, dict):

            class SimpleLLMConfig:
                def __init__(self, config_dict):
                    self._config = config_dict

                def deepcopy(self):
                    import copy

                    return SimpleLLMConfig(copy.deepcopy(self._config))

                def model_dump(self):
                    return self._config

                def __setattr__(self, name, value):
                    if name.startswith("_"):
                        super().__setattr__(name, value)
                    else:
                        self._config[name] = value

            llm_config = SimpleLLMConfig(llm_config)

        from .targets.transition_target import TransitionTarget

        guardrail = LLMGuardrail(
            name="temp_safeguard_check",
            condition=condition,
            target=TransitionTarget(),
            llm_config=llm_config,
            activation_message="Content violates safeguard conditions",
        )

        try:
            result = guardrail.check(content)
            return result.activated, result.justification
        except Exception as e:
            raise RuntimeError(f"Safeguard check failed: {e}")

    def _check_pattern_condition(self, content: str, pattern: str) -> tuple[bool, str]:
        """Check if content matches a regex pattern."""
        try:
            if re.search(pattern, content, re.IGNORECASE):
                return True, f"Content matched pattern: {pattern}"
        except re.error as e:
            print(colored(f"Invalid regex pattern '{pattern}': {e}", "red"))

        return False, "No pattern match"

    def _apply_action(
        self,
        action: str,
        content: str | dict[str, Any] | list[Any],
        disallow_items: list[str],
        explanation: str,
        custom_message: str | None = None,
        pattern: str | None = None,
    ) -> str | dict[str, Any] | list[Any]:
        """Apply the specified action to content."""
        message = custom_message or explanation

        if action == "block":
            print(colored(f"🚨 BLOCKED: {message}", "red"))
            return self._handle_blocked_content(content, message)
        elif action == "mask":
            print(colored(f"🎭 MASKED: {message}", "yellow"))

            def mask_func(text: str) -> str:
                return self._mask_content(text, disallow_items, explanation, pattern)

            return self._handle_masked_content(content, mask_func)
        elif action == "warning":
            print(colored(f"⚠️ WARNING: {message}", "yellow"))
            return content
        else:
            return content

    def _mask_content(
        self, content: str, disallow_items: list[str], explanation: str, pattern: str | None = None
    ) -> str:
        """Mask sensitive content using LLM, pattern-based, or simple replacement."""
        # If we have a specific pattern from a regex guardrail, use it first
        if pattern:
            try:
                masked = re.sub(pattern, "[SENSITIVE_INFO]", content, flags=re.IGNORECASE)
                if masked != content:  # Only return if something was actually masked
                    return masked
            except re.error as e:
                print(colored(f"Pattern masking failed: {e}", "red"))

        # Try LLM-based masking if available
        if self.mask_agent and disallow_items:
            mask_prompt = f"""
            Mask the sensitive information in this content with [SENSITIVE_INFO]:

            Content: {content}
            Sensitive categories: {", ".join(disallow_items)}
            Reason: {explanation}

            Return only the masked content, nothing else.
            """

            try:
                response = self.mask_agent.generate_oai_reply(messages=[{"role": "user", "content": mask_prompt}])

                if response[0] and response[1]:
                    masked = response[1].get("content", content) if isinstance(response[1], dict) else str(response[1])
                    return masked
            except Exception as e:
                print(colored(f"LLM masking failed: {e}", "red"))

        return masked

    def _handle_blocked_content(
        self, content: str | dict[str, Any] | list[Any], block_message: str
    ) -> str | dict[str, Any] | list[Any]:
        """Handle blocked content based on its structure."""
        block_msg = f"🛡️ BLOCKED: {block_message}"

        if isinstance(content, dict):
            blocked_content = content.copy()

            # Handle tool_responses (like in tool outputs)
            if "tool_responses" in blocked_content and blocked_content["tool_responses"]:
                blocked_content["content"] = block_msg
                blocked_content["tool_responses"] = [
                    {**response, "content": block_msg} for response in blocked_content["tool_responses"]
                ]
            # Handle tool_calls (like in tool inputs)
            elif "tool_calls" in blocked_content and blocked_content["tool_calls"]:
                blocked_content["tool_calls"] = [
                    {**tool_call, "function": {**tool_call["function"], "arguments": block_msg}}
                    for tool_call in blocked_content["tool_calls"]
                ]
            # Handle regular content
            elif "content" in blocked_content:
                blocked_content["content"] = block_msg
            # Handle arguments (for some tool formats)
            elif "arguments" in blocked_content:
                blocked_content["arguments"] = block_msg
            else:
                # Default case - add content field
                blocked_content["content"] = block_msg

            return blocked_content

        elif isinstance(content, list):
            # Handle list of messages (like LLM inputs)
            blocked_list = []
            for item in content:
                if isinstance(item, dict):
                    blocked_item = item.copy()
                    if "content" in blocked_item:
                        blocked_item["content"] = block_msg
                    if "tool_calls" in blocked_item:
                        blocked_item["tool_calls"] = [
                            {**tool_call, "function": {**tool_call["function"], "arguments": block_msg}}
                            for tool_call in blocked_item["tool_calls"]
                        ]
                    if "tool_responses" in blocked_item:
                        blocked_item["tool_responses"] = [
                            {**response, "content": block_msg} for response in blocked_item["tool_responses"]
                        ]
                    blocked_list.append(blocked_item)
                else:
                    blocked_list.append({"content": block_msg, "role": "function"})
            return blocked_list

        else:
            # String or other content - return as function message
            return {"content": block_msg, "role": "function"}

    def _handle_masked_content(
        self, content: str | dict[str, Any] | list[Any], mask_func: Callable[[str], str]
    ) -> str | dict[str, Any] | list[Any]:
        """Handle masked content based on its structure."""
        if isinstance(content, dict):
            masked_content = content.copy()

            # Handle tool_responses
            if "tool_responses" in masked_content and masked_content["tool_responses"]:
                if "content" in masked_content:
                    masked_content["content"] = mask_func(str(masked_content["content"]))
                masked_content["tool_responses"] = [
                    {**response, "content": mask_func(str(response.get("content", "")))}
                    for response in masked_content["tool_responses"]
                ]
            # Handle tool_calls
            elif "tool_calls" in masked_content and masked_content["tool_calls"]:
                masked_content["tool_calls"] = [
                    {
                        **tool_call,
                        "function": {
                            **tool_call["function"],
                            "arguments": mask_func(str(tool_call["function"].get("arguments", ""))),
                        },
                    }
                    for tool_call in masked_content["tool_calls"]
                ]
            # Handle regular content
            elif "content" in masked_content:
                masked_content["content"] = mask_func(str(masked_content["content"]))
            # Handle arguments
            elif "arguments" in masked_content:
                masked_content["arguments"] = mask_func(str(masked_content["arguments"]))

            return masked_content

        elif isinstance(content, list):
            # Handle list of messages
            masked_list = []
            for item in content:
                if isinstance(item, dict):
                    masked_item = item.copy()
                    if "content" in masked_item:
                        masked_item["content"] = mask_func(str(masked_item["content"]))
                    if "tool_calls" in masked_item:
                        masked_item["tool_calls"] = [
                            {
                                **tool_call,
                                "function": {
                                    **tool_call["function"],
                                    "arguments": mask_func(str(tool_call["function"].get("arguments", ""))),
                                },
                            }
                            for tool_call in masked_item["tool_calls"]
                        ]
                    if "tool_responses" in masked_item:
                        masked_item["tool_responses"] = [
                            {**response, "content": mask_func(str(response.get("content", "")))}
                            for response in masked_item["tool_responses"]
                        ]
                    masked_list.append(masked_item)
                else:
                    # For non-dict items, wrap the masked content in a dict
                    masked_item_content: str = mask_func(str(item))
                    masked_list.append({"content": masked_item_content, "role": "function"})
            return masked_list

        else:
            # String content
            return mask_func(str(content))

    def _check_inter_agent_communication(
        self, sender_name: str, recipient_name: str, message: str | dict[str, Any]
    ) -> str | dict[str, Any]:
        """Check inter-agent communication."""
        content = message.get("content", "") if isinstance(message, dict) else str(message)

        for rule in self.inter_agent_rules:
            if rule["type"] == "agent_transition":
                # Check if this rule applies
                source_match = rule["source"] == "*" or rule["source"] == sender_name
                target_match = rule["target"] == "*" or rule["target"] == recipient_name

                if source_match and target_match:
                    # Log the detection check
                    print(colored("\n🔍 Checking inter-agent communication:", "cyan"))
                    print(colored(f"  • From: {sender_name}", "cyan"))
                    print(colored(f"  • To: {recipient_name}", "cyan"))
                    print(colored(f"  • Rule action: {rule.get('action', 'N/A')}", "cyan"))
                    print(colored(f"  • Content: {content[:100]}{'...' if len(content) > 100 else ''}", "cyan"))

                    # Use guardrail if available
                    if "guardrail" in rule and rule["guardrail"]:
                        print(colored(f"  • Checking with guardrail: {type(rule['guardrail']).__name__}", "blue"))
                        try:
                            result = rule["guardrail"].check(content)
                            if result.activated:
                                print(colored(f"  • 🚨 VIOLATION DETECTED: {result.justification}", "red"))
                                # Pass the pattern if it's a regex guardrail
                                pattern = rule.get("pattern") if isinstance(rule["guardrail"], RegexGuardrail) else None
                                action_result = self._apply_action(
                                    rule["action"],
                                    message,
                                    [],
                                    result.justification,
                                    rule.get("activation_message", result.justification),
                                    pattern,
                                )
                                if isinstance(action_result, (str, dict)):
                                    return action_result
                                else:
                                    return message
                            else:
                                print(colored("  • ✅ Content passed guardrail check", "green"))
                        except Exception as e:
                            print(colored(f"  • ❌ Guardrail check failed: {e}", "red"))

                    # Handle legacy pattern-based rules
                    elif "pattern" in rule and rule["pattern"]:
                        print(colored(f"  • Checking with pattern: {rule['pattern']}", "blue"))
                        is_violation, explanation = self._check_pattern_condition(content, rule["pattern"])
                        if is_violation:
                            print(colored(f"  • 🚨 PATTERN VIOLATION: {explanation}", "red"))
                            result_value = self._apply_action(
                                rule["action"],
                                message,
                                [],
                                explanation,
                                rule.get("activation_message"),
                                rule["pattern"],
                            )
                            if isinstance(result_value, (str, dict)):
                                return result_value
                            else:
                                return message
                        else:
                            print(colored("  • ✅ Content passed pattern check", "green"))

                    # Handle legacy disallow-based rules and custom prompts
                    elif "disallow" in rule or "custom_prompt" in rule:
                        if "custom_prompt" in rule:
                            print(colored("  • Checking with custom prompt", "blue"))
                            is_violation, explanation = self._check_safeguard_condition(
                                content, custom_prompt=rule["custom_prompt"]
                            )
                        else:
                            print(colored(f"  • Checking with disallow items: {rule['disallow']}", "blue"))
                            is_violation, explanation = self._check_safeguard_condition(
                                content, disallow_items=rule["disallow"]
                            )

                        if is_violation:
                            violation_type = "CUSTOM PROMPT" if "custom_prompt" in rule else "DISALLOW"
                            print(colored(f"  • 🚨 {violation_type} VIOLATION: {explanation}", "red"))
                            result_value = self._apply_action(
                                rule["action"], message, rule.get("disallow", []), explanation
                            )
                            if isinstance(result_value, (str, dict)):
                                return result_value
                            else:
                                return message
                        else:
                            print(colored("  • ✅ Content passed check", "green"))

        return message

    def _check_tool_interaction(self, agent_name: str, data: dict[str, Any], direction: str) -> dict[str, Any]:
        """Check tool interactions."""
        for rule in self.environment_rules:
            if rule["type"] == "tool_interaction":
                # Handle format with message_source/message_destination
                if "message_source" in rule and "message_destination" in rule:
                    # Extract tool name from data
                    tool_name = data.get("name", data.get("tool_name", ""))

                    # Check if this rule applies to the current interaction
                    # For tool output: message_source = tool, message_destination = agent
                    # For tool input: message_source = agent, message_destination = tool
                    rule_applies = False
                    if direction == "output":
                        rule_applies = rule["message_source"] == tool_name and rule["message_destination"] == agent_name
                    else:  # input
                        rule_applies = rule["message_source"] == agent_name and rule["message_destination"] == tool_name

                    if rule_applies:
                        content = (
                            str(data.get("content", "")) if direction == "output" else str(data.get("arguments", ""))
                        )

                        # Log the detection check
                        print(colored("\n🔍 Checking tool interaction:", "cyan"))
                        print(colored(f"  • Agent: {agent_name}", "cyan"))
                        print(colored(f"  • Tool: {tool_name}", "cyan"))
                        print(colored(f"  • Direction: {direction}", "cyan"))
                        print(colored(f"  • Rule action: {rule.get('action', 'N/A')}", "cyan"))
                        print(colored(f"  • Content: {content[:100]}{'...' if len(content) > 100 else ''}", "cyan"))

                        # Check method-specific handling
                        check_method = rule.get("check_method", "regex")

                        if check_method == "llm":
                            # LLM-based checking - require safeguard_llm_config
                            if not self.safeguard_llm_config:
                                raise ValueError(
                                    f"safeguard_llm_config is required for LLM-based tool interaction rule: {rule['message_source']} -> {rule['message_destination']}"
                                )

                            if "custom_prompt" in rule:
                                print(colored("  • Checking with custom LLM prompt", "blue"))
                                is_violation, explanation = self._check_safeguard_condition(
                                    content, custom_prompt=rule["custom_prompt"]
                                )
                            elif "disallow" in rule:
                                print(colored(f"  • Checking with LLM disallow items: {rule['disallow']}", "blue"))
                                is_violation, explanation = self._check_safeguard_condition(
                                    content, disallow_items=rule["disallow"]
                                )
                            else:
                                raise ValueError(
                                    f"Either custom_prompt or disallow must be provided for LLM-based tool interaction: {rule['message_source']} -> {rule['message_destination']}"
                                )

                            if is_violation:
                                violation_type = "CUSTOM LLM" if "custom_prompt" in rule else "LLM DISALLOW"
                                print(colored(f"  • 🚨 {violation_type} VIOLATION: {explanation}", "red"))
                                result_value = self._apply_action(
                                    rule["action"], data, [], explanation, rule.get("message")
                                )
                                if isinstance(result_value, dict):
                                    return result_value
                                else:
                                    return data
                            else:
                                print(colored("  • ✅ Content passed LLM check", "green"))

                        elif "pattern" in rule:
                            # Regex pattern-based checking
                            print(colored(f"  • Checking with regex pattern: {rule['pattern']}", "blue"))
                            is_violation, explanation = self._check_pattern_condition(content, rule["pattern"])
                            if is_violation:
                                print(colored(f"  • 🚨 REGEX VIOLATION: {explanation}", "red"))
                                result_value = self._apply_action(
                                    rule["action"], data, [], explanation, rule.get("message")
                                )
                                # For tool output blocking, preserve tool message structure
                                if (
                                    direction == "output"
                                    and rule["action"] == "block"
                                    and isinstance(result_value, dict)
                                    and "content" in result_value
                                    or isinstance(result_value, dict)
                                ):
                                    return result_value
                                else:
                                    return data
                            else:
                                print(colored("  • ✅ Content passed regex check", "green"))

                # Handle simple pattern-based format
                elif "pattern" in rule and "message_source" not in rule:
                    content = str(data.get("content", "")) if direction == "output" else str(data.get("arguments", ""))

                    print(colored("\n🔍 Checking tool interaction (simple regex pattern):", "cyan"))
                    print(colored(f"  • Agent: {agent_name}", "cyan"))
                    print(colored(f"  • Direction: {direction}", "cyan"))
                    print(colored(f"  • Content: {content[:100]}{'...' if len(content) > 100 else ''}", "cyan"))
                    print(colored(f"  • Checking with regex pattern: {rule['pattern']}", "blue"))

                    is_violation, explanation = self._check_pattern_condition(content, rule["pattern"])
                    if is_violation:
                        print(colored(f"  • 🚨 REGEX VIOLATION: {explanation}", "red"))
                        result_value = self._apply_action(rule["action"], data, [], explanation, rule.get("message"))
                        # For tool output blocking, preserve tool message structure
                        if isinstance(result_value, dict):
                            return result_value
                        else:
                            return data
                    else:
                        print(colored("  • ✅ Content passed regex check", "green"))

        return data

    def _check_llm_interaction(
        self, agent_name: str, data: str | dict[str, Any] | list[dict[str, Any]], direction: str
    ) -> str | dict[str, Any] | list[dict[str, Any]] | None:
        """Check LLM interactions."""
        for rule in self.environment_rules:
            if rule["type"] == "llm_interaction":
                # Handle format with message_source/message_destination
                if "message_source" in rule and "message_destination" in rule:
                    # Check if this rule applies to the current interaction
                    # For LLM input: message_source = agent, message_destination = "llm"
                    # For LLM output: message_source = "llm", message_destination = agent
                    rule_applies = False
                    if direction == "input":
                        rule_applies = (
                            rule["message_source"] == agent_name and rule["message_destination"].lower() == "llm"
                        )
                    else:  # output
                        rule_applies = (
                            rule["message_source"].lower() == "llm" and rule["message_destination"] == agent_name
                        )

                    if rule_applies:
                        content = str(data)
                        is_violation, explanation = self._check_pattern_condition(content, rule["pattern"])
                        if is_violation:
                            result = self._apply_action(rule["action"], data, [], explanation, rule.get("message"))
                            return result

                # Handle simple pattern-based format
                elif "pattern" in rule and "message_source" not in rule:
                    content = str(data)
                    is_violation, explanation = self._check_pattern_condition(content, rule["pattern"])
                    if is_violation:
                        result_value = self._apply_action(rule["action"], data, [], explanation, rule.get("message"))
                        return result_value

        return data

    def _check_user_interaction(self, agent_name: str, user_input: str) -> str | None:
        """Check user interactions."""
        for rule in self.environment_rules:
            if rule["type"] == "user_interaction":
                # Check if this rule applies to the current interaction
                # For user input: message_source = "user", message_destination = agent
                rule_applies = False
                if "message_source" in rule and "message_destination" in rule:
                    rule_applies = (
                        rule["message_source"].lower() == "user" and rule["message_destination"] == agent_name
                    )

                if rule_applies:
                    if not self.safeguard_llm_config:
                        raise ValueError(
                            f"safeguard_llm_config is required for user interaction rule for agent: {agent_name}"
                        )

                    is_violation, explanation = self._check_safeguard_condition(
                        user_input, disallow_items=rule["disallow"]
                    )
                    if is_violation:
                        result_value = self._apply_action(rule["action"], user_input, rule["disallow"], explanation)
                        return result_value if isinstance(result_value, str) else user_input

        return user_input

    def check_and_act(
        self, src_agent_name: str, dst_agent_name: str, message_content: str | dict[str, Any]
    ) -> str | dict[str, Any] | None:
        """Check and act on inter-agent communication for GroupChat integration.

        This method is called by GroupChat._run_inter_agent_guardrails to check
        messages between agents and potentially modify or block them.

        Args:
            src_agent_name: Name of the source agent
            dst_agent_name: Name of the destination agent
            message_content: The message content to check

        Returns:
            Optional replacement message if a safeguard triggers, None otherwise
        """
        # Store original content for comparison
        original_content = (
            message_content.get("content", "") if isinstance(message_content, dict) else str(message_content)
        )

        result = self._check_inter_agent_communication(src_agent_name, dst_agent_name, message_content)

        if result != original_content:
            # Return the complete modified message structure to preserve tool_calls/tool_responses pairing
            return result

        return None


def apply_safeguards(
    *,
    agents: list[ConversableAgent] | None = None,
    groupchat_manager: GroupChatManager | None = None,
    manifest: dict[str, Any] | str,
    llm_config: LLMConfig | None = None,
    safeguard_llm_config: LLMConfig | None = None,
    mask_llm_config: LLMConfig | None = None,
) -> SafeguardManager:
    """Apply safeguards to agents using a manifest file.

    This is the main function for applying safeguards. It supports the manifest format
    with 'inter_agent_safeguards' and 'agent_environment_safeguards' sections.

    Args:
        agents: List of agents to apply safeguards to (optional if groupchat_manager provided)
        groupchat_manager: GroupChatManager to apply safeguards to (optional if agents provided)
        manifest: Safeguard manifest dict or path to JSON file (follows hospitalgpt format)
        llm_config: Default LLM configuration
        safeguard_llm_config: LLM configuration for safeguard checks
        mask_llm_config: LLM configuration for masking

    Returns:
        SafeguardManager instance for further configuration

    Example:
        ```python
        from autogen.agentchat.group.safeguard import apply_safeguards

        # Apply safeguards to agents
        safeguard_manager = apply_safeguards(
            agents=[agent1, agent2, agent3],
            manifest="path/to/manifest.json",
            llm_config=llm_config,
        )

        # Or apply to GroupChatManager
        safeguard_manager = apply_safeguards(
            groupchat_manager=manager,
            manifest="path/to/manifest.json",
            llm_config=llm_config,
        )
        ```
    """
    manager = SafeguardManager(
        manifest=manifest,
        llm_config=llm_config,
        safeguard_llm_config=safeguard_llm_config,
        mask_llm_config=mask_llm_config,
    )

    # Determine which agents to apply safeguards to
    target_agents: list[ConversableAgent | Agent] = []
    all_agent_names = []

    if groupchat_manager:
        from ..groupchat import GroupChatManager

        if not isinstance(groupchat_manager, GroupChatManager):
            raise ValueError("groupchat_manager must be an instance of GroupChatManager")

        target_agents.extend([agent for agent in groupchat_manager.groupchat.agents if hasattr(agent, "hook_lists")])
        all_agent_names = [agent.name for agent in groupchat_manager.groupchat.agents]
        all_agent_names.append(groupchat_manager.name)

        # Register inter-agent guardrails with the groupchat
        # Ensure the list exists and append our manager
        if not hasattr(groupchat_manager.groupchat, "_inter_agent_guardrails"):
            groupchat_manager.groupchat._inter_agent_guardrails = []
        groupchat_manager.groupchat._inter_agent_guardrails.clear()  # Clear any existing
        groupchat_manager.groupchat._inter_agent_guardrails.append(manager)
        print(colored("✅ Registered inter-agent safeguards with GroupChat", "green"))
    elif agents:
        target_agents.extend(agents)
        all_agent_names = [agent.name for agent in agents]
    else:
        raise ValueError("Either agents or groupchat_manager must be provided")

    # Validate agent names referenced in manifest
    try:
        manager._validate_agent_names(all_agent_names)
    except ValueError as e:
        raise ValueError(f"Agent name validation failed: {e}")

    # Validate tool names if we can extract them from agents
    all_tool_names = []
    for agent in target_agents:
        if hasattr(agent, "tool_names") and agent.tool_names:
            all_tool_names.extend(agent.tool_names)
        elif hasattr(agent, "_tools") and agent._tools:
            # Try to extract tool names from tool objects
            for tool in agent._tools:
                if hasattr(tool, "name"):
                    all_tool_names.append(tool.name)
                elif hasattr(tool, "__name__"):
                    all_tool_names.append(tool.__name__)

    if all_tool_names:
        try:
            manager._validate_tool_names(list(set(all_tool_names)))  # Remove duplicates
        except ValueError as e:
            raise ValueError(f"Tool name validation failed: {e}")

    # Apply hooks to each agent
    for agent in target_agents:
        if hasattr(agent, "hook_lists"):
            hooks = manager.create_agent_hooks(agent.name)
            for hook_name, hook_func in hooks.items():
                if hook_name in agent.hook_lists:
                    agent.hook_lists[hook_name].append(hook_func)
                    print(colored(f"✅ Registered {hook_name} hook for agent {agent.name}", "green"))
        else:
            print(
                colored(
                    f"⚠️ Agent {agent.name} does not support hooks. Please ensure it inherits from ConversableAgent.",
                    "yellow",
                )
            )

    return manager
