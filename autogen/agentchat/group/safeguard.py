# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any
from uuid import UUID

from termcolor import colored

from ...events.base_event import BaseEvent, wrap_event
from ...io.base import IOStream
from ...llm_config import LLMConfig
from .guardrails import LLMGuardrail, RegexGuardrail
from .targets.transition_target import TransitionTarget

if TYPE_CHECKING:
    from ..agent import Agent
    from ..conversable_agent import ConversableAgent
    from ..groupchat import GroupChatManager


@wrap_event
class SafeguardEvent(BaseEvent):
    """Event for safeguard actions"""
    
    event_type: str  # e.g., "load", "check", "violation", "action"
    message: str
    source_agent: str | None = None
    target_agent: str | None = None
    guardrail_type: str | None = None
    action: str | None = None
    content_preview: str | None = None
    
    def __init__(
        self,
        *,
        uuid: UUID | None = None,
        event_type: str,
        message: str,
        source_agent: str | None = None,
        target_agent: str | None = None,
        guardrail_type: str | None = None,
        action: str | None = None,
        content_preview: str | None = None,
    ):
        super().__init__(
            uuid=uuid,
            event_type=event_type,
            message=message,
            source_agent=source_agent,
            target_agent=target_agent,
            guardrail_type=guardrail_type,
            action=action,
            content_preview=content_preview,
        )
    
    def print(self, f: Callable[..., Any] | None = None) -> None:
        f = f or print
        
        # Choose color based on event type
        color = "green"
        if self.event_type == "load":
            color = "green"
        elif self.event_type == "check":
            color = "cyan"
        elif self.event_type == "violation":
            color = "red"
        elif self.event_type == "action":
            color = "yellow"
        
        # Choose emoji based on event type
        emoji = ""
        if self.event_type == "load":
            emoji = "✅"
        elif self.event_type == "check":
            emoji = "🔍"
        elif self.event_type == "violation":
            emoji = "🛡️"
        elif self.event_type == "action":
            if self.action == "block":
                emoji = "🚨"
            elif self.action == "mask":
                emoji = "🎭"
            elif self.action == "warning":
                emoji = "⚠️"
            else:
                emoji = "⚙️"
        
        # Create header based on event type (skip for load events)
        if self.event_type == "check":
            header = f"***** Safeguard Check: {self.message} *****"
            f(colored(header, color), flush=True)
        elif self.event_type == "violation":
            header = f"***** Safeguard Violation: DETECTED *****"
            f(colored(header, color), flush=True)
        elif self.event_type == "action":
            header = f"***** Safeguard Enforcement Action: {self.action.upper() if self.action else 'APPLIED'} *****"
            f(colored(header, color), flush=True)
        
        # Format the output
        output_parts = [f"{emoji} {self.message}" if emoji else self.message]
        
        if self.source_agent and self.target_agent:
            output_parts.append(f"  • From: {self.source_agent}")
            output_parts.append(f"  • To: {self.target_agent}")
        elif self.source_agent:
            output_parts.append(f"  • Agent: {self.source_agent}")
        
        if self.guardrail_type:
            output_parts.append(f"  • Guardrail: {self.guardrail_type}")
        
        if self.action:
            output_parts.append(f"  • Action: {self.action}")
            
        if self.content_preview:
            # Replace actual newlines with \n for display
            content_display = self.content_preview.replace('\n', '\\n').replace('\r', '\\r')
            output_parts.append(f"  • Content: {content_display}")
        
        f(colored("\n".join(output_parts), color), flush=True)
        
        # Print footer with matching length (skip for load events)
        if self.event_type in ["check", "violation", "action"]:
            footer = "*" * len(header)
            f(colored(footer, color), flush=True)


class SafeguardManager:
    """Main safeguard manager"""

    def __init__(
        self,
        manifest: dict[str, Any] | str,
        safeguard_llm_config: LLMConfig | dict[str, Any] | None = None,
        mask_llm_config: LLMConfig | dict[str, Any] | None = None,
    ):
        """Initialize the safeguard manager.

        Args:
            manifest: Safeguard manifest dict or path to JSON file
            safeguard_llm_config: LLM configuration for safeguard checks
            mask_llm_config: LLM configuration for masking
        """
        self.manifest = self._load_manifest(manifest)
        self.safeguard_llm_config = safeguard_llm_config
        self.mask_llm_config = mask_llm_config

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

        # Send load event
        self._send_safeguard_event(
            event_type="load",
            message=f"Loaded {len(self.inter_agent_rules)} inter-agent and {len(self.environment_rules)} environment safeguard rules"
        )

    def _send_safeguard_event(
        self,
        event_type: str,
        message: str,
        source_agent: str | None = None,
        target_agent: str | None = None,
        guardrail_type: str | None = None,
        action: str | None = None,
        content_preview: str | None = None,
    ) -> None:
        """Send a safeguard event to the IOStream."""
        iostream = IOStream.get_default()
        event = SafeguardEvent(
            event_type=event_type,
            message=message,
            source_agent=source_agent,
            target_agent=target_agent,
            guardrail_type=guardrail_type,
            action=action,
            content_preview=content_preview,
        )
        iostream.send(event)

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

                elif "disallow_item" in rule:
                    # Disallow items for LLM guardrail
                    condition = f"Check if this content contains any of these disallowed categories: {', '.join(rule['disallow_item'])}"

                else:
                    raise ValueError(
                        f"Either custom_prompt or disallow_item must be provided for LLM guardrail: {rule['message_src']} -> {rule['message_dst']}"
                    )

                # Create LLM guardrail - handle dict config by converting to LLMConfig
                llm_config = self.safeguard_llm_config
                if isinstance(llm_config, dict):
                    llm_config = LLMConfig(config_list=[llm_config])

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
                elif "disallow_item" in rule:
                    parsed_rule["disallow"] = rule["disallow_item"]

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

            hooks["safeguard_tool_inputs"] = tool_input_hook
            hooks["safeguard_tool_outputs"] = tool_output_hook

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

            hooks["safeguard_llm_inputs"] = llm_input_hook
            hooks["safeguard_llm_outputs"] = llm_output_hook

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

            hooks["safeguard_human_inputs"] = human_input_hook

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
        # Handle dict config by converting to LLMConfig
        llm_config = self.safeguard_llm_config
        if isinstance(llm_config, dict):
            llm_config = LLMConfig(config_list=[llm_config])

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
            raise ValueError(f"Invalid regex pattern '{pattern}': {e}")

        return False, "No pattern match"

    def _apply_action(
        self,
        action: str,
        content: str | dict[str, Any] | list[Any],
        disallow_items: list[str],
        explanation: str,
        custom_message: str | None = None,
        pattern: str | None = None,
        guardrail_type: str | None = None,
        source_agent: str | None = None,
        target_agent: str | None = None,
        content_preview: str | None = None,
    ) -> str | dict[str, Any] | list[Any]:
        """Apply the specified action to content."""
        message = custom_message or explanation

        if action == "block":
            self._send_safeguard_event(
                event_type="action",
                message=f"BLOCKED: {message}",
                action="block",
                source_agent=source_agent,
                target_agent=target_agent,
                content_preview=content_preview
            )
            return self._handle_blocked_content(content, message)
        elif action == "mask":
            self._send_safeguard_event(
                event_type="action",
                message=f"MASKED: {message}",
                action="mask",
                source_agent=source_agent,
                target_agent=target_agent,
                content_preview=content_preview
            )

            def mask_func(text: str) -> str:
                return self._mask_content(text, disallow_items, explanation, pattern)

            return self._handle_masked_content(content, mask_func)
        elif action == "warning":
            self._send_safeguard_event(
                event_type="action",
                message=f"WARNING: {message}",
                action="warning",
                source_agent=source_agent,
                target_agent=target_agent,
                content_preview=content_preview
            )
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
                raise ValueError(f"Pattern masking failed: {e}")

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
                raise ValueError(f"LLM masking failed: {e}")

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
                    # Prepare content preview
                    content_preview = content[:100] + ('...' if len(content) > 100 else '')
                    
                    # Use guardrail if available
                    if "guardrail" in rule and rule["guardrail"]:
                        # Send single check event with guardrail info
                        self._send_safeguard_event(
                            event_type="check",
                            message="Checking inter-agent communication",
                            source_agent=sender_name,
                            target_agent=recipient_name,
                            guardrail_type=type(rule["guardrail"]).__name__,
                            # action=rule.get('action', 'N/A'),
                            content_preview=content_preview
                        )

                        try:
                            result = rule["guardrail"].check(content)
                            if result.activated:
                                self._send_safeguard_event(
                                    event_type="violation",
                                    message=f"VIOLATION DETECTED: {result.justification}",
                                    source_agent=sender_name,
                                    target_agent=recipient_name,
                                    guardrail_type=type(rule['guardrail']).__name__,
                                    content_preview=content_preview
                                )
                                # Pass the pattern if it's a regex guardrail
                                pattern = rule.get("pattern") if isinstance(rule["guardrail"], RegexGuardrail) else None
                                action_result = self._apply_action(
                                    action=rule["action"],
                                    content=message,
                                    disallow_items=[],
                                    explanation=result.justification,
                                    custom_message=rule.get("activation_message", result.justification),
                                    pattern=pattern,
                                    guardrail_type=type(rule["guardrail"]).__name__,
                                    source_agent=sender_name,
                                    target_agent=recipient_name,
                                    content_preview=content_preview
                                )
                                if isinstance(action_result, (str, dict)):
                                    return action_result
                                else:
                                    return message
                            else:
                                # Content passed - no additional event needed, already sent check event above
                                pass
                        except Exception as e:
                            raise ValueError(f"Guardrail check failed: {e}")

                    # Handle legacy pattern-based rules
                    elif "pattern" in rule and rule["pattern"]:
                        # Send single check event for pattern-based rules
                        self._send_safeguard_event(
                            event_type="check",
                            message="Checking inter-agent communication",
                            source_agent=sender_name,
                            target_agent=recipient_name,
                            guardrail_type="RegexGuardrail",
                            # action=rule.get('action', 'N/A'),
                            content_preview=content_preview
                        )
                        is_violation, explanation = self._check_pattern_condition(content, rule["pattern"])
                        if is_violation:
                            result_value = self._apply_action(
                                action=rule["action"],
                                content=message,
                                disallow_items=[],
                                explanation=explanation,
                                custom_message=rule.get("activation_message"),
                                pattern=rule["pattern"],
                                guardrail_type="RegexGuardrail",
                                source_agent=sender_name,
                                target_agent=recipient_name,
                                content_preview=content_preview
                            )
                            if isinstance(result_value, (str, dict)):
                                return result_value
                            else:
                                return message
                        else:
                            pass

                    # Handle legacy disallow-based rules and custom prompts  
                    elif "disallow" in rule or "custom_prompt" in rule:
                        # Send single check event for LLM-based legacy rules
                        self._send_safeguard_event(
                            event_type="check",
                            message="Checking inter-agent communication",
                            source_agent=sender_name,
                            target_agent=recipient_name,
                            guardrail_type="LLMGuardrail",
                            # action=rule.get('action', 'N/A'),
                            content_preview=content_preview
                        )
                        if "custom_prompt" in rule:
                            is_violation, explanation = self._check_safeguard_condition(
                                content, custom_prompt=rule["custom_prompt"]
                            )
                        else:
                            is_violation, explanation = self._check_safeguard_condition(
                                content, disallow_items=rule["disallow"]
                            )

                        if is_violation:
                            violation_type = "CUSTOM PROMPT" if "custom_prompt" in rule else "DISALLOW"
                            result_value = self._apply_action(
                                action=rule["action"], 
                                content=message, 
                                disallow_items=rule.get("disallow", []), 
                                explanation=explanation,
                                custom_message=None,
                                pattern=None,
                                guardrail_type="LLMGuardrail",
                                source_agent=sender_name,
                                target_agent=recipient_name,
                                content_preview=content_preview
                            )
                            if isinstance(result_value, (str, dict)):
                                return result_value
                            else:
                                return message
                        else:
                            pass

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

                        # Send check event for tool interaction
                        content_preview = content[:100] + ('...' if len(content) > 100 else '')
                        self._send_safeguard_event(
                            event_type="check",
                            message=f"Checking tool interaction: {agent_name} <-> {tool_name} ({direction})",
                            source_agent=agent_name,
                            target_agent=tool_name,
                            guardrail_type="LLMGuardrail" if rule.get("check_method") == "llm" else "RegexGuardrail",
                            # action=rule.get('action', 'N/A'),
                            content_preview=content_preview
                        )

                        # Check method-specific handling
                        check_method = rule.get("check_method", "regex")

                        if check_method == "llm":
                            # LLM-based checking - require safeguard_llm_config
                            if not self.safeguard_llm_config:
                                raise ValueError(
                                    f"safeguard_llm_config is required for LLM-based tool interaction rule: {rule['message_source']} -> {rule['message_destination']}"
                                )

                            if "custom_prompt" in rule:
                                is_violation, explanation = self._check_safeguard_condition(
                                    content, custom_prompt=rule["custom_prompt"]
                                )
                            elif "disallow" in rule:
                                is_violation, explanation = self._check_safeguard_condition(
                                    content, disallow_items=rule["disallow"]
                                )
                            else:
                                raise ValueError(
                                    f"Either custom_prompt or disallow must be provided for LLM-based tool interaction: {rule['message_source']} -> {rule['message_destination']}"
                                )

                            if is_violation:
                                violation_type = "CUSTOM LLM" if "custom_prompt" in rule else "LLM DISALLOW"
                                result_value = self._apply_action(
                                    action=rule["action"], 
                                    content=data, 
                                    disallow_items=[], 
                                    explanation=explanation, 
                                    custom_message=rule.get("message"),
                                    pattern=None,
                                    guardrail_type="LLMGuardrail",
                                    source_agent=agent_name,
                                    target_agent=tool_name,
                                    content_preview=content_preview
                                )
                                if isinstance(result_value, dict):
                                    return result_value
                                else:
                                    return data
                            else:
                                pass

                        elif "pattern" in rule:
                            # Regex pattern-based checking
                            is_violation, explanation = self._check_pattern_condition(content, rule["pattern"])
                            if is_violation:
                                result_value = self._apply_action(
                                    action=rule["action"], 
                                    content=data, 
                                    disallow_items=[], 
                                    explanation=explanation, 
                                    custom_message=rule.get("message")
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
                                pass

                # Handle simple pattern-based format
                elif "pattern" in rule and "message_source" not in rule:
                    content = str(data.get("content", "")) if direction == "output" else str(data.get("arguments", ""))
                    content_preview = content[:100] + "..." if len(content) > 100 else content

                    self._send_safeguard_event(
                        event_type="check",
                        message=f"Checking tool interaction: {agent_name} <-> tool ({direction})",
                        source_agent=agent_name,
                        target_agent="tool",
                        guardrail_type="RegexGuardrail",
                        # action=rule.get('action', 'N/A'),
                        content_preview=content_preview
                    )
                    
                    is_violation, explanation = self._check_pattern_condition(content, rule["pattern"])
                    if is_violation:
                        self._send_safeguard_event(
                            event_type="violation",
                            message=f"REGEX VIOLATION: {explanation}",
                            source_agent=agent_name,
                            target_agent="tool",
                            guardrail_type="RegexGuardrail",
                            content_preview=content_preview
                        )
                        result_value = self._apply_action(
                            action=rule["action"], 
                            content=data, 
                            disallow_items=[], 
                            explanation=explanation, 
                            custom_message=rule.get("message"),
                            pattern=rule.get("pattern"),
                            guardrail_type="RegexGuardrail",
                            source_agent=agent_name,
                            target_agent="tool",
                            content_preview=content_preview
                        )
                        # For tool output blocking, preserve tool message structure
                        if isinstance(result_value, dict):
                            return result_value
                        else:
                            return data

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
                        content_preview = content[:100] + "..." if len(content) > 100 else content
                        target_name = "llm" if direction == "input" else agent_name
                        source_name = agent_name if direction == "input" else "llm"

                        self._send_safeguard_event(
                            event_type="check",
                            message=f"Checking LLM interaction: {source_name} <-> {target_name} ({direction})",
                            source_agent=source_name,
                            target_agent=target_name,
                            guardrail_type="RegexGuardrail",
                            # action=rule.get('action', 'N/A'),
                            content_preview=content_preview
                        )
                        
                        is_violation, explanation = self._check_pattern_condition(content, rule["pattern"])
                        if is_violation:
                            self._send_safeguard_event(
                                event_type="violation",
                                message=f"REGEX VIOLATION: {explanation}",
                                source_agent=source_name,
                                target_agent=target_name,
                                guardrail_type="RegexGuardrail",
                                content_preview=content_preview
                            )
                            result = self._apply_action(
                                action=rule["action"], 
                                content=data, 
                                disallow_items=[], 
                                explanation=explanation, 
                                custom_message=rule.get("message"),
                                pattern=rule.get("pattern"),
                                guardrail_type="RegexGuardrail",
                                source_agent=source_name,
                                target_agent=target_name,
                                content_preview=content_preview
                            )
                            return result

                # Handle simple pattern-based format
                elif "pattern" in rule and "message_source" not in rule:
                    content = str(data)
                    content_preview = content[:100] + "..." if len(content) > 100 else content

                    self._send_safeguard_event(
                        event_type="check",
                        message=f"Checking LLM interaction: {agent_name} <-> llm ({direction})",
                        source_agent=agent_name,
                        target_agent="llm",
                        guardrail_type="RegexGuardrail",
                        # action=rule.get('action', 'N/A'),
                        content_preview=content_preview
                    )
                    
                    is_violation, explanation = self._check_pattern_condition(content, rule["pattern"])
                    if is_violation:
                        self._send_safeguard_event(
                            event_type="violation",
                            message=f"REGEX VIOLATION: {explanation}",
                            source_agent=agent_name,
                            target_agent="llm",
                            guardrail_type="RegexGuardrail",
                            content_preview=content_preview
                        )
                        result_value = self._apply_action(
                            action=rule["action"], 
                            content=data, 
                            disallow_items=[], 
                            explanation=explanation, 
                            custom_message=rule.get("message"),
                            pattern=rule.get("pattern"),
                            guardrail_type="RegexGuardrail",
                            source_agent=agent_name,
                            target_agent="llm",
                            content_preview=content_preview
                        )
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

                    content_preview = user_input[:100] + "..." if len(user_input) > 100 else user_input
                    
                    self._send_safeguard_event(
                        event_type="check",
                        message=f"Checking user interaction: user <-> {agent_name}",
                        source_agent="user",
                        target_agent=agent_name,
                        guardrail_type="LLMGuardrail",
                        # action=rule.get('action', 'N/A'),
                        content_preview=content_preview
                    )

                    is_violation, explanation = self._check_safeguard_condition(
                        user_input, disallow_items=rule["disallow"]
                    )
                    if is_violation:
                        self._send_safeguard_event(
                            event_type="violation",
                            message=f"LLM VIOLATION: {explanation}",
                            source_agent="user",
                            target_agent=agent_name,
                            guardrail_type="LLMGuardrail",
                            content_preview=content_preview
                        )
                        result_value = self._apply_action(
                            action=rule["action"], 
                            content=user_input, 
                            disallow_items=rule["disallow"], 
                            explanation=explanation,
                            custom_message=None,
                            pattern=None,
                            guardrail_type="LLMGuardrail",
                            source_agent="user",
                            target_agent=agent_name,
                            content_preview=content_preview
                        )
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
            safeguard_llm_config=safeguard_llm_config,
        )

        # Or apply to GroupChatManager
        safeguard_manager = apply_safeguards(
            groupchat_manager=manager,
            manifest="path/to/manifest.json",
            safeguard_llm_config=safeguard_llm_config,
            mask_llm_config=mask_llm_config,
        )
        ```
    """
    manager = SafeguardManager(
        manifest=manifest,
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
        else:
            raise ValueError(
                f"Agent {agent.name} does not support hooks. Please ensure it inherits from ConversableAgent."
            )

    return manager
