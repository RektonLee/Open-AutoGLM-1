"""Model client for Qwen3.5 with specialized parsing for <function=...><parameter=...> format."""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from phone_agent.model.client_tool import (
    ModelClient as ModelClientV2,
    ModelConfig as ModelConfigV2,
    ModelResponse,
)


@dataclass
class ModelConfigQwen35(ModelConfigV2):
    """Configuration for Qwen3.5 model client."""
    pass  # 继承 V2 配置，可以后续添加 Qwen3.5 特定配置


class ModelClientQwen35(ModelClientV2):
    """
    Model client specialized for Qwen3.5 model outputs.

    Qwen3.5 uses <function=...><parameter=...> format in its chat template,
    which differs from the standard tool_calls format.
    """

    def _parse_content_based_tool_calls(
        self, content: str
    ) -> tuple[str, list[dict[str, Any]] | None]:
        """
        Parse tool calls from content with Qwen3.5 format support.

        Supports multiple formats:
        1. Direct JSON format: {"name": "...", "arguments": {...}}
        2. Qwen3.5 format: <tool_call><function=...><parameter=...>...</parameter></function></tool_call>
        3. Legacy XML format: <tool_call>...<arg_key>...</arg_key><arg_value>...</arg_value>...</tool_call>
        """
        # Extract thinking
        think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        thinking = think_match.group(1).strip() if think_match else ""

        # Get content after </think> for tool call parsing
        content_after_think = content
        if "</think>" in content:
            content_after_think = content.split("</think>", 1)[1].strip()

        # Priority 1: Try to parse direct JSON format
        json_match = re.search(
            r'\{[\s]*"name"[\s]*:[\s]*"([^"]+)"[\s]*,[\s]*"arguments"[\s]*:[\s]*(\{.*\}|\[\])\s*\}',
            content_after_think,
            re.DOTALL
        )
        if json_match:
            try:
                function_name = json_match.group(1)
                arguments_str = json_match.group(2)
                # Handle nested braces
                brace_count = 0
                end_pos = 0
                for i, char in enumerate(arguments_str):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_pos = i + 1
                            break
                if end_pos > 0:
                    arguments_str = arguments_str[:end_pos]

                arguments = json.loads(arguments_str)

                tool_calls = [
                    {
                        "id": f"call_{int(time.time() * 1000)}",
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "arguments": json.dumps(arguments, ensure_ascii=False),
                        },
                    }
                ]
                return thinking, tool_calls
            except (json.JSONDecodeError, AttributeError):
                pass

        # Priority 1.5: Try to find and parse complete JSON object
        json_start = content_after_think.find('{"name"')
        if json_start == -1:
            json_start = content_after_think.find('{"name":')
        if json_start >= 0:
            brace_count = 0
            json_end = -1
            for i in range(json_start, len(content_after_think)):
                if content_after_think[i] == '{':
                    brace_count += 1
                elif content_after_think[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break
            if json_end > json_start:
                json_str = content_after_think[json_start:json_end]
                try:
                    parsed = json.loads(json_str)
                    if "name" in parsed and "arguments" in parsed:
                        tool_calls = [
                            {
                                "id": f"call_{int(time.time() * 1000)}",
                                "type": "function",
                                "function": {
                                    "name": parsed["name"],
                                    "arguments": json.dumps(parsed["arguments"], ensure_ascii=False),
                                },
                            }
                        ]
                        return thinking, tool_calls
                except json.JSONDecodeError:
                    pass

        # Priority 2: Try <tool_call> format
        if "<tool_call>" not in content:
            return content, None

        tool_call_match = re.search(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL)
        if not tool_call_match:
            return thinking, None

        tool_call_content = tool_call_match.group(1).strip()

        # Priority 2.1: Try <function=...><parameter=...> format (Qwen3.5 chat template)
        function_match = re.search(r"<function=([^>]+)>", tool_call_content)
        if function_match:
            function_name = function_match.group(1).strip()
            # Parse <parameter=key>value</parameter> pairs
            param_matches = re.findall(
                r"<parameter=([^>]+)>(.*?)</parameter>",
                tool_call_content,
                re.DOTALL
            )
            arguments = {}
            for key, value in param_matches:
                key = key.strip()
                value = value.strip()
                # Try to parse value as JSON
                try:
                    parsed_value = json.loads(value)
                    arguments[key] = parsed_value
                except json.JSONDecodeError:
                    arguments[key] = value

            tool_calls = [
                {
                    "id": f"call_{int(time.time() * 1000)}",
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                }
            ]
            return thinking, tool_calls

        # Priority 2.2: Try legacy <arg_key>/<arg_value> format
        lines = tool_call_content.split("\n")
        function_name = lines[0].strip() if lines else ""

        arg_keys = re.findall(r"<arg_key>(.*?)</arg_key>", tool_call_content)
        arg_values = re.findall(r"<arg_value>(.*?)</arg_value>", tool_call_content)

        arguments = {}
        for key, value in zip(arg_keys, arg_values):
            key = key.strip()
            value = value.strip()
            try:
                parsed_value = json.loads(value)
                arguments[key] = parsed_value
            except json.JSONDecodeError:
                arguments[key] = value

        tool_calls = [
            {
                "id": f"call_{int(time.time() * 1000)}",
                "type": "function",
                "function": {
                    "name": function_name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        ]

        return thinking, tool_calls
