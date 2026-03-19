"""Model client for AI inference using OpenAI-compatible API with Tool Calls support."""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI


@dataclass
class ModelConfig:
    """Configuration for the AI model."""

    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    model_name: str = "autoglm-phone-9b"
    max_tokens: int = 3000
    temperature: float = 0.0
    top_p: float = 0.85
    frequency_penalty: float = 0.2
    extra_body: dict[str, Any] = field(default_factory=dict)
    lang: str = "cn"  # Language for UI messages: 'cn' or 'en'
    tools: list[dict[str, Any]] = field(default_factory=list)  # Tool definitions
    enable_stream: bool = True  # Enable streaming mode


@dataclass
class ModelResponse:
    """Response from the AI model."""

    content: str  # Content before <tool_call>, preserving <think> tags
    tool_calls: list[dict[str, Any]] | None  # Tool calls from the model
    finished: bool  # True if no tool calls (task finished)
    # Performance metrics
    time_to_first_token: float | None = None  # Time to first token (seconds)
    time_to_thinking_end: float | None = None  # Time to thinking end (seconds)
    total_time: float | None = None  # Total inference time (seconds)


class ModelClient:
    """
    Client for interacting with OpenAI-compatible vision-language models.
    Supports both OpenAI tool_calls format and content-embedded XML format (v2).

    Args:
        config: Model configuration.
    """

    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()
        self.client = OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            max_retries=0  # 禁用自动重试，避免重复请求
        )

    def _fix_incomplete_think_tags(self, content: str) -> str:
        """
        Fix incomplete <think> tags in model output.

        Sometimes the model outputs <think> but forgets to close it with </think>
        before outputting <tool_call>. This method detects and fixes such cases.

        Args:
            content: Raw content from model response.

        Returns:
            Fixed content with proper </think> tags.
        """
        # Check if there's a <think> tag
        if "<think>" not in content:
            return content

        # Check if </think> is missing
        if "</think>" in content:
            return content  # Already has closing tag

        # If we have <think> but no </think>, and there's a <tool_call>
        # Insert </think> before <tool_call>
        if "<tool_call>" in content:
            # Find the position of <tool_call>
            tool_call_pos = content.find("<tool_call>")
            # Insert </think> before <tool_call>
            fixed_content = content[:tool_call_pos] + "</think>\n" + content[tool_call_pos:]
            return fixed_content

        # If no <tool_call> found, add </think> at the end
        return content + "\n</think>"

    def _parse_content_based_tool_calls(
        self, content: str
    ) -> tuple[str, list[dict[str, Any]] | None]:
        """
        Parse tool calls from content when they're embedded in the response.

        Supports multiple formats:
        1. Direct JSON format (Qwen3 SFT trained):
           <think>思考过程</think>
           {"name": "function_name", "arguments": {...}}

        2. XML format (original autoglm):
           <think>思考过程</think>
           <tool_call>FunctionName
           <arg_key>param1</arg_key>
           <arg_value>value1</arg_value>
           </tool_call>

        Args:
            content: The content string from the model response.

        Returns:
            Tuple of (thinking, tool_calls). If no tool calls, returns (thinking, None).
        """
        # Extract thinking
        think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        thinking = think_match.group(1).strip() if think_match else ""

        # Get content after </think> for tool call parsing
        content_after_think = content
        if "</think>" in content:
            content_after_think = content.split("</think>", 1)[1].strip()

        # Priority 1: Try to parse direct JSON format (Qwen3 SFT trained format)
        # Pattern: {"name": "...", "arguments": {...}}
        # Use a more robust approach: find JSON object and parse it

        # First, try simple regex for well-formed JSON
        json_match = re.search(r'\{[\s]*"name"[\s]*:[\s]*"([^"]+)"[\s]*,[\s]*"arguments"[\s]*:[\s]*(\{.*\}|\[\])\s*\}', content_after_think, re.DOTALL)
        if json_match:
            try:
                function_name = json_match.group(1)
                arguments_str = json_match.group(2)
                # Handle nested braces - find the matching closing brace
                # Count braces to find the correct end
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
            except (json.JSONDecodeError, AttributeError) as e:
                # Try alternative: parse the full JSON object
                pass

        # Priority 1.5: Try to find and parse complete JSON object
        # Look for {"name": and find the matching closing brace
        json_start = content_after_think.find('{"name"')
        if json_start == -1:
            json_start = content_after_think.find('{"name":')  # without space
        if json_start >= 0:
            # Find matching closing brace
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

        # Priority 2: Try XML format
        if "<tool_call>" not in content:
            # No tool calls found
            return content, None

        # Extract tool call content
        tool_call_match = re.search(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL)
        if not tool_call_match:
            return thinking, None

        tool_call_content = tool_call_match.group(1).strip()

        # Parse function name (first line before any tags)
        lines = tool_call_content.split("\n")
        function_name = lines[0].strip() if lines else ""

        # Parse arguments
        arg_keys = re.findall(r"<arg_key>(.*?)</arg_key>", tool_call_content)
        arg_values = re.findall(r"<arg_value>(.*?)</arg_value>", tool_call_content)

        # Build arguments dict
        arguments = {}
        for key, value in zip(arg_keys, arg_values):
            key = key.strip()
            value = value.strip()

            # Try to parse value as JSON (handles numbers, booleans, arrays, objects, null)
            try:
                parsed_value = json.loads(value)
                arguments[key] = parsed_value
            except json.JSONDecodeError:
                # If JSON parsing fails, keep as string
                arguments[key] = value

        # Build tool call in OpenAI format
        tool_calls = [
            {
                "id": f"call_{int(time.time() * 1000)}",  # Generate a simple ID
                "type": "function",
                "function": {
                    "name": function_name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        ]

        return thinking, tool_calls

    def request(self, messages: list[dict[str, Any]]) -> ModelResponse:
        """
        Send a request to the model with tool calls support.

        Args:
            messages: List of message dictionaries in OpenAI format.

        Returns:
            ModelResponse containing thinking and tool calls.

        Raises:
            ValueError: If the response cannot be parsed.
        """
        if self.config.enable_stream:
            return self._request_stream(messages)
        else:
            return self._request_non_stream(messages)

    def _request_stream(self, messages: list[dict[str, Any]]) -> ModelResponse:
        """
        Send a streaming request to the model.

        Args:
            messages: List of message dictionaries in OpenAI format.

        Returns:
            ModelResponse containing thinking and tool calls.
        """
        # Start timing
        start_time = time.time()
        time_to_first_token = None
        time_to_thinking_end = None

        stream = self.client.chat.completions.create(
            messages=messages,
            model=self.config.model_name,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            frequency_penalty=self.config.frequency_penalty,
            extra_body=self.config.extra_body,
            tools=self.config.tools if self.config.tools else None,
            stream=True,
        )

        raw_content = ""
        buffer = ""  # Buffer to hold content that might be part of a marker
        in_think_tag = False  # Track if we're inside <think> tag
        in_tool_call_phase = False  # Track if we've entered the tool call phase
        first_token_received = False
        stream_tool_calls = None  # Track tool_calls from streaming

        # Markers to detect
        think_start_marker = "<think>"
        think_end_marker = "</think>"
        tool_call_marker = "<tool_call>"
        json_tool_marker = '{"name":'  # Direct JSON format marker

        for chunk in stream:
            if len(chunk.choices) == 0:
                continue

            delta = chunk.choices[0].delta

            # Record time to first token
            if not first_token_received:
                time_to_first_token = time.time() - start_time
                first_token_received = True

            # Handle content (thinking process) - print <think> content in real-time
            if delta.content is not None:
                content = delta.content
                raw_content += content

                # Don't print if we're already in tool call phase
                if in_tool_call_phase:
                    continue

                buffer += content

                # Check if we're entering <think> tag
                if not in_think_tag and think_start_marker in buffer:
                    # Found <think> tag, split and start printing
                    parts = buffer.split(think_start_marker, 1)
                    buffer = parts[1] if len(parts) > 1 else ""
                    in_think_tag = True
                    continue

                # If we're inside <think> tag, check for end markers
                if in_think_tag:
                    # Check for </think> marker
                    if think_end_marker in buffer:
                        # Found end of thinking, print content before marker
                        thinking_part = buffer.split(think_end_marker, 1)[0]
                        print(thinking_part, end="", flush=True)
                        print()  # Newline after thinking
                        in_tool_call_phase = True

                        # Record time to thinking end
                        if time_to_thinking_end is None:
                            time_to_thinking_end = time.time() - start_time

                        buffer = ""
                        continue

                    # Check for <tool_call> marker
                    if tool_call_marker in buffer:
                        # Found tool call, print content before marker
                        thinking_part = buffer.split(tool_call_marker, 1)[0]
                        print(thinking_part, end="", flush=True)
                        print()  # Newline after thinking
                        in_tool_call_phase = True

                        # Record time to thinking end
                        if time_to_thinking_end is None:
                            time_to_thinking_end = time.time() - start_time

                        buffer = ""
                        continue

                    # Check for direct JSON format {"name":
                    if json_tool_marker in buffer:
                        # Found JSON tool call, print content before marker
                        thinking_part = buffer.split(json_tool_marker, 1)[0]
                        print(thinking_part, end="", flush=True)
                        print()  # Newline after thinking
                        in_tool_call_phase = True

                        # Record time to thinking end
                        if time_to_thinking_end is None:
                            time_to_thinking_end = time.time() - start_time

                        buffer = ""
                        continue

                    # Check if buffer ends with a prefix of any end marker
                    is_potential_marker = False
                    for marker in [think_end_marker, tool_call_marker, json_tool_marker]:
                        for i in range(1, len(marker)):
                            if buffer.endswith(marker[:i]):
                                is_potential_marker = True
                                break
                        if is_potential_marker:
                            break

                    if not is_potential_marker:
                        # Safe to print the buffer
                        print(buffer, end="", flush=True)
                        buffer = ""

            # Handle tool_calls in streaming mode (if supported by API)
            if hasattr(delta, 'tool_calls') and delta.tool_calls:
                # When we receive tool_calls, mark that thinking phase is over
                if not in_tool_call_phase:
                    # Print any remaining buffer content
                    if buffer and in_think_tag:
                        print(buffer, end="", flush=True)
                        buffer = ""
                    print()  # Newline after thinking
                    in_tool_call_phase = True

                    # Record time to thinking end
                    if time_to_thinking_end is None:
                        time_to_thinking_end = time.time() - start_time

                if stream_tool_calls is None:
                    stream_tool_calls = []
                # Accumulate tool calls from stream
                for tc_delta in delta.tool_calls:
                    if tc_delta.index >= len(stream_tool_calls):
                        stream_tool_calls.append({
                            "id": tc_delta.id or "",
                            "type": tc_delta.type or "function",
                            "function": {"name": "", "arguments": ""}
                        })
                    if tc_delta.function:
                        if tc_delta.function.name:
                            stream_tool_calls[tc_delta.index]["function"]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            stream_tool_calls[tc_delta.index]["function"]["arguments"] += tc_delta.function.arguments

        # Print any remaining buffer content (in case stream ended without markers)
        if buffer and in_think_tag and not in_tool_call_phase:
            print(buffer, end="", flush=True)
            print()

        # Record time to thinking end if not recorded yet
        if time_to_thinking_end is None:
            time_to_thinking_end = time.time() - start_time

        # Calculate total time
        total_time = time.time() - start_time

        # Fix incomplete <think> tags before parsing (DISABLED)
        # raw_content = self._fix_incomplete_think_tags(raw_content)

        # Parse tool calls from the collected content
        content = raw_content  # Default: use full raw_content
        tool_calls = None
        finished = False

        # Priority 1: Use tool_calls from streaming if available
        if stream_tool_calls:
            tool_calls = stream_tool_calls
            # Extract content before <tool_call> (if present)
            if "<tool_call>" in raw_content:
                content = raw_content.split("<tool_call>", 1)[0]
        # Priority 2: Check for XML-embedded tool calls in content
        elif "<tool_call>" in raw_content:
            # Content-embedded XML format (new standard)
            # Extract content before <tool_call>, preserving <think> tags
            content = raw_content.split("<tool_call>", 1)[0]
            # Parse tool calls from the XML
            _, tool_calls = self._parse_content_based_tool_calls(raw_content)
        # Priority 3: Check for direct JSON format {"name": ...}
        elif '{"name":' in raw_content:
            # Direct JSON format (Qwen3 SFT trained)
            _, tool_calls = self._parse_content_based_tool_calls(raw_content)
            if tool_calls:
                # Extract content before JSON
                content = raw_content.split('{"name":', 1)[0]
        # Priority 4: No tool calls (task finished)
        else:
            # Use full raw_content as content
            content = raw_content

        # Determine if task is finished
        if tool_calls is None:
            # No tool calls means task finished
            finished = True

        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            finished=finished,
            time_to_first_token=time_to_first_token,
            time_to_thinking_end=time_to_thinking_end,
            total_time=total_time,
        )

    def _request_non_stream(self, messages: list[dict[str, Any]]) -> ModelResponse:
        """
        Send a non-streaming request to the model.

        Args:
            messages: List of message dictionaries in OpenAI format.

        Returns:
            ModelResponse containing thinking and tool calls.
        """
        # Start timing
        start_time = time.time()

        response = self.client.chat.completions.create(
            messages=messages,
            model=self.config.model_name,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            frequency_penalty=self.config.frequency_penalty,
            extra_body=self.config.extra_body,
            tools=self.config.tools if self.config.tools else None,
            stream=False,
        )

        # Calculate total time
        total_time = time.time() - start_time

        # Extract response data
        choice = response.choices[0]
        message = choice.message
        raw_content = message.content or ""

        # Fix incomplete <think> tags before parsing (DISABLED)
        # raw_content = self._fix_incomplete_think_tags(raw_content)

        # Parse tool calls from the response
        content = raw_content  # Default: use full raw_content
        tool_calls = None
        finished = False

        # Priority 1: Use tool_calls from API response if available
        if hasattr(message, 'tool_calls') and message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]
            # Extract content before <tool_call> (if present)
            if "<tool_call>" in raw_content:
                content = raw_content.split("<tool_call>", 1)[0].rstrip()
        # Priority 2: Check for XML-embedded tool calls in content
        elif "<tool_call>" in raw_content:
            # Content-embedded XML format
            # Extract content before <tool_call>, preserving <think> tags
            content = raw_content.split("<tool_call>", 1)[0].rstrip()
            # Parse tool calls from the XML
            _, tool_calls = self._parse_content_based_tool_calls(raw_content)
        # Priority 3: Check for direct JSON format {"name": ...}
        elif '{"name":' in raw_content:
            # Direct JSON format (Qwen3 SFT trained)
            _, tool_calls = self._parse_content_based_tool_calls(raw_content)
            if tool_calls:
                # Extract content before JSON
                content = raw_content.split('{"name":', 1)[0].rstrip()
        # Priority 4: No tool calls (task finished)
        else:
            # Use full raw_content as content
            content = raw_content

        # Determine if task is finished
        if tool_calls is None:
            # No tool calls means task finished
            finished = True

        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            finished=finished,
            time_to_first_token=None,  # Not applicable in non-streaming mode
            time_to_thinking_end=None,  # Not applicable in non-streaming mode
            total_time=total_time,
        )


class MessageBuilder:
    """Helper class for building conversation messages."""

    @staticmethod
    def create_system_message(content: str) -> dict[str, Any]:
        """Create a system message."""
        return {"role": "system", "content": content}

    @staticmethod
    def create_user_message(
        text: str, image_base64: str | None = None
    ) -> dict[str, Any]:
        """
        Create a user message with optional image.

        Args:
            text: Text content.
            image_base64: Optional base64-encoded image.

        Returns:
            Message dictionary.
        """
        content = []

        if image_base64:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                }
            )

        content.append({"type": "text", "text": text})

        return {"role": "user", "content": content}

    @staticmethod
    def create_assistant_message(
        content: str,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Create an assistant message with content and optional tool calls.

        Args:
            content: Assistant content (before <tool_call>, with <think> tags preserved).
            tool_calls: Optional tool calls.

        Returns:
            Message dictionary.
        """
        message = {"role": "assistant", "content": content}

        if tool_calls:
            # Convert to OpenAI tool call format
            message["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": tc["type"],
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    },
                }
                for tc in tool_calls
            ]

        return message

    @staticmethod
    def create_observation_message(observation: str, image_base64: str | None = None) -> dict[str, Any]:
        """
        Create an observation message for action result.

        Note: Uses 'user' role instead of 'tool' to match the Qwen3 SFT training format,
        where 'observation' role is converted to user-like messages during training.

        Args:
            observation: Observation text (e.g., "Tapped at (499, 669)").
            image_base64: Optional base64-encoded screenshot.

        Returns:
            Message dictionary with role='user'.
        """
        content = []

        # Add image first if provided (matching training data format: <image> comes before text)
        if image_base64:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                }
            )

        content.append({"type": "text", "text": observation})

        return {"role": "user", "content": content}

    @staticmethod
    def remove_images_from_message(message: dict[str, Any]) -> dict[str, Any]:
        """
        Remove image content from a message to save context space.

        Args:
            message: Message dictionary.

        Returns:
            Message with images removed.
        """
        if isinstance(message.get("content"), list):
            message["content"] = [
                item for item in message["content"] if item.get("type") == "text"
            ]
        return message

    @staticmethod
    def build_screen_info(current_app: str) -> str:
        """
        Build screen info string for the model.

        Args:
            current_app: Current app name.

        Returns:
            System reminder string.
        """
        from datetime import datetime

        today = datetime.today()
        date_str = today.strftime("%Y.%m.%d")
        weekday_names = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
        weekday = weekday_names[today.weekday()]

        return f"<system-reminder>current date: {date_str}, {weekday};current app: {current_app}</system-reminder>"
