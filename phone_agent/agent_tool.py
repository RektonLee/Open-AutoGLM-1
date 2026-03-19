"""Main PhoneAgent class for orchestrating phone automation (Tool Calls Version)."""

import json
import os
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from phone_agent.actions.handler_tool import ActionHandler
from phone_agent.config.i18n import get_message, get_messages
from phone_agent.device_factory import get_device_factory
from phone_agent.model.client_tool import MessageBuilder, ModelClient, ModelConfig

# Default config paths (relative to package root)
_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SYSTEM_PROMPT_FILE = os.path.join(_PACKAGE_DIR, "config", "sys_prompt.md")
DEFAULT_TOOLS_FILE = os.path.join(_PACKAGE_DIR, "config", "tool.json")


def _load_default_system_prompt() -> str:
    """Load system prompt from default config file."""
    if os.path.exists(DEFAULT_SYSTEM_PROMPT_FILE):
        with open(DEFAULT_SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
            return f.read()
    # Fallback to hardcoded prompt if file not found
    return "You are an Android automation assistant."


def _load_default_tools() -> list[dict[str, Any]]:
    """Load tools from default config file."""
    if os.path.exists(DEFAULT_TOOLS_FILE):
        with open(DEFAULT_TOOLS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    # Fallback to empty list if file not found
    return []


@dataclass
class AgentConfig:
    """Configuration for the PhoneAgent (v2)."""

    max_steps: int = 100
    device_id: str | None = None
    lang: str = "cn"
    system_prompt: str | None = None
    tools: list[dict[str, Any]] | None = None
    verbose: bool = True
    save_messages: bool = True  # Save messages to JSON file after task completion
    messages_output_dir: str = "output"  # Directory to save messages
    keep_recent_images: int = 3  # Number of recent images to keep in context

    def __post_init__(self):
        # Load system prompt from default config file
        if self.system_prompt is None:
            self.system_prompt = _load_default_system_prompt()

        # Load tools from default config file
        if self.tools is None:
            self.tools = _load_default_tools()


@dataclass
class StepResult:
    """Result of a single agent step."""

    success: bool
    finished: bool
    tool_calls: list[dict[str, Any]] | None
    thinking: str
    observation: str | None = None
    special_tag: str | None = None  # 特殊标签: finish, sensitive, preference, captcha, verification, notool, toxic


# 特殊标签定义
SPECIAL_TAGS = {
    "[finish]": "task_completed",      # 任务完成
    "[notool]": "no_action_needed",    # 无需手机操作，直接回答
    "[sensitive]": "sensitive_screen", # 敏感页面（登录、支付等），需要用户处理
    "[captcha]": "captcha_detected",   # 验证码，需要用户处理
    "[verification]": "need_confirm",  # 敏感操作需要用户确认
    "[preference]": "need_user_input", # 需要用户提供选择或输入
    "[toxic]": "request_refused",      # 违反安全政策，拒绝执行
}


class PhoneAgentV2:
    """
    AI-powered agent for automating Android phone interactions (Tool Calls Version).

    The agent uses a vision-language model to understand screen content
    and decide on actions to complete user tasks.

    Args:
        model_config: Configuration for the AI model.
        agent_config: Configuration for the agent behavior.
        confirmation_callback: Optional callback for sensitive action confirmation.
        takeover_callback: Optional callback for takeover requests. Should return
            user input as a string, which will be included in the next user message.

    Example:
        >>> from phone_agent.agent_tool import PhoneAgentV2
        >>> from phone_agent.model.client_tool import ModelConfig
        >>>
        >>> model_config = ModelConfig(base_url="http://localhost:8000/v1")
        >>> agent = PhoneAgentV2(model_config)
        >>> agent.run("Open WeChat and send a message to John")
    """

    def __init__(
        self,
        model_config: ModelConfig | None = None,
        agent_config: AgentConfig | None = None,
        confirmation_callback: Callable[[str], bool] | None = None,
        takeover_callback: Callable[[str], str] | None = None,
    ):
        self.agent_config = agent_config or AgentConfig()

        # Set tools in model config
        if model_config is None:
            model_config = ModelConfig()
        model_config.tools = self.agent_config.tools
        model_config.lang = self.agent_config.lang

        self.model_config = model_config
        self.model_client = ModelClient(self.model_config)
        self.action_handler = ActionHandler(
            device_id=self.agent_config.device_id,
            confirmation_callback=confirmation_callback,
            takeover_callback=takeover_callback,
        )

        self._context: list[dict[str, Any]] = []
        self._step_count = 0
        self._pending_takeover_input: str | None = None  # Store takeover user input

    def run(self, task: str) -> str:
        """
        Run the agent to complete a task.

        Args:
            task: Natural language description of the task.

        Returns:
            Final message from the agent.
        """
        self._context = []
        self._step_count = 0

        # First step with user prompt
        result = self._execute_step(task, is_first=True)

        if result.finished:
            # Save messages if enabled
            if self.agent_config.save_messages:
                saved_path = self.save_messages()
                if self.agent_config.verbose:
                    msgs = get_messages(self.agent_config.lang)
                    print(f"\n💾 {msgs.get('messages_saved', 'Messages saved to')}: {saved_path}")
                    print("=" * 50 + "\n")
            return result.observation or "Task completed"

        # Continue until finished or max steps reached
        while self._step_count < self.agent_config.max_steps:
            result = self._execute_step(is_first=False)

            if result.finished:
                # Save messages if enabled
                if self.agent_config.save_messages:
                    saved_path = self.save_messages()
                    if self.agent_config.verbose:
                        msgs = get_messages(self.agent_config.lang)
                        print(f"\n💾 {msgs.get('messages_saved', 'Messages saved to')}: {saved_path}")
                        print("=" * 50 + "\n")
                return result.observation or "Task completed"

        # Save messages even if max steps reached
        if self.agent_config.save_messages:
            saved_path = self.save_messages()
            if self.agent_config.verbose:
                msgs = get_messages(self.agent_config.lang)
                print(f"\n💾 {msgs.get('messages_saved', 'Messages saved to')}: {saved_path}")
                print("=" * 50 + "\n")

        return "Max steps reached"

    def step(self, task: str | None = None) -> StepResult:
        """
        Execute a single step of the agent.

        Useful for manual control or debugging.

        Args:
            task: Task description (only needed for first step).

        Returns:
            StepResult with step details.
        """
        is_first = len(self._context) == 0

        if is_first and not task:
            raise ValueError("Task is required for the first step")

        return self._execute_step(task, is_first)

    def reset(self) -> None:
        """Reset the agent state for a new task."""
        self._context = []
        self._step_count = 0
        self._pending_takeover_input = None

    def save_messages(self, output_path: str | None = None) -> str:
        """
        Save the complete conversation data to a JSON file, including:
        - Messages (conversation history)
        - Tools definitions
        - Model configuration
        - Agent configuration
        - Metadata (timestamp, steps, etc.)

        Args:
            output_path: Optional custom output path. If not provided,
                        will use auto-generated filename in configured output directory.

        Returns:
            Path to the saved file.
        """
        if not self._context:
            raise ValueError("No messages to save. Run a task first.")

        # Create output directory if it doesn't exist
        output_dir = self.agent_config.messages_output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Generate filename if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"conversation_{timestamp}.json")

        # Build complete conversation data
        conversation_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "agent_version": "v2",
                "total_steps": self._step_count,
                "device_id": self.agent_config.device_id,
            },
            "model_config": {
                "base_url": self.model_config.base_url,
                "model_name": self.model_config.model_name,
                "api_key": "***" if self.model_config.api_key != "EMPTY" else "EMPTY",  # Mask API key
                "max_tokens": self.model_config.max_tokens,
                "temperature": self.model_config.temperature,
                "top_p": self.model_config.top_p,
                "frequency_penalty": self.model_config.frequency_penalty,
                "enable_stream": self.model_config.enable_stream,
                "extra_body": self.model_config.extra_body,
            },
            "tools": self.agent_config.tools,
            "system_prompt": self.agent_config.system_prompt,
            "messages": self._context,
        }

        # Save to file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(conversation_data, f, ensure_ascii=False, indent=2)

        return output_path

    def _execute_step(
        self, user_prompt: str | None = None, is_first: bool = False
    ) -> StepResult:
        """Execute a single step of the agent loop."""
        self._step_count += 1

        # Capture current screen state
        device_factory = get_device_factory()
        screenshot = device_factory.get_screenshot(self.agent_config.device_id)
        current_app = device_factory.get_current_app(self.agent_config.device_id)

        # Build messages
        if is_first:
            # First turn: system + user (with task and image)
            self._context.append(
                MessageBuilder.create_system_message(self.agent_config.system_prompt)
            )

            screen_info = MessageBuilder.build_screen_info(current_app)
            text_content = f"{user_prompt}\n\n{screen_info}"

            self._context.append(
                MessageBuilder.create_user_message(
                    text=text_content, image_base64=screenshot.base64_data
                )
            )
        else:
            # Subsequent turns: screenshot is already in the previous observation message
            # Only add a user message if there's takeover input
            if self._pending_takeover_input:
                text_content = f"User provided information: {self._pending_takeover_input}"
                self._pending_takeover_input = None  # Clear after use
                self._context.append(
                    MessageBuilder.create_user_message(
                        text=text_content, image_base64=None  # No image, it's in observation
                    )
                )
            # Note: We don't add any message here if no takeover input
            # because the observation (with screenshot) was already added in previous turn

        # Remove old images from context to save space (keep recent N images)
        # This must be done BEFORE model request so model only sees recent N images
        self._remove_old_images_from_context()

        # Get model response
        try:
            msgs = get_messages(self.agent_config.lang)

            # Print header for thinking section
            if self.agent_config.verbose:
                print("\n" + "=" * 50)
                print(f"💭 {msgs['thinking']}:")
                print("-" * 50)

            # Request will print thinking in real-time during streaming
            response = self.model_client.request(self._context)

            # Print performance metrics
            if self.agent_config.verbose:
                print()
                print("=" * 50)
                print(f"⏱️  {msgs['performance_metrics']}:")
                print("-" * 50)
                if response.time_to_first_token is not None:
                    print(f"{msgs['time_to_first_token']}: {response.time_to_first_token:.3f}s")
                if response.time_to_thinking_end is not None:
                    print(f"{msgs['time_to_thinking_end']}:        {response.time_to_thinking_end:.3f}s")
                if response.total_time is not None:
                    print(f"{msgs['total_inference_time']}:          {response.total_time:.3f}s")
                print("=" * 50)
        except Exception as e:
            if self.agent_config.verbose:
                traceback.print_exc()
            return StepResult(
                success=False,
                finished=True,
                tool_calls=None,
                thinking="",
                observation=f"Model error: {e}",
            )

        # Check if finished (no tool calls)
        if response.finished:
            msgs = get_messages(self.agent_config.lang)

            # Parse content to extract thinking and final summary
            thinking_text = ""
            final_summary = ""

            if "</think>" in response.content:
                # Split by </think> tag
                parts = response.content.split("</think>", 1)
                # Extract thinking part (remove <think> tag)
                thinking_part = parts[0]
                if "<think>" in thinking_part:
                    thinking_text = thinking_part.split("<think>", 1)[1].strip()
                else:
                    thinking_text = thinking_part.strip()

                # Extract final summary (content after </think>)
                if len(parts) > 1:
                    final_summary = parts[1].strip()
            else:
                # No </think> tag, treat entire content as thinking
                thinking_text = response.content.replace("<think>", "").replace("</think>", "").strip()

            # Parse special tag from final_summary
            special_tag = None
            tag_message = final_summary
            for tag in SPECIAL_TAGS:
                if final_summary.startswith(tag):
                    special_tag = tag
                    tag_message = final_summary[len(tag):].strip()
                    break

            if self.agent_config.verbose:
                # Print thinking section
                if thinking_text:
                    print("\n" + "=" * 50)
                    print(f"💭 {msgs['thinking']}:")
                    print("-" * 50)
                    print(thinking_text)
                    print()

                # Print final summary section with special tag indicator
                if final_summary:
                    print("=" * 50)
                    if special_tag:
                        tag_emoji = {
                            "[finish]": "✅",
                            "[notool]": "💬",
                            "[sensitive]": "🔒",
                            "[captcha]": "🔐",
                            "[verification]": "⚠️",
                            "[preference]": "❓",
                            "[toxic]": "🚫",
                        }
                        print(f"{tag_emoji.get(special_tag, '📝')} {special_tag} {msgs.get('final_summary', 'Response')}:")
                    else:
                        print(f"📝 {msgs['final_summary']}:")
                    print("-" * 50)
                    print(tag_message if special_tag else final_summary)
                    print()

            # Handle special tags
            should_finish = True
            should_wait_user = False

            if special_tag in ("[sensitive]", "[captcha]"):
                # 敏感页面或验证码，暂停等待用户处理
                should_wait_user = True
                if self.agent_config.verbose:
                    print("=" * 50)
                    print("⏸️  " + msgs.get('waiting_user', 'Waiting for user to handle...'))
                    print("=" * 50 + "\n")

                # 如果有 takeover_callback，调用它
                if hasattr(self, 'action_handler') and self.action_handler.takeover_callback:
                    user_input = self.action_handler.takeover_callback(tag_message)
                    if user_input:
                        self._pending_takeover_input = user_input
                        should_finish = False  # 继续执行

            elif special_tag == "[verification]":
                # 敏感操作需要确认
                if self.agent_config.verbose:
                    print("=" * 50)
                    print("⚠️  " + msgs.get('need_confirm', 'Sensitive action requires confirmation'))
                    print("=" * 50 + "\n")

                # 如果有 confirmation_callback，调用它
                if hasattr(self, 'action_handler') and self.action_handler.confirmation_callback:
                    confirmed = self.action_handler.confirmation_callback(tag_message)
                    if confirmed:
                        # 用户确认，添加确认信息并继续
                        self._pending_takeover_input = "用户已确认执行该操作"
                        should_finish = False

            elif special_tag == "[preference]":
                # 需要用户输入或选择
                if self.agent_config.verbose:
                    print("=" * 50)
                    print("❓ " + msgs.get('need_input', 'User input required'))
                    print("=" * 50 + "\n")

                # 调用 takeover_callback 获取用户输入
                if hasattr(self, 'action_handler') and self.action_handler.takeover_callback:
                    user_input = self.action_handler.takeover_callback(tag_message)
                    if user_input:
                        self._pending_takeover_input = user_input
                        should_finish = False

            elif special_tag == "[toxic]":
                # 拒绝执行
                if self.agent_config.verbose:
                    print("=" * 50)
                    print("🚫 " + msgs.get('request_refused', 'Request refused due to policy violation'))
                    print("=" * 50 + "\n")
                should_finish = True

            elif special_tag in ("[finish]", "[notool]"):
                # 正常完成
                if self.agent_config.verbose:
                    print("=" * 50)
                    print()
                    print("✅ " + msgs['task_completed'])
                    print("=" * 50 + "\n")

            else:
                # 无特殊标签，正常完成
                if self.agent_config.verbose:
                    print("=" * 50)
                    print()
                    print("✅ " + msgs['task_completed'])
                    print("=" * 50 + "\n")

            # Add assistant response to context (preserve <think> tags)
            self._context.append(
                MessageBuilder.create_assistant_message(
                    content=response.content,
                    tool_calls=None,
                )
            )

            return StepResult(
                success=True,
                finished=should_finish,
                tool_calls=None,
                thinking=response.content,
                observation=response.content,
                special_tag=special_tag,
            )

        # Execute tool calls
        observations = []
        for tool_call in response.tool_calls:
            if self.agent_config.verbose:
                print("\n" + "🎯 " + msgs['action'] + ":")
                print(
                    json.dumps(
                        {
                            "name": tool_call["function"]["name"],
                            "arguments": json.loads(
                                tool_call["function"]["arguments"]
                            ),
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                print("=" * 50)

            try:
                result = self.action_handler.execute(
                    tool_call, screenshot.width, screenshot.height
                )

                # Check if observation contains takeover user input
                if "[TAKEOVER_USER_INPUT]" in result.observation:
                    import re
                    match = re.search(r'\[TAKEOVER_USER_INPUT\](.*?)\[/TAKEOVER_USER_INPUT\]', result.observation, re.DOTALL)
                    if match:
                        self._pending_takeover_input = match.group(1).strip()
                        # Remove the marker from observation for cleaner output
                        result.observation = re.sub(r'\n?\[TAKEOVER_USER_INPUT\].*?\[/TAKEOVER_USER_INPUT\]', '', result.observation, flags=re.DOTALL)

                observations.append(result.observation)

                # Print observation result
                if self.agent_config.verbose:
                    print(f"\n📋 {msgs['observation']}:")
                    print(f"   {result.observation}")
                    print("=" * 50)

                if result.should_finish:
                    # Add assistant response as two messages (matching training format)
                    thinking_content = response.content
                    if response.tool_calls:
                        tool_call = response.tool_calls[0]
                        tool_call_json = json.dumps({
                            "name": tool_call["function"]["name"],
                            "arguments": json.loads(tool_call["function"]["arguments"])
                        }, ensure_ascii=False)

                        self._context.append(
                            MessageBuilder.create_assistant_message(
                                content=thinking_content,
                                tool_calls=None,
                            )
                        )
                        self._context.append(
                            MessageBuilder.create_assistant_message(
                                content=tool_call_json,
                                tool_calls=None,
                            )
                        )
                    else:
                        self._context.append(
                            MessageBuilder.create_assistant_message(
                                content=thinking_content,
                                tool_calls=None,
                            )
                        )

                    return StepResult(
                        success=False,
                        finished=True,
                        tool_calls=response.tool_calls,
                        thinking=response.content,
                        observation=result.observation,
                    )
            except Exception as e:
                if self.agent_config.verbose:
                    traceback.print_exc()
                observations.append(f"Action failed: {e}")

        # Add assistant response to context as TWO separate messages (matching training format):
        # 1. First message: <think>...</think>
        # 2. Second message: {"name": "...", "arguments": {...}}

        # Extract thinking part (content before tool call)
        thinking_content = response.content

        # Build tool call JSON string
        if response.tool_calls:
            tool_call = response.tool_calls[0]  # Take first tool call
            tool_call_json = json.dumps({
                "name": tool_call["function"]["name"],
                "arguments": json.loads(tool_call["function"]["arguments"])
            }, ensure_ascii=False)

            # Add thinking message
            self._context.append(
                MessageBuilder.create_assistant_message(
                    content=thinking_content,
                    tool_calls=None,
                )
            )

            # Add tool call message (JSON only, no tool_calls field)
            self._context.append(
                MessageBuilder.create_assistant_message(
                    content=tool_call_json,
                    tool_calls=None,
                )
            )
        else:
            # No tool calls, just add thinking
            self._context.append(
                MessageBuilder.create_assistant_message(
                    content=thinking_content,
                    tool_calls=None,
                )
            )

        # Build observation text from all observations
        observation_text = "\n".join(observations)

        # Add system info to observation (matching training data format)
        # Format: {"success": true, "message": "...", "system": {"current_app": "...", "current_time": "..."}}
        device_factory = get_device_factory()
        current_app = device_factory.get_current_app(self.agent_config.device_id)
        from datetime import datetime
        now = datetime.now()
        weekday_names = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
        current_time = now.strftime("%Y-%m-%d %H:%M:%S") + ", " + weekday_names[now.weekday()]

        # Build observation JSON matching training format
        observation_json = {
            "success": True,
            "message": observation_text,
            "screenshot": False,  # Will be updated below
            "data": None,
            "system": {
                "current_app": current_app,
                "current_time": current_time
            }
        }

        # Add observation with optional screenshot to context
        # Training data pattern: GUI operations include screenshots,
        # while non-GUI operations (TodoWrite, Skill, etc.) don't include screenshots

        # Check if any tool call is a GUI operation that needs screenshot
        # GUI tools: mcp__phone-use__*, mcp__phone-api__*, or simplified names like Launch, Tap, Swipe, etc.
        GUI_TOOL_NAMES = {
            "mcp__phone-use__tap", "mcp__phone-use__swipe", "mcp__phone-use__type",
            "mcp__phone-use__launch", "mcp__phone-use__back", "mcp__phone-use__home",
            "mcp__phone-use__wait", "mcp__phone-use__longpress", "mcp__phone-use__doubleclick",
            "mcp__phone-use__take_screen_shot",
            # Simplified names (model may output these)
            "Tap", "tap", "Swipe", "swipe", "Type", "type", "Launch", "launch",
            "Back", "back", "Home", "home", "Wait", "wait", "LongPress", "longpress",
            "DoubleClick", "doubleclick", "take_screen_shot",
        }
        NON_GUI_TOOL_NAMES = {"TodoWrite", "Skill"}

        needs_screenshot = False
        if response.tool_calls:
            for tc in response.tool_calls:
                tool_name = tc["function"]["name"]
                # Check if it's a non-GUI operation first
                if tool_name in NON_GUI_TOOL_NAMES:
                    continue
                # Check if it's a known GUI operation or starts with mcp__phone
                if tool_name in GUI_TOOL_NAMES or tool_name.startswith("mcp__phone-use__") or tool_name.startswith("mcp__phone-api__"):
                    needs_screenshot = True
                    break

        if needs_screenshot:
            new_screenshot = device_factory.get_screenshot(self.agent_config.device_id)
            observation_json["screenshot"] = True
            # Format observation as JSON string
            observation_str = json.dumps(observation_json, ensure_ascii=False)
            self._context.append(
                MessageBuilder.create_observation_message(
                    observation=observation_str,
                    image_base64=new_screenshot.base64_data
                )
            )
        else:
            # Non-GUI operations don't need screenshot
            observation_json["screenshot"] = False
            observation_str = json.dumps(observation_json, ensure_ascii=False)
            self._context.append(
                MessageBuilder.create_observation_message(
                    observation=observation_str,
                    image_base64=None
                )
            )

        return StepResult(
            success=True,
            finished=False,
            tool_calls=response.tool_calls,
            thinking=response.content,
            observation=observation_str,
        )

    def _remove_old_images_from_context(self) -> None:
        """
        Remove images from old user messages, keeping only the most recent N images.

        This helps reduce context size while maintaining recent visual information.
        """
        # Find all user messages with images
        user_message_indices = []
        for i, msg in enumerate(self._context):
            if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                # Check if this message has an image
                has_image = any(
                    item.get("type") == "image_url" for item in msg["content"]
                )
                if has_image:
                    user_message_indices.append(i)

        # Calculate how many images to remove
        if len(user_message_indices) <= self.agent_config.keep_recent_images:
            # No need to remove any images
            return

        # Remove images from old messages (keep only the most recent N)
        num_to_remove = len(user_message_indices) - self.agent_config.keep_recent_images
        for i in range(num_to_remove):
            msg_idx = user_message_indices[i]
            self._context[msg_idx] = MessageBuilder.remove_images_from_message(
                self._context[msg_idx]
            )

    @property
    def context(self) -> list[dict[str, Any]]:
        """Get the current conversation context."""
        return self._context.copy()

    @property
    def step_count(self) -> int:
        """Get the current step count."""
        return self._step_count
