"""Action handler for processing AI model outputs in Tool Calls format."""

import json
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Callable

from phone_agent.config.timing import TIMING_CONFIG
from phone_agent.device_factory import get_device_factory


@dataclass
class ActionResult:
    """Result of an action execution."""

    success: bool
    observation: str  # Observation text to be added to context
    should_finish: bool = False
    requires_confirmation: bool = False


class ActionHandler:
    """
    Handles execution of actions from AI model tool calls (v2).

    Args:
        device_id: Optional ADB device ID for multi-device setups.
        confirmation_callback: Optional callback for sensitive action confirmation.
            Should return True to proceed, False to cancel.
        takeover_callback: Optional callback for takeover requests (login, captcha).
            Should return user input as a string, which will be included in the next
            user message to the model.
    """

    def __init__(
        self,
        device_id: str | None = None,
        confirmation_callback: Callable[[str], bool] | None = None,
        takeover_callback: Callable[[str], str] | None = None,
    ):
        self.device_id = device_id
        self.confirmation_callback = confirmation_callback or self._default_confirmation
        self.takeover_callback = takeover_callback or self._default_takeover

    def execute(
        self, tool_call: dict[str, Any], screen_width: int, screen_height: int
    ) -> ActionResult:
        """
        Execute a tool call from the AI model.

        Args:
            tool_call: The tool call dictionary from the model.
            screen_width: Current screen width in pixels.
            screen_height: Current screen height in pixels.

        Returns:
            ActionResult with observation text.
        """
        function_name = tool_call["function"]["name"]
        arguments_str = tool_call["function"]["arguments"]

        # Parse arguments
        try:
            arguments = json.loads(arguments_str)
        except json.JSONDecodeError as e:
            return ActionResult(
                success=False,
                observation=f"Failed to parse arguments: {e}",
                should_finish=True,
            )

        handler_method = self._get_handler(function_name)

        if handler_method is None:
            return ActionResult(
                success=False,
                observation=f"Unknown action: {function_name}",
                should_finish=True,
            )

        try:
            return handler_method(arguments, screen_width, screen_height)
        except Exception as e:
            return ActionResult(
                success=False,
                observation=f"Action failed: {e}",
                should_finish=False,
            )

    def _get_handler(self, action_name: str) -> Callable | None:
        """Get the handler method for an action."""
        handlers = {
            # Simplified names (legacy)
            "Tap": self._handle_tap,
            "DoubleTap": self._handle_double_tap,
            "Launch": self._handle_launch,
            "Swipe": self._handle_swipe,
            "Back": self._handle_back,
            "Home": self._handle_home,
            "LongPress": self._handle_long_press,
            "Type": self._handle_type,
            "Take_over": self._handle_takeover,
            "Wait": self._handle_wait,
            "TodoWrite": self._handle_todowrite,
            # MCP format names (training data format)
            "mcp__phone-use__tap": self._handle_tap,
            "mcp__phone-use__longpress": self._handle_long_press,
            "mcp__phone-use__doubleclick": self._handle_double_tap,
            "mcp__phone-use__type": self._handle_type,
            "mcp__phone-use__swipe": self._handle_swipe,
            "mcp__phone-use__launch": self._handle_launch,
            "mcp__phone-use__back": self._handle_back,
            "mcp__phone-use__home": self._handle_home,
            "mcp__phone-use__wait": self._handle_wait,
            "mcp__phone-use__take_screen_shot": self._handle_take_screenshot,
            # MCP API and search tools
            "mcp__phone-api__execute_api": self._handle_execute_api,
            "mcp__phone-search__web_search": self._handle_web_search,
            "mcp__phone-search__web_fetch": self._handle_web_fetch,
            # Skill tool
            "Skill": self._handle_skill,
        }
        return handlers.get(action_name)

    def _convert_coordinate(
        self, coordinate: list[int], screen_width: int, screen_height: int
    ) -> tuple[int, int]:
        """Convert relative coordinates (0-999) to absolute pixels."""
        # Ensure coordinate values are integers (defensive programming)
        x = int(float(coordinate[0]) * screen_width // 999)
        y = int(float(coordinate[1]) * screen_height // 999)
        return x, y

    def _handle_tap(
        self, arguments: dict, width: int, height: int
    ) -> ActionResult:
        """Handle tap action."""
        coordinate = arguments.get("coordinate")
        if not coordinate:
            return ActionResult(
                False, "No coordinate specified", should_finish=False
            )

        x, y = self._convert_coordinate(coordinate, width, height)

        device_factory = get_device_factory()
        device_factory.tap(x, y, self.device_id)

        return ActionResult(
            True, f"Tapped at ({coordinate[0]}, {coordinate[1]})"
        )

    def _handle_double_tap(
        self, arguments: dict, width: int, height: int
    ) -> ActionResult:
        """Handle double tap action."""
        coordinate = arguments.get("coordinate")
        if not coordinate:
            return ActionResult(
                False, "No coordinate specified", should_finish=False
            )

        x, y = self._convert_coordinate(coordinate, width, height)

        device_factory = get_device_factory()
        device_factory.double_tap(x, y, self.device_id)

        return ActionResult(
            True, f"Double Tapped at ({coordinate[0]}, {coordinate[1]})"
        )

    def _handle_launch(
        self, arguments: dict, width: int, height: int
    ) -> ActionResult:
        """Handle app launch action."""
        # Support both "app_name" (legacy) and "name" (MCP format)
        app_name = arguments.get("app_name") or arguments.get("name")
        if not app_name:
            return ActionResult(False, "No app name specified", should_finish=False)

        device_factory = get_device_factory()
        success = device_factory.launch_app(app_name, self.device_id)

        if success:
            return ActionResult(True, f"Launched {app_name}")
        return ActionResult(
            False, f"App not found: {app_name}", should_finish=False
        )

    def _handle_swipe(
        self, arguments: dict, width: int, height: int
    ) -> ActionResult:
        """Handle swipe action."""
        start_coord = arguments.get("start_coordinate")
        end_coord = arguments.get("end_coordinate")

        if not start_coord or not end_coord:
            return ActionResult(
                False, "Missing swipe coordinates", should_finish=False
            )

        start_x, start_y = self._convert_coordinate(start_coord, width, height)
        end_x, end_y = self._convert_coordinate(end_coord, width, height)

        device_factory = get_device_factory()
        device_factory.swipe(
            start_x, start_y, end_x, end_y, device_id=self.device_id
        )

        return ActionResult(
            True,
            f"Swiped from [{start_coord[0]}, {start_coord[1]}] to [{end_coord[0]}, {end_coord[1]}]",
        )

    def _handle_back(
        self, arguments: dict, width: int, height: int
    ) -> ActionResult:
        """Handle back button action."""
        device_factory = get_device_factory()
        device_factory.back(self.device_id)
        return ActionResult(True, "Back to previous screen")

    def _handle_home(
        self, arguments: dict, width: int, height: int
    ) -> ActionResult:
        """Handle home button action."""
        device_factory = get_device_factory()
        device_factory.home(self.device_id)
        return ActionResult(True, "Return to the system homepage")

    def _handle_long_press(
        self, arguments: dict, width: int, height: int
    ) -> ActionResult:
        """Handle long press action."""
        coordinate = arguments.get("coordinate")
        if not coordinate:
            return ActionResult(
                False, "No coordinate specified", should_finish=False
            )

        x, y = self._convert_coordinate(coordinate, width, height)

        device_factory = get_device_factory()
        device_factory.long_press(x, y, device_id=self.device_id)

        return ActionResult(
            True,
            f"Long Pressed ({coordinate[0]}, {coordinate[1]}) for 2 seconds",
        )

    def _handle_type(
        self, arguments: dict, width: int, height: int
    ) -> ActionResult:
        """Handle text input action."""
        text = arguments.get("text", "")

        # Ensure text is a string (handle cases where LLM returns numbers)
        if not isinstance(text, str):
            text = str(text)

        device_factory = get_device_factory()

        # Switch to ADB keyboard
        original_ime = device_factory.detect_and_set_adb_keyboard(self.device_id)
        time.sleep(TIMING_CONFIG.action.keyboard_switch_delay)

        # Clear existing text and type new text
        device_factory.clear_text(self.device_id)
        time.sleep(TIMING_CONFIG.action.text_clear_delay)

        # Handle multiline text by splitting on newlines
        device_factory.type_text(text, self.device_id)
        time.sleep(TIMING_CONFIG.action.text_input_delay)

        # Restore original keyboard
        device_factory.restore_keyboard(original_ime, self.device_id)
        time.sleep(TIMING_CONFIG.action.keyboard_restore_delay)

        return ActionResult(True, f"Typed {text}")

    def _handle_takeover(
        self, arguments: dict, width: int, height: int
    ) -> ActionResult:
        """Handle takeover request (login, captcha, etc.)."""
        message = arguments.get("message", "User intervention required")
        user_input = self.takeover_callback(message)
        # Include user input in observation with special marker for agent to pick up
        observation = f"Takeover completed: {message}"
        if user_input:
            observation += f"\n[TAKEOVER_USER_INPUT]{user_input}[/TAKEOVER_USER_INPUT]"
        return ActionResult(True, observation, should_finish=False)

    def _handle_wait(
        self, arguments: dict, width: int, height: int
    ) -> ActionResult:
        """Handle wait action."""
        # Support both "duration" (legacy) and "seconds" (MCP format)
        duration = arguments.get("duration") or arguments.get("seconds", 1.0)
        # Ensure duration is a float (defensive programming)
        duration = float(duration)

        time.sleep(duration)
        return ActionResult(True, f"Waited {duration} seconds")

    def _handle_todowrite(
        self, arguments: dict, width: int, height: int
    ) -> ActionResult:
        """Handle TodoWrite action - no actual execution, just acknowledgment."""
        todos = arguments.get("todos", [])
        # TodoWrite is only for context management, no actual execution
        return ActionResult(True, f"TodoWrite recorded with {len(todos)} items")

    @staticmethod
    def _default_confirmation(message: str) -> bool:
        """Default confirmation callback using console input."""
        response = input(f"Sensitive operation: {message}\nConfirm? (Y/N): ")
        return response.upper() == "Y"

    @staticmethod
    def _default_takeover(message: str) -> str:
        """Default takeover callback using console input.

        Returns:
            User input string, or empty string if no input provided.
        """
        user_input = input(f"{message}\nPlease enter any information to provide to the agent (or press Enter to skip): ")
        return user_input

    def _handle_take_screenshot(
        self, arguments: dict, width: int, height: int
    ) -> ActionResult:
        """Handle take_screen_shot action - screenshot is taken automatically by agent."""
        # Screenshot is already handled by the agent loop, this is just for completion
        return ActionResult(True, "Screenshot taken")

    def _handle_execute_api(
        self, arguments: dict, width: int, height: int
    ) -> ActionResult:
        """Handle mcp__phone-api__execute_api action."""
        api_name = arguments.get("api_name", "")
        parameters = arguments.get("parameters", {})
        # TODO: Implement actual API execution
        return ActionResult(True, f"API {api_name} executed with parameters: {parameters}")

    def _handle_web_search(
        self, arguments: dict, width: int, height: int
    ) -> ActionResult:
        """Handle mcp__phone-search__web_search action."""
        query = arguments.get("query", "")
        # TODO: Implement actual web search
        return ActionResult(True, f"Web search for: {query}")

    def _handle_web_fetch(
        self, arguments: dict, width: int, height: int
    ) -> ActionResult:
        """Handle mcp__phone-search__web_fetch action."""
        url = arguments.get("url", "")
        # TODO: Implement actual web fetch
        return ActionResult(True, f"Fetched URL: {url}")

    def _handle_skill(
        self, arguments: dict, width: int, height: int
    ) -> ActionResult:
        """Handle Skill tool call."""
        skill = arguments.get("skill", "")
        args = arguments.get("args", "")
        # TODO: Implement actual skill execution
        return ActionResult(True, f"Skill {skill} invoked with args: {args}")
