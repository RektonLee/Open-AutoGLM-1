"""Action handler for Qwen3.5 model outputs."""

from typing import Callable

from phone_agent.actions.handler_tool import ActionHandler, ActionResult


class ActionHandlerQwen35(ActionHandler):
    """
    Handles execution of actions from Qwen3.5 model outputs.

    Qwen3.5 uses <function=...><parameter=...> format in its chat template,
    so we need to clean up the function name format.
    """

    def _get_handler(self, action_name: str) -> Callable | None:
        """Get the handler method for an action, with Qwen3.5 format cleanup."""
        # 清理 <function=...> 格式 (Qwen3.5 chat template 输出格式)
        if action_name.startswith("<function=") and action_name.endswith(">"):
            action_name = action_name[10:-1]  # 去掉 <function= 和 >
        elif action_name.startswith("<function="):
            action_name = action_name[10:]  # 只去掉 <function=

        # 调用父类的 _get_handler
        return super()._get_handler(action_name)
