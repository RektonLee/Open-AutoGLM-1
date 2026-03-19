"""Phone agent for Qwen3.5 model with specialized parsing."""

from typing import Callable

from phone_agent.agent_tool import PhoneAgentV2, AgentConfig
from phone_agent.actions.handler_qwen35 import ActionHandlerQwen35
from phone_agent.model.client_qwen35 import ModelClientQwen35, ModelConfigQwen35


class PhoneAgentQwen35(PhoneAgentV2):
    """
    Phone agent specialized for Qwen3.5 model.

    Uses Qwen3.5-specific client and handler to properly parse
    <function=...><parameter=...> format.
    """

    def __init__(
        self,
        model_config: ModelConfigQwen35,
        agent_config: AgentConfig,
        confirmation_callback: Callable[[str], bool] | None = None,
        takeover_callback: Callable[[str], str] | None = None,
    ):
        # 保存配置
        self.agent_config = agent_config or AgentConfig()

        # 设置 model_config 的 tools 和 lang
        model_config.tools = self.agent_config.tools
        model_config.lang = self.agent_config.lang

        self.model_config = model_config

        # 使用 Qwen3.5 专用的 client
        self.model_client = ModelClientQwen35(model_config)

        # 使用 Qwen3.5 专用的 handler
        self.action_handler = ActionHandlerQwen35(
            device_id=self.agent_config.device_id,
            confirmation_callback=confirmation_callback,
            takeover_callback=takeover_callback,
        )

        # 初始化上下文和状态
        self._context: list[dict] = []
        self._step_count = 0
        self._pending_takeover_input: str | None = None
