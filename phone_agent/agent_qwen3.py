"""Phone agent for Qwen3 model with standard tool calls support."""

from typing import Callable

from phone_agent.agent_tool import PhoneAgentV2, AgentConfig
from phone_agent.actions.handler_tool import ActionHandler
from phone_agent.model.client_tool import ModelClient, ModelConfig


class PhoneAgentQwen3(PhoneAgentV2):
    """
    Phone agent specialized for Qwen3 model.

    Uses standard OpenAI-compatible tool calls format.
    """

    def __init__(
        self,
        model_config: ModelConfig,
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

        # 使用标准的 tool client
        self.model_client = ModelClient(model_config)

        # 使用标准的 action handler
        self.action_handler = ActionHandler(
            device_id=self.agent_config.device_id,
            confirmation_callback=confirmation_callback,
            takeover_callback=takeover_callback,
        )

        # 初始化上下文和状态
        self._context: list[dict] = []
        self._step_count = 0
        self._pending_takeover_input: str | None = None
