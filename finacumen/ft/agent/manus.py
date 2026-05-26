"""
Manus Agent - 精简版通用执行模板

这个版本是当前仓库推荐的通用模板，目标是：
1. 给 `main.py` 提供一个稳定的单智能体执行器；
2. 作为后续构建领域 Agent（例如金融 Agent）的参考模板。

重要说明（架构演进）：
- Manus 已从当前多智能体执行架构中移除，以降低执行器冗余和路由歧义。
- 本文件代码被刻意保留，作为未来新增执行 Agent 的模板。
- 若后续扩展新 Agent，建议复用本类的结构（身份字段、prompt 注入、工具集、special tool）。
"""

from pydantic import Field

from finacumen.ft.agent.toolcall import ToolCallAgent
from finacumen.ft.config import config
from finacumen.ft.prompt.manus import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from finacumen.ft.tool import Terminate, ToolCollection
from finacumen.ft.tool.python_execute import PythonExecute
from finacumen.ft.tool.str_replace_editor import StrReplaceEditor


class Manus(ToolCallAgent):
    """
    通用执行 Agent（面向精简框架）

    说明（模板用途）：
    - 保留 OpenManus 的核心执行模式（ToolCall + ReAct）。
    - 移除已下线分支（MCP、浏览器上下文、沙箱）相关复杂逻辑。
    - 默认开启最终答复收口（由 ToolCallAgent 统一处理）。
    - 当前默认不在多智能体流程中注册，仅作为扩展模板与单智能体入口保留。
    """

    # 1) 基础身份信息
    name: str = "Manus"
    description: str = (
        "A compact general-purpose OpenManus agent for tool-driven task execution."
    )

    # 2) Prompt 配置
    system_prompt: str = SYSTEM_PROMPT.format(directory=config.workspace_root)
    next_step_prompt: str = NEXT_STEP_PROMPT

    # 3) 执行约束
    max_observe: int = 10000
    max_steps: int = 20

    # 4) 默认工具集（保持通用但尽量精简）
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            PythonExecute(),
            StrReplaceEditor(),
            Terminate(),
        )
    )

    # 5) 特殊工具：命中后会将 Agent 状态置为 FINISHED
    special_tool_names: list[str] = Field(default_factory=lambda: [Terminate().name])

    @classmethod
    async def create(cls, **kwargs) -> "Manus":
        """
        异步工厂方法，保留统一调用风格。

        当前精简版本没有额外异步初始化动作，
        但保留该接口可避免上层入口逻辑频繁改动。
        """
        return cls(**kwargs)
