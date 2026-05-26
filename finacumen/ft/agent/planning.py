from typing import List, Optional

from pydantic import Field, model_validator

from finacumen.ft.agent.toolcall import ToolCallAgent
from finacumen.ft.config import config
from finacumen.ft.llm import LLM
from finacumen.ft.prompt.planning import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from finacumen.ft.tool import Terminate, ToolCollection, WorkflowStateTool


class PlanningAgent(ToolCallAgent):
    """
    专注创建执行计划的规划 Agent。

    工具范围严格限定：仅 WorkflowStateTool（创建/更新计划）+ Terminate（结束规划）。
    不包含 PythonExecute、StrReplaceEditor 等执行类工具。
    """

    name: str = "Planning"
    description: str = (
        "A specialized planning agent that decomposes tasks into actionable steps "
        "for flow-controlled execution."
    )

    system_prompt: str = SYSTEM_PROMPT.format(directory=config.workspace_root)
    next_step_prompt: str = NEXT_STEP_PROMPT

    max_observe: int = 10000
    max_steps: int = 12
    enable_final_answer: bool = False

    workflow_state_tool: Optional[WorkflowStateTool] = Field(default=None)
    available_tools: Optional[ToolCollection] = None
    special_tool_names: list[str] = Field(default_factory=lambda: [Terminate().name])

    # 多模态任务时使用 vision 模型，以便 Planning 能阅读图像并生成更准确的提取步骤
    llm_vision: Optional[LLM] = Field(default_factory=lambda: LLM(config_name="vision"))

    async def run(
        self,
        request: Optional[str] = None,
        base64_image: Optional[str] = None,
        base64_images: Optional[List[str]] = None,
    ) -> str:
        """多模态任务时使用 vision 模型并传入图像，纯文本任务保持原逻辑。支持多图。"""
        has_images = base64_images or base64_image
        if has_images and self.llm_vision is not None:
            original_llm = self.llm
            self.llm = self.llm_vision
            try:
                return await super().run(
                    request,
                    base64_image=base64_image,
                    base64_images=base64_images,
                )
            finally:
                self.llm = original_llm
        return await super().run(
            request,
            base64_image=base64_image,
            base64_images=base64_images,
        )

    @model_validator(mode="after")
    def init_tools(self) -> "PlanningAgent":
        """
        规划 Agent 仅持有：WorkflowStateTool（创建计划）+ Terminate。
        使用 to_planning_param 描述，鼓励 LLM 调用工具创建计划。
        """
        if self.available_tools is None:
            if self.workflow_state_tool is not None:
                self.available_tools = ToolCollection(
                    self.workflow_state_tool,
                    Terminate(),
                    use_planning_param_for="workflow_state",
                )
            else:
                self.available_tools = ToolCollection(Terminate())
        return self
