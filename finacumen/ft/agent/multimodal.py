"""
MultimodalAgent - 多模态任务专用智能体

使用 Qwen3-VL-8B-Instruct 等视觉模型处理带图片的输入。
仅在多模态输入和任务时被调用，不影响纯文本流程。

当 vision 模型不返回原生 tool_calls 时，think() 从 content 中解析
"Thought / Action / 代码块" 格式的文本输出，转为 synthetic tool_calls。
"""
import json
import re
from typing import List, Optional

from pydantic import Field

from finacumen.ft.agent.toolcall import ToolCallAgent
from finacumen.ft.config import config
from finacumen.ft.llm import LLM
from finacumen.ft.prompt.multimodal import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from finacumen.ft.schema import Function, Message, ToolCall
from finacumen.ft.skill import FinanceExtractionSkill
from finacumen.ft.tool import OcrExtract, Terminate, ToolCollection
from finacumen.ft.tool.python_execute import PythonExecute
from finacumen.ft.tool.str_replace_editor import StrReplaceEditor


class MultimodalAgent(ToolCallAgent):
    """
    多模态任务专用智能体

    使用 vision 配置的 LLM（如 Qwen3-VL-8B-Instruct）处理图片+文本输入。
    能力：图像理解、图表数据提取、视觉证据转文本、多模态推理。
    """

    name: str = "Multimodal"
    description: str = (
        "A specialized multimodal agent for image-text tasks: chart/table extraction, "
        "visual evidence understanding, and vision-language reasoning. Uses vision models. "
        "Prefer direct visual reading plus python_execute for clearly labeled charts and simple visual facts. "
        "Use finance_extraction_skill only for dense structured tables, row-column intersections, or OCR-friendly footnote tables."
    )

    system_prompt: str = SYSTEM_PROMPT.format(directory=config.workspace_root)
    next_step_prompt: str = NEXT_STEP_PROMPT

    max_observe: int = 15000
    max_steps: int = 20

    # 使用 vision 配置的 LLM（Qwen3-VL-8B-Instruct 等）
    llm: LLM = Field(default_factory=lambda: LLM(config_name="vision"))

    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            FinanceExtractionSkill(),
            OcrExtract(),
            PythonExecute(),
            StrReplaceEditor(),
            Terminate(),
        )
    )
    special_tool_names: list[str] = Field(default_factory=lambda: [Terminate().name])

    async def think(self) -> bool:
        """Process current state; parse 'Action:' text format if no native tool_calls."""
        result = await super().think()

        if self.tool_calls or not result:
            return result

        content = ""
        for msg in reversed(self.memory.messages):
            if getattr(msg, "role", "") == "assistant":
                content = msg.content or ""
                break
        if not content:
            return result

        parsed = self._parse_action_text(content)
        if not parsed:
            return result

        self.tool_calls = parsed
        for i in range(len(self.memory.messages) - 1, -1, -1):
            if getattr(self.memory.messages[i], "role", "") == "assistant":
                self.memory.messages[i] = Message.from_tool_calls(
                    content=content, tool_calls=parsed
                )
                break
        self._consecutive_zero_tool_rounds = 0
        return True

    def _parse_action_text(self, content: str) -> Optional[List[ToolCall]]:
        """Parse 'Action: tool_name' text format into synthetic ToolCall objects."""
        content = content or ""
        action_match = re.search(
            r"Action:\s*(finance_extraction_skill|ocr_extract|python_execute|terminate)",
            content,
            re.IGNORECASE,
        )
        if not action_match:
            return None
        action = action_match.group(1).lower()

        if action == "finance_extraction_skill":
            vars_match = re.search(
                r"[Vv]ariables?\s*[:\[]\s*\[([^\]]*)\]", content, re.IGNORECASE
            )
            if vars_match:
                vars_str = vars_match.group(1)
                variables = [v.strip().strip('"\'') for v in vars_str.split(",") if v.strip()]
            else:
                vars_match = re.search(
                    r"[Vv]ariables?\s*[:=]\s*([^\n]+)", content, re.IGNORECASE
                )
                if not vars_match:
                    return None
                variables = [
                    v.strip().strip('"\'')
                    for v in vars_match.group(1).split(",")
                    if v.strip()
                ]
            if not variables:
                return None
            return [
                ToolCall(
                    id="fallback_multimodal_1",
                    type="function",
                    function=Function(
                        name="finance_extraction_skill",
                        arguments=json.dumps(
                            {"variables": variables, "use_context_image": True}
                        ),
                    ),
                )
            ]
        if action == "ocr_extract":
            return [
                ToolCall(
                    id="fallback_multimodal_1",
                    type="function",
                    function=Function(
                        name="ocr_extract",
                        arguments=json.dumps({"use_context_image": True}),
                    ),
                )
            ]
        if action == "python_execute":
            code_match = re.search(r"```python\s*\n(.*?)```", content, re.DOTALL)
            if not code_match:
                return None
            code = code_match.group(1).strip()
            if not code:
                return None
            return [
                ToolCall(
                    id="fallback_multimodal_1",
                    type="function",
                    function=Function(
                        name="python_execute",
                        arguments=json.dumps({"code": code}),
                    ),
                )
            ]
        if action == "terminate":
            status = "failure" if re.search(r"failure|DATA_UNREADABLE", content, re.I) else "success"
            return [
                ToolCall(
                    id="fallback_multimodal_1",
                    type="function",
                    function=Function(
                        name="terminate",
                        arguments=json.dumps({"status": status}),
                    ),
                )
            ]
        return None
