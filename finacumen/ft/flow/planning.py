"""
PlanningFlow - 基于 archive/app/flow/planning.py 的简洁实现。

保持原始逻辑：规划阶段使用 to_planning_param() 鼓励 LLM 调用工具，
完整 plan_status 传给 executor，复用同一 executor，步骤执行清晰。
"""
import json
import re
import time
from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import Field

from finacumen.ft.agent.base import BaseAgent
from finacumen.ft.flow.base import BaseFlow
from finacumen.ft.flow.extraction_state_bridge import (
    get_variable_semantic_type,
    reset_variable_semantic_registry,
    set_shared_python_execute,
)
from finacumen.ft.llm import LLM
from finacumen.ft.logger import logger
from finacumen.ft.schema import AgentState, Message, ToolChoice
from finacumen.ft.tool import PlanningTool, ToolCollection
from finacumen.ft.tool.anti_loop import AntiLoopInterceptor
from finacumen.ft.tool.ocr import set_step_context, set_step_images_for_ocr
from finacumen.ft.tool.python_execute import PythonExecute


class PlanStepStatus(str, Enum):
    """Enum class defining possible statuses of a plan step"""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"

    @classmethod
    def get_all_statuses(cls) -> list[str]:
        return [status.value for status in cls]

    @classmethod
    def get_active_statuses(cls) -> list[str]:
        return [cls.NOT_STARTED.value, cls.IN_PROGRESS.value]

    @classmethod
    def get_status_marks(cls) -> Dict[str, str]:
        return {
            cls.COMPLETED.value: "[✓]",
            cls.IN_PROGRESS.value: "[→]",
            cls.BLOCKED.value: "[!]",
            cls.NOT_STARTED.value: "[ ]",
        }


_RATE_NAME_RE = re.compile(
    r"(rate|margin|yield|percentage|growth|return|ratio|discount|interest|coupon|spread|premium)",
    re.IGNORECASE,
)
_CURRENCY_NAME_RE = re.compile(
    r"(price|cost|value|amount|fee|revenue|income|expense|salary|wage|"
    r"payment|proceeds|principal|balance|deposit|dividend|earning|profit|loss|"
    r"cash|debt|asset|liability|equity|capital|fund|budget|rent|tax)",
    re.IGNORECASE,
)
_COUNT_NAME_RE = re.compile(
    r"(count|number|shares|units|quantity|days|months|years|periods|n_)",
    re.IGNORECASE,
)


class PlanningFlow(BaseFlow):
    """A flow that manages planning and execution of tasks using agents."""

    llm: LLM = Field(default_factory=lambda: LLM())
    planning_tool: PlanningTool = Field(default_factory=PlanningTool)
    executor_keys: List[str] = Field(default_factory=list)
    active_plan_id: str = Field(default_factory=lambda: f"plan_{int(time.time())}")
    current_step_index: Optional[int] = None
    base64_images: Optional[List[str]] = None  # 多模态输入，execute() 时设置
    _shared_python_execute: Optional[PythonExecute] = None  # 跨 Agent 共享的 PythonExecute 实例
    step_handoff_records: List[Dict[str, str]] = Field(default_factory=list, exclude=True)

    def __init__(
        self, agents: Union[BaseAgent, List[BaseAgent], Dict[str, BaseAgent]], **data
    ):
        if "executors" in data:
            data["executor_keys"] = data.pop("executors")
        if "plan_id" in data:
            data["active_plan_id"] = data.pop("plan_id")
        if "workflow_state_tool" in data:
            data["planning_tool"] = data.pop("workflow_state_tool")
        if "planning_tool" not in data:
            data["planning_tool"] = PlanningTool()
        super().__init__(agents, **data)
        if not self.executor_keys:
            self.executor_keys = list(self.agents.keys())

    def get_executor(self, step_type: Optional[str] = None) -> BaseAgent:
        """Get an appropriate executor agent for the current step."""
        if step_type and step_type in self.agents:
            return self.agents[step_type]
        for key in self.executor_keys:
            if key in self.agents:
                return self.agents[key]
        return self.primary_agent

    def _ensure_shared_python_execute(self) -> PythonExecute:
        """确保所有 Agent 共享同一个 PythonExecute 实例，实现变量跨步骤持久化。
        
        问题背景：FinanceAgent 和 MultimodalAgent 使用 Field(default_factory) 创建 ToolCollection，
        导致每次访问 available_tools 都返回新的 PythonExecute 实例，变量无法跨 Agent 共享。
        
        解决方案：在 Flow 级别创建共享的 PythonExecute 实例，并注入到所有 Agent 的 ToolCollection 中。
        """
        if self._shared_python_execute is None:
            self._shared_python_execute = PythonExecute()
        return self._shared_python_execute

    def _remove_shared_python_variables(self, variable_names: List[str]) -> None:
        """Remove accidentally created variables from the shared python state."""
        if not variable_names:
            return
        shared_python = self._ensure_shared_python_execute()
        for variable_name in variable_names:
            shared_python._global_env.pop(variable_name, None)

    def _snapshot_user_variables(self) -> Dict[str, str]:
        """Snapshot user-defined scalar variables from the shared Python environment."""
        if self._shared_python_execute is None:
            return {}
        env = self._shared_python_execute._global_env
        snapshot: Dict[str, str] = {}
        for k, v in env.items():
            if k.startswith("_") or k == "__builtins__":
                continue
            if callable(v) or isinstance(v, type):
                continue
            try:
                if isinstance(v, (int, float, str, bool)):
                    snapshot[k] = repr(v)
                elif isinstance(v, (list, dict, tuple)):
                    s = repr(v)
                    if len(s) < 200:
                        snapshot[k] = s
            except Exception:
                continue
        return snapshot

    # ---- Variable semantic classification (Phase 1: semantic metadata) ----

    @staticmethod
    def _infer_variable_semantic_type(
        name: str, value_repr: str, step_text: str = ""
    ) -> str:
        """Infer the semantic type of a variable from its name, value, and step context.

        Returns one of: "rate/percentage", "currency/price", "count/quantity", "general".
        """
        name_lower = name.lower()
        try:
            val = float(value_repr.strip("'\""))
        except (ValueError, TypeError):
            val = None

        if _RATE_NAME_RE.search(name_lower):
            if val is not None and 0 < abs(val) < 1:
                return "rate/percentage (decimal form)"
            return "rate/percentage"

        if _CURRENCY_NAME_RE.search(name_lower):
            return "currency/price"

        if _COUNT_NAME_RE.search(name_lower):
            return "count/quantity"

        if step_text:
            step_lower = step_text.lower()
            if any(kw in step_lower for kw in ("%", "percent", "rate", "margin", "yield")):
                if val is not None and 0 < abs(val) < 1:
                    return "rate/percentage (decimal form)"

        return "general"

    def _build_variable_semantics(
        self, new_vars: Dict[str, str], step_text: str = ""
    ) -> Dict[str, str]:
        """Build a dict mapping variable name -> semantic type for handoff metadata.

        Prioritizes Skill-registered semantics (from extraction_state_bridge) over
        name-based inference, since Skill has access to raw cell text and headers.
        """
        if not new_vars:
            return {}
        result: Dict[str, str] = {}
        for name, val in new_vars.items():
            registered = get_variable_semantic_type(name)
            if registered:
                result[name] = registered
            else:
                result[name] = self._infer_variable_semantic_type(name, val, step_text)
        return result

    def _extract_explicit_output_unit_hint(self, text: str) -> Optional[str]:
        scope = (text or "").strip()
        if not scope:
            return None
        match = re.search(
            r"(?:final\s+unit|output\s+unit|answer\s+unit|unit)\s*[:=]\s*([a-zA-Z_%\s]+)",
            scope,
            re.IGNORECASE,
        )
        if not match:
            return None
        return re.sub(r"\s+", " ", match.group(1)).strip(" .,)(")

    def _normalize_output_unit_hint(self, unit_hint: str) -> str:
        hint = (unit_hint or "").strip().lower()
        if not hint:
            return "base unit"
        if any(token in hint for token in ("percent", "percentage", "%", "百分点")):
            return "percent"
        if any(token in hint for token in ("raw ratio", "decimal ratio", "ratio", "raw decimal", "decimal form")):
            return "raw ratio (decimal)"
        if any(token in hint for token in ("billion", "billions")):
            return "billions"
        if any(token in hint for token in ("million", "millions")):
            return "millions"
        if any(token in hint for token in ("thousand", "thousands")):
            return "thousands"
        if any(token in hint for token in ("base", "full", "absolute")):
            return "base unit"
        return re.sub(r"\s+", " ", unit_hint).strip()

    def _infer_output_unit_hint(self, step_text: str, user_request: str = "") -> str:
        explicit = self._extract_explicit_output_unit_hint(step_text)
        if explicit:
            return self._normalize_output_unit_hint(explicit)
        return "base unit"

    def _build_output_unit_contract(self, step_text: str, user_request: str = "") -> str:
        final_unit = self._infer_output_unit_hint(step_text, user_request)
        return (
            "OUTPUT UNIT CONTRACT:\n"
            f"- Final answer unit for this step: {final_unit}.\n"
            "- Keep ratio-like intermediates in raw decimal form during computation unless the step explicitly asks for percent output.\n"
            "- Apply scale conversion only at final formatting when the requested answer unit is percent/millions/billions/thousands; otherwise keep base-unit values."
        )

    def _inject_shared_python_execute(self, agent: BaseAgent) -> None:
        """将共享的 PythonExecute 实例注入到 Agent 的 ToolCollection 中。"""
        if not hasattr(agent, "available_tools"):
            return
        
        shared_py = self._ensure_shared_python_execute()
        
        # 获取当前 Agent 的 available_tools
        tools = agent.available_tools
        if tools is None:
            return
        
        # 检查是否已有 python_execute
        existing = tools.get_tool("python_execute")
        if existing is not None and existing is shared_py:
            # 已经是共享实例，无需操作
            return
        
        # 替换为共享实例
        if existing is not None:
            # 移除旧的实例
            new_tools = list(tools.tools)
            new_tools = [t for t in new_tools if t.name != "python_execute"]
            new_tools.append(shared_py)
            agent.available_tools = ToolCollection(*new_tools)
        else:
            # 添加共享实例
            tools.add_tool(shared_py)

    def _reset_python_execute_env(self) -> None:
        """Flow 开始时重置所有 executor 的 python_execute 环境，确保变量不跨请求泄露。"""
        # 首先确保所有 Agent 使用共享的 PythonExecute 实例
        seen = set()
        for key in self.executor_keys:
            if key in self.agents:
                agent = self.agents[key]
                if id(agent) not in seen:
                    seen.add(id(agent))
                    self._inject_shared_python_execute(agent)
        
        if self.primary_agent and id(self.primary_agent) not in seen:
            self._inject_shared_python_execute(self.primary_agent)
        
        # 然后重置共享实例的环境
        shared_py = self._ensure_shared_python_execute()
        if hasattr(shared_py, "reset_env"):
            shared_py.reset_env()
        reset_variable_semantic_registry()

    async def execute(
        self,
        input_text: str,
        base64_images: Optional[List[str]] = None,
    ) -> str:
        """Execute the planning flow with agents.

        Args:
            input_text: User request text.
            base64_images: Optional list of base64-encoded images for multimodal tasks.
                          When provided, Planning may assign [multimodal] to steps.
        """
        try:
            if not self.primary_agent:
                raise ValueError("No primary agent available")

            self.base64_images = base64_images
            self.step_handoff_records = []
            self._reset_python_execute_env()

            # 防死循环拦截器：每个样本开始时创建并注入到所有 agent
            anti_loop = AntiLoopInterceptor()
            for agent in self.agents.values():
                if hasattr(agent, "anti_loop_interceptor"):
                    agent.anti_loop_interceptor = anti_loop

            if input_text:
                await self._create_initial_plan(input_text)
                if self.active_plan_id not in self.planning_tool.plans:
                    logger.error(
                        f"Plan creation failed. Plan ID {self.active_plan_id} not found in planning tool."
                    )
                    return f"Failed to create plan for: {input_text}"

            execution_result = ""
            while True:
                self.current_step_index, step_info = await self._get_current_step_info()

                if self.current_step_index is None:
                    execution_result += await self._finalize_plan(
                        execution_result=execution_result,
                        user_request=input_text or "",
                    )
                    break

                step_type = step_info.get("type") if step_info else None
                executor = self.get_executor(step_type)
                step_result, step_success = await self._execute_step(
                    executor,
                    step_info,
                    previous_output=execution_result,
                    user_request=input_text or "",
                )
                execution_result += step_result + "\n"

                if not step_success:
                    execution_result += (
                        "\n[FLOW STOPPED] Current step failed or produced no valid handoff. "
                        "Subsequent steps were not executed to avoid propagating invalid results.\n"
                    )
                    execution_result += await self._finalize_plan(
                        execution_result=execution_result,
                        user_request=input_text or "",
                    )
                    break

                if hasattr(executor, "state") and executor.state == AgentState.FINISHED:
                    break

            return execution_result
        except Exception as e:
            logger.error(f"Error in PlanningFlow: {str(e)}")
            return f"Execution failed: {str(e)}"

    def _get_planning_agent(self):
        """获取可用于创建计划的 PlanningAgent（持有 workflow_state_tool 且工具范围限定为规划相关）。"""
        for key, agent in self.agents.items():
            if (
                hasattr(agent, "workflow_state_tool")
                and agent.workflow_state_tool is not None
                and agent.workflow_state_tool is self.planning_tool
            ):
                return agent
        return None

    async def _create_initial_plan(self, request: str) -> None:
        """创建初始计划。若存在 PlanningAgent 则委托其执行，否则使用 flow 的 LLM 直接调用 planning_tool。"""
        logger.info(f"Creating initial plan with ID: {self.active_plan_id}")

        planning_agent = self._get_planning_agent()
        if planning_agent is not None:
            await self._create_plan_via_planning_agent(planning_agent, request)
            return

        await self._create_plan_via_flow_llm(request)

    def _extract_time_range_hint(self, request: str) -> str:
        """
        从问题中提取时间范围提示，帮助 Planning Agent 正确理解时间范围。
        
        例如：
        - "October 2018" -> "Focus ONLY on October 2018 data. Do NOT include data from other months like November or September unless explicitly requested."
        - "Q3 2024" -> "Focus ONLY on Q3 2024 data (July-September)."
        
        返回空字符串表示没有检测到特定时间范围提示。
        """
        import re
        
        request_lower = request.lower()
        hints = []
        
        # 检测月份 + 年份模式 (e.g., "October 2018", "oct 2018")
        month_year_patterns = [
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\s+(\d{4})\b',
            r'\b(\d{4})\s+(january|february|march|april|may|june|july|august|september|october|november|december)\b',
        ]
        
        for pattern in month_year_patterns:
            match = re.search(pattern, request_lower)
            if match:
                month = match.group(1) if match.group(1).isalpha() else match.group(2)
                year = match.group(2) if match.group(1).isalpha() else match.group(1)
                month_name = month.capitalize() if len(month) > 3 else month.capitalize()
                hints.append(
                    f"Focus ONLY on {month_name} {year}. If the table has multiple period rows, use ONLY the row(s) "
                    f"that fall within {month_name} {year} (e.g. 09/30-10/27 for October). Do NOT sum rows from other months (e.g. 10/28-11/24 is November)."
                )
                break
        
        # 检测季度模式 (e.g., "Q3 2024", "third quarter 2024")
        quarter_map = {
            'first': 'Q1', 'second': 'Q2', 'third': 'Q3', 'fourth': 'Q4',
            'q1': 'Q1', 'q2': 'Q2', 'q3': 'Q3', 'q4': 'Q4'
        }
        quarter_patterns = [
            r'\b(q[1-4])\s+(\d{4})\b',
            r'\b(first|second|third|fourth)\s+quarter\s+(\d{4}|of\s+\d{4})\b',
        ]
        
        for pattern in quarter_patterns:
            match = re.search(pattern, request_lower)
            if match:
                quarter_raw = match.group(1).lower()
                quarter = quarter_map.get(quarter_raw, quarter_raw.upper())
                year = match.group(2)
                # 清理 year 中的 "of " 前缀
                if year.startswith('of '):
                    year = year[3:]
                hints.append(f"Focus ONLY on {quarter} {year} data. Do NOT include data from other quarters unless explicitly requested.")
                break
        
        return " ".join(hints) if hints else ""

    def _build_plan_prompt(self, request: str) -> str:
        """构建规划任务 prompt（含 plan_id、executor 描述、规划规则）。"""
        lines = [
            f"Create a plan for the following task. On your FIRST response, you MUST call the planning tool (do not output analysis without calling it). Use plan_id='{self.active_plan_id}'.",
            "",
            "**Task:**",
            request,
            "",
            "**Planning rules:**",
            "- For formula-based computation: use 2 steps. Extraction step MUST list ALL variable names required by the formula—never omit any. Specify output order (e.g. 'output as var_a, var_b') so downstream maps by position. Computation step: apply formula and output result.",
            "- For critical calculations (ratios, differences, etc.): ensure extraction step extracts exactly the distinct inputs required—each variable from a different source. Do not use the same value as both numerator and denominator, or omit a required input.",
            "- When the question constrains the final answer format, include an explicit unit hint in the compute step (for example: 'final unit: percent', 'final unit: raw ratio', 'final unit: millions', or 'final unit: base unit').",
            "- For simple tasks: use 1 step.",
            "- Do not over-split (avoid 7+ steps for a single formula).",
            "- Do NOT hardcode numerical values in step descriptions. Extraction steps must specify WHAT to extract (e.g. 'extract X from table'), not assume or invent values.",
            "- If text context already maps multiple values with wording like 'A and B, respectively' for the requested metric/periods, prefer text-context extraction instead of visual extraction.",
            "- If narrative text gives approximate values but exact computable inputs exist in the image/table, plan exact extraction + computation. Do NOT substitute approximate narrative values for exact computation.",
            "- If the question asks for sum/total of multiple values (e.g. 'A 与 B 之和', 'total of X and Y', 'A and B combined'), include a step to sum/combine the extracted values after extraction.",
        ]
        
        # 添加时间范围验证提示（针对 easy-test-194 类型的问题）
        time_range_hint = self._extract_time_range_hint(request)
        if time_range_hint:
            lines.extend([
                "",
                f"**Time Range Guidance:** {time_range_hint}",
            ])
        if self.base64_images and "multimodal" in self.agents:
            lines.extend(
                [
                    "",
                    "**Multimodal (MANDATORY for image tasks):** User has provided images. Any step that EXTRACTS data from charts/tables/images (提取、extract) MUST use [multimodal]. Only [multimodal] can see the image; [finance] cannot access images. Use [finance] only for computation steps that operate on already-extracted values (e.g. sum, formula, compare). Do NOT assign extraction from charts/tables to [finance].",
                ]
            )
            # 多图：变量–图片映射 + 每图独立 [multimodal] 步骤（针对 easy-test-22）
            if len(self.base64_images) > 1:
                n = len(self.base64_images)
                img_refs = ", ".join(f"image {i+1}" for i in range(n))
                lines.extend(
                    [
                        "",
                        f"**Multi-image ({n} images: {img_refs}):**",
                        "- Map each variable required by the formula to its source image. Create exactly one [multimodal] step per image.",
                        "- Example: if formula needs A, B from image 1 and C from image 2 → Step 0: [multimodal] from image 1 extract A, B; Step 1: [multimodal] from image 2 extract C; Step 2: [finance] compute.",
                        "- NEVER skip a [multimodal] step for an image that holds required data. Each [multimodal] step MUST explicitly say 'from image 1' or 'from image 2'.",
                        "- Never combine 'from image 1' and 'from image 2' in one step—each step receives only one image.",
                    ]
                )
        elif not self.base64_images and "multimodal" in self.agents:
            lines.extend(
                [
                    "",
                    "**No images provided:** If the task requires extracting data from charts/tables/images (提取、图中、表格), but NO images are provided, the task CANNOT be completed. Create a minimal plan with 1 step: [finance] state 'Data missing - no image provided for extraction' and call terminate(status='failure'). Do NOT create [multimodal] extraction steps when no images exist.",
                ]
            )
        agents_description = []
        for key in self.executor_keys:
            if key in self.agents:
                agents_description.append(
                    {"name": key, "description": self.agents[key].description}
                )
        if agents_description:
            executor_names = [a["name"] for a in agents_description]
            lines.extend(
                [
                    "",
                    f"**Executors (use these exact keys in step labels):** {executor_names}. EVERY step MUST start with [executor_key], e.g. [multimodal] or [finance]. Do NOT create steps without an executor tag. Do NOT use non-existent keys like [text_agent] or [compute_agent].",
                ]
            )
        lines.extend(
            [
                "",
                f"Call the planning tool with command='create', plan_id='{self.active_plan_id}', title, and steps. Then terminate.",
            ]
        )
        return "\n".join(lines)

    def _extract_step_executor_tag(self, step_text: str) -> Optional[str]:
        text = (step_text or "").strip()
        if not text:
            return None
        match = re.match(r"^\[([a-zA-Z_]+)\]\s*", text)
        if not match:
            return None
        return match.group(1).lower()

    def _strip_step_executor_tag(self, step_text: str) -> str:
        text = (step_text or "").strip()
        return re.sub(r"^\[[a-zA-Z_]+\]\s*", "", text).strip()

    def _is_output_only_plan_step(self, step_text: str) -> bool:
        text = (step_text or "").strip()
        if not text or self._extract_step_executor_tag(text):
            return False
        lower = text.lower()
        return lower.startswith("output as ") or lower.startswith("输出为")

    def _normalize_candidate_plan_steps(self, steps: List[str]) -> List[str]:
        """Lightly normalize planner output into executable step contracts.

        Current normalizations are intentionally conservative:
        1. Drop empty strings.
        2. Merge orphaned `output as ...` pseudo-steps into the previous executor step.
        """
        normalized: List[str] = []
        for raw_step in steps or []:
            step = (raw_step or "").strip()
            if not step:
                continue
            if self._is_output_only_plan_step(step):
                if not normalized:
                    continue
                prev = normalized[-1].rstrip(" .")
                if "output as" not in prev.lower():
                    normalized[-1] = f"{prev}, {step.rstrip('.')}"
                continue
            normalized.append(step)
        return normalized

    def _step_has_output_contract(self, step_text: str) -> bool:
        text = (step_text or "").lower()
        return "save_as" in text or "output as" in text or "输出为" in text

    def _step_requires_output_contract(self, step_text: str) -> bool:
        text = (step_text or "").lower()
        if not text:
            return False
        return any(
            key in text
            for key in (
                "extract",
                "提取",
                "compute",
                "calculate",
                "ratio",
                "sum",
                "total",
                "compare",
                "应用公式",
                "计算",
                "输出",
            )
        )

    def _validate_normalized_plan_steps(self, steps: List[str]) -> tuple[bool, str]:
        if not steps:
            return False, "candidate plan has no executable steps"

        known_executors = set(self.executor_keys) | set(self.agents.keys())
        for idx, step in enumerate(steps):
            executor = self._extract_step_executor_tag(step)
            if not executor:
                return False, f"step {idx} is missing executor tag: {step}"
            if executor not in known_executors:
                return False, f"step {idx} uses unknown executor [{executor}]"
            if self._step_requires_output_contract(step) and not self._step_has_output_contract(step):
                logger.info(
                    "Plan step %d has no explicit output contract but contains action keywords — accepting anyway: %s",
                    idx,
                    step[:120],
                )
        return True, ""

    def _select_candidate_plan_id(self, existing_plan_ids: set[str]) -> Optional[str]:
        if self.active_plan_id in self.planning_tool.plans:
            return self.active_plan_id

        current_plan_id = getattr(self.planning_tool, "_current_plan_id", None)
        new_plan_ids = [
            plan_id
            for plan_id in self.planning_tool.plans.keys()
            if plan_id not in existing_plan_ids
        ]
        if len(new_plan_ids) == 1:
            return new_plan_ids[0]
        if current_plan_id and current_plan_id in self.planning_tool.plans:
            return current_plan_id
        if new_plan_ids:
            return new_plan_ids[-1]
        return None

    def _adopt_candidate_plan(self, candidate_plan_id: str, normalized_steps: List[str]) -> None:
        candidate = self.planning_tool.plans[candidate_plan_id]
        adopted_plan = {
            "plan_id": self.active_plan_id,
            "title": (candidate.get("title") or "").strip() or f"Plan for {self.active_plan_id}",
            "steps": normalized_steps,
            "step_statuses": ["not_started"] * len(normalized_steps),
            "step_notes": [""] * len(normalized_steps),
        }
        self.planning_tool.plans[self.active_plan_id] = adopted_plan
        self.planning_tool._current_plan_id = self.active_plan_id
        if candidate_plan_id != self.active_plan_id:
            self.planning_tool.plans.pop(candidate_plan_id, None)

    def _cleanup_invalid_candidate_plans(
        self, existing_plan_ids: set[str], candidate_plan_id: Optional[str]
    ) -> None:
        plan_ids_to_delete = {
            plan_id
            for plan_id in self.planning_tool.plans.keys()
            if plan_id not in existing_plan_ids
        }
        if self.active_plan_id in self.planning_tool.plans and self.active_plan_id not in existing_plan_ids:
            plan_ids_to_delete.add(self.active_plan_id)
        if candidate_plan_id and candidate_plan_id in self.planning_tool.plans:
            plan_ids_to_delete.add(candidate_plan_id)
        for plan_id in plan_ids_to_delete:
            self.planning_tool.plans.pop(plan_id, None)
        if getattr(self.planning_tool, "_current_plan_id", None) in plan_ids_to_delete:
            self.planning_tool._current_plan_id = None

    def _validate_and_adopt_candidate_plan(
        self, existing_plan_ids: set[str]
    ) -> tuple[bool, str]:
        candidate_plan_id = self._select_candidate_plan_id(existing_plan_ids)
        if not candidate_plan_id:
            return False, "planning agent did not create a candidate plan"

        candidate = self.planning_tool.plans.get(candidate_plan_id) or {}
        steps = candidate.get("steps") or []
        if not isinstance(steps, list) or not all(isinstance(step, str) for step in steps):
            return False, f"candidate plan '{candidate_plan_id}' has invalid steps payload"

        normalized_steps = self._normalize_candidate_plan_steps(steps)
        ok, reason = self._validate_normalized_plan_steps(normalized_steps)
        if not ok:
            return False, reason

        self._adopt_candidate_plan(candidate_plan_id, normalized_steps)
        return True, ""

    async def _create_plan_via_planning_agent(
        self, planning_agent: BaseAgent, request: str
    ) -> None:
        """由 PlanningAgent 创建计划。多图时传入全部图以便 Planning 分配任务；单图传单张。"""
        existing_plan_ids = set(self.planning_tool.plans.keys())
        plan_prompt = self._build_plan_prompt(request)
        base64_image = self.base64_images[0] if self.base64_images else None
        base64_images = self.base64_images if len(self.base64_images or []) > 1 else None
        await planning_agent.run(
            plan_prompt,
            base64_image=base64_image,
            base64_images=base64_images,
        )

        is_valid, reason = self._validate_and_adopt_candidate_plan(existing_plan_ids)
        if is_valid:
            return

        logger.warning(
            "PlanningAgent did not create a valid executable plan (%s); falling back to flow LLM.",
            reason,
        )
        self._cleanup_invalid_candidate_plans(
            existing_plan_ids, self._select_candidate_plan_id(existing_plan_ids)
        )
        await self._create_plan_via_flow_llm(request)

    async def _create_plan_via_flow_llm(self, request: str) -> None:
        """由 flow 的 LLM 直接调用 planning_tool 创建计划（无 PlanningAgent 时）。"""
        system_message_content = (
            "You are a planning assistant. Create a concise, actionable plan with clear steps. "
            "Focus on key milestones rather than detailed sub-steps. Optimize for clarity and efficiency. "
            "For critical calculations (ratios, differences): ensure extraction step extracts exactly the distinct inputs required. "
            "For formula-based computation (e.g. EBITDA=X+Y+Z) where the user provides the formula and data: "
            "1) Use 2 steps: extraction + computation. "
            "2) The extraction step MUST explicitly list the variable names from the formula, e.g. '提取合并净利润、所得税费用、利息支出、固定资产折旧、无形资产摊销' (not generic '提取所需数据'). "
            "3) The computation step: '应用公式计算并输出结果'. "
            "4) If the final answer has a required presentation unit, include it explicitly in the compute step such as 'final unit: percent' or 'final unit: raw ratio'. "
            "For simple tasks without a formula, use 1 step. Do not over-split (e.g. avoid 7 steps for a single formula). python_execute variables persist across calls."
        )
        agents_description = []
        for key in self.executor_keys:
            if key in self.agents:
                agents_description.append(
                    {
                        "name": key,
                        "description": self.agents[key].description,
                    }
                )
        if agents_description:
            executor_names = [a["name"] for a in agents_description]
            system_message_content += (
                f"\n**Executors (use these exact keys in step labels):** {executor_names}. "
                f"Use format [executor_key] in step text, e.g. [finance]. "
                "Do NOT use non-existent keys like [text_agent] or [compute_agent]."
            )
        if self.base64_images and "multimodal" in self.agents:
            system_message_content += (
                "\n**Multimodal (MANDATORY for image tasks):** User has provided images. "
                "Any step that EXTRACTS data from charts/tables/images MUST use [multimodal]. "
                "Only [multimodal] can see the image; [finance] cannot access images. "
                "Use [finance] only for computation steps that operate on already-extracted values."
            )

        system_message = Message.system_message(system_message_content)
        user_message = Message.user_message(
            f"Create a reasonable plan with clear steps to accomplish the task: {request}"
        )

        tool_param = (
            self.planning_tool.to_planning_param()
            if hasattr(self.planning_tool, "to_planning_param")
            else self.planning_tool.to_param()
        )

        response = await self.llm.ask_tool(
            messages=[user_message],
            system_msgs=[system_message],
            tools=[tool_param],
            tool_choice=ToolChoice.AUTO,
        )

        if response and response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call.function.name in ("planning", "workflow_state"):
                    args = tool_call.function.arguments
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse tool arguments: {args}")
                            continue

                    args["plan_id"] = self.active_plan_id
                    result = await self.planning_tool.execute(**args)
                    logger.info(f"Plan creation result: {str(result)}")
                    return

        logger.warning("Creating default plan")
        await self.planning_tool.execute(
            **{
                "command": "create",
                "plan_id": self.active_plan_id,
                "title": f"Plan for: {request[:50]}{'...' if len(request) > 50 else ''}",
                "steps": ["Analyze request", "Execute task", "Verify results"],
            }
        )

    def _infer_step_type(self, step_text: str) -> str:
        """
        当步骤无 [executor] 标签时，根据文本推断 executor。
        """
        text = (step_text or "").strip().lower()
        # 提取类：图中、表格、extract、from image
        extract_patterns = [
            r"从图|图中|表格|extract|提取",
            r"from\s+image\s*\d*",
            r"image\s*\d+\s*(中|里|的)",
        ]
        for p in extract_patterns:
            if re.search(p, text, re.IGNORECASE):
                if self.base64_images and "multimodal" in self.agents:
                    return "multimodal"
                break
        # 计算类：公式、计算、比例、ratio、formula
        compute_patterns = [
            r"计算|应用公式|formula|calculate|ratio|比例",
            r"sum|total|差值|difference|compare",
        ]
        for p in compute_patterns:
            if re.search(p, text, re.IGNORECASE):
                return "finance"
        # 默认：计算步骤更常见
        return "finance"

    async def _get_current_step_info(self) -> tuple[Optional[int], Optional[dict]]:
        """
        Parse the current plan to identify the first non-completed step's index and info.
        Returns (None, None) if no active step is found.
        """
        if (
            not self.active_plan_id
            or self.active_plan_id not in self.planning_tool.plans
        ):
            logger.error(f"Plan with ID {self.active_plan_id} not found")
            return None, None

        try:
            plan_data = self.planning_tool.plans[self.active_plan_id]
            steps = plan_data.get("steps", [])
            step_statuses = plan_data.get("step_statuses", [])

            for i, step in enumerate(steps):
                if i >= len(step_statuses):
                    status = PlanStepStatus.NOT_STARTED.value
                else:
                    status = step_statuses[i]

                if status in PlanStepStatus.get_active_statuses():
                    step_info = {"text": step}
                    type_match = re.search(r"\[([a-zA-Z_]+)\]", step)
                    if type_match:
                        step_info["type"] = type_match.group(1).lower()
                    else:
                        # 无 [executor] 标签时根据步骤文本推断，避免 step_type=None 导致错误分配
                        step_info["type"] = self._infer_step_type(step)

                    try:
                        await self.planning_tool.execute(
                            command="mark_step",
                            plan_id=self.active_plan_id,
                            step_index=i,
                            step_status=PlanStepStatus.IN_PROGRESS.value,
                        )
                    except Exception as e:
                        logger.warning(f"Error marking step as in_progress: {e}")
                        if i < len(step_statuses):
                            step_statuses[i] = PlanStepStatus.IN_PROGRESS.value
                        else:
                            while len(step_statuses) < i:
                                step_statuses.append(PlanStepStatus.NOT_STARTED.value)
                            step_statuses.append(PlanStepStatus.IN_PROGRESS.value)
                        plan_data["step_statuses"] = step_statuses

                    return i, step_info

            return None, None

        except Exception as e:
            logger.warning(f"Error finding current step index: {e}")
            return None, None

    def _extract_primary_tool_evidence(self, step_result: str) -> str:
        """Extract the last successful non-terminate tool's observation from a step result."""
        if not step_result or not step_result.strip():
            return ""
        tool_pattern = re.compile(
            r"Observed output of cmd `([^`]+)` executed:\n(.*?)(?=(?:\n\nObserved output of cmd `)|\Z)",
            re.IGNORECASE | re.DOTALL,
        )
        obs_pattern = re.compile(
            r"['\"]?observation['\"]?\s*:\s*['\"]([^'\"]*)['\"]",
            re.IGNORECASE,
        )
        tool_blocks = tool_pattern.findall(step_result)
        if not tool_blocks:
            return step_result.strip()

        chosen = None
        for name, body in reversed(tool_blocks):
            body_clean = body.strip()
            if name.lower() == "terminate":
                continue
            if "'success': True" in body_clean or '"success": true' in body_clean.lower():
                chosen = (name, body_clean)
                break
        if chosen is None:
            for name, body in reversed(tool_blocks):
                if name.lower() != "terminate":
                    chosen = (name, body.strip())
                    break
        if chosen is None:
            return ""

        tool_name, tool_body = chosen
        observation_matches = obs_pattern.findall(tool_body)
        if observation_matches:
            content = observation_matches[-1].strip().replace("\\n", "\n").strip()
        else:
            content = tool_body.strip()
        return f"[{tool_name}] {content}" if content else ""

    def _is_successful_tool_message(self, tool_content: str) -> bool:
        body = (tool_content or "").strip()
        lower_body = body.lower()
        if not body:
            return False
        if "'success': true" in lower_body or '"success": true' in lower_body:
            return True
        if (
            "tool execution failed:" in lower_body
            or "system intercept:" in lower_body
            or "'success': false" in lower_body
            or '"success": false' in lower_body
            or "[ocr_error]" in lower_body
            or body.startswith("Error:")
        ):
            return False
        return True

    def _sanitize_observation_text(self, content: str) -> str:
        text = re.sub(r"\s+", " ", (content or "")).strip()
        if not text:
            return ""
        sentence_like_parts = [
            part.strip(" -")
            for part in re.split(r"(?<=[.!?])\s+|\n+", text)
            if part and part.strip(" -")
        ]
        generic_patterns = [
            r"\bi will now terminate\b",
            r"\bi will terminate\b",
            r"\bterminate the process\b",
            r"\bterminate the interaction\b",
            r"\bthe task is complete\b",
            r"^the values have been successfully extracted(?: and (?:saved|assigned|confirmed))?\.?$",
            r"^the required value has been extracted(?: and saved)?\.?$",
            r"^the extraction and verification are complete\.?$",
            r"^the .* has been successfully extracted and confirmed\.?$",
        ]
        kept: List[str] = []
        for part in sentence_like_parts:
            lower = part.lower()
            if any(re.search(pattern, lower) for pattern in generic_patterns):
                continue
            kept.append(part)
        return " ".join(kept).strip()

    def _is_low_value_observation(self, content: str) -> bool:
        return not bool(self._sanitize_observation_text(content))

    def _extract_executor_observation(
        self, executor: BaseAgent, start_message_idx: int
    ) -> str:
        """
        Extract the executor's own natural-language observation from messages added in this run.
        Prefer the assistant text immediately preceding the latest successful non-terminate tool.
        Fall back to the latest informative assistant text only when the step has no such tool.
        """
        if not hasattr(executor, "memory") or executor.memory is None:
            return ""
        new_messages = executor.memory.messages[start_message_idx:]
        latest_informative_assistant = ""
        latest_pre_tool_informative_assistant = ""
        current_assistant = ""
        for msg in new_messages:
            role = getattr(msg, "role", None)
            if role == "assistant":
                content = (getattr(msg, "content", None) or "").strip()
                if not content:
                    continue
                compact = self._sanitize_observation_text(content)
                if not compact:
                    continue
                current_assistant = compact
                latest_informative_assistant = compact
                continue

            if role != "tool":
                continue
            tool_name = (getattr(msg, "name", None) or "").strip().lower()
            tool_content = (getattr(msg, "content", None) or "").strip()
            if tool_name == "terminate":
                continue
            if not self._is_successful_tool_message(tool_content):
                continue
            if current_assistant:
                latest_pre_tool_informative_assistant = current_assistant

        return latest_pre_tool_informative_assistant or latest_informative_assistant

    def _record_step_handoff(
        self,
        executor: BaseAgent,
        step_info: dict,
        step_result: str,
        start_message_idx: int,
        step_new_variables: Optional[Dict[str, str]] = None,
        step_variable_semantics: Optional[Dict[str, str]] = None,
    ) -> None:
        tool_evidence = self._extract_primary_tool_evidence(step_result)
        executor_observation = self._extract_executor_observation(
            executor, start_message_idx
        )
        self.step_handoff_records.append(
            {
                "step_index": str(self.current_step_index if self.current_step_index is not None else ""),
                "step_type": (step_info.get("type") or "").strip(),
                "executor_name": getattr(executor, "name", type(executor).__name__),
                "step_text": (step_info.get("text") or "").strip(),
                "tool_evidence": tool_evidence,
                "executor_observation": executor_observation,
                "step_variables": step_new_variables or {},
                "step_variable_semantics": step_variable_semantics or {},
            }
        )

    def _format_structured_previous_output(self, previous_output: str) -> str:
        """
        将原始 execution_result 格式化为按步骤分组的证据块。
        优先保留每步最后一个成功的非 terminate 工具输出，避免把上一步结果压扁成裸数字。
        """
        if not previous_output or not previous_output.strip():
            return ""
        if self.step_handoff_records:
            step_blocks = []
            for idx, record in enumerate(self.step_handoff_records):
                parts = []
                executor_observation = (record.get("executor_observation") or "").strip()
                tool_evidence = (record.get("tool_evidence") or "").strip()
                step_vars = record.get("step_variables") or {}
                var_semantics = record.get("step_variable_semantics") or {}
                step_type = (record.get("step_type") or "").strip()
                executor_name = (record.get("executor_name") or "").strip()
                if step_vars:
                    var_parts = []
                    for k, v in step_vars.items():
                        sem = var_semantics.get(k, "")
                        if sem and sem != "general":
                            var_parts.append(f"{k} = {v} [semantic: {sem}]")
                        else:
                            var_parts.append(f"{k} = {v}")
                    var_lines = ", ".join(var_parts)
                    parts.append("Saved variables: " + var_lines)
                if executor_observation:
                    parts.append("Executor observation:\n" + executor_observation)
                if tool_evidence:
                    parts.append("Tool evidence:\n" + tool_evidence)
                if parts:
                    label = f"{executor_name}/{step_type}" if step_type else executor_name
                    step_blocks.append(f"Step {idx} [{label}]:\n" + "\n".join(parts))
            if step_blocks:
                return "\n\n".join(step_blocks)
        tool_pattern = re.compile(
            r"Observed output of cmd `([^`]+)` executed:\n(.*?)(?=(?:\n\nObserved output of cmd `)|\Z)",
            re.IGNORECASE | re.DOTALL,
        )
        obs_pattern = re.compile(
            r"['\"]?observation['\"]?\s*:\s*['\"]([^'\"]*)['\"]",
            re.IGNORECASE,
        )
        segments = re.split(
            r"Observed output of cmd `terminate` executed:\nThe interaction has been completed with status: \w+",
            previous_output,
            flags=re.IGNORECASE,
        )
        step_blocks = []

        for idx, seg in enumerate(segments):
            if not seg.strip():
                continue
            tool_blocks = tool_pattern.findall(seg)
            if not tool_blocks:
                continue

            chosen = None
            for name, body in reversed(tool_blocks):
                body_clean = body.strip()
                if name.lower() == "terminate":
                    continue
                if "'success': True" in body_clean or '"success": true' in body_clean.lower():
                    chosen = (name, body_clean)
                    break
            if chosen is None:
                name, body = tool_blocks[-1]
                chosen = (name, body.strip())

            tool_name, tool_body = chosen
            observation_matches = obs_pattern.findall(tool_body)
            if observation_matches:
                content = observation_matches[-1].strip().replace("\\n", "\n").strip()
            else:
                content = tool_body.strip()

            step_blocks.append(f"Step {idx} [{tool_name}]:\n{content}")

        return "\n\n".join(step_blocks) if step_blocks else previous_output.strip()

    def _is_extraction_like_step(self, step_text: str) -> bool:
        text = (step_text or "").strip().lower()
        if not text:
            return False
        patterns = [
            r"\bextract\b",
            r"\bsave_as\b",
            r"\boutput\s+as\b",
            r"提取",
            r"输出为",
            r"保存为",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    def _is_step_result_successful(self, step_result: str, step_text: str = "") -> bool:
        """
        判断 step 是否得到有效结果。
        - terminate(status='failure') -> 失败
        - 存在失败/拦截且没有后续成功的非 terminate 工具 -> 失败
        - 仅 terminate(status='success') -> 仅在分析型步骤视为成功
        """
        if not step_result or not step_result.strip():
            return False

        tool_pattern = re.compile(
            r"Observed output of cmd `([^`]+)` executed:\n(.*?)(?=(?:\n\nObserved output of cmd `)|\Z)",
            re.IGNORECASE | re.DOTALL,
        )
        blocks = tool_pattern.findall(step_result)
        if not blocks:
            return "Error:" not in step_result and "Tool Execution Failed:" not in step_result

        has_successful_non_terminate = False
        has_successful_extraction_evidence = False
        has_failure = False

        for name, body in blocks:
            body_clean = body.strip()
            lower_body = body_clean.lower()

            if name.lower() == "terminate":
                if "status: failure" in lower_body:
                    return False
                continue

            if "'success': true" in lower_body or '"success": true' in lower_body:
                has_successful_non_terminate = True
                if name.lower() in {"python_execute", "finance_extraction_skill"}:
                    has_successful_extraction_evidence = True
                continue

            if (
                "tool execution failed:" in lower_body
                or "system intercept:" in lower_body
                or "'success': false" in lower_body
                or '"success": false' in lower_body
                or "[ocr_error]" in lower_body
                or body_clean.startswith("Error:")
            ):
                has_failure = True

        if has_failure and not has_successful_non_terminate:
            return False
        if self._is_extraction_like_step(step_text) and not has_successful_extraction_evidence:
            return False
        return True

    def _get_image_index_for_step(self, step_info: dict) -> int:
        """
        P1: 从步骤文本解析目标图片索引，支持多图按需提取。
        步骤中写 "from image 1" / "image 2" 时使用对应图片；未指定时用 0。
        """
        text = (step_info.get("text") or "").strip()
        m = re.search(r"(?:from\s+)?image\s*(\d+)", text, re.IGNORECASE)
        if m:
            return max(0, int(m.group(1)) - 1)  # 1-based in plan → 0-based
        return 0

    async def _execute_step(
        self,
        executor: BaseAgent,
        step_info: dict,
        previous_output: str = "",
        user_request: str = "",
    ) -> tuple[str, bool]:
        """Execute the current step with the specified agent using agent.run()."""
        # Reset anti-loop per step: different steps legitimately produce similar
        # python_execute code (e.g., multimodal saves x=123, finance reuses x).
        # Without reset, the shared interceptor blocks the second call.
        for ag in self.agents.values():
            if hasattr(ag, "anti_loop_interceptor") and ag.anti_loop_interceptor:
                ag.anti_loop_interceptor.reset_memory()
                break  # all agents share the same instance

        step_type = step_info.get("type")
        step_text = step_info.get("text", f"Step {self.current_step_index}")
        route_text_context_direct = self._should_route_text_context_direct(
            step_type=step_type,
            step_text=step_text,
            user_request=user_request,
        )
        if route_text_context_direct:
            logger.info(
                f"Step {self.current_step_index}: step explicitly requests text context; "
                "routing directly to finance executor"
            )
            result, success = await self._run_text_context_executor(
                step_info=step_info,
                previous_output=previous_output,
                user_request=user_request,
                step_text=step_text,
                routing_note=(
                    "The plan explicitly marked this extraction as coming from text context. "
                    "Do not inspect or infer from images for this step."
                ),
            )
            if success:
                await self._mark_step_completed()
            else:
                await self._mark_step_blocked(result)
            return result, success
        # P0 无图快速失败：multimodal 步骤无图时直接 block，避免无效 LLM 调用
        if step_type == "multimodal" and not self.base64_images:
            logger.warning(
                f"Step {self.current_step_index} skipped: multimodal step but no images provided"
            )
            try:
                await self.planning_tool.execute(
                    command="mark_step",
                    plan_id=self.active_plan_id,
                    step_index=self.current_step_index,
                    step_status=PlanStepStatus.BLOCKED.value,
                    step_notes="No image provided; cannot extract from image/table.",
                )
            except Exception as e:
                logger.warning(f"Error marking step as blocked: {e}")
            return (
                "Step blocked: No image provided. Cannot extract data from image/table. "
                "Data missing.",
                False,
            )

        plan_status = await self._get_plan_text()

        prev_block = ""
        prev_block_extra = ""
        if previous_output.strip():
            # A: 结构化格式，避免 Finance 误将多步输出映射到同一变量
            structured = self._format_structured_previous_output(previous_output)
            prev_block = f"""
        PREVIOUS STEPS OUTPUT:
        - "Executor observation" = the previous executor's own semantic conclusion / summary.
        - "Tool evidence" = grounded outputs from tools such as python_execute or finance_extraction_skill.
        For analytical or text-based follow-up steps, preserve and build on the executor observation.
        For computational steps, prefer named variables and grounded tool evidence.
        Reuse variables already stored in shared python state whenever possible.
        Do NOT remap values by position unless the previous step explicitly labeled them.
        If the previous output is unlabeled and ambiguous, stop and state what is missing instead of guessing or swapping values.
        ---
        {structured}
        ---
        """
            # 当上一步 python_execute 输出为空时，禁止 Finance 编造数值（如 easy-test-14）
            if re.search(r"observation['\"]?\s*:\s*['\"]?['\"]", previous_output):
                prev_block_extra = """
        WARNING: Previous step returned empty observation. Do NOT invent values. Use ONLY numbers from PREVIOUS STEPS OUTPUT. If none, state extraction failed.
        """

        step_type = step_info.get("type")
        if step_type == "multimodal":
            step_prompt = f"""YOUR TASK: {step_text}

Look at the image and extract the required data or answer the visual question.
Choose the best extraction strategy yourself based on the visual structure you observe.
Use only numbers you can ground in the image.

If you extract values, save them via `python_execute` (assign + print), then call terminate(status="success").

**NO MENTAL MATH**: Do NOT do any calculation in your head. ALL computations MUST be done via python_execute."""
        else:
            user_context_block = ""
            if user_request.strip():
                user_context_block = f"""
        USER REQUEST / 原始数据（从此处提取数值，勿编造）:
        ---
        {user_request.strip()}
        ---

        """
            output_unit_contract = self._build_output_unit_contract(step_text, user_request)
            step_prompt = f"""
        CURRENT PLAN STATUS:
        {plan_status}
        {user_context_block}
        {prev_block}{prev_block_extra}
        YOUR CURRENT TASK:
        You are now working on step {self.current_step_index}: "{step_text}"

        python_execute supports multi-step: variables persist across calls. If the required variables were already stored by previous steps, use those exact variable names directly in python_execute instead of re-binding raw numbers from memory. If the plan step explicitly lists variables to extract, extract EXACTLY those items—use the exact names from the formula, not substitutes. Apply the user's formula verbatim.

        IMPORTANT: Use numbers ONLY from PREVIOUS STEPS OUTPUT (if present), existing shared python variables, or from the user context. Do NOT invent values. If the previous step already produced a text conclusion that your current step depends on, use that conclusion faithfully instead of discarding it. In python_execute, use: variable = value  # Source: 'exact snippet' (SINGLE quotes only—never triple quotes \"\"\" which break Python).

        PERCENTAGE / RATE SAFETY:
        Check PREVIOUS STEPS OUTPUT for [semantic: ...] annotations on variables:
        - [semantic: rate/percentage (decimal form)] → already normalized, do NOT divide again.
        - [semantic: rate/percentage] with raw value > 1 → convert: x = x / 100.
        - [semantic: currency/price] → NEVER divide by 100.
        Without annotation: convert only when "%" in source AND variable name/context indicates rate/margin/yield. NEVER normalize price, cost, amount, revenue, or values with currency symbols ($, €, ¥).

        {output_unit_contract}

        **NO MENTAL MATH**: Do NOT do any calculation in your head. ALL computations (arithmetic, ratios, formulas) MUST be done via python_execute. Never output a computed result without calling python_execute.

        Please only execute this current step using the appropriate tools. When you're done, provide a summary of what you accomplished.
        """

        try:
            step_type = step_info.get("type")
            base64_image_count = len(self.base64_images) if self.base64_images else 0
            start_message_idx = len(executor.memory.messages) if hasattr(executor, "memory") and executor.memory else 0
            logger.info(
                f"Step {self.current_step_index} executor={getattr(executor, 'name', type(executor).__name__)} "
                f"step_type={step_type!r} base64_images_count={base64_image_count}"
            )
            base64_image = None
            if (
                self.base64_images
                and step_type == "multimodal"
                and hasattr(executor, "run")
            ):
                # P1: 多图路由 - 从步骤文本解析 image 1/image 2，支持按需指定图片
                img_idx = self._get_image_index_for_step(step_info)
                base64_image = (
                    self.base64_images[img_idx]
                    if img_idx < len(self.base64_images)
                    else self.base64_images[0]
                )
            logger.info(
                f"Step {self.current_step_index} passing base64_image to executor: "
                f"{'yes' if base64_image else 'no'} (len={len(base64_image) if base64_image else 0})"
            )
            if step_type == "multimodal" and base64_image:
                set_step_images_for_ocr([base64_image])
                set_step_context(
                    {
                        "step_text": step_text,
                        "user_request": user_request or "",
                        "effective_question": user_request or "",
                    }
                )
            else:
                set_step_images_for_ocr(None)
                set_step_context(None)

            set_shared_python_execute(self._ensure_shared_python_execute())
            pre_vars = self._snapshot_user_variables()
            step_result = await executor.run(step_prompt, base64_image=base64_image)
            post_vars = self._snapshot_user_variables()
            new_vars = {k: v for k, v in post_vars.items() if k not in pre_vars or pre_vars[k] != v}
            var_semantics = self._build_variable_semantics(new_vars, step_text) if new_vars else None
            self._record_step_handoff(
                executor=executor,
                step_info=step_info,
                step_result=step_result,
                start_message_idx=start_message_idx,
                step_new_variables=new_vars if new_vars else None,
                step_variable_semantics=var_semantics,
            )
            step_success = self._is_step_result_successful(
                step_result, step_text=step_text
            )

            # --- Text-context fallback ---
            # When a multimodal step explicitly fails (agent says "not in image"),
            # and the original question carries meaningful text context,
            # retry the same task via the finance (text-only) executor.
            if (
                not step_success
                and step_type == "multimodal"
                and self._should_attempt_text_fallback(step_result, user_request)
            ):
                logger.info(
                    f"Step {self.current_step_index}: visual extraction failed; "
                    "attempting text-context fallback via finance executor"
                )
                fb_result, fb_success = await self._attempt_text_fallback(
                    step_info=step_info,
                    previous_output=previous_output,
                    user_request=user_request,
                    step_text=step_text,
                )
                if fb_success:
                    if self.step_handoff_records:
                        self.step_handoff_records.pop()
                    step_result = fb_result
                    step_success = True

            # --- save_as variable contract enforcement ---
            if step_success:
                expected_vars = self._extract_step_save_as_variables(step_text)
                if expected_vars:
                    current_vars = self._snapshot_user_variables()
                    missing_vars = [v for v in expected_vars if v not in current_vars]
                    if missing_vars:
                        logger.warning(
                            f"Step {self.current_step_index}: reported success but "
                            f"save_as variables {missing_vars} not found in python state"
                        )
                        step_success = False
                        step_result = (
                            f"{step_result}\n[VARIABLE_CONTRACT] Step reported success "
                            f"but missing expected variables: {', '.join(missing_vars)}"
                        )

            if step_success:
                await self._mark_step_completed()
            else:
                await self._mark_step_blocked(step_result)
            return step_result, step_success
        except Exception as e:
            logger.error(f"Error executing step {self.current_step_index}: {e}")
            return f"Error executing step {self.current_step_index}: {str(e)}", False
        finally:
            set_step_images_for_ocr(None)
            set_step_context(None)

    async def _mark_step_completed(self) -> None:
        """Mark the current step as completed."""
        if self.current_step_index is None:
            return

        try:
            await self.planning_tool.execute(
                command="mark_step",
                plan_id=self.active_plan_id,
                step_index=self.current_step_index,
                step_status=PlanStepStatus.COMPLETED.value,
            )
            logger.info(
                f"Marked step {self.current_step_index} as completed in plan {self.active_plan_id}"
            )
        except Exception as e:
            logger.warning(f"Failed to update plan status: {e}")
            if self.active_plan_id in self.planning_tool.plans:
                plan_data = self.planning_tool.plans[self.active_plan_id]
                step_statuses = plan_data.get("step_statuses", [])

                while len(step_statuses) <= self.current_step_index:
                    step_statuses.append(PlanStepStatus.NOT_STARTED.value)

                step_statuses[self.current_step_index] = PlanStepStatus.COMPLETED.value
                plan_data["step_statuses"] = step_statuses

    async def _mark_step_blocked(self, step_result: str) -> None:
        """Mark the current step as blocked when execution produced no valid result."""
        if self.current_step_index is None:
            return

        note = "Step execution failed or produced no valid handoff."
        if step_result:
            compact = " ".join(step_result.strip().split())
            if compact:
                note = compact[:240]

        try:
            await self.planning_tool.execute(
                command="mark_step",
                plan_id=self.active_plan_id,
                step_index=self.current_step_index,
                step_status=PlanStepStatus.BLOCKED.value,
                step_notes=note,
            )
            logger.info(
                f"Marked step {self.current_step_index} as blocked in plan {self.active_plan_id}"
            )
        except Exception as e:
            logger.warning(f"Failed to mark step as blocked: {e}")

    # ------------------------------------------------------------------
    # Text-context fallback: when multimodal step fails because the image
    # does not contain the needed data, re-route to a text-only executor.
    # ------------------------------------------------------------------

    def _strip_image_tags(self, text: str) -> str:
        return re.sub(r"<image\s*\d*\s*/?\s*>", "", text or "", flags=re.IGNORECASE).strip()

    def _has_meaningful_text_context(self, user_request: str) -> bool:
        return len(self._strip_image_tags(user_request)) >= 30

    def _step_explicitly_requests_text_context(self, step_text: str) -> bool:
        step_lower = (step_text or "").lower()
        markers = [
            "from the text context",
            "from text context",
            "use the text context",
            "extract from text",
            "from the user request",
            "from user request",
        ]
        return any(marker in step_lower for marker in markers)

    def _step_likely_prefers_text_context(self, step_text: str, user_request: str) -> bool:
        """
        Narrow heuristic for cases where the needed values are explicitly stated
        in narrative text (often with 'respectively') and visual read is unstable.

        Trigger only when all of these hold:
        1) request contains 'respectively' and multiple numeric values
        2) step is extraction-like and references a specific year (in 20xx)
        3) the metric phrase from step appears in text context
        """
        step_lower = (step_text or "").lower()
        text_ctx = self._strip_image_tags(user_request or "").lower()
        if not step_lower or not text_ctx:
            return False
        if "respectively" not in text_ctx:
            return False
        if len(re.findall(r"\$?\d[\d,]*(?:\.\d+)?", text_ctx)) < 2:
            return False
        if "extract" not in step_lower:
            return False
        if not re.search(r"\bin\s+(?:19|20)\d{2}[a-z]?\b", step_lower):
            return False

        m_metric = re.search(
            r"extract(?:\s+the\s+value\s+for)?\s+(.+?)\s+in\s+(?:19|20)\d{2}[a-z]?",
            step_lower,
            re.IGNORECASE,
        )
        if not m_metric:
            return False
        metric_phrase = re.sub(
            r"\bfrom\s+image\s*\d+\b", "", m_metric.group(1), flags=re.IGNORECASE
        )
        metric_phrase = re.sub(r"\s+", " ", metric_phrase).strip(" ,.-")
        if len(metric_phrase) < 5:
            return False
        return metric_phrase in text_ctx

    def _step_requires_visual_source(self, step_text: str) -> bool:
        """Return True if the step explicitly references image/table/chart as
        the data source.  When True, text-context routing must NOT override."""
        step_lower = (step_text or "").lower()
        visual_markers = [
            "from image", "from the image",
            "from the table", "from table",
            "from the chart", "from chart",
            "from the figure", "from figure",
            "from the graph", "from graph",
            "look at the image", "look at image",
            "in the image", "in image",
        ]
        return any(m in step_lower for m in visual_markers)

    def _should_route_text_context_direct(
        self, step_type: str, step_text: str, user_request: str
    ) -> bool:
        if step_type != "multimodal":
            return False
        if not self._has_meaningful_text_context(user_request):
            return False
        if self._step_requires_visual_source(step_text):
            return False
        return (
            self._step_explicitly_requests_text_context(step_text)
            or self._step_likely_prefers_text_context(step_text, user_request)
        )

    def _extract_step_save_as_variables(self, step_text: str) -> List[str]:
        step_lower = (step_text or "").lower()
        match = re.search(
            r"\bsave_as\s+([a-z_][a-z0-9_]*(?:\s*,\s*[a-z_][a-z0-9_]*)*)",
            step_lower,
        )
        if not match:
            return []
        return [v.strip() for v in match.group(1).split(",") if v.strip()]

    def _get_future_steps_save_as_variables(self) -> set:
        """Collect save_as variables from all steps AFTER the current one."""
        if (
            self.active_plan_id is None
            or self.active_plan_id not in self.planning_tool.plans
            or self.current_step_index is None
        ):
            return set()
        steps = self.planning_tool.plans[self.active_plan_id].get("steps", [])
        future_vars: set = set()
        for i in range(self.current_step_index + 1, len(steps)):
            future_vars.update(self._extract_step_save_as_variables(steps[i]))
        return future_vars

    def _should_attempt_text_fallback(
        self, step_result: str, user_request: str
    ) -> bool:
        """Decide whether a failed multimodal step should be retried with text context.

        Conditions (all must hold):
        1. The original user request contains meaningful text beyond image tags
           (at least 30 chars after stripping <image ...> tags).
        2. The agent explicitly terminated with failure (not an infra / tool error).
        """
        if not self._has_meaningful_text_context(user_request):
            return False

        result_lower = (step_result or "").lower()
        has_agent_failure = "status: failure" in result_lower
        infra_markers = [
            "tool execution failed:",
            "system intercept:",
            "[ocr_error]",
            "error executing step",
        ]
        has_infra_error = any(m in result_lower for m in infra_markers)
        return has_agent_failure and not has_infra_error

    async def _run_text_context_executor(
        self,
        step_info: dict,
        previous_output: str,
        user_request: str,
        step_text: str,
        routing_note: str,
    ) -> tuple[str, bool]:
        """Execute a step via the finance agent using only text context."""
        finance_executor = self.get_executor("finance")
        plan_status = await self._get_plan_text()
        text_context_only = self._strip_image_tags(user_request)
        expected_vars = self._extract_step_save_as_variables(step_text)

        user_context_block = ""
        if user_request.strip():
            user_context_block = f"""
        USER REQUEST / 原始请求:
        ---
        {user_request.strip()}
        ---
        """

        text_context_block = ""
        if text_context_only:
            text_context_block = f"""
        TEXT CONTEXT ONLY / 去除图片标记后的纯文本上下文:
        ---
        {text_context_only}
        ---
        """

        prev_block = ""
        if previous_output.strip():
            structured = self._format_structured_previous_output(previous_output)
            prev_block = f"""
        PREVIOUS STEPS OUTPUT:
        ---
        {structured}
        ---
        """

        expected_vars_block = ""
        if expected_vars:
            expected_vars_block = f"""
        EXPECTED VARIABLES FOR THIS STEP:
        - Save EXACTLY these variables for this step: {", ".join(expected_vars)}
        - Use these exact variable names. Do NOT rename them.
        - Do NOT skip any listed variable.
        - For extraction steps, do NOT create extra derived/result variables from later plan steps.
        - In python_execute, assign and print the variables in the same order.
        """

        output_unit_contract = self._build_output_unit_contract(step_text, user_request)

        fallback_prompt = f"""
        CURRENT PLAN STATUS:
        {plan_status}
        {user_context_block}
        {text_context_block}
        {prev_block}
        {expected_vars_block}
        YOUR CURRENT TASK:
        You are now working on step {self.current_step_index}: "{step_text}"

        ROUTING NOTE:
        {routing_note}

        Treat this as a TEXT-CONTEXT task.
        Do NOT inspect, infer from, or rely on images for this step.
        Extract the required values from the TEXT CONTEXT ONLY block above when possible.
        Use the full USER REQUEST block only for surrounding phrasing and constraints.
        Execute ONLY the current step. If this is an extraction step, save ONLY the variables requested by this step and do NOT perform downstream calculations from later plan steps.

        IMPORTANT EXTRACTION RULES:
        - Use numbers ONLY from the text context or PREVIOUS STEPS OUTPUT.
        - Prefer values that explicitly match the requested entity, date, period, and metric.
        - If nearby text contains multiple numbers, do NOT choose by proximity alone; match the target phrase faithfully.
        - If the text context does not contain the needed information, terminate(status="failure").
        - Do NOT invent values.

        UNIT / SCALE NORMALIZATION (CRITICAL):
        When the text says a value with a scale word, you MUST convert it to the full
        numeric value in python_execute BEFORE saving the variable. Examples:
          "$15.3 million"  → var = 15.3 * 1e6   # 15300000
          "$2.5 billion"   → var = 2.5 * 1e9     # 2500000000
          "12.5 thousand"  → var = 12.5 * 1e3    # 12500
          "$ 37 million"   → var = 37 * 1e6      # 37000000
        PERCENTAGE DECIMAL CONVERSION:
          Convert to decimal (x/100) only when "%" appears in source AND context/variable name
          indicates rate/margin/yield/percentage/growth/return.
          NEVER convert currency/price values ($, €, ¥, or named *_price, *_cost, *_amount,
          *_revenue, *_income, *_expense) — even if "%" appears nearby.
        {output_unit_contract}
        EXCEPTION: If the plan step or the question EXPLICITLY states the answer unit
        (e.g., "in millions", "in percent", "in thousand"), keep the value in that unit
        and note it in a python comment.
        When in doubt about scale words (million/billion/thousand), convert to the base
        (full) numeric form. When in doubt about percentage, add a comment noting the
        ambiguity for downstream verification.

        **NO MENTAL MATH**: ALL computations MUST be done via python_execute.

        When done, provide a summary of what you accomplished.
        """

        try:
            set_step_images_for_ocr(None)
            set_step_context(None)
            set_shared_python_execute(self._ensure_shared_python_execute())

            start_msg_idx = (
                len(finance_executor.memory.messages)
                if hasattr(finance_executor, "memory") and finance_executor.memory
                else 0
            )
            pre_vars = self._snapshot_user_variables()
            fb_result = await finance_executor.run(fallback_prompt)
            post_vars = self._snapshot_user_variables()
            new_vars = {
                k: v
                for k, v in post_vars.items()
                if k not in pre_vars or pre_vars[k] != v
            }
            extra_new_vars = [
                var for var in new_vars if expected_vars and var not in expected_vars
            ]
            if extra_new_vars:
                future_vars = self._get_future_steps_save_as_variables()
                polluting_vars = [v for v in extra_new_vars if v in future_vars]
                benign_vars = [v for v in extra_new_vars if v not in future_vars]
                if polluting_vars:
                    logger.warning(
                        f"Step {self.current_step_index}: removing polluting variables "
                        f"{polluting_vars} (belong to future steps)"
                    )
                    self._remove_shared_python_variables(polluting_vars)
                    new_vars = {
                        k: v for k, v in new_vars.items() if k not in polluting_vars
                    }
                if benign_vars:
                    logger.info(
                        f"Step {self.current_step_index}: keeping extra variables "
                        f"{benign_vars} (not in any future step's save_as)"
                    )
                fb_result += (
                    f"\n[TEXT_CONTEXT_NOTE] Extra variables produced: "
                    f"{', '.join(extra_new_vars)}."
                    + (f" Removed (future-step conflict): {', '.join(polluting_vars)}." if polluting_vars else "")
                )

            missing_expected_vars = [
                var for var in expected_vars if var not in post_vars
            ]

            fb_var_semantics = self._build_variable_semantics(new_vars, step_text) if new_vars else None
            self._record_step_handoff(
                executor=finance_executor,
                step_info=step_info,
                step_result=fb_result,
                start_message_idx=start_msg_idx,
                step_new_variables=new_vars if new_vars else None,
                step_variable_semantics=fb_var_semantics,
            )
            fb_success = self._is_step_result_successful(fb_result, step_text=step_text)
            if fb_success and missing_expected_vars:
                logger.warning(
                    f"Step {self.current_step_index}: text-context execution missing expected "
                    f"variables {missing_expected_vars}"
                )
                fb_success = False
                fb_result = (
                    f"{fb_result}\n[TEXT_CONTEXT_VALIDATION] Missing expected variables: "
                    f"{', '.join(missing_expected_vars)}"
                )
            return fb_result, fb_success
        except Exception as e:
            logger.warning(
                f"Step {self.current_step_index}: text-context execution error: {e}"
            )
            return f"Text-context execution error: {e}", False

    async def _attempt_text_fallback(
        self,
        step_info: dict,
        previous_output: str,
        user_request: str,
        step_text: str,
    ) -> tuple[str, bool]:
        """Re-execute a failed multimodal step using the finance (text) executor."""
        try:
            fb_result, fb_success = await self._run_text_context_executor(
                step_info=step_info,
                previous_output=previous_output,
                user_request=user_request,
                step_text=step_text,
                routing_note=(
                    "The visual context (image) did not contain the needed information for this step. "
                    "Retry the task using text context only."
                ),
            )
            if fb_success:
                logger.info(
                    f"Step {self.current_step_index}: text-context fallback succeeded"
                )
            else:
                logger.info(
                    f"Step {self.current_step_index}: text-context fallback also failed"
                )
            return fb_result, fb_success
        except Exception as e:
            logger.warning(
                f"Step {self.current_step_index}: text-context fallback error: {e}"
            )
            return f"Text fallback error: {e}", False

    async def _get_plan_text(self) -> str:
        """Get the current plan as formatted text."""
        try:
            result = await self.planning_tool.execute(
                command="get", plan_id=self.active_plan_id
            )
            return result.output if hasattr(result, "output") else str(result)
        except Exception as e:
            logger.error(f"Error getting plan: {e}")
            return self._generate_plan_text_from_storage()

    def _generate_plan_text_from_storage(self) -> str:
        """Generate plan text directly from storage if the planning tool fails."""
        try:
            if self.active_plan_id not in self.planning_tool.plans:
                return f"Error: Plan with ID {self.active_plan_id} not found"

            plan_data = self.planning_tool.plans[self.active_plan_id]
            title = plan_data.get("title", "Untitled Plan")
            steps = plan_data.get("steps", [])
            step_statuses = plan_data.get("step_statuses", [])
            step_notes = plan_data.get("step_notes", [])

            while len(step_statuses) < len(steps):
                step_statuses.append(PlanStepStatus.NOT_STARTED.value)
            while len(step_notes) < len(steps):
                step_notes.append("")

            status_counts = {status: 0 for status in PlanStepStatus.get_all_statuses()}
            for status in step_statuses:
                if status in status_counts:
                    status_counts[status] += 1

            completed = status_counts[PlanStepStatus.COMPLETED.value]
            total = len(steps)
            progress = (completed / total) * 100 if total > 0 else 0

            plan_text = f"Plan: {title} (ID: {self.active_plan_id})\n"
            plan_text += "=" * len(plan_text) + "\n\n"

            plan_text += (
                f"Progress: {completed}/{total} steps completed ({progress:.1f}%)\n"
            )
            plan_text += f"Status: {status_counts[PlanStepStatus.COMPLETED.value]} completed, {status_counts[PlanStepStatus.IN_PROGRESS.value]} in progress, "
            plan_text += f"{status_counts[PlanStepStatus.BLOCKED.value]} blocked, {status_counts[PlanStepStatus.NOT_STARTED.value]} not started\n\n"
            plan_text += "Steps:\n"

            status_marks = PlanStepStatus.get_status_marks()

            for i, (step, status, notes) in enumerate(
                zip(steps, step_statuses, step_notes)
            ):
                status_mark = status_marks.get(
                    status, status_marks[PlanStepStatus.NOT_STARTED.value]
                )
                plan_text += f"{i}. {status_mark} {step}\n"
                if notes:
                    plan_text += f"   Notes: {notes}\n"

            return plan_text
        except Exception as e:
            logger.error(f"Error generating plan text from storage: {e}")
            return f"Error: Unable to retrieve plan with ID {self.active_plan_id}"

    async def _finalize_plan(
        self,
        execution_result: str = "",
        user_request: str = "",
    ) -> str:
        """Finalize the plan: 基于执行输出提取并呈现最终答案，直接回应用户请求。"""
        plan_text = await self._get_plan_text()

        try:
            system_message = Message.system_message(
                "You are a planning assistant. Your task is to produce the FINAL ANSWER for the user based on the execution output. "
                "Rules: 1) Extract the actual computed result (numbers, values) from the execution output. "
                "2) Present it clearly as the direct answer to the user's request. "
                "3) Do NOT invent or hallucinate—only use what appears in the execution output. "
                "4) If the user asked for a calculation (e.g. ratio, percentage), state the numeric result prominently. "
                "5) Keep the summary concise; the final answer is the priority."
            )

            user_content = f"""The plan has been completed.

**User request:**
{user_request}

**Plan status:**
{plan_text}

**Execution output (tool results, computed values):**
{execution_result}

Based on the execution output above, provide the FINAL ANSWER that directly addresses the user's request. Extract and state any computed numeric results clearly. Do not invent data."""
            user_message = Message.user_message(user_content)

            response = await self.llm.ask(
                messages=[user_message], system_msgs=[system_message]
            )

            return f"\n\n---\n\n**最终答案**\n\n{response}"
        except Exception as e:
            logger.error(f"Error finalizing plan with LLM: {e}")
            return f"\n\n---\n\n**执行输出**\n\n{execution_result}"
