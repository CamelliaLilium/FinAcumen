"""
AntiLoopInterceptor - 防死循环与低级错误拦截器。

解决：
1. same_args_repeated：同一工具同一参数重复调用
2. tool_execution_error：变量名以数字开头等语法错误
3. 错误后无反思的重复尝试

优化点（相对示例代码）：
- terminate/planning/workflow_state 允许重复（控制流工具）
- 仅对 python_execute 等执行类工具做重复拦截
- 区分「上次失败」与「上次成功」：仅在上次失败时拦截重复
- 支持 async 工具执行
"""
import hashlib
import json
import re
from typing import Any, Callable, Dict, Optional, Set, Tuple

from finacumen.ft.logger import logger

# 允许重复调用的工具（控制流，同一参数多次调用属正常）
ALLOW_REPEAT_TOOLS: Set[str] = {"terminate", "planning", "workflow_state"}

# 需拦截重复的工具
INTERCEPT_REPEAT_TOOLS: Set[str] = {"python_execute", "str_replace_editor"}


def _normalize_code_for_hash(code: str) -> str:
    """
    归一化供哈希查重：剥离注释、合并空白，使语义相同的代码产生相同哈希。
    例如：ebitda_2018 = 984 与 ebitda_2018 = 984  # Source: 'Income before...' 视为相同。
    """
    if not code or not code.strip():
        return (code or "").strip()
    lines = []
    for line in code.split("\n"):
        # 移除行末 # 注释（# Source: '...' 等），避免微调注释绕过查重
        if "#" in line:
            line = re.sub(r"\s*#\s*Source:.*$", "", line)  # 先移除 # Source: '...'
            if "#" in line:
                line = re.sub(r"\s*#.*$", "", line)  # 再移除其余行末注释
        line = " ".join(line.split()).strip()  # 合并行内空白
        if line:
            lines.append(line)
    return "\n".join(lines)


def _generate_hash(tool_name: str, kwargs: Dict[str, Any]) -> str:
    """生成确定性的工具调用指纹。python_execute 使用 code 归一化后的哈希以识别语义重复。"""
    args_for_hash = dict(kwargs)
    if tool_name == "python_execute" and "code" in args_for_hash:
        args_for_hash["code"] = _normalize_code_for_hash(
            args_for_hash.get("code") or ""
        )
    canonical_args = json.dumps(args_for_hash, sort_keys=True)
    payload = f"{tool_name}::{canonical_args}"
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _sanitize_variable_names(code: str) -> str:
    """
    自动修改变量名，确保不以数字开头。
    
    例如：
    - '2025_PE = 100' -> 'pe_2025 = 100'
    - '2024_net_profit = 50' -> 'net_profit_2024 = 50'
    - 'value_2025 = 200' -> 'value_2025 = 200' (无需修改)
    
    此功能用于自动修复 LLM 生成的非法变量名，避免 SyntaxError。
    """
    # 匹配以数字开头的标识符（变量名）
    # 模式：数字 + 下划线 + 字母/下划线开头 + 字母数字下划线
    # 例如：2025_PE, 2024_net_profit
    pattern = r'\b(\d+)_([a-zA-Z_][a-zA-Z0-9_]*)\b'
    
    def replace_illegal_name(match):
        number = match.group(1)
        name = match.group(2)
        # 将数字移到名称末尾，并将名称转换为小写
        return f"{name.lower()}_{number}"
    
    # 替换所有以数字开头的变量名
    sanitized_code = re.sub(pattern, replace_illegal_name, code)
    
    return sanitized_code


def _normalize_indent(code: str) -> str:
    """
    Fix stray single leading space that causes 'unexpected indent'.
    Only fixes lines with exactly 1 leading space when prev line did not start a block.
    """
    lines = code.split("\n")
    if not lines:
        return code
    result = []
    for i, line in enumerate(lines):
        if i > 0 and line.startswith(" ") and not line.startswith("  "):
            prev = lines[i - 1].rstrip()
            if prev and not prev.endswith(":"):
                result.append(line.lstrip())
                continue
        result.append(line)
    return "\n".join(result)


def _pre_flight_check(tool_name: str, kwargs: Dict[str, Any]) -> Tuple[bool, str]:
    """
    飞行前检查：执行前拦截低级语法错误。
    1. 变量名不能以数字开头（如 2024_net_profit）- 自动修正
    2. # Source: 后使用三引号会破坏 Python 语法，需用单引号
    3. 单行前多余空格导致 unexpected indent - 自动修正
    """
    if tool_name == "python_execute" and "code" in kwargs:
        code_snippet = kwargs["code"] or ""
        
        # 修复非法缩进（单行前多余空格）
        code_snippet = _normalize_indent(code_snippet)
        # 自动修改变量名（将非法变量名如 2024_net_profit 修正为 net_profit_2024）
        original_code = code_snippet
        sanitized_code = _sanitize_variable_names(code_snippet)
        if sanitized_code != original_code:
            # 变量名被修正，更新 kwargs
            kwargs["code"] = sanitized_code
            logger.info(f"🔄 Auto-sanitized variable names in python_execute code")
        
        # 匹配 # Source: """ — 三引号在注释中会开启多行字符串，导致 unexpected indent（easy-test-1）
        if re.search(r'#\s*Source:\s*["\']{3}', code_snippet):
            return False, (
                "System Pre-flight Error: In # Source comments, use SINGLE quotes only "
                "(e.g. # Source: 'exact snippet'). NEVER use triple quotes (\")—they break Python syntax. "
                "Rewrite: variable = value  # Source: 'your snippet here'"
            )
    return True, ""


def _is_tool_failure(result: Any) -> bool:
    """判断工具执行是否失败"""
    if result is None:
        return True
    if isinstance(result, dict):
        return result.get("success", True) is False
    if hasattr(result, "error") and getattr(result, "error", None):
        return True
    return False


class AntiLoopInterceptor:
    """
    防死循环拦截器。每个 flow 执行前应 reset_memory()。
    """

    def __init__(self):
        # hash -> (count, last_was_error, last_error_detail)
        self._hash_history: Dict[str, Tuple[int, bool, str]] = {}

    def _should_block_repeat(
        self, tool_name: str, call_hash: str
    ) -> Tuple[bool, str]:
        """
        是否应拦截此次重复调用。
        返回 (should_block, reason)。
        """
        if tool_name not in INTERCEPT_REPEAT_TOOLS:
            return False, ""
        if tool_name in ALLOW_REPEAT_TOOLS:
            return False, ""

        if call_hash not in self._hash_history:
            return False, ""

        count, last_was_error, last_error_detail = self._hash_history[call_hash]
        if last_was_error:
            err_hint = (
                f"\n\nLast execution error was: {last_error_detail}"
                if last_error_detail
                else ""
            )
            return True, (
                f"System Intercept: You are calling `{tool_name}` with the EXACT same arguments "
                "that previously FAILED. Do not retry unchanged. "
                "Fix your code based on the error below, then try a NEW approach."
                f"{err_hint}"
            )
        # P0: 成功 1 次后即拦截重复，避免前 2 次重复无法避免
        if count >= 1:
            return True, (
                f"System Intercept: You have already called `{tool_name}` with these same arguments "
                f"{count} time(s) successfully. Stop repeating. Proceed to the next step or call terminate."
            )
        return False, ""

    def _record_call(
        self,
        tool_name: str,
        call_hash: str,
        was_error: bool,
        error_detail: str = "",
    ) -> None:
        if tool_name in ALLOW_REPEAT_TOOLS:
            return
        if tool_name not in INTERCEPT_REPEAT_TOOLS:
            return
        if call_hash in self._hash_history:
            count, _, _ = self._hash_history[call_hash]
            self._hash_history[call_hash] = (count + 1, was_error, error_detail)
        else:
            self._hash_history[call_hash] = (1, was_error, error_detail)

    async def execute_with_reflection(
        self,
        tool_name: str,
        kwargs: Dict[str, Any],
        actual_execute_func: Callable,
    ) -> Any:
        """
        包装工具执行，注入预检、重复拦截与错误反思。

        Args:
            tool_name: 工具名
            kwargs: 工具参数
            actual_execute_func: 实际执行函数，签名为 async def fn(**kwargs) -> Any

        Returns:
            工具返回结果，或拦截/错误时的提示字符串
        """
        # 1. 飞行前语法校验
        is_valid, error_msg = _pre_flight_check(tool_name, kwargs)
        if not is_valid:
            logger.warning(f"🛑 Pre-flight blocked: {error_msg[:80]}...")
            return error_msg

        # 2. 哈希查重
        call_hash = _generate_hash(tool_name, kwargs)
        should_block, block_reason = self._should_block_repeat(tool_name, call_hash)
        if should_block:
            logger.warning(f"🛑 Repeat blocked for {tool_name}: {block_reason[:60]}...")
            return block_reason

        # 3. 执行
        try:
            result = await actual_execute_func(**kwargs)
            was_error = _is_tool_failure(result)
            error_detail = ""
            if was_error:
                error_detail = (
                    str(result.get("observation", result))
                    if isinstance(result, dict)
                    else str(result)
                )
            self._record_call(tool_name, call_hash, was_error, error_detail)

            if was_error:
                return (
                    f"Tool Execution Failed:\n{error_detail}\n\n"
                    "ACTION REQUIRED: Do not retry unchanged. "
                    "Analyze the error above, fix your code (e.g., variable names, syntax), and try a NEW approach."
                )
            return result

        except Exception as e:
            error_detail = str(e)
            self._record_call(tool_name, call_hash, was_error=True, error_detail=error_detail)
            error_traceback = str(e)
            logger.exception(f"Tool {tool_name} failed: {error_traceback}")
            return (
                f"Tool Execution Failed with Exception:\n{error_traceback}\n\n"
                "ACTION REQUIRED: Do not retry unchanged. "
                "Analyze the traceback above, fix your code, and try a NEW approach."
            )

    def reset_memory(self) -> None:
        """每个新样本开始时调用，清理状态"""
        self._hash_history.clear()
