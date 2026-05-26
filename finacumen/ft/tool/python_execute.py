"""
PythonExecute - 持久化命名空间的 Python 执行工具。

支持多步计算：变量在多次调用间保留。每次 flow 开始时重置环境。
支持 REPL 风格：代码末尾的裸表达式（如 interest_expense, a, b）会自动 print 输出。
"""
import ast
import re
import sys
from io import StringIO
from typing import Any, Dict, Optional

from pydantic import PrivateAttr

from finacumen.ft.tool.base import BaseTool


def _fix_literal_newlines(code: str) -> str:
    """
    修复 JSON 中字面量 \\n 导致的 EMPTY_OUTPUT（easy-test-14 类问题）。
    LLM 在 JSON 中输出 \\n 时，解析后可能得到字面量 \\n（反斜杠+n 两字符），
    导致整行被当作注释或 print 未执行。将字面量 \\n 替换为实际换行符。
    """
    return code.replace("\\n", "\n")


def _fix_print_after_comment(code: str) -> str:
    """
    修复「print() 与 # 注释同行」导致的输出为空问题。
    LLM 常生成:  variable = 9300  # Source: '...'  print(variable)
    Python 会把 print 当作注释的一部分，导致无输出。
    将 print 提取到单独一行。
    """
    lines = code.split("\n")
    result = []
    for line in lines:
        hash_pos = line.find("#")
        if hash_pos < 0:
            result.append(line)
            continue
        after_hash = line[hash_pos + 1 :]
        # 在 # 之后查找 print(...)，支持 print(x) 或 print(a, b)
        match = re.search(r"\bprint\s*\([^)]*\)", after_hash)
        if match:
            before_hash = line[:hash_pos].rstrip()
            print_stmt = match.group(0)
            indent = len(line) - len(line.lstrip())
            result.append(before_hash)
            result.append(" " * indent + print_stmt)
        else:
            result.append(line)
    return "\n".join(result)


def _default_global_env() -> Dict[str, Any]:
    """构建初始 globals，含 __builtins__。"""
    if isinstance(__builtins__, dict):
        return {"__builtins__": __builtins__}
    return {"__builtins__": __builtins__.__dict__.copy()}


def _wrap_last_expression(code: str) -> str:
    """
    若代码最后一条是裸表达式（如 x 或 x, y），自动添加 print 以产生输出。
    解决「哑巴代码」：单写变量名在 exec() 中无输出。
    若最后一条已是 print(...) 调用，则跳过，避免 print(print(...)) 产生冗余 None。
    """
    code = code.strip()
    if not code:
        return code
    try:
        tree = ast.parse(code)
        if not tree.body:
            return code
        last = tree.body[-1]
        if not isinstance(last, ast.Expr):
            return code
        # 若已是 print(...) 调用，不重复包装
        if isinstance(last.value, ast.Call):
            if isinstance(last.value.func, ast.Name):
                if last.value.func.id == "print":
                    return code
        # 裸表达式，需要 print
        try:
            expr_src = ast.unparse(last.value)
        except AttributeError:
            return code
        return code + f"\nprint({expr_src})"
    except SyntaxError:
        pass
    return code


def _normalize_indent(code: str) -> str:
    """
    Fix stray single leading space that causes 'unexpected indent' errors.
    Only fixes lines with exactly 1 leading space when the previous line did not
    start a block (:) - common LLM mistake. Does not touch valid block indentation.
    """
    lines = code.split("\n")
    if not lines:
        return code
    result = []
    for i, line in enumerate(lines):
        # Exactly 1 leading space (common mistake); 2+ spaces may be valid block indent
        if i > 0 and line.startswith(" ") and not line.startswith("  "):
            prev = lines[i - 1].rstrip()
            if prev and not prev.endswith(":"):
                result.append(line.lstrip())
                continue
        result.append(line)
    return "\n".join(result)


def _sanitize_variable_names(code: str) -> str:
    """
    自动修改变量名，确保不以数字开头。
    
    例如：
    - '2025_PE = 100' -> 'pe_2025 = 100'
    - '2024_net_profit = 50' -> 'net_profit_2024 = 50'
    - 'value_2025 = 200' -> 'value_2025 = 200' (无需修改)
    
    此功能用于自动修复 LLM 生成的非法变量名，避免 SyntaxError。
    """
    import re
    
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


class PythonExecute(BaseTool):
    """A tool for executing Python code with persistent namespace across calls."""

    name: str = "python_execute"
    description: str = (
        "Executes Python code for deterministic computation and data transformation. "
        "Use this tool when arithmetic, parsing, ratio/aggregation, or algorithmic processing is required, "
        "including cases where numeric inputs are already provided in text context. "
        "Typical fit: growth rate, ROA/ROE, ratio comparison, unit conversion, multi-step financial formulas. "
        "Variables PERSIST across calls within the same flow—Step 1 can use variables defined in Step 0. "
        "For multi-step calculations, you may split: Step 0 extract/assign, Step 1 compute using those variables. "
        "Always use print() for output, or write a bare expression as the last line (e.g. result or a, b, c) "
        "which will be auto-printed. Only printed outputs are visible; function return values are not captured. "
        "IMPORTANT: Use numbers from the user's context directly—do NOT invent fake text or placeholder values. "
        "Avoid f-strings with nested braces (e.g. f\"{{x: {y}}}\")—use print(a, b, c) or string concatenation instead. "
        "You MUST perform the actual arithmetic/formula and output the result in the SAME code block as variable assignment. Do not stop at just printing the variables. "
        "When assigning values, use: variable = value  # Source: 'exact snippet' (SINGLE quotes only—never triple quotes). "
        "Python rules: variable names cannot start with a digit (use net_profit_2024 not 2024_net_profit). "
        "If execution fails, read the error message and fix the code before retrying."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code to execute. Variables persist across calls.",
            },
        },
        "required": ["code"],
    }

    _global_env: Dict[str, Any] = PrivateAttr(default_factory=_default_global_env)

    def reset_env(self) -> None:
        """重置执行环境。Flow 开始时调用，确保新请求不受旧请求影响。"""
        self._global_env.clear()
        self._global_env.update(_default_global_env())

    def _run_code(self, code: str) -> tuple[str, bool]:
        """执行代码，返回 (observation, success)。"""
        original_stdout = sys.stdout
        try:
            output_buffer = StringIO()
            sys.stdout = output_buffer

            # 修复 JSON 字面量 \n 转义（easy-test-14 类问题）
            code = _fix_literal_newlines(code)
            # 修复 print 与 # 同行被注释（easy-test-18 类问题）
            code = _fix_print_after_comment(code)
            # 修复非法缩进（如单行前多余空格导致的 unexpected indent）
            code = _normalize_indent(code)
            # 自动修改变量名，确保不以数字开头
            code = _sanitize_variable_names(code)
            
            # REPL 风格：末尾裸表达式自动 print
            code = _wrap_last_expression(code)

            exec(code, self._global_env, self._global_env)
            # 显式 flush，避免某些环境下缓冲导致输出丢失
            if hasattr(output_buffer, "flush"):
                output_buffer.flush()
            observation = output_buffer.getvalue()
            # 空输出检测：success 但无输出时，视为失败并返回明确提示，避免下游 Agent 编造数值
            if observation.strip() == "" and "print" in code.lower():
                return (
                    "[EMPTY_OUTPUT] python_execute completed but produced no output. "
                    "Check: (1) print() is on its own line, not after # comment; "
                    "(2) variables exist before print; (3) use print(x) not bare x for output.",
                    False,
                )
            return observation, True
        except Exception as e:
            return str(e), False
        finally:
            sys.stdout = original_stdout

    async def execute(
        self,
        code: Optional[str] = None,
        timeout: int = 5,
        **kwargs: Any,
    ) -> Dict:
        """
        Executes the provided Python code. Variables persist across calls.

        Args:
            code (str): The Python code to execute. May be missing when LLM returns empty/truncated arguments.
            timeout (int): Reserved for future use; currently execution runs in-process.
            **kwargs: Additional args (e.g. from tool_input); code may come from kwargs when called as tool(**args).

        Returns:
            Dict: Contains 'observation' with execution output or error message and 'success' status.
        """
        code = code or kwargs.get("code") or ""
        if not code or not str(code).strip():
            return {
                "observation": "[ERROR] python_execute requires 'code' argument. "
                "The model may have returned empty or truncated arguments (e.g. due to token limit). "
                "Please retry with valid code.",
                "success": False,
            }
        observation, success = self._run_code(str(code))
        return {"observation": observation, "success": success}
