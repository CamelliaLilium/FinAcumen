"""Lazy exports for tool classes."""
from importlib import import_module

_LAZY_EXPORTS = {
    "BaseTool":            ("finacumen.ft.tool.base", "BaseTool"),
    "ToolCollection":      ("finacumen.ft.tool.tool_collection", "ToolCollection"),
    "Terminate":           ("finacumen.ft.tool.terminate", "Terminate"),
    "WorkflowStateTool":   ("finacumen.ft.tool.workflow_state", "WorkflowStateTool"),
    "PlanningTool":        ("finacumen.ft.tool.workflow_state", "PlanningTool"),
    "PythonExecute":       ("finacumen.ft.tool.python_execute", "PythonExecute"),
    "StrReplaceEditor":    ("finacumen.ft.tool.str_replace_editor", "StrReplaceEditor"),
    "OcrExtract":          ("finacumen.ft.tool.ocr", "OcrExtract"),
    "FinancialDataLookup": ("finacumen.ft.tool.financial_data_lookup", "FinancialDataLookup"),
}

__all__ = list(_LAZY_EXPORTS.keys())


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'finacumen.ft.tool' has no attribute '{name}'")
    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
