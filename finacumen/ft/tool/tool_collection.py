"""Collection classes for managing multiple tools."""
from typing import Any, Dict, List, Optional

from finacumen.ft.exceptions import ToolError
from finacumen.ft.logger import logger
from finacumen.ft.tool.base import BaseTool, ToolFailure, ToolResult


class ToolCollection:
    """A collection of defined tools."""

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        *tools: BaseTool,
        use_planning_param_for: Optional[str] = None,
    ):
        """
        Args:
            *tools: Tools to include.
            use_planning_param_for: If set (e.g. "workflow_state"), to_params() will
                use to_planning_param() for that tool instead of to_param().
        """
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in tools}
        self.alias_map = {}
        for tool in tools:
            for alias in getattr(tool, "aliases", []) or []:
                if alias and alias not in self.tool_map:
                    self.alias_map[alias] = tool
        self.use_planning_param_for = use_planning_param_for

    def __iter__(self):
        return iter(self.tools)

    def to_params(self) -> List[Dict[str, Any]]:
        result = []
        for tool in self.tools:
            if (
                self.use_planning_param_for
                and getattr(tool, "name", None) == self.use_planning_param_for
                and hasattr(tool, "to_planning_param")
            ):
                result.append(tool.to_planning_param())
            else:
                result.append(tool.to_param())
        return result

    async def execute(
        self, *, name: str, tool_input: Dict[str, Any] = None
    ) -> ToolResult:
        tool = self.tool_map.get(name) or self.alias_map.get(name)
        if not tool:
            return ToolFailure(error=f"Tool {name} is invalid")
        try:
            result = await tool(**tool_input)
            return result
        except ToolError as e:
            return ToolFailure(error=e.message)

    async def execute_all(self) -> List[ToolResult]:
        """Execute all tools in the collection sequentially."""
        results = []
        for tool in self.tools:
            try:
                result = await tool()
                results.append(result)
            except ToolError as e:
                results.append(ToolFailure(error=e.message))
        return results

    def get_tool(self, name: str) -> BaseTool:
        return self.tool_map.get(name) or self.alias_map.get(name)

    def add_tool(self, tool: BaseTool):
        """Add a single tool to the collection.

        If a tool with the same name already exists, it will be skipped and a warning will be logged.
        """
        if tool.name in self.tool_map:
            logger.warning(f"Tool {tool.name} already exists in collection, skipping")
            return self

        self.tools += (tool,)
        self.tool_map[tool.name] = tool
        for alias in getattr(tool, "aliases", []) or []:
            if alias and alias not in self.tool_map:
                self.alias_map[alias] = tool
        return self

    def add_tools(self, *tools: BaseTool):
        """Add multiple tools to the collection.

        If any tool has a name conflict with an existing tool, it will be skipped and a warning will be logged.
        """
        for tool in tools:
            self.add_tool(tool)
        return self
