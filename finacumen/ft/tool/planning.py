"""Backward-compatible module for legacy planning tool imports.

DEPRECATED:
- This shim is intentionally kept for migration safety.
- New code should import `WorkflowStateTool` from `app.tool.workflow_state`.
- Remove this file only after confirming no runtime paths rely on `PlanningTool`.
"""

from finacumen.ft.tool.workflow_state import WorkflowStateTool

# Keep old symbol available for existing imports.
PlanningTool = WorkflowStateTool

