from finacumen.ft.tool.base import BaseTool


_TERMINATE_DESCRIPTION = """Terminate the interaction when the request is met OR if the assistant cannot proceed further with the task.
When you have finished all the tasks, call this tool to end the work."""


class Terminate(BaseTool):
    name: str = "terminate"
    description: str = _TERMINATE_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "description": "The finish status of the interaction.",
                "enum": ["success", "failure", "forced"],
            },
            "final_answer": {
                "type": "string",
                "description": "The final answer to the question. Set this when you have determined the answer.",
            },
        },
        "required": ["final_answer"],
    }

    async def execute(self, status: str = None, final_answer: str = None) -> str:
        """Finish the current execution"""
        parts = []
        if status:
            parts.append(f"The interaction has been completed with status: {status}")
        if final_answer:
            parts.append(f"Final answer: {final_answer}")
        if not parts:
            parts.append("The interaction has been completed.")
        return ". ".join(parts)
