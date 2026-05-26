from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import List, Optional

import re

from pydantic import BaseModel, Field, model_validator

from finacumen.ft.llm import LLM
from finacumen.ft.logger import logger
# from finacumen.ft.sandbox.client import SANDBOX_CLIENT  # 如果不用 sandbox，可以注释掉
from finacumen.ft.schema import ROLE_TYPE, AgentState, Memory, Message


class BaseAgent(BaseModel, ABC):
    """Abstract base class for managing agent state and execution.

    Provides foundational functionality for state transitions, memory management,
    and a step-based execution loop. Subclasses must implement the `step` method.
    """

    # Core attributes
    name: str = Field(..., description="Unique name of the agent")
    description: Optional[str] = Field(None, description="Optional agent description")

    # Prompts（系统prompt：定义Agent角色和能力，和next_step_prompt下一步做什么）
    system_prompt: Optional[str] = Field(
        None, description="System-level instruction prompt"
    )
    next_step_prompt: Optional[str] = Field(
        None, description="Prompt for determining next action"
    )

    # Dependencies
    llm: LLM = Field(default_factory=LLM, description="Language model instance")
    memory: Memory = Field(default_factory=Memory, description="Agent's memory store")
    state: AgentState = Field(
        default=AgentState.IDLE, description="Current agent state"
    )

    # Execution control
    max_steps: int = Field(default=10, description="Maximum steps before termination")
    current_step: int = Field(default=0, description="Current step in execution")

    duplicate_threshold: int = 2

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"  # Allow extra fields for flexibility in subclasses

    @model_validator(mode="after")
    def initialize_agent(self) -> "BaseAgent":
        """Initialize agent with default settings if not provided."""
        if self.llm is None or not isinstance(self.llm, LLM):
            self.llm = LLM(config_name=self.name.lower())
        if not isinstance(self.memory, Memory):
            self.memory = Memory()
        return self

    @asynccontextmanager
    async def state_context(self, new_state: AgentState):  # 状态上下文管理器，为了安全的状态转换
        """Context manager for safe agent state transitions.

        Args:
            new_state: The state to transition to during the context.

        Yields:
            None: Allows execution within the new state.

        Raises:
            ValueError: If the new_state is invalid.
        """
        if not isinstance(new_state, AgentState):
            raise ValueError(f"Invalid state: {new_state}")

        previous_state = self.state
        self.state = new_state
        try:
            yield
        except Exception as e:
            self.state = AgentState.ERROR  # Transition to ERROR on failure
            raise e
        finally:
            self.state = previous_state  # Revert to previous state

    def update_memory(
        self,
        role: ROLE_TYPE,  # type: ignore
        content: str,
        base64_image: Optional[str] = None,
        base64_images: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        """Add a message to the agent's memory.

        Args:
            role: The role of the message sender (user, system, assistant, tool).
            content: The message content.
            base64_image: Optional base64 encoded image (单图).
            base64_images: Optional list of base64 images (多图，优先于 base64_image).
            **kwargs: Additional arguments (e.g., tool_call_id for tool messages).

        Raises:
            ValueError: If the role is unsupported.
        """
        message_map = {
            "user": Message.user_message,
            "system": Message.system_message,
            "assistant": Message.assistant_message,
            "tool": lambda content, **kw: Message.tool_message(content, **kw),
        }

        if role not in message_map:
            raise ValueError(f"Unsupported message role: {role}")

        msg_kw = kwargs if role == "tool" else {}
        if base64_images:
            msg_kw["base64_images"] = base64_images
        else:
            msg_kw["base64_image"] = base64_image
        self.memory.add_message(message_map[role](content, **msg_kw))

    async def run(
        self,
        request: Optional[str] = None,
        base64_image: Optional[str] = None,
        base64_images: Optional[List[str]] = None,
    ) -> str:
        """Execute the agent's main loop asynchronously.

        Args:
            request: Optional initial user request to process.
            base64_image: Optional base64-encoded image for multimodal agents (单图).
            base64_images: Optional list of base64 images (多图，优先于 base64_image).

        Returns:
            A string summarizing the execution results.

        Raises:
            RuntimeError: If the agent is not in IDLE state at start.
        """
        if self.state != AgentState.IDLE:
            raise RuntimeError(f"Cannot run agent from state: {self.state}")

        # Each run starts with a fresh step counter.
        # Without this reset, reusing the same agent instance across flow steps
        # can inherit stale step progress and cause unexpected looping behavior.
        self.current_step = 0

        if request:
            self.update_memory(
                "user",
                request,
                base64_image=base64_image,
                base64_images=base64_images,
            )

        results: List[str] = []
        async with self.state_context(AgentState.RUNNING):
            while (
                self.current_step < self.max_steps and self.state != AgentState.FINISHED
            ):
                self.current_step += 1
                logger.info(f"Executing step {self.current_step}/{self.max_steps}")
                step_result = await self.step()

                # Check for stuck state
                if self.is_stuck():
                    self.handle_stuck_state()

                results.append(f"Step {self.current_step}: {step_result}")

            if self.current_step >= self.max_steps:
                # Preserve actual step count for downstream diagnostics
                self._forced_answer = _extract_last_computed_value(self)
                self.state = AgentState.IDLE
                results.append(f"Terminated: Reached max steps ({self.max_steps})")
        # await SANDBOX_CLIENT.cleanup()  # 如果不用 sandbox，可以注释掉
        return "\n".join(results) if results else "No steps executed"

    @abstractmethod
    async def step(self) -> str:
        """Execute a single step in the agent's workflow.

        Must be implemented by subclasses to define specific behavior.
        """

    def handle_stuck_state(self):
        """Detect stuck type and inject targeted recovery prompt."""
        last_content = ""
        if hasattr(self, "memory") and getattr(self.memory, "messages", None):
            for msg in reversed(self.memory.messages):
                if getattr(msg, "role", "") == "assistant" and getattr(msg, "content", ""):
                    last_content = msg.content
                    break

        if last_content and re.search(
            r"(?:answer|result|margin|value)\s*(?:is|=|:)\s*[-+]?[\d,.]+",
            last_content,
            re.IGNORECASE,
        ):
            stuck_prompt = (
                "DETECTED: you have a conclusion but did not call terminate. "
                "Your ONLY valid next action: terminate(final_answer='<your numeric value>'). "
                "Do NOT explain — call terminate with your answer NOW."
            )
        else:
            stuck_prompt = (
                "Observed duplicate responses. Consider new strategies and avoid "
                "repeating ineffective paths already attempted."
            )
        self.next_step_prompt = f"{stuck_prompt}\n{self.next_step_prompt}"
        logger.warning(f"Agent stuck. Recovery: {stuck_prompt[:80]}...")

    def is_stuck(self) -> bool:
        """Check if the agent is stuck in a loop by detecting duplicate content"""
        if len(self.memory.messages) < 2:
            return False

        last_message = self.memory.messages[-1]
        if not last_message.content:
            return False

        # Count identical content occurrences
        duplicate_count = sum(
            1
            for msg in reversed(self.memory.messages[:-1])
            if msg.role == "assistant" and msg.content == last_message.content
        )

        return duplicate_count >= self.duplicate_threshold

    @property
    def messages(self) -> List[Message]:
        """Retrieve a list of messages from the agent's memory."""
        return self.memory.messages

    @messages.setter
    def messages(self, value: List[Message]):
        """Set the list of messages in the agent's memory."""
        self.memory.messages = value


def _extract_last_computed_value(agent) -> str:
    """Fallback: extract the last python_execute output value when agent exhausts steps."""
    for msg in reversed(getattr(agent, "memory", None) and agent.memory.messages or []):
        if getattr(msg, "role", "") != "tool":
            continue
        content = getattr(msg, "content", "") or ""
        numbers = re.findall(r"[-+]?\d+\.?\d*", str(content))
        if numbers:
            return numbers[-1]
    return ""
