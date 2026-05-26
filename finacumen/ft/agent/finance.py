"""
FinanceAgent - 简化版自定义智能体示例

本文件展示了如何快速创建一个专注于特定领域的简化版智能体。
相比 Manus 的通用模板，这是一个更聚焦金融场景的轻量级模板。

简化版 vs 全功能版 (manus.py) 的区别：
- 工具集合更聚焦金融分析任务
- 默认不依赖联网检索工具，便于后续接入经验记忆检索等功能
- 保持统一的最终答复输出链路
- 更少的工具和更专注的能力

适合场景：
- 单一领域的专业任务
- 不需要浏览器自动化
- 工具集相对固定
- 快速原型开发
"""
from pydantic import Field

from finacumen.ft.agent.toolcall import ToolCallAgent
from finacumen.ft.config import config
from finacumen.ft.prompt.finance import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from finacumen.ft.tool import Terminate, ToolCollection
from finacumen.ft.tool.python_execute import PythonExecute
from finacumen.ft.tool.str_replace_editor import StrReplaceEditor


class FinanceAgent(ToolCallAgent):
    """
    金融分析专用智能体

    这是一个简化版自定义智能体的最佳实践示例，展示了：
    1. 如何继承 ToolCallAgent 基类
    2. 如何配置领域特定的工具集
    3. 如何定义专业的角色定位

    设计思路：
    - 专注：只提供金融分析相关的核心工具
    - 简洁：省略不需要的复杂功能（如 MCP、浏览器）
    - 专业：通过 system_prompt 定义金融领域的专业角色

    工具集说明：
    - PythonExecute: 执行数据分析和计算代码
    - StrReplaceEditor: 读取和编辑金融报告文件
    - Terminate: 完成任务并返回结果
    """

    # ==================== 1. 基础身份配置 ====================
    # name: 智能体名称，用于日志和识别
    # description: 智能体描述，说明其专业领域和能力范围
    name: str = "Finance"
    description: str = (
        "A specialized financial agent that focuses on deterministic financial "
        "analysis with local tools and computation-first execution."
    )

    # ==================== 2. Prompt 配置 ====================
    # system_prompt: 定义智能体的专业角色和系统行为
    # 通过 format() 注入动态变量（如工作目录）
    system_prompt: str = SYSTEM_PROMPT.format(directory=config.workspace_root)

    # next_step_prompt: 指导智能体如何规划下一步行动
    # 通常包含领域特定的思考框架和约束条件
    next_step_prompt: str = NEXT_STEP_PROMPT

    # ==================== 3. 运行限制配置 ====================
    # max_observe: 控制单次工具输出的最大长度
    # 金融分析可能需要较长的数据输出，可适当放宽
    max_observe: int = 15000

    # max_steps: 单次任务的最大执行步数
    # 防止复杂分析任务陷入无限循环
    max_steps: int = 20

    # ==================== 4. 工具配置 ====================
    # available_tools 定义智能体可调用的所有工具
    # 使用 Field + default_factory 确保每个实例有独立的工具集合
    # 避免 Python 可变对象作为类属性的常见问题
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            PythonExecute(),      # 数据分析和计算
            StrReplaceEditor(),   # 文件读写和编辑
            Terminate(),          # 任务终止
        )
    )

    # ==================== 5. 特殊工具配置 ====================
    # special_tool_names 中的工具会触发任务结束
    # Terminate 被调用后，Agent 状态会变为 FINISHED
    special_tool_names: list[str] = Field(default_factory=lambda: [Terminate().name])

    # ==================== 6. 可选：自定义方法 ====================
    # 如果不需要特殊逻辑，可以不重写任何方法
    # ToolCallAgent 的默认 think() 和 act() 已经足够应对大多数场景

    # 如需自定义，可参考以下模板：
    #
    # async def think(self) -> bool:
    #     """在父类思考逻辑前后添加自定义处理"""
    #     # 前置处理：如加载数据、准备上下文
    #     result = await super().think()
    #     # 后置处理：如分析结果、调整策略
    #     return result
    #
    # async def act(self) -> str:
    #     """自定义工具执行逻辑"""
    #     return await super().act()
    #
    # async def cleanup(self):
    #     """清理自定义资源"""
    #     await super().cleanup()

    # ==================== 快速创建新 Agent 的清单 ====================
    # 基于本模板创建新 Agent 的步骤：
    #
    # 1. 复制本文件，重命名为你的领域（如 medical.py, legal.py）
    # 2. 修改类名（如 MedicalAgent, LegalAgent）
    # 3. 更新 name 和 description
    # 4. 创建对应的 prompt 文件（如 app/prompt/medical.py）
    # 5. 调整 available_tools 中的工具列表
    # 6. 根据需要调整 max_observe 和 max_steps
    # 7. （可选）添加自定义方法处理特殊需求
    # 8. 在应用入口处实例化并使用你的 Agent
    #
    # 最佳实践：
    # - 保持工具集精简，只包含领域必需的工具
    # - 使用清晰的命名和详细的 docstring
    # - 通过 prompt 定义明确的专业角色和行为边界
    # - 测试不同场景下的表现，调整参数优化效果
