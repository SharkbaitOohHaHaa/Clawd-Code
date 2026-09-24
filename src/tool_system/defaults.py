from __future__ import annotations

from .registry import ToolRegistry
from .tools import (
    AskUserQuestionTool,
    BriefTool,
    CronCreateTool,
    CronDeleteTool,
    CronListTool,
    DataInspectTool,
    DataTransformTool,
    EnterPlanModeTool,
    EnterWorktreeTool,
    ExitPlanModeTool,
    ExitWorktreeTool,
    FileEditTool,
    FileReadTool,
    FileWriteTool,
    GlobTool,
    GrepTool,
    GeminiThinkTool,
    LSPTool,
    MemoryTool,
    NotebookEditTool,
    OsvQueryTool,
    ListMcpResourcesTool,
    ReadMcpResourceTool,
    ListMcpToolsTool,
    MCPTool,
    QwenMediaAnalyzeTool,
    SendMessageTool,
    SendUserMessageTool,
    SkillTool,
    SleepTool,
    StructuredOutputTool,
    TeamCreateTool,
    TeamDeleteTool,
    TaskCreateTool,
    TaskGetTool,
    TaskListTool,
    TaskOutputTool,
    TaskStopTool,
    TaskUpdateTool,
    TodoWriteTool,
    WebFetchTool,
    WebSearchTool,
    YouTubeAnalyzeTool,
)
from .tools.agent import AgentTool
from .tools.tool_search import ToolSearchTool


def build_default_registry(*, include_user_tools: bool = False) -> ToolRegistry:
    registry = ToolRegistry(
        tools=[
            SendUserMessageTool(),
            FileReadTool(),
            FileWriteTool(),
            FileEditTool(),
            DataInspectTool(),
            DataTransformTool(),
            GlobTool(),
            GrepTool(),
            GeminiThinkTool(),
            LSPTool(),
            QwenMediaAnalyzeTool(),
            WebSearchTool(),
            WebFetchTool(),
            OsvQueryTool(),
            YouTubeAnalyzeTool(),
            SleepTool(),
            TaskStopTool(),
            BriefTool(),
            AskUserQuestionTool(),
            MemoryTool(),
            NotebookEditTool(),
            ListMcpResourcesTool(),
            ReadMcpResourceTool(),
            ListMcpToolsTool(),
            MCPTool(),
            TodoWriteTool(),
            TaskCreateTool(),
            TaskGetTool(),
            TaskListTool(),
            TaskUpdateTool(),
            TaskOutputTool(),
            TeamCreateTool(),
            TeamDeleteTool(),
            EnterPlanModeTool(),
            EnterWorktreeTool(),
            ExitPlanModeTool(),
            ExitWorktreeTool(),
            CronCreateTool(),
            CronListTool(),
            CronDeleteTool(),
            SendMessageTool(),
            StructuredOutputTool(),
            SkillTool(),
        ]
    )
    registry.register(AgentTool(registry))
    registry.register(ToolSearchTool(registry))

    if include_user_tools:
        raise RuntimeError(
            "Direct ~/.clawd/tools Python imports are disabled. "
            "Use the operator-trusted Python plugin runtime instead."
        )

    return registry
