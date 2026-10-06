"""工作流编排层：用 LangGraph 把多步 LLM 任务串成可持久化、可恢复的流程图。

当前提供学习测评工作流（app/workflows/learning_assessment.py）：
- 启动图：prepare_learning_context -> generate_quiz -> END
- 提交图：evaluate_and_review -> END

图本身不碰数据库：状态在这一层是纯数据（LearningWorkflowState），
跨请求的持久化 / 恢复由 app/services/learning_workflow_service.py 负责。
"""

from app.workflows.learning_assessment import (
    LearningWorkflowState,
    WorkflowGenerationError,
    build_start_graph,
    build_submit_graph,
)

__all__ = [
    "LearningWorkflowState",
    "WorkflowGenerationError",
    "build_start_graph",
    "build_submit_graph",
]
