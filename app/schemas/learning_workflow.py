"""校验层：学习测评工作流的请求 / 响应模型。

前后端边界：
- 后端只返回标准化 quiz JSON（question_id / type / stem / options），
  **不含标准答案**（答案留在服务端 state 里供阅卷）；
- 前端按 type 渲染对应表单组件；
- 用户只提交 question_id + 对应答案 value。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

QuestionType = str  # 取值见 app/workflows/learning_assessment.QUESTION_TYPES 的说明


class LearningWorkflowStartRequest(BaseModel):
    """发起测评：在某个会话下，按用户需求生成一套题目。"""

    model_config = ConfigDict(str_strip_whitespace=True)

    session_id: int = Field(ge=1, description="所属会话 id（校验归属）")
    request_text: str = Field(min_length=1, max_length=4000, description="测评需求，如「Java 并发基础，10 题」")
    question_count: int = Field(default=5, ge=1, le=20, description="题目数量，1-20")
    provider: str | None = Field(default=None, max_length=32, description="模型 provider，留空用默认")


class QuizAnswer(BaseModel):
    """单题作答：用户只提交 question_id 与对应 value。"""

    model_config = ConfigDict(str_strip_whitespace=True)

    question_id: str = Field(min_length=1, max_length=64, description="题目 id（来自返回的 quiz）")
    value: str | list[str] = Field(default="", description="答案：单选/判断/填空/简答为字符串，多选为字符串数组")


class LearningWorkflowSubmitRequest(BaseModel):
    """提交整套答案：触发评分、逐题点评、薄弱点提取与学习建议。"""

    answers: list[QuizAnswer] = Field(min_length=1, max_length=50, description="整套答案，至少一题")


class QuizOption(BaseModel):
    """选项：key 供作答引用（如 "A"），text 供渲染。"""

    key: str
    text: str


class QuizQuestion(BaseModel):
    """标准题目：前端按 type 选择表单组件。

    type：single_choice（单选）/ multiple_choice（多选）/ true_false（判断）/
    fill_blank（填空）/ short_answer（简答）。
    """

    question_id: str
    type: QuestionType
    stem: str
    options: list[QuizOption] | None = None


class EvaluationItem(BaseModel):
    """逐题点评：correct 为 None 表示模型未判定对错。"""

    question_id: str
    correct: bool | None = None
    comment: str = ""


class LearningWorkflowStartResponse(BaseModel):
    """发起测评响应：run_id 用于第二次提交，quiz 供前端渲染答题表单。"""

    run_id: str
    status: str
    quiz: list[QuizQuestion]


class LearningWorkflowSubmitResponse(BaseModel):
    """提交响应：总分、逐题点评、薄弱点与学习建议。"""

    run_id: str
    status: str
    score: float | None = None
    evaluation: list[EvaluationItem] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    review: str = ""


class LearningWorkflowDetailResponse(BaseModel):
    """查询响应：run_id + 状态 + 完整状态快照（跨请求恢复用）。"""

    run_id: str
    status: str
    state: dict[str, Any] = Field(default_factory=dict)
