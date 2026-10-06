"""学习测评工作流：LangGraph 流程编排 + 结构化题目 / 评分结果的生产与归一化。

拆分两次请求（两段图）：

    start   : prepare_learning_context -> generate_quiz -> END     产出题目
    submit  : evaluate_and_review -> END                           一次性评分 + 点评 + 薄弱点 + 建议

设计要点：

- **状态纯数据**：``LearningWorkflowState`` 只保留闭环必须字段，节点返回「增量 dict」，
  由 LangGraph 合并；服务层把整份 state 序列化进 ``workflow_runs.state_json``，
  第二次请求按 run_id 恢复题目继续评分。
- **节点不碰数据库 / HTTP**：只做轻量状态标记与 LLM 调用，便于单测与替换模型。
- **模型按 provider 解析**：图构建时传入 resolver，节点执行时才取模型实例
  （provider 存在 state 里），测试注入假模型即可离线跑。
- **题目答案只留在服务端 state**：``quiz`` 里可带 ``answer``（供阅卷），
  对外响应由 ``public_quiz`` 剥掉，不泄露答案。

不做：不落库（服务层）、不鉴权（路由层）、不拼最终提示词模板（直接用固定 system）。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from typing import Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from app.core.exceptions import SystemError
from app.core.json_parse import extract_json, extract_json_object

logger = logging.getLogger("app.workflows.learning")

__all__ = [
    "LearningWorkflowState",
    "WorkflowGenerationError",
    "build_start_graph",
    "build_submit_graph",
    "prepare_learning_context",
    "generate_quiz",
    "evaluate_and_review",
    "normalize_quiz",
    "normalize_evaluation",
    "public_quiz",
    "public_state",
    "QUESTION_TYPES",
]

# 工作流类型 / 状态：与 workflow_runs.status 列保持一致
WORKFLOW_TYPE_LEARNING = "learning_assessment"
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_QUIZ_READY = "quiz_ready"
STATUS_EVALUATED = "evaluated"
STATUS_FAILED = "failed"

# 前端可渲染的题型：单选 / 多选 / 判断 / 填空 / 简答
QUESTION_TYPES = frozenset(
    {"single_choice", "multiple_choice", "true_false", "fill_blank", "short_answer"}
)
CHOICE_TYPES = frozenset({"single_choice", "multiple_choice"})
MIN_QUESTION_COUNT = 1
MAX_QUESTION_COUNT = 20
DEFAULT_QUESTION_COUNT = 5
MAX_WEAKNESSES = 8

# 模型输出解析失败 / 结构不合法时抛出（服务层据此把 run 标记为 failed）
class WorkflowGenerationError(SystemError):
    """系统异常：模型没有按约定产出可用的结构化结果。"""

    code = "WORKFLOW_GENERATION_FAILED"
    message = "题目生成或评分失败，请稍后重试"


class LearningWorkflowState(TypedDict, total=False):
    """学习测评闭环所需的全部状态（total=False：节点按需增量写入）。

    输入：run_id / user_id / session_id / request_text / provider / question_count
    中间：quiz（含答案，仅服务端）/ answers
    输出：evaluation / score / weaknesses / review
    控制：status
    """

    run_id: str
    user_id: int
    session_id: int
    request_text: str
    provider: str | None
    question_count: int
    quiz: list[dict[str, Any]]
    answers: list[dict[str, Any]]
    evaluation: list[dict[str, Any]]
    score: float | None
    weaknesses: list[str]
    review: str
    status: str


ModelResolver = Callable[[LearningWorkflowState], BaseChatModel]

_QUIZ_SYSTEM_PROMPT = """你是学习测评出题老师。请根据用户的测评需求生成结构化题目。
只输出一个 JSON 对象，不要输出任何解释或 Markdown 代码块。
JSON 结构：
{"questions": [{"question_id": "q1", "type": "single_choice", "stem": "题干", "options": [{"key": "A", "text": "选项"}], "answer": "A"}]}
type 只能是：single_choice（单选）、multiple_choice（多选）、true_false（判断）、fill_blank（填空）、short_answer（简答）。
要求：
1. question_id 从 q1 开始连续编号，数量严格等于用户要求的题目数量；
2. 单选 / 多选必须给 4 个选项；判断可省略 options（前端渲染「正确 / 错误」）；
   填空 / 简答不需要 options；
3. answer 给出标准答案（单选填选项 key，多选填 key 数组，其余填参考答案），供后端阅卷，不要写进题干；
4. 题干明确、可评测，覆盖用户要求的主题。"""

_EVALUATION_SYSTEM_PROMPT = """你是学习测评阅卷老师。请根据题目、标准答案与用户作答，一次性完成评分、逐题点评、薄弱点提取和学习建议。
只输出一个 JSON 对象，不要输出任何解释或 Markdown 代码块。
JSON 结构：
{
  "score": 0 到 100 的数字,
  "evaluation": [{"question_id": "q1", "correct": true, "comment": "逐题点评"}],
  "weaknesses": ["薄弱点1", "薄弱点2"],
  "review": "整体学习建议与下一步练习方向"
}
要求：
1. evaluation 必须覆盖每一道题，question_id 与题目一一对应；
2. 评分按百分制：客观题按对错，主观题按要点覆盖度；
3. comment 要具体，指出错因和正确思路；未作答的题也要点评；
4. weaknesses 精炼到 3-5 条；review 给出可执行的学习建议。"""


# ---------------------------------------------------------------------------
# 节点 1：prepare_learning_context（只做轻量状态标记）
# ---------------------------------------------------------------------------
def prepare_learning_context(state: LearningWorkflowState) -> dict[str, Any]:
    """轻量预处理：归一化入参 + 标记状态，不做任何 LLM / 数据库调用。"""
    request_text = str(state.get("request_text") or "").strip()
    count = _clamp_int(state.get("question_count"), DEFAULT_QUESTION_COUNT, MIN_QUESTION_COUNT, MAX_QUESTION_COUNT)
    provider = (state.get("provider") or "").strip() or None
    logger.info("学习测评准备 run_id=%s 题量=%s provider=%s", state.get("run_id"), count, provider)
    return {
        "request_text": request_text,
        "question_count": count,
        "provider": provider,
        "quiz": [],
        "answers": [],
        "evaluation": [],
        "score": None,
        "weaknesses": [],
        "review": "",
        "status": STATUS_RUNNING,
    }


# ---------------------------------------------------------------------------
# 节点 2：generate_quiz（LLM 生成结构化题目 JSON）
# ---------------------------------------------------------------------------
def generate_quiz(state: LearningWorkflowState, model: BaseChatModel) -> dict[str, Any]:
    """调用 LLM 生成题目，解析并归一化成前端可渲染的结构。"""
    request_text = str(state.get("request_text") or "").strip()
    count = _clamp_int(state.get("question_count"), DEFAULT_QUESTION_COUNT, MIN_QUESTION_COUNT, MAX_QUESTION_COUNT)
    human = f"测评需求：{request_text}\n题目数量：{count}"

    raw = _invoke_text(model, _QUIZ_SYSTEM_PROMPT, human)
    questions = _pick_questions(extract_json(raw))
    quiz = normalize_quiz(questions)
    if not quiz:
        raise WorkflowGenerationError(detail=f"模型未产出有效题目 raw={raw[:200]!r}")
    logger.info("学习测评出题完成 run_id=%s 题量=%s", state.get("run_id"), len(quiz))
    return {"quiz": quiz, "status": STATUS_QUIZ_READY}


# ---------------------------------------------------------------------------
# 节点 3：evaluate_and_review（一次 LLM 完成评分 / 点评 / 薄弱点 / 建议）
# ---------------------------------------------------------------------------
def evaluate_and_review(state: LearningWorkflowState, model: BaseChatModel) -> dict[str, Any]:
    """调用 LLM 一次性完成评分、逐题点评、薄弱点提取与学习建议。"""
    quiz = _as_dict_list(state.get("quiz"))
    answers = _as_dict_list(state.get("answers"))
    if not quiz:
        raise WorkflowGenerationError(detail="state 中没有题目，无法评分")

    human = (
        "题目与标准答案：\n"
        f"{_dump_json(quiz)}\n\n"
        "用户作答（question_id -> value，缺项视为未作答）：\n"
        f"{_dump_json(answers)}"
    )
    raw = _invoke_text(model, _EVALUATION_SYSTEM_PROMPT, human)
    payload = extract_json_object(raw)
    if payload is None:
        raise WorkflowGenerationError(detail=f"模型评分结果不是合法 JSON raw={raw[:200]!r}")

    score, evaluation, weaknesses, review = normalize_evaluation(payload, quiz)
    logger.info(
        "学习测评评分完成 run_id=%s 得分=%s 薄弱点=%s",
        state.get("run_id"),
        score,
        len(weaknesses),
    )
    return {
        "score": score,
        "evaluation": evaluation,
        "weaknesses": weaknesses,
        "review": review,
        "status": STATUS_EVALUATED,
    }


# ---------------------------------------------------------------------------
# 图构建
# ---------------------------------------------------------------------------
def build_start_graph(resolve_model: ModelResolver):
    """启动图：prepare_learning_context -> generate_quiz -> END。"""
    graph = StateGraph(LearningWorkflowState)
    graph.add_node("prepare_learning_context", prepare_learning_context)
    graph.add_node("generate_quiz", lambda state: generate_quiz(state, resolve_model(state)))
    graph.add_edge(START, "prepare_learning_context")
    graph.add_edge("prepare_learning_context", "generate_quiz")
    graph.add_edge("generate_quiz", END)
    return graph.compile()


def build_submit_graph(resolve_model: ModelResolver):
    """提交图：evaluate_and_review -> END。"""
    graph = StateGraph(LearningWorkflowState)
    graph.add_node("evaluate_and_review", lambda state: evaluate_and_review(state, resolve_model(state)))
    graph.add_edge(START, "evaluate_and_review")
    graph.add_edge("evaluate_and_review", END)
    return graph.compile()


# ---------------------------------------------------------------------------
# 归一化 / 解析辅助
# ---------------------------------------------------------------------------
def normalize_quiz(questions: Sequence[Any]) -> list[dict[str, Any]]:
    """把模型给的题目列表归一化成标准结构（非法项丢弃，question_id 兜底补全）。"""
    quiz: list[dict[str, Any]] = []
    for item in questions:
        if not isinstance(item, dict):
            continue
        stem = str(item.get("stem") or item.get("question") or item.get("title") or "").strip()
        if not stem:
            continue
        qtype = str(item.get("type") or "short_answer").strip().lower()
        if qtype not in QUESTION_TYPES:
            qtype = "short_answer"
        # 兜底 id 按「已接受题目数」连续编号，避免中间丢项导致 q1、q3 这种断号
        fallback_id = f"q{len(quiz) + 1}"
        question: dict[str, Any] = {
            "question_id": str(item.get("question_id") or fallback_id).strip() or fallback_id,
            "type": qtype,
            "stem": stem,
        }
        options = _normalize_options(item.get("options"))
        if options and qtype in CHOICE_TYPES:
            question["options"] = options
        answer = item.get("answer")
        if answer not in (None, "", []):
            question["answer"] = answer if isinstance(answer, list) else str(answer)
        quiz.append(question)
    return quiz


def normalize_evaluation(
    payload: dict[str, Any],
    quiz: Sequence[dict[str, Any]],
) -> tuple[float | None, list[dict[str, Any]], list[str], str]:
    """归一化评分结果：分数夹到 0-100、逐题结果按题目补齐、薄弱点去重截断。"""
    score = _to_score(payload.get("score"))
    by_id = {str(item.get("question_id")): item for item in quiz}

    evaluation: list[dict[str, Any]] = []
    covered: set[str] = set()
    raw_evaluation = payload.get("evaluation")
    if isinstance(raw_evaluation, list):
        for item in raw_evaluation:
            if not isinstance(item, dict):
                continue
            question_id = str(item.get("question_id") or "").strip()
            if question_id not in by_id or question_id in covered:
                continue
            covered.add(question_id)
            evaluation.append(_normalize_evaluation_item(question_id, item))
    # 模型漏评的题目补空项，保证前端逐题都能渲染
    for question in quiz:
        question_id = str(question.get("question_id"))
        if question_id not in covered:
            evaluation.append({"question_id": question_id, "correct": None, "comment": ""})

    weaknesses = _to_str_list(payload.get("weaknesses"), limit=MAX_WEAKNESSES)
    review = str(payload.get("review") or payload.get("suggestion") or payload.get("advice") or "").strip()
    return score, evaluation, weaknesses, review


def public_quiz(quiz: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """对外题目：剥掉 answer（答案只留在服务端 state 里，供阅卷）。"""
    public: list[dict[str, Any]] = []
    for item in quiz:
        public.append({key: value for key, value in item.items() if key != "answer"})
    return public


def public_state(state: dict[str, Any]) -> dict[str, Any]:
    """对外状态快照：同样对 quiz 剥答案，避免 GET 接口泄露标准答案。"""
    public = dict(state)
    quiz = public.get("quiz")
    if isinstance(quiz, list):
        public["quiz"] = public_quiz([item for item in quiz if isinstance(item, dict)])
    return public


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------
def _invoke_text(model: BaseChatModel, system: str, human: str) -> str:
    """同步调用模型并拍平成文本（节点在图里同步执行，由服务层放进线程池）。"""
    messages: list[BaseMessage] = [SystemMessage(content=system), HumanMessage(content=human)]
    return _message_text(model.invoke(messages))


def _message_text(message: object) -> str:
    """从模型返回 / 消息对象里拍平出文本（兼容内容块列表）。"""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, (list, tuple)):
        pieces: list[str] = []
        for item in content:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                pieces.append(item["text"])
        return "".join(pieces).strip()
    return str(content).strip()


def _pick_questions(payload: Any) -> list[Any]:
    """题目列表可能藏在 {"questions": [...]} 或 "quiz"/"items" 键下，也兼容直接给数组。"""
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in ("questions", "quiz", "items", "data"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return []


def _normalize_options(options: Any) -> list[dict[str, str]]:
    """选项归一化：支持 {"key","text"} 与纯字符串两种写法。"""
    if not isinstance(options, list):
        return []
    result: list[dict[str, str]] = []
    for index, option in enumerate(options):
        default_key = chr(ord("A") + index)
        if isinstance(option, dict):
            text = str(option.get("text") or option.get("label") or option.get("value") or "").strip()
            key = str(option.get("key") or default_key).strip() or default_key
        elif isinstance(option, str):
            text, key = option.strip(), default_key
        else:
            continue
        if text:
            result.append({"key": key, "text": text})
    return result


def _normalize_evaluation_item(question_id: str, item: dict[str, Any]) -> dict[str, Any]:
    """单题点评归一化：correct 缺失时保持 None（不臆断对错）。"""
    correct = item.get("correct")
    if isinstance(correct, str):
        correct = correct.strip().lower() in {"true", "yes", "1", "正确", "对"}
    elif correct is not None:
        correct = bool(correct)
    return {
        "question_id": question_id,
        "correct": correct,
        "comment": str(item.get("comment") or item.get("feedback") or "").strip(),
    }


def _to_score(value: Any) -> float | None:
    """分数转 float 并夹到 0-100；解析不了返回 None。"""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(100.0, score))


def _to_str_list(value: Any, *, limit: int) -> list[str]:
    """字符串列表去重截断（模型偶尔给字符串 / 混入非字符串）。"""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in result:
            result.append(text)
        if len(result) >= limit:
            break
    return result


def _as_dict_list(value: Any) -> list[dict[str, Any]]:
    """过滤出 dict 列表（state 来自 DB JSON，理论上已是 list[dict]）。"""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _clamp_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def _dump_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)
