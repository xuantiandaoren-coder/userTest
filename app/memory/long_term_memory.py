"""长期记忆：跨会话用户画像的读取、写入意图判断与异步沉淀。

设计目标（与 ``InstantMemory`` / 搜索增强并列的第三种记忆）：

===========  ==========================================================
读取          ``load_memory_context``：读 ``user_profiles``，拼成可注入 system 的画像文本
写入意图      ``should_write``：fastembed + ``BAAI/bge-small-zh-v1.5`` 语义相似度，
              判断这条消息是不是「用户在交代自己的情况」，命中才值得落库
异步沉淀      ``submit_profile_update_async``：后台线程里用大模型从对话上下文抽取结构化
              画像，回写 ``user_profiles`` 对应字段（learning_goal / learning_style /
              interview_focus / long_term_summary）
===========  ==========================================================

三条不变量：

1. **读写分离**：读用请求级会话（``profiles``），写用独立会话工厂（``session_factory``），
   后者天然适合丢进后台线程，不受请求生命周期影响；
2. **绝不阻塞对话**：读失败 / 向量化失败 / 抽取失败都只记日志并降级，不向上抛异常；
3. **依赖可注入**：向量后端（``EmbeddingBackend``）与模型都从外部注入，测试不联网、
   不下载模型也能跑。

配置统一走 ``app.core.config.settings``，本模块不直接读环境变量。
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.json_parse import extract_json_object
from app.db.models import UserProfile
from app.db.user_profile_repository import UserProfileRepository

logger = logging.getLogger("app.memory.long_term")

__all__ = [
    "LongTermMemory",
    "EmbeddingBackend",
    "FastEmbedBackend",
    "PROFILE_INTENT_EXAMPLES",
]

# 默认本地向量模型：中文小模型，fastembed 原生支持，首次使用时下载
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"

# 画像写入意图的语义参照句：与这些句子越像，越说明用户在提供可长期沉淀的信息
PROFILE_INTENT_EXAMPLES: tuple[str, ...] = (
    "我的学习目标是转行做后端开发，半年内拿到 offer",
    "我打算系统学习算法和数据结构，每天坚持刷题",
    "我习惯通过看视频和动手做项目来学习，不太喜欢纯看文档",
    "我更喜欢先看例题再自己推导，喜欢有人给我讲思路",
    "我面试主要准备 Java 后端和分布式相关的问题",
    "我投的是高级工程师岗位，面试重点会放在系统设计上",
    "我的薄弱点是并发编程和 MySQL 索引优化",
    "我已经掌握 Python 和 Flask，但没怎么用过微服务",
    "我有三年工作经验，目前在一家创业公司做全栈",
    "记住我的情况：目标是数据分析岗，擅长 SQL，需要补统计学",
)

# 反向参照句：与这些句子越像，越说明用户只是提问 / 寒暄，不该写画像。
# 正负对照能把「MySQL 索引优化怎么做」这类知识型提问和「我在准备 MySQL 面试」区分开。
NON_PROFILE_EXAMPLES: tuple[str, ...] = (
    "帮我看看这段代码为什么报错",
    "MySQL 索引优化的原理是什么",
    "什么是死锁，怎么避免",
    "继续讲一下这个知识点",
    "谢谢，我明白了",
    "今天天气怎么样",
    "帮我写一段快速排序",
    "这道题的标准答案是什么",
)

# 语义判断兜底：fastembed 不可用（未安装 / 模型未下载）时退化为这些关键词规则
FALLBACK_INTENT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(我的|我)(学习|职业|求职|面试)?(目标|计划|打算|方向)"),
    re.compile(r"我(习惯|喜欢|更愿意|偏好|倾向于)"),
    re.compile(r"我(面试|求职|准备)(主要|重点|侧重|会|要)?"),
    re.compile(r"我(的)?(薄弱|弱项|短板|不足|不擅长|不太会|不会)"),
    re.compile(r"我(已经|已)?(掌握|会|熟悉|用过|用过|精通|擅长)"),
    re.compile(r"我(有|做了|工作了)?\s*\d+\s*年(的)?(经验|工作)"),
    re.compile(r"(记住|记录|备注|帮我记)(一下|住)?"),
)

# 抽取结果的长度上限，避免脏数据 / 模型幻觉把字段撑爆
MAX_GOAL_CHARS = 255
MAX_STYLE_CHARS = 255
MAX_FOCUS_ITEMS = 10
MAX_FOCUS_ITEM_CHARS = 64
MAX_SUMMARY_CHARS = 4000
MAX_CONTEXT_CHARS = 4000

# 抽取返回的键（只接受这四个，别的一律丢弃）
EXTRACT_KEYS = ("learning_goal", "learning_style", "interview_focus", "long_term_summary")

_EXTRACTION_SYSTEM = """你是用户画像抽取器。请阅读对话上下文，抽取与用户长期学习 / 求职相关的稳定信息。
只输出一个 JSON 对象，不要输出任何解释或 Markdown 代码块。
字段说明：
- learning_goal：字符串，用户的学习 / 求职目标；没有新信息则给 null
- learning_style：字符串，用户偏好的学习方式；没有新信息则给 null
- interview_focus：字符串数组，用户面试关注或准备的方向；没有新信息则给 null
- long_term_summary：字符串，结合「已有画像」与「新对话」更新后的长期画像摘要；没有变化则给 null
没有把握的字段一律给 null，不要编造。"""

class EmbeddingBackend(Protocol):
    """向量后端协议：换成别的本地 / 远程向量化实现时只要实现 ``embed``。"""

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class FastEmbedBackend:
    """默认向量后端：fastembed + ``BAAI/bge-small-zh-v1.5``（本地 ONNX，无需联网调用）。

    ``fastembed`` 与模型都按需加载（lazy）：构造本对象不触发下载，
    首次 ``embed`` 才加载；加载后进程内复用，并用锁避免并发重复加载。
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model: Any = None
        self._lock = threading.Lock()

    def _ensure_model(self) -> Any:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from fastembed import TextEmbedding  # 延迟导入：缺失时只影响本功能

                    logger.info("加载长期记忆向量模型 model=%s", self.model_name)
                    self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._ensure_model()
        return [[float(value) for value in vector] for vector in model.embed(list(texts))]


class LongTermMemory:
    """长期记忆服务：读画像 / 判断写入意图 / 异步抽取并回写画像。"""

    def __init__(
        self,
        *,
        profiles: UserProfileRepository | None = None,
        session_factory: Callable[[], Session] | None = None,
        model_factory: Callable[[], BaseChatModel] | None = None,
        embeddings: EmbeddingBackend | None = None,
        similarity_threshold: float | None = None,
        similarity_margin: float | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.profiles = profiles
        self.session_factory = session_factory
        self.model_factory = model_factory
        self.embeddings = embeddings if embeddings is not None else FastEmbedBackend(
            settings.long_term_embedding_model
        )
        self.similarity_threshold = (
            settings.long_term_similarity_threshold if similarity_threshold is None else similarity_threshold
        )
        self.similarity_margin = (
            settings.long_term_similarity_margin if similarity_margin is None else similarity_margin
        )
        self.enabled = settings.long_term_memory_enabled if enabled is None else enabled
        self._example_vectors: list[list[float]] | None = None
        self._negative_vectors: list[list[float]] | None = None
        # 向量后端是否可用：一次失败后置 False，避免每条消息都重试 / 刷警告
        self._embedding_available: bool | None = None

    # ------------------------------------------------------------------
    # 1. 读取：画像 -> 可注入 system 的文本
    # ------------------------------------------------------------------
    def load_memory_context(self, user_id: int, *, max_chars: int = 1200) -> str:
        """读用户画像并排版成记忆文本；没有可注入的信息返回空字符串。"""
        if not self.enabled:
            return ""
        try:
            profile = self._load_profile(user_id)
        except Exception as exc:  # noqa: BLE001 - 读画像失败不能让聊天整体不可用
            logger.warning("读取长期画像失败 user_id=%s：%r", user_id, exc)
            return ""
        if profile is None:
            return ""
        text = format_profile_memory(profile)
        if not text:
            return ""
        return text[:max_chars]

    def _load_profile(self, user_id: int) -> UserProfile | None:
        if self.profiles is None:
            return None
        return self.profiles.get_by_user(user_id)

    # ------------------------------------------------------------------
    # 2. 写入意图：语义相似度判断
    # ------------------------------------------------------------------
    def should_write(self, text: str, user_id: int) -> bool:
        """判断这条消息是否值得写入长期画像。

        fastembed 可用时用 ``BAAI/bge-small-zh-v1.5`` 把消息与画像例句一起向量化，
        采用正负对照：既要与「画像句」足够像，又要比「非画像句」明显更像，
        避免「MySQL 索引优化怎么做」这类知识型提问被误判；向量化不可用时退化为关键词规则。
        """
        if not self.enabled:
            return False
        normalized = (text or "").strip()
        if len(normalized) < 2:
            return False

        scores = self._intent_scores(normalized)
        if scores is None:
            hit = any(pattern.search(normalized) for pattern in FALLBACK_INTENT_PATTERNS)
            logger.debug("长期记忆意图判断（关键词兜底）user_id=%s hit=%s", user_id, hit)
            return hit
        positive, negative = scores
        hit = positive >= self.similarity_threshold and (positive - negative) >= self.similarity_margin
        logger.debug(
            "长期记忆意图判断 user_id=%s positive=%.3f negative=%.3f hit=%s",
            user_id,
            positive,
            negative,
            hit,
        )
        return hit

    def _intent_scores(self, text: str) -> tuple[float, float] | None:
        """返回 (与画像句的最大相似度, 与非画像句的最大相似度)；向量化失败返回 None。"""
        if self._embedding_available is False:
            return None
        try:
            if self._example_vectors is None:
                self._example_vectors = [list(vector) for vector in self.embeddings.embed(PROFILE_INTENT_EXAMPLES)]
                self._negative_vectors = [list(vector) for vector in self.embeddings.embed(NON_PROFILE_EXAMPLES)]
            query_vector = self.embeddings.embed([text])[0]
        except Exception as exc:  # noqa: BLE001 - 向量化是增强能力，失败退化为关键词
            self._embedding_available = False
            logger.warning("画像写入意图向量化失败，退化为关键词规则（后续不再重试）：%r", exc)
            return None
        self._embedding_available = True
        if not self._example_vectors:
            return None
        positive = max(_cosine(query_vector, example) for example in self._example_vectors)
        negative = max((_cosine(query_vector, example) for example in self._negative_vectors), default=0.0)
        return positive, negative

    # ------------------------------------------------------------------
    # 3. 写入：后台线程抽取 + 回写
    # ------------------------------------------------------------------
    def submit_profile_update_async(
        self,
        user_id: int,
        request_text: str,
        context: str | Sequence[BaseMessage],
    ) -> threading.Thread | None:
        """在后台线程里抽取画像并回写；未配置会话 / 模型时安全跳过。"""
        if not self.enabled:
            return None
        if self.session_factory is None or self.model_factory is None:
            logger.warning("长期记忆未配置会话工厂或模型，跳过画像更新 user_id=%s", user_id)
            return None
        thread = threading.Thread(
            target=self._update_profile_sync,
            args=(user_id, request_text, context),
            name=f"long-term-memory-{user_id}",
            daemon=True,
        )
        thread.start()
        logger.info("长期记忆画像更新已提交后台线程 user_id=%s", user_id)
        return thread

    def _update_profile_sync(
        self,
        user_id: int,
        request_text: str,
        context: str | Sequence[BaseMessage],
    ) -> None:
        """线程体：抽取 -> 合并 -> 回写（独立会话，全过程只记日志不抛异常）。"""
        try:
            session = self.session_factory()  # type: ignore[misc]
            try:
                repository = UserProfileRepository(session)
                current = repository.get_by_user(user_id)
                extracted = self._extract_profile(user_id, request_text, context, current)
                if not extracted:
                    logger.info("长期记忆未抽取到新画像 user_id=%s", user_id)
                    return
                repository.upsert(user_id, **extracted)
                session.commit()
                changed = _changed_fields(current, extracted)
                logger.info(
                    "长期记忆写入成功 user_id=%s fields=%s changed=%s",
                    user_id,
                    sorted(extracted),
                    changed,
                )
                logger.debug("长期记忆写入内容 user_id=%s detail=%s", user_id, _field_detail(extracted))
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()
        except Exception as exc:  # noqa: BLE001 - 后台任务失败不能影响主流程
            logger.error("长期记忆画像更新失败 user_id=%s：%r", user_id, exc, exc_info=exc)

    def _extract_profile(
        self,
        user_id: int,
        request_text: str,
        context: str | Sequence[BaseMessage],
        current: UserProfile | None,
    ) -> dict[str, Any]:
        """调用大模型抽取结构化画像，返回可直接 upsert 的字段字典。"""
        if self.model_factory is None:
            return {}
        model = self.model_factory()
        human = _build_extraction_input(request_text, context, current)
        message = model.invoke([SystemMessage(content=_EXTRACTION_SYSTEM), HumanMessage(content=human)])
        raw = _message_text(message)
        data = _parse_json_object(raw)
        if not data:
            logger.debug("长期记忆抽取结果无法解析 user_id=%s raw=%r", user_id, raw[:200])
            return {}
        return _sanitize_extracted(data)


# ---------------------------------------------------------------------------
# 纯函数
# ---------------------------------------------------------------------------
def format_profile_memory(profile: UserProfile) -> str:
    """把画像行排版成可读文本（只输出有值的行，便于控制注入长度）。"""
    lines: list[str] = []
    if profile.target_job:
        lines.append(f"目标岗位：{profile.target_job}")
    if profile.target_level:
        lines.append(f"目标等级：{profile.target_level}")
    if profile.learning_goal:
        lines.append(f"学习目标：{profile.learning_goal}")
    if profile.learning_style:
        lines.append(f"学习风格：{profile.learning_style}")
    if profile.target_skills:
        lines.append(f"已掌握技能：{'、'.join(str(item) for item in profile.target_skills)}")
    if profile.weak_topics:
        lines.append(f"薄弱点：{'、'.join(str(item) for item in profile.weak_topics)}")
    if profile.interview_focus:
        lines.append(f"面试关注点：{'、'.join(str(item) for item in profile.interview_focus)}")
    if profile.long_term_summary:
        lines.append(f"画像摘要：{profile.long_term_summary}")
    if not lines:
        return ""
    return "【用户长期画像】\n" + "\n".join(lines)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """余弦相似度（纯 Python 实现，零向量返回 0，不引入 numpy 依赖）。"""
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _build_extraction_input(
    request_text: str,
    context: str | Sequence[BaseMessage],
    current: UserProfile | None,
) -> str:
    """拼抽取输入：本轮问题 + 对话上下文 + 已有画像。"""
    sections = [f"【本轮用户消息】\n{(request_text or '').strip()}"]
    rendered = render_context(context)
    if rendered:
        sections.append(f"【对话上下文】\n{rendered}")
    if current is not None:
        existing = {
            "learning_goal": current.learning_goal,
            "learning_style": current.learning_style,
            "interview_focus": current.interview_focus,
            "long_term_summary": current.long_term_summary,
        }
        if any(value for value in existing.values()):
            sections.append("【已有画像】\n" + json.dumps(existing, ensure_ascii=False))
    return "\n\n".join(sections)[:MAX_CONTEXT_CHARS]


def render_context(context: str | Sequence[BaseMessage]) -> str:
    """对话上下文 -> 纯文本（支持直接传字符串或 LangChain 消息序列）。"""
    if isinstance(context, str):
        return context.strip()
    lines: list[str] = []
    for message in context or []:
        content = _message_text(message)
        if content:
            lines.append(f"{_role_of(message)}：{content}")
    return "\n".join(lines)


def _role_of(message: object) -> str:
    """消息 -> 中文角色名。"""
    name = type(message).__name__
    if name.startswith("Human"):
        return "用户"
    if name.startswith("AI"):
        return "助手"
    if name.startswith("System"):
        return "系统"
    return "对话"


def _message_text(message: object) -> str:
    """从模型返回 / LangChain 消息里拍平出文本。"""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, Sequence):
        pieces: list[str] = []
        for item in content:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                pieces.append(item["text"])
        return "".join(pieces).strip()
    return str(content).strip()


def _parse_json_object(raw: str) -> dict[str, Any] | None:
    """从模型输出里抠出 JSON 对象（复用 app/core/json_parse 的容错提取）。"""
    return extract_json_object(raw)


def _sanitize_extracted(data: dict[str, Any]) -> dict[str, Any]:
    """只保留四个允许的键，做类型 / 长度清洗，空值不写入。"""
    result: dict[str, Any] = {}

    for key, limit in (("learning_goal", MAX_GOAL_CHARS), ("learning_style", MAX_STYLE_CHARS)):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            result[key] = value.strip()[:limit]

    focus = data.get("interview_focus")
    if isinstance(focus, str):
        focus = [focus]
    if isinstance(focus, Sequence) and not isinstance(focus, (bytes, bytearray)):
        items = [str(item).strip()[:MAX_FOCUS_ITEM_CHARS] for item in focus if str(item).strip()]
        if items:
            result["interview_focus"] = items[:MAX_FOCUS_ITEMS]

    summary = data.get("long_term_summary")
    if isinstance(summary, str) and summary.strip():
        result["long_term_summary"] = summary.strip()[:MAX_SUMMARY_CHARS]

    return result


def _changed_fields(current: UserProfile | None, extracted: dict[str, Any]) -> list[str]:
    """相对已有画像真正发生变化的字段名（便于从日志看出这次写入有没有实质更新）。"""
    if current is None:
        return sorted(extracted)
    return sorted(
        key for key, value in extracted.items() if getattr(current, key, None) != value
    )


def _field_detail(fields: dict[str, Any], *, limit: int = 60) -> str:
    """把写入内容整理成一行日志（值截断，避免长摘要刷屏）。"""
    parts: list[str] = []
    for key in EXTRACT_KEYS:
        if key not in fields:
            continue
        value = fields[key]
        text = "、".join(str(item) for item in value) if isinstance(value, (list, tuple)) else str(value)
        parts.append(f"{key}={text[:limit]}")
    return "; ".join(parts)
