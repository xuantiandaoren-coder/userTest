"""提示词变量注入：验证层 -> 转换层 -> 填充层（三层各管一件事）。

    原始值（user_profiles 行 / 调用方入参）
        │  ① 验证层 validate_variables：白名单 + 类型 + 长度 + 控制字符 + 注入特征
        ▼
    已校验的原始值
        │  ② 转换层 transform_variables：数组 -> 顿号文本、数字 -> "3 年"、空值 -> 默认值
        ▼
    {变量名: 文本}
        │  ③ 填充层 fill_template：单次正则替换 {target_job} -> 实际值
        ▼
    最终提示词

安全设计的两个关键点：
1. **白名单**：只接受 ALLOWED_VARIABLES 里的变量名，模板作者改不动的东西用户也注入不了；
2. **单次替换**：`re.sub` 一遍扫完，替换进来的文本不会被二次解析，
   因此用户填 `{weak_topics}` 这类字面量不会造成“二次占位符注入”。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Mapping, Sequence

from app.core.exceptions import BusinessError
from app.db.models import UserProfile

# 占位符语法：{变量名}，变量名限定为标识符，模板里写 JSON 示例（如 {"a": 1}）不会被误替换
PLACEHOLDER_PATTERN = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")

# 去掉除换行 / 制表符外的控制字符（防提示词走私与日志注入）
CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# 常见注入特征：命中即拒绝该字段（宁可 422 让用户改词，也不把控制指令送进模型）
INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("忽略指令", re.compile(r"忽略(以上|上面|之前|前面|上述|前面所有|所有)(的)?(指令|提示|规则|设定)")),
    ("指令覆盖", re.compile(r"(ignore|disregard|forget)\s+(all\s+)?(the\s+)?(previous|above|prior|earlier)\s+(instruction|prompt|rule|message)", re.I)),
    ("角色越权", re.compile(r"(你现在是|从现在开始你是|you are now|act as)\s*(一个新|a new)?\s*(system|系统)")),
    ("泄露系统提示", re.compile(r"(system\s*prompt|系统提示词?|开发者指令).{0,10}(输出|打印|告诉我|repeat|print|show)", re.I)),
    ("分隔符走私", re.compile(r"(<\|im_start\|>|<\|im_end\|>|\[/?INST\]|```\s*system)")),
)

# 允许注入的变量白名单（模板变量必须落在集合内，其余一律丢弃并记入 warnings）
ALLOWED_VARIABLES: frozenset[str] = frozenset(
    {
        "user_name",
        "target_job",
        "years_experience",
        "target_level",
        "target_skills",
        "weak_topics",
        "agent_name",
        "agent_label",
        "scene",
        "current_date",
    }
)

# 「没填」时的占位文本：让模型知道信息缺失，而不是接到空字符串后自由发挥
DEFAULT_TEXT = "未填写"
PROFILE_DEFAULTS: dict[str, str] = {
    "user_name": "同学",
    "target_job": DEFAULT_TEXT,
    "years_experience": DEFAULT_TEXT,
    "target_level": DEFAULT_TEXT,
    "target_skills": DEFAULT_TEXT,
    "weak_topics": DEFAULT_TEXT,
}

MAX_VALUE_CHARS = 500      # 单变量长度上限
MAX_TOTAL_CHARS = 4000     # 所有变量长度之和上限（防止把用户信息塞成一篇长文）
MAX_LIST_ITEMS = 20        # 数组类变量最多取几项
MAX_PLACEHOLDER_COUNT = 60  # 模板占位符数量上限（防模板异常膨胀）


class PromptVariableError(BusinessError):
    """业务异常：变量非法（含注入特征 / 超长 / 类型不支持），提示词拒绝渲染。"""

    code = "PROMPT_VARIABLE_INVALID"
    http_status = 422
    message = "提示词变量不合法"


@dataclass
class InjectionResult:
    """注入结果：渲染好的提示词 + 实际注入的变量 + 诊断信息。"""

    text: str
    variables: dict[str, str]
    unfilled: list[str] = field(default_factory=list)   # 模板里有、但没给值的占位符
    warnings: list[str] = field(default_factory=list)   # 被丢弃的非法变量等


# --------------------------------------------------------------------------
# ① 验证层
# --------------------------------------------------------------------------
def validate_variables(raw: Mapping[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
    """白名单 + 类型 + 长度 + 控制字符 + 注入特征校验，返回 (合法变量, 告警)。"""
    if not raw:
        return {}, []

    accepted: dict[str, Any] = {}
    warnings: list[str] = []
    total = 0

    for key, value in raw.items():
        name = str(key).strip()
        if name not in ALLOWED_VARIABLES:
            warnings.append(f"变量 {name!r} 不在白名单内，已忽略")
            continue
        coerced = _coerce(name, value)
        if coerced is None:
            continue
        if isinstance(coerced, str):
            _reject_injection(name, coerced)
            if len(coerced) > MAX_VALUE_CHARS:
                raise PromptVariableError(detail=f"变量 {name} 长度 {len(coerced)} 超过上限 {MAX_VALUE_CHARS}")
            total += len(coerced)
        accepted[name] = coerced

    if total > MAX_TOTAL_CHARS:
        raise PromptVariableError(detail=f"变量总长度 {total} 超过上限 {MAX_TOTAL_CHARS}")
    return accepted, warnings


def _coerce(name: str, value: Any) -> Any:
    """基础类型校验：只放行 str / 数字 / 布尔 / 字符串数组，其余丢弃并告警。

    （告警由调用方汇总，这里用异常表达「明确错误」，用 None 表达「静默丢弃」。）
    """
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = CONTROL_CHAR_PATTERN.sub("", value).strip()
        return cleaned or None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        items = [CONTROL_CHAR_PATTERN.sub("", str(item)).strip() for item in value][:MAX_LIST_ITEMS]
        return [item for item in items if item]
    raise PromptVariableError(detail=f"变量 {name} 类型 {type(value).__name__} 不支持")


def _reject_injection(name: str, value: str) -> None:
    """注入特征检测：命中即拒绝，并在 detail 里带上命中的规则名便于定位。"""
    for label, pattern in INJECTION_PATTERNS:
        if pattern.search(value):
            raise PromptVariableError(detail=f"变量 {name} 命中注入特征「{label}」，请修改后重试")


# --------------------------------------------------------------------------
# ② 转换层
# --------------------------------------------------------------------------
def transform_variables(validated: Mapping[str, Any]) -> dict[str, str]:
    """把已校验的值转成可直接填进模板的文本（数组 -> 顿号分隔，空值 -> 默认值）。"""
    values: dict[str, str] = {}
    for name, value in validated.items():
        text = _to_text(name, value)
        if text:
            values[name] = text[:MAX_VALUE_CHARS]
    # 补默认值：没给 / 给了空值的变量，用「未填写」占位，避免模型面对空字符串瞎猜
    for name, default in PROFILE_DEFAULTS.items():
        values.setdefault(name, default)
    return values


def _to_text(name: str, value: Any) -> str:
    """单值转换规则。"""
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (list, tuple)):
        return "、".join(str(item) for item in value)
    if name == "years_experience" and isinstance(value, (int, float)):
        return f"{int(value)} 年"
    return str(value).strip()


# --------------------------------------------------------------------------
# ③ 填充层
# --------------------------------------------------------------------------
def fill_template(template: str, values: Mapping[str, str]) -> tuple[str, list[str]]:
    """单次正则替换占位符，返回 (渲染结果, 未填的占位符名)。

    单次替换（而不是循环 replace）保证替换进来的文本不会再被解析，
    从根上避免「用户输入里带 {xxx} 造成二次注入」。
    """
    unfilled: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in values:
            return values[name]
        unfilled.append(name)
        return match.group(0)  # 未知占位符原样保留，便于模板作者发现拼写错误

    return PLACEHOLDER_PATTERN.sub(_replace, _check_template(template)), unfilled


def _check_template(template: str) -> str:
    """模板自身的基本保护：非空 + 占位符数量上限。"""
    if not template or not template.strip():
        raise PromptVariableError(detail="模板内容为空，无法渲染")
    if len(PLACEHOLDER_PATTERN.findall(template)) > MAX_PLACEHOLDER_COUNT:
        raise PromptVariableError(detail=f"模板占位符超过 {MAX_PLACEHOLDER_COUNT} 个，疑似模板异常")
    return template


def render(template: str, raw: Mapping[str, Any] | None) -> InjectionResult:
    """三层处理入口：验证 -> 转换 -> 填充。"""
    validated, warnings = validate_variables(raw)
    values = transform_variables(validated)
    text, unfilled = fill_template(template, values)
    if unfilled:
        warnings.append(f"模板占位符未赋值：{','.join(sorted(set(unfilled)))}")
    return InjectionResult(text=text, variables=values, unfilled=sorted(set(unfilled)), warnings=warnings)


def profile_to_variables(
    profile: UserProfile | None,
    *,
    user_name: str | None = None,
    agent_name: str | None = None,
    agent_label: str | None = None,
    scene: str | None = None,
    today: date | None = None,
) -> dict[str, Any]:
    """把 user_profiles 行 + 运行时信息拍平成原始变量字典（注入前不做任何格式化）。"""
    variables: dict[str, Any] = {
        "user_name": user_name,
        "agent_name": agent_name,
        "agent_label": agent_label or agent_name,
        "scene": scene,
        "current_date": (today or date.today()).isoformat(),
    }
    if profile is not None:
        variables.update(
            {
                "target_job": profile.target_job,
                "years_experience": profile.years_experience,
                "target_level": profile.target_level,
                "target_skills": profile.target_skills,
                "weak_topics": profile.weak_topics,
            }
        )
    return variables


def collect_variables(*groups: Sequence[str] | None) -> list[str]:
    """合并模板声明的变量名（去重、保持首次出现顺序），用于响应里展示。"""
    merged: list[str] = []
    for group in groups:
        for name in group or []:
            if name not in merged:
                merged.append(name)
    return merged
