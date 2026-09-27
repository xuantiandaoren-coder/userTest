"""插入提示词模板种子数据（可重复执行，已存在生效模板的分组会被跳过）。

流式聊天（`POST /sessions/{session_id}/stream-chat`）要求目标 `(agent_name, scene)`
至少有一条生效模板，否则接口返回 404 `PROMPT_TEMPLATE_NOT_FOUND`；空库部署后跑本脚本
即可拿到一套能直接运行的初始模板（每个场景一条公共模板 + 每个智能体一条私有模板），
也可以在前端模板管理页维护。

用法：
    uv run python scripts/seed_prompt_templates.py            # 只报告要插入什么（默认）
    uv run python scripts/seed_prompt_templates.py --apply    # 真正写入数据库与 Redis 缓存

写入走的是正式链路 `PromptTemplateManager.create_version()`：自动算版本号、置为生效、
下线同组旧版本、刷新 Redis 缓存（key=`PROMPT_CACHE_KEY`，field=`{agent}:{scene}`）。
"""

from __future__ import annotations

import argparse

from sqlalchemy.orm import Session

from app.core.config import AGENT_CONFIG, DEFAULT_AGENT, DEFAULT_SCENE
from app.core.redis_client import get_redis_client
from app.db.models import UserProfile
from app.db.prompt_template_repository import PromptTemplateRepository
from app.db.session import get_session_factory
from app.prompts.injector import profile_to_variables
from app.prompts.prompt_layer import build_system_prompt
from app.prompts.prompt_template_manager import COMMON_AGENT_KEY, PromptTemplateManager

# 公共模板（agent_name=NULL, template_type=2）：所有智能体按 scene 复用
COMMON_TMPL = """你是一位耐心、严谨的 AI 学习与求职助手，当前对话场景：{scene}。
今天是 {current_date}。
回答用中文，先给结论再给依据；信息不足时先说明缺什么，不要编造事实或引用不存在的资料。"""

# 学员画像：所有私有模板共用的开头（变量由 injector 注入）
PROFILE_BLOCK = """当前学员信息（缺失项显示为「未填写」，不要臆造）：
- 目标岗位：{target_job}
- 工作经验：{years_experience}
- 目标等级：{target_level}
- 已掌握技能：{target_skills}
- 薄弱点：{weak_topics}
"""

# 私有模板（agent_name=<智能体>, template_type=1）：学员画像 + 该智能体的专属职责
PRIVATE_TMPL: dict[str, str] = {
    "flow_controller": PROFILE_BLOCK + """
职责：流程总控。
1. 先判断学员当前处于哪个阶段（简历 -> 出题 -> 面试 -> 评测 -> 难点学习 -> 笔记）；
2. 每次只推进一个阶段，明确说出下一步交给哪个智能体、需要学员补什么材料；
3. 目标不清时，先用一个问题补齐目标岗位与目标等级再动手。""",
    "resume_parser": PROFILE_BLOCK + """
职责：资料 & 简历解析。
1. 从简历 / 上传资料中抽取教育背景、工作经历、项目、技能与量化结果；
2. 缺字段一律标「未填写」，不要替学员编造经历或数字；
3. 用列表或表格输出结构化结果，并指出与目标岗位最不匹配的三处。""",
    "quiz_generate_workflow": PROFILE_BLOCK + """
职责：出题题库。
1. 围绕目标岗位与薄弱点出题，一次 1~3 题，难度从易到难；
2. 学员作答后先判对错、再讲错因，最后给同类变式题；
3. 题目与解析按「题干 / 选项 / 答案 / 解析」结构输出，便于入库。""",
    "interview_host": PROFILE_BLOCK + """
职责：面试主考官。
1. 按目标岗位与目标等级逐轮追问，一次只问一个问题；
2. 学员回答后先简短追问细节，不要替学员作答；
3. 保持面试节奏，记录每轮问答要点，面试结束后交给评测环节。""",
    "interview_evaluator": PROFILE_BLOCK + """
职责：面试评测。
1. 针对面试问答逐条打分（表达 / 技术深度 / 岗位匹配），并给出依据；
2. 指出三个最关键的改进点，每条附一个可立即执行的练习建议；
3. 结论先行，评分用表格，避免笼统的「很好 / 一般」。""",
    "difficulty_learner": PROFILE_BLOCK + """
职责：难点学习。
1. 围绕薄弱点组织回答，一次只讲透一个知识点，先补强最薄弱的；
2. 讲完概念后配一个可执行的小练习或下一步行动；
3. 需要对比时用列表或表格，控制单次回答长度。""",
    "note_archiver": PROFILE_BLOCK + """
职责：笔记归档。
1. 把会话内容提炼成结构化笔记：结论、关键知识点、待办与疑问；
2. 保留关键代码 / 公式 / 面试原题，删掉寒暄与重复表述；
3. 给每条笔记打上主题标签，便于之后按标签检索。""",
}

# 场景 -> 中文名：公共模板按场景各存一版，私有模板按智能体各存一版
SCENE_LABELS: dict[str, str] = {
    "workflow": "流程总控",
    "resume": "资料 & 简历解析",
    "quiz": "出题题库",
    "interview": "面试主考官",
    "evaluation": "面试评测",
    "study": "难点学习",
    "note": "笔记归档",
}

SEEDS: tuple[dict[str, object], ...] = (
    # 历史兼容：老 (tutor, chat) 分组，老调用方仍按 tutor/chat 请求时可命中
    {
        "agent_name": None,
        "scene": "chat",
        "description": "公共模板：通用交流约定（历史兼容，种子）",
        "template_content": COMMON_TMPL,
    },
    {
        "agent_name": "tutor",
        "scene": "chat",
        "description": "tutor 私有模板：学员画像与辅导要求（历史兼容，种子）",
        "template_content": PRIVATE_TMPL["difficulty_learner"],
    },
    # 公共模板：每个场景一版（agent_name=NULL, template_type=2）
    *(
        {
            "agent_name": None,
            "scene": scene,
            "description": f"公共模板：{label}场景通用约定（种子）",
            "template_content": COMMON_TMPL,
        }
        for scene, label in SCENE_LABELS.items()
    ),
    # 私有模板：AGENT_CONFIG 里每个智能体一版（直接按配置生成，改配置即同步）
    *(
        {
            "agent_name": name,
            "scene": setting.scene,
            "description": f"{name} 私有模板：{setting.label}（种子）",
            "template_content": PRIVATE_TMPL[name],
        }
        for name, setting in AGENT_CONFIG.items()
    ),
)

# 预览用的示例画像（不落库，只为了让操作者看到渲染后的样子）
SAMPLE_PROFILE = UserProfile(
    user_id=0,
    target_job="后端工程师",
    years_experience=3,
    target_level="P6",
    target_skills=["Python", "MySQL"],
    weak_topics=["并发", "索引"],
)


def seed(session: Session, manager: PromptTemplateManager, *, apply: bool) -> list[str]:
    """按 SEEDS 插入模板，返回每行处理结果；已生效的分组跳过（可重复执行）。"""
    lines: list[str] = []
    for seed_item in SEEDS:
        agent_name = seed_item["agent_name"]  # type: ignore[assignment]
        scene = str(seed_item["scene"])
        label = agent_name or COMMON_AGENT_KEY

        active = manager.get_template(agent_name, scene)  # type: ignore[arg-type]
        if active is not None:
            lines.append(f"skip  {label}:{scene} 已有生效模板 v{active.version}（template_id={active.template_id}）")
            continue

        if not apply:
            lines.append(f"plan  {label}:{scene} 将新增 v1（{seed_item['description']}）")
            continue

        view = manager.create_version(
            agent_name=agent_name,  # type: ignore[arg-type]
            scene=scene,
            template_content=str(seed_item["template_content"]),
            description=str(seed_item["description"]),
        )
        session.commit()
        lines.append(
            f"done  {label}:{scene} 已新增 v{view.version}（template_id={view.template_id}，"
            f"变量={view.variables}）"
        )
    return lines


def preview(manager: PromptTemplateManager) -> str:
    """把默认智能体 (flow_controller:workflow) 的最终 system 提示词渲染出来，便于人工确认。"""
    composed = manager.compose(DEFAULT_AGENT, DEFAULT_SCENE)
    variables = profile_to_variables(
        SAMPLE_PROFILE,
        user_name="小明",
        agent_name=DEFAULT_AGENT,
        agent_label=AGENT_CONFIG[DEFAULT_AGENT].label,
        scene=DEFAULT_SCENE,
    )
    return build_system_prompt(composed, variables).text


def main() -> int:
    parser = argparse.ArgumentParser(description="插入提示词模板种子数据（默认只报告，--apply 才写入）")
    parser.add_argument("--apply", action="store_true", help="真正写入数据库并刷新 Redis 缓存")
    args = parser.parse_args()

    session = get_session_factory()()
    redis = get_redis_client()
    try:
        manager = PromptTemplateManager(PromptTemplateRepository(session), redis)
        for line in seed(session, manager, apply=args.apply):
            print(line)

        if not args.apply:
            print("\n以上为预演结果，未改动任何数据；确认无误后加 --apply 执行。")
            return 0

        print(f"\n缓存：key={manager.cache_key}，Redis={'可用' if redis else '未启用'}")
        if redis:
            fields = [f"{COMMON_AGENT_KEY}:{scene}" for scene in SCENE_LABELS]
            fields += [f"{name}:{setting.scene}" for name, setting in AGENT_CONFIG.items()]
            for field in fields:
                print(f"  {field} -> {'已写入' if redis.hget(manager.cache_key, field) else '缺失'}")

        print(f"\n=== {DEFAULT_AGENT}:{DEFAULT_SCENE} 渲染预览（示例画像）===")
        print(preview(manager))
        return 0
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
