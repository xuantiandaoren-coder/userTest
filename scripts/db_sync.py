"""一键同步表结构：根据 ORM 模型自动生成迁移脚本并升级数据库。

用法：
    uv run python scripts/db_sync.py -m "新增 xxx 字段"

等价于依次执行：
    alembic upgrade head                      # 先让数据库追上已有迁移
    alembic revision --autogenerate -m "..."  # 再按模型差异生成新迁移
    alembic upgrade head                      # 最后应用新迁移

若模型相对数据库没有结构变更，则不会留下空的迁移文件。
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALEMBIC_INI = PROJECT_ROOT / "alembic.ini"


def _run_alembic(*args: str) -> None:
    """以当前解释器调用 alembic，避免依赖 PATH 中的同名命令行。"""
    subprocess.run([sys.executable, "-m", "alembic", *args], cwd=PROJECT_ROOT, check=True)


def _latest_revision_file() -> Path:
    """返回当前 head 对应的迁移脚本路径。"""
    config = Config(str(ALEMBIC_INI))
    config.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    script = ScriptDirectory.from_config(config)
    head = script.get_revision(script.get_current_head())
    return Path(head.path)


_DOCSTRING_PATTERN = re.compile(r'"""(?:.|\n)*?"""')


def _is_empty_migration(path: Path) -> bool:
    """判断迁移脚本的 upgrade() 是否为空（即模型没有结构变更）。"""
    upgrade_body = path.read_text(encoding="utf-8").split("def upgrade() -> None:", 1)[1]
    upgrade_body = upgrade_body.split("def downgrade()", 1)[0]
    upgrade_body = _DOCSTRING_PATTERN.sub("", upgrade_body)  # 去掉模板自带的 docstring
    statements = [
        line.strip()
        for line in upgrade_body.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    return not statements or statements == ["pass"]


def main() -> int:
    parser = argparse.ArgumentParser(description="自动生成迁移并同步表结构到数据库")
    parser.add_argument("-m", "--message", default="auto sync schema", help="迁移说明，会写进迁移脚本文件名")
    args = parser.parse_args()

    # 先应用已有迁移，避免在空库上把“建表”重复生成为新迁移
    _run_alembic("upgrade", "head")

    _run_alembic("revision", "--autogenerate", "-m", args.message)

    revision_file = _latest_revision_file()
    if _is_empty_migration(revision_file):
        revision_file.unlink()
        print("模型与数据库结构一致，未生成迁移，数据库无需变更。")
        return 0

    print(f"已生成迁移脚本：{revision_file.relative_to(PROJECT_ROOT)}")
    _run_alembic("upgrade", "head")
    print("表结构已同步到数据库。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
