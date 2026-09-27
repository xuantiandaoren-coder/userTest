"""把历史头像（本地磁盘上的文件）补传到 SeaweedFS。

改造后原文件统一存 SeaweedFS，但早期上传的头像还留在 `STORAGE_ROOT/<user_id>/` 下，
DB 里的 `user.avatar` 键在对象存储里并不存在，所以 `avatar_url` 会 404。
本脚本扫描 `user.avatar`，把本地文件按**原键**补传上去，DB 无需改动。

用法：
    uv run python scripts/migrate_local_avatars.py           # 只报告，不改动（默认）
    uv run python scripts/migrate_local_avatars.py --apply   # 真正补传
"""

from __future__ import annotations

import argparse
import mimetypes
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.seaweedfs import SeaweedFSClient, get_seaweedfs_client
from app.db.models import User


def migrate(session: Session, storage: SeaweedFSClient, root: Path, *, apply: bool) -> list[str]:
    """补传缺失的头像对象，返回处理结果说明（每行一条）。"""
    lines: list[str] = []
    users = session.scalars(select(User).where(User.avatar.is_not(None)).order_by(User.id)).all()

    for user in users:
        key = user.avatar or ""
        if storage.object_exists(key):
            lines.append(f"skip   user={user.id} {key}（对象已存在）")
            continue

        local_file = root / key
        if not local_file.is_file():
            lines.append(f"miss   user={user.id} {key}（对象和本地文件都没有，需重新上传）")
            continue

        if apply:
            content_type = mimetypes.guess_type(key)[0] or "application/octet-stream"
            storage.put_object(key, local_file.read_bytes(), content_type)
            lines.append(f"done   user={user.id} {key}（已补传 {local_file.stat().st_size} bytes）")
        else:
            lines.append(f"todo   user={user.id} {key}（本地 {local_file} 待补传）")

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="把本地历史头像补传到 SeaweedFS")
    parser.add_argument("--apply", action="store_true", help="真正执行补传（默认只报告）")
    args = parser.parse_args()

    engine = create_engine(settings.sqlalchemy_url, connect_args={"connect_timeout": settings.db_connect_timeout})
    try:
        with Session(engine) as session:
            lines = migrate(session, get_seaweedfs_client(), settings.storage_root, apply=args.apply)
    finally:
        engine.dispose()

    for line in lines:
        print(line)
    print(f"\n共 {len(lines)} 个头像；{'已执行补传' if args.apply else '未做改动（加 --apply 执行）'}")


if __name__ == "__main__":
    main()
