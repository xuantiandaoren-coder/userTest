"""教学保留的本地落盘实现（app/services/upload_service.py）仍可用，但不在运行链路上。"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import pytest
from sqlalchemy.orm import Session
from starlette.datastructures import Headers, UploadFile

from app.db.models import User
from app.db.user_repository import UserRepository
from app.services.upload_service import FileService

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
)


@pytest.mark.anyio
async def test_legacy_local_upload_writes_file_and_avatar(
    db_session: Session, storage_root: Path
) -> None:
    user = User(user_name="legacy", password="hash")
    db_session.add(user)
    db_session.flush()

    upload = UploadFile(
        file=io.BytesIO(PNG_BYTES),
        filename="old-avatar.png",
        headers=Headers({"content-type": "image/png"}),
    )
    result = await FileService(UserRepository(db_session)).upload(user, upload)

    assert result.kind == "image"
    assert result.resource_type == 1
    assert (storage_root / result.path).read_bytes() == PNG_BYTES
    assert user.avatar == result.path
