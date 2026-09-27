"""密码安全：强度校验规则。

策略（三条同时满足）：常见弱口令黑名单 + 必须同时包含字母和数字 + 不得包含用户名。
校验失败抛 `ValueError`，由 FastAPI 统一转成 422 `PARAMETER_ERROR` 响应。
"""

from __future__ import annotations

import re

# 常见弱口令。重点覆盖能通过“字母 + 数字”规则的组合（如 abc123、qwerty123），
# 纯数字、纯字母的组合会被下面的组合规则直接拦掉。
WEAK_PASSWORDS: frozenset[str] = frozenset(
    {
        "123456", "12345678", "123456789", "1234567890", "111111", "000000", "888888", "666666",
        "password", "password1", "passw0rd", "admin", "admin123", "root1234", "test1234",
        "qwerty", "qwerty123", "1qaz2wsx", "1q2w3e4r", "zaq12wsx", "abc123", "abcd1234",
        "a123456", "aa123456", "123456a", "123456q", "qwe123", "123qwe", "iloveyou", "letmein",
    }
)

_HAS_LETTER = re.compile(r"[A-Za-z]")
_HAS_DIGIT = re.compile(r"\d")


def validate_password_strength(password: str, username: str = "") -> None:
    """校验密码强度，不通过时抛 ValueError（提示面向用户、中文）。"""
    if password.lower() in WEAK_PASSWORDS:
        raise ValueError("密码过于简单，属于常见弱口令")
    if not (_HAS_LETTER.search(password) and _HAS_DIGIT.search(password)):
        raise ValueError("密码必须同时包含字母和数字")
    if username and username.lower() in password.lower():
        raise ValueError("密码不能包含用户名")
