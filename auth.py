# -*- coding: utf-8 -*-
"""本機 JSON 帳號儲存：密碼以 PBKDF2-HMAC-SHA256 加鹽雜湊後儲存，不落地明文。"""
import os
import json
import time
import hashlib
import binascii

USERS_PATH = os.path.join("data_auth", "users.json")
PBKDF2_ITERATIONS = 200_000


def _ensure_store():
    os.makedirs(os.path.dirname(USERS_PATH), exist_ok=True)
    if not os.path.exists(USERS_PATH):
        with open(USERS_PATH, "w", encoding="utf-8") as f:
            json.dump({}, f)


def _load():
    _ensure_store()
    with open(USERS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(users):
    with open(USERS_PATH, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


def _hash_password(password, salt_hex=None):
    salt = bytes.fromhex(salt_hex) if salt_hex else os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return salt.hex(), digest.hex()


def register(username, password):
    """回傳 (成功與否, 原因代碼)。原因代碼: ok / empty / exists"""
    username = username.strip()
    if not username or not password:
        return False, "empty"

    users = _load()
    if username in users:
        return False, "exists"

    salt_hex, hash_hex = _hash_password(password)
    users[username] = {"salt": salt_hex, "hash": hash_hex, "created_at": time.time()}
    _save(users)
    return True, "ok"


def verify(username, password):
    username = username.strip()
    if not username or not password:
        return False

    users = _load()
    record = users.get(username)
    if not record:
        return False

    _, hash_hex = _hash_password(password, record["salt"])
    return hash_hex == record["hash"]
