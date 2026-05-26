from __future__ import annotations

import os
import traceback
from pathlib import Path

from dotenv import load_dotenv
import requests

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


def send(msg: str) -> None:
    if not _TOKEN or not _CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{_TOKEN}/sendMessage",
            json={"chat_id": _CHAT_ID, "text": msg},
            timeout=10,
        )
    except Exception:
        pass


def send_photo(path: str | Path) -> None:
    if not _TOKEN or not _CHAT_ID:
        return
    try:
        with open(path, "rb") as f:
            requests.post(
                f"https://api.telegram.org/bot{_TOKEN}/sendPhoto",
                data={"chat_id": _CHAT_ID},
                files={"photo": f},
                timeout=30,
            )
    except Exception:
        pass


def send_document(path: str | Path) -> None:
    if not _TOKEN or not _CHAT_ID:
        return
    try:
        with open(path, "rb") as f:
            requests.post(
                f"https://api.telegram.org/bot{_TOKEN}/sendDocument",
                data={"chat_id": _CHAT_ID},
                files={"document": f},
                timeout=30,
            )
    except Exception:
        pass


def send_error(context: str, exc: BaseException) -> None:
    tb = traceback.format_exc()
    send(f"[ERRO] {context}\n{type(exc).__name__}: {exc}\n\n{tb[:1000]}")
