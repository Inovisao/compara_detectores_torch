from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


def _fresh_module(token: str = "", chat_id: str = "") -> types.ModuleType:
    import importlib
    import telegram_notify as _orig

    with patch.dict(
        "os.environ",
        {"TELEGRAM_BOT_TOKEN": token, "TELEGRAM_CHAT_ID": chat_id},
        clear=False,
    ):
        mod = types.ModuleType("telegram_notify_fresh")
        mod._TOKEN = token
        mod._CHAT_ID = chat_id
        mod.send = _orig.send
        mod.send_error = _orig.send_error
    return mod


def test_send_no_token_is_noop():
    with patch("telegram_notify._TOKEN", ""), patch("telegram_notify._CHAT_ID", ""):
        import telegram_notify as tg
        with patch("requests.post") as mock_post:
            tg.send("hello")
            mock_post.assert_not_called()


def test_send_no_chat_id_is_noop():
    with patch("telegram_notify._TOKEN", "sometoken"), patch("telegram_notify._CHAT_ID", ""):
        import telegram_notify as tg
        with patch("requests.post") as mock_post:
            tg.send("hello")
            mock_post.assert_not_called()


def test_send_calls_api():
    with (
        patch("telegram_notify._TOKEN", "tok123"),
        patch("telegram_notify._CHAT_ID", "999"),
    ):
        import telegram_notify as tg
        with patch("requests.post") as mock_post:
            tg.send("test message")
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args
            assert "sendMessage" in call_kwargs[0][0]
            assert call_kwargs[1]["json"]["text"] == "test message"
            assert call_kwargs[1]["json"]["chat_id"] == "999"


def test_send_swallows_network_error():
    with (
        patch("telegram_notify._TOKEN", "tok123"),
        patch("telegram_notify._CHAT_ID", "999"),
    ):
        import telegram_notify as tg
        with patch("requests.post", side_effect=Exception("network down")):
            tg.send("msg")


def test_send_error_includes_context_and_type():
    with (
        patch("telegram_notify._TOKEN", "tok123"),
        patch("telegram_notify._CHAT_ID", "999"),
    ):
        import telegram_notify as tg
        captured = []

        def fake_send(msg: str) -> None:
            captured.append(msg)

        with patch("telegram_notify.send", side_effect=fake_send):
            try:
                raise ValueError("boom")
            except ValueError as exc:
                tg.send_error("test_context", exc)

        assert len(captured) == 1
        assert "[ERRO]" in captured[0]
        assert "test_context" in captured[0]
        assert "ValueError" in captured[0]


def test_send_photo_calls_api(tmp_path):
    img = tmp_path / "plot.png"
    img.write_bytes(b"fake")
    with (
        patch("telegram_notify._TOKEN", "tok123"),
        patch("telegram_notify._CHAT_ID", "999"),
    ):
        import telegram_notify as tg
        with patch("requests.post") as mock_post:
            tg.send_photo(img)
            mock_post.assert_called_once()
            assert "sendPhoto" in mock_post.call_args[0][0]
            assert mock_post.call_args[1]["data"]["chat_id"] == "999"


def test_send_photo_noop_without_token(tmp_path):
    img = tmp_path / "plot.png"
    img.write_bytes(b"fake")
    with patch("telegram_notify._TOKEN", ""), patch("telegram_notify._CHAT_ID", "999"):
        import telegram_notify as tg
        with patch("requests.post") as mock_post:
            tg.send_photo(img)
            mock_post.assert_not_called()


def test_send_document_calls_api(tmp_path):
    doc = tmp_path / "results.csv"
    doc.write_text("a,b\n1,2")
    with (
        patch("telegram_notify._TOKEN", "tok123"),
        patch("telegram_notify._CHAT_ID", "999"),
    ):
        import telegram_notify as tg
        with patch("requests.post") as mock_post:
            tg.send_document(doc)
            mock_post.assert_called_once()
            assert "sendDocument" in mock_post.call_args[0][0]


def test_send_document_swallows_error(tmp_path):
    doc = tmp_path / "results.csv"
    doc.write_text("a,b\n1,2")
    with (
        patch("telegram_notify._TOKEN", "tok123"),
        patch("telegram_notify._CHAT_ID", "999"),
    ):
        import telegram_notify as tg
        with patch("requests.post", side_effect=Exception("timeout")):
            tg.send_document(doc)


def test_send_oi():
    import telegram_notify as tg
    tg.send("oi")
