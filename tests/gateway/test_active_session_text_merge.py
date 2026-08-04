"""Regression tests for active-session TEXT follow-up queueing.

When the agent is actively running, rapid text follow-ups should survive as
one next-turn pending message instead of clobbering each other. In
``busy_text_mode=queue`` those active follow-ups first pass through a short
debounce so bursty multi-message thoughts are merged before the active drain
hands off the next turn.
"""

from __future__ import annotations

import asyncio
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Minimal telegram stub so importing gateway.platforms.base does not pull
# in the real python-telegram-bot dependency.
_tg = sys.modules.get("telegram") or types.ModuleType("telegram")
_tg.constants = sys.modules.get("telegram.constants") or types.ModuleType("telegram.constants")
_ct = MagicMock()
_ct.PRIVATE = "private"
_ct.GROUP = "group"
_ct.SUPERGROUP = "supergroup"
_tg.constants.ChatType = _ct
sys.modules.setdefault("telegram", _tg)
sys.modules.setdefault("telegram.constants", _tg.constants)
sys.modules.setdefault("telegram.ext", types.ModuleType("telegram.ext"))

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.session import SessionSource, build_session_key


def _make_event(
    text: str,
    chat_id: str = "12345",
    *,
    chat_type: str = "dm",
    user_id: str = "u1",
    user_name: str | None = None,
    thread_id: str | None = None,
) -> MessageEvent:
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_type=chat_type,
        user_id=user_id,
        user_name=user_name,
        thread_id=thread_id,
    )
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        message_id=f"msg-{text[:8]}",
    )


class _DummyAdapter(BasePlatformAdapter):  # type: ignore[misc]
    async def connect(self, *, is_reconnect: bool = False):
        pass

    async def disconnect(self):
        pass

    async def get_chat_info(self, chat_id):
        return None

    async def send(self, *args, **kwargs):
        return SendResult(success=True, message_id="x")


def _make_initialized_adapter() -> BasePlatformAdapter:
    return _DummyAdapter(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)


def _make_adapter() -> BasePlatformAdapter:
    """Build a BasePlatformAdapter without running its heavy __init__."""
    adapter = object.__new__(_DummyAdapter)
    adapter.config = PlatformConfig(enabled=True, token="***")
    adapter.platform = Platform.TELEGRAM
    adapter._message_handler = AsyncMock(return_value=None)
    adapter._busy_session_handler = None
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    adapter._session_tasks = {}
    adapter._background_tasks = set()
    adapter._post_delivery_callbacks = {}
    adapter._expected_cancelled_tasks = set()
    adapter._fatal_error_code = None
    adapter._fatal_error_message = None
    adapter._fatal_error_retryable = True
    adapter._fatal_error_handler = None
    adapter._running = True
    adapter._busy_text_mode = "queue"
    adapter._busy_text_debounce_seconds = 0.1
    adapter._busy_text_hard_cap_seconds = 1.0
    adapter._text_debounce = {}
    adapter._ingress_batch_seconds = 0.0
    adapter._ingress_batch_hard_cap_seconds = 0.0
    adapter._ingress_batches = {}
    adapter._auto_tts_default = False
    adapter._auto_tts_enabled_chats = set()
    adapter._auto_tts_disabled_chats = set()
    adapter._typing_paused = set()
    return adapter


def _debounced_event(adapter: BasePlatformAdapter, session_key: str) -> MessageEvent:
    return adapter._text_debounce[session_key].event


def _make_media_event(
    text: str,
    path: str,
    media_type: str,
    *,
    message_type: MessageType = MessageType.DOCUMENT,
    chat_id: str = "12345",
    user_id: str = "u1",
    thread_id: str | None = None,
) -> MessageEvent:
    event = _make_event(
        text,
        chat_id=chat_id,
        user_id=user_id,
        thread_id=thread_id,
    )
    event.message_type = message_type
    event.media_urls = [path]
    event.media_types = [media_type]
    return event


@pytest.mark.asyncio
async def test_idle_ingress_batches_three_text_messages_before_starting_turn():
    adapter = _make_adapter()
    adapter._ingress_batch_seconds = 0.05
    adapter._ingress_batch_hard_cap_seconds = 0.5
    started: list[MessageEvent] = []

    def _fake_start(event, session_key, *, interrupt_event=None):
        started.append(event)
        return True

    adapter._start_session_processing = _fake_start  # type: ignore[method-assign]

    await adapter.handle_message(_make_event("one"))
    await adapter.handle_message(_make_event("two"))
    await adapter.handle_message(_make_event("three"))

    assert started == []
    await asyncio.sleep(0.12)

    assert len(started) == 1
    assert started[0].text == "one\ntwo\nthree"
    assert started[0].media_urls == []
    assert adapter._ingress_batches == {}


@pytest.mark.asyncio
async def test_idle_ingress_batches_text_and_file_into_one_turn():
    adapter = _make_adapter()
    adapter._ingress_batch_seconds = 0.05
    started: list[MessageEvent] = []
    adapter._start_session_processing = lambda event, session_key, *, interrupt_event=None: started.append(event) or True  # type: ignore[method-assign]

    await adapter.handle_message(_make_event("task text"))
    await adapter.handle_message(_make_media_event("file caption", "/tmp/a.pdf", "application/pdf"))

    await asyncio.sleep(0.12)

    assert len(started) == 1
    assert started[0].text == "task text\nfile caption"
    assert started[0].media_urls == ["/tmp/a.pdf"]
    assert started[0].media_types == ["application/pdf"]
    assert started[0].message_type == MessageType.DOCUMENT


@pytest.mark.asyncio
async def test_idle_ingress_batches_text_and_image_into_one_turn():
    adapter = _make_adapter()
    adapter._ingress_batch_seconds = 0.05
    started: list[MessageEvent] = []
    adapter._start_session_processing = lambda event, session_key, *, interrupt_event=None: started.append(event) or True  # type: ignore[method-assign]

    await adapter.handle_message(_make_event("look at this"))
    await adapter.handle_message(
        _make_media_event(
            "image caption",
            "/tmp/image.jpg",
            "image/jpeg",
            message_type=MessageType.PHOTO,
        )
    )

    await asyncio.sleep(0.12)

    assert len(started) == 1
    assert started[0].text == "look at this\nimage caption"
    assert started[0].media_urls == ["/tmp/image.jpg"]
    assert started[0].media_types == ["image/jpeg"]
    assert started[0].message_type == MessageType.PHOTO


@pytest.mark.asyncio
async def test_idle_ingress_batch_keys_include_user_and_thread():
    adapter = _make_adapter()
    adapter._ingress_batch_seconds = 0.05
    started: list[MessageEvent] = []
    adapter._start_session_processing = lambda event, session_key, *, interrupt_event=None: started.append(event) or True  # type: ignore[method-assign]

    await adapter.handle_message(_make_event("alice t1", chat_type="group", user_id="alice", thread_id="topic-1"))
    await adapter.handle_message(_make_event("bob t1", chat_type="group", user_id="bob", thread_id="topic-1"))
    await adapter.handle_message(_make_event("alice t2", chat_type="group", user_id="alice", thread_id="topic-2"))

    await asyncio.sleep(0.12)

    assert sorted(event.text for event in started) == ["alice t1", "alice t2", "bob t1"]


@pytest.mark.asyncio
async def test_idle_ingress_discards_pending_batch_when_reset_command_arrives():
    adapter = _make_adapter()
    adapter._ingress_batch_seconds = 0.05
    started: list[MessageEvent] = []
    adapter._start_session_processing = lambda event, session_key, *, interrupt_event=None: started.append(event) or True  # type: ignore[method-assign]

    await adapter.handle_message(_make_event("stale task text"))
    await adapter.handle_message(_make_event("/new"))

    await asyncio.sleep(0.12)

    assert [event.text for event in started] == ["/new"]
    assert adapter._ingress_batches == {}


@pytest.mark.asyncio
async def test_idle_ingress_preserves_order_when_session_becomes_busy_before_next_event():
    adapter = _make_adapter()
    adapter._ingress_batch_seconds = 0.2
    adapter._busy_text_mode = ""  # expose direct active-session pending merge order
    first = _make_event("first")
    session_key = build_session_key(first.source)

    await adapter.handle_message(first)
    adapter._active_sessions[session_key] = asyncio.Event()
    await adapter.handle_message(_make_event("second"))

    await asyncio.sleep(0.25)

    assert adapter._pending_messages[session_key].text == "first\nsecond"
    assert adapter._ingress_batches == {}


@pytest.mark.asyncio
async def test_ingress_batch_cleanup_in_cancel_background_tasks():
    adapter = _make_adapter()
    adapter._ingress_batch_seconds = 1.0
    started: list[MessageEvent] = []
    adapter._start_session_processing = lambda event, session_key, *, interrupt_event=None: started.append(event) or True  # type: ignore[method-assign]

    await adapter.handle_message(_make_event("cleanup ingress"))
    assert adapter._ingress_batches

    await adapter.cancel_background_tasks()
    await asyncio.sleep(0.05)

    assert adapter._ingress_batches == {}
    assert started == []


def test_ingress_batch_config_ignores_malformed_values():
    adapter = _DummyAdapter(
        PlatformConfig(
            enabled=True,
            token="***",
            extra={
                "ingress_batch_seconds": "not-a-float",
                "ingress_batch_hard_cap_seconds": "also-bad",
            },
        ),
        Platform.TELEGRAM,
    )

    assert adapter._ingress_batch_seconds == 0.0
    assert adapter._ingress_batch_hard_cap_seconds == 0.0


@pytest.mark.asyncio
async def test_idle_ingress_flush_queues_if_session_becomes_busy():
    adapter = _make_adapter()
    adapter._ingress_batch_seconds = 0.05
    event = _make_event("queued after busy")
    session_key = build_session_key(event.source)

    await adapter.handle_message(event)
    adapter._active_sessions[session_key] = asyncio.Event()

    await asyncio.sleep(0.12)

    assert session_key in adapter._pending_messages
    assert adapter._pending_messages[session_key].text == "queued after busy"
    assert adapter._ingress_batches == {}


@pytest.mark.asyncio
async def test_idle_ingress_batch_hard_cap_splits_late_message():
    adapter = _make_adapter()
    adapter._ingress_batch_seconds = 0.2
    adapter._ingress_batch_hard_cap_seconds = 0.05
    started: list[MessageEvent] = []

    def _fake_start(event, session_key, *, interrupt_event=None):
        started.append(event)
        adapter._active_sessions[session_key] = interrupt_event or asyncio.Event()
        return True

    adapter._start_session_processing = _fake_start  # type: ignore[method-assign]

    await adapter.handle_message(_make_event("first"))
    await asyncio.sleep(0.08)
    await adapter.handle_message(_make_event("late"))
    await asyncio.sleep(0.12)

    assert [event.text for event in started] == ["first"]
    # The second event arrived after the first batch's hard cap and is queued
    # behind the now-active session instead of being merged into the first turn.
    assert build_session_key(_make_event("late").source) in adapter._pending_messages


@pytest.mark.asyncio
async def test_idle_ingress_preserves_order_for_distinct_users_in_shared_thread_session():
    adapter = _make_adapter()
    adapter._ingress_batch_seconds = 0.2
    started: list[str] = []

    def _fake_start(event, session_key, *, interrupt_event=None):
        started.append(event.text)
        adapter._active_sessions[session_key] = interrupt_event or asyncio.Event()
        return True

    adapter._start_session_processing = _fake_start  # type: ignore[method-assign]

    alice = _make_event(
        "alice first",
        chat_id="group-1",
        chat_type="group",
        user_id="alice",
        thread_id="topic-1",
    )
    bob = _make_event(
        "bob second",
        chat_id="group-1",
        chat_type="group",
        user_id="bob",
        thread_id="topic-1",
    )
    session_key = build_session_key(alice.source)
    assert session_key == build_session_key(bob.source)
    assert adapter._build_ingress_batch_key(alice) != adapter._build_ingress_batch_key(bob)

    await adapter.handle_message(alice)
    await adapter.handle_message(bob)

    assert started == ["alice first"]
    assert adapter._pending_messages[session_key].text == "bob second"
    assert adapter._ingress_batches == {}


@pytest.mark.asyncio
async def test_idle_ingress_preserves_order_for_distinct_users_when_group_sessions_shared():
    adapter = _make_adapter()
    adapter.config.extra["group_sessions_per_user"] = False
    adapter._ingress_batch_seconds = 0.2
    started: list[str] = []

    def _fake_start(event, session_key, *, interrupt_event=None):
        started.append(event.text)
        adapter._active_sessions[session_key] = interrupt_event or asyncio.Event()
        return True

    adapter._start_session_processing = _fake_start  # type: ignore[method-assign]

    alice = _make_event("alice first", chat_id="group-1", chat_type="group", user_id="alice")
    bob = _make_event("bob second", chat_id="group-1", chat_type="group", user_id="bob")
    session_key = build_session_key(alice.source, group_sessions_per_user=False)
    assert session_key == build_session_key(bob.source, group_sessions_per_user=False)
    assert adapter._build_ingress_batch_key(alice) != adapter._build_ingress_batch_key(bob)

    await adapter.handle_message(alice)
    await adapter.handle_message(bob)

    assert started == ["alice first"]
    assert adapter._pending_messages[session_key].text == "bob second"
    assert adapter._ingress_batches == {}


@pytest.mark.asyncio
async def test_idle_ingress_key_includes_session_key_to_prevent_cross_session_merge():
    adapter = _make_adapter()
    adapter._ingress_batch_seconds = 0.05
    started: list[tuple[str, str]] = []

    def _fake_start(event, session_key, *, interrupt_event=None):
        started.append((session_key, event.text))
        return True

    adapter._start_session_processing = _fake_start  # type: ignore[method-assign]

    dm_event = _make_event("dm text", chat_id="same", chat_type="dm", user_id="u1")
    group_event = _make_event("group text", chat_id="same", chat_type="group", user_id="u1")
    dm_session = build_session_key(dm_event.source)
    group_session = build_session_key(group_event.source)
    assert dm_session != group_session
    assert adapter._build_ingress_batch_key(dm_event, dm_session) != adapter._build_ingress_batch_key(group_event, group_session)

    await adapter.handle_message(dm_event)
    await adapter.handle_message(group_event)
    await asyncio.sleep(0.12)

    assert started == [(dm_session, "dm text"), (group_session, "group text")]
    assert adapter._ingress_batches == {}


@pytest.mark.asyncio
async def test_non_dm_message_does_not_wait_for_topic_recovery_executor(monkeypatch):
    """Group messages must not queue behind the shared thread pool.

    Topic recovery only applies to Telegram DM topic mode. Offloading that
    no-op check for every group message makes ingress wait behind unrelated
    blocking jobs when the default executor is saturated.
    """
    adapter = _make_adapter()
    recovery = MagicMock(return_value=None)
    adapter.set_topic_recovery_fn(recovery)
    executor_called = False
    never_release = asyncio.Event()

    async def _blocked_to_thread(*args, **kwargs):
        nonlocal executor_called
        executor_called = True
        await never_release.wait()

    monkeypatch.setattr(asyncio, "to_thread", _blocked_to_thread)

    await asyncio.wait_for(
        adapter.handle_message(_make_event("/status", chat_type="group")),
        timeout=1.0,
    )
    await asyncio.sleep(0)

    assert executor_called is False
    recovery.assert_not_called()


@pytest.mark.asyncio
async def test_dm_topic_recovery_stays_offloaded(monkeypatch):
    """Real Telegram DM topic recovery must still run outside the event loop."""
    adapter = _make_adapter()
    recovery = MagicMock(return_value="topic-222")
    adapter.set_topic_recovery_fn(recovery)
    offloaded = False

    async def _inline_to_thread(func, *args, **kwargs):
        nonlocal offloaded
        offloaded = True
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", _inline_to_thread)
    event = _make_event("hello", chat_type="dm", thread_id="1")
    original_source = event.source

    await adapter.handle_message(event)
    await asyncio.sleep(0)

    assert offloaded is True
    assert recovery.call_count == 1
    assert recovery.call_args.args[0] is original_source
    assert event.source.thread_id == "topic-222"


@pytest.mark.asyncio
async def test_rapid_text_followups_accumulate_instead_of_replacing():
    """Rapid TEXT follow-ups must all survive in the pending event."""
    adapter = _make_adapter()
    adapter._busy_text_mode = ""  # direct-merge behavior, no debounce
    first = _make_event("part one")
    session_key = build_session_key(first.source)
    adapter._active_sessions[session_key] = asyncio.Event()

    await adapter.handle_message(_make_event("part two"))
    await adapter.handle_message(_make_event("part three"))

    pending = adapter._pending_messages[session_key]
    assert pending.text == "part two\npart three"
    assert not adapter._active_sessions[session_key].is_set()


@pytest.mark.asyncio
async def test_debounce_resets_timer_on_new_arrival():
    adapter = _make_adapter()
    adapter._busy_text_debounce_seconds = 0.1

    first = _make_event("one")
    session_key = build_session_key(first.source)
    adapter._active_sessions[session_key] = asyncio.Event()

    await adapter.handle_message(first)
    task1 = adapter._text_debounce[session_key].task
    assert task1 is not None
    assert not task1.done()

    await adapter.handle_message(_make_event("two"))
    task2 = adapter._text_debounce[session_key].task
    assert task2 is not None
    assert task2 is not task1
    await asyncio.sleep(0)
    assert task1.cancelled() or task1.done()
    assert adapter._text_debounce[session_key].task is task2

    await adapter.handle_message(_make_event("three"))
    task3 = adapter._text_debounce[session_key].task
    assert task3 is not None
    assert task3 is not task2

    await asyncio.sleep(0.2)
    assert session_key not in adapter._text_debounce
    assert adapter._pending_messages[session_key].text == "one\ntwo\nthree"


@pytest.mark.asyncio
async def test_control_and_clarify_messages_bypass_text_debounce():
    adapter = _make_adapter()
    started: list[str] = []

    def _fake_start(event, session_key, *, interrupt_event=None):
        started.append(event.text)
        return True

    adapter._start_session_processing = _fake_start  # type: ignore[method-assign]

    await adapter.handle_message(_make_event("/status"))
    assert started == ["/status"]
    assert adapter._text_debounce == {}

    answer = _make_event("clarify answer")
    session_key = build_session_key(answer.source)
    adapter._active_sessions[session_key] = asyncio.Event()
    adapter._message_handler = AsyncMock(return_value=None)

    with patch("tools.clarify_gateway.get_pending_for_session", return_value=object()):
        await adapter.handle_message(answer)

    adapter._message_handler.assert_awaited_once_with(answer)
    assert session_key not in adapter._text_debounce
    assert session_key not in adapter._pending_messages


def test_adapter_defaults_to_interrupt_mode(monkeypatch):
    monkeypatch.delenv("HERMES_GATEWAY_BUSY_TEXT_MODE", raising=False)
    adapter = _make_initialized_adapter()
    assert adapter._busy_text_mode == "interrupt"
    assert not adapter._is_queue_text_debounce_candidate(_make_event("hello"))


def test_command_messages_bypass_debounce_even_in_queue_mode():
    adapter = _make_adapter()
    assert not adapter._is_queue_text_debounce_candidate(_make_event(""))
    assert not adapter._is_queue_text_debounce_candidate(_make_event("/stop"))


