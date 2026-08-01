"""Regression coverage for environment-only WhatsApp allowlists.

The Baileys bridge accepts ``WHATSAPP_ALLOWED_USERS`` from the environment,
but the Python adapter must load the same value before enforcing its own
intake gate.  Otherwise an allowed message is silently dropped between the
bridge and the gateway handler.
"""

import pytest

from agent import secret_scope
from gateway.config import Platform, PlatformConfig, load_gateway_config


DM_USER = "6281234567890"
OTHER_DM_USER = "6289999999999"
GROUP_JID = "120363001234567890@g.us"
OTHER_GROUP_JID = "120363009999999999@g.us"


@pytest.fixture(autouse=True)
def _reset_secret_scope_mode():
    secret_scope.set_multiplex_active(False)
    yield
    secret_scope.set_multiplex_active(False)


def _instantiate_adapter(monkeypatch, tmp_path, config):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    monkeypatch.setattr(
        WhatsAppAdapter,
        "_DEFAULT_BRIDGE_DIR",
        tmp_path / "bridge",
    )
    return WhatsAppAdapter(config)


def _build_adapter(monkeypatch, tmp_path, *, extra=None, env=None):
    for key in (
        "WHATSAPP_DM_POLICY",
        "WHATSAPP_ALLOWED_USERS",
        "WHATSAPP_GROUP_POLICY",
        "WHATSAPP_GROUP_ALLOWED_USERS",
        "WHATSAPP_REQUIRE_MENTION",
        "WHATSAPP_FREE_RESPONSE_CHATS",
        "WHATSAPP_MENTION_PATTERNS",
        "WHATSAPP_ALLOW_ALL_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)

    config = PlatformConfig(
        enabled=True,
        extra=extra if extra is not None else {},
    )
    return _instantiate_adapter(monkeypatch, tmp_path, config)


def test_env_only_dm_allowlist_reaches_intake(monkeypatch, tmp_path):
    adapter = _build_adapter(
        monkeypatch,
        tmp_path,
        env={
            "WHATSAPP_DM_POLICY": "allowlist",
            "WHATSAPP_ALLOWED_USERS": DM_USER,
        },
    )

    assert adapter._allow_from == {DM_USER}
    assert adapter._is_dm_intake_allowed(f"{DM_USER}@s.whatsapp.net") is True
    assert adapter._is_dm_intake_allowed(f"{OTHER_DM_USER}@s.whatsapp.net") is False
    assert adapter._should_process_message(
        {
            "isGroup": False,
            "body": "hello",
            "senderId": f"{DM_USER}@s.whatsapp.net",
            "from": f"{DM_USER}@s.whatsapp.net",
        }
    ) is True


def test_env_only_group_allowlist_reaches_intake(monkeypatch, tmp_path):
    adapter = _build_adapter(
        monkeypatch,
        tmp_path,
        env={
            "WHATSAPP_GROUP_POLICY": "allowlist",
            "WHATSAPP_GROUP_ALLOWED_USERS": GROUP_JID,
        },
    )

    assert adapter._group_allow_from == {GROUP_JID}
    assert adapter._is_group_allowed(GROUP_JID) is True
    assert adapter._is_group_allowed(OTHER_GROUP_JID) is False
    assert adapter._should_process_message(
        {
            "isGroup": True,
            "body": "hello",
            "chatId": GROUP_JID,
            "mentionedIds": [],
            "botIds": [],
        }
    ) is True


@pytest.mark.parametrize("config_key", ("allow_from", "allowFrom"))
def test_config_dm_allowlist_takes_precedence_over_environment(
    monkeypatch,
    tmp_path,
    config_key,
):
    adapter = _build_adapter(
        monkeypatch,
        tmp_path,
        extra={"dm_policy": "allowlist", config_key: [DM_USER]},
        env={"WHATSAPP_ALLOWED_USERS": OTHER_DM_USER},
    )

    assert adapter._is_dm_intake_allowed(f"{DM_USER}@s.whatsapp.net") is True
    assert adapter._is_dm_intake_allowed(f"{OTHER_DM_USER}@s.whatsapp.net") is False


@pytest.mark.parametrize("empty_value", (None, [], ""))
def test_explicit_empty_dm_allowlist_does_not_fall_back_to_environment(
    monkeypatch,
    tmp_path,
    empty_value,
):
    adapter = _build_adapter(
        monkeypatch,
        tmp_path,
        extra={"dm_policy": "allowlist", "allow_from": empty_value},
        env={"WHATSAPP_ALLOWED_USERS": DM_USER},
    )

    assert adapter._allow_from == set()
    assert adapter._is_dm_intake_allowed(f"{DM_USER}@s.whatsapp.net") is False


@pytest.mark.parametrize("empty_value", (None, [], ""))
def test_explicit_empty_group_allowlist_does_not_fall_back_to_environment(
    monkeypatch,
    tmp_path,
    empty_value,
):
    adapter = _build_adapter(
        monkeypatch,
        tmp_path,
        extra={"group_policy": "allowlist", "group_allow_from": empty_value},
        env={"WHATSAPP_GROUP_ALLOWED_USERS": GROUP_JID},
    )

    assert adapter._group_allow_from == set()
    assert adapter._is_group_allowed(GROUP_JID) is False


def test_env_only_allowlist_survives_gateway_config_loading(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("WHATSAPP_ENABLED", "true")
    monkeypatch.setenv("WHATSAPP_DM_POLICY", "allowlist")
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", DM_USER)

    config = load_gateway_config()
    adapter = _instantiate_adapter(
        monkeypatch,
        tmp_path,
        config.platforms[Platform.WHATSAPP],
    )

    assert adapter._is_dm_intake_allowed(f"{DM_USER}@s.whatsapp.net") is True
    assert adapter._is_dm_intake_allowed(f"{OTHER_DM_USER}@s.whatsapp.net") is False


def test_profile_scope_does_not_leak_process_allowlist(monkeypatch, tmp_path):
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", OTHER_DM_USER)
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({"WHATSAPP_ALLOWED_USERS": DM_USER})
    try:
        adapter = _instantiate_adapter(
            monkeypatch,
            tmp_path,
            PlatformConfig(
                enabled=True,
                extra={"dm_policy": "allowlist"},
            ),
        )
    finally:
        secret_scope.reset_secret_scope(token)

    assert adapter._is_dm_intake_allowed(f"{DM_USER}@s.whatsapp.net") is True
    assert adapter._is_dm_intake_allowed(f"{OTHER_DM_USER}@s.whatsapp.net") is False
