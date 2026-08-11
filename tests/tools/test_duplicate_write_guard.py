"""Deterministic duplicate guards on the memory and skill mutation paths.

The background reviewer re-proposes the same content across sessions. Only
provably zero-information writes are refused here — anything requiring a
judgement call is left to the model, by design.

The two stores get deliberately DIFFERENT comparisons:

  * memory  — whitespace-collapsed only. NOT casefolded: memory holds
    usernames, paths, env vars and product names whose case is load-bearing.
  * skills  — exact, no folding at all. SKILL.md has structural YAML
    frontmatter and skills ship scripts/ and templates/ files, so folding
    either case or indentation would classify a real fix as a duplicate.
"""

import json

import pytest

from tools.memory_tool import MemoryStore, _normalize_memory_entry


# ---------------------------------------------------------------------------
# memory — normalized
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path, monkeypatch):
    import tools.memory_tool as mt

    monkeypatch.setattr(mt, "get_memory_dir", lambda: tmp_path)
    s = MemoryStore()
    s.load_from_disk()
    return s


class TestMemoryNormalizedDuplicates:
    @pytest.mark.parametrize("variant", [
        "User prefers short answers",           # identical
        "User  prefers   short  answers",       # internal whitespace
        "  User prefers short answers  ",       # surrounding whitespace
        "User prefers\nshort answers",          # newline as separator
        "User prefers\tshort answers",          # tab as separator
    ])
    def test_whitespace_only_duplicate_is_not_added(self, store, variant):
        first = store.add("user", "User prefers short answers")
        assert first["success"]

        result = store.add("user", variant)
        assert result["success"], "duplicates report success, they just don't write"
        assert "already exists" in result["message"].lower()
        assert len(store._entries_for("user")) == 1

    def test_genuinely_different_entry_is_added(self, store):
        store.add("user", "User prefers short answers")
        store.add("user", "User prefers long answers")
        assert len(store._entries_for("user")) == 2

    @pytest.mark.parametrize("first,second", [
        # Case is load-bearing in memory: these are DIFFERENT facts, and
        # casefolding would silently drop the second one.
        ("User's GitHub handle is Bob", "User's GitHub handle is bob"),
        ("Config lives at ~/.hermes/Config.yaml", "Config lives at ~/.hermes/config.yaml"),
        ("Deploy target is PROD", "Deploy target is prod"),
        ("Env var is HERMES_HOME", "Env var is hermes_home"),
    ])
    def test_case_differences_are_preserved(self, store, first, second):
        store.add("user", first)
        store.add("user", second)
        assert len(store._entries_for("user")) == 2, (
            "casefolding would merge identifiers that differ only in case"
        )

    def test_normalization_does_not_fold_distinct_words(self, store):
        """Collapsing whitespace must not merge tokens across the boundary."""
        assert _normalize_memory_entry("a b") != _normalize_memory_entry("ab")

    def test_punctuation_is_still_significant(self, store):
        """Only whitespace is folded — nothing semantic."""
        store.add("user", "User works on Hermes")
        store.add("user", "User works on Hermes?")
        assert len(store._entries_for("user")) == 2


# ---------------------------------------------------------------------------
# skills — byte-exact, and refused BEFORE the approval gate stages anything
# ---------------------------------------------------------------------------

SKILL_MD = """---
name: demo
description: A demo skill
---

# Demo

- Always check the config first.
"""


@pytest.fixture
def skill(tmp_path, monkeypatch):
    import tools.skill_manager_tool as sm

    d = tmp_path / "demo"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
    monkeypatch.setattr(sm, "_find_skill", lambda name: {"path": d} if name == "demo" else None)
    return d


class TestSkillNoOpGuard:
    def test_write_file_with_identical_content_is_refused(self, skill):
        from tools.skill_manager_tool import _skill_write_is_noop

        (skill / "references").mkdir()
        (skill / "references" / "api.md").write_text("# API\n", encoding="utf-8")

        assert _skill_write_is_noop(
            "write_file", "demo",
            file_path="references/api.md", file_content="# API\n",
        ) is True

    def test_write_file_with_changed_content_passes(self, skill):
        from tools.skill_manager_tool import _skill_write_is_noop

        (skill / "references").mkdir()
        (skill / "references" / "api.md").write_text("# API\n", encoding="utf-8")

        assert _skill_write_is_noop(
            "write_file", "demo",
            file_path="references/api.md", file_content="# API v2\n",
        ) is False

    def test_write_file_whitespace_change_is_allowed_through(self, skill):
        """Indentation is semantic in templates/ and scripts/ — never folded."""
        from tools.skill_manager_tool import _skill_write_is_noop

        (skill / "templates").mkdir()
        (skill / "templates" / "cfg.yaml").write_text("a:\n  b: 1\n", encoding="utf-8")

        assert _skill_write_is_noop(
            "write_file", "demo",
            file_path="templates/cfg.yaml", file_content="a:\n    b: 1\n",
        ) is False

    def test_write_file_case_change_is_allowed_through(self, skill):
        """A YAML False -> false fix is a real edit, not a duplicate."""
        from tools.skill_manager_tool import _skill_write_is_noop

        (skill / "templates").mkdir()
        (skill / "templates" / "cfg.yaml").write_text("debug: False\n", encoding="utf-8")

        assert _skill_write_is_noop(
            "write_file", "demo",
            file_path="templates/cfg.yaml", file_content="debug: false\n",
        ) is False

    def test_comparison_is_text_level_not_byte_level(self, skill):
        """CRLF-vs-LF is a no-op: atomic_write_text would produce the same file.

        The write path goes through a text-mode handle, so a '\\n' in the
        model's content lands as os.linesep on disk. Comparing raw bytes would
        call this a real change and stage a write that alters nothing.
        """
        from tools.skill_manager_tool import _skill_write_is_noop

        (skill / "references").mkdir()
        # Write CRLF bytes explicitly, bypassing newline translation.
        (skill / "references" / "notes.md").write_bytes(b"line one\r\nline two\r\n")

        assert _skill_write_is_noop(
            "write_file", "demo",
            file_path="references/notes.md", file_content="line one\nline two\n",
        ) is True

    def test_patch_replacing_text_with_itself_is_refused(self, skill):
        from tools.skill_manager_tool import _skill_write_is_noop

        assert _skill_write_is_noop(
            "patch", "demo",
            old_string="Always check the config first.",
            new_string="Always check the config first.",
        ) is True

    def test_patch_is_a_noop_when_fuzzy_matching_lands_on_identical_text(self, skill):
        """old_string != new_string can still resolve to a byte-identical file.

        The patch path matches whitespace-tolerantly, so an old_string that
        differs from the file only in spacing, replaced by the file's actual
        text, changes nothing.
        """
        from tools.skill_manager_tool import _skill_write_is_noop

        assert _skill_write_is_noop(
            "patch", "demo",
            old_string="Always  check   the config first.",   # loose spacing
            new_string="Always check the config first.",      # what's on disk
        ) is True

    def test_patch_that_changes_text_passes(self, skill):
        from tools.skill_manager_tool import _skill_write_is_noop

        assert _skill_write_is_noop(
            "patch", "demo",
            old_string="Always check the config first.",
            new_string="Always check the config and SOUL.md first.",
        ) is False

    def test_patch_case_change_is_allowed_through(self, skill):
        """A YAML True/true fix is a real edit, not a duplicate."""
        from tools.skill_manager_tool import _skill_write_is_noop

        assert _skill_write_is_noop(
            "patch", "demo",
            old_string="Always check the config first.",
            new_string="ALWAYS check the config first.",
        ) is False

    def test_unmatched_patch_defers_to_the_real_handler(self, skill):
        """No match is a real error with a useful message — don't mask it here."""
        from tools.skill_manager_tool import _skill_write_is_noop

        assert _skill_write_is_noop(
            "patch", "demo",
            old_string="text that is not in the file",
            new_string="whatever",
        ) is False

    def test_unknown_skill_defers(self, skill):
        from tools.skill_manager_tool import _skill_write_is_noop

        assert _skill_write_is_noop(
            "patch", "nonexistent", old_string="x", new_string="x",
        ) is False

    @pytest.mark.parametrize("action", ["create", "edit", "delete", "remove_file"])
    def test_guard_only_covers_patch_and_write_file(self, skill, action):
        from tools.skill_manager_tool import _skill_write_is_noop

        assert _skill_write_is_noop(action, "demo") is False


class TestNoOpRefusedBeforeStaging:
    """With write_approval on, the gate stages the payload and the real write is
    deferred to approval-replay. A no-op check that only ran in the handler
    would therefore still land a useless proposal in the user's queue."""

    def test_noop_patch_never_reaches_the_approval_gate(self, skill, monkeypatch):
        import tools.skill_manager_tool as sm

        called = []
        monkeypatch.setattr(
            sm, "_apply_skill_write_gate",
            lambda *a, **k: called.append(a) or None,
        )

        out = json.loads(sm.skill_manage(
            action="patch", name="demo",
            old_string="Always check the config first.",
            new_string="Always check the config first.",
        ))
        assert out["success"] is False
        assert "no-op" in out["error"].lower()
        assert called == [], "gate must not run — nothing should be staged"

    def test_real_patch_still_reaches_the_gate(self, skill, monkeypatch):
        import tools.skill_manager_tool as sm

        called = []
        monkeypatch.setattr(
            sm, "_apply_skill_write_gate",
            lambda *a, **k: called.append(a) or None,
        )

        sm.skill_manage(
            action="patch", name="demo",
            old_string="Always check the config first.",
            new_string="Always check SOUL.md first.",
        )
        assert called, "a genuine edit must still be gated/staged"
