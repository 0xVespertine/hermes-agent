"""discard_pending() archives decided proposals instead of deleting them.

The background reviewer proposes constantly and most proposals get rejected.
Deleting them threw away the only record of what the reviewer actually emits,
which is the corpus needed to tell whether a prompt change helped.

``discard_pending`` is called from BOTH the approve path (after the write
landed) and the reject path, so ``outcome`` is mandatory — a default would
silently mislabel half the archive and make it useless as evidence.
"""

import json

import pytest

from tools import write_approval as wa


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    monkeypatch.setattr(wa, "get_hermes_home", lambda: tmp_path)
    return tmp_path


def _stage(subsystem="memory", summary="user prefers short answers"):
    return wa.stage_write(
        subsystem,
        {"action": "add", "target": "user", "content": summary},
        summary=summary,
        origin="background_review",
    )


class TestOutcomeIsMandatory:
    def test_missing_outcome_raises(self, hermes_home):
        rec = _stage()
        with pytest.raises(TypeError):
            wa.discard_pending("memory", rec["id"])

    def test_pending_record_survives_a_missing_outcome(self, hermes_home):
        """The loud failure must not consume the record."""
        rec = _stage()
        with pytest.raises(TypeError):
            wa.discard_pending("memory", rec["id"])
        assert [r["id"] for r in wa.list_pending("memory")] == [rec["id"]]

    def test_unknown_outcome_raises(self, hermes_home):
        rec = _stage()
        with pytest.raises(ValueError):
            wa.discard_pending("memory", rec["id"], outcome="maybe")
        assert wa.pending_count("memory") == 1


class TestArchiving:
    @pytest.mark.parametrize("outcome", ["approved", "rejected"])
    def test_record_moves_to_its_outcome_dir(self, hermes_home, outcome):
        rec = _stage()
        assert wa.discard_pending("memory", rec["id"], outcome=outcome) is True

        archived = hermes_home / "pending" / "memory" / f".{outcome}" / f"{rec['id']}.json"
        assert archived.exists()
        data = json.loads(archived.read_text(encoding="utf-8"))
        assert data["decision"] == outcome
        assert isinstance(data["decided_at"], float)
        # Original payload is preserved for replay/analysis.
        assert data["payload"]["content"] == "user prefers short answers"
        assert data["origin"] == "background_review"

    def test_approved_and_rejected_do_not_mix(self, hermes_home):
        keep, drop = _stage(summary="keep me"), _stage(summary="drop me")
        wa.discard_pending("memory", keep["id"], outcome="approved")
        wa.discard_pending("memory", drop["id"], outcome="rejected")

        base = hermes_home / "pending" / "memory"
        assert [p.stem for p in (base / ".approved").glob("*.json")] == [keep["id"]]
        assert [p.stem for p in (base / ".rejected").glob("*.json")] == [drop["id"]]

    def test_archived_records_leave_the_pending_queue(self, hermes_home):
        rec = _stage()
        wa.discard_pending("memory", rec["id"], outcome="rejected")
        assert wa.list_pending("memory") == []
        assert wa.pending_count("memory") == 0
        assert wa.get_pending("memory", rec["id"]) is None

    def test_archive_dirs_are_invisible_to_the_pending_listing(self, hermes_home):
        """list_pending/pending_count glob *.json non-recursively — keep it that way."""
        for i in range(3):
            r = _stage(summary=f"entry {i}")
            wa.discard_pending("memory", r["id"], outcome="rejected")
        live = _stage(summary="still pending")

        assert wa.pending_count("memory") == 1
        assert [r["id"] for r in wa.list_pending("memory")] == [live["id"]]

    def test_missing_record_returns_false(self, hermes_home):
        assert wa.discard_pending("memory", "nope", outcome="rejected") is False

    def test_corrupt_record_is_archived_verbatim(self, hermes_home):
        """A record that won't parse is still evidence — keep its bytes."""
        rec = _stage()
        raw = hermes_home / "pending" / "memory" / f"{rec['id']}.json"
        raw.write_text("{not valid json at all", encoding="utf-8")

        assert wa.discard_pending("memory", rec["id"], outcome="rejected") is True

        archived = hermes_home / "pending" / "memory" / ".rejected" / f"{rec['id']}.json"
        assert archived.read_text(encoding="utf-8") == "{not valid json at all"

    def test_record_is_never_deleted_when_archiving_fails(self, hermes_home, monkeypatch):
        """On any archive failure the proposal stays pending — never dropped.

        Deleting on failure would destroy exactly the evidence the archive
        exists to collect.
        """
        rec = _stage()
        monkeypatch.setattr(
            wa.os, "replace",
            lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")),
        )

        assert wa.discard_pending("memory", rec["id"], outcome="rejected") is False
        assert [r["id"] for r in wa.list_pending("memory")] == [rec["id"]]
        assert wa.get_pending("memory", rec["id"])["summary"] == "user prefers short answers"

    def test_failed_metadata_stamp_still_keeps_the_record(self, hermes_home, monkeypatch):
        """The move already happened; a stamp failure costs the timestamp only."""
        rec = _stage()
        real_replace = wa.os.replace
        calls = {"n": 0}

        def flaky(src, dst):
            calls["n"] += 1
            if calls["n"] == 1:          # the archive move itself
                return real_replace(src, dst)
            raise OSError("stamp failed")  # the metadata rewrite

        monkeypatch.setattr(wa.os, "replace", flaky)

        assert wa.discard_pending("memory", rec["id"], outcome="rejected") is True
        archived = hermes_home / "pending" / "memory" / ".rejected" / f"{rec['id']}.json"
        assert archived.exists(), "the record survives a failed stamp"
        # Outcome is still recoverable from the directory it landed in.
        assert archived.parent.name == ".rejected"
        assert wa.pending_count("memory") == 0

    def test_skills_subsystem_archives_separately(self, hermes_home):
        mem = _stage("memory", "a memory")
        skill = _stage("skills", "a skill")
        wa.discard_pending("memory", mem["id"], outcome="rejected")
        wa.discard_pending("skills", skill["id"], outcome="rejected")

        base = hermes_home / "pending"
        assert (base / "memory" / ".rejected" / f"{mem['id']}.json").exists()
        assert (base / "skills" / ".rejected" / f"{skill['id']}.json").exists()
