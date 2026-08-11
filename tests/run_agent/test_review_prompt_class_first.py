"""Behavior tests for the skill review / combined review prompts.

The review prompts gate what the background review agent is allowed to
propose. "Nothing to save." is the DEFAULT outcome: the reviewer runs after
every turn, so a prompt that pushes it to act produces mostly noise for the
user to reject. When it does act, it keeps a strong bias toward:
  1. Patching currently-loaded skills first,
  2. Patching existing umbrellas next,
  3. Adding references/ files under an existing umbrella,
  4. Creating a new class-level umbrella only when nothing else fits.

User-preference corrections (style, format, verbosity, legibility) are
first-class skill signals, not just memory signals — they are also the one
signal that is self-evidencing, since the correction is in the transcript.

These tests assert behavioral *instructions* are present — they do NOT
snapshot the full prompt text (change-detector).
"""

import pytest

from run_agent import AIAgent


ACTING_PROMPTS = [
    ("_SKILL_REVIEW_PROMPT", AIAgent._SKILL_REVIEW_PROMPT),
    ("_COMBINED_REVIEW_PROMPT", AIAgent._COMBINED_REVIEW_PROMPT),
    ("_MEMORY_REVIEW_PROMPT", AIAgent._MEMORY_REVIEW_PROMPT),
]


# ---------------------------------------------------------------------------
# Output bias. The review fork used to be told "Be ACTIVE — most sessions
# produce at least one skill update. A pass that does nothing is a missed
# learning opportunity". That mandates output regardless of whether the
# session contained anything worth saving, and with write_approval on it
# lands as a queue of proposals the user has to reject by hand. The default
# is now inaction; these tests are the tripwire against reintroducing it.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,prompt", ACTING_PROMPTS)
def test_review_prompts_default_to_nothing_to_save(label, prompt):
    """'Nothing to save.' must be stated as the DEFAULT, not a fallback."""
    lower = prompt.lower()
    assert "nothing to save." in lower, f"{label}: must offer the no-op reply verbatim"
    assert "correct default" in lower, (
        f"{label}: must name inaction as the correct default outcome"
    )


@pytest.mark.parametrize("label,prompt", ACTING_PROMPTS)
def test_review_prompts_do_not_mandate_output(label, prompt):
    """No prompt may frame a do-nothing pass as a failure or missed chance."""
    lower = prompt.lower()
    for banned in ("missed learning", "missed opportunity", "be active",
                   "should not be the default", "not a neutral outcome"):
        assert banned not in lower, (
            f"{label}: must not pressure the reviewer to act ({banned!r})"
        )


@pytest.mark.parametrize("label,prompt", ACTING_PROMPTS)
def test_review_prompts_require_evidence(label, prompt):
    """Claims about databases/schemas/scripts/files need transcript evidence.

    The review fork is whitelisted to memory + skill tools — no terminal, no
    read_file, no SQL — so it cannot verify a factual claim it makes. The
    prompt must forbid asserting one rather than leaving it to judgement.
    """
    lower = prompt.lower()
    assert "cannot point to" in lower or "point to" in lower, (
        f"{label}: must require the reviewer to point at its evidence"
    )
    # The two admissible sources, and only those two.
    assert "conversation above" in lower or "transcript" in lower, (
        f"{label}: must name the replayed conversation as an evidence source"
    )


@pytest.mark.parametrize("label,prompt", ACTING_PROMPTS)
def test_review_prompts_skip_already_covered(label, prompt):
    """Content already in SOUL.md / config / an existing skill is not a finding."""
    lower = prompt.lower()
    assert "soul.md" in lower, f"{label}: must name SOUL.md as pre-existing coverage"
    assert "configuration" in lower or "config" in lower, (
        f"{label}: must name configuration as pre-existing coverage"
    )


def test_acting_prompts_name_skill_view_as_evidence():
    """skill_view output is admissible evidence — it is inside the whitelist.

    Restricting evidence to the transcript alone would forbid patching a skill
    the fork just read, which the read-before-write guard in
    tools/skill_manager_tool.py actively requires it to do first.
    """
    for label, prompt in ACTING_PROMPTS[:2]:  # skill + combined touch skills
        assert "skill_view" in prompt, (
            f"{label}: must admit skill_view output as evidence"
        )


# ---------------------------------------------------------------------------
# _SKILL_REVIEW_PROMPT
# ---------------------------------------------------------------------------


def test_skill_review_prompt_treats_user_corrections_as_skill_signal():
    """Style/format/verbosity complaints must be FIRST-CLASS skill signals, not just memory."""
    prompt = AIAgent._SKILL_REVIEW_PROMPT
    lower = prompt.lower()
    # Must mention style/format/verbosity-family corrections
    assert any(k in lower for k in ("style", "format", "verbos", "legib", "tone")), (
        "must name style/format/verbosity/legibility as signals"
    )
    # Must frame these as first-class skill signals (not memory-only)
    assert "FIRST-CLASS" in prompt or "first-class" in prompt, (
        "must explicitly label user-preference corrections as first-class skill signals"
    )
    # Must mention the correction-type phrases to tune the model's ear
    assert "stop doing" in lower or "don't" in lower or "hate" in lower or "frustrat" in lower, (
        "must give concrete phrasing examples so the model recognizes corrections"
    )
















# ---------------------------------------------------------------------------
# _COMBINED_REVIEW_PROMPT
# ---------------------------------------------------------------------------

def test_combined_review_prompt_has_memory_section():
    """Memory half must still cover user facts and preferences."""
    prompt = AIAgent._COMBINED_REVIEW_PROMPT
    assert "**Memory**" in prompt
    assert "memory tool" in prompt














# ---------------------------------------------------------------------------
# Anti-pattern guidance — see issue #6051. The reviewer was learning transient
# environment failures (e.g. "browser tools do not work" from a fresh-install
# Playwright miss) as durable skill rules, then citing them against itself for
# weeks after the environment was fixed. Both review prompts must explicitly
# tell the reviewer not to capture environment-dependent or negative-framing
# content as skills.
# ---------------------------------------------------------------------------


def _assert_anti_pattern_guidance(prompt: str, label: str) -> None:
    """Both review prompts must carry the same anti-pattern section."""
    lower = prompt.lower()
    assert "do not capture" in lower, (
        f"{label}: must have an explicit 'Do NOT capture' section"
    )
    # Environment-dependent failures (the #6051 root cause)
    assert any(k in lower for k in ("missing binar", "command not found", "uninstalled", "fresh-install")), (
        f"{label}: must call out environment/setup failures as not-skill-worthy"
    )
    # Negative-framing avoidance
    assert any(k in lower for k in ("negative claim", "do not work", "is broken")), (
        f"{label}: must call out negative-claim phrasings as the failure mode"
    )
    # Positive reframing — "capture the fix, not the failure"
    assert "capture the fix" in lower or "capture the fix " in lower, (
        f"{label}: must redirect tool-failure capture toward the fix, not the constraint"
    )
    # One-off task narratives (#12812 family)
    assert "one-off" in lower, (
        f"{label}: must call out one-off task narratives as not-skill-worthy"
    )


def _assert_unresolved_failure_guidance(prompt: str, label: str) -> None:
    """Unresolved task attempts must not become persistent skill guidance."""
    lower = prompt.lower()
    assert "unresolved failures" in lower, f"{label}: must identify unresolved failures"
    assert "working method" in lower, f"{label}: must require a working method"
    assert "told the user to check manually" in lower, (
        f"{label}: must recognize an explicitly unresolved session"
    )
    assert "never the dead ends" in lower, f"{label}: must exclude failed attempts"
    assert "independently confident" in lower, (
        f"{label}: must limit exceptions to verified alternatives"
    )


def test_skill_review_prompt_rejects_unresolved_failures():
    _assert_unresolved_failure_guidance(AIAgent._SKILL_REVIEW_PROMPT, "_SKILL_REVIEW_PROMPT")


def test_combined_review_prompt_rejects_unresolved_failures():
    _assert_unresolved_failure_guidance(AIAgent._COMBINED_REVIEW_PROMPT, "_COMBINED_REVIEW_PROMPT")






# ---------------------------------------------------------------------------
# _MEMORY_REVIEW_PROMPT — unchanged, still memory-focused
# ---------------------------------------------------------------------------
