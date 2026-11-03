"""Tests for the AI pipeline. The LLMs are faked, so no API key or network is needed."""
import pytest

import tweet_engine as engine
from tweet_engine import (
    MAX_TWEET_CHARS,
    clean_tweet,
    enforce_limit,
    load_style_examples,
    parse_candidates,
    parse_ranking,
    rate_limit_wait,
)


class FakeResponse:
    def __init__(self, content):
        self.content = content


class ScriptedLLM:
    """Returns the scripted outputs in order, repeating the last one forever."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0
        self.last_prompt = ""

    def invoke(self, messages):
        self.last_prompt = "\n".join(m.content for m in messages)
        item = self.outputs[min(self.calls, len(self.outputs) - 1)]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return FakeResponse(item) if isinstance(item, str) else item


def ranking(scores, feedback="because"):
    """What the critic model replies with, as plain text."""
    return f"SCORES: {', '.join(str(x) for x in scores)}\nFEEDBACK: {feedback}"


THREE = "tweet A\n---\ntweet B\n---\ntweet C"


@pytest.fixture
def fake_llms(monkeypatch):
    monkeypatch.setattr(engine, "NUM_CANDIDATES", 3)
    monkeypatch.setattr(engine, "APPROVAL_SCORE", 8)  # keep tests independent of .env

    def install(generator, critic, optimizer=("new X\n---\nnew Y",), fallback=None):
        g, c, o = ScriptedLLM(generator), ScriptedLLM(critic), ScriptedLLM(optimizer)
        monkeypatch.setattr(engine, "generator_llm", g)
        monkeypatch.setattr(engine, "critic_llm", c)
        monkeypatch.setattr(engine, "optimizer_llm", o)
        monkeypatch.setattr(engine, "fallback_critic_llm", ScriptedLLM(fallback) if fallback else None)
        return g, c, o

    return install


# ------------------------------------------------------------ clean_tweet
def test_clean_strips_wrapping_quotes():
    assert clean_tweet('"Coffee is my love language."') == "Coffee is my love language."
    assert clean_tweet("“Smart quotes too”") == "Smart quotes too"


def test_clean_keeps_quotes_that_are_part_of_the_joke():
    text = '"Wow" I said, and "nice" he replied'
    assert clean_tweet(text) == text


def test_clean_removes_preamble():
    assert clean_tweet("Here's a funny tweet: Mondays are a scam.") == "Mondays are a scam."
    assert clean_tweet("Sure! Here's the improved tweet:\nMondays are a scam.") == "Mondays are a scam."


def test_clean_does_not_eat_legit_colons():
    text = "Okay so my coffee order is: oat milk, regret, repeat"
    assert clean_tweet(text) == text


def test_clean_removes_trailing_explanation():
    raw = "Mondays are a scam.\n\n(This tweet uses irony to be relatable)"
    assert clean_tweet(raw) == "Mondays are a scam."


def test_clean_strips_trailing_hashtags_only():
    assert clean_tweet("Mondays are a scam. #Monday #Mood") == "Mondays are a scam."
    assert clean_tweet("I love #coffee more than people") == "I love #coffee more than people"


def test_clean_hashtag_stripping_can_be_turned_off(monkeypatch):
    monkeypatch.setattr(engine, "STRIP_TRAILING_HASHTAGS", False)
    assert clean_tweet("Mondays #Monday") == "Mondays #Monday"


def test_clean_handles_none_and_whitespace():
    assert clean_tweet(None) == ""
    assert clean_tweet("   hi   there  ") == "hi there"


# -------------------------------------------------------- parse_candidates
def test_parse_splits_on_dashes():
    assert parse_candidates("one\n---\ntwo\n---\nthree") == ["one", "two", "three"]


def test_parse_dedupes_and_drops_empty():
    assert parse_candidates("one\n---\nONE\n---\n\n---\ntwo") == ["one", "two"]


def test_parse_strips_numbering_and_intro():
    raw = "Here are 3 tweets:\n---\n1. first joke\n---\n2) second joke"
    assert parse_candidates(raw) == ["first joke", "second joke"]


def test_parse_respects_limit_and_single_output():
    assert parse_candidates("a\n---\nb\n---\nc", limit=2) == ["a", "b"]
    assert parse_candidates("just one tweet") == ["just one tweet"]
    assert parse_candidates("") == []


# ----------------------------------------------------------- parse_ranking
def test_parse_ranking_standard_format():
    r = parse_ranking("SCORES: 5, 9, 6\nFEEDBACK: Sharp ending. Trim the opening.")
    assert r.scores == [5, 9, 6]
    assert r.feedback == "Sharp ending. Trim the opening."


def test_parse_ranking_handles_the_markdown_reply_that_crashed_before():
    raw = "**Scores:** [5]\n\n**Feedback:** The tweet mashes up AI hype with a quirky image."
    r = parse_ranking(raw)
    assert r.scores == [5]
    assert r.feedback.startswith("The tweet mashes up")


def test_parse_ranking_handles_out_of_ten_and_brackets():
    assert parse_ranking("scores: [4/10, 7/10, 6/10]\nfeedback: ok").scores == [4, 7, 6]


def test_parse_ranking_missing_feedback_gets_a_default():
    assert parse_ranking("SCORES: 7").feedback == "No specific feedback."


@pytest.mark.parametrize("bad", ["", None, "I think the second one is best", "SCORES: none\nFEEDBACK: x"])
def test_parse_ranking_rejects_garbage(bad):
    with pytest.raises(ValueError):
        parse_ranking(bad)


# ---------------------------------------------------------- enforce_limit
def test_enforce_limit_leaves_short_text_alone():
    assert enforce_limit("short") == "short"
    assert enforce_limit("x" * MAX_TWEET_CHARS) == "x" * MAX_TWEET_CHARS


def test_enforce_limit_trims_long_text():
    out = enforce_limit("word " * 100)
    assert len(out) <= MAX_TWEET_CHARS
    assert out.endswith("…")


def test_enforce_limit_hard_cuts_text_without_spaces():
    assert len(enforce_limit("a" * 1000)) == MAX_TWEET_CHARS


# ---------------------------------------------------------- style examples
def test_style_examples_ignore_comments_and_blanks(tmp_path):
    f = tmp_path / "ex.txt"
    f.write_text("# my favourites\n\nfirst funny tweet\n  second funny tweet  \n# end\n", encoding="utf-8")
    assert load_style_examples(f) == ["first funny tweet", "second funny tweet"]


def test_style_examples_missing_file_is_fine(tmp_path):
    assert load_style_examples(tmp_path / "nope.txt") == []


# --------------------------------------------------------------- workflow
def test_best_scoring_candidate_wins_and_stops(fake_llms):
    g, c, o = fake_llms([THREE], [ranking([5, 9, 6], "sharp ending")])
    result = engine.run_workflow("coffee")
    assert result["tweet"] == "tweet B"
    assert result["score"] == 9
    assert result["evaluation"] == "approved"
    assert result["iteration"] == 1
    assert (g.calls, c.calls, o.calls) == (1, 1, 0)
    assert result["tweet_history"] == ["tweet B"]
    assert result["feedback_history"] == ["Score 9/10 — sharp ending"]


def test_critic_sees_all_candidates_numbered(fake_llms):
    g, c, o = fake_llms([THREE], [ranking([9, 1, 1])])
    engine.run_workflow("coffee")
    assert "1. tweet A" in c.last_prompt and "3. tweet C" in c.last_prompt


def test_low_scores_trigger_optimize_then_approve(fake_llms):
    g, c, o = fake_llms(
        [THREE],
        [ranking([4, 5, 3], "too flat"), ranking([5, 9, 6], "much better")],
        ["new X\n---\nnew Y"],
    )
    result = engine.run_workflow("coffee")
    # round 2 pool = [previous best "tweet B", "new X", "new Y"]; critic picks index 2 = "new X"
    assert result["tweet"] == "new X"
    assert result["iteration"] == 2
    assert result["tweet_history"] == ["tweet B", "new X"]
    assert len(result["feedback_history"]) == len(result["score_history"]) == 2
    assert "tweet B" in o.last_prompt and "too flat" in o.last_prompt


def test_stall_stops_the_loop(fake_llms):
    # round 2 pool = [B, X, Y]; the critic picks B again (6 beats 5 and 3): nothing improved
    g, c, o = fake_llms([THREE], [ranking([4, 6, 3]), ranking([6, 5, 3])], ["new X\n---\nnew Y"])
    result = engine.run_workflow("coffee", max_iteration=5)
    assert result["iteration"] == 2
    assert c.calls == 2
    assert result["tweet"] == "tweet B"
    assert result["evaluation"] == "needs_improvement"


def test_returns_best_tweet_of_all_rounds_not_the_last_one(fake_llms):
    # round 1 best = B (6). Round 2 the critic is harsher: B scores 2, X wins with 5.
    g, c, o = fake_llms([THREE], [ranking([4, 6, 3]), ranking([2, 5, 3])], ["new X\n---\nnew Y"])
    result = engine.run_workflow("coffee", max_iteration=2)
    assert result["tweet_history"] == ["tweet B", "new X"]
    assert result["tweet"] == "tweet B"  # the 6, not the later 5
    assert result["score"] == 6
    assert result["feedback"].startswith("Score 6/10")


def test_stops_at_max_iteration(fake_llms):
    # a new winner every round, but never good enough to approve
    g, c, o = fake_llms([THREE], [ranking([3, 4, 3]), ranking([3, 5, 3]), ranking([3, 6])], ["new X\n---\nnew Y"])
    result = engine.run_workflow("coffee", max_iteration=3)
    assert result["iteration"] == 3
    assert c.calls == 3
    assert result["evaluation"] == "needs_improvement"
    assert len(result["tweet_history"]) == len(result["feedback_history"]) == len(result["score_history"]) == 3
    assert result["tweet"] == "new Y" and result["score"] == 6


def test_over_limit_candidates_never_reach_the_critic(fake_llms):
    too_long = "x" * (MAX_TWEET_CHARS + 50)
    g, c, o = fake_llms([f"{too_long}\n---\nshort one"], [ranking([9])])
    result = engine.run_workflow("coffee")
    assert too_long not in c.last_prompt
    assert result["tweet"] == "short one"


def test_all_over_limit_skips_critic_and_optimizes(fake_llms):
    too_long = "word " * 100
    g, c, o = fake_llms([f"{too_long}\n---\n{too_long} more"], [ranking([9, 1])], ["tiny joke"])
    result = engine.run_workflow("coffee")
    assert "Auto-rejected" in result["feedback_history"][0]
    assert c.calls == 1  # only the second round reached the critic
    assert len(result["tweet"]) <= MAX_TWEET_CHARS


def test_final_tweet_never_exceeds_limit(fake_llms):
    too_long = "word " * 200
    fake_llms([too_long], [ranking([9])], [too_long])
    result = engine.run_workflow("coffee", max_iteration=2)
    assert len(result["tweet"]) <= MAX_TWEET_CHARS


def test_scores_are_padded_and_clamped(fake_llms):
    fake_llms([THREE], [ranking([15])])  # too few, and out of range
    result = engine.run_workflow("coffee")
    assert result["tweet"] == "tweet A" and result["score"] == 10

    fake_llms([THREE], [ranking([0, 4, 2], "meh")])
    result = engine.run_workflow("coffee", max_iteration=1)
    assert result["tweet"] == "tweet B" and result["score"] == 4


def test_generator_output_is_cleaned(fake_llms):
    fake_llms(['Here is your tweet: "Monday is a hostage situation" #Monday'], [ranking([9])])
    assert engine.run_workflow("monday")["tweet"] == "Monday is a hostage situation"


def test_falls_back_to_writer_model_when_critic_fails(fake_llms):
    g, c, o = fake_llms([THREE], [RuntimeError("tool_use_failed")], fallback=[ranking([2, 9, 2], "ok")])
    result = engine.run_workflow("coffee")
    assert result["tweet"] == "tweet B"
    assert "unavailable" not in result["feedback"]


def test_unparseable_critic_reply_also_triggers_the_fallback(fake_llms):
    g, c, o = fake_llms([THREE], ["I like the second one best."], fallback=[ranking([2, 9, 2], "ok")])
    assert engine.run_workflow("coffee")["tweet"] == "tweet B"


def test_both_critics_failing_does_not_break_the_request(fake_llms):
    g, c, o = fake_llms([THREE], [RuntimeError("down")], fallback=[RuntimeError("down")])
    result = engine.run_workflow("coffee")
    assert result["tweet"] == "tweet A"
    assert result["evaluation"] == "approved"
    assert "unavailable" in result["feedback"]


def test_empty_generation_raises_a_clear_error(fake_llms):
    fake_llms([""], [ranking([9])])
    with pytest.raises(RuntimeError, match="usable tweet"):
        engine.run_workflow("coffee")


def test_single_candidate_mode(fake_llms, monkeypatch):
    g, c, o = fake_llms(["only tweet"], [ranking([9])])
    monkeypatch.setattr(engine, "NUM_CANDIDATES", 1)
    assert engine.run_workflow("coffee")["tweet"] == "only tweet"
    assert "separated by a line" not in g.last_prompt


def test_models_ready_false_without_models(monkeypatch):
    monkeypatch.setattr(engine, "generator_llm", None)
    assert engine.models_ready() is False


# ------------------------------------------------- rate limits and empty replies
RATE_SHORT = "Error code: 429 - Rate limit reached ... Please try again in 6.735s. Need more tokens?"
RATE_LONG = "Error code: 429 - Rate limit reached ... tokens per day ... Please try again in 6m14.975s."


@pytest.fixture
def slept(monkeypatch):
    calls = []
    monkeypatch.setattr(engine.time, "sleep", lambda s: calls.append(s))
    return calls


def test_rate_limit_wait_parsing():
    assert rate_limit_wait(Exception(RATE_SHORT)) == pytest.approx(6.735)
    assert rate_limit_wait(Exception(RATE_LONG)) == pytest.approx(374.975)
    assert rate_limit_wait(Exception("429 too many requests")) == 2.0
    assert rate_limit_wait(Exception("500 server error")) is None


def test_short_rate_limit_is_waited_out_and_retried(fake_llms, slept):
    g, c, o = fake_llms([Exception(RATE_SHORT), THREE], [ranking([5, 9, 6])])
    result = engine.run_workflow("coffee")
    assert result["tweet"] == "tweet B"
    assert g.calls == 2 and len(slept) == 1 and slept[0] > 6


def test_long_rate_limit_gives_a_friendly_error(fake_llms, slept):
    fake_llms([Exception(RATE_LONG)], [ranking([9])])
    with pytest.raises(RuntimeError, match="try again in about 7 min"):
        engine.run_workflow("coffee")
    assert slept == []  # did not sit there for 6 minutes


def test_gives_up_after_repeated_rate_limits(fake_llms, slept):
    g, c, o = fake_llms([Exception(RATE_SHORT)], [ranking([9])])
    with pytest.raises(Exception, match="429"):
        engine.run_workflow("coffee")
    assert g.calls == engine.MAX_LLM_RETRIES + 1


def test_non_rate_limit_errors_are_not_retried(fake_llms, slept):
    g, c, o = fake_llms([RuntimeError("500 boom")], [ranking([9])])
    with pytest.raises(RuntimeError, match="boom"):
        engine.run_workflow("coffee")
    assert g.calls == 1 and slept == []


def test_empty_reply_from_the_writer_is_retried_once(fake_llms):
    g, c, o = fake_llms(["", THREE], [ranking([5, 9, 6])])
    assert engine.run_workflow("coffee")["tweet"] == "tweet B"
    assert g.calls == 2


def test_critic_rate_limit_falls_back_instead_of_crashing(fake_llms, slept):
    g, c, o = fake_llms([THREE], [Exception(RATE_LONG)], fallback=[ranking([2, 9, 2])])
    assert engine.run_workflow("coffee")["tweet"] == "tweet B"


# ------------------------------------------- separators the model actually produces
def test_parse_splits_inline_dashes_like_the_cricket_case():
    raw = "Cricket fever is a six-hit dance. ---\n\nI told my boss I had cricket fever, he laughed."
    assert parse_candidates(raw) == [
        "Cricket fever is a six-hit dance.",
        "I told my boss I had cricket fever, he laughed.",
    ]


def test_parse_does_not_split_dashes_inside_words():
    assert parse_candidates("well---maybe not") == ["well---maybe not"]


def test_parse_falls_back_to_lines_when_model_ignores_the_separator():
    raw = "tweet one\ntweet two\ntweet three"
    assert parse_candidates(raw, split_lines=True) == ["tweet one", "tweet two", "tweet three"]
    assert parse_candidates(raw) == [raw]  # without the flag it stays one tweet


def test_parse_line_fallback_handles_blank_lines_intro_and_labels():
    raw = "Here are 3 tweets:\n\nTweet 1: first joke\n\nTweet 2: second joke"
    assert parse_candidates(raw, split_lines=True) == ["first joke", "second joke"]


def test_pipeline_survives_a_writer_that_uses_plain_lines(fake_llms):
    g, c, o = fake_llms(["tweet A\ntweet B\ntweet C"], [ranking([5, 9, 6])])
    result = engine.run_workflow("coffee")
    assert result["tweet"] == "tweet B"
    assert "1. tweet A" in c.last_prompt and "3. tweet C" in c.last_prompt
    assert "Auto-rejected" not in result["feedback"]
