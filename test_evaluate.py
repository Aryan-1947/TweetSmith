"""Tests for the benchmark helpers. No API calls."""
import json

import evaluate


def fake_run(topic, max_iteration=5):
    if topic == "boom":
        raise RuntimeError("groq down")
    return {"tweet": f"joke about {topic}", "iteration": 2, "score": 8,
            "evaluation": "approved", "feedback": "Score 8/10 — nice"}


def fake_judge(topic, tweet):
    return 7, "ok"


def test_run_benchmark_collects_results_and_survives_errors():
    logs = []
    results = evaluate.run_benchmark(["a", "boom", "b"], fake_run, fake_judge, log=logs.append)
    assert [r["topic"] for r in results] == ["a", "boom", "b"]
    assert results[1]["error"] == "groq down"
    assert results[0]["judge_score"] == 7 and results[0]["length"] == len("joke about a")
    assert len(logs) == 3


def test_judge_failure_is_recorded_not_fatal():
    def bad_judge(topic, tweet):
        raise RuntimeError("judge 404")

    results = evaluate.run_benchmark(["a"], fake_run, bad_judge, log=lambda s: None)
    assert results[0]["judge_score"] is None and "judge 404" in results[0]["judge_error"]


def test_summarize():
    results = evaluate.run_benchmark(["a", "boom", "b"], fake_run, fake_judge, log=lambda s: None)
    s = evaluate.summarize(results)
    assert (s["topics"], s["succeeded"], s["failed"]) == (3, 2, 1)
    assert s["avg_judge_score"] == 7 and s["approval_rate_pct"] == 100.0 and s["avg_iterations"] == 2


def test_summarize_with_nothing_succeeding():
    s = evaluate.summarize([{"topic": "x", "error": "bad"}])
    assert s["succeeded"] == 0 and s["avg_judge_score"] is None and s["approval_rate_pct"] is None


def test_critic_unavailable_is_counted():
    def run(topic, max_iteration=5):
        return {"tweet": "t", "iteration": 1, "score": 0, "evaluation": "approved",
                "feedback": "Critic was unavailable, so this tweet was accepted without review."}

    s = evaluate.summarize(evaluate.run_benchmark(["a"], run, fake_judge, log=lambda s: None))
    assert s["critic_unavailable"] == 1


def test_compare_prints_both_runs_and_deltas():
    a = {"label": "baseline", "summary": {"avg_judge_score": 4.0, "failed": 0}}
    b = {"label": "improved", "summary": {"avg_judge_score": 6.5, "failed": 0}}
    text = evaluate.format_comparison(a, b)
    assert "baseline" in text and "improved" in text and "+2.50" in text


def test_load_run_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "RESULTS_DIR", tmp_path)
    (tmp_path / "x.json").write_text(json.dumps({"label": "x", "summary": {}}), encoding="utf-8")
    assert evaluate.load_run("x")["label"] == "x"


def test_parse_args_defaults():
    args = evaluate.parse_args(["--label", "v2", "--candidates", "1"])
    assert args.label == "v2" and args.candidates == 1 and args.topics == len(evaluate.TOPICS)


def test_parse_judge_standard_and_markdown():
    assert evaluate.parse_judge("SCORE: 7\nREASON: Tight and specific.") == (7, "Tight and specific.")
    assert evaluate.parse_judge("**Score:** 8/10\n**Reason:** Sharp ending.") == (8, "Sharp ending.")


def test_parse_judge_clamps_and_rejects_garbage():
    assert evaluate.parse_judge("SCORE: 42")[0] == 10
    assert evaluate.parse_judge("score: 0")[0] == 1
    import pytest
    with pytest.raises(ValueError):
        evaluate.parse_judge("it was fine")
