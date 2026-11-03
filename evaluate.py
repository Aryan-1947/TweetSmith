"""Quality benchmark for the TweetSmith pipeline.

Runs 15 fixed topics through the pipeline, then has a *separate* judge model with a
fixed rubric score each final tweet (so the pipeline's own critic can't grade itself).

Usage (from the project folder, with the virtual environment active):

    python evaluate.py --label improved
    python evaluate.py --label baseline --candidates 1 --critic-model openai/gpt-oss-20b
    python evaluate.py --compare baseline improved

Results are saved to eval_results/<label>.json. Needs GROQ_API_KEY; uses no X credentials.
Note: scores come from an LLM judge. Use them to compare runs, and still read the tweets yourself.
"""
import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

TOPICS = [
    "Monday mornings",
    "Coffee addiction",
    "Work from home",
    "Food delivery apps",
    "Traffic jams",
    "Social media",
    "Gym resolutions",
    "Group chats",
    "Online shopping",
    "Weather complaints",
    "Cooking disasters",
    "Phone battery at 1%",
    "Wedding season",
    "Cricket fever",
    "Exam stress",
]

RESULTS_DIR = Path("eval_results")
DEFAULT_JUDGE_MODEL = "openai/gpt-oss-120b"

JUDGE_PROMPT = """You are a strict judge of tweet comedy. Rate this tweet from 1 to 10.

Topic: "{topic}"
Tweet: "{tweet}"

Scale: 1-3 = not funny or broken, 4-5 = average and forgettable, 6-7 = decent,
8-9 = genuinely funny and shareable, 10 = exceptional.
Penalize: question-answer or setup-punchline format, clichés, explaining the joke,
hashtags, emoji spam, ad-like or motivational tone, a weak last line.
Reward: a specific surprising observation, tight wording, a sharp final line.

Reply in EXACTLY this format and nothing else:
SCORE: <integer from 1 to 10>
REASON: <one sentence>"""


# ------------------------------------------------------------- core logic
def summarize(results: list[dict]) -> dict:
    ok = [r for r in results if not r.get("error")]

    def avg(key):
        vals = [r[key] for r in ok if r.get(key) is not None]
        return round(statistics.mean(vals), 2) if vals else None

    return {
        "topics": len(results),
        "succeeded": len(ok),
        "failed": len(results) - len(ok),
        "avg_judge_score": avg("judge_score"),
        "avg_critic_score": avg("critic_score"),
        "approval_rate_pct": round(100 * sum(1 for r in ok if r.get("approved")) / len(ok), 1) if ok else None,
        "avg_iterations": avg("iterations"),
        "avg_length": avg("length"),
        "avg_seconds": avg("seconds"),
        "critic_unavailable": sum(1 for r in ok if r.get("critic_unavailable")),
    }


def run_benchmark(topics, run_workflow, judge, max_iteration=4, log=print) -> list[dict]:
    """`run_workflow(topic, max_iteration=...)` and `judge(topic, tweet) -> (score, reason)` are injected."""
    results = []
    for i, topic in enumerate(topics, 1):
        entry = {"topic": topic}
        start = time.time()
        try:
            res = run_workflow(topic, max_iteration=max_iteration)
            tweet = res["tweet"]
            entry.update(
                tweet=tweet,
                length=len(tweet),
                iterations=res.get("iteration", 1),
                critic_score=res.get("score"),
                approved=res.get("evaluation") == "approved",
                critic_unavailable="unavailable" in res.get("feedback", ""),
            )
        except Exception as e:
            entry["error"] = str(e)
        entry["seconds"] = round(time.time() - start, 1)

        if "tweet" in entry:
            try:
                score, reason = judge(topic, entry["tweet"])
                entry["judge_score"], entry["judge_reason"] = score, reason
            except Exception as e:
                entry["judge_score"], entry["judge_error"] = None, str(e)

        results.append(entry)
        if "error" in entry:
            log(f"[{i}/{len(topics)}] {topic}: ERROR {entry['error']}")
        else:
            log(
                f"[{i}/{len(topics)}] {topic}: judge={entry.get('judge_score')} "
                f"critic={entry.get('critic_score')} rounds={entry['iterations']} ({entry['seconds']}s)\n"
                f"      {entry['tweet']}"
            )
    return results


def format_comparison(a: dict, b: dict) -> str:
    """Side-by-side summary of two saved result files."""
    rows = [
        ("Avg judge score (1-10)", "avg_judge_score", True),
        ("Avg critic score (1-10)", "avg_critic_score", True),
        ("Approval rate %", "approval_rate_pct", True),
        ("Avg rounds", "avg_iterations", False),
        ("Avg length (chars)", "avg_length", False),
        ("Avg seconds / tweet", "avg_seconds", False),
        ("Critic unavailable", "critic_unavailable", False),
        ("Failed topics", "failed", False),
    ]
    sa, sb = a["summary"], b["summary"]
    name_a, name_b = a["label"], b["label"]
    lines = [f"{'Metric':<26}{name_a:>14}{name_b:>14}{'change':>10}", "-" * 64]
    for title, key, _ in rows:
        va, vb = sa.get(key), sb.get(key)
        delta = f"{vb - va:+.2f}" if isinstance(va, (int, float)) and isinstance(vb, (int, float)) else "-"
        lines.append(f"{title:<26}{str(va):>14}{str(vb):>14}{delta:>10}")
    return "\n".join(lines)


# --------------------------------------------------------------------- CLI
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Benchmark the TweetSmith tweet pipeline.")
    p.add_argument("--label", default="run", help="name for this run, e.g. baseline or improved")
    p.add_argument("--topics", type=int, default=len(TOPICS), help=f"how many topics to use (max {len(TOPICS)})")
    p.add_argument("--max-iteration", type=int, default=4)
    p.add_argument("--candidates", type=int, help="override NUM_CANDIDATES for this run")
    p.add_argument("--writer-model", help="override GROQ_MODEL for this run")
    p.add_argument("--critic-model", help="override GROQ_CRITIC_MODEL for this run")
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="independent judge (keep it fixed between runs)")
    p.add_argument("--compare", nargs=2, metavar=("LABEL_A", "LABEL_B"), help="compare two saved runs")
    return p.parse_args(argv)


def load_run(label: str) -> dict:
    path = RESULTS_DIR / f"{label}.json"
    if not path.exists():
        sys.exit(f"No saved results at {path}. Run: python evaluate.py --label {label}")
    return json.loads(path.read_text(encoding="utf-8"))


def parse_judge(text) -> tuple:
    """Read 'SCORE: 7' and 'REASON: ...' from plain text (markdown tolerated)."""
    text = str(text or "")
    m = re.search(r"score[\s*_]*[:=][\s*_]*\[?(\d+)", text, re.IGNORECASE)
    if not m:
        raise ValueError("judge reply had no SCORE line")
    reason = re.search(r"reason[\s*_]*[:=][\s*_]*(.+)", text, re.IGNORECASE | re.DOTALL)
    return min(10, max(1, int(m.group(1)))), (reason.group(1).strip() if reason else "")


def make_judge(model_name: str):
    from langchain_groq import ChatGroq
    from tweet_engine import _invoke

    llm = ChatGroq(model_name=model_name, temperature=0)

    def judge(topic, tweet):
        last_error = None
        for _ in range(2):  # one retry if the reply can't be parsed
            try:
                return parse_judge(_invoke(llm, JUDGE_PROMPT.format(topic=topic, tweet=tweet)).content)
            except Exception as e:
                last_error = e
        raise last_error

    return judge


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # emoji-safe on Windows
    args = parse_args(argv)

    if args.compare:
        print(format_comparison(load_run(args.compare[0]), load_run(args.compare[1])))
        return 0

    # These settings are read when tweet_engine is imported, so set them first.
    if args.candidates:
        os.environ["NUM_CANDIDATES"] = str(args.candidates)
    if args.writer_model:
        os.environ["GROQ_MODEL"] = args.writer_model
    if args.critic_model:
        os.environ["GROQ_CRITIC_MODEL"] = args.critic_model

    import tweet_engine as engine

    if not engine.init_models():
        print("Could not start the models. Check GROQ_API_KEY in your .env file.")
        return 1

    config = {
        "writer_model": engine.MODEL_NAME,
        "critic_model": engine.CRITIC_MODEL,
        "judge_model": args.judge_model,
        "num_candidates": engine.NUM_CANDIDATES,
        "approval_score": engine.APPROVAL_SCORE,
        "generator_temperature": engine.GENERATOR_TEMPERATURE,
        "max_iteration": args.max_iteration,
    }
    print(f"Running '{args.label}' with {config}\n")

    topics = TOPICS[: max(1, min(args.topics, len(TOPICS)))]
    results = run_benchmark(topics, engine.run_workflow, make_judge(args.judge_model), args.max_iteration)
    summary = summarize(results)

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{args.label}.json"
    out_path.write_text(
        json.dumps({"label": args.label, "config": config, "summary": summary, "results": results},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== Summary ===")
    for key, value in summary.items():
        print(f"{key:<22} {value}")
    if summary["critic_unavailable"]:
        print("\n⚠️  The critic failed on some topics. Check GROQ_CRITIC_MODEL (or try openai/gpt-oss-20b).")
    print(f"\nSaved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
