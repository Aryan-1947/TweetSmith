"""AI tweet pipeline for TweetSmith.

Flow:
    generate N candidate tweets
      -> critic scores every candidate 1-10 and picks the best
      -> approved if best score >= APPROVAL_SCORE (or the iteration cap is hit)
      -> otherwise optimize: write N rewrites of the best tweet, score again

Kept separate from app.py so it can be tested without Flask or X OAuth.
"""
import logging
import operator
import os
import re
import time
from pathlib import Path
from typing import Annotated, Literal, NamedTuple, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

load_dotenv(override=True)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------- settings
# Everything below can be overridden from .env
MAX_TWEET_CHARS = 280
MAX_TOPIC_CHARS = 200
DEFAULT_MAX_ITERATION = 4

MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")            # writes tweets
CRITIC_MODEL = os.getenv("GROQ_CRITIC_MODEL", "openai/gpt-oss-120b")  # judges tweets
GENERATOR_TEMPERATURE = float(os.getenv("GENERATOR_TEMPERATURE", "1"))
OPTIMIZER_TEMPERATURE = float(os.getenv("OPTIMIZER_TEMPERATURE", "0.8"))
NUM_CANDIDATES = max(1, int(os.getenv("NUM_CANDIDATES", "3")))
APPROVAL_SCORE = int(os.getenv("APPROVAL_SCORE", "7"))
STRIP_TRAILING_HASHTAGS = os.getenv("STRIP_TRAILING_HASHTAGS", "1") != "0"
STYLE_EXAMPLES_FILE = Path(__file__).with_name("style_examples.txt")

# ------------------------------------------------------------------ models
generator_llm = None
optimizer_llm = None
critic_llm = None           # the big critic model
fallback_critic_llm = None  # the writer model, used as critic if the critic model fails


class Ranking(NamedTuple):
    scores: list
    feedback: str


def init_models() -> bool:
    """Create the Groq models. Returns False (instead of crashing) if that fails."""
    global generator_llm, optimizer_llm, critic_llm, fallback_critic_llm
    try:
        generator_llm = ChatGroq(model_name=MODEL_NAME, temperature=GENERATOR_TEMPERATURE)
        optimizer_llm = ChatGroq(model_name=MODEL_NAME, temperature=OPTIMIZER_TEMPERATURE)
        critic_llm = ChatGroq(model_name=CRITIC_MODEL, temperature=0)
        fallback_critic_llm = ChatGroq(model_name=MODEL_NAME, temperature=0)
        return True
    except Exception as e:  # missing GROQ_API_KEY, bad config, ...
        print(f"⚠️  WARNING: Could not initialize Groq models: {e}")
        print("   Please ensure GROQ_API_KEY is set in your .env file")
        generator_llm = optimizer_llm = critic_llm = fallback_critic_llm = None
        return False


def models_ready() -> bool:
    return all([generator_llm, optimizer_llm, critic_llm])


# ------------------------------------------------------- calling the models
MAX_LLM_RETRIES = 3
MAX_RATE_LIMIT_WAIT = 20  # seconds; longer waits (e.g. a daily limit) are reported instead of waited out


def rate_limit_wait(error):
    """Seconds Groq asks us to wait if `error` is a rate-limit (429) error, else None."""
    msg = str(error)
    if "429" not in msg and "rate_limit" not in msg.lower():
        return None
    m = re.search(r"try again in\s+(?:(\d+)m)?\s*([\d.]+)s", msg)
    if not m:
        return 2.0
    return int(m.group(1) or 0) * 60 + float(m.group(2))


def _invoke(llm, messages, retries: int = MAX_LLM_RETRIES):
    """llm.invoke() that waits out short rate limits instead of failing the request."""
    for attempt in range(retries + 1):
        try:
            return llm.invoke(messages)
        except Exception as e:
            wait = rate_limit_wait(e)
            if wait is None or attempt == retries:
                raise
            if wait > MAX_RATE_LIMIT_WAIT:
                minutes = int(-(-wait // 60))
                raise RuntimeError(
                    f"Groq rate limit reached. Please try again in about {minutes} min."
                ) from e
            logger.warning("Rate limited, retrying in %.1fs", wait)
            time.sleep(wait + 0.5)


# ----------------------------------------------------------- text cleaning
_PREAMBLE = re.compile(
    r"^\s*(?:here(?:'s|’s| is| are)|sure|okay|certainly)"
    r"[^:\n]{0,60}\b(?:tweets?|versions?|rewrites?|takes?|options?|candidates?)\b[^:\n]{0,20}:\s*",
    re.IGNORECASE,
)
_TRAILING_NOTE = re.compile(
    r"\n\s*\n\s*(?:\(|note\b|this tweet\b|explanation\b).*\Z",
    re.IGNORECASE | re.DOTALL,
)
_TRAILING_HASHTAGS = re.compile(r"(?:\s+#\w+)+\s*$")
_QUOTE_PAIRS = {'"': '"', "“": "”", "'": "'", "‘": "’"}
_CANDIDATE_SPLIT = re.compile(r"(?<![\w-])-{3,}(?![\w-])")
_LABEL = re.compile(r"^\s*(?:tweet|option|candidate|version)\s*\d+\s*[:.)-]\s*", re.IGNORECASE)


def clean_tweet(text) -> str:
    """Strip the usual LLM noise: intros, wrapping quotes, trailing notes and trailing hashtags."""
    if text is None:
        return ""
    text = str(text).strip()
    text = _PREAMBLE.sub("", text, count=1)
    text = _TRAILING_NOTE.sub("", text).strip()
    if STRIP_TRAILING_HASHTAGS:  # first, so '"joke" #tag' still shows its wrapping quotes
        text = _TRAILING_HASHTAGS.sub("", text)

    # Remove wrapping quotes, but only if the same quote doesn't also appear inside
    while len(text) >= 2 and _QUOTE_PAIRS.get(text[0]) == text[-1]:
        inner = text[1:-1]
        if text[0] in inner or text[-1] in inner:
            break
        text = inner.strip()

    if STRIP_TRAILING_HASHTAGS:
        text = _TRAILING_HASHTAGS.sub("", text)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_candidates(text, limit=None, split_lines=False) -> list[str]:
    """Split model output into cleaned, de-duplicated tweets.

    Splits on '---'. With split_lines=True, if the model ignored that and just wrote the
    tweets one per line (or one per paragraph), each line becomes a candidate instead of
    all of them being glued into one over-long tweet.
    """
    chunks = _CANDIDATE_SPLIT.split(str(text or ""))
    non_empty = [c for c in chunks if c.strip()]
    if split_lines and len(non_empty) == 1:
        lines = [ln for ln in non_empty[0].splitlines() if ln.strip()]
        if len(lines) > 1:
            chunks = lines

    seen, tweets = set(), []
    for chunk in chunks:
        chunk = _LABEL.sub("", chunk.strip())
        chunk = re.sub(r"^\s*(?:\d+[.)]|[-*•])\s+", "", chunk)  # drop "1." / "- " markers
        tweet = clean_tweet(chunk)
        if tweet and tweet.lower() not in seen:
            seen.add(tweet.lower())
            tweets.append(tweet)
    return tweets[:limit] if limit else tweets


def parse_ranking(text) -> Ranking:
    """Read 'SCORES: 5, 9, 6' and 'FEEDBACK: ...' from plain text.

    Plain text is used instead of tool-calling because the critic model sometimes
    answers in text and Groq then rejects the whole request. Markdown is tolerated.
    """
    text = str(text or "")
    m = re.search(r"scores?[\s*_]*[:=][\s*_]*(.*)", text, re.IGNORECASE)
    if not m:
        raise ValueError("critic reply had no SCORES line")
    line = re.sub(r"[\[\]()*_]", " ", m.group(1))
    scores = []
    for token in re.split(r"[,;\s]+", line.strip()):
        num = re.match(r"(\d+)", token)
        if num:
            scores.append(int(num.group(1)))
    if not scores:
        raise ValueError("critic reply had no numbers in its SCORES line")
    fb = re.search(r"feedback[\s*_]*[:=][\s*_]*(.+)", text, re.IGNORECASE | re.DOTALL)
    feedback = fb.group(1).strip() if fb else ""
    return Ranking(scores=scores, feedback=feedback or "No specific feedback.")


def enforce_limit(text: str, limit: int = MAX_TWEET_CHARS) -> str:
    """Last-resort safety net: trim to `limit` characters at a word boundary, ending with …

    Note: len() is an approximation. X counts URLs as 23 chars and some emoji as 2.
    """
    if len(text) <= limit:
        return text
    cut = text[: limit - 1]
    space = cut.rfind(" ")
    if space >= int(limit * 0.6):
        cut = cut[:space]
    return cut.rstrip(" ,;:-–—") + "…"


def load_style_examples(path: Path = None, limit: int = 8) -> list[str]:
    """Read your favourite tweets from style_examples.txt (one per line, '#' lines ignored)."""
    path = path or STYLE_EXAMPLES_FILE
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")][:limit]


def _examples_block() -> str:
    examples = load_style_examples()
    if not examples:
        return ""
    bullets = "\n".join(f"- {e}" for e in examples)
    return f"\nExamples of the quality bar (match their energy, do NOT copy them):\n{bullets}\n"


# ------------------------------------------------------------------- state
class TweetState(TypedDict):
    topic: str
    candidates: list[str]
    tweet: str
    score: int
    stalled: bool
    evaluation: Literal["approved", "needs_improvement"]
    feedback: str
    iteration: int
    max_iteration: int
    tweet_history: Annotated[list[str], operator.add]     # best tweet of each round
    feedback_history: Annotated[list[str], operator.add]  # critic feedback for each round
    score_history: Annotated[list[int], operator.add]     # critic score for each round


# ------------------------------------------------------------------- nodes
def generate_tweet(state: TweetState):
    n = NUM_CANDIDATES
    how_many = f"{n} different" if n > 1 else "a"
    messages = [
        SystemMessage(content="You are a funny and clever Twitter/X influencer."),
        HumanMessage(content=f"""
Write {how_many} short, original, and hilarious tweet{'s' if n > 1 else ''} on the topic: "{state['topic']}".

Rules:
- One or two short sentences, ideally under 200 characters. Never over {MAX_TWEET_CHARS}.
- Do NOT use question-answer or setup-punchline format.
- Use observational humor, irony, sarcasm, cultural references, meme logic, or relatable takes.
- Use simple, day to day English. No run-on sentences.
- No hashtags. No emoji spam.
- Avoid clichés and the first joke everyone thinks of. Be specific and surprising.
{'- Each tweet must take a genuinely different angle, not a rewording.' if n > 1 else ''}
{_examples_block()}
Output ONLY the tweet{'s, separated by a line containing only ---' if n > 1 else ''}. No numbering, no intro, no explanation.
""")
    ]
    candidates = []
    for _ in range(2):  # the model occasionally returns an empty reply: try once more
        raw = _invoke(generator_llm, messages).content
        candidates = parse_candidates(raw, NUM_CANDIDATES, split_lines=n > 1)
        if candidates:
            break
        logger.warning("Writer returned nothing usable. Raw reply: %r", str(raw)[:400])
    if not candidates:
        raise RuntimeError("The model did not return a usable tweet.")
    return {"candidates": candidates}


def _clamp_scores(scores, count: int) -> list[int]:
    """Make sure there is exactly one score in 1..10 per candidate."""
    clean = []
    for s in list(scores or [])[:count]:
        try:
            clean.append(min(10, max(1, int(s))))
        except (TypeError, ValueError):
            clean.append(1)
    return clean + [1] * (count - len(clean))


def _verdict(tweet, score, evaluation, feedback, stalled=False):
    return {
        "tweet": tweet,
        "score": score,
        "stalled": stalled,
        "evaluation": evaluation,
        "feedback": feedback,
        "tweet_history": [tweet],
        "feedback_history": [feedback],
        "score_history": [score],
    }


def evaluate_tweet(state: TweetState):
    pool = [c for c in state.get("candidates", []) if c]

    # Hard rule checked in code: over-long candidates never reach the critic.
    fitting = [c for c in pool if len(c) <= MAX_TWEET_CHARS]
    if not fitting:
        shortest = enforce_limit(min(pool, key=len)) if pool else ""
        feedback = (
            f"Auto-rejected: every candidate was over {MAX_TWEET_CHARS} characters. "
            "Shorten it while keeping the punchline."
        )
        return _verdict(shortest, 0, "needs_improvement", feedback)

    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(fitting, 1))
    messages = [
        SystemMessage(content="You are a ruthless, no-laugh-given Twitter comedy critic."),
        HumanMessage(content=f"""
Topic: "{state['topic']}"

Score each candidate tweet from 1 to 10.

Be strict. Most tweets are mediocre: an average, forgettable joke is a 4 or 5.
A decent, solid joke is a 6 or 7. Give 8 or more ONLY if it would genuinely make a
stranger laugh and want to share it.

Penalize: question-answer or setup-punchline format, clichés and overused jokes,
explaining the joke, hashtags, emoji spam, run-on sentences, anything that sounds like
an ad or a motivational poster, a weak last line, anything over {MAX_TWEET_CHARS} characters.
Reward: a specific and surprising observation, tight wording, a sharp final line.
{_examples_block()}
Candidates:
{numbered}

Reply in EXACTLY this format and nothing else:
SCORES: <{len(fitting)} integers from 1 to 10 separated by commas, one per candidate, in order>
FEEDBACK: <2-3 sentences about the BEST candidate: what works, and the single most important thing to improve. Describe the tweet itself; do not refer to candidate numbers.>
""")
    ]

    ranking = None
    for name, critic in ((CRITIC_MODEL, critic_llm), (MODEL_NAME, fallback_critic_llm)):
        if critic is None:
            continue
        try:
            ranking = parse_ranking(_invoke(critic, messages).content)
            break
        except Exception as e:
            logger.warning("Critic %s failed: %s", name, str(e)[:300])

    if ranking is None:
        # Never fail the whole request because the critic is down: keep the first candidate.
        feedback = "Critic was unavailable, so this tweet was accepted without review."
        return _verdict(fitting[0], 0, "approved", feedback)

    scores = _clamp_scores(ranking.scores, len(fitting))
    best_index = max(range(len(fitting)), key=lambda i: scores[i])  # first one wins ties
    best, score = fitting[best_index], scores[best_index]
    feedback = f"Score {score}/10 — {ranking.feedback}"
    evaluation = "approved" if score >= APPROVAL_SCORE else "needs_improvement"
    # If the same tweet wins again, the rewrites could not beat it: stop wasting calls.
    stalled = bool(state.get("tweet")) and best == state["tweet"]
    return _verdict(best, score, evaluation, feedback, stalled)


def optimize_tweet(state: TweetState):
    n = NUM_CANDIDATES
    best = state["tweet"]
    messages = [
        SystemMessage(content="You punch up tweets for humor and shareability."),
        HumanMessage(content=f"""
Topic: "{state['topic']}"
Current best tweet: {best}
Critic's verdict: {state['feedback']}

Write {n} improved rewrite{'s' if n > 1 else ''} that fix the critic's main complaint.
Keep it tight: one or two short sentences, ideally under 200 characters, never over {MAX_TWEET_CHARS}.
No hashtags and no question-answer format.
{'Take a different angle in each one; a fresh joke beats a polished weak one.' if n > 1 else ''}
{_examples_block()}
Output ONLY the tweet{'s, separated by a line containing only ---' if n > 1 else ''}. No numbering, no intro, no explanation.
""")
    ]
    variants = parse_candidates(_invoke(optimizer_llm, messages).content, NUM_CANDIDATES, split_lines=n > 1)
    # The previous best stays in the pool, so a round can never make things worse.
    candidates = [best] + [v for v in variants if v.lower() != best.lower()]
    return {"candidates": candidates, "iteration": state["iteration"] + 1}


def route_evaluation(state: TweetState):
    if (
        state["evaluation"] == "approved"
        or state["iteration"] >= state["max_iteration"]
        or state.get("stalled")
    ):
        return "approved"
    return "needs_improvement"


# --------------------------------------------------------------- workflow
def build_workflow():
    graph = StateGraph(TweetState)
    graph.add_node("generate", generate_tweet)
    graph.add_node("evaluate", evaluate_tweet)
    graph.add_node("optimize", optimize_tweet)
    graph.add_edge(START, "generate")
    graph.add_edge("generate", "evaluate")
    graph.add_conditional_edges(
        "evaluate", route_evaluation, {"approved": END, "needs_improvement": "optimize"}
    )
    graph.add_edge("optimize", "evaluate")
    return graph.compile()


# Compiled once at import time instead of on every request.
# Nodes look up the LLMs when they run, so this works even before init_models().
WORKFLOW = build_workflow()


def run_workflow(topic: str, max_iteration: int = DEFAULT_MAX_ITERATION):
    state = {
        "topic": topic,
        "candidates": [],
        "tweet": "",
        "score": 0,
        "stalled": False,
        "iteration": 1,
        "max_iteration": max_iteration,
        "tweet_history": [],
        "feedback_history": [],
        "score_history": [],
    }
    result = WORKFLOW.invoke(state)

    # Return the best tweet of ALL rounds (latest wins ties), not just the last round's winner.
    tweets = list(result.get("tweet_history", []))
    scores = list(result.get("score_history", []))
    feedbacks = list(result.get("feedback_history", []))
    if tweets and len(scores) == len(tweets):
        best = max(range(len(tweets)), key=lambda i: (scores[i], i))
        if best != len(tweets) - 1:
            result["evaluation"] = "approved" if scores[best] >= APPROVAL_SCORE else "needs_improvement"
        result["tweet"] = tweets[best]
        result["score"] = scores[best]
        if best < len(feedbacks):
            result["feedback"] = feedbacks[best]

    # Guarantee the final tweet fits, even in the worst case.
    result["tweet"] = enforce_limit(result["tweet"])
    return result
