"""Route tests. run_workflow is faked, so no API key or network is needed."""
import pytest

import app as app_module
import tweet_engine


@pytest.fixture
def client():
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


@pytest.fixture
def ready(monkeypatch):
    monkeypatch.setattr(tweet_engine, "models_ready", lambda: True)


def fake_result(topic, max_iteration=5):
    return {
        "tweet": "final tweet",
        "feedback": "nice",
        "evaluation": "approved",
        "iteration": 2,
        "tweet_history": ["draft", "final tweet"],
        "feedback_history": ["flat", "nice"],
    }


def test_empty_topic_rejected(client, ready):
    assert client.post("/generate-tweet", json={"topic": "   "}).status_code == 400


def test_missing_json_rejected(client, ready):
    assert client.post("/generate-tweet", data="not json").status_code == 400


def test_non_string_topic_rejected(client, ready):
    assert client.post("/generate-tweet", json={"topic": 123}).status_code == 400


def test_overlong_topic_rejected(client, ready):
    resp = client.post("/generate-tweet", json={"topic": "a" * 500})
    assert resp.status_code == 400
    assert "too long" in resp.get_json()["error"]


def test_models_unavailable_returns_500(client, monkeypatch):
    monkeypatch.setattr(tweet_engine, "models_ready", lambda: False)
    resp = client.post("/generate-tweet", json={"topic": "coffee"})
    assert resp.status_code == 500
    assert "GROQ_API_KEY" in resp.get_json()["error"]


def test_success_returns_history_and_counts(client, ready, monkeypatch):
    monkeypatch.setattr(tweet_engine, "run_workflow", fake_result)
    resp = client.post("/generate-tweet", json={"topic": "  coffee  "})
    body = resp.get_json()
    assert resp.status_code == 200
    assert body["tweet"] == "final tweet"
    assert body["char_count"] == len("final tweet")
    assert body["iterations"] == 2
    assert body["tweet_history"] == ["draft", "final tweet"]
    assert body["feedback_history"] == ["flat", "nice"]


def test_workflow_error_returns_500(client, ready, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("groq down")

    monkeypatch.setattr(tweet_engine, "run_workflow", boom)
    resp = client.post("/generate-tweet", json={"topic": "coffee"})
    assert resp.status_code == 500


def test_check_auth_defaults_to_false(client):
    assert client.get("/check-auth").get_json() == {"authenticated": False}


def test_post_tweet_requires_login(client):
    assert client.post("/post-tweet", json={"tweet": "hi"}).status_code == 401
