import json

import httpx
import pytest

from metacog.judge import (
    DEFAULT_CHOOSE_INSTRUCTIONS,
    FINISHED_QUESTIONS,
    JudgeError,
    SystemOneJudge,
)


def _choice_response(labels, choice="B", conf=0.8):
    probs = {label: 0.0 for label in labels}
    probs[choice] = 1.0
    return {
        "model": "reflex-latest",
        "answers": {
            "best_path": {
                "type": "choice",
                "choice": choice,
                "probabilities": probs,
                "confidence": conf,
            }
        },
        "usage": {"input_tokens": 10},
    }


def _judge(handler, **kw):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return SystemOneJudge(client=client, **kw)


def test_choose_request_body_and_parsing():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=_choice_response(["A", "B", "C"]))

    judge = _judge(handler, base_url="http://localhost:8008", model="reflex-latest", permutations=2)
    v = judge.choose("the problem", ["path one", "path two", "path three"])

    assert seen["url"] == "http://localhost:8008/v1/systemone"
    body = seen["body"]
    assert body["model"] == "reflex-latest"
    assert body["state"]["problem"] == "the problem"
    assert body["state"]["paths"] == {"A": "path one", "B": "path two", "C": "path three"}
    q = body["questions"]["best_path"]
    assert q["type"] == "choice"
    assert q["instructions"] == DEFAULT_CHOOSE_INSTRUCTIONS
    assert q["criteria"] == {"A": "path A", "B": "path B", "C": "path C"}
    assert body["permutations"] == 2
    assert seen["auth"] is None  # reflex without key: no auth header

    assert v.choice == 1
    assert v.probabilities == [0.0, 1.0, 0.0]
    assert v.confidence == 0.8


def test_permutations_omitted_when_none():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_choice_response(["A", "B"], "A"))

    judge = _judge(handler, base_url="http://localhost:8008")
    judge.choose("p", ["x", "y"])
    assert "permutations" not in seen["body"]


def test_jev_sends_bearer_auth():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=_choice_response(["A", "B"], "A"))

    judge = _judge(handler, api_key="sk-test")
    judge.choose("p", ["x", "y"])
    assert seen["auth"] == "Bearer sk-test"


def test_tail_truncation_keeps_last_chars():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_choice_response(["A", "B"], "A"))

    judge = _judge(handler, base_url="http://localhost:8008", max_chars_per_path=10)
    long_path = "0123456789" + "TAIL-END!!"  # >10 chars
    judge.choose("p", [long_path, "short"])
    path_a = seen["body"]["state"]["paths"]["A"]
    assert path_a == "…TAIL-END!!"
    assert seen["body"]["state"]["paths"]["B"] == "short"


def test_more_than_26_candidates_raises():
    judge = _judge(lambda r: httpx.Response(200), base_url="http://localhost:8008")
    with pytest.raises(ValueError):
        judge.choose("p", ["x"] * 27)


def test_http_400_raises_judge_error_with_body():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "bad request body"})

    judge = _judge(handler, base_url="http://localhost:8008")
    with pytest.raises(JudgeError, match="bad request body"):
        judge.choose("p", ["x", "y"])


def test_5xx_retried_once_then_raises():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500, text="boom")

    judge = _judge(handler, base_url="http://localhost:8008")
    with pytest.raises(JudgeError):
        judge.choose("p", ["x", "y"])
    assert calls["n"] == 2


def test_assess_maps_noul_per_key():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["state"]["path"] == "the candidate"
        assert body["questions"]["is_complete"]["type"] == "noul"
        return httpx.Response(
            200,
            json={
                "model": "m",
                "answers": {
                    k: {"type": "noul", "noul": v}
                    for k, v in [("is_complete", 0.9), ("is_correct", 0.4)]
                },
                "usage": {"input_tokens": 1},
            },
        )

    judge = _judge(handler, base_url="http://localhost:8008")
    out = judge.assess("p", "the candidate", questions=FINISHED_QUESTIONS)
    assert out == {"is_complete": 0.9, "is_correct": 0.4}
