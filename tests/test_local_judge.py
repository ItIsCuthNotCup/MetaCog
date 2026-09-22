import json
import math

import httpx
import pytest

from metacog.judge import FINISHED_QUESTIONS, JudgeError
from metacog.local_judge import LogitJudge, build_prompt, yes_probability


def test_yes_probability_sums_variants():
    lp = {
        " yes": math.log(0.5),
        "Yes": math.log(0.1),
        " no": math.log(0.3),
        "other": math.log(0.1),
    }
    assert yes_probability(lp) == pytest.approx(0.6 / 0.9)


def test_yes_probability_missing_families_is_half():
    assert yes_probability({"foo": -1.0, "bar": -2.0}) == 0.5
    assert yes_probability({}) == 0.5


def _fake(probs: dict[str, float]):
    """Backend returning P(yes) from a substring of the graded path."""

    def backend(messages: list[dict]) -> dict[str, float]:
        user = messages[-1]["content"]
        path = user.split("path:\n", 1)[1].rsplit("\n\nAnswer", 1)[0]
        p = probs.get(path, 0.5)
        return {" yes": math.log(p), " no": math.log(1 - p)}

    return backend


def test_score_matches_systemone_shape():
    judge = LogitJudge(_fake({"a": 0.2, "b": 0.9, "c": 0.1}))
    v = judge.score("p", ["a", "b", "c"])
    assert v.raw == pytest.approx([0.2, 0.9, 0.1])
    assert v.probabilities == pytest.approx([0.2 / 1.2, 0.9 / 1.2, 0.1 / 1.2])
    assert v.choice == 1
    assert v.confidence == pytest.approx(0.9)


def test_choose_picks_highest():
    judge = LogitJudge(_fake({"a": 0.4, "b": 0.6}))
    v = judge.choose("p", ["a", "b"])
    assert v.choice == 1


def test_assess_one_readout_per_question():
    judge = LogitJudge(_fake({}))
    out = judge.assess("p", "cand", questions=FINISHED_QUESTIONS)
    assert set(out) == set(FINISHED_QUESTIONS)
    assert all(isinstance(v, float) for v in out.values())


def test_build_prompt_shape_and_tail_truncation():
    seen = {}

    def backend(messages):
        seen["m"] = messages
        return {" yes": 0.0, " no": 0.0}

    judge = LogitJudge(backend, max_chars=10)
    judge.score("P", ["0123456789" + "TAIL-END!!"])
    msgs = seen["m"]
    assert msgs[0]["role"] == "system"
    assert "yes or no" in msgs[0]["content"]
    assert "path:\n…TAIL-END!!\n" in msgs[1]["content"]
    msgs = build_prompt("P", "short", "INSTR")
    assert "INSTR" in msgs[1]["content"]
    assert "path:\nshort\n" in msgs[1]["content"]


def test_openai_compat_parses_top_logprobs():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "logprobs": {
                            "content": [
                                {
                                    "token": " yes",
                                    "logprob": math.log(0.7),
                                    "top_logprobs": [
                                        {"token": " yes", "logprob": math.log(0.7)},
                                        {"token": " no", "logprob": math.log(0.3)},
                                    ],
                                }
                            ]
                        }
                    }
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    judge = LogitJudge.openai_compat("http://x/v1", model="m", api_key="sk-t", client=client)
    v = judge.score("p", ["a", "b"])
    assert v.raw == pytest.approx([0.7, 0.7])
    body = seen["body"]
    assert body["max_tokens"] == 1
    assert body["logprobs"] is True
    assert body["top_logprobs"] == 20
    assert seen["auth"] == "Bearer sk-t"


def test_openai_compat_missing_logprobs_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "yes"}}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    judge = LogitJudge.openai_compat("http://x", model="m", client=client)
    with pytest.raises(JudgeError):
        judge.score("p", ["a"])
