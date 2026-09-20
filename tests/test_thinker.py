import json

import httpx
import pytest

from metacog.thinker import OpenAICompatThinker, ThinkerError


def _thinker(handler, **kw):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenAICompatThinker(client=client, **kw)


def test_completions_prompt_concat_and_finish_reason():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"text": " continuation A", "finish_reason": "stop"},
                    {"text": " continuation B", "finish_reason": "length"},
                ],
                "usage": {"completion_tokens": 20},
            },
        )

    t = _thinker(
        handler,
        base_url="http://x",
        model="m",
        api="completions",
        system_prompt="You are careful.",
    )
    gens = t.generate("the problem", "PREFIX", n=2, max_tokens=64, temperature=0.5)

    assert seen["url"] == "http://x/v1/completions"
    body = seen["body"]
    assert body["prompt"] == "You are careful.\n\nthe problem\n\nPREFIX"
    assert body["n"] == 2 and body["max_tokens"] == 64 and body["temperature"] == 0.5
    assert gens[0].finished and not gens[1].finished
    assert gens[0].tokens == 10  # total split evenly across n


def test_chat_prefill_message_and_extra_body():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": "cont"}, "finish_reason": "stop"}
                ],
                "usage": {"completion_tokens": 3},
            },
        )

    t = _thinker(
        handler,
        base_url="http://x",
        model="m",
        api="chat",
        system_prompt="sys",
        extra_body={"chat_template_kwargs": {"a": 1}},
    )
    gens = t.generate("prob", "partial answer", n=1, max_tokens=16, temperature=0.2)

    body = seen["body"]
    assert body["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "prob"},
        {"role": "assistant", "content": "partial answer"},
    ]
    assert body["continue_final_message"] is True
    assert body["add_generation_prompt"] is False
    assert body["chat_template_kwargs"] == {"a": 1}
    assert gens[0].text == "cont" and gens[0].finished


def test_chat_no_prefix_no_prefill_flags():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]},
        )

    t = _thinker(handler, base_url="http://x", model="m", api="chat")
    t.generate("prob", "", n=1, max_tokens=16, temperature=0.2)
    assert "continue_final_message" not in seen["body"]
    assert len(seen["body"]["messages"]) == 1


def test_n_greater_than_1_400_falls_back_to_sequential():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        body = json.loads(request.content)
        if body["n"] > 1:
            return httpx.Response(400, json={"error": "n must be 1"})
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": f"gen{calls['n']}"}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 5},
            },
        )

    t = _thinker(handler, base_url="http://x", model="m", api="chat")
    gens = t.generate("prob", "", n=3, max_tokens=16, temperature=0.2)
    assert len(gens) == 3
    assert calls["n"] == 4  # one failed n=3 + three sequential n=1


def test_http_error_raises_thinker_error_with_body():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server exploded")

    t = _thinker(handler, base_url="http://x", model="m", api="completions")
    with pytest.raises(ThinkerError, match="server exploded"):
        t.generate("prob", "", n=1, max_tokens=8, temperature=0.0)


def test_bearer_header_sent_when_key():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"choices": [{"text": "x", "finish_reason": "stop"}]})

    t = _thinker(handler, base_url="http://x", model="m", api="completions", api_key="k")
    t.generate("p", "", n=1, max_tokens=4, temperature=0.0)
    assert seen["auth"] == "Bearer k"
