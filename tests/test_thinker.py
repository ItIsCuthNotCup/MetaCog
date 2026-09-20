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


def test_server_ignoring_n_falls_back_to_sequential():
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body["n"])
        return httpx.Response(
            200,
            json={
                # always returns exactly one choice, whatever n was asked for
                "choices": [{"text": f"gen{len(bodies)}", "finish_reason": "stop"}],
                "usage": {"completion_tokens": 5},
            },
        )

    t = _thinker(handler, base_url="http://x", model="m", api="completions")
    gens = t.generate("p", "", n=3, max_tokens=8, temperature=0.7)
    assert len(gens) == 3
    assert bodies == [3, 1, 1]  # first ask for 3, then top up with n=1
    assert t.supports_n is False

    bodies.clear()
    t.generate("p", "", n=2, max_tokens=8, temperature=0.7)
    assert bodies == [1, 1]  # later calls go straight to sequential


def test_finish_reason_stop_but_token_truncated_is_unfinished():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"text": "cut off mid-sen", "finish_reason": "stop"}],
                "usage": {"completion_tokens": 8},  # == max_tokens -> truncated
            },
        )

    t = _thinker(handler, base_url="http://x", model="m", api="completions")
    gens = t.generate("p", "", n=1, max_tokens=8, temperature=0.7)
    assert not gens[0].finished


def test_finish_reason_stop_fewer_tokens_is_finished():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"text": "done.", "finish_reason": "stop"}],
                "usage": {"completion_tokens": 3},
            },
        )

    t = _thinker(handler, base_url="http://x", model="m", api="completions")
    gens = t.generate("p", "", n=1, max_tokens=8, temperature=0.7)
    assert gens[0].finished


def test_stop_string_truncates_and_marks_finished():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"text": "answer\n\nSTOP here is junk", "finish_reason": "stop"}],
                "usage": {"completion_tokens": 8},  # would look truncated otherwise
            },
        )

    t = _thinker(handler, base_url="http://x", model="m", api="completions")
    gens = t.generate("p", "", n=1, max_tokens=8, temperature=0.7, stop=["STOP"])
    assert gens[0].text == "answer\n\n"
    assert gens[0].finished


def test_prefix_mode_prompt_folds_prefix_into_user_turn():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "cont"}, "finish_reason": "stop"}]},
        )

    t = _thinker(handler, base_url="http://x", model="m", api="chat", prefix_mode="prompt")
    t.generate("the problem", "partial answer", n=1, max_tokens=16, temperature=0.2)
    body = seen["body"]
    assert len(body["messages"]) == 1
    assert body["messages"][0]["role"] == "user"
    assert "the problem" in body["messages"][0]["content"]
    assert "partial answer" in body["messages"][0]["content"]
    assert "Continue the following partial solution" in body["messages"][0]["content"]
    assert "continue_final_message" not in body


def test_http_200_with_error_body_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": "completions not supported"})

    t = _thinker(handler, base_url="http://x", model="m", api="completions")
    with pytest.raises(ThinkerError, match="completions not supported"):
        t.generate("p", "", n=1, max_tokens=8, temperature=0.0)


def test_transport_error_retried_then_succeeds():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("dropped", request=request)
        return httpx.Response(200, json={"choices": [{"text": "x", "finish_reason": "stop"}]})

    t = _thinker(handler, base_url="http://x", model="m", api="completions")
    gens = t.generate("p", "", n=1, max_tokens=8, temperature=0.0)
    assert len(gens) == 1
    assert calls["n"] == 2


def test_transport_error_exhaustion_raises(monkeypatch):
    monkeypatch.setattr("metacog.thinker.time.sleep", lambda s: None)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.RemoteProtocolError("Server disconnected", request=request)

    t = _thinker(handler, base_url="http://x", model="m", api="completions")
    with pytest.raises(ThinkerError, match="transport error"):
        t.generate("p", "", n=1, max_tokens=8, temperature=0.0)
    assert calls["n"] == 3  # initial + max_retries(2)


def test_bearer_header_sent_when_key():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"choices": [{"text": "x", "finish_reason": "stop"}]})

    t = _thinker(handler, base_url="http://x", model="m", api="completions", api_key="k")
    t.generate("p", "", n=1, max_tokens=4, temperature=0.0)
    assert seen["auth"] == "Bearer k"
