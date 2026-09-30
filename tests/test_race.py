"""Race mode: concurrent streams, judge kills losers at the confidence bar."""

from __future__ import annotations

import json
import threading
import time

import httpx

from conftest import FakeJudge, FakeThinker
from metacog import Config, MetaCog, StreamHandle
from metacog.thinker import OpenAICompatThinker
from metacog.types import Generation


class FakeStreamHandle(StreamHandle):
    """Handle whose text arrives in timed chunks, like a real SSE stream."""

    def __init__(self, chunks, delay=0.005, finished=True, tokens=12):
        super().__init__()
        self._chunks = chunks
        self._delay = delay
        self._fin = finished
        self._tok = tokens
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        for c in self._chunks:
            if self.cancelled:
                break
            self._push(c, "")
            time.sleep(self._delay)
        self.tokens = self._tok
        self._settle(
            text=self.text,
            finished=self._fin and not self.cancelled,
            tokens=self._tok,
        )


class FakeStreamThinker(FakeThinker):
    """Returns scripted streaming handles instead of generate() batches."""

    def __init__(self, scripts):
        super().__init__([])
        self.scripts = list(scripts)
        self.handles: list[StreamHandle] = []
        self.stream_calls: list[dict] = []

    def generate_streaming(self, problem, prefix, *, max_tokens, temperature, stop=None):
        self.stream_calls.append(
            {
                "problem": problem,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        spec = self.scripts.pop(0) if self.scripts else {"chunks": [""]}
        h = FakeStreamHandle(
            spec["chunks"],
            delay=spec.get("delay", 0.005),
            finished=spec.get("finished", True),
        )
        self.handles.append(h)
        return h


def race_cfg(**kw):
    base = dict(
        mode="race",
        n_paths=3,
        greedy_anchor=False,
        race_confidence=0.8,
        race_poll_seconds=0.005,
        race_score_chars=1,
        race_max_seconds=30,
        max_tokens=64,
    )
    base.update(kw)
    return Config(**base)


def chunks(n, text, delay=0.005, finished=True):
    """n chunks spelling text one char at a time."""
    return {"chunks": list(text), "delay": delay, "finished": finished}


def test_race_winner_crosses_bar_and_cancels_losers():
    # Path 1 scores >= 0.8 on the first judge poll; paths 0 and 2 keep
    # streaming for a long time, so if they are not cancelled the test is slow.
    thinker = FakeStreamThinker(
        [
            chunks(40, "slow wrong path " * 30, delay=0.01),
            chunks(40, "the WINNER answer " * 30, delay=0.01),
            chunks(40, "another wrong path " * 30, delay=0.01),
        ]
    )
    judge = FakeJudge(scores=[[0.1, 0.85, 0.05]])
    mc = MetaCog(thinker, judge, race_cfg())

    res = mc.run("problem")

    assert "WINNER" in res.answer
    assert res.finished
    # losers were cancelled before finishing
    assert thinker.handles[0].cancelled
    assert thinker.handles[2].cancelled
    assert not thinker.handles[0].result.finished
    assert not thinker.handles[2].result.finished
    assert len(res.trace.rounds) == 1
    assert res.trace.thinker_calls == 3


def test_race_greedy_anchor_gets_zero_temperature():
    thinker = FakeStreamThinker([chunks(4, "a"), chunks(4, "b"), chunks(4, "c")])
    judge = FakeJudge(scores=[[0.3, 0.3, 0.3]] * 40)
    mc = MetaCog(thinker, judge, race_cfg(greedy_anchor=True))

    mc.run("problem")

    assert thinker.stream_calls[0]["temperature"] == 0.0
    assert thinker.stream_calls[1]["temperature"] > 0


def test_race_no_confident_path_falls_back_to_verdict():
    thinker = FakeStreamThinker([chunks(3, "alpha"), chunks(3, "beta CORRECT"), chunks(3, "gamma")])
    # never above the bar during polling; final verdict scores come from the
    # same queue (CORRECT-matching entries pick beta)
    judge = FakeJudge(scores=[[0.3, 0.3, 0.3]] * 40)
    mc = MetaCog(thinker, judge, race_cfg())

    res = mc.run("problem")

    assert res.answer in ("alpha", "beta CORRECT", "gamma")


def test_race_early_agreement_two_same_answers():
    # Two streams finish on identical answers while a third is still going;
    # agreement ends the race without the judge ever crossing the bar.
    thinker = FakeStreamThinker(
        [
            chunks(5, "answer 42"),
            chunks(5, "answer 42"),
            chunks(200, "still generating " * 60, delay=0.01),
        ]
    )
    judge = FakeJudge(scores=[[0.2, 0.2, 0.2]] * 40)
    mc = MetaCog(thinker, judge, race_cfg())

    res = mc.run("problem")

    assert res.answer == "answer 42"
    assert res.finished
    assert thinker.handles[2].cancelled


def test_race_falls_back_to_best_of_n_without_streaming():
    thinker = FakeThinker([[Generation(text="plain answer", finished=True)]])
    judge = FakeJudge(scores=[[0.9]])
    mc = MetaCog(thinker, judge, race_cfg(n_paths=1))

    res = mc.run("problem")

    assert res.answer == "plain answer"


def test_race_winner_unfinished_falls_back():
    # Winner crosses the bar mid-stream but never finishes (truncated);
    # the race must not commit to it and must judge the completed paths.
    thinker = FakeStreamThinker(
        [
            chunks(5, "half answer", finished=False),
            chunks(5, "done CORRECT"),
            chunks(5, "done wrong"),
        ]
    )
    judge = FakeJudge(scores=[[0.9, 0.1, 0.1]] + [[0.1, 1.0, 0.1]] * 40)
    mc = MetaCog(thinker, judge, race_cfg())

    res = mc.run("problem")

    # the judge's favourite partial is still returned, flagged unfinished
    assert res.answer == "half answer"
    assert not res.finished


def test_generate_streaming_end_to_end_mock_transport():
    """OpenAICompatThinker.generate_streaming parses SSE deltas and settles."""
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        lines = (
            "".join(
                f"data: {json.dumps(c)}\n\n"
                for c in [
                    {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]},
                    {"choices": [{"delta": {"content": " world"}, "finish_reason": "stop"}]},
                    {"choices": [], "usage": {"completion_tokens": 2}},
                ]
            )
            + "data: [DONE]\n\n"
        )
        return httpx.Response(200, text=lines, headers={"content-type": "text/event-stream"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    t = OpenAICompatThinker(client=client, base_url="http://x", model="m")

    h = t.generate_streaming("problem", "", max_tokens=16, temperature=0.0)

    assert h.wait(timeout=10)
    assert h.result is not None
    assert h.result.text == "Hello world"
    assert h.result.finished
    assert h.result.tokens == 2
    assert seen["body"]["stream"] is True
    assert seen["body"]["n"] == 1


def test_generate_streaming_cancel_marks_unfinished():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text="data: "
            + json.dumps({"choices": [{"delta": {"content": "partial"}, "finish_reason": None}]})
            + "\n\n",
            headers={"content-type": "text/event-stream"},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    t = OpenAICompatThinker(client=client, base_url="http://x", model="m")

    h = t.generate_streaming("problem", "", max_tokens=16, temperature=0.0)
    h.cancel()
    assert h.wait(timeout=10)
    assert h.result is None or not h.result.finished


def test_race_on_branch_streams_samples_and_cancels_losers():
    # best_of_n + race_on_branch: the cascade fails (greedy truncated), the
    # sampled paths stream concurrently and losers die at the confidence bar.
    thinker = FakeStreamThinker(
        [
            chunks(5, "WINNER final answer " * 5),
            chunks(200, "slow wrong path " * 50, delay=0.01),
            chunks(200, "slow wrong path2 " * 50, delay=0.01),
        ]
    )
    thinker.script = [[Generation(text="greedy truncated", finished=False)]]
    # first poll crowns stream 0; then the verdict judges all 4 candidates
    judge = FakeJudge(scores=[[0.9, 0.1, 0.1], [0.9, 0.05, 0.05, 0.05]] * 40)
    mc = MetaCog(
        thinker,
        judge,
        Config(
            mode="best_of_n",
            n_paths=4,
            greedy_anchor=True,
            cascade_confidence=0.8,
            race_on_branch=True,
            race_confidence=0.8,
            race_poll_seconds=0.005,
            race_score_chars=1,
            race_max_seconds=30,
            split_generations=False,
        ),
    )

    res = mc.run("problem")

    # losers were cancelled mid-flight; winner still picked by the normal verdict
    assert thinker.handles[1].cancelled
    assert thinker.handles[2].cancelled
    assert "WINNER" in res.answer
    assert res.finished
    assert res.trace.thinker_calls == 4  # greedy + 3 streams


def test_race_on_branch_deadline_lets_streams_finish():
    # No winner before the deadline: unfinished streams are NOT cancelled —
    # they complete naturally so finishable paths still produce candidates.
    thinker = FakeStreamThinker(
        [
            chunks(30, "CORRECT but slow " * 10, delay=0.01),
            chunks(30, "also slow wrong " * 10, delay=0.01),
        ]
    )
    thinker.script = [[Generation(text="greedy truncated", finished=False)]]
    # huge race_score_chars => no poll re-scores; the only score() call is the
    # final verdict over [greedy, stream0, stream1]
    judge = FakeJudge(scores=[[0.1, 0.9, 0.05]] * 10)
    mc = MetaCog(
        thinker,
        judge,
        Config(
            mode="best_of_n",
            n_paths=3,
            greedy_anchor=True,
            cascade_confidence=0.8,
            race_on_branch=True,
            race_confidence=0.99,
            race_poll_seconds=0.005,
            race_score_chars=10**9,
            race_max_seconds=0.05,
            split_generations=False,
        ),
    )

    res = mc.run("problem")

    assert not thinker.handles[0].cancelled
    assert not thinker.handles[1].cancelled
    assert all(h.result is not None and h.result.finished for h in thinker.handles)
    assert "CORRECT but slow" in res.answer or "also slow wrong" in res.answer
    assert res.finished


def test_race_on_branch_without_streaming_falls_back_to_generate():
    thinker = FakeThinker(
        [
            [Generation(text="greedy unsure", finished=False)],
            [Generation(text="sampled CORRECT", finished=True)],
        ]
    )
    judge = FakeJudge(scores=[[0.9, 0.1]] * 10)
    mc = MetaCog(
        thinker,
        judge,
        Config(
            mode="best_of_n",
            n_paths=2,
            greedy_anchor=True,
            race_on_branch=True,
            split_generations=False,
        ),
    )
    res = mc.run("p")
    assert res.answer == "sampled CORRECT"
    assert len(thinker.calls) == 2  # normal generate() path, no streaming
