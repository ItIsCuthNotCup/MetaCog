from conftest import FakeJudge, FakeThinker, gen
from metacog import Config, MetaCog
from metacog.controller import DIVERSITY_HINTS


def test_diversity_hints_fan_out_with_hint_per_branch():
    thinker = FakeThinker(
        [
            [gen("b0", finished=True)],
            [gen("b1", finished=True)],
            [gen("b2", finished=True)],
        ]
    )
    judge = FakeJudge(scores=[[0.6, 0.5, 0.4]])
    mc = MetaCog(
        thinker,
        judge,
        Config(
            mode="best_of_n",
            n_paths=3,
            split_generations=False,
            diversity_hints=["hint A", "hint B", "hint C"],
        ),
    )
    result = mc.run("p")
    assert len(thinker.calls) == 3
    assert {c["problem"] for c in thinker.calls} == {
        "p\n\nhint A",
        "p\n\nhint B",
        "p\n\nhint C",
    }
    assert all(c["n"] == 1 for c in thinker.calls)
    assert result.trace.thinker_calls == 3


def test_diversity_hints_none_behaves_normally():
    thinker = FakeThinker([[gen("a"), gen("b")]])
    judge = FakeJudge()
    mc = MetaCog(thinker, judge, Config(mode="best_of_n", n_paths=2, split_generations=False))
    mc.run("p")
    assert len(thinker.calls) == 1
    assert thinker.calls[0]["n"] == 2
    assert result_hints_absent(thinker.calls[0]["problem"])


def result_hints_absent(problem: str) -> bool:
    return all(h not in problem for h in DIVERSITY_HINTS)


def test_escalate_thinker_called_when_unconfident():
    thinker = FakeThinker(
        [
            [gen("greedy", finished=True)],
            [gen(f"branch {i}", finished=True) for i in range(4)],
        ]
    )
    escalate = FakeThinker([[gen("escalated CORRECT answer", finished=True)]])
    # greedy 0.5 -> n_br = 4; pool pick scores all 0.5 (< stop) -> escalate;
    # final pick over pool + escalate prefers the new candidate.
    judge = FakeJudge(scores=[[0.5], [0.5] * 5, [0.1] * 5 + [0.9]])
    mc = MetaCog(
        thinker,
        judge,
        Config(
            mode="adaptive",
            stop_confidence=0.95,
            n_min=2,
            n_max=6,
            split_generations=False,
            escalate_thinker=escalate,
        ),
    )
    result = mc.run("p")
    assert len(escalate.calls) == 1
    assert escalate.calls[0]["n"] == 1
    assert escalate.calls[0]["temperature"] == 0.0
    assert result.trace.escalated
    assert result.answer == "escalated CORRECT answer"
    last = result.trace.rounds[-1]
    assert last.candidates[last.kept[0]].source == "escalate"


def test_escalate_thinker_skipped_when_greedy_confident():
    thinker = FakeThinker([[gen("greedy CORRECT answer", finished=True)]])
    escalate = FakeThinker([[gen("escalated", finished=True)]])
    judge = FakeJudge(scores=[[0.99]])
    mc = MetaCog(
        thinker,
        judge,
        Config(
            mode="adaptive",
            stop_confidence=0.95,
            split_generations=False,
            escalate_thinker=escalate,
        ),
    )
    result = mc.run("p")
    assert escalate.calls == []
    assert not result.trace.escalated
