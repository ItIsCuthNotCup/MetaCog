import pytest

from conftest import FakeJudge, FakeThinker, gen
from metacog import Config, MetaCog


@pytest.mark.parametrize("strategy", ["noul", "choice"])
def test_best_of_n_picks_correct_candidate(strategy):
    thinker = FakeThinker([[gen("wrong answer"), gen("this is CORRECT and finished", True)]])
    judge = FakeJudge()
    mc = MetaCog(thinker, judge, Config(mode="best_of_n", n_paths=2, strategy=strategy))
    result = mc.run("p")
    assert "CORRECT" in result.answer
    assert result.finished
    assert len(result.trace.rounds) == 1
    # noul = one request per candidate; choice = one request total
    assert result.trace.judge_calls == (2 if strategy == "noul" else 1)
    assert result.trace.thinker_calls == 1


@pytest.mark.parametrize("strategy", ["noul", "choice"])
def test_single_candidate_skips_judge(strategy):
    thinker = FakeThinker([[gen("only path", finished=True)]])
    judge = FakeJudge()
    mc = MetaCog(thinker, judge, Config(mode="best_of_n", n_paths=1, strategy=strategy))
    result = mc.run("p")
    assert result.answer == "only path"
    assert judge.calls == []
    assert result.trace.judge_calls == 0


@pytest.mark.parametrize("strategy", ["noul", "choice"])
def test_stepwise_continues_chosen_prefix_and_stops_on_finished(strategy):
    thinker = FakeThinker(
        [
            [gen("alpha reasoning "), gen("beta reasoning ")],
            [gen("done CORRECT", finished=True)],
        ]
    )
    # judge picks whichever candidate contains CORRECT
    judge = FakeJudge()
    mc = MetaCog(
        thinker,
        judge,
        Config(
            mode="stepwise",
            n_paths=2,
            max_steps=5,
            split_generations=False,
            strategy=strategy,
        ),
    )
    # step 0: FakeJudge sees no CORRECT, picks index 0 ("alpha reasoning")
    result = mc.run("p")
    assert result.finished
    assert result.answer == "alpha reasoning done CORRECT"
    # second generate() call must extend the chosen prefix
    assert thinker.calls[1]["prefix"] == "alpha reasoning "
    assert len(result.trace.rounds) == 2


@pytest.mark.parametrize("strategy", ["noul", "choice"])
def test_commit_confidence_triggers_greedy_finish(strategy):
    thinker = FakeThinker(
        [
            [gen("good path "), gen("bad path ")],
            [gen("final answer", finished=True)],
        ]
    )
    judge = FakeJudge(confidence=0.95)
    mc = MetaCog(
        thinker,
        judge,
        Config(
            mode="stepwise",
            n_paths=2,
            commit_confidence=0.9,
            split_generations=False,
            strategy=strategy,
        ),
    )
    result = mc.run("p")
    commit_call = thinker.calls[1]
    assert commit_call["n"] == 1
    assert commit_call["temperature"] == 0.0
    assert result.answer == "good path final answer"
    assert result.finished


@pytest.mark.parametrize("strategy", ["noul", "choice"])
def test_max_steps_exhaustion_returns_unfinished(strategy):
    thinker = FakeThinker([[gen("chunk")], [gen("chunk")]])
    judge = FakeJudge()
    mc = MetaCog(
        thinker,
        judge,
        Config(
            mode="stepwise",
            n_paths=1,
            max_steps=2,
            split_generations=False,
            strategy=strategy,
        ),
    )
    result = mc.run("p")
    assert not result.finished
    assert len(result.trace.rounds) == 2


def test_more_than_26_candidates_truncated_for_choice():
    gens = [gen(f"candidate number {i} with enough text") for i in range(30)]
    thinker = FakeThinker([gens])
    judge = FakeJudge()
    mc = MetaCog(
        thinker,
        judge,
        Config(mode="best_of_n", n_paths=30, split_generations=False, strategy="choice"),
    )
    result = mc.run("p")
    assert len(judge.calls[0]["candidates"]) == 26
    assert len(result.trace.rounds[0].candidates) == 30


def test_noul_has_no_candidate_cap():
    gens = [gen(f"candidate number {i} with enough text") for i in range(30)]
    thinker = FakeThinker([gens])
    judge = FakeJudge()
    mc = MetaCog(
        thinker,
        judge,
        Config(mode="best_of_n", n_paths=30, split_generations=False, strategy="noul"),
    )
    result = mc.run("p")
    assert len(judge.calls[0]["candidates"]) == 30
    assert result.trace.judge_calls == 30


@pytest.mark.parametrize("strategy", ["noul", "choice"])
def test_splits_labelled_source_split(strategy):
    multi = (
        "Shared setup text for the problem at hand.\n\n"
        "Approach 1: one way to solve it, described here at sufficient length.\n\n"
        "Approach 2: another way to solve it, also described at sufficient length."
    )
    thinker = FakeThinker([[gen(multi)]])
    judge = FakeJudge()
    mc = MetaCog(
        thinker,
        judge,
        Config(mode="best_of_n", n_paths=1, split_generations=True, strategy=strategy),
    )
    result = mc.run("p")
    sources = {c.source for c in result.trace.rounds[0].candidates}
    assert sources == {"sample", "split"}
    assert all(c.parent == 0 for c in result.trace.rounds[0].candidates if c.source == "split")


def test_only_last_split_piece_inherits_finished():
    multi = (
        "Shared setup text for the problem at hand.\n\n"
        "Approach 1: one way to solve it, described here at sufficient length.\n\n"
        "Approach 2: another way to solve it, also described at sufficient length."
    )
    thinker = FakeThinker([[gen(multi, finished=True)]])
    judge = FakeJudge()
    mc = MetaCog(
        thinker,
        judge,
        Config(mode="best_of_n", n_paths=1, split_generations=True, strategy="noul"),
    )
    result = mc.run("p")
    splits = [c for c in result.trace.rounds[0].candidates if c.source == "split"]
    assert len(splits) >= 2
    assert splits[-1].finished
    assert all(not c.finished for c in splits[:-1])


@pytest.mark.parametrize("strategy", ["noul", "choice"])
def test_verify_finished_records_assessment(strategy):
    thinker = FakeThinker([[gen("CORRECT answer", finished=True), gen("bad")]])
    judge = FakeJudge()
    mc = MetaCog(
        thinker,
        judge,
        Config(
            mode="best_of_n",
            n_paths=2,
            split_generations=False,
            verify_finished=True,
            strategy=strategy,
        ),
    )
    result = mc.run("p")
    rnd = result.trace.rounds[0]
    assert rnd.assessment is not None
    assert set(rnd.assessment) == {"is_complete", "is_correct"}
    assert result.trace.judge_calls == (3 if strategy == "noul" else 2)
