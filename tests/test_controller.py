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
def test_best_of_n_prefers_finished_over_higher_scored_unfinished(strategy):
    # The judge scores the UNFINISHED candidate higher (it contains CORRECT); the
    # finished candidate must still win — an unfinished path has no final answer.
    thinker = FakeThinker(
        [
            [
                gen("CORRECT but truncated mid-thought", finished=False),
                gen("a complete final answer", finished=True),
            ]
        ]
    )
    judge = FakeJudge()
    mc = MetaCog(
        thinker,
        judge,
        Config(mode="best_of_n", n_paths=2, split_generations=False, strategy=strategy),
    )
    result = mc.run("p")
    assert result.answer == "a complete final answer"
    assert result.finished
    rnd = result.trace.rounds[0]
    assert rnd.kept[0] == 1
    assert rnd.candidates[rnd.kept[0]].finished


@pytest.mark.parametrize("strategy", ["noul", "choice"])
def test_greedy_anchor_adds_temperature0_candidate_first(strategy):
    thinker = FakeThinker(
        [
            [gen("greedy CORRECT answer", finished=True)],
            [gen("sampled A"), gen("sampled B")],
        ]
    )
    judge = FakeJudge()
    mc = MetaCog(
        thinker,
        judge,
        Config(
            mode="best_of_n",
            n_paths=3,
            split_generations=False,
            greedy_anchor=True,
            strategy=strategy,
        ),
    )
    result = mc.run("p")
    assert thinker.calls[0]["n"] == 1
    assert thinker.calls[0]["temperature"] == 0.0
    assert thinker.calls[1]["n"] == 2
    assert thinker.calls[1]["temperature"] == 0.8
    cands = result.trace.rounds[0].candidates
    assert len(cands) == 3
    assert cands[0].source == "greedy"
    assert [c.source for c in cands[1:]] == ["sample", "sample"]


def test_greedy_anchor_with_n_paths_1_is_greedy_only():
    thinker = FakeThinker([[gen("greedy answer", finished=True)]])
    judge = FakeJudge()
    mc = MetaCog(
        thinker,
        judge,
        Config(mode="best_of_n", n_paths=1, greedy_anchor=True, split_generations=False),
    )
    mc.run("p")
    assert len(thinker.calls) == 1
    assert thinker.calls[0]["n"] == 1
    assert thinker.calls[0]["temperature"] == 0.0


def test_cascade_accepts_high_confidence_greedy_without_sampling():
    thinker = FakeThinker([[gen("greedy CORRECT answer", finished=True)]])
    judge = FakeJudge(scores=[[0.99]])
    mc = MetaCog(
        thinker,
        judge,
        Config(
            mode="best_of_n",
            n_paths=3,
            greedy_anchor=True,
            cascade_confidence=0.95,
            split_generations=False,
        ),
    )
    result = mc.run("p")
    assert len(thinker.calls) == 1  # no sampled generations
    assert thinker.calls[0]["n"] == 1 and thinker.calls[0]["temperature"] == 0.0
    rnd = result.trace.rounds[0]
    assert len(rnd.candidates) == 1
    assert rnd.kept == [0]
    assert result.trace.judge_calls == 1
    assert result.answer == "greedy CORRECT answer"
    assert result.finished


def test_cascade_below_threshold_falls_through_to_full_pool():
    thinker = FakeThinker(
        [[gen("greedy answer", finished=True)], [gen("sampled A"), gen("sampled B")]]
    )
    # first score() call (cascade check) low; second (full pool) picks greedy
    judge = FakeJudge(scores=[[0.3], [0.9, 0.1, 0.1]])
    mc = MetaCog(
        thinker,
        judge,
        Config(
            mode="best_of_n",
            n_paths=3,
            greedy_anchor=True,
            cascade_confidence=0.95,
            split_generations=False,
            strategy="noul",
        ),
    )
    result = mc.run("p")
    assert len(thinker.calls) == 2
    assert thinker.calls[1]["n"] == 2 and thinker.calls[1]["temperature"] == 0.8
    assert len(result.trace.rounds[0].candidates) == 3
    assert result.trace.judge_calls == 1 + 3  # cascade check + full-pool noul
    assert result.answer == "greedy answer"


def test_cascade_only_considers_finished_candidates():
    # Greedy expands into the full text plus split pieces; an unfinished split
    # piece scoring above threshold must not win over the finished greedy text.
    greedy = (
        "The greedy approach computes the result step by step carefully here. "
        "Alternatively, a second unfinished approach wanders off mid-thought."
    )
    thinker = FakeThinker([[gen(greedy, finished=True)], [gen("sampled A"), gen("sampled B")]])
    judge = FakeJudge(scores=[[0.96, 0.99, 0.5]])  # unfinished piece scores highest
    mc = MetaCog(
        thinker,
        judge,
        Config(
            mode="best_of_n",
            n_paths=3,
            greedy_anchor=True,
            cascade_confidence=0.95,
            split_generations=True,
        ),
    )
    result = mc.run("p")
    assert len(thinker.calls) == 1  # no sampled generations
    assert result.answer == greedy
    assert result.finished
    rnd = result.trace.rounds[0]
    assert rnd.candidates[rnd.kept[0]].finished


def test_adaptive_stops_at_confident_greedy():
    thinker = FakeThinker([[gen("greedy CORRECT answer", finished=True)]])
    judge = FakeJudge(scores=[[0.97]])
    mc = MetaCog(
        thinker,
        judge,
        Config(mode="adaptive", stop_confidence=0.95, split_generations=False),
    )
    result = mc.run("p")
    assert result.answer == "greedy CORRECT answer"
    assert result.finished
    assert len(thinker.calls) == 1
    assert thinker.calls[0]["temperature"] == 0.0
    assert len(result.trace.rounds) == 1


@pytest.mark.parametrize("g_score,n_br", [(0.9, 2), (0.1, 6)])
def test_adaptive_branching_scales_with_impurity(g_score, n_br):
    # u = 1 - g_score; n_br = round(n_min + u * (n_max - n_min)) with n_min=2, n_max=6.
    thinker = FakeThinker(
        [
            [gen("greedy answer", finished=True)],
            [gen(f"sampled {i}", finished=True) for i in range(n_br)],
        ]
    )
    judge = FakeJudge(scores=[[g_score], [0.5] * (1 + n_br)])
    mc = MetaCog(
        thinker,
        judge,
        Config(
            mode="adaptive",
            stop_confidence=0.95,
            n_min=2,
            n_max=6,
            split_generations=False,
        ),
    )
    mc.run("p")
    assert len(thinker.calls) == 2
    assert thinker.calls[1]["n"] == n_br


def test_adaptive_sketch_prunes_and_expands():
    from metacog.controller import SKETCH_JUDGE_INSTRUCTIONS

    thinker = FakeThinker(
        [
            [gen("greedy answer", finished=True)],
            [gen(f"sketch idea number {i} with a plan") for i in range(4)],
            [gen("expanded CORRECT solution zero", finished=True)],
            [gen("expanded solution one", finished=True)],
        ]
    )
    # greedy 0.5 -> u = 0.5 -> n_br = round(2 + 0.5*4) = 4 sketches;
    # k = round(1 + 0.5*2) = 2; prune_margin 0.25 keeps scores >= 0.9 - 0.25.
    judge = FakeJudge(scores=[[0.5], [0.9, 0.8, 0.3, 0.2], [0.5, 0.9, 0.8]])
    mc = MetaCog(
        thinker,
        judge,
        Config(
            mode="adaptive",
            stop_confidence=0.95,
            n_min=2,
            n_max=6,
            sketch_tokens=200,
            expand_max=3,
            prune_margin=0.25,
            split_generations=False,
        ),
    )
    result = mc.run("p")
    assert len(thinker.calls) == 4  # greedy + sketches + 2 expands
    # the sketches were judged with the sketch-specific instructions
    sketch_judge_call = judge.calls[1]
    assert sketch_judge_call["kind"] == "score"
    assert sketch_judge_call["instructions"] == SKETCH_JUDGE_INSTRUCTIONS
    # exactly 2 expansions, prompted with the kept sketches' text
    expand_calls = thinker.calls[2:]
    assert len(expand_calls) == 2
    assert all(c["n"] == 1 for c in expand_calls)
    assert "sketch idea number 0" in expand_calls[0]["problem"]
    assert "sketch idea number 1" in expand_calls[1]["problem"]
    # round 0 is the sketch level; sketches are never the final answer
    rnd = result.trace.rounds[0]
    assert all(c.source == "sketch" for c in rnd.candidates)
    assert rnd.kept == [0, 1]
    final = result.trace.rounds[-1].candidates[result.trace.rounds[-1].kept[0]]
    assert final.finished
    assert final.source in {"greedy", "expand"}
    assert result.finished


def test_adaptive_unfinished_greedy_never_stops_early():
    thinker = FakeThinker(
        [
            [gen("greedy truncated", finished=False)],
            [gen(f"sampled {i}", finished=True) for i in range(6)],
        ]
    )
    # greedy scored 0.99 but unfinished -> no finished candidate to stop on
    judge = FakeJudge(scores=[[0.99], [0.5] * 7])
    mc = MetaCog(
        thinker,
        judge,
        Config(mode="adaptive", stop_confidence=0.95, split_generations=False),
    )
    mc.run("p")
    assert len(thinker.calls) == 2  # it branched anyway
    assert thinker.calls[1]["n"] == 6  # u = 1 - 0.0 -> n_max


def test_cascade_requires_greedy_anchor():
    with pytest.raises(ValueError, match="greedy_anchor"):
        MetaCog(FakeThinker([]), FakeJudge(), Config(cascade_confidence=0.9))


def test_cascade_skipped_when_greedy_unfinished():
    thinker = FakeThinker([[gen("greedy truncated", finished=False)], [gen("sA"), gen("sB")]])
    judge = FakeJudge(scores=[[0.9, 0.1, 0.1]])  # full-pool call; cascade never runs
    mc = MetaCog(
        thinker,
        judge,
        Config(
            mode="best_of_n",
            n_paths=3,
            greedy_anchor=True,
            cascade_confidence=0.95,
            split_generations=False,
            strategy="noul",
        ),
    )
    result = mc.run("p")
    assert len(thinker.calls) == 2  # no short-circuit on an unfinished path
    assert len(result.trace.rounds[0].candidates) == 3


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


def test_stepwise_finish_paths_expands_survivors_and_judges_full_thoughts():
    prefix = "pathA more "
    thinker = FakeThinker(
        [
            [gen("pathA "), gen("pathB ")],  # step 0, unfinished continuations
            [gen("more "), gen("other ")],  # step 1 continuations of "pathA "
            [gen("done CORRECT", finished=True), gen("still going")],  # tail, finished
        ]
    )
    judge = FakeJudge()
    mc = MetaCog(
        thinker,
        judge,
        Config(
            mode="stepwise",
            n_paths=2,
            step_tokens=64,
            max_steps=2,
            max_tokens=2000,
            finish_paths=True,
            split_generations=False,
            strategy="noul",
        ),
    )
    result = mc.run("p")
    assert len(result.trace.rounds) == 3
    tail = result.trace.rounds[-1]
    assert all(c.text.startswith(prefix) for c in tail.candidates)
    # tail call used the full remaining budget, not step_tokens
    assert thinker.calls[-1]["max_tokens"] == 2000 - len(prefix) // 4
    assert result.finished
    assert "CORRECT" in result.answer


def test_stepwise_without_finish_paths_keeps_old_tail():
    thinker = FakeThinker(
        [
            [gen("pathA "), gen("pathB ")],
            [gen("more "), gen("other ")],
        ]
    )
    judge = FakeJudge()
    mc = MetaCog(
        thinker,
        judge,
        Config(
            mode="stepwise",
            n_paths=2,
            max_steps=2,
            split_generations=False,
            strategy="noul",
        ),
    )
    result = mc.run("p")
    assert len(result.trace.rounds) == 2
    assert not result.finished
    assert len(thinker.calls) == 2  # no tail call


def test_stepwise_finished_beats_higher_scored_unfinished_mid_tree():
    thinker = FakeThinker(
        [[gen("unfinished but highly scored"), gen("finished CORRECT answer", finished=True)]]
    )
    judge = FakeJudge(scores=[[0.9, 0.4]])  # unfinished scores higher
    mc = MetaCog(
        thinker,
        judge,
        Config(
            mode="stepwise",
            n_paths=2,
            max_steps=5,
            split_generations=False,
            strategy="noul",
        ),
    )
    result = mc.run("p")
    assert result.finished
    assert result.answer == "finished CORRECT answer"
    assert len(result.trace.rounds) == 1


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
