"""The metacognition loop: branch with the thinker, prune with the judge, repeat."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from pydantic import BaseModel

from .answers import extract_answer
from .judge import ANSWER_PRIOR_INSTRUCTIONS, FINISHED_QUESTIONS, TRIAGE_INSTRUCTIONS, Judge
from .split import split_paths
from .thinker import Thinker
from .types import Candidate, Result, Round, Trace, Verdict

MAX_JUDGE_CANDIDATES = 26

SKETCH_PROMPT = (
    "{problem}\n\nDo NOT solve this yet. In at most 150 words, outline the approach you "
    "would take: the key idea, the main steps, and any pitfall to avoid. No final answer."
)
EXPAND_PROMPT = (
    "{problem}\n\nSolve it by following this approach (fix it if it is flawed):\n{sketch}"
)
RESKETCH_PROMPT = (
    "{problem}\n\nA previous attempt reached this answer (it may be wrong):\n{answer}\n\n"
    "Do NOT solve this yet. In at most 150 words, outline a different approach or a fix "
    "for the flaw in that attempt: the key idea, the main steps, and any pitfall. No final answer."
)
SKETCH_JUDGE_INSTRUCTIONS = (
    "The response is only an outline of an approach, not a full solution. Judge whether "
    "following this approach would lead to the correct final answer."
)


def _est_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token). We do not ship a tokenizer, so the
    stepwise token cap is approximate; treat ``max_tokens`` as a soft bound."""
    return len(text) // 4


class Config(BaseModel):
    mode: Literal["best_of_n", "stepwise", "adaptive"] = "stepwise"
    # noul: one isolated "is this correct?" request per candidate (no cap);
    # choice: one-shot pick over all candidates (max 26).
    strategy: Literal["noul", "choice"] = "noul"
    n_paths: int = 4  # samples per kept prefix per step
    # best_of_n: one of the n_paths candidates is a temperature-0 sample
    greedy_anchor: bool = False
    # best_of_n + greedy_anchor: accept the greedy path alone if the judge scores it >= this
    cascade_confidence: float | None = None
    # stepwise: after the last step, expand each kept prefix into n_paths full
    # completions and judge those
    finish_paths: bool = False
    keep_top_k: int = 1  # beam width after judging
    step_tokens: int = 256  # stepwise: tokens per extension
    max_steps: int = 8
    max_tokens: int = 2048  # best_of_n: per sample; stepwise: total cap per path
    temperature: float = 0.8
    stop: list[str] | None = None
    split_generations: bool = True  # run split_paths on each sample
    commit_confidence: float | None = None  # stepwise: >= this -> finish winner greedily
    judge_instructions: str | None = None
    verify_finished: bool = False  # assess() a finished winner, record on the Round
    # -- adaptive (decision-tree heuristics; branching scales with the judge's
    #    uncertainty about the greedy answer) ----------------------------------
    # greedy noul >= stop_confidence -> leaf, return it after one generation
    stop_confidence: float = 0.95
    # branching factor at a node grows linearly with impurity u = 1 - greedy score:
    # n_branches = round(n_min + u * (n_max - n_min))
    n_min: int = 2
    n_max: int = 6
    # 0 -> branches are full sampled solutions (adaptive best-of-N). > 0 -> branches
    # are short sketches of this many tokens; the judge prunes them and only the
    # kept sketches are expanded into full solutions.
    sketch_tokens: int = 0
    # sketches kept for expansion: round(1 + u * (expand_max - 1)), and a sketch is
    # dropped when its score is below best_sketch_score - prune_margin
    expand_max: int = 3
    prune_margin: float = 0.25
    # after each level, if the judge's score for the best full solution is still
    # below stop_confidence, open another level (branching from the best answer so
    # far) — up to this many levels in total
    max_rounds: int = 1
    # adaptive: when set, the judge first scores the bare problem (path = ""); a
    # score >= triage runs the normal flow, a lower score fires the greedy root
    # and the level-0 branches concurrently, sizing branching from u = 1 - easy.
    triage: float | None = None
    # weight of the judge's "answer prior": each candidate's noul score is raised by
    # answer_prior * P(its bare final answer is correct), judged without the reasoning.
    # None disables. Only applied when >1 candidate is judged (never to the cascade).
    answer_prior: float | None = None
    answer_extractor: Callable[[str], str | None] = extract_answer


class MetaCog:
    def __init__(self, thinker: Thinker, judge: Judge, config: Config | None = None):
        self.thinker = thinker
        self.judge = judge
        self.config = config or Config()
        if self.config.cascade_confidence is not None and not self.config.greedy_anchor:
            raise ValueError("cascade_confidence requires greedy_anchor=True")

    # -- internals -----------------------------------------------------------

    def _expand(
        self,
        gens,
        prefix: str,
        *,
        source: Literal["sample", "greedy", "expand"] = "sample",
        parent: int | None = None,
    ) -> list[Candidate]:
        """Turn raw generations (continuations of ``prefix``) into candidates."""
        cands: list[Candidate] = []
        for idx, g in enumerate(gens):
            full = prefix + g.text
            pid = idx if parent is None else parent
            cands.append(Candidate(text=full, finished=g.finished, source=source, parent=pid))
            if self.config.split_generations:
                pieces = split_paths(g.text)
                for i, piece in enumerate(pieces):
                    if len(pieces) > 1:
                        # Only the last piece can be finished; earlier alternatives
                        # were abandoned mid-generation by definition.
                        cands.append(
                            Candidate(
                                text=prefix + piece,
                                finished=g.finished if i == len(pieces) - 1 else False,
                                source="split",
                                parent=pid,
                            )
                        )
        return cands

    def _verdict(self, problem: str, cands: list[Candidate], trace: Trace) -> Verdict:
        """Judge candidates; more than 26 are truncated in source order (samples first)."""
        if len(cands) == 1:
            return Verdict(probabilities=[1.0], choice=0, confidence=1.0)
        if self.config.strategy == "noul":
            cfg = self.config
            v = self.judge.score(
                problem,
                [c.text for c in cands],
                instructions=cfg.judge_instructions,
            )
            trace.judge_calls += len(cands)
            if cfg.answer_prior is None:
                return v
            # Answer prior: score each distinct bare final answer on its own and
            # add answer_prior * P(answer correct) to the full-text score.
            answers = [cfg.answer_extractor(c.text) for c in cands]
            distinct = list(dict.fromkeys(a for a in answers if a is not None))
            if not distinct:
                return v
            pv = self.judge.score(
                problem,
                [f"Final answer: {a}" for a in distinct],
                instructions=ANSWER_PRIOR_INSTRUCTIONS,
            )
            trace.judge_calls += len(distinct)
            prior = dict(zip(distinct, pv.raw or pv.probabilities, strict=True))
            base = v.raw or v.probabilities
            new_raw = [
                base[i] + cfg.answer_prior * (prior[answers[i]] if answers[i] is not None else 0.5)
                for i in range(len(cands))
            ]
            total = sum(new_raw)
            probs = [p / total for p in new_raw] if total else [1.0 / len(new_raw)] * len(new_raw)
            # probabilities/choice use the combined score; raw stays the text-only
            # noul so confidence thresholds keep their calibrated meaning.
            return Verdict(
                probabilities=probs,
                choice=max(range(len(new_raw)), key=lambda i: new_raw[i]),
                confidence=max(new_raw),
                raw=list(base),
            )
        judged = cands[:MAX_JUDGE_CANDIDATES]
        v = self.judge.choose(
            problem,
            [c.text for c in judged],
            instructions=self.config.judge_instructions,
        )
        trace.judge_calls += 1
        if len(cands) > MAX_JUDGE_CANDIDATES:
            # Pad unseen tail with zero probability so indexing stays aligned.
            v = Verdict(
                probabilities=v.probabilities + [0.0] * (len(cands) - MAX_JUDGE_CANDIDATES),
                choice=v.choice,
                confidence=v.confidence,
            )
        return v

    def _generate(
        self,
        problem: str,
        prefix: str,
        *,
        n: int,
        max_tokens: int,
        temperature: float,
        trace: Trace,
    ):
        gens = self.thinker.generate(
            problem,
            prefix,
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=self.config.stop,
        )
        trace.thinker_calls += 1
        trace.thinker_tokens += sum(g.tokens for g in gens)
        return gens

    def _verify(self, problem: str, answer: str, rnd: Round, trace: Trace) -> None:
        if self.config.verify_finished:
            rnd.assessment = self.judge.assess(problem, answer, questions=FINISHED_QUESTIONS)
            trace.judge_calls += 1

    # -- modes ---------------------------------------------------------------

    def run(self, problem: str) -> Result:
        if self.config.mode == "best_of_n":
            return self._best_of_n(problem)
        if self.config.mode == "adaptive":
            return self._adaptive(problem)
        return self._stepwise(problem)

    def _pick(
        self, problem: str, cands: list[Candidate], step: int, trace: Trace
    ) -> tuple[Result, float]:
        """Judge full candidates, finished-first, record the round; return the best
        and the judge's score for it (raw noul when available)."""
        cfg = self.config
        verdict = self._verdict(problem, cands, trace)
        ranked = sorted(
            range(len(cands)),
            key=lambda i: (not cands[i].finished, -verdict.probabilities[i]),
        )
        kept = ranked[: cfg.keep_top_k]
        rnd = Round(step=step, prefix="", candidates=cands, verdict=verdict, kept=kept)
        trace.rounds.append(rnd)
        best = cands[ranked[0]]
        if best.finished:
            self._verify(problem, best.text, rnd, trace)
        score = (verdict.raw or verdict.probabilities)[ranked[0]] if best.finished else 0.0
        return Result(answer=best.text, finished=best.finished, trace=trace), score

    def _adaptive(self, problem: str) -> Result:
        """Decision-tree style search: the greedy answer is the root; the judge's
        uncertainty about it (impurity u = 1 - score) decides whether to split at
        all, how many branches to open, and how many to expand to full depth.
        Only full solutions are ever committed to as answers."""
        cfg, trace = self.config, Trace()
        easy: float | None = None
        pre = None  # level-0 branches pre-generated concurrently with the greedy root
        if cfg.triage is not None:
            tv = self.judge.score(problem, [""], instructions=TRIAGE_INSTRUCTIONS)
            trace.judge_calls += 1
            easy = (tv.raw or tv.probabilities)[0]
            trace.triage = easy
        if easy is not None and easy < cfg.triage:
            # Hard problem: don't wait for the greedy answer before branching.
            u0 = 1.0 - easy
            n_br0 = max(1, round(cfg.n_min + u0 * (cfg.n_max - cfg.n_min)))

            def gen_greedy():
                return self.thinker.generate(
                    problem,
                    "",
                    n=1,
                    max_tokens=cfg.max_tokens,
                    temperature=0.0,
                    stop=cfg.stop,
                )

            def gen_level0():
                if cfg.sketch_tokens <= 0:
                    return self.thinker.generate(
                        problem,
                        "",
                        n=n_br0,
                        max_tokens=cfg.max_tokens,
                        temperature=cfg.temperature,
                        stop=cfg.stop,
                    )
                return self.thinker.generate(
                    SKETCH_PROMPT.format(problem=problem),
                    "",
                    n=n_br0,
                    max_tokens=cfg.sketch_tokens,
                    temperature=cfg.temperature,
                    stop=cfg.stop,
                )

            with ThreadPoolExecutor(2) as ex:
                f_greedy = ex.submit(gen_greedy)
                f_pre = ex.submit(gen_level0)
                greedy, pre = f_greedy.result(), f_pre.result()
            for gens in (greedy, pre):
                trace.thinker_calls += 1
                trace.thinker_tokens += sum(g.tokens for g in gens)
        else:
            greedy = self._generate(
                problem, "", n=1, max_tokens=cfg.max_tokens, temperature=0.0, trace=trace
            )
        root = self._expand(greedy, "", source="greedy")
        v = self.judge.score(problem, [c.text for c in root], instructions=cfg.judge_instructions)
        trace.judge_calls += len(root)
        score_vals = v.raw or v.probabilities
        finished = [i for i, c in enumerate(root) if c.finished]
        g_score = max((score_vals[i] for i in finished), default=0.0)
        if finished and g_score >= cfg.stop_confidence:
            choice = max(finished, key=lambda i: score_vals[i])
            verdict = Verdict(
                probabilities=v.probabilities, choice=choice, confidence=g_score, raw=v.raw
            )
            rnd = Round(step=0, prefix="", candidates=root, verdict=verdict, kept=[choice])
            trace.rounds.append(rnd)
            self._verify(problem, root[choice].text, rnd, trace)
            return Result(answer=root[choice].text, finished=True, trace=trace)
        cands = list(root)
        best_text = root[finished[0]].text if finished else None
        score, step = g_score, 0
        for level in range(max(1, cfg.max_rounds)):
            # level 0 was sized from triage uncertainty (u = 1 - easy) and may
            # already be generated concurrently with the greedy root.
            u = 1.0 - easy if level == 0 and pre is not None else 1.0 - score
            n_br = max(1, round(cfg.n_min + u * (cfg.n_max - cfg.n_min)))
            if cfg.sketch_tokens <= 0:
                gens = (
                    pre
                    if level == 0 and pre is not None
                    else self._generate(
                        problem,
                        "",
                        n=n_br,
                        max_tokens=cfg.max_tokens,
                        temperature=cfg.temperature,
                        trace=trace,
                    )
                )
                cands = cands + self._expand(gens, "")
            else:
                # Sketch level: cheap outlines, judged and pruned before any is expanded.
                # Later levels branch from the best full answer so far.
                if level == 0 or best_text is None:
                    prompt = SKETCH_PROMPT.format(problem=problem)
                else:
                    prompt = RESKETCH_PROMPT.format(problem=problem, answer=best_text)
                sk_gens = (
                    pre
                    if level == 0 and pre is not None
                    else self._generate(
                        prompt,
                        "",
                        n=n_br,
                        max_tokens=cfg.sketch_tokens,
                        temperature=cfg.temperature,
                        trace=trace,
                    )
                )
                sketches = [
                    Candidate(text=g.text, finished=False, source="sketch", parent=i)
                    for i, g in enumerate(sk_gens)
                ]
                sv = self.judge.score(
                    problem,
                    [s.text for s in sketches],
                    instructions=SKETCH_JUDGE_INSTRUCTIONS,
                )
                trace.judge_calls += len(sketches)
                s_vals = sv.raw or sv.probabilities
                order = sorted(range(len(sketches)), key=lambda i: -s_vals[i])
                k = max(1, round(1 + u * (cfg.expand_max - 1)))
                kept = [i for i in order[:k] if s_vals[i] >= s_vals[order[0]] - cfg.prune_margin]
                trace.rounds.append(
                    Round(step=step, prefix="", candidates=sketches, verdict=sv, kept=kept)
                )
                step += 1
                for i in kept:
                    gens = self._generate(
                        EXPAND_PROMPT.format(problem=problem, sketch=sketches[i].text),
                        "",
                        n=1,
                        max_tokens=cfg.max_tokens,
                        temperature=cfg.temperature,
                        trace=trace,
                    )
                    cands = cands + self._expand(gens, "", source="expand", parent=i)
            result, score = self._pick(problem, cands, step, trace)
            step += 1
            if result.finished:
                best_text = result.answer
            if result.finished and score >= cfg.stop_confidence:
                break
        return result

    def _best_of_n(self, problem: str) -> Result:
        cfg, trace = self.config, Trace()
        cands: list[Candidate] = []
        if cfg.greedy_anchor:
            greedy = self._generate(
                problem,
                "",
                n=1,
                max_tokens=cfg.max_tokens,
                temperature=0.0,
                trace=trace,
            )
            cands.extend(self._expand(greedy, "", source="greedy"))
            # Confidence cascade: if the greedy path finished and the judge's
            # noul score clears the threshold, skip the sampled generations
            # entirely (measured: noul >= 0.95 is right ~98% of the time).
            if cfg.cascade_confidence is not None and cands and cands[0].finished:
                v = self.judge.score(
                    problem,
                    [c.text for c in cands],
                    instructions=cfg.judge_instructions,
                )
                trace.judge_calls += len(cands)
                # Threshold on the raw noul when available: normalised probabilities
                # degenerate to 1.0 for a single candidate.
                score_vals = v.raw or v.probabilities
                # Only finished candidates can carry a final answer — an
                # unfinished split piece must never short-circuit the cascade.
                choice = max(
                    (i for i, c in enumerate(cands) if c.finished),
                    key=lambda i: score_vals[i],
                )
                if score_vals[choice] >= cfg.cascade_confidence:
                    verdict = Verdict(
                        probabilities=v.probabilities,
                        choice=choice,
                        confidence=score_vals[choice],
                        raw=v.raw,
                    )
                    rnd = Round(
                        step=0,
                        prefix="",
                        candidates=cands,
                        verdict=verdict,
                        kept=[choice],
                    )
                    trace.rounds.append(rnd)
                    best = cands[choice]
                    self._verify(problem, best.text, rnd, trace)
                    return Result(answer=best.text, finished=best.finished, trace=trace)
        n_sampled = cfg.n_paths - 1 if cfg.greedy_anchor else cfg.n_paths
        if n_sampled:
            gens = self._generate(
                problem,
                "",
                n=n_sampled,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                trace=trace,
            )
            cands.extend(self._expand(gens, ""))
        verdict = self._verdict(problem, cands, trace)
        # A finished candidate always outranks an unfinished one (a truncated sample
        # or an abandoned split-off path has no final answer to commit to).
        ranked = sorted(
            range(len(cands)), key=lambda i: (not cands[i].finished, -verdict.probabilities[i])
        )
        kept = ranked[: cfg.keep_top_k]
        rnd = Round(step=0, prefix="", candidates=cands, verdict=verdict, kept=kept)
        trace.rounds.append(rnd)
        best = cands[ranked[0]]
        if best.finished:
            self._verify(problem, best.text, rnd, trace)
        return Result(answer=best.text, finished=best.finished, trace=trace)

    def _stepwise(self, problem: str) -> Result:
        cfg, trace = self.config, Trace()
        prefixes = [""]
        best_cand: Candidate | None = None
        for step in range(cfg.max_steps):
            cands: list[Candidate] = []
            offset = 0
            for prefix in prefixes:
                remaining = cfg.max_tokens - _est_tokens(prefix)
                if remaining <= 0:
                    continue
                gens = self._generate(
                    problem,
                    prefix,
                    n=cfg.n_paths,
                    max_tokens=min(cfg.step_tokens, remaining),
                    temperature=cfg.temperature,
                    trace=trace,
                )
                new = self._expand(gens, prefix)
                for c in new:
                    if c.parent is not None:
                        c.parent += offset
                offset += len(gens)
                cands.extend(new)
            if not cands:
                break
            verdict = self._verdict(problem, cands, trace)
            # Finished-first: a branch that reached a final answer outranks a
            # higher-scored truncated one at every level.
            ranked = sorted(
                range(len(cands)),
                key=lambda i: (not cands[i].finished, -verdict.probabilities[i]),
            )
            kept = ranked[: cfg.keep_top_k]
            rnd = Round(
                step=step,
                prefix="\n".join(prefixes),
                candidates=cands,
                verdict=verdict,
                kept=kept,
            )
            trace.rounds.append(rnd)
            best_cand = cands[ranked[0]]
            if best_cand.finished:
                self._verify(problem, best_cand.text, rnd, trace)
                return Result(answer=best_cand.text, finished=True, trace=trace)
            if cfg.commit_confidence is not None and verdict.confidence >= cfg.commit_confidence:
                remaining = max(1, cfg.max_tokens - _est_tokens(best_cand.text))
                gens = self._generate(
                    problem,
                    best_cand.text,
                    n=1,
                    max_tokens=remaining,
                    temperature=0.0,
                    trace=trace,
                )
                text = best_cand.text + gens[0].text
                return Result(answer=text, finished=gens[0].finished, trace=trace)
            prefixes = [cands[i].text for i in kept]
        if cfg.finish_paths and prefixes:
            return self._finish_tail(problem, prefixes, trace)
        answer = best_cand.text if best_cand else ""
        return Result(answer=answer, finished=False, trace=trace)

    def _finish_tail(self, problem: str, prefixes: list[str], trace: Trace) -> Result:
        """Final level of the decision tree: expand each surviving prefix into
        ``n_paths`` full completions and let the judge pick among those."""
        cfg = self.config
        cands: list[Candidate] = []
        offset = 0
        for prefix in prefixes:
            gens = self._generate(
                problem,
                prefix,
                n=cfg.n_paths,
                max_tokens=max(1, cfg.max_tokens - _est_tokens(prefix)),
                temperature=cfg.temperature,
                trace=trace,
            )
            new = self._expand(gens, prefix)
            for c in new:
                if c.parent is not None:
                    c.parent += offset
            offset += len(gens)
            cands.extend(new)
        verdict = self._verdict(problem, cands, trace)
        ranked = sorted(
            range(len(cands)),
            key=lambda i: (not cands[i].finished, -verdict.probabilities[i]),
        )
        kept = ranked[: cfg.keep_top_k]
        rnd = Round(
            step=trace.rounds[-1].step + 1 if trace.rounds else 0,
            prefix="\n".join(prefixes),
            candidates=cands,
            verdict=verdict,
            kept=kept,
        )
        trace.rounds.append(rnd)
        best = cands[ranked[0]]
        if best.finished:
            self._verify(problem, best.text, rnd, trace)
        return Result(answer=best.text, finished=best.finished, trace=trace)
