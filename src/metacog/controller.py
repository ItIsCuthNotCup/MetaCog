"""The metacognition loop: branch with the thinker, prune with the judge, repeat."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from .judge import FINISHED_QUESTIONS, Judge
from .split import split_paths
from .thinker import Thinker
from .types import Candidate, Result, Round, Trace, Verdict

MAX_JUDGE_CANDIDATES = 26


def _est_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token). We do not ship a tokenizer, so the
    stepwise token cap is approximate; treat ``max_tokens`` as a soft bound."""
    return len(text) // 4


class Config(BaseModel):
    mode: Literal["best_of_n", "stepwise"] = "stepwise"
    n_paths: int = 4  # samples per kept prefix per step
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


class MetaCog:
    def __init__(self, thinker: Thinker, judge: Judge, config: Config | None = None):
        self.thinker = thinker
        self.judge = judge
        self.config = config or Config()

    # -- internals -----------------------------------------------------------

    def _expand(self, gens, prefix: str) -> list[Candidate]:
        """Turn raw generations (continuations of ``prefix``) into candidates."""
        cands: list[Candidate] = []
        for idx, g in enumerate(gens):
            full = prefix + g.text
            cands.append(Candidate(text=full, finished=g.finished, source="sample", parent=idx))
            if self.config.split_generations:
                for piece in split_paths(g.text):
                    if piece != g.text.strip():
                        cands.append(
                            Candidate(
                                text=prefix + piece,
                                finished=g.finished,
                                source="split",
                                parent=idx,
                            )
                        )
        return cands

    def _verdict(self, problem: str, cands: list[Candidate], trace: Trace) -> Verdict:
        """Judge candidates; more than 26 are truncated in source order (samples first)."""
        if len(cands) == 1:
            return Verdict(probabilities=[1.0], choice=0, confidence=1.0)
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
        return self._stepwise(problem)

    def _best_of_n(self, problem: str) -> Result:
        cfg, trace = self.config, Trace()
        gens = self._generate(
            problem,
            "",
            n=cfg.n_paths,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            trace=trace,
        )
        cands = self._expand(gens, "")
        verdict = self._verdict(problem, cands, trace)
        ranked = sorted(range(len(cands)), key=lambda i: -verdict.probabilities[i])
        kept = ranked[: cfg.keep_top_k]
        rnd = Round(step=0, prefix="", candidates=cands, verdict=verdict, kept=kept)
        trace.rounds.append(rnd)
        best = cands[verdict.choice]
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
            ranked = sorted(range(len(cands)), key=lambda i: -verdict.probabilities[i])
            kept = ranked[: cfg.keep_top_k]
            rnd = Round(
                step=step,
                prefix="\n".join(prefixes),
                candidates=cands,
                verdict=verdict,
                kept=kept,
            )
            trace.rounds.append(rnd)
            best_cand = cands[verdict.choice]
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
        answer = best_cand.text if best_cand else ""
        return Result(answer=answer, finished=False, trace=trace)
