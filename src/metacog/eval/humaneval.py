"""HumanEval evaluation harness: ``metacog-eval humaneval ...``.

WARNING: this module EXECUTES model-generated Python code
(``run_test`` runs ``code + test + check(entry_point)`` in a subprocess).
Only run it inside a disposable container or VM with no secrets mounted.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

from ..controller import Config, MetaCog
from ..judge import SystemOneJudge
from ..thinker import OpenAICompatThinker
from ..types import Trace

INSTRUCTION = (
    "Complete the following Python function. "
    "Return the full function in a single ```python code block."
)

_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def build_problem(prompt: str) -> str:
    """The HumanEval prompt preceded by the fixed instruction line."""
    return f"{INSTRUCTION}\n\n{prompt}"


def extract_code(text: str, prompt: str, entry_point: str) -> str:
    """Last ```python block if present, else raw text; if the extracted code does
    not define ``entry_point``, the original HumanEval prompt is prepended."""
    blocks = _CODE_BLOCK_RE.findall(text)
    code = blocks[-1].strip() if blocks else text.strip()
    if f"def {entry_point}" not in code:
        code = prompt.rstrip() + "\n" + code
    return code


def run_test(code: str, test: str, entry_point: str, timeout: int = 10) -> bool:
    """Run model-generated code in a subprocess. ONLY CALL INSIDE A CONTAINER/VM."""
    program = code + "\n" + test + f"\ncheck({entry_point})\n"
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    try:
        proc = subprocess.run(
            [sys.executable, "-c", program],
            timeout=timeout,
            cwd=tempfile.mkdtemp(prefix="metacog-"),
            env=env,
            capture_output=True,
        )
    except subprocess.TimeoutExpired:
        return False
    return proc.returncode == 0


def _load_tasks(data: str | None, limit: int | None) -> list[dict]:
    if data:
        tasks = [json.loads(line) for line in open(data) if line.strip()]
    else:
        from datasets import load_dataset  # lazy: requires the ``eval`` extra

        tasks = list(load_dataset("openai_humaneval", split="test"))
    return tasks[:limit] if limit else tasks


def _run_task(task: dict, mc: MetaCog | None) -> dict:
    problem = build_problem(task["prompt"])
    result = mc.run(problem)
    code = extract_code(result.answer, task["prompt"], task["entry_point"])
    passed = run_test(code, task["test"], task["entry_point"])
    return {
        "task_id": task["task_id"],
        "passed": passed,
        "answer": result.answer,
        "finished": result.finished,
        "trace": _trace_summary(result.trace),
    }


def _trace_summary(trace: Trace) -> dict:
    return {
        "rounds": len(trace.rounds),
        "thinker_calls": trace.thinker_calls,
        "judge_calls": trace.judge_calls,
        "thinker_tokens": trace.thinker_tokens,
    }


class _BaselineMetaCog:
    """--judge none: single greedy sample, no judging."""

    def __init__(self, thinker: OpenAICompatThinker):
        self.thinker = thinker

    def run(self, problem: str):
        from ..types import Result

        gens = self.thinker.generate(problem, "", n=1, max_tokens=2048, temperature=0.0, stop=None)
        trace = Trace(thinker_calls=1, thinker_tokens=sum(g.tokens for g in gens))
        return Result(answer=gens[0].text, finished=gens[0].finished, trace=trace)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="metacog-eval")
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("humaneval", help="evaluate on HumanEval")
    h.add_argument("--thinker-url", required=True)
    h.add_argument("--thinker-model", required=True)
    h.add_argument("--thinker-api", choices=["chat", "completions"], default="chat")
    h.add_argument("--thinker-api-key", default=None)
    h.add_argument("--judge", choices=["jev", "reflex", "none"], default="reflex")
    h.add_argument("--judge-url", default=None)
    h.add_argument("--judge-model", default=None)
    h.add_argument("--mode", choices=["best_of_n", "stepwise"], default="stepwise")
    h.add_argument("--strategy", choices=["noul", "choice"], default="noul")
    h.add_argument("--n", type=int, default=4, dest="n_paths")
    h.add_argument("--limit", type=int, default=None)
    h.add_argument("--data", default=None, help="local JSONL with HumanEval fields")
    h.add_argument("--out", default="runs/humaneval.jsonl")
    h.add_argument("--workers", type=int, default=1)
    h.add_argument("--step-tokens", type=int, default=256)
    h.add_argument("--max-steps", type=int, default=8)
    h.add_argument("--temperature", type=float, default=0.8)
    h.add_argument("--test-timeout", type=int, default=10)
    args = ap.parse_args(argv)

    thinker = OpenAICompatThinker(
        base_url=args.thinker_url,
        model=args.thinker_model,
        api=args.thinker_api,
        api_key=args.thinker_api_key,
    )
    if args.judge == "none":
        mc: MetaCog | _BaselineMetaCog = _BaselineMetaCog(thinker)
    else:
        if args.judge == "jev":
            judge = SystemOneJudge.jev(model=args.judge_model or "jev-latest")
            if args.judge_url:
                judge.base_url = args.judge_url.rstrip("/")
        else:
            judge = SystemOneJudge.reflex(
                base_url=args.judge_url or "http://localhost:8008",
                model=args.judge_model or "reflex-latest",
            )
        mc = MetaCog(
            thinker,
            judge,
            Config(
                mode=args.mode,
                strategy=args.strategy,
                n_paths=args.n_paths,
                step_tokens=args.step_tokens,
                max_steps=args.max_steps,
                temperature=args.temperature,
            ),
        )

    tasks = _load_tasks(args.data, args.limit)
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    results: list[dict] = []
    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(lambda t: _run_task(t, mc), tasks))
    else:
        for t in tasks:
            r = _run_task(t, mc)
            results.append(r)
            print(f"{r['task_id']}: {'PASS' if r['passed'] else 'FAIL'}", flush=True)

    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    passed = sum(1 for r in results if r["passed"])
    print(f"pass@1 = {passed}/{len(results)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
