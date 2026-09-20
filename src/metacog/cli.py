"""``metacog solve "<problem>" ...`` command line interface."""

from __future__ import annotations

import argparse
import sys

from .controller import Config, MetaCog
from .judge import SystemOneJudge
from .thinker import OpenAICompatThinker


def _build_thinker(args) -> OpenAICompatThinker:
    return OpenAICompatThinker(
        base_url=args.thinker_url,
        model=args.thinker_model,
        api=args.thinker_api,
        api_key=args.thinker_api_key,
        system_prompt=args.system_prompt,
    )


def _build_judge(args) -> SystemOneJudge:
    if args.judge == "jev":
        judge = SystemOneJudge.jev(model=args.judge_model or "jev-latest")
    else:
        judge = SystemOneJudge.reflex(
            base_url=args.judge_url or "http://localhost:8008",
            model=args.judge_model or "reflex-latest",
        )
    if args.judge_url and args.judge == "jev":
        judge.base_url = args.judge_url.rstrip("/")
    return judge


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--thinker-url", required=True)
    p.add_argument("--thinker-model", required=True)
    p.add_argument("--thinker-api", choices=["chat", "completions"], default="chat")
    p.add_argument("--thinker-api-key", default=None)
    p.add_argument("--system-prompt", default=None)
    p.add_argument("--judge", choices=["jev", "reflex"], default="reflex")
    p.add_argument("--judge-url", default=None)
    p.add_argument("--judge-model", default=None)
    p.add_argument("--mode", choices=["best_of_n", "stepwise"], default="stepwise")
    p.add_argument("--n", type=int, default=4, dest="n_paths")
    p.add_argument("--keep-top-k", type=int, default=1)
    p.add_argument("--step-tokens", type=int, default=256)
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--stop", nargs="*", default=None)
    p.add_argument("--no-split", action="store_true")
    p.add_argument("--commit-confidence", type=float, default=None)
    p.add_argument("--verify-finished", action="store_true")


def _config(args) -> Config:
    return Config(
        mode=args.mode,
        n_paths=args.n_paths,
        keep_top_k=args.keep_top_k,
        step_tokens=args.step_tokens,
        max_steps=args.max_steps,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        stop=args.stop,
        split_generations=not args.no_split,
        commit_confidence=args.commit_confidence,
        verify_finished=args.verify_finished,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="metacog")
    sub = ap.add_subparsers(dest="cmd", required=True)
    solve = sub.add_parser("solve", help="run the metacognition loop on one problem")
    solve.add_argument("problem")
    _add_common(solve)
    solve.add_argument("--json", action="store_true", help="print full Result JSON")
    args = ap.parse_args(argv)

    mc = MetaCog(_build_thinker(args), _build_judge(args), _config(args))
    result = mc.run(args.problem)
    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        print(result.answer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
