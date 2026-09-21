"""Live demo: wrap any OpenAI-compatible thinker in MetaCog with a Jev judge and dump full traces.

Env: THINKER_URL, THINKER_MODEL, THINKER_KEY, THINKER_MAX_TOKENS, N_PATHS, PROBLEM_SET=hard|harder or PROBLEMS_FILE=path.jsonl, OUT (resumable).
Render the JSON files with examples/render_demo.py.
"""

import json
import os
import re
import sys
import time

from metacog import Config, MetaCog, OpenAICompatThinker, SystemOneJudge
from metacog.thinker import ThinkerError

SYSTEM = (
    "Solve the problem. Think step by step but briefly, then finish with a final line "
    "of the form 'Answer: <number>' (or 'Answer: <letter>' for multiple choice)."
)

PROBLEMS = [
    (
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How many cents does the ball cost?",
        "5",
    ),
    (
        "If 5 machines take 5 minutes to make 5 widgets, how many minutes would 100 machines take to make 100 widgets?",
        "5",
    ),
    (
        "A clock shows 3:15. What is the smaller angle in degrees between the hour and minute hands?",
        "7.5",
    ),
    (
        "Janet's ducks lay 16 eggs per day. She eats three for breakfast and bakes muffins with four. She sells the rest at $2 each. How many dollars does she make per day?",
        "18",
    ),
    (
        "A train travels 60 miles in 1.5 hours, then 90 miles in 1.5 hours. What is its average speed in mph for the whole trip?",
        "50",
    ),
    ("The sum of three consecutive even integers is 78. What is the largest of the three?", "28"),
    (
        "A shirt is discounted 20%, then the sale price is discounted a further 25%. What is the total percent discount off the original price?",
        "40",
    ),
    ("How many positive divisors does 36 have?", "9"),
    (
        "Alice has 3 brothers and 2 sisters. How many sisters does one of Alice's brothers have?",
        "3",
    ),
    ("What is the 1000th digit after the decimal point in the decimal expansion of 1/7?", "8"),
    ("How many integers from 1 to 1000 inclusive are divisible by 3 or 5 but not by 15?", "401"),
    ("What is the sum of all two-digit positive integers whose digits add up to 9?", "486"),
    (
        "A snail at the bottom of a 10-foot well climbs 3 feet each day and slides back 2 feet each night. On which day does it first reach the top?",
        "8",
    ),
    ("What is the smallest positive integer with exactly 12 positive divisors?", "60"),
]


HARD = [
    ("How many positive integers n with 1 <= n <= 1000 make n^2 + 1 divisible by 5?", "400"),
    ("What is the sum of the decimal digits of 2^20?", "31"),
    ("How many distinct arrangements are there of the letters of MISSISSIPPI?", "34650"),
    ("How many prime numbers are strictly less than 200?", "46"),
    ("How many lattice points (x, y) with integer coordinates satisfy x^2 + y^2 <= 25?", "81"),
    (
        "How many ordered triples of positive integers (x, y, z) satisfy x + y + z = 20 with x <= 10?",
        "135",
    ),
    ("What is the smallest positive integer n such that n! ends in more than 100 zeros?", "410"),
    ("How many non-empty subsets of {1, 2, ..., 10} contain no two consecutive integers?", "143"),
    (
        "How many Pythagorean triples (a, b, c) with a <= b < c and all of a, b, c less than 100 satisfy a^2 + b^2 = c^2?",
        "50",
    ),
    ("What are the last two digits of 7^2024? Give the answer as a number (e.g. 01 -> 1).", "1"),
    ("What is the sum of all positive divisors of 360?", "1170"),
    ("How many three-digit numbers n are palindromes such that n^2 is also a palindrome?", "5"),
]
# Ground truths brute-forced by examples/verify_harder.py.
HARDER = [
    ("How many positive integers n with 1 <= n <= 1000 make n^2 + n + 1 prime?", "189"),
    (
        "How many ordered pairs of integers (a, b) with 1 <= a, b <= 100 have gcd(a, b) = 1 and a + b divisible by 7?",
        "756",
    ),
    (
        "How many six-digit positive integers have strictly increasing digits and a digit sum divisible by 3?",
        "30",
    ),
    (
        "How many integers n with 1 <= n <= 2024 can be written as x^2 - y^2 for non-negative integers x, y?",
        "1518",
    ),
    (
        "How many permutations p of {1, ..., 8} have no fixed point (p(i) != i for all i) and satisfy p(1) = 2?",
        "2119",
    ),
    ("How many binary strings of length 12 contain no three consecutive equal bits?", "466"),
    (
        "How many lattice paths from (0, 0) to (8, 8) using unit steps right or up never pass through (4, 4)?",
        "7970",
    ),
    ("How many ordered triples of positive integers (a, b, c) satisfy a * b * c = 720?", "270"),
    (
        "For how many integers n with 1 <= n <= 500 does n! have exactly 3 more trailing zeros than (n-1)!?",
        "4",
    ),
    (
        "How many 4x4 matrices with entries in {0, 1} have every row sum and every column sum even?",
        "512",
    ),
    ("What is the sum of the decimal digits of 3^100?", "153"),
    (
        "How many positive integers less than or equal to 10000 have a digit sum of exactly 20?",
        "633",
    ),
    (
        "How many subsets of {1, 2, ..., 12} (including the empty set) have an element sum divisible by 12?",
        "344",
    ),
    ("How many primes p < 1000 are such that p + 2 and p + 6 are also prime?", "15"),
    ("What are the last three digits of 3^2024? Give the answer as a number.", "481"),
]
PROBLEM_SETS = {"hard": HARD, "harder": HARDER}
if os.environ.get("PROBLEM_SET") in PROBLEM_SETS:
    PROBLEMS = PROBLEM_SETS[os.environ["PROBLEM_SET"]]
if os.environ.get("PROBLEMS_FILE"):  # JSONL rows: {"problem": ..., "answer": ...}
    with open(os.environ["PROBLEMS_FILE"]) as fh:
        PROBLEMS = [(r["problem"], str(r["answer"])) for r in map(json.loads, fh) if r]


def final_answer(text: str, truth: str = "0") -> str | None:
    if re.fullmatch(r"[A-J]", truth):  # multiple choice
        m = re.findall(r"(?:Answer:\s*\(?|\\boxed\{)\s*([A-J])\b", text)
    else:
        m = re.findall(r"Answer:\s*\$?\s*(-?\d+(?:\.\d+)?)", text)
    return m[-1] if m else None


def correct(text: str, truth: str) -> bool:
    a = final_answer(text, truth)
    if a is None:
        return False
    if re.fullmatch(r"[A-J]", truth):
        return a == truth
    return abs(float(a) - float(truth)) < 1e-6


def main() -> None:
    url = os.environ.get("THINKER_URL", "http://100.119.198.25:8031")
    model = os.environ.get("THINKER_MODEL", "minicpm5-2b-casual")
    max_tokens = int(os.environ.get("THINKER_MAX_TOKENS", "700"))
    out_path = os.environ.get("OUT", "live_demo.json")
    thinker = OpenAICompatThinker(
        url,
        model=model,
        api_key=os.environ.get("THINKER_KEY"),
        system_prompt=SYSTEM,
        stream=os.environ.get("THINKER_STREAM", "1") != "0",
    )
    judge = SystemOneJudge.jev()
    mc = MetaCog(
        thinker,
        judge,
        Config(
            mode="best_of_n",
            n_paths=int(os.environ.get("N_PATHS", "3")),
            max_tokens=max_tokens,
            temperature=0.9,
            greedy_anchor=True,  # candidate 0 IS the baseline: the judge can only gain
        ),
    )
    out = []
    if os.path.exists(out_path):  # resume a run interrupted by provider errors
        out = json.load(open(out_path))["rows"]
    done = {r["problem"] for r in out}
    for i, (problem, truth) in enumerate(PROBLEMS):
        if problem in done:
            continue
        t0 = time.time()
        try:
            res = mc.run(problem)
        except ThinkerError as e:
            print(f"[{i + 1}/{len(PROBLEMS)}] skipped: {str(e)[:200]}", flush=True)
            continue
        t2 = time.time()
        rnd = res.trace.rounds[0] if res.trace.rounds else None
        # the greedy anchor candidate doubles as the baseline row (no separate call)
        base = next(
            (c for c in rnd.candidates if c.source == "greedy"),
            rnd.candidates[0] if rnd and rnd.candidates else None,
        )
        row = {
            "problem": problem,
            "truth": truth,
            "baseline": {
                "text": base.text if base else res.answer,
                "answer": final_answer(base.text if base else res.answer, truth),
                "correct": correct(base.text if base else res.answer, truth),
                "seconds": None,  # folded into the metacog call; no separate timing
            },
            "metacog": {
                "answer": final_answer(res.answer, truth),
                "correct": correct(res.answer, truth),
                "seconds": round(t2 - t0, 1),
                "pick": rnd.kept[0] if rnd else 0,
                "candidates": [
                    {
                        "text": c.text,
                        "source": c.source,
                        "finished": c.finished,
                        "answer": final_answer(c.text, truth),
                        "correct": correct(c.text, truth),
                        "score": (rnd.verdict.raw or rnd.verdict.probabilities)[k],
                    }
                    for k, c in enumerate(rnd.candidates)
                ]
                if rnd
                else [
                    {
                        "text": res.answer,
                        "source": "sample",
                        "finished": res.finished,
                        "answer": final_answer(res.answer, truth),
                        "correct": correct(res.answer, truth),
                        "score": None,
                    }
                ],
                "judge_calls": res.trace.judge_calls,
                "thinker_tokens": res.trace.thinker_tokens,
            },
        }
        out.append(row)
        print(
            f"[{i + 1}/{len(PROBLEMS)}] base={row['baseline']['answer']} ({row['baseline']['correct']}) metacog={row['metacog']['answer']} ({row['metacog']['correct']}) truth={truth}",
            flush=True,
        )
        json.dump({"thinker": model, "url": url, "rows": out}, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
