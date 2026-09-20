"""Agent-in-the-loop use: any agent (human, Devin, Claude) writes several independent drafts;
Jev scores each one blind and MetaCog picks. Ground truth included so you can check the pick."""

import json

from metacog import SystemOneJudge

PROBLEMS = [
    {
        "problem": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
        "How much does the ball cost, in cents?",
        "truth": "5",
        "candidates": [
            "The total is 1.10 and the bat is 1.00 more, so the ball is 0.10. Answer: 10 cents.",
            "Let ball = x. Bat = x + 1.00. Sum: 2x + 1.00 = 1.10, so 2x = 0.10, x = 0.05. "
            "Check: 0.05 + 1.05 = 1.10. Answer: 5 cents.",
            "Ball = x, bat = 1.00 + x, total 1.10 => x = 0.10 / 2 = 0.05... but the bat must be "
            "exactly a dollar, so the ball is 10 cents. Answer: 10 cents.",
        ],
    },
    {
        "problem": "How many times does the letter 'r' appear in the word 'strawberry'?",
        "truth": "3",
        "candidates": [
            "s-t-r-a-w-b-e-r-r-y. r at position 3, 8, 9. Answer: 3.",
            "strawberry has 'straw' (one r) and 'berry' (one r). Answer: 2.",
            "Count: st(r)awbe(r)(r)y -> 3. Answer: 3.",
        ],
    },
    {
        "problem": "In Python, what does `sorted([3, 1, 2], key=lambda x: -x)[0]` evaluate to?",
        "truth": "3",
        "candidates": [
            "key=-x sorts descending, so the list becomes [3, 2, 1]; index 0 is 3. Answer: 3.",
            "sorted is ascending by default; key only maps values for comparison, so the smallest "
            "original value comes first. Answer: 1.",
            "The key negates each element: [-3, -1, -2]. Sorted ascending: [-3, -2, -1], which "
            "corresponds to original [3, 2, 1]. First element: 3. Answer: 3.",
        ],
    },
    {
        "problem": "A clock shows 3:15. What is the angle in degrees between the hour and "
        "minute hands (smaller angle)?",
        "truth": "7.5",
        "candidates": [
            "At 3:15 the minute hand is at 90 degrees (on the 3) and the hour hand is on the 3 "
            "as well, so the angle is 0. Answer: 0.",
            "Minute hand: 15 * 6 = 90 degrees. Hour hand: 3 * 30 + 15 * 0.5 = 97.5 degrees. "
            "Difference: 7.5 degrees. Answer: 7.5.",
            "Minute hand at 90. Hour hand moves 30 degrees per hour, so at 3:15 it is at "
            "3.25 * 30 = 97.5. Angle = 97.5 - 90 = 7.5. Answer: 7.5.",
            "Hour hand at 3 = 90 degrees; minute hand at 15 min = 15 * 6 = 90. But the hour "
            "hand also moves: 15 minutes = 1/4 hour = 7.5 degrees, giving 82.5. "
            "Angle = 82.5 - 90 = -7.5, so 7.5. Actually hour hand is at 90 - 7.5 = 82.5? "
            "No: it advances, so 97.5. "
            "Answer: 15 (rounding up from 7.5 for the reflex angle).",
        ],
    },
]


def main() -> None:
    judge = SystemOneJudge.jev()
    rows = []
    for p in PROBLEMS:
        v = judge.score(p["problem"], p["candidates"])
        pick = p["candidates"][v.choice]
        correct = f"Answer: {p['truth']}" in pick
        rows.append(
            {
                "problem": p["problem"][:60],
                "raw": [round(r, 2) for r in v.raw],
                "pick": v.choice,
                "pick_correct": correct,
                "n_correct_candidates": sum(f"Answer: {p['truth']}" in c for c in p["candidates"]),
            }
        )
        print(json.dumps(rows[-1]))
    print("picked correctly:", sum(r["pick_correct"] for r in rows), "/", len(rows))


if __name__ == "__main__":
    main()
