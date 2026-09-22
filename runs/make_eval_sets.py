"""Build the locked evaluation sets committed under docs/eval/.

- docs/eval/confirm_set.json: fresh problems — 60 GPQA Diamond + 40 AIME whose
  sha1 does not appear in ANY saved runs/*_*.json row. Read once per promotion
  decision; never tune on it. Stores benchmark/dataset-index/sha1 only (no
  problem text — GPQA/AIME are not ours to redistribute).
- docs/eval/quick_screen.json: the ~40-row first gate for generation-side
  ideas — rows from the v0.3 runs (adaptP_* = adaptive + answer_prior) where
  MetaCog was wrong or the winner's judge score was < 0.8, balanced across
  thinkers, deterministic order.

Usage: .venv/bin/python runs/make_eval_sets.py   (run from the repo root)
"""

import glob
import hashlib
import json
import os

BENCH_FILES = {
    "gpqa_diamond": "runs/bench/gpqa_diamond.jsonl",
    "aime": "runs/bench/aime_2024_2025.jsonl",
}
V03_GLOB = "runs/adaptP_*.json"
SCORE_FLOOR = 0.8
CONFIRM_COUNTS = {"gpqa_diamond": 60, "aime": 40}
QUICK_MAX = 40


def sha1(text):
    return hashlib.sha1(text.encode()).hexdigest()


def seen_problems():
    """sha1 of every problem text in every saved runs row."""
    seen = set()
    for f in glob.glob("runs/*_*.json"):
        try:
            rows = json.load(open(f))["rows"]
        except Exception:  # noqa: BLE001
            continue
        for r in rows:
            if "problem" in r:
                seen.add(sha1(r["problem"]))
    return seen


def confirm_set(seen):
    out = []
    for bench, path in BENCH_FILES.items():
        items = [json.loads(line) for line in open(path) if line.strip()]
        fresh = [
            {"benchmark": bench, "index": i, "sha1": sha1(d["problem"])}
            for i, d in enumerate(items)
            if sha1(d["problem"]) not in seen
        ]
        take = CONFIRM_COUNTS[bench]
        print(f"{bench}: {len(fresh)} unseen of {len(items)} -> take {min(take, len(fresh))}")
        out.extend(sorted(fresh, key=lambda x: x["sha1"])[:take])
    return out


def quick_screen():
    """v0.3 rows where the pick was wrong or uncertain, balanced across thinkers."""
    per_model = {}
    for f in sorted(glob.glob(V03_GLOB)):
        stem = os.path.basename(f).removesuffix(".json").removeprefix("adaptP_")
        for prefix in ("gpqa_diamond_", "aime_"):
            stem = stem.removeprefix(prefix)
        model = stem
        rows = json.load(open(f))["rows"]
        bench = "aime" if "_aime_" in f else "gpqa_diamond"
        for i, r in enumerate(rows):
            pick = r["metacog"].get("pick", 0)
            cands = r["metacog"].get("candidates", [])
            score = cands[pick].get("score") if pick < len(cands) else None
            ok = r["metacog"]["correct"]
            if ok and (score is None or score >= SCORE_FLOOR):
                continue
            per_model.setdefault(model, []).append(
                {
                    "benchmark": bench,
                    "model": model,
                    "row_index": i,
                    "sha1": sha1(r["problem"]),
                    "v0_3_correct": bool(ok),
                }
            )
    # interleave thinkers (sha1 order inside each) until QUICK_MAX
    for m, lst in per_model.items():
        lst.sort(key=lambda x: x["sha1"])
        print(f"{m}: {len(lst)} flagged rows")
    out = []
    models = sorted(per_model)
    while len(out) < QUICK_MAX and any(per_model[m] for m in models):
        for m in models:
            if per_model[m] and len(out) < QUICK_MAX:
                out.append(per_model[m].pop(0))
    return out


if __name__ == "__main__":
    os.makedirs("docs/eval", exist_ok=True)
    confirm = confirm_set(seen_problems())
    json.dump(confirm, open("docs/eval/confirm_set.json", "w"), indent=1)
    print(f"confirm_set.json: {len(confirm)} problems")
    quick = quick_screen()
    json.dump(quick, open("docs/eval/quick_screen.json", "w"), indent=1)
    print(f"quick_screen.json: {len(quick)} rows")
