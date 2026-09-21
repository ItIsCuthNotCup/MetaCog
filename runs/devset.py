"""Dev set for fast, statistically honest iteration on the *picking* side.

Pools: every saved row where finished full candidates disagree on the final
answer (the only rows a picking change can flip). Each pool is assigned to
split A (tune) or B (test) by a stable hash of its problem text, so the same
problem never lands in both. Report on A while iterating; look at B once.

Variants evaluated against the current best arm (text noul + 0.5 * bare-answer
prior = `prior`):
  text     : Jev noul on the full text only (v0.2 pick)
  prior    : text + 0.5 * bare-answer prior              (current best arm)
  group    : #2 — sum prior-scores per distinct answer, pick the best path
             inside the winning answer group
  tie      : #4 — when the top-2 prior totals are within TIE_MARGIN, one Jev
             `choice` call on those two full texts decides
  group+tie: both

Jev calls are cached in runs/devset_cache.json; no thinker calls are made.
Usage: .venv/bin/python runs/devset.py [A|B|all]
"""

import glob
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from math import comb

sys.path.insert(0, "src")
sys.path.insert(0, "examples")
from live_demo import final_answer  # noqa: E402

from metacog.judge import SystemOneJudge  # noqa: E402

CACHE = "runs/devset_cache.json"
OLD = "runs/rejudge.json"
TIE_MARGIN = 0.05
PRIOR_W = 0.5
PRIOR_INSTR = "`path` states only a proposed final answer to `problem`. Is it correct?"

cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
old = json.load(open(OLD)) if os.path.exists(OLD) else {}


def norm(a, truth):
    if a is None:
        return None
    if re.fullmatch(r"[A-J]", truth):
        return a
    try:
        return str(float(a))
    except ValueError:
        return None


def load_pools():
    pools = {}
    for f in sorted(glob.glob("runs/*_*.json")):
        if "rejudge" in f or "devset" in f:
            continue
        try:
            rs = json.load(open(f))["rows"]
        except Exception:  # noqa: BLE001
            continue
        for j, r in enumerate(rs):
            truth = norm(r["truth"], r["truth"])
            cands = []
            for c in r["metacog"]["candidates"]:
                if c["source"] in ("sketch", "split") or not c["finished"]:
                    continue
                a = norm(c.get("answer") or final_answer(c["text"], r["truth"]), r["truth"])
                if a is not None:
                    cands.append({"text": c["text"], "answer": a})
            if len({c["answer"] for c in cands}) > 1:
                h = hashlib.sha1(r["problem"].encode()).hexdigest()
                pools[f"{f}#{j}"] = {
                    "id": f"{f}#{j}",
                    "split": "A" if int(h[:8], 16) % 2 == 0 else "B",
                    "problem": r["problem"],
                    "truth": truth,
                    "cands": cands,
                }
    return list(pools.values())


judge = SystemOneJudge.jev()


def ensure_scores(p):
    """Fill text + prior scores (per candidate) into cache; reuse rejudge.json where possible."""
    c = cache.setdefault(p["id"], {})
    o = old.get(p["id"], {})
    if "text" not in c:
        if o.get("tail6k") and len(o["tail6k"]) == len(p["cands"]):
            c["text"] = o["tail6k"]
        else:
            c["text"] = judge.score(p["problem"], [x["text"] for x in p["cands"]]).raw
    if "ans" not in c:
        if o.get("ansonly") and len(o["ansonly"]) == len(p["cands"]):
            c["ans"] = o["ansonly"]
        else:
            distinct = list(dict.fromkeys(x["answer"] for x in p["cands"]))
            pv = judge.score(
                p["problem"], [f"Final answer: {a}" for a in distinct], instructions=PRIOR_INSTR
            ).raw
            m = dict(zip(distinct, pv, strict=True))
            c["ans"] = [m[x["answer"]] for x in p["cands"]]
    return p["id"]


def totals(p):
    c = cache[p["id"]]
    return [t + PRIOR_W * a for t, a in zip(c["text"], c["ans"], strict=True)]


def ensure_tie(p):
    """#4: pairwise Jev choice on the top-2 by total when they are within TIE_MARGIN."""
    c = cache[p["id"]]
    tot = totals(p)
    order = sorted(range(len(tot)), key=lambda i: -tot[i])
    i, j = order[0], order[1]
    key = f"tie:{i},{j}"
    if tot[i] - tot[j] <= TIE_MARGIN and key not in c:
        v = judge.choose(p["problem"], [p["cands"][i]["text"], p["cands"][j]["text"]])
        c[key] = v.probabilities
    return p["id"]


# ---- pickers ---------------------------------------------------------------
def pick_text(p):
    s = cache[p["id"]]["text"]
    return p["cands"][max(range(len(s)), key=lambda k: s[k])]["answer"]


def pick_prior(p):
    s = totals(p)
    return p["cands"][max(range(len(s)), key=lambda k: s[k])]["answer"]


def pick_group(p):
    s = totals(p)
    g = {}
    for x, sc in zip(p["cands"], s, strict=True):
        g[x["answer"]] = g.get(x["answer"], 0.0) + sc
    return max(g, key=g.get)


def pick_tie(p, base=pick_prior):
    s = totals(p)
    order = sorted(range(len(s)), key=lambda k: -s[k])
    i, j = order[0], order[1]
    pr = cache[p["id"]].get(f"tie:{i},{j}")
    if s[i] - s[j] <= TIE_MARGIN and pr:
        return p["cands"][i if pr[0] >= pr[1] else j]["answer"]
    return base(p)


def pick_group_tie(p):
    return pick_tie(p, base=pick_group)


PICKERS = {
    "text": pick_text,
    "prior": pick_prior,
    "group": pick_group,
    "tie": pick_tie,
    "group+tie": pick_group_tie,
}


def mcnemar(b, c):
    """Exact two-sided McNemar p for discordant counts b (fixed) and c (lost)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def report(pools, split):
    rows = [p for p in pools if split == "all" or p["split"] == split]
    ceil = sum(any(x["answer"] == p["truth"] for x in p["cands"]) for p in rows)
    print(f"\nsplit {split}: {len(rows)} disagreeing pools, ceiling {ceil}")
    print(f"{'picker':10s} {'right':>5s} {'fixed':>5s} {'lost':>4s} {'p':>7s}   (vs prior)")
    ref = {p["id"]: pick_prior(p) == p["truth"] for p in rows}
    for name, fn in PICKERS.items():
        right = fixed = lost = 0
        for p in rows:
            ok = fn(p) == p["truth"]
            right += ok
            fixed += ok and not ref[p["id"]]
            lost += (not ok) and ref[p["id"]]
        print(f"{name:10s} {right:5d} {fixed:5d} {lost:4d} {mcnemar(fixed, lost):7.3f}")


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "A"
    pools = load_pools()
    na = sum(p["split"] == "A" for p in pools)
    print(f"{len(pools)} pools: A={na} B={len(pools) - na}", file=sys.stderr)
    with ThreadPoolExecutor(8) as ex:
        for n, _ in enumerate(ex.map(ensure_scores, pools), 1):
            if n % 20 == 0:
                json.dump(cache, open(CACHE, "w"))
        json.dump(cache, open(CACHE, "w"))
        for _ in ex.map(ensure_tie, pools):
            pass
    json.dump(cache, open(CACHE, "w"))
    for s in ["A", "B", "all"] if split == "all" else [split]:
        report(pools, s)
