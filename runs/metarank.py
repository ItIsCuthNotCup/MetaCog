"""Calibrated judge ensemble (research report item F), offline on saved pools.

Per candidate features: Jev text score t, Jev bare-answer score a, answer-frequency
share f (fraction of the pool's candidates with the same final answer), and, on
split A only, the local LogitJudge text score l. Label = candidate answer == truth.
A regularised logistic model is fit on split A, then read ONCE on split B against
the v0.3 picker (t + 0.5 a). Local-judge features have no B scores, so they are
evaluated only by 5-fold CV inside A (weaker evidence, labelled as such).

Usage: .venv/bin/python runs/metarank.py            (needs TYPESAFE_API_KEY)
"""

from __future__ import annotations

import json
import os
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from math import comb

import numpy as np

os.environ.pop("JUDGE", None)
_src = open("runs/devset.py").read().split("# ---- pickers")[0]
d = types.ModuleType("devset")
exec(compile(_src, "runs/devset.py", "exec"), d.__dict__)  # noqa: S102

CACHE = "runs/devset_cache.json"
LOCAL = "runs/devset_cache_local.json"
cache = d.cache
lock = threading.Lock()


def score_pool(p):
    """Thread-safe version of devset.ensure_scores: compute into a fresh dict."""
    with lock:
        c = dict(cache.get(p["id"], {}))
    o = d.old.get(p["id"], {})
    n = len(p["cands"])
    if not (c.get("text") and len(c["text"]) == n):
        c["text"] = (
            o["tail6k"]
            if o.get("tail6k") and len(o["tail6k"]) == n
            else d.judge.score(p["problem"], [x["text"] for x in p["cands"]]).raw
        )
    if not (c.get("ans") and len(c["ans"]) == n):
        if o.get("ansonly") and len(o["ansonly"]) == n:
            c["ans"] = o["ansonly"]
        else:
            distinct = list(dict.fromkeys(x["answer"] for x in p["cands"]))
            pv = d.judge.score(
                p["problem"], [f"Final answer: {x}" for x in distinct], instructions=d.PRIOR_INSTR
            ).raw
            m = dict(zip(distinct, pv, strict=True))
            c["ans"] = [m[x["answer"]] for x in p["cands"]]
    with lock:
        cache[p["id"]] = c
    return p["id"]


def features(p, c, loc=None):
    n = len(p["cands"])
    answers = [x["answer"] for x in p["cands"]]
    rows, y = [], []
    for i, x in enumerate(p["cands"]):
        f = answers.count(x["answer"]) / n
        row = [c["text"][i], c["ans"][i], f]
        if loc is not None:
            row.append(loc["text"][i])
        rows.append(row)
        y.append(1.0 if x["answer"] == p["truth"] else 0.0)
    return np.array(rows), np.array(y)


def fit_logreg(X, y, l2=1.0, iters=500):
    """L2-regularised logistic regression by Newton's method; bias unpenalised."""
    Xb = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(Xb.shape[1])
    reg = np.full(Xb.shape[1], l2)
    reg[-1] = 0.0
    for _ in range(iters):
        z = Xb @ w
        p = 1 / (1 + np.exp(-z))
        g = Xb.T @ (p - y) + reg * w
        h = (Xb * (p * (1 - p))[:, None]).T @ Xb + np.diag(reg)
        step = np.linalg.solve(h, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def predict(w, X):
    Xb = np.hstack([X, np.ones((len(X), 1))])
    return 1 / (1 + np.exp(-(Xb @ w)))


def pick_prior(p, c):
    s = [t + 0.5 * a for t, a in zip(c["text"], c["ans"], strict=True)]
    return int(np.argmax(s))


def evaluate(pools, w, cols, name, loc_cache=None):
    right = fixed = lost = 0
    brier_m = brier_p = 0.0
    for p in pools:
        c = cache[p["id"]]
        loc = loc_cache.get(p["id"]) if loc_cache else None
        X, y = features(p, c, loc)
        pm = predict(w, X[:, cols])
        i = int(np.argmax(pm))
        j = pick_prior(p, c)
        ok, ref = y[i] == 1, y[j] == 1
        right += ok
        fixed += ok and not ref
        lost += (not ok) and ref
        brier_m += (pm[i] - y[i]) ** 2
        brier_p += (c["text"][j] - y[j]) ** 2
    n = len(pools)
    ceil = sum(any(x["answer"] == p["truth"] for x in p["cands"]) for p in pools)
    prior_right = right - fixed + lost
    print(
        f"{name:22s} n={n:3d} ceiling={ceil:3d} prior={prior_right:3d} model={right:3d} "
        f"fixed={fixed:2d} lost={lost:2d} p={mcnemar(fixed, lost):.3f} "
        f"brier model={brier_m / n:.3f} jev-text={brier_p / n:.3f}"
    )


def stack(pools, loc_cache=None):
    Xs, ys = [], []
    for p in pools:
        X, y = features(p, cache[p["id"]], loc_cache.get(p["id"]) if loc_cache else None)
        Xs.append(X)
        ys.append(y)
    return np.vstack(Xs), np.concatenate(ys)


def main():
    pools = d.load_pools()
    A = [p for p in pools if p["split"] == "A"]
    B = [p for p in pools if p["split"] == "B"]
    print(f"{len(pools)} disagreeing pools: A={len(A)} B={len(B)}", file=sys.stderr)
    with ThreadPoolExecutor(8) as ex:
        for k, _ in enumerate(ex.map(score_pool, pools), 1):
            if k % 20 == 0:
                with lock:
                    json.dump(cache, open(CACHE, "w"))
                print(f"scored {k}/{len(pools)}", file=sys.stderr, flush=True)
    json.dump(cache, open(CACHE, "w"))

    XA, yA = stack(A)
    for name, cols in [("t+a (learned w)", [0, 1]), ("t+a+freq", [0, 1, 2]), ("t+freq", [0, 2])]:
        w = fit_logreg(XA[:, cols], yA)
        print(f"\n== {name}: weights {np.round(w, 3).tolist()} (last = bias)")
        evaluate(A, w, cols, "  fit split A (in-sample)")
        evaluate(B, w, cols, "  READ split B (held-out)")

    # local judge: 5-fold CV inside A only (no B scores exist)
    loc = json.load(open(LOCAL)) if os.path.exists(LOCAL) else {}
    AL = [p for p in A if p["id"] in loc and len(loc[p["id"]].get("text", [])) == len(p["cands"])]
    if len(AL) >= 40:
        print(f"\n== local-judge fusion, 5-fold CV inside A ({len(AL)} pools; weaker evidence)")
        rng = np.random.default_rng(0)
        order = rng.permutation(len(AL))
        for name, cols in [("t+a+freq", [0, 1, 2]), ("t+a+freq+local", [0, 1, 2, 3])]:
            right = fixed = lost = 0
            for f in range(5):
                test_idx = set(order[f::5].tolist())
                train = [AL[i] for i in range(len(AL)) if i not in test_idx]
                test = [AL[i] for i in sorted(test_idx)]
                X, y = stack(train, loc)
                w = fit_logreg(X[:, cols], y)
                for p in test:
                    Xp, yp = features(p, cache[p["id"]], loc[p["id"]])
                    i = int(np.argmax(predict(w, Xp[:, cols])))
                    j = pick_prior(p, cache[p["id"]])
                    ok, ref = yp[i] == 1, yp[j] == 1
                    right += ok
                    fixed += ok and not ref
                    lost += (not ok) and ref
            print(
                f"  {name:18s} model={right:3d}/{len(AL)} vs prior fixed={fixed} lost={lost} "
                f"p={mcnemar(fixed, lost):.3f}"
            )


if __name__ == "__main__":
    main()
