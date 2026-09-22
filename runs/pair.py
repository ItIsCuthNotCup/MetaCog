"""Paired comparison of one arm against another on identical (bench, model, problem).
Usage: .venv/bin/python runs/pair.py divP adaptP   (ref arm '' = current MetaCog files)"""

import glob
import json
import sys
from math import comb


def pval(fixed, lost):
    n = fixed + lost
    if not n:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(fixed, lost) + 1)) / 2**n)


def load(arm, bench, model):
    name = f"runs/{arm + '_' if arm else ''}{bench}_{model}.json"
    try:
        return json.load(open(name))["rows"]
    except Exception:  # noqa: BLE001
        return []


arm, ref = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "")
models = sorted(
    {f.split(f"{arm}_")[1].split("_", 2)[-1][:-5] for f in glob.glob(f"runs/{arm}_*.json")}
)
tot = dict(n=0, e=0, r=0, b=0, f=0, l=0, se=0.0, sr=0.0, te=0, tr=0, esc=0)
for bench in ("gpqa_diamond", "aime"):
    for m in models:
        m = m.replace(f"{bench}_", "")
        e_rows = load(arm, bench, m)
        r_rows = {r["problem"]: r for r in load(ref, bench, m)}
        fixed = lost = n = 0
        for e in e_rows:
            r = r_rows.get(e["problem"])
            if not r:
                continue
            n += 1
            ec, rc = e["metacog"]["correct"], r["metacog"]["correct"]
            fixed += ec and not rc
            lost += rc and not ec
            tot["e"] += ec
            tot["r"] += rc
            tot["b"] += e["baseline"]["correct"]
            tot["se"] += e["metacog"]["seconds"] or 0
            tot["sr"] += r["metacog"]["seconds"] or 0
            tot["te"] += e["metacog"]["thinker_tokens"] or 0
            tot["tr"] += r["metacog"]["thinker_tokens"] or 0
            tot["esc"] += bool(e["metacog"].get("escalated"))
        tot["n"] += n
        tot["f"] += fixed
        tot["l"] += lost
        if n:
            print(
                f"{bench:13s} {m:32s} n={n:3d} fixed/lost vs {ref or 'current'}: +{fixed}/-{lost}"
            )
n = tot["n"] or 1
print(
    f"\n{arm} vs {ref or 'current'}: n={tot['n']}  "
    f"alone {100 * tot['b'] / n:.1f}%  ref {100 * tot['r'] / n:.1f}%  "
    f"{arm} {100 * tot['e'] / n:.1f}%  +{tot['f']}/-{tot['l']} "
    f"p={pval(tot['f'], tot['l']):.2f}  "
    f"time {tot['se'] / max(tot['sr'], 1):.2f}x  "
    f"tokens {tot['te'] / max(tot['tr'], 1):.2f}x  escalated {tot['esc']}"
)
