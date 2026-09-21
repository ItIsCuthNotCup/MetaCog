"""Regenerate runs/dashboard.html every 30s from runs/<bench>_<model>.json progress."""

import json
import time
from datetime import datetime, timezone

BENCHES = {
    "harder": ("Custom harder counting set", 15),
    "math500_l5_int": ("MATH-500 level 5 (integer-answer subset)", 74),
    "gpqa_diamond": ("GPQA Diamond", 198),
    "aime": ("AIME 2024 + 2025", 60),
}
MODELS = [
    "deepseek_deepseek-v4-flash", "Qwen_Qwen3.8-Flash", "z-ai_glm-5.3-flash", "moonshotai_Kimi-K2.5",
    "MiniMaxAI_MiniMax-M2.5", "xiaomi_mimo-v2.5", "stepfun_Step-3.5-Flash", "meituan_LongCat-2.0",
    "zai-org_GLM-5.1", "Qwen_Qwen3.7-Flash",
]
JEV_USD = 0.000058  # per Jev decision (TypeSafe pricing as measured on the HumanEval runs)
CSS = """tr.grp td{background:#eef2f7;padding-top:12px}
body{font:14px system-ui;max-width:1200px;margin:24px auto;color:#222}
table{border-collapse:collapse;margin:10px 0 28px}td,th{border:1px solid #ddd;padding:5px 10px;text-align:left}
th{background:#f3f3f3}.up{background:#e6f6e6}.down{background:#fbe6e6}small{color:#666}
.sum th{background:#e8eef7}.charts{display:grid;grid-template-columns:1fr 1fr;gap:12px 24px;margin:12px 0 20px}
.chart small{display:block;max-width:560px}"""


def load(key, m):
    try:
        return json.load(open(f"runs/{key}_{m}.json"))["rows"]
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def stats(d):
    n = len(d)
    b = sum(r["baseline"]["correct"] for r in d)
    mc = sum(r["metacog"]["correct"] for r in d)
    cov = sum(any(c["correct"] for c in r["metacog"]["candidates"]) for r in d)
    fixed = sum(r["metacog"]["correct"] and not r["baseline"]["correct"] for r in d)
    lost = sum(r["baseline"]["correct"] and not r["metacog"]["correct"] for r in d)
    sb = [r["baseline"]["seconds"] for r in d if r["baseline"]["seconds"] is not None]
    sm = [r["metacog"]["seconds"] for r in d if r["metacog"]["seconds"] is not None]
    gens = [sum(c["source"] != "split" for c in r["metacog"]["candidates"]) for r in d]
    tok = sum(r["metacog"]["thinker_tokens"] or 0 for r in d)
    jc = sum(r["metacog"]["judge_calls"] for r in d)
    miss_judge = sum(not r["metacog"]["correct"] and any(c["correct"] for c in r["metacog"]["candidates"]) for r in d)
    casc = [r for r, g in zip(d, gens) if g == 1]
    return dict(n=n, b=b, mc=mc, cov=cov, fixed=fixed, lost=lost, sb=sum(sb), nsb=len(sb),
                sm=sum(sm), nsm=len(sm), gens=sum(gens), tok=tok, jc=jc,
                miss_judge=miss_judge, miss_cov=n - cov, casc=len(casc),
                casc_ok=sum(r["metacog"]["correct"] for r in casc))


def add(a, s):
    for k, v in s.items():
        a[k] = a.get(k, 0) + v
    return a


def fmt(s, total=None):
    n = s["n"]
    if not n:
        return "<td>0</td>" + "<td>—</td>" * 8
    tb = s["sb"] / s["nsb"] if s["nsb"] else 0
    tm = s["sm"] / s["nsm"] if s["nsm"] else 0
    prog = f"{n}/{total}" if total else str(n)
    speed = f"<td>{tb:.0f}s → {tm:.0f}s ({tm / tb:.1f}×)</td>" if tb else f"<td>— → {tm:.0f}s</td>"
    return (f"<td>{prog}</td><td>{s['b']}/{n} ({s['b'] / n:.0%})</td><td><b>{s['mc']}/{n} ({s['mc'] / n:.0%})</b></td>"
            f"<td>{s['cov']}/{n}</td><td>+{s['fixed']} / −{s['lost']}</td>{speed}"
            f"<td>{s['gens'] / n:.1f}</td><td>{s['tok'] / n / 1000:.1f}k</td><td>${s['jc'] * JEV_USD / n:.4f}</td>")


HEAD = ("<tr><th>thinker</th><th>problems</th><th>baseline</th><th>MetaCog</th><th>ceiling<br>(any path)</th>"
        "<th>fixed / lost</th><th>avg time/problem<br>baseline → MetaCog</th><th>thinker gens<br>/problem</th>"
        "<th>thinker tokens<br>/problem</th><th>Jev cost<br>/problem</th></tr>")


def bars(title, note, groups, series, colors, ymax, fmt_v, w=560, h=230):
    """Grouped bar chart as inline SVG. groups: labels; series: {name: [v per group]}."""
    left, top, bot = 40, 30, 50
    gw = (w - left - 10) / len(groups)
    bw = gw / (len(series) + 1)
    svg = [f"<svg width={w} height={h} font-family=system-ui font-size=11>",
           f"<text x={left} y=14 font-weight=bold font-size=13>{title}</text>"]
    for f in (0, 0.25, 0.5, 0.75, 1):
        y = top + (h - top - bot) * (1 - f)
        svg.append(f"<line x1={left} x2={w - 10} y1={y:.0f} y2={y:.0f} stroke=#eee />"
                   f"<text x={left - 4} y={y + 4:.0f} text-anchor=end fill=#888>{fmt_v(f * ymax)}</text>")
    for gi, g in enumerate(groups):
        for si, (name, vals) in enumerate(series.items()):
            v = vals[gi]
            bh = (h - top - bot) * min(v, ymax) / ymax
            x = left + gi * gw + (si + 0.5) * bw
            y = h - bot - bh
            svg.append(f"<rect x={x:.0f} y={y:.0f} width={bw - 2:.0f} height={bh:.0f} fill={colors[si]} />"
                       f"<text x={x + bw / 2 - 1:.0f} y={y - 3:.0f} text-anchor=middle>{fmt_v(v)}</text>")
        svg.append(f"<text x={left + (gi + 0.5) * gw:.0f} y={h - bot + 14} text-anchor=middle>{g}</text>")
    lx = left
    for si, name in enumerate(series):
        svg.append(f"<rect x={lx} y={h - 14} width=10 height=10 fill={colors[si]} /><text x={lx + 14} y={h - 5}>{name}</text>")
        lx += 14 + 7 * len(name) + 16
    svg.append("</svg>")
    return f"<div class=chart>{''.join(svg)}<small>{note}</small></div>"


def charts(per_bench):
    groups = [t.split(" (")[0].replace("Custom harder counting set", "harder") for t, _ in per_bench]
    S = [s for _, s in per_bench]
    pct = lambda v: f"{v:.0%}"  # noqa: E731
    acc = bars("1. Accuracy: baseline → MetaCog → ceiling",
               "Ceiling = some path was right. Gap MetaCog→ceiling is what a better judge could recover; gap ceiling→100% needs more/better paths.",
               groups, {"baseline": [s["b"] / s["n"] for s in S], "MetaCog": [s["mc"] / s["n"] for s in S],
                        "ceiling": [s["cov"] / s["n"] for s in S]},
               ["#bbb", "#3a7", "#69c"], 1, pct)
    miss = bars("2. Why MetaCog still misses (share of problems)",
                "'no path right' → generate more or more diverse paths. 'Jev picked wrong' → improve the judge. Judge errors are the minority everywhere.",
                groups, {"no path right": [s["miss_cov"] / s["n"] for s in S],
                         "Jev picked wrong": [s["miss_judge"] / s["n"] for s in S]},
                ["#e88", "#fc6"], max(0.05, max((s["miss_cov"] / s["n"] for s in S)) * 1.2), pct)
    cost = bars("3. Cost vs one greedy answer (× multiplier)",
                "Generations ≈ token cost. Time is higher than generations because greedy → samples → judge run in sequence; parallelising stages is the speed lever.",
                groups, {"thinker gens": [s["gens"] / s["n"] for s in S],
                         "wall time": [(s["sm"] / s["nsm"]) / (s["sb"] / s["nsb"]) if s["nsb"] and s["nsm"] else 0 for s in S]},
                ["#69c", "#c96"], 6, lambda v: f"{v:.1f}×")
    casc = bars("4. Cascade: problems stopped after the greedy path",
                "Share of problems where Jev was ≥0.95 on the greedy answer so no samples were made, and how often that answer was right. Raising this share (without losing accuracy) is the cost lever.",
                groups, {"stopped early": [s["casc"] / s["n"] for s in S],
                         "…and correct": [s["casc_ok"] / s["casc"] if s["casc"] else 0 for s in S]},
                ["#9c6", "#3a7"], 1, pct)
    return f"<div class=charts>{acc}{miss}{cost}{casc}</div>"


# key -> (group, short name, plain-English description)
EXPERIMENTS = {
    "prior": ("B", "Answer prior", "Current MetaCog, plus: Jev also rates each distinct final answer on its own (no reasoning shown) and that is added to the score before picking."),
    "adaptP": ("B", "Answer prior + adaptive paths", "Answer-prior pick on top of adaptive branching (2–6 full answers when Jev is unsure)."),
    "adaptN": ("A", "Adaptive paths", "Answer once; if Jev is ≥95% sure, stop. Otherwise write 2–6 more full answers (more when Jev is less sure) and let Jev pick."),
    "sketch": ("A", "Adaptive sketches", "Same, but the extra attempts are 800-token outlines; Jev keeps the best 1–3 and only those are written out in full."),
    "sketchC": ("A", "Cheaper sketches", "500-token outlines, at most 2 written out in full."),
    "adaptW": ("A", "Wider & earlier", "Stop at 90% sure instead of 95%; branch up to 8 full answers when unsure."),
    "deep": ("A", "Deep tree", "4–8 outlines of 500 tokens; Jev prunes; best 1–2 written in full; if Jev is still <95% sure, branch again from the best answer — up to 3 levels."),
    "divP": ("A", "Diverse thought paths", "Answer prior + adaptive paths, but each extra thought path is prompted with a different approach (eliminate options / verify a second way / work backwards…) so the paths don't all make the same mistake."),
    "escP": ("A", "Escalate to a bigger model", "Answer prior + adaptive paths; if Jev is still <95% sure after branching, one more thought path is written by a stronger model (DeepSeek → Kimi, Kimi → DeepSeek) and joins the pool."),
    "triage": ("C", "Triage (full stack)", "Jev first rates how hard the problem is from the statement alone; hard problems start 2–6 answers at the same time as the first one (no waiting), easy ones stay single-answer. Uses adaptive + answer prior."),
}
GROUPS = {
    "A": ("1 · How many answers to write, and when", "Changes the generation side: spend more attempts only where Jev is unsure. Goal: accuracy at equal or lower cost."),
    "B": ("2 · How to pick the winner", "Changes the judging side only: same attempts, better selection. Goal: accuracy at ~zero extra cost."),
    "C": ("3 · How to be faster", "Changes scheduling: decide effort up front and run attempts in parallel. Goal: wall-clock time."),
}
EXP_MODELS = ["deepseek_deepseek-v4-flash", "moonshotai_Kimi-K2.5", "z-ai_glm-5.3-flash"]


def pval(f, l):
    from math import comb
    n = f + l
    if not n:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(f, l) + 1)) / 2**n)


def experiments():
    rows = {g: [] for g in GROUPS}
    for key, (group, name, desc) in EXPERIMENTS.items():
        exp, ref = [], []
        for bench in ("gpqa_diamond", "aime"):
            for m in EXP_MODELS:
                d = load(f"{key}_{bench}", m)
                if not d:
                    continue
                base = {r["problem"]: r for r in load(bench, m)}
                for r in d:
                    if r["problem"] in base:
                        exp.append(r)
                        ref.append(base[r["problem"]])
        n = len(exp)
        if not n:
            continue
        acc_e = sum(r["metacog"]["correct"] for r in exp)
        acc_r = sum(r["metacog"]["correct"] for r in ref)
        acc_b = sum(r["baseline"]["correct"] for r in exp)
        fixed = sum(e["metacog"]["correct"] and not r["metacog"]["correct"] for e, r in zip(exp, ref))
        lost = sum(r["metacog"]["correct"] and not e["metacog"]["correct"] for e, r in zip(exp, ref))
        fb = sum(e["metacog"]["correct"] and not e["baseline"]["correct"] for e in exp)
        lb = sum(e["baseline"]["correct"] and not e["metacog"]["correct"] for e in exp)
        sec_e = sum(r["metacog"]["seconds"] or 0 for r in exp) / n
        sec_r = sum(r["metacog"]["seconds"] or 0 for r in ref) / n
        tok_e = sum(r["metacog"]["thinker_tokens"] or 0 for r in exp) / n
        tok_r = sum(r["metacog"]["thinker_tokens"] or 0 for r in ref) / n
        p = pval(fixed, lost)
        verdict = ("<b style=color:#1a7f1a>better</b>" if p < 0.05 and fixed > lost else
                   "<b style=color:#b00>worse</b>" if p < 0.05 else
                   f"leaning {'better' if fixed > lost else 'worse' if lost > fixed else 'equal'}, not yet significant")
        cls = "up" if fixed > lost else "down" if lost > fixed else ""
        rows[group].append((p if fixed > lost else 2 - p,
            f"<tr class='{cls}'><td><b>{name}</b><br><small>{desc}</small></td><td>{n}</td>"
            f"<td>{100 * acc_b / n:.1f}%</td><td>{100 * acc_r / n:.1f}%</td><td><b>{100 * acc_e / n:.1f}%</b></td>"
            f"<td>+{fixed} / −{lost}<br><small>vs model alone +{fb} / −{lb}</small></td>"
            f"<td>{sec_e / sec_r:.2f}×<br><small>{sec_e:.0f}s vs {sec_r:.0f}s</small></td>"
            f"<td>{tok_e / tok_r:.2f}×<br><small>{tok_e / 1000:.1f}k vs {tok_r / 1000:.1f}k</small></td>"
            f"<td>{verdict}<br><small>p={p:.2f}</small></td></tr>"))
    if not any(rows.values()):
        return ""
    body = ""
    for g, (title, blurb) in GROUPS.items():
        if rows[g]:
            body += f"<tr class=grp><td colspan=9><b>{title}</b> — <small>{blurb}</small></td></tr>"
            body += "".join(r for _, r in sorted(rows[g]))
    return ("<h2>Live experiments — new configurations vs current MetaCog</h2>"
            "<small>Same thinker, same problems (GPQA Diamond first 60 + AIME 60; DeepSeek V4 Flash, Kimi K2.5, GLM-5.3 Flash). "
            "‘Current MetaCog’ = the best-of-N rows in the tables below. Speed/tokens are the new config relative to current MetaCog "
            "(below 1× = faster/cheaper). Refreshes every minute.</small>"
            "<p><small><b>How to read a row:</b> <i>problems</i> = paired problems finished so far · <i>model alone</i> = one plain answer · "
            "<i>current MetaCog</i> = the shipped version (3 answers, Jev picks) on the same problems · <i>new config</i> = the experiment · "
            "<i>fixed / lost</i> = problems the new config got right that current MetaCog got wrong / the reverse · "
            "<i>verdict</i> = ‘better’ only when fixed-vs-lost is significant (p&lt;0.05); ‘leaning’ = trend, not proven. "
            "Within each group rows are sorted best-first.</small></p>"
            "<table><tr><th>configuration</th><th>problems</th><th>model alone</th><th>current MetaCog</th><th>new config</th>"
            f"<th>fixed / lost vs current</th><th>time</th><th>tokens</th><th>verdict</th></tr>{body}</table>")


def build():
    out = [f"<!doctype html><meta charset=utf-8><meta http-equiv=refresh content=60><title>MetaCog benchmark results</title><style>{CSS}</style>",
           "<h1>MetaCog vs baseline — 10 hosted thinkers, Jev judge</h1>"
           "<small>Paired: same model, same problem. Baseline = one greedy answer; MetaCog = greedy anchor + sampled paths "
           "(3, AIME 5), Jev scores each, cascade stops after the greedy path when Jev ≥ 0.95. Raw chat completions, no tools. "
           "Denominators are partial (provider usage caps), so compare baseline vs MetaCog on the same rows, not to published scores. "
           "Thinker gens/problem = generations actually made (1 = cascade stopped early); CommandCode publishes no per-model prices, "
           f"so thinker cost is given as tokens. Updated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC.</small>"]
    summary, grand = [], {}
    sections, per_bench = [], []
    for key, (title, total) in BENCHES.items():
        rows, agg = [], {}
        for m in MODELS:
            s = stats(load(key, m))
            if not s["n"]:
                continue
            add(agg, s)
            cls = "up" if s["mc"] > s["b"] else "down" if s["mc"] < s["b"] else ""
            rows.append(f"<tr class='{cls}'><td>{m.replace('_', '/', 1)}</td>{fmt(s, total)}</tr>")
        add(grand, agg)
        per_bench.append((title, agg))
        summary.append(f"<tr><td>{title}</td>{fmt(agg)}</tr>")
        sections.append(f"<h2>{title}</h2><table>{HEAD}{''.join(rows)}<tr class=sum><th>all</th>{fmt(agg).replace('<td>', '<th>').replace('</td>', '</th>')}</tr></table>")
    out.append(experiments())
    out.append(charts(per_bench))
    out.append(f"<h2>Summary (pooled over thinkers)</h2><table>{HEAD.replace('thinker</th>', 'benchmark</th>')}{''.join(summary)}"
               f"<tr class=sum><th>all benchmarks</th>{fmt(grand).replace('<td>', '<th>').replace('</td>', '</th>')}</tr></table>")
    out.extend(sections)
    open("runs/dashboard.html", "w").write("".join(out))


if __name__ == "__main__":
    while True:
        build()
        time.sleep(30)
