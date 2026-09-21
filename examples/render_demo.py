"""Render runs/*.json traces from live_demo.py into one self-contained HTML report."""

import glob
import html
import json
import re
import sys


def extract(text):
    m = re.findall(r"Answer:\s*\$?\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        return m[-1]
    tail = text.split("</think>")[-1]
    nums = re.findall(r"-?\d+(?:\.\d+)?", tail.replace(",", ""))
    return nums[-1] if nums else None


def regrade(block, truth):
    a = extract(block["text"])
    block["answer"] = a
    block["correct"] = a is not None and abs(float(a) - float(truth)) < 1e-6


def load(path):
    d = json.load(open(path))
    if isinstance(d, list):
        d = {"thinker": "minicpm5-2b-casual", "url": "http://100.119.198.25:8031", "rows": d}
    for r in d["rows"]:
        regrade(r["baseline"], r["truth"])
        for c in r["metacog"]["candidates"]:
            regrade(c, r["truth"])
        pick = r["metacog"]["candidates"][r["metacog"]["pick"]]
        r["metacog"]["answer"], r["metacog"]["correct"] = pick["answer"], pick["correct"]
    return d


def esc(s):
    return html.escape(s or "")


def badge(ok):
    return '<span class="ok">correct</span>' if ok else '<span class="bad">wrong</span>'


def stats(rows):
    nb = sum(r["baseline"]["correct"] for r in rows)
    nm = sum(r["metacog"]["correct"] for r in rows)
    cov = sum(any(c["correct"] for c in r["metacog"]["candidates"]) for r in rows)
    return nb, nm, cov, len(rows)


CSS = """
body{font:15px/1.45 system-ui,sans-serif;max-width:1150px;margin:30px auto;padding:0 16px;color:#1b1b1b}
h1{margin-bottom:4px} .sub{color:#666;margin-top:0}
table{border-collapse:collapse;margin:14px 0} td,th{border:1px solid #ddd;padding:6px 12px;text-align:left}
th{background:#f3f3f3} tr.hi td{background:#eef9ee}
details.model{border:1px solid #ccc;border-radius:10px;margin:18px 0;padding:6px 16px}
details.model>summary{font-size:18px;font-weight:600;cursor:pointer;padding:8px 0}
.prob{border:1px solid #ddd;border-radius:10px;margin:16px 0;padding:14px}
.q{font-weight:600;font-size:16px} .truth{color:#666}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:10px}
.cand{border:1px solid #e3e3e3;border-radius:8px;padding:10px;margin:8px 0;background:#fafafa}
.cand.pick{border:2px solid #2b6cb0;background:#eef4fc}
.hdr{display:flex;justify-content:space-between;font-size:13px;color:#444;margin-bottom:6px;gap:8px}
pre{white-space:pre-wrap;font:12.5px/1.4 ui-monospace,monospace;margin:0;max-height:240px;overflow:auto}
.ok{color:#1a7f37;font-weight:600} .bad{color:#c0392b;font-weight:600}
.score{font-weight:700} .tag{background:#e8e8e8;border-radius:4px;padding:1px 6px;margin-left:6px;font-size:13px;font-weight:400}
.pill{display:inline-block;padding:2px 8px;border-radius:12px;font-size:12px;background:#2b6cb0;color:#fff}
.flag{color:#b7791f;font-weight:600}
"""


def render_run(d):
    rows = d["rows"]
    nb, nm, cov, n = stats(rows)
    out = [
        f'<details class="model"><summary>{esc(d["thinker"])} &nbsp; <span class="tag">baseline {nb}/{n}</span> <span class="tag">MetaCog {nm}/{n}</span> <span class="tag">ceiling {cov}/{n}</span></summary>'
        f'<p class="sub">thinker endpoint: <code>{esc(d["url"])}</code> · judge: Jev · {n} problems</p>'
    ]
    for i, r in enumerate(rows, 1):
        b, m = r["baseline"], r["metacog"]
        bsec = f"{b['seconds']}s" if b["seconds"] is not None else "—"
        msec = f"{m['seconds']}s" if m["seconds"] is not None else "—"
        interesting = (
            b["correct"] != m["correct"] or len({c["answer"] for c in m["candidates"]}) > 1
        )
        flag = ' <span class="flag">candidates disagree</span>' if interesting else ""
        out.append(
            f'<div class="prob"><div class="q">{i}. {esc(r["problem"])}</div><div class="truth">ground truth: <b>{esc(r["truth"])}</b>{flag}</div><div class="cols"><div>'
            f'<h3>Baseline &nbsp;{badge(b["correct"])} <span class="tag">answer {esc(b["answer"])}</span> <span class="tag">{bsec}</span></h3>'
            f'<div class="cand"><pre>{esc(b["text"])}</pre></div></div><div>'
            f'<h3>MetaCog &nbsp;{badge(m["correct"])} <span class="tag">answer {esc(m["answer"])}</span> <span class="tag">{msec} · {m["judge_calls"]} Jev calls</span></h3>'
        )
        for k, c in enumerate(m["candidates"]):
            sc = "–" if c["score"] is None else f"{c['score']:.2f}"
            pick = " pick" if k == m["pick"] else ""
            pill = ' <span class="pill">MetaCog picked this</span>' if pick else ""
            out.append(
                f'<div class="cand{pick}"><div class="hdr"><span>candidate {k + 1} · answer {esc(c["answer"])} · {badge(c["correct"])}{pill}</span><span>Jev P(correct) <span class="score">{sc}</span></span></div><pre>{esc(c["text"])}</pre></div>'
            )
        out.append("</div></div></div>")
    out.append("</details>")
    return "".join(out)


def main(paths, out_path, title):
    runs = [load(p) for p in paths]
    runs.sort(key=lambda d: d["thinker"])
    table = [
        "<table><tr><th>thinker</th><th>baseline (1 greedy answer)</th><th>MetaCog (Jev picks of 3)</th><th>ceiling (any candidate)</th><th>problems where candidates disagreed</th></tr>"
    ]
    tb = tm = tc = tn = 0
    for d in runs:
        nb, nm, cov, n = stats(d["rows"])
        dis = sum(len({c["answer"] for c in r["metacog"]["candidates"]}) > 1 for r in d["rows"])
        tb, tm, tc, tn = tb + nb, tm + nm, tc + cov, tn + n
        cls = ' class="hi"' if nm > nb else ""
        table.append(
            f"<tr{cls}><td>{esc(d['thinker'])}</td><td>{nb}/{n}</td><td>{nm}/{n}</td><td>{cov}/{n}</td><td>{dis}</td></tr>"
        )
    table.append(
        f"<tr><th>total</th><th>{tb}/{tn}</th><th>{tm}/{tn}</th><th>{tc}/{tn}</th><th></th></tr></table>"
    )
    doc = (
        f'<!doctype html><html><head><meta charset="utf-8"><title>{esc(title)}</title><style>{CSS}</style></head><body>'
        f"<h1>{esc(title)}</h1>"
        '<p class="sub">Each thinker is wrapped by <code>metacog.MetaCog(thinker, SystemOneJudge.jev(), Config(mode="best_of_n", n_paths=3, temperature=0.9))</code>. '
        "The thinker samples 3 candidate solutions; Jev (a separate System-One judge) scores each one in isolation with P(correct); MetaCog commits to the highest. "
        "Baseline = the same thinker's single greedy answer. Every text block is verbatim model output. Final answers are auto-extracted (the <code>Answer:</code> line, else the last number).</p>"
        + "".join(table)
        + "<p>Expand a thinker to see every problem, every candidate, and Jev's score for each. Rows highlighted green are thinkers where MetaCog beat the baseline. "
        "This is a demonstration that the loop works end to end on live models — a small sample, not a benchmark.</p>"
        + "".join(render_run(d) for d in runs)
        + "<p class='sub'>Generated by <code>examples/live_demo.py</code> + <code>examples/render_demo.py</code> in the MetaCog repo.</p></body></html>"
    )
    open(out_path, "w").write(doc)
    print(out_path, tb, tm, tc, tn)


if __name__ == "__main__":
    pattern, out_path, title = sys.argv[1], sys.argv[2], sys.argv[3]
    main(sorted(glob.glob(pattern)), out_path, title)
