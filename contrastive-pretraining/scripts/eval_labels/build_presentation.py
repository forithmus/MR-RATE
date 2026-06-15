"""Generate a self-contained HTML presentation of the merged-group labels.

Reads the produced splits_merged / splits_merged_32col folders and emits
merged_labels_presentation.html (inline CSS, no external dependencies).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).parent
RAW = HERE / "splits_merged_majority"          # primary: 3-way majority
LEGACY = HERE / "splits_merged"                # baseline: 2-model Claude AND GPT
OUT = HERE / "merged_labels_presentation.html"

# No per-column "recovered" badges in the majority story.
RECOVERED = {}

SCHEMES = {
    "Pathophysiologie": ("PP_", "#2563eb"),
    "Bildphänotyp": ("BP_", "#9333ea"),
}


def load(folder: Path):
    cols = next(csv.reader(open(folder / "mrrate_merged_labels.csv")))[1:]
    defs = json.loads((folder / "group_definitions.json").read_text())
    split = {r["study_uid"]: r["split"] for r in csv.DictReader(open(folder / "splits.csv"))}
    pos = {c: {"train": 0, "val": 0, "test": 0, "all": 0} for c in cols}
    n = {"train": 0, "val": 0, "test": 0, "all": 0}
    for row in csv.DictReader(open(folder / "mrrate_merged_labels.csv")):
        s = split[row["study_uid"]]
        n[s] += 1
        n["all"] += 1
        for c in cols:
            if row[c] == "1":
                pos[c][s] += 1
                pos[c]["all"] += 1
    return cols, defs, pos, n


def pct(a, b):
    return 100 * a / b if b else 0


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def main() -> None:
    cols, defs, pos, n = load(RAW)
    _, _, lpos, _ = load(LEGACY)

    label = lambda c: c.replace("PP_", "").replace("BP_", "").replace("_", " ")
    scheme_of = lambda c: "Pathophysiologie" if c.startswith("PP_") else "Bildphänotyp"
    color_of = lambda c: SCHEMES[scheme_of(c)][1]
    maxpct = max(pct(pos[c]["all"], n["all"]) for c in cols)

    # ---- group cards ----
    def group_card(c):
        col = color_of(c)
        members = defs[c]
        chips = "".join(
            f'<span class="chip" style="border-color:{col}33;color:{col}">{esc(m)}</span>'
            for m in members
        )
        rec = RECOVERED.get(c, [])
        rec_html = ""
        if rec:
            rec_html = (
                '<div class="rec">recovered: '
                + ", ".join(esc(m) for m in rec)
                + "</div>"
            )
        p = pct(pos[c]["all"], n["all"])
        return f"""
        <div class="card">
          <div class="card-head" style="border-left:4px solid {col}">
            <div class="card-title">{esc(c)}</div>
            <div class="card-prev"><b>{pos[c]['all']:,}</b> <span>({p:.1f}%)</span></div>
          </div>
          <div class="bar"><div class="bar-fill" style="width:{p/maxpct*100:.1f}%;background:{col}"></div></div>
          <div class="chips">{chips}</div>
          {rec_html}
        </div>"""

    pp_cards = "".join(group_card(c) for c in cols if c.startswith("PP_"))
    bp_cards = "".join(group_card(c) for c in cols if c.startswith("BP_"))

    # ---- prevalence table ----
    rows_html = ""
    for c in cols:
        col = color_of(c)
        diff = pos[c]["all"] - lpos[c]["all"]
        if diff > 0:
            diff_html = f'<span class="up">+{diff:,}</span>'
        elif diff < 0:
            diff_html = f'<span class="down">{diff:,}</span>'
        else:
            diff_html = '<span class="same">—</span>'
        rows_html += f"""
        <tr>
          <td><span class="dot" style="background:{col}"></span>{esc(c)}</td>
          <td class="num">{pos[c]['train']:,}</td>
          <td class="num">{pos[c]['val']:,}</td>
          <td class="num">{pos[c]['test']:,}</td>
          <td class="num"><b>{pos[c]['all']:,}</b></td>
          <td class="num">{pct(pos[c]['all'], n['all']):.1f}%</td>
          <td class="num">{diff_html}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MR-RATE · Merged Pathology Group Labels</title>
<style>
:root {{ --bg:#0f172a; --panel:#ffffff; --ink:#0f172a; --muted:#64748b; --line:#e2e8f0; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  color:var(--ink); background:#f1f5f9; line-height:1.5; }}
.hero {{ background:linear-gradient(135deg,#0f172a,#1e3a8a); color:#fff; padding:56px 32px 40px; }}
.wrap {{ max-width:1080px; margin:0 auto; padding:0 24px; }}
.hero h1 {{ margin:0 0 8px; font-size:30px; letter-spacing:-.02em; }}
.hero p {{ margin:0; color:#cbd5e1; max-width:760px; }}
.stats {{ display:flex; gap:16px; flex-wrap:wrap; margin-top:28px; }}
.stat {{ background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.15);
  border-radius:12px; padding:14px 20px; min-width:120px; }}
.stat b {{ display:block; font-size:24px; }}
.stat span {{ color:#94a3b8; font-size:13px; }}
section {{ padding:40px 0; }}
h2 {{ font-size:20px; margin:0 0 4px; }}
.sub {{ color:var(--muted); margin:0 0 24px; font-size:14px; }}
.flow {{ display:flex; gap:12px; flex-wrap:wrap; align-items:stretch; }}
.flow .step {{ flex:1; min-width:160px; background:var(--panel); border:1px solid var(--line);
  border-radius:12px; padding:16px; }}
.flow .step .k {{ font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:.05em; }}
.flow .step .v {{ font-weight:600; margin-top:4px; }}
.arrow {{ align-self:center; color:#94a3b8; font-size:22px; }}
.scheme-h {{ display:flex; align-items:center; gap:10px; margin:8px 0 16px; }}
.scheme-h .badge {{ width:12px; height:12px; border-radius:3px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(320px,1fr)); gap:14px; }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:0 0 14px;
  overflow:hidden; box-shadow:0 1px 2px rgba(0,0,0,.04); }}
.card-head {{ display:flex; justify-content:space-between; align-items:center; padding:14px 16px 10px; }}
.card-title {{ font-weight:600; font-size:14px; }}
.card-prev b {{ font-size:15px; }}
.card-prev span {{ color:var(--muted); font-size:13px; }}
.bar {{ height:6px; background:#f1f5f9; margin:0 16px 12px; border-radius:4px; overflow:hidden; }}
.bar-fill {{ height:100%; border-radius:4px; }}
.chips {{ display:flex; flex-wrap:wrap; gap:6px; padding:0 16px; }}
.chip {{ font-size:11.5px; padding:3px 8px; border:1px solid var(--line); border-radius:999px;
  background:#fff; white-space:nowrap; }}
.rec {{ margin:10px 16px 0; font-size:12px; color:#059669; background:#ecfdf5;
  border:1px solid #a7f3d0; border-radius:8px; padding:6px 10px; }}
table {{ width:100%; border-collapse:collapse; background:var(--panel); border:1px solid var(--line);
  border-radius:12px; overflow:hidden; font-size:13.5px; }}
th,td {{ padding:10px 14px; text-align:left; border-bottom:1px solid var(--line); }}
th {{ background:#f8fafc; color:var(--muted); font-weight:600; font-size:12px; text-transform:uppercase; letter-spacing:.03em; }}
td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
tr:last-child td {{ border-bottom:none; }}
.dot {{ display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:8px; vertical-align:middle; }}
.up {{ color:#059669; font-weight:600; }}
.down {{ color:#dc2626; font-weight:600; }}
.same {{ color:#cbd5e1; }}
.note {{ background:#fffbeb; border:1px solid #fde68a; border-radius:12px; padding:16px 18px; font-size:13.5px; }}
pre {{ background:#0f172a; color:#e2e8f0; padding:16px 18px; border-radius:12px; overflow:auto;
  font-size:13px; line-height:1.6; }}
.files {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(240px,1fr)); gap:12px; }}
.file {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px 14px; font-size:13px; }}
.file b {{ font-family:ui-monospace,Menlo,monospace; font-size:12.5px; }}
.file span {{ color:var(--muted); display:block; margin-top:3px; }}
footer {{ color:var(--muted); font-size:12.5px; padding:32px 0 60px; text-align:center; }}
.tag {{ display:inline-block; font-size:11px; padding:2px 8px; border-radius:6px; background:#dbeafe; color:#1e40af; font-weight:600; }}
</style></head>
<body>

<div class="hero"><div class="wrap">
  <div class="tag">MR-RATE · linear-probe targets</div>
  <h1>Merged Pathology Group Labels</h1>
  <p>Radiology-report findings collapsed from 37 fine-grained pathologies into the
  neuroradiologist's clinically-grounded groups, derived from a <b>3-model majority vote</b>.</p>
  <div class="stats">
    <div class="stat"><b>{n['all']:,}</b><span>studies labeled</span></div>
    <div class="stat"><b>{len(cols)}</b><span>group targets</span></div>
    <div class="stat"><b>3</b><span>voting models</span></div>
    <div class="stat"><b>{n['train']:,} / {n['val']:,} / {n['test']:,}</b><span>train / val / test</span></div>
  </div>
</div></div>

<div class="wrap">

<section>
  <h2>How the ground-truth labels are derived</h2>
  <p class="sub">Each study is independently labeled by three models; a finding is positive when a <b>majority</b> agree (≥2 of the available votes). Groups are then the logical OR of their member pathologies.</p>
  <div class="flow">
    <div class="step"><div class="k">Anthropic</div><div class="v">Claude Opus 4.7</div></div>
    <div class="step"><div class="k">OpenAI</div><div class="v">GPT-5.5</div></div>
    <div class="step"><div class="k">NVIDIA</div><div class="v">Nemotron-3 Super 120B</div></div>
    <div class="arrow">→</div>
    <div class="step"><div class="k">Majority vote</div><div class="v">37 pathology labels</div></div>
    <div class="arrow">→</div>
    <div class="step"><div class="k">OR per group</div><div class="v">{len(cols)} merged targets</div></div>
  </div>
</section>

<section>
  <h2>Grouping scheme 1 — Pathophysiologie</h2>
  <p class="sub">Eight mutually-exclusive groups by underlying mechanism. Chips list member pathologies.</p>
  <div class="grid">{pp_cards}</div>
</section>

<section>
  <h2>Grouping scheme 2 — Bildphänotyp</h2>
  <p class="sub">Six groups by imaging phenotype. (Three further phenotype groups are identical to their pathophysiology counterparts and are not duplicated.)</p>
  <div class="grid">{bp_cards}</div>
</section>

<section>
  <h2>Prevalence by split</h2>
  <p class="sub">Positive study counts per group. The last column shows the change versus the 2-model (Claude ∧ GPT) agreement labels.</p>
  <table>
    <tr><th>Group</th><th style="text-align:right">train+</th><th style="text-align:right">val+</th>
    <th style="text-align:right">test+</th><th style="text-align:right">total+</th>
    <th style="text-align:right">prev.</th><th style="text-align:right">Δ vs 2-model</th></tr>
    {rows_html}
  </table>
</section>

<section>
  <h2>Label variants available</h2>
  <p class="sub">Same 14 columns and study set throughout — only the voting rule differs.</p>
  <div class="note">
    <b>splits_merged_majority/</b> &nbsp;— this report. 3-way majority of Claude Opus 4.7 + GPT-5.5 + Nemotron-3 Super 120B (≥2 of available votes), over all 37 pathologies.<br><br>
    <b>splits_merged/</b> &nbsp;— 2-model strict agreement (Claude ∧ GPT).<br><br>
    <b>splits_merged_32col/</b> &nbsp;— legacy 2-model agreement built from the 32-column labels (5 pathologies dropped).
  </div>
</section>

<section>
  <h2>Files &amp; usage</h2>
  <div class="files">
    <div class="file"><b>mrrate_merged_labels.csv</b><span>study_uid + {len(cols)} binary columns</span></div>
    <div class="file"><b>train.csv / val.csv / test.csv</b><span>per-split, same columns</span></div>
    <div class="file"><b>splits.csv</b><span>study_uid → split</span></div>
    <div class="file"><b>pathologies.json</b><span>group names &amp; prompts</span></div>
    <div class="file"><b>group_definitions.json</b><span>group → member pathologies</span></div>
  </div>
  <pre>--labels_csv  splits_merged/mrrate_merged_labels.csv
--splits_csv  splits_merged/splits.csv
--split       train        # / val / test</pre>
</section>

<footer>MR-RATE · merged-group labels for linear probing · 3-model majority: Claude Opus 4.7 · GPT-5.5 · Nemotron-3 Super 120B</footer>
</div>
</body></html>"""

    OUT.write_text(html)
    print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
