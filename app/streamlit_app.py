"""Catch the Drift: replay a real learner's eye movements during a video lecture and see when a tutor would step in.

Gaze data and mind-wandering reports: Wandering Eyes (Zhang, Miller, Sun & Cortina, 2020), OSF ptj75, CC-BY 4.0.
Detector scores are out-of-fold (the learner was never in the training data). Catch-up questions were generated once
per lecture paragraph by gpt-4o-mini (src/catchup.py); the demo makes no model calls.
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

DATA = Path(__file__).parent / "data"
DRIFT, PROMPT, REPORT, INK = (217, 70, 46), (37, 99, 235), (23, 34, 43), (23, 34, 43)
FONT = ImageFont.load_default(size=13)

st.set_page_config(page_title="Catch the Drift", page_icon="👀", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=Public+Sans:wght@400;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, label, button, input { font-family: 'Public Sans', sans-serif !important; }
[data-testid="stMainBlockContainer"] { padding-top: 3.2rem; max-width: 1200px; }
h1 { font-family: 'Source Serif 4', serif !important; font-weight: 600; color: #1d2733; font-size: 2.2rem !important; margin-bottom: 0 !important; }
.lede { color: #4a5563; font-size: 1.05rem; max-width: 66ch; margin: .3rem 0 1.2rem; }
.card { background: #fff; border: 1px solid #dde3ea; border-radius: 8px; padding: .8rem 1rem; margin: .4rem 0; }
.card b { color: #1d2733; }
.q { border-left: 4px solid #2563EB; }
.miss { border-left: 4px solid #D9462E; }
.small { color: #4a5563; font-size: .9rem; }
.project-popup { display: flex; align-items: center; gap: 12px; width: fit-content; max-width: 100%; margin: 0 auto 18px;
  padding: 8px 10px 8px 14px; background: #fff; color: #1d2733; font-size: 14px; border: 1px solid #dde3ea;
  border-left: 4px solid #D9462E; border-radius: 10px; }
.project-popup a { padding: 6px 12px; background: #1d2733; color: #fff !important; font-weight: 700;
  text-decoration: none !important; border-radius: 7px; }
@media (max-width: 640px) { .project-popup { flex-wrap: wrap; } }
</style>
<aside class="project-popup" aria-label="Portfolio project">
  <span>This app was developed by Amrina as part of a portfolio.</span>
  <a href="https://sayamrina.github.io/" target="_blank" rel="noopener">View the portfolio</a>
</aside>
""", unsafe_allow_html=True)


@st.cache_data
def load():
    scores = pd.read_csv(DATA / "scores.csv")
    reports = pd.read_csv(DATA / "reports.csv")
    fix = pd.read_csv(DATA / "fixations.csv")
    meta = json.load(open(DATA / "meta.json"))
    qs = json.load(open(DATA / "catchup_questions.json"))["items"]
    return scores, reports, fix, meta, qs


scores, reports, fix, meta, questions = load()
thr = meta["threshold"]


def paragraph_at(t_s):
    for q in questions:
        if q["start_s"] <= t_s < q["end_s"]:
            return q
    return questions[-1]


def policy(g):
    trig, last = [], -1e12
    for e, p in zip(g.end, g.p):
        if p >= thr and e - last >= 60_000:
            trig.append(e)
            last = e
    k = len(trig)
    ends = g.end.to_numpy()
    fixed = [] if k == 0 else [int(ends[int(round(i))]) for i in pd.Series(range(1, k + 1)) * (len(ends) - 1) / (k + 1)]
    return trig, fixed


def timeline(g, rep, trig, fixed, width=1100, height=220):
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    left, right, top, bottom = 40, width - 10, 18, height - 70
    T = max(g.end.max(), 1)
    sx = lambda t: left + (right - left) * t / T  # noqa: E731
    sy = lambda p: bottom - (bottom - top) * p  # noqa: E731
    d.line([(left, sy(thr)), (right, sy(thr))], fill=(200, 200, 200), width=1)
    pts = [(sx(e), sy(p)) for e, p in zip(g.end, g.p)]
    d.line(pts, fill=DRIFT, width=3)
    for t in rep:
        d.line([(sx(t), top), (sx(t), bottom)], fill=REPORT, width=2)
    box = d.textbbox((right - 4, sy(thr) - 4), "prompt threshold", font=FONT, anchor="rb")
    d.rectangle([box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2], fill="white")
    d.text((right - 4, sy(thr) - 4), "prompt threshold", fill=(120, 120, 120), font=FONT, anchor="rb")
    for t in trig:
        d.polygon([(sx(t) - 6, bottom + 16), (sx(t) + 6, bottom + 16), (sx(t), bottom + 5)], fill=PROMPT)
    for t in fixed:
        d.ellipse([sx(t) - 5, bottom + 24, sx(t) + 5, bottom + 34], outline=(110, 110, 110), width=2)
    for m in range(0, int(T / 60000) + 1, 2):
        d.text((sx(m * 60000), height - 6), f"{m} min", fill=INK, font=FONT, anchor="mb")
    return img


def gaze(f, end):
    img = Image.open(DATA / "education_snapshot.jpg").convert("RGB")
    W, H = img.size
    sxf, syf = W / 1280, H / 1024
    w = f[(f.t >= end - 30_000) & (f.t < end)]
    d = ImageDraw.Draw(img, "RGBA")
    prev = None
    for x, y, dur in zip(w.x, w.y, w.dur):
        r = 4 + dur / 60
        c = (x * sxf, y * syf)
        if prev:
            d.line([prev, c], fill=(29, 39, 51, 140), width=2)
        d.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], fill=(217, 70, 46, 120), outline=(217, 70, 46, 230))
        prev = c
    return img


st.title("Catch the Drift")
st.markdown('<p class="lede">A student watches a 19-minute video lecture and presses a key whenever they notice their '
            'mind has wandered. A detector that has never seen this student reads only their eye movements. '
            'Where would a tutor step in, and what would it ask?</p>', unsafe_allow_html=True)

pids = sorted(scores.pid.unique(), key=lambda s: int(s.split("_s")[-1]))
label = {p: f"Student {p.split('_s')[-1]}" for p in pids}
pid = st.selectbox("Choose a real student from the study", pids, format_func=label.get)
g = scores[scores.pid == pid].sort_values("end")
rep = reports[reports.pid == pid].t.tolist()
trig, fixed = policy(g)
hit = lambda t: any(0 <= r - t < 30_000 for r in rep)  # noqa: E731

st.image(timeline(g, rep, trig, fixed), use_container_width=True)
st.markdown(
    f'<p class="small"><span style="color:#D9462E">■</span> detector score for the last 30 s &nbsp; '
    f'<span style="color:#1d2733">|</span> the student pressed the key ({len(rep)}×) &nbsp; '
    f'<span style="color:#2563EB">▲</span> prompt triggered by the detector ({len(trig)}×, '
    f'{sum(map(hit, trig))} within 30 s before a report) &nbsp; '
    f'○ same number of prompts on a fixed schedule ({sum(map(hit, fixed))} within 30 s before a report)</p>',
    unsafe_allow_html=True)

if trig:
    pick = st.select_slider("Step through the detector's prompts", options=trig,
                            format_func=lambda t: f"{int(t / 60000)}:{int(t / 1000) % 60:02d}")
    c1, c2 = st.columns([1.15, 1])
    with c1:
        st.image(gaze(fix[fix.pid == pid], pick), use_container_width=True,
                 caption="Fixations in the 30 s before the prompt (larger dot = longer fixation)")
    with c2:
        q = paragraph_at(pick / 1000)
        miss = "The student reported mind wandering within the next 30 s." if hit(pick) else \
            "No report followed within 30 s. This prompt may have interrupted an attentive student."
        st.markdown(f'<div class="card miss"><b>What happened</b><br>{miss}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="card q"><b>Catch-up question</b><br>{q["question"]}</div>', unsafe_allow_html=True)
        with st.expander("Answer key and the passage it comes from"):
            st.write(q["answer"])
            st.caption(f"Lecture paragraph {q['index']} (approx. {int(q['start_s'] // 60)}:{int(q['start_s'] % 60):02d}"
                       f"–{int(q['end_s'] // 60)}:{int(q['end_s'] % 60):02d}): {q['text']}")
else:
    st.info("The detector never crossed the threshold for this student, so the tutor stays silent.")

st.markdown("---")
r = meta["results"]
st.markdown(f"""
**How good is the detector?** Not good. Across all {r['participants']} students in the self-report group, the 30 s
before a report could not be told apart from other moments: AUC {r['auc']:.2f}, where 0.50 is chance, tested on students
the model had never seen. Detector-triggered prompts landed within 30 s before a real report in {r['hit_triggered']:.0%}
of cases, against {r['hit_fixed']:.0%} for a fixed schedule (preregistered test, p = {r['p_vs_fixed']:.3f}) but also
{r['hit_random']:.0%} for prompts at random times (exploratory, p = {r['p_vs_random']:.2f}). So the tutor timing in this
demo is not better than chance; the demo shows how such a system would behave, not that it works.

**Limits.** Self-reports only capture the moments students noticed; unreported mind wandering counts as "attentive" here.
The lecture paragraph is matched to time by word count, which is approximate. Full methods, the preregistration and
all results: [project page](https://sayamrina.github.io/catch-the-drift/).
""")
