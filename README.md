# Catch the Drift

**Can eye movements tell a tutor when a student's mind wanders during a video lecture? A preregistered test on open data**

[Project page](https://sayamrina.github.io/catch-the-drift/) · [Demo](https://catch-the-drift.streamlit.app/) · [Preregistration](PREREGISTRATION.md) · [Portfolio](https://sayamrina.github.io/)

Students drift away during video lectures, and nobody notices. An adaptive system could pause and ask a short catch-up question, but only if it knows when. This project tests, on open eye-tracking data from 143 students, whether mind wandering can be detected from gaze for a student the system has never seen, how early, and whether a tutor that acts on the detector prompts at better moments than one that asks at fixed times.

The nine hypotheses, the features, models and tests were written in [PREREGISTRATION.md](PREREGISTRATION.md) and committed before any model was fitted. Deviations and exploratory checks are logged in the same file.

## Short answer

- **The detector does not work.** Summary features of 30 s of gaze could not separate the moments before a mind-wandering report from other moments (AUC 0.49 for key presses, 0.50 for probes; chance = 0.50).
- **The tutor beat a fixed schedule, but not chance.** Detector-triggered prompts were better timed than evenly spaced ones (27% vs 21%, p = .026, as preregistered), but not better than prompts at random times (24%, exploratory p = .36).
- **The order of fixations may matter.** A small GRU on the raw fixation sequence reached AUC 0.57 (exploratory 95% CI 0.52–0.63), the one video result clearly above chance.
- **In reading, gaze carries a weak signal.** On the ReMind data, the same kind of model detects the onset of mind wandering above chance (AUC 0.55, p = .001).

## Data

**Part A: Wandering Eyes** (Zhang, Miller, Sun and Cortina, 2020; [OSF ptj75](https://osf.io/ptj75/), CC-BY 4.0). 143 undergraduates watched a 19-minute (Education) or 22-minute (Genetics) lecture video while an Eye Tribe tracker recorded gaze at 60 Hz. The probe-caught group answered four beeps (272 answers, 87 yes); the self-caught group pressed a key when they noticed mind wandering (815 presses). Pre- and post-tests for everyone.

**Part B: ReMind** (Sun, 2025; [OSF ajme7](https://osf.io/ajme7/), CC-BY 4.0). 45 readers marked mind-wandering onsets; the authors' gaze features for the 2 s before each onset and before matched controls (2,211 windows).

The raw data are not included in this repository; `src/build.py` expects them under `data/raw/`.

## Method

- **Windows.** 30 s of gaze before each probe or key press. For key presses, negatives are 30 s windows on a grid that end at least 30 s before and 45 s after any press.
- **Features (12).** Fixation rate; mean, median and SD of fixation duration; share of fixations ≥ 500 ms; share on the instructor vs the slides (AOIs from the authors' notebook); share off the video; dispersion on the slides; mean and SD of saccade amplitude; share of horizontal saccades; tracking coverage.
- **Exclusions.** Windows with less than 50% tracking coverage; participants with fewer than 2 windows. This removed 38% of probe windows (168 of 272 remain, 49 of 68 participants) and 28% of key-press windows (1,698 of 2,349, 71 of 75 participants).
- **Models.** L2 logistic regression (primary), random forest, and a one-layer GRU on fixation sequences.
- **Evaluation.** Person-independent only: leave-one-participant-out (probes) or 10-fold grouped by participant (key presses). Permutation tests with 1,000 label shuffles; participant bootstrap CIs.
- **Tutor simulation.** In 30 s steps, the triggered tutor prompts when the out-of-fold score is above the 80th percentile, with at least 60 s between prompts. Baselines: the same number of evenly spaced prompts, and random prompts. A prompt is well timed if a report follows within 30 s.
- **Catch-up questions.** gpt-4o-mini wrote one question and answer key per paragraph of the Education lecture script, using only that paragraph (`src/catchup.py`, temperature 0). Video time is mapped to paragraphs by word count.

## Results

| Hypothesis | Result | Verdict |
|---|---|---|
| **H1** Gaze detects probe-caught mind wandering | AUC 0.50 [0.42, 0.59], p = .38; forest 0.58 | Not supported |
| **H2** Gaze detects the 30 s before a key press | AUC 0.49 [0.42, 0.56], p = .57; forest 0.54 | Not supported |
| **H3** AUC falls as the window moves 0–30 s earlier | 0.49, 0.48, 0.44, 0.45, 0.47; ρ = −.60, p = .29 | Not supported |
| **H4** GRU gains no more than 0.02 AUC | GRU 0.57 vs 0.49 (+0.09) | Not supported (GRU better than expected) |
| **H5** Transfer to the other lecture stays above chance | 0.48 and 0.45 (within: 0.53, 0.59) | Not supported |
| **H6** Predicted mind wandering predicts lower post-test | partial r = +.16, n = 140 | Not supported (wrong sign) |
| **H7** Triggered prompts are better timed than fixed | 27% vs 21%, Wilcoxon p = .026; random 24% | Supported, but not better than random (exploratory p = .36) |
| **H8** Gaze detects mind-wandering onset in reading | AUC 0.55, p = .001; with pupil and blink 0.54 | Supported |
| **H9** Feature directions agree between reading and video | dispersion agrees; fixation count and saccade length disagree; horizontal saccades missing in ReMind | Descriptive |

All numbers: `results/part_a.json`, `results/part_b.json`, `results/exploratory.json`.

## Demo

`app/streamlit_app.py` replays the 33 self-caught students of the Education lecture: detector score over time, the student's reports, where the triggered and fixed tutors would prompt, fixations in the 30 s before a prompt over the lecture slide, and the catch-up question for that moment. It makes no model calls.

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-analysis.txt   # torch: use the CPU wheel index if needed
cd src
../.venv/bin/python build.py        # windows and features -> data/derived/
../.venv/bin/python analysis.py     # H1-H7 -> results/part_a.json
../.venv/bin/python part_b.py       # H8-H9 -> results/part_b.json
../.venv/bin/python exploratory.py  # exploratory checks
../.venv/bin/python export.py       # demo data -> app/data/
```

`catchup.py` needs an `OPENAI_API_KEY`; its output is already in `results/catchup_questions.json`.

## Limits

Self-reports only capture the mind wandering a student noticed, so "negative" windows contain unnoticed drifting; one dataset and one consumer tracker; the tutor simulation measures timing, not learning; the GRU result is one run with a fixed seed; paragraph timing is approximate.

## Credits

Data: Zhang, H., Miller, K. F., Sun, X., and Cortina, K. S. (2020). Wandering eyes: Eye movements during mind wandering in video lectures. *Applied Cognitive Psychology, 34*(2), 449–464 (CC-BY 4.0); Sun, H. (2025). ReMind: A retrospective self-report paradigm for studying mind-wandering onset during reading [Data set], OSF ajme7 (CC-BY 4.0).

Built by Amrina, with AI as a supporting tool for coding. Code: MIT licence.
