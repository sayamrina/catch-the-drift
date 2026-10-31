# Study plan: Catch the Drift

Written 31 October 2026, before any model was fitted. At this point I had only counted participants, labels and recording lengths, checked that the probe times in the processed file match the paper, and read the authors' AOI definitions. Any later change is logged under "Deviations" at the end, with date and reason.

## Question

Can a learner's mind wandering during a video lecture be detected from eye movements alone, early enough and reliably enough that an adaptive system should act on it? And if it acts, is a detector-triggered prompt better timed than a prompt on a fixed schedule?

## Data

**Part A: Wandering Eyes** (Zhang, Miller, Sun & Cortina, 2020, *Applied Cognitive Psychology*; OSF ptj75, CC-BY 4.0). Undergraduates watched a 19-minute (Education) or 22-minute (Genetics) video lecture with slides and instructor; Eye Tribe tracker, 60 Hz, fixations provided. Two report conditions:
- *Probe-caught:* four beeps at fixed times; participant answers Y (mind wandering) or N. 68 participants, 272 probes, 87 Y.
- *Self-caught:* participant presses a key whenever they notice mind wandering. 75 participants, 815 key presses.
- Pre-test (5 items) and post-test (18 items) per participant.

**Part B: ReMind** (OSF ajme7, CC-BY 4.0): eye tracking during reading, with readers marking where mind wandering started and ended. The Part B plan will be added as a dated amendment after I have read the data documentation and before I look at labels in relation to gaze.

## Preprocessing (fixed)

- Fixations shorter than 80 ms or longer than 2000 ms are dropped (the authors' rule).
- AOIs exactly as in the authors' notebook: Slides, Teacher, white space inside the video, outside/missing.
- Self-caught: if two key presses fall within 5 s, the first is dropped (the authors' rule).
- A window is excluded if valid fixations cover less than 50% of its duration. A participant is excluded from an analysis if fewer than 2 windows remain.

## Windows and labels

- Primary window: the **30 s before** the probe or key press.
- Part A1 (probe-caught): one window per probe, label = answer.
- Part A2 (self-caught): positives = windows ending at each key press. Negatives = windows on a 30 s grid over the whole video that end at least 30 s before or 15 s after any key press. Negatives are "not reported", not "on task"; this is a known limitation of self-caught labels.

## Features (fixed list, per window)

Number of fixations per second; mean, median and SD of fixation duration; proportion of fixations ≥ 500 ms; proportion of fixations on Teacher among Slides + Teacher; proportion outside/missing; dispersion of fixations on Slides (RMS distance to centroid, in px); mean and SD of distance between successive fixations; proportion of mostly horizontal successive moves; share of window time covered by fixations.

## Models and evaluation

- Primary model: L2 logistic regression on standardised features. Secondary: random forest (500 trees, balanced class weights).
- Sequence model (A2 only): one-layer GRU (hidden 32) on the fixation sequence of each window (duration, x, y, AOI, distance to previous fixation), class-weighted loss, fixed 30 epochs.
- **All evaluation is person-independent**: leave-one-participant-out in A1; 10-fold grouped by participant in A2. No participant ever appears in training and test at once.
- Metrics: AUC-ROC (primary) and AUC-PR, reported next to the base rate.
- Significance of AUC: permutation test, 1,000 label shuffles, one-sided, α = .05.
- Seeds fixed at 0.

## Hypotheses

- **H1 (probe-caught detection):** the primary model detects probe-caught mind wandering above chance (AUC-ROC > .5, permutation p < .05).
- **H2 (self-caught detection):** the primary model separates the 30 s before a self-caught report from unreported windows above chance.
- **H3 (how early):** in A2, AUC falls as the window is moved earlier, ending 0, 5, 10, 20 and 30 s before the report (negative Spearman correlation between lag and AUC).
- **H4 (sequence vs. features):** the GRU will **not** beat the feature-based primary model by more than .02 AUC on A2. I expect this because the data are small; a larger gain would count against H4.
- **H5 (transfer across lectures):** a model trained on one lecture (A2) and tested on the other stays above chance (AUC > .5) but below the within-lecture AUC.
- **H6 (link to learning):** a participant's mean out-of-fold predicted probability of mind wandering across all 30 s windows of the lecture correlates negatively with the post-test score, controlling for the pre-test (partial correlation, one-sided, all participants with data).
- **H7 (when to prompt):** in a simulation on A2, a detector-triggered policy (prompt when out-of-fold probability exceeds the participant-independent threshold, then wait at least 60 s) places a larger share of its prompts in the 30 s before a real self-caught report than a fixed schedule with the same number of prompts per participant (paired Wilcoxon, one-sided). A random schedule with the same count is reported as a second baseline. The threshold is the 80th percentile of out-of-fold scores, fixed now.

## Amendment 1: Part B plan (31 October 2026, before fitting any Part B model)

Written after reading the column names, label counts and window durations of the ReMind group files, and after seeing the authors' own AUC file (`comparison_permutation_test_results.csv`, their AUCs of about .62 to .65). I have not related any gaze feature to the labels myself.

- **Data:** `group_features_end2_tp.csv`, the 2 s window that ends at a mind-wandering onset (positive) or at a matched control point (negative); 45 readers, 2,211 windows, right eye. Label = `is_MWreported`.
- **Features:** gaze only, chosen to match Part A as closely as the files allow: `fix_num`, `norm_fix_word_num`, `norm_in_word_reg`, `norm_out_word_reg`, `norm_total_viewing`, `fix_dispersion`, `sacc_num`, `sacc_length`, `horizontal_sacc`, `zipf_fixdur_corr`, `word_length_fixdur_corr`. Pupil and blink features are added only in a secondary model.
- **H8:** the primary model (same L2 logistic regression, 10-fold grouped by reader) detects mind-wandering onset in reading above chance (AUC-ROC > .5, permutation test, 1,000 shuffles).
- **H9 (exploratory, descriptive):** for the features that exist in both parts (fixation count/rate, dispersion, saccade length, horizontal moves), I compare the sign of the standardised coefficient between reading (B) and the video lecture (A2). No test; reported as agree or disagree.
- Missing values are imputed with the median inside each training fold, as in Part A.

## What I will report regardless of outcome

All hypotheses with effect sizes and confidence intervals, including those that fail; numbers of excluded windows and participants; base rates next to every AUC-PR.

## Deviations

1. **A2 negatives (31 October 2026, written before the first model was fitted, logged after).** The code excludes negative windows that *end* less than 30 s before or less than 45 s after a key press, instead of "15 s after". With 30 s windows, "ends 15 s after" would still let a negative contain the press itself; 45 s keeps the whole window at least 15 s clear of it. This is stricter than written, not looser.
2. **H6/H7 scoring (31 October 2026, implementation detail not covered by the plan).** Lecture windows of the 4 self-caught participants who were excluded from A2 (fewer than 2 usable windows) and all probe-caught participants were scored by the model fitted on all of A2; they were never in its training data, so these scores are still person-independent. All other H6/H7 scores are out-of-fold.
3. **Part B feature (31 October 2026, found after the first Part B run).** `horizontal_sacc` has no values in `group_features_end2_tp.csv` (right eye). The imputer silently dropped it, so the first run's coefficient names were shifted by one from that feature on; the AUC and permutation test were unaffected. The feature is now removed explicitly, the analysis was rerun, and the "horizontal moves" pair in H9 cannot be compared.

## Exploratory analyses (not planned in advance)

Added after seeing the Part A results, and labelled exploratory wherever they appear (`src/exploratory.py`): a participant bootstrap CI for the GRU AUC, and a Wilcoxon test of the triggered policy against the random schedule (the random schedule was planned in advance as a reported baseline, but without a test).
