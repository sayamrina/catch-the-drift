# Preregistration: Catch the Drift

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

## What I will report regardless of outcome

All hypotheses with effect sizes and confidence intervals, including those that fail; numbers of excluded windows and participants; base rates next to every AUC-PR.

## Deviations

(none yet)
