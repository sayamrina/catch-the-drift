"""Exploratory checks that were NOT planned in advance (labelled as such everywhere they are reported)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
rng = np.random.default_rng(0)

a2 = pd.read_csv(RES / "a2_oof.csv")
y, g = a2.y.to_numpy(), a2.pid.to_numpy()


def boot_ci(p, n=1000):
    ids = np.unique(g)
    idx = {k: np.where(g == k)[0] for k in ids}
    out = []
    for _ in range(n):
        s = np.concatenate([idx[k] for k in rng.choice(ids, len(ids))])
        out.append(roc_auc_score(y[s], p[s]))
    return [round(float(np.percentile(out, 2.5)), 3), round(float(np.percentile(out, 97.5)), 3)]


sim = pd.read_csv(RES / "policy_simulation.csv")
w = stats.wilcoxon(sim.triggered, sim.random, alternative="greater")
out = {"gru_auc_ci95": boot_ci(a2.p_gru.to_numpy()),
       "triggered_vs_random_wilcoxon_p": round(float(w.pvalue), 4),
       "participants_triggered_better_than_fixed": int((sim.triggered > sim.fixed).sum()),
       "participants_fixed_better_than_triggered": int((sim.triggered < sim.fixed).sum()),
       "participants": int(len(sim))}
json.dump(out, open(RES / "exploratory.json", "w"), indent=1)
print(out)
