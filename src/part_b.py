"""Part B (Amendment 1): mind-wandering onset in reading (ReMind), H8 and H9."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
F = ROOT / "data" / "raw" / "remind" / "processed_data" / "group_file" / "group_features_end2_tp.csv"
GAZE = ["fix_num", "norm_fix_word_num", "norm_in_word_reg", "norm_out_word_reg", "norm_total_viewing", "fix_dispersion",
        "sacc_num", "sacc_length", "horizontal_sacc", "zipf_fixdur_corr", "word_length_fixdur_corr"]
EXTRA = ["blink_num", "blink_dur", "ibi", "norm_ibi", "page_norm_pupil", "pupil_slope"]
rng = np.random.default_rng(0)
gkf = GroupKFold(n_splits=10)


def model():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000))


def oof(X, y, g):
    p = np.zeros(len(y))
    for tr, te in gkf.split(X, y, g):
        p[te] = model().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
    return p


d = pd.read_csv(F)
d = d[d.eye == "R"].reset_index(drop=True)
EMPTY = [c for c in GAZE + EXTRA if d[c].isna().all()]  # horizontal_sacc has no values in this file (Deviation 3)
GAZE, EXTRA = [c for c in GAZE if c not in EMPTY], [c for c in EXTRA if c not in EMPTY]
y, g = d.is_MWreported.astype(int).to_numpy(), d.sub_id.to_numpy()
p = oof(d[GAZE], y, g)
obs = roc_auc_score(y, p)
null = np.array([roc_auc_score(yp, oof(d[GAZE], yp, g)) for yp in (rng.permutation(y) for _ in range(1000))])
p_extra = oof(d[GAZE + EXTRA], y, g)
coef = dict(zip(GAZE, np.round(model().fit(d[GAZE], y)[-1].coef_[0], 3).tolist()))
a2 = json.load(open(ROOT / "results" / "part_a.json"))["H2"]["coefficients"]
pairs = {"fixation count / rate": ("fix_num", "fix_rate"), "dispersion": ("fix_dispersion", "slide_dispersion"),
         "saccade length / step": ("sacc_length", "step_mean"), "horizontal moves": ("horizontal_sacc", "horizontal_prop")}
h9 = {k: {"reading": coef[b], "video": a2[v], "agree": bool(np.sign(coef[b]) == np.sign(a2[v]))} if b in coef
      else {"reading": None, "video": a2[v], "agree": None} for k, (b, v) in pairs.items()}
out = {"empty_features_dropped": EMPTY, "n_windows": int(len(d)), "n_readers": int(d.sub_id.nunique()),
       "H8": {"auc_roc": round(obs, 3), "auc_pr": round(average_precision_score(y, p), 3), "base_rate": round(y.mean(), 3),
              "perm_p": float((1 + (null >= obs).sum()) / 1001)},
       "H8_with_pupil_blink": {"auc_roc": round(roc_auc_score(y, p_extra), 3)},
       "coefficients": coef, "H9": h9}
json.dump(out, open(ROOT / "results" / "part_b.json", "w"), indent=1)
print(json.dumps(out, indent=1))
