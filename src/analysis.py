"""Run the planned analyses H1-H7 (Part A) and write results/part_a.json."""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURES

ROOT = Path(__file__).resolve().parents[1]
DER, RES = ROOT / "data" / "derived", ROOT / "results"
RES.mkdir(exist_ok=True)
RAW = ROOT / "data" / "raw" / "wandering_eyes" / "Data"
SEED, N_PERM = 0, 1000
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)


def logreg():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))


def forest():
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestClassifier(500, class_weight="balanced", random_state=SEED, n_jobs=-1))


def oof(df, make, splitter, y=None):
    """Out-of-fold probabilities, folds grouped by participant."""
    y = df.y.to_numpy() if y is None else y
    p = np.zeros(len(df))
    for tr, te in splitter.split(df, y, df.pid):
        m = make().fit(df.iloc[tr][FEATURES], y[tr])
        p[te] = m.predict_proba(df.iloc[te][FEATURES])[:, 1]
    return p


def scores(y, p):
    return {"auc_roc": round(roc_auc_score(y, p), 3), "auc_pr": round(average_precision_score(y, p), 3),
            "base_rate": round(float(np.mean(y)), 3), "n": int(len(y))}


def boot_ci(y, p, groups, n=1000):
    """Participant-level bootstrap 95% CI for AUC-ROC."""
    ids = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ids}
    out = []
    for _ in range(n):
        s = np.concatenate([idx[g] for g in rng.choice(ids, len(ids))])
        if len(np.unique(y[s])) == 2:
            out.append(roc_auc_score(y[s], p[s]))
    return [round(float(np.percentile(out, 2.5)), 3), round(float(np.percentile(out, 97.5)), 3)]


def oof_perm(df, make, splitter, y):
    yp = rng.permutation(y)
    return roc_auc_score(yp, oof(df, make, splitter, yp))


def perm_test(df, make, splitter, observed):
    y = df.y.to_numpy()
    null = np.array([oof_perm(df, make, splitter, y) for _ in range(N_PERM)])
    return float((1 + np.sum(null >= observed)) / (N_PERM + 1))


out = {"exclusions": json.load(open(DER / "exclusions.json"))}
gkf = GroupKFold(n_splits=10)

# H1: probe-caught, leave-one-participant-out
a1 = pd.read_csv(DER / "a1_probe.csv")
logo = LeaveOneGroupOut()
p_lr = oof(a1, logreg, logo)
p_rf = oof(a1, forest, logo)
out["H1"] = {"logreg": scores(a1.y, p_lr) | {"ci95": boot_ci(a1.y.to_numpy(), p_lr, a1.pid.to_numpy()),
                                              "perm_p": perm_test(a1, logreg, logo, roc_auc_score(a1.y, p_lr))},
             "forest": scores(a1.y, p_rf)}
print("H1", out["H1"])

# H2: self-caught, grouped 10-fold
a2 = pd.read_csv(DER / "a2_self_lag0.csv")
p2 = oof(a2, logreg, gkf)
p2_rf = oof(a2, forest, gkf)
out["H2"] = {"logreg": scores(a2.y, p2) | {"ci95": boot_ci(a2.y.to_numpy(), p2, a2.pid.to_numpy()),
                                            "perm_p": perm_test(a2, logreg, gkf, roc_auc_score(a2.y, p2))},
             "forest": scores(a2.y, p2_rf)}
# coefficients of the full-data model, for interpretation only
m = logreg().fit(a2[FEATURES], a2.y)
out["H2"]["coefficients"] = dict(zip(FEATURES, np.round(m[-1].coef_[0], 3).tolist()))
print("H2", out["H2"])

# H3: how early
lags = [0, 5, 10, 20, 30]
auc_lag = []
for lag in lags:
    d = pd.read_csv(DER / f"a2_self_lag{lag}.csv")
    auc_lag.append(roc_auc_score(d.y, oof(d, logreg, gkf)))
rho, p_rho = stats.spearmanr(lags, auc_lag)
out["H3"] = {"lag_s": lags, "auc": [round(a, 3) for a in auc_lag], "spearman_rho": round(float(rho), 3),
             "p": round(float(p_rho), 3)}
print("H3", out["H3"])

# H5: transfer across lectures
tr = {}
for train, test in [("Education", "Genetics"), ("Genetics", "Education")]:
    a, b = a2[a2.video == train], a2[a2.video == test]
    pt = logreg().fit(a[FEATURES], a.y).predict_proba(b[FEATURES])[:, 1]
    within = oof(b.reset_index(drop=True), logreg, gkf)
    tr[f"{train}->{test}"] = {"transfer_auc": round(roc_auc_score(b.y, pt), 3),
                              "within_auc": round(roc_auc_score(b.y, within), 3)}
out["H5"] = tr
print("H5", tr)

# H4: GRU on fixation sequences vs. features, same folds
seqs = pickle.load(open(DER / "a2_sequences.pkl", "rb"))
key = {(s[0], s[2], s[3]): s[4] for s in seqs}
X = [key[(r.pid, r.end, r.y)] for r in a2.itertuples()]


class GRU(torch.nn.Module):
    def __init__(self, d=9, h=32):
        super().__init__()
        self.rnn = torch.nn.GRU(d, h, batch_first=True)
        self.out = torch.nn.Linear(h, 1)

    def forward(self, x, lengths):
        packed = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed)
        return self.out(h[-1]).squeeze(-1)


def batchify(idx):
    seq = [torch.tensor(X[i]) for i in idx]
    return torch.nn.utils.rnn.pad_sequence(seq, batch_first=True), torch.tensor([len(s) for s in seq])


y2 = a2.y.to_numpy()
p_gru = np.zeros(len(a2))
for tr_i, te_i in gkf.split(a2, y2, a2.pid):
    torch.manual_seed(SEED)
    net = GRU()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    pos_w = torch.tensor((y2[tr_i] == 0).sum() / max(1, (y2[tr_i] == 1).sum()), dtype=torch.float32)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
    for _ in range(30):
        perm = rng.permutation(tr_i)
        for b in range(0, len(perm), 64):
            idx = perm[b:b + 64]
            xb, lb = batchify(idx)
            opt.zero_grad()
            loss = loss_fn(net(xb, lb), torch.tensor(y2[idx], dtype=torch.float32))
            loss.backward()
            opt.step()
    with torch.no_grad():
        xb, lb = batchify(te_i)
        p_gru[te_i] = torch.sigmoid(net(xb, lb)).numpy()
out["H4"] = {"gru": scores(y2, p_gru), "logreg": scores(y2, p2),
             "difference_auc": round(roc_auc_score(y2, p_gru) - roc_auc_score(y2, p2), 3)}
print("H4", out["H4"])

# H6 and H7 use every 30 s window of the lecture
survey = pd.read_csv(RAW / "survey_data.csv")
survey["pid"] = survey.probe_type.map({"Self-caught": "self", "Probe-caught": "probe"}) + "_" + survey.video + "_" + survey.SubjectName
gs = pd.read_csv(DER / "grid_self.csv")
gp = pd.read_csv(DER / "grid_probe.csv")
gs, gp = gs[gs.coverage >= 0.5].reset_index(drop=True), gp[gp.coverage >= 0.5].reset_index(drop=True)
gs["p"] = np.nan
folds = list(gkf.split(a2, y2, a2.pid))
for tr_i, te_i in folds:
    m = logreg().fit(a2.iloc[tr_i][FEATURES], y2[tr_i])
    held = set(a2.iloc[te_i].pid)
    sel = gs.pid.isin(held)
    gs.loc[sel, "p"] = m.predict_proba(gs.loc[sel, FEATURES])[:, 1]
# self-caught participants not in a2 after exclusions are scored by the full model
full = logreg().fit(a2[FEATURES], y2)
miss = gs.p.isna()
gs.loc[miss, "p"] = full.predict_proba(gs.loc[miss, FEATURES])[:, 1]
gp["p"] = full.predict_proba(gp[FEATURES])[:, 1]

per = pd.concat([gs, gp]).groupby("pid").p.mean().rename("mw_pred").reset_index().merge(survey, on="pid")
per = per.dropna(subset=["PRE_TEST", "POST_TEST"])
rx = stats.linregress(per.PRE_TEST, per.mw_pred)
ry = stats.linregress(per.PRE_TEST, per.POST_TEST)
res_x = per.mw_pred - (rx.intercept + rx.slope * per.PRE_TEST)
res_y = per.POST_TEST - (ry.intercept + ry.slope * per.PRE_TEST)
r, p_two = stats.pearsonr(res_x, res_y)
out["H6"] = {"partial_r": round(float(r), 3), "p_one_sided": round(float(p_two / 2 if r < 0 else 1 - p_two / 2), 4),
             "n": int(len(per)), "raw_r": round(float(stats.pearsonr(per.mw_pred, per.POST_TEST)[0]), 3)}
print("H6", out["H6"])
per.to_csv(RES / "participants.csv", index=False)

# H7: when to prompt (self-caught participants only)
from data import self_reports  # noqa: E402

reports = pd.concat([self_reports(v) for v in ["Education", "Genetics"]])
thr = float(np.percentile(gs.p, 80))


def hit_share(times, r):
    if len(times) == 0:
        return np.nan
    return float(np.mean([np.any((r - t >= 0) & (r - t < 30_000)) for t in times]))


rows = []
for pid, g in gs.sort_values("end").groupby("pid"):
    r = reports[reports.pid == pid].t.to_numpy()
    trig, last = [], -1e12
    for e, p in zip(g.end, g.p):
        if p >= thr and e - last >= 60_000:
            trig.append(e)
            last = e
    k = len(trig)
    if k == 0 or len(r) == 0:
        continue
    ends = g.end.to_numpy()
    fixed = ends[np.linspace(0, len(ends) - 1, k + 2)[1:-1].round().astype(int)] if k else []
    rand = np.mean([hit_share(rng.choice(ends, k, replace=False), r) for _ in range(100)])
    rows.append({"pid": pid, "prompts": k, "reports": len(r), "triggered": hit_share(trig, r),
                 "fixed": hit_share(fixed, r), "random": rand})
sim = pd.DataFrame(rows)
w = stats.wilcoxon(sim.triggered, sim.fixed, alternative="greater")
out["H7"] = {"threshold": round(thr, 3), "participants": int(len(sim)),
             "mean_prompts": round(float(sim.prompts.mean()), 2),
             "hit_share_triggered": round(float(sim.triggered.mean()), 3),
             "hit_share_fixed": round(float(sim.fixed.mean()), 3),
             "hit_share_random": round(float(sim.random.mean()), 3),
             "wilcoxon_p": round(float(w.pvalue), 4)}
sim.to_csv(RES / "policy_simulation.csv", index=False)
print("H7", out["H7"])

a2.assign(p_logreg=p2, p_gru=p_gru).to_csv(RES / "a2_oof.csv", index=False)
gs.to_csv(RES / "grid_self_scored.csv", index=False)
json.dump(out, open(RES / "part_a.json", "w"), indent=1)
