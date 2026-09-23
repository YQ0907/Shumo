"""Harmonize F_Q1-1 and F_Q1-2 mixture-model comparisons on one CV metric.

No external test labels enter model selection; results are still post-hoc because
both sessions previously inspected those test tables.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import helmert
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mixture_compare import ROOT, cv_score, load_pair, rank


OUT = ROOT / "问题一研究" / "cross_session_results"


def ilr(x, eps):
    z = np.where(x == 0, eps, x)
    z = z / z.sum(axis=1, keepdims=True)
    return np.log(z) @ helmert(17, full=False).T


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = {key: load_pair(*key) for key in [("train", "1m"), ("test", "1m"),
                ("test", "60m"), ("test", "1B"), ("est", "10b"), ("est", "70b")]}
    _, x, y, _, _, _ = data[("train", "1m")]
    folds = list(KFold(n_splits=5, shuffle=True, random_state=20260923).split(x))
    report = {"protocol": "Same A4/A5 five folds, training-fold 13-domain standard deviations, 16D raw or ilr features; no test-label model choice",
              "grids": {}, "selected": {}}
    candidates = []
    for geometry in ("raw16", "ilr"):
        for eps in ([None] if geometry == "raw16" else [.0001, .001, .01]):
            xx = x[:, :16] if eps is None else ilr(x, eps)
            for alpha in [.001, .01, .1, 1.]:
                for gamma in [.001, .01, .1]:
                    oof = np.empty_like(y)
                    for tr, va in folds:
                        model = make_pipeline(StandardScaler(), KernelRidge(kernel="rbf", alpha=alpha, gamma=gamma))
                        model.fit(xx[tr], y[tr])
                        oof[va] = model.predict(xx[va])
                    score = cv_score(y, oof, folds)
                    item = {"geometry": geometry, "eps": eps, "alpha": alpha, "gamma": gamma, "cv": score}
                    candidates.append(item)
    report["grids"]["candidates"] = candidates
    for geometry, eps in [("raw16", None), ("ilr", .0001), ("ilr", .001), ("ilr", .01)]:
        subset = [v for v in candidates if v["geometry"] == geometry and v["eps"] == eps]
        best = min(subset, key=lambda v: v["cv"]["domain_standardized_mse"])
        xx = x[:, :16] if eps is None else ilr(x, eps)
        model = make_pipeline(StandardScaler(), KernelRidge(kernel="rbf", alpha=best["alpha"], gamma=best["gamma"]))
        model.fit(xx, y)
        entry = {"eps": eps, "alpha": best["alpha"], "gamma": best["gamma"], "train_cv": best["cv"], "checks": {}}
        for key, (_, tx, ty, _, _, _) in data.items():
            if key[0] == "train":
                continue
            xt = tx[:, :16] if eps is None else ilr(tx, eps)
            pred = model.predict(xt)
            score = rank(ty, pred)
            if key == ("test", "1m"):
                score.update({"overall_mean_loss_mae": float(np.mean(np.abs(ty.mean(axis=1) - pred.mean(axis=1)))),
                              "domain_mae_mean": float(np.mean(np.abs(ty - pred)))})
            entry["checks"]["_".join(key)] = score
        name = geometry if eps is None else f"ilr_eps_{eps:g}"
        report["selected"][name] = entry
        print(name, "CV", best["cv"]["domain_standardized_mse"], "1M domain MAE",
              entry["checks"]["test_1m"]["domain_mae_mean"], flush=True)
    (OUT / "harmonized_kernel_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
