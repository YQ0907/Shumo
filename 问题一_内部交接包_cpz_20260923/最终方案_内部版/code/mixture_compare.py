"""Problem 1 mixture-response comparison. Reads A4-A15; never edits source data.

Model selection uses only A4/A5 five-fold CV. Cross-scale metrics are rank-only.
The A12-A15 targets are estimates and their mixtures overlap training mixtures.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "real_attachments" / "A_data_value" / "regmix_tables"


def load_pair(kind: str, scale: str):
    mix = pd.read_csv(DATA / f"{kind}_mixture_{scale}.csv")
    loss = pd.read_csv(DATA / f"{kind}_pile_loss_{scale}.csv")
    if not mix["index"].is_unique or not loss["index"].is_unique:
        raise ValueError(f"Duplicate mixture or loss index: {kind}/{scale}")
    joined = mix.merge(loss, on="index", validate="one_to_one", indicator=True)
    if len(joined) != len(mix) or len(joined) != len(loss) or not joined["_merge"].eq("both").all():
        raise ValueError(f"Unmatched mixture/loss index: {kind}/{scale}")
    xcols = [c for c in mix if c.startswith("train_the_pile_")]
    ycols = [c for c in loss if c.endswith("_val_loss")]
    x = joined[xcols].to_numpy(float)
    y = joined[ycols].to_numpy(float)
    if x.shape[1] != 17 or y.shape[1] != 13 or not np.isfinite(x).all() or not np.isfinite(y).all() or np.min(x) < 0:
        raise ValueError(f"Invalid shape/value: {kind}/{scale}")
    sums = x.sum(axis=1)
    if np.min(sums) <= 0 or np.max(np.abs(sums - 1)) > .01:
        raise ValueError(f"Invalid proportion sum: {kind}/{scale}")
    return joined["index"].to_numpy(), x / sums[:, None], y, xcols, ycols, float(np.max(np.abs(sums - 1)))


class LogExcessRidge:
    """Positive excess-loss mixture law: L_j(p)=c_j+exp(a_j+b_j' p).

    The floor is estimated from each training fold only. This is a constrained
    exponential response surrogate, not the full Data Mixing Laws model.
    """
    def __init__(self, alpha: float):
        self.alpha = alpha

    def fit(self, x, y):
        spread = np.maximum(np.ptp(y, axis=0), 1e-6)
        self.floor = np.min(y, axis=0) - .10 * spread
        self.model = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))
        self.model.fit(x, np.log(y - self.floor))
        return self

    def predict(self, x):
        return self.floor + np.exp(np.clip(self.model.predict(x), -30, 30))


def make_model(name: str, param: float):
    if name == "linear_ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=param))
    if name == "quadratic_ridge":
        return make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(), Ridge(alpha=param))
    if name == "rbf_kernel":
        return make_pipeline(StandardScaler(), KernelRidge(alpha=param, kernel="rbf", gamma=.01))
    if name == "extra_trees":
        return ExtraTreesRegressor(n_estimators=240, min_samples_leaf=int(param), max_features=1.0,
                                   random_state=20260923, n_jobs=-1)
    if name == "log_excess_law":
        return LogExcessRidge(param)
    raise ValueError(name)


def rank(y, pred):
    def rho(a, b):
        value = spearmanr(a, b).statistic
        return float(value) if np.isfinite(value) else None
    domains = [rho(y[:, j], pred[:, j]) for j in range(y.shape[1])]
    k = max(1, int(np.ceil(.1 * len(y))))
    actual_top = set(np.argsort(y.mean(axis=1))[:k])
    pred_top = set(np.argsort(pred.mean(axis=1))[:k])
    return {"mean_loss_spearman": rho(y.mean(axis=1), pred.mean(axis=1)),
            "top10pct_recall": len(actual_top & pred_top) / k,
            "top10pct_k": k,
            "domain_spearman_median": float(np.median([v for v in domains if v is not None]))
            if any(v is not None for v in domains) else None,
            "domain_spearman": domains}


def cv_score(y, pred, fold_ids):
    # Scales are computed on training rows of each fold, never on validation labels.
    domain_mse, overall_mae = [], []
    for tr, va in fold_ids:
        scale = np.maximum(y[tr].std(axis=0), 1e-8)
        domain_mse.append(float(np.mean(((y[va] - pred[va]) / scale) ** 2)))
        overall_mae.append(float(np.mean(np.abs(y[va].mean(axis=1) - pred[va].mean(axis=1)))))
    return {"domain_standardized_mse": float(np.mean(domain_mse)),
            "overall_mean_loss_mae": float(np.mean(overall_mae)),
            **rank(y, pred)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=ROOT / "问题一研究" / "mixture_results")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    datasets = {key: load_pair(*key) for key in [("train", "1m"), ("test", "1m"),
                ("test", "60m"), ("test", "1B"), ("est", "10b"), ("est", "70b")]}
    idx, x, y, xcols, ycols, train_error = datasets[("train", "1m")]
    z = x[:, :16]  # 17th fraction is determined by the simplex sum.
    folds = list(KFold(n_splits=args.folds, shuffle=True, random_state=20260923).split(z))
    grids = {"linear_ridge": [.01, .1, 1., 10., 100.],
             "quadratic_ridge": [1., 10., 100., 1000.],
             "rbf_kernel": [.001, .01, .1, 1.],
             "extra_trees": [1., 2., 4.],
             "log_excess_law": [.01, .1, 1., 10., 100.]}
    report = {"protocol": {"folds": args.folds, "seed": 20260923, "selection": "training CV standardized 13-domain MSE",
                           "scale_metrics": "1M absolute and rank; 60M/1B rank only; 10B/70B estimated and nonindependent",
                           "x_features": "first 16 normalized proportions; 17th is 1 minus sum",
                           "test_independence_caveat": "An earlier workspace experiment examined the test sets; comparisons are post-hoc exploratory."},
              "data": {}, "models": {}}
    train_keys = {tuple(np.round(v, 6)) for v in x}
    for key, (ids, xx, yy, _, _, err) in datasets.items():
        report["data"]["_".join(key)] = {"n": len(xx), "max_raw_sum_error": err,
             "overlap_train": int(sum(tuple(np.round(v, 6)) in train_keys for v in xx)) if key[0] != "train" else len(xx)}

    predictions = []
    for name, params in grids.items():
        candidates = []
        for param in params:
            oof = np.empty_like(y)
            for tr, va in folds:
                model = make_model(name, param).fit(z[tr], y[tr])
                oof[va] = model.predict(z[va])
            candidates.append((param, cv_score(y, oof, folds)))
        best_param, best_cv = min(candidates, key=lambda item: item[1]["domain_standardized_mse"])
        model = make_model(name, best_param).fit(z, y)
        entry = {"best_param": best_param, "train_cv": best_cv,
                 "candidate_cv": [{"param": p, **s} for p, s in candidates], "checks": {}}
        for key, (ids, xx, yy, _, _, _) in datasets.items():
            if key[0] == "train":
                continue
            pred = model.predict(xx[:, :16])
            score = rank(yy, pred)
            if key == ("test", "1m"):
                score.update({"overall_mean_loss_mae": float(np.mean(np.abs(yy.mean(axis=1) - pred.mean(axis=1)))),
                              "overall_mean_loss_rmse": float(np.sqrt(np.mean((yy.mean(axis=1) - pred.mean(axis=1)) ** 2))),
                              "domain_mae_mean": float(np.mean(np.abs(yy - pred)))})
                predictions.append(pd.DataFrame({"index": ids, "model": name,
                    "actual_mean_loss": yy.mean(axis=1), "pred_mean_loss": pred.mean(axis=1)}))
            entry["checks"]["_".join(key)] = score
        report["models"][name] = entry
        print(name, best_param, best_cv["domain_standardized_mse"], entry["checks"]["test_1m"], flush=True)

    baseline = np.repeat(y.mean(axis=0)[None, :], len(datasets[("test", "1m")][2]), axis=0)
    yt = datasets[("test", "1m")][2]
    report["constant_1m_baseline"] = {"overall_mean_loss_mae": float(np.mean(np.abs(yt.mean(axis=1) - baseline.mean(axis=1)))),
                                      "domain_mae_mean": float(np.mean(np.abs(yt - baseline)))}
    by_model = {frame.model.iloc[0]: frame for frame in predictions}
    draw = np.random.default_rng(20260923).integers(0, len(yt), size=(3000, len(yt)))
    report["posthoc_paired_bootstrap"] = {}
    for name, frame in by_model.items():
        current = np.abs(frame.actual_mean_loss.to_numpy() - frame.pred_mean_loss.to_numpy())
        comparisons = {}
        for reference in ("linear_ridge", "log_excess_law"):
            ref = by_model[reference]
            if not np.array_equal(frame["index"], ref["index"]):
                raise ValueError("Prediction indices differ across models")
            delta = current - np.abs(ref.actual_mean_loss.to_numpy() - ref.pred_mean_loss.to_numpy())
            comparisons[f"mae_delta_vs_{reference}_95ci"] = [float(v) for v in np.quantile(delta[draw].mean(axis=1), [.025, .975])]
        report["posthoc_paired_bootstrap"][name] = comparisons
    (args.output / "model_comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.concat(predictions, ignore_index=True).to_csv(args.output / "test_1m_predictions.csv", index=False)


if __name__ == "__main__":
    main()
