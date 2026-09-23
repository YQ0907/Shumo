"""问题一方案对比的只读数据实验；输出不写回赛题附件。"""
from __future__ import annotations

import hashlib
import json
import lzma
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import helmert
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "real_attachments" / "A_data_value"
OUT = ROOT / "问题一方案对比实验"
OUT.mkdir(exist_ok=True)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def load_pair(prefix: str, scale: str):
    base = DATA / "regmix_tables"
    mix_path = base / f"{prefix}_mixture_{scale}.csv"
    loss_path = base / f"{prefix}_pile_loss_{scale}.csv"
    mix = pd.read_csv(mix_path)
    loss = pd.read_csv(loss_path)
    assert mix["index"].is_unique and loss["index"].is_unique
    merged = mix.merge(loss, on="index", validate="one_to_one", indicator=True)
    assert len(merged) == len(mix) == len(loss)
    assert (merged["_merge"] == "both").all()
    xcols = [c for c in mix if c.startswith("train_the_pile_")]
    ycols = [c for c in loss if c.endswith("_val_loss")]
    x = merged[xcols].to_numpy(dtype=float)
    y = merged[ycols].to_numpy(dtype=float)
    assert x.shape[1] == 17 and y.shape[1] == 13
    assert np.isfinite(x).all() and np.isfinite(y).all() and (x >= 0).all()
    row_error = np.abs(x.sum(axis=1) - 1)
    x = x / x.sum(axis=1, keepdims=True)
    return x, y, {"rows": len(x), "max_row_sum_error": float(row_error.max()),
                  "mixture_sha256": sha(mix_path), "loss_sha256": sha(loss_path)}


def ilr(x: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    y = np.where(x == 0, eps, x)
    y = y / y.sum(axis=1, keepdims=True)
    log = np.log(y)
    return log @ helmert(17, full=False).T


def pred_score(y: np.ndarray, yp: np.ndarray) -> dict:
    mean_y = y.mean(axis=1)
    mean_yp = yp.mean(axis=1)
    rho = spearmanr(mean_y, mean_yp).statistic if np.std(mean_yp) > 1e-12 else float("nan")
    domain_rhos = [spearmanr(y[:, j], yp[:, j]).statistic
                   if np.std(yp[:, j]) > 1e-12 else float("nan") for j in range(y.shape[1])]
    return {"overall_mae": float(np.mean(np.abs(mean_y - mean_yp))),
            "overall_rmse": float(np.sqrt(np.mean((mean_y - mean_yp)**2))),
            "overall_spearman": float(rho),
            "median_domain_spearman": float(np.nanmedian(domain_rhos))
            if np.isfinite(domain_rhos).any() else float("nan"),
            "domain_mae_mean": float(np.mean(np.abs(y - yp)))}


def composition_experiment():
    xtr, ytr, train_info = load_pair("train", "1m")
    xte, yte, test_info = load_pair("test", "1m")
    x60, y60, info60 = load_pair("test", "60m")
    x1b, y1b, info1b = load_pair("test", "1B")
    x10, y10, info10 = load_pair("est", "10b")
    x70, y70, info70 = load_pair("est", "70b")
    train_keys = {tuple(np.round(row, 7)) for row in xtr}
    overlaps = {name: sum(tuple(np.round(row, 7)) in train_keys for row in xx)
                for name, xx in [("test_1m", xte), ("test_60m", x60),
                                 ("test_1B", x1b), ("est_10b", x10), ("est_70b", x70)]}
    ymean = ytr.mean(axis=0)
    cv = KFold(n_splits=5, shuffle=True, random_state=20260923)
    alphas = [0.01, 0.1, 1, 10, 100]
    features = {
        "raw16_linear": (xtr[:, :16], lambda x: x[:, :16],
                         make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": alphas}),
        "raw16_quadratic": (xtr[:, :16], lambda x: x[:, :16],
                            make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(), Ridge()),
                            {"ridge__alpha": [1, 10, 100, 1000]}),
        "ilr_linear_eps1e-3": (ilr(xtr), ilr,
                                make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": alphas}),
        "ilr_rbf_eps1e-3": (ilr(xtr), ilr,
                             make_pipeline(StandardScaler(), KernelRidge(kernel="rbf")),
                             {"kernelridge__alpha": [0.001, 0.01, 0.1, 1],
                              "kernelridge__gamma": [0.001, 0.01, 0.1]}),
        "raw16_rbf": (xtr[:, :16], lambda x: x[:, :16],
                      make_pipeline(StandardScaler(), KernelRidge(kernel="rbf")),
                      {"kernelridge__alpha": [0.001, 0.01, 0.1, 1],
                       "kernelridge__gamma": [0.001, 0.01, 0.1]}),
        "ilr_rbf_eps1e-4": (ilr(xtr, 1e-4), lambda x: ilr(x, 1e-4),
                             make_pipeline(StandardScaler(), KernelRidge(kernel="rbf")),
                             {"kernelridge__alpha": [0.001, 0.01, 0.1, 1],
                              "kernelridge__gamma": [0.001, 0.01, 0.1]}),
        "ilr_rbf_eps1e-2": (ilr(xtr, 1e-2), lambda x: ilr(x, 1e-2),
                             make_pipeline(StandardScaler(), KernelRidge(kernel="rbf")),
                             {"kernelridge__alpha": [0.001, 0.01, 0.1, 1],
                              "kernelridge__gamma": [0.001, 0.01, 0.1]}),
    }
    result = {"data": {"train": train_info, "test_1m": test_info,
                       "test_60m": info60, "test_1B": info1b,
                       "est_10b": info10, "est_70b": info70,
                       "overlap_with_train": overlaps}, "models": {}}
    constant = np.tile(ymean, (len(yte), 1))
    result["models"]["constant_baseline"] = {"test_1m": pred_score(yte, constant)}
    predictions = {}
    predictions_all = {}
    def balanced_fold_mse(estimator, xval, yval):
        """Validation-fold metric; no held-out target enters training preprocessing."""
        pred = estimator.predict(xval)
        scale = np.maximum(yval.std(axis=0), 1e-8)
        return -float(np.mean(((yval - pred) / scale) ** 2))
    for name, (xfit, transform, model, grid) in features.items():
        search = GridSearchCV(model, grid, scoring=balanced_fold_mse, cv=cv, n_jobs=1)
        search.fit(xfit, ytr)
        fitted = search.best_estimator_
        predict = lambda x: fitted.predict(transform(x))
        pte = predict(xte)
        predictions[name] = pte.mean(axis=1)
        predictions_all[name] = pte
        result["models"][name] = {
            "train_cv_mse_standardized": float(-search.best_score_),
            "best_params": search.best_params_,
            "test_1m": pred_score(yte, pte),
            "test_60m_rank_only": pred_score(y60, predict(x60)),
            "test_1B_rank_only": pred_score(y1b, predict(x1b)),
            "est_10b_rank_only_nonindependent": pred_score(y10, predict(x10)),
            "est_70b_rank_only_nonindependent": pred_score(y70, predict(x70)),
        }
    # Paired bootstrap intervals for differences in overall absolute error vs raw linear.
    base_err = np.abs(yte.mean(axis=1) - predictions["raw16_linear"])
    rng = np.random.default_rng(20260923)
    ids = rng.integers(0, len(yte), size=(2000, len(yte)))
    for name, pred in predictions.items():
        delta = np.abs(yte.mean(axis=1) - pred) - base_err
        result["models"][name]["mae_difference_vs_raw_linear_95ci"] = [
            float(v) for v in np.quantile(delta[ids].mean(axis=1), [0.025, 0.975])]
    quad_err = np.abs(yte.mean(axis=1) - predictions["raw16_quadratic"])
    for name, pred in predictions.items():
        delta = np.abs(yte.mean(axis=1) - pred) - quad_err
        result["models"][name]["mae_difference_vs_quadratic_95ci"] = [
            float(v) for v in np.quantile(delta[ids].mean(axis=1), [0.025, 0.975])]
        domain_delta = np.abs(yte - predictions_all[name]).mean(axis=1) - np.abs(
            yte - predictions_all["raw16_quadratic"]).mean(axis=1)
        result["models"][name]["domain_mae_difference_vs_quadratic_95ci"] = [
            float(v) for v in np.quantile(domain_delta[ids].mean(axis=1), [0.025, 0.975])]
    return result


def quality_audit():
    files = [("A1", DATA / "slimpajama_quality_signal_sample.jsonl.xz", None),
             ("A2", next((DATA / "slimpajama_quality_extended").glob("arxiv_*.xz")), "arxiv"),
             ("A3", next((DATA / "slimpajama_quality_extended").glob("github_*.xz")), "github")]
    seen = {}
    out = {}
    sample_rows_by_domain = {}
    sampled_seen = {}
    rng = np.random.default_rng(20260923)
    for label, path, domain_hint in files:
        ids = set()
        domain_counts = {}
        parse_error = 0
        missing = {}
        with lzma.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except Exception:
                    parse_error += 1
                    continue
                ident = row.get("id")
                if ident is not None:
                    ids.add(ident)
                domain = domain_hint or row.get("_source_domain", "unknown")
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
                if label == "A1":
                    sampled_seen[domain] = sampled_seen.get(domain, 0) + 1
                    sample_rows_by_domain.setdefault(domain, [])
                    bucket = sample_rows_by_domain[domain]
                    if len(bucket) < 1000:
                        bucket.append(row)
                    else:
                        j = rng.integers(0, sampled_seen[domain])
                        if j < 1000:
                            bucket[j] = row
                if label == "A1":
                    for k, v in row.items():
                        if v is None:
                            missing[k] = missing.get(k, 0) + 1
        seen[label] = ids
        out[label] = {"file": str(path), "sha256": sha(path), "rows": sum(domain_counts.values()),
                      "unique_ids": len(ids), "parse_error": parse_error,
                      "domain_counts": domain_counts, "null_counts": missing}
    out["id_overlap"] = {"A1_A2": len(seen["A1"] & seen["A2"]),
                         "A1_A3": len(seen["A1"] & seen["A3"]),
                         "A2_A3": len(seen["A2"] & seen["A3"])}
    # The PCA is a structural diagnostic, not a label-based quality evaluation.
    sample_rows = [row for bucket in sample_rows_by_domain.values() for row in bucket]
    fields = [k for k in sample_rows[0] if k not in
              {"id", "content", "sub_path", "_source_domain", "_source_path"}]
    matrix = []
    for row in sample_rows:
        vals = []
        for k in fields:
            v = row.get(k)
            if isinstance(v, str) and v.startswith("["):
                try: v = json.loads(v)
                except Exception: v = None
            if isinstance(v, list):
                if not v: v = None
                elif k == "ad_en" or k == "fluency_en": v = float(v[1]) - float(v[0])
                elif k.startswith("modernbert_"):
                    logits = np.array(v, dtype=float)
                    p = np.exp(logits - logits.max())
                    v = float((p * np.arange(len(v))).sum() / p.sum())
                elif k == "qurater": v = float(v[3])
                else: v = float(v[0])
            try: v = float(v)
            except (TypeError, ValueError): v = np.nan
            vals.append(v)
        matrix.append(vals)
    arr = np.array(matrix)
    from scipy.stats import rankdata, norm
    for j in range(arr.shape[1]):
        col = arr[:, j]
        good = np.isfinite(col)
        if good.sum() < 100: continue
        q = rankdata(col[good]) / (good.sum() + 1)
        col[good] = norm.ppf(q)
        col[~good] = 0
        arr[:, j] = col
    pca = PCA().fit(arr)
    out["pca_diagnostic_A1_stratified_sample"] = {
        "sample_rows": len(sample_rows),
        "sample_by_domain": {k: len(v) for k, v in sample_rows_by_domain.items()},
        "fields": len(fields), "pc1_explained_variance": float(pca.explained_variance_ratio_[0]),
        "first_4_cumulative": float(pca.explained_variance_ratio_[:4].sum()),
        "note": "仅结构诊断，未经22指标方向校正，不代表质量真值"}
    return out


if __name__ == "__main__":
    result = {"quality_audit": quality_audit(), "mixture_experiment": composition_experiment()}
    target = OUT / "experiment_results.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(target)
    for model, row in result["mixture_experiment"]["models"].items():
        print(model, row.get("test_1m"))
