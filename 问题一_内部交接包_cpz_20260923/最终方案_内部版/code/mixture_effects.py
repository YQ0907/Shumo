"""Small feasible composition-transfer scenarios, descriptive model responses only."""
from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from mixture_compare import ROOT, load_pair, make_model


def main():
    out = ROOT / "问题一研究" / "mixture_results"
    report = json.loads((out / "model_comparison.json").read_text(encoding="utf-8"))
    _, x, y, names, _, _ = load_pair("train", "1m")
    fits = {name: make_model(name, report["models"][name]["best_param"]).fit(x[:, :16], y)
            for name in ("linear_ridge", "quadratic_ridge", "log_excess_law", "extra_trees")}
    nn = NearestNeighbors(n_neighbors=2).fit(x)
    loo = nn.kneighbors(x, return_distance=True)[0][:, 1]
    radius = float(np.quantile(loo, .95))
    marginal_max = np.quantile(x, .95, axis=0)
    source = names.index("train_the_pile_pile_cc")
    delta = .01
    base = x[x[:, source] >= 2 * delta]

    def supported(candidates, target_indices):
        flag = np.ones(len(candidates[0]), bool)
        for candidate in candidates:
            flag &= nn.kneighbors(candidate, n_neighbors=1, return_distance=True)[0][:, 0] <= radius
            for j in target_indices:
                flag &= candidate[:, j] <= marginal_max[j]
        return flag

    def pred(model, xx):
        return model.predict(xx[:, :16]).mean(axis=1)

    rows = []
    for j in range(17):
        if j == source:
            continue
        one = base.copy(); one[:, source] -= delta; one[:, j] += delta
        keep = supported([one], [j])
        row = {"source": names[source], "target": names[j], "transfer_fraction": delta,
               "n_local_supported": int(keep.sum())}
        for name, model in fits.items():
            change = pred(model, one[keep]) - pred(model, base[keep]) if keep.any() else np.array([])
            row[f"{name}_mean_delta_loss"] = float(change.mean()) if len(change) else np.nan
            row[f"{name}_p10_delta_loss"] = float(np.quantile(change, .1)) if len(change) else np.nan
            row[f"{name}_p90_delta_loss"] = float(np.quantile(change, .9)) if len(change) else np.nan
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "one_point_transfers.csv", index=False)

    pairs = []
    for j, k in combinations([i for i in range(17) if i != source], 2):
        aj = base.copy(); aj[:, source] -= delta; aj[:, j] += delta
        ak = base.copy(); ak[:, source] -= delta; ak[:, k] += delta
        both = base.copy(); both[:, source] -= 2 * delta; both[:, j] += delta; both[:, k] += delta
        keep = supported([aj, ak, both], [j, k])
        row = {"source": names[source], "target_a": names[j], "target_b": names[k],
               "transfer_fraction_each": delta, "n_local_supported": int(keep.sum())}
        for name, model in fits.items():
            interaction = (pred(model, both[keep]) - pred(model, aj[keep]) -
                           pred(model, ak[keep]) + pred(model, base[keep])) if keep.any() else np.array([])
            row[f"{name}_mean_interaction"] = float(interaction.mean()) if len(interaction) else np.nan
        pairs.append(row)
    pd.DataFrame(pairs).to_csv(out / "two_domain_interactions.csv", index=False)
    (out / "effect_protocol.json").write_text(json.dumps({"source": names[source], "transfer_each": delta,
        "nearest_neighbor_radius_p95": radius, "base_n": len(base),
        "support": "nonnegative/simplex, target shares below training p95, and candidate nearest-neighbor distance below training leave-one-out p95",
        "interpretation": "Fitted 1M model scenario response, not a randomized causal effect or validated optimization result"},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows)[["target", "n_local_supported", "log_excess_law_mean_delta_loss",
                               "quadratic_ridge_mean_delta_loss", "extra_trees_mean_delta_loss"]].to_string(index=False))


if __name__ == "__main__":
    main()
