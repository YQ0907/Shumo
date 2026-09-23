"""Audit AI-agent blind readings against quality signal routes (not human validation)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parent / "quality_results"
LABELS = ["人工信息价值_0到3", "人工可读性_0到3", "人工噪声_0到3"]
ROUTES = ["group_equal", "flat_equal", "group_median", "pca_structural"]


def read_ratings(path: Path):
    df = pd.read_csv(path, dtype={"id": str})
    if df.id.duplicated().any() or df[LABELS].isna().any().any():
        raise ValueError(f"Incomplete/duplicate ratings: {path}")
    for col in LABELS:
        if not df[col].isin(range(4)).all():
            raise ValueError(f"Noninteger score: {path}/{col}")
    return df


def correlation(a, b):
    if len(a) < 5 or np.std(a) == 0 or np.std(b) == 0:
        return None
    v = spearmanr(a, b).statistic
    return float(v) if np.isfinite(v) else None


def main():
    blind = pd.read_csv(ROOT / "原文盲评样本.csv", dtype={"id": str})
    a = read_ratings(ROOT / "盲评_子代理A_1到68.csv")
    b = read_ratings(ROOT / "盲评_子代理B_69到135.csv")
    if a.id.tolist() != blind.id.iloc[:68].tolist() or b.id.tolist() != blind.id.iloc[68:].tolist():
        raise ValueError("Agent ratings do not match assigned blind rows")
    ratings = pd.concat([a, b], ignore_index=True)
    if ratings.id.nunique() != len(blind) or len(ratings) != len(blind):
        raise ValueError("Blind-rating coverage error")
    filled = blind.drop(columns=[*LABELS, "人工备注"]).merge(ratings, on="id", validate="one_to_one")
    filled = filled.rename(columns={LABELS[0]: "AI信息价值_0到3", LABELS[1]: "AI可读性_0到3",
                                    LABELS[2]: "AI噪声_0到3", "人工备注": "AI备注"})
    filled["评分来源"] = "AI子代理，仅前1500字符"
    filled.to_csv(ROOT / "原文盲评样本_子代理AI评分.csv", index=False)
    scores = pd.read_csv(ROOT / "sample_scores.csv", usecols=["source", "id", "domain", "conflict", *ROUTES])
    scores = scores[scores.source.eq("A1")].copy()
    if scores.id.duplicated().any():
        raise ValueError("Duplicate A1 score id")
    merged = blind[["id", "domain"]].merge(ratings, on="id", validate="one_to_one")
    merged = merged.merge(scores.drop(columns="source"), on=["id", "domain"], validate="one_to_one")
    if len(merged) != len(blind):
        raise ValueError("A1 score matching failed")
    merged = merged.rename(columns={LABELS[0]: "AI信息价值_0到3", LABELS[1]: "AI可读性_0到3",
                                    LABELS[2]: "AI噪声_0到3", "人工备注": "AI备注"})
    merged["AI片段综合分"] = (merged["AI信息价值_0到3"] + merged["AI可读性_0到3"] + 3 - merged["AI噪声_0到3"]) / 9
    route_summary = {}
    for name in ROUTES:
        per_domain = {domain: correlation(part["AI片段综合分"], part[name])
                      for domain, part in merged.groupby("domain")}
        valid = [v for v in per_domain.values() if v is not None]
        route_summary[name] = {"pooled_spearman": correlation(merged["AI片段综合分"], merged[name]),
                               "median_within_domain_spearman": float(np.median(valid)) if valid else None,
                               "within_domain_spearman": per_domain}
        merged[f"{name}_域内秩差"] = merged.groupby("domain")[name].rank(pct=True) - merged.groupby("domain")["AI片段综合分"].rank(pct=True)

    policies = pd.read_csv(ROOT.parent / "conflict_results" / "route_scores.csv",
                           usecols=["source", "id", "consensus_correction", "manual_review_abstain"])
    policies = policies[policies.source.eq("A1")].drop(columns="source")
    merged = merged.merge(policies, on="id", validate="one_to_one")
    conflict_routes = {}
    for name in ["group_equal", "group_median", "consensus_correction", "manual_review_abstain"]:
        for subset, part in [("all", merged), ("flagged", merged[merged.conflict.eq(1)])]:
            part = part[part[name].notna()]
            conflict_routes[f"{name}/{subset}"] = {"n": len(part),
                "spearman": correlation(part["AI片段综合分"], part[name])}

    overlap = pd.concat([read_ratings(ROOT / "复评_子代理A_20条.csv"),
                         read_ratings(ROOT / "复评_子代理B_20条.csv")], ignore_index=True)
    if len(overlap) != 40 or overlap.id.nunique() != 40:
        raise ValueError("Expected 40 unique cross-rated items")
    same = ratings.merge(overlap, on="id", suffixes=("_first", "_second"), validate="one_to_one")
    agreement = {}
    for col in LABELS:
        first, second = same[col + "_first"], same[col + "_second"]
        agreement[col] = {"n": len(same), "exact_fraction": float(np.mean(first == second)),
                          "within_one_fraction": float(np.mean(np.abs(first - second) <= 1)),
                          "quadratic_weighted_kappa": float(cohen_kappa_score(first, second, weights="quadratic"))}
    conflicts = merged.groupby("conflict").agg(n=("id", "size"), mean_ai_score=("AI片段综合分", "mean"),
                                                 mean_group_score=("group_equal", "mean")).reset_index()
    summary = {"provenance": "Two AI subagents read disjoint 1500-character snippets blind to model scores; 40 snippets independently cross-rated. This is not human ground truth.",
               "n": len(merged), "domain_counts": merged.domain.value_counts().to_dict(),
               "routes": route_summary, "conflict_route_checks": conflict_routes,
               "inter_agent_agreement": agreement,
               "conflict_strata": conflicts.to_dict(orient="records")}
    merged.to_csv(ROOT / "AI盲评与模型对照_内部.csv", index=False)
    (ROOT / "AI盲评核验摘要.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
