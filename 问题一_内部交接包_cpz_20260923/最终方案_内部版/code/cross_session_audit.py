"""Read-only reconciliation of F_Q1-1 and F_Q1-2 quality/conflict outputs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "问题一研究" / "cross_session_results"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    first = pd.read_csv(ROOT / "问题一研究" / "quality_results" / "sample_scores.csv",
                        usecols=["source", "id", "domain", "group_equal", "conflict"])
    human_proxy = pd.read_csv(ROOT / "问题一研究" / "quality_results" / "AI盲评与模型对照_内部.csv",
                              usecols=["id", "domain", "AI片段综合分"])
    summary = {"provenance": "F_Q1-1 and F_Q1-2 workspace outputs; 135 AI snippet ratings are not human labels",
               "sources": {}, "ai_snippet_check": {}}
    for source in ("A1", "A2", "A3"):
        second = pd.read_csv(ROOT / "问题一质量实验" / f"{source}_record_scores.csv",
                             usecols=["id", "Q", "semantic_conflict", "fineweb_vs_qurater_education",
                                      "fluency_vs_readability", "professionalism_vs_expertise", "two_vs_three_gram"])
        part = first[first.source.eq(source)].merge(second, on="id", validate="one_to_one")
        sem = part[["fineweb_vs_qurater_education", "fluency_vs_readability",
                    "professionalism_vs_expertise"]].any(axis=1)
        ngram = part.two_vs_three_gram.eq(1)
        summary["sources"][source] = {"n": len(part),
            "mean_group_q_f_q1_1": float(part.group_equal.mean()),
            "mean_group_q_f_q1_2": float(part.Q.mean()),
            "record_score_spearman": float(spearmanr(part.group_equal, part.Q).statistic),
            "record_score_mae": float(np.mean(np.abs(part.group_equal - part.Q))),
            "all4_conflicts_f_q1_1": int(part.conflict.sum()),
            "all4_conflicts_f_q1_2": int(part.semantic_conflict.sum()),
            "semantic3_conflicts_f_q1_2": int(sem.sum()),
            "ngram_conflicts_f_q1_2": int(ngram.sum()),
            "ngram_only_f_q1_2": int((ngram & ~sem).sum()),
            "f_q1_2_only": int((part.semantic_conflict.eq(1) & part.conflict.eq(0)).sum()),
            "f_q1_1_only": int((part.conflict.eq(1) & part.semantic_conflict.eq(0)).sum())}
        if source == "A1":
            labeled = part.merge(human_proxy, on=["id", "domain"], validate="one_to_one")
            for name in ("group_equal", "Q"):
                within = [spearmanr(g["AI片段综合分"], g[name]).statistic
                          for _, g in labeled.groupby("domain")]
                summary["ai_snippet_check"][name] = {"n": len(labeled),
                    "pooled_spearman": float(spearmanr(labeled["AI片段综合分"], labeled[name]).statistic),
                    "median_within_domain_spearman": float(np.median(within))}
            other_only = labeled.semantic_conflict.eq(1) & labeled.conflict.eq(0)
            summary["ai_snippet_check"]["f_q1_2_only_conflicts"] = {
                "n": int(other_only.sum()),
                "mean_ai_snippet_score": float(labeled.loc[other_only, "AI片段综合分"].mean())}
    (OUT / "quality_conflict_reconciliation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
