"""Compare conflict policies on A1-A3 without using AI blind ratings for fitting."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import quality_compare as q

OUT = Path(__file__).resolve().parent / "conflict_results"
PAIRS = [("fineweb_edu", "qurater_3"),
         ("fluency_en", "modernbert_readability"),
         ("modernbert_professionalism", "qurater_1"),
         ("rps_doc_frac_chars_top_2gram", "rps_doc_frac_chars_top_3gram")]


def policy_consensus(u, raw, domains, refs):
    """Replace only a clear pairwise outlier using other indicators in its group.

    For ambiguous ties, keep both and retain an unresolved flag. No text labels
    or model scores are used to decide the correction.
    """
    corrected = u.copy()
    flagged_any = np.zeros(len(u), bool)
    unresolved_any = np.zeros(len(u), bool)
    details = {}
    raw_groups = {}
    for group, fields in q.GROUPS.items():
        raw_groups[group] = [f"qurater_{i}" if n == "qurater" else n
                             for n in fields for i in (range(4) if n == "qurater" else range(1))]
    for a, b in PAIRS:
        ia, ib = q.RAW.index(a), q.RAW.index(b)
        group = next(name for name, fields in raw_groups.items() if a in fields and b in fields)
        others = [q.RAW.index(name) for name in raw_groups[group] if name not in (a, b)]
        reliable = np.zeros(len(u), bool)
        for domain in sorted(set(domains)):
            mask = np.asarray(domains) == domain
            alo, ahi = np.quantile(refs[(domain, a)], [.01, .99])
            blo, bhi = np.quantile(refs[(domain, b)], [.01, .99])
            reliable[mask] = (np.isfinite(raw[mask, ia]) & np.isfinite(raw[mask, ib]) &
                              (raw[mask, ia] >= alo) & (raw[mask, ia] <= ahi) &
                              (raw[mask, ib] >= blo) & (raw[mask, ib] <= bhi))
        flag = reliable & (((u[:, ia] >= .8) & (u[:, ib] <= .2)) |
                           ((u[:, ib] >= .8) & (u[:, ia] <= .2)))
        consensus = np.nanmedian(u[:, others], axis=1)
        da, db = np.abs(u[:, ia] - consensus), np.abs(u[:, ib] - consensus)
        choose_a = flag & np.isfinite(consensus) & (da - db > .1)
        choose_b = flag & np.isfinite(consensus) & (db - da > .1)
        unresolved = flag & ~(choose_a | choose_b)
        corrected[choose_a, ia] = consensus[choose_a]
        corrected[choose_b, ib] = consensus[choose_b]
        flagged_any |= flag
        unresolved_any |= unresolved
        details[f"{a}|{b}"] = {"flagged": int(flag.sum()), "a_replaced": int(choose_a.sum()),
                               "b_replaced": int(choose_b.sum()), "unresolved": int(unresolved.sum())}
    return corrected, flagged_any, unresolved_any, details


def main():
    paths = [("A1", q.DATA / "slimpajama_quality_signal_sample.jsonl.xz", None),
             ("A2", next((q.DATA / "slimpajama_quality_extended").glob("arxiv_*.xz")), "arxiv"),
             ("A3", next((q.DATA / "slimpajama_quality_extended").glob("github_*.xz")), "github")]
    arrays, meta = [], []
    for label, path, hint in paths:
        values, rows = q.load(label, path, hint, None)
        arrays.append(values); meta.extend(rows)
    raw = np.concatenate(arrays)
    domains = [row[2] for row in meta]
    refs = q.reference(arrays[0], domains[:len(arrays[0])])
    u = q.orient(raw, domains, refs)
    base, _, eligible = q.score(u, u[:len(arrays[0])])
    corrected, flagged, unresolved, details = policy_consensus(u, raw, domains, refs)
    fixed, _, fixed_eligible = q.score(corrected, u[:len(arrays[0])])
    if not np.array_equal(eligible, fixed_eligible):
        raise ValueError("Correction changed eligibility")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "route_scores.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.writer(fh)
        wr.writerow(["source", "id", "domain", "conflict", "unresolved", "baseline_group_equal",
                     "robust_group_median", "consensus_correction", "manual_review_abstain"])
        for i, (source, ident, domain, _) in enumerate(meta):
            wr.writerow([source, ident, domain, int(flagged[i]), int(unresolved[i]),
                         base["group_equal"][i], base["group_median"][i],
                         fixed["group_equal"][i], "" if flagged[i] else base["group_equal"][i]])
    summary = {"policy": {"baseline": "keep both conflicting values, group mean",
                          "robust": "four-group median, no feature deletion",
                          "consensus": "if one conflicting value is >0.1 farther from other same-group indicators, replace it by their median; ties remain unresolved",
                          "abstain": "flag conflicting text for manual review, retain raw score only for others"},
               "pair_details": details, "sources": {}}
    for label in ("A1", "A2", "A3"):
        mask = np.fromiter((m[0] == label for m in meta), dtype=bool)
        summary["sources"][label] = {"n": int(mask.sum()), "conflict_n": int((flagged & mask).sum()),
                                      "unresolved_n": int((unresolved & mask).sum()),
                                      "mean_baseline": float(np.mean(base["group_equal"][mask])),
                                      "mean_consensus": float(np.mean(fixed["group_equal"][mask])),
                                      "mean_abs_change_conflict": float(np.mean(np.abs(
                                          fixed["group_equal"][mask & flagged] - base["group_equal"][mask & flagged])))
                                      if (mask & flagged).any() else 0.0}
    (OUT / "policy_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["sources"], ensure_ascii=False))


if __name__ == "__main__":
    main()
