"""Question 1 quality-signal comparison. Exploratory scores, not human ground truth.

Run from the project root:
    python 问题一研究/quality_compare.py --limit 200 --output 问题一研究/smoke
    python 问题一研究/quality_compare.py --output 问题一研究/quality_results
"""
from __future__ import annotations

import argparse
import csv
import json
import lzma
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "real_attachments" / "A_data_value"

FIELDS = [
    "fineweb_edu", "fluency_en", "modernbert_cleanliness",
    "modernbert_readability", "modernbert_reasoning",
    "modernbert_professionalism", "dsir_books", "dsir_wiki", "dsir_math",
    "qurater", "ad_en", "rps_doc_word_count", "rps_doc_num_sentences",
    "rps_doc_unigram_entropy", "rps_doc_frac_unique_words",
    "rps_doc_frac_no_alph_words", "rps_doc_frac_chars_top_2gram",
    "rps_doc_frac_chars_top_3gram", "rps_lines_uppercase_letter_fraction",
    "rps_lines_ending_with_terminal_punctution_mark",
    "rps_lines_numerical_chars_fraction", "rps_doc_mean_word_length",
]
RAW = [f"qurater_{i}" if f == "qurater" else f for f in FIELDS for i in (range(4) if f == "qurater" else range(1))]
POS = {
    "fineweb_edu", "fluency_en", "modernbert_cleanliness",
    "modernbert_readability", "modernbert_reasoning",
    "modernbert_professionalism", "dsir_books", "dsir_wiki", "dsir_math",
    "qurater_0", "qurater_1", "qurater_2", "qurater_3", "ad_en",
}
NEG = {"rps_doc_frac_chars_top_2gram", "rps_doc_frac_chars_top_3gram"}
INTERVAL = set(RAW) - POS - NEG
GROUPS = {
    "education": ["fineweb_edu", "modernbert_reasoning", "modernbert_professionalism",
                  "dsir_books", "dsir_wiki", "dsir_math", "qurater"],
    "readability": ["fluency_en", "modernbert_readability",
                    "rps_lines_ending_with_terminal_punctution_mark"],
    "cleanliness": ["modernbert_cleanliness", "ad_en",
                    "rps_doc_frac_chars_top_2gram", "rps_doc_frac_chars_top_3gram"],
    "structure": ["rps_doc_word_count", "rps_doc_num_sentences",
                  "rps_doc_unigram_entropy", "rps_doc_frac_unique_words",
                  "rps_doc_frac_no_alph_words", "rps_lines_uppercase_letter_fraction",
                  "rps_lines_numerical_chars_fraction", "rps_doc_mean_word_length"],
}


def scalarize(row: dict) -> list[float]:
    out = []
    for name in FIELDS:
        value = row.get(name)
        if name == "qurater":
            seq = value if isinstance(value, list) and len(value) == 4 else [math.nan] * 4
            out.extend(float(x) for x in seq)
        elif name in {"ad_en", "fluency_en"}:
            out.append(float(value[1]) - float(value[0]) if isinstance(value, list) and len(value) == 2 else math.nan)
        elif name.startswith("modernbert_"):
            if isinstance(value, list) and len(value) == 6:
                z = np.asarray(value, dtype=float)
                if np.isfinite(z).all():
                    p = np.exp(z - z.max()); out.append(float((p @ np.arange(6)) / p.sum()))
                else:
                    out.append(math.nan)
            else:
                out.append(math.nan)
        elif name == "fineweb_edu":
            out.append(float(value[0]) if isinstance(value, list) and len(value) == 1 else math.nan)
        else:
            x = float(value) if value is not None else math.nan
            if name in {"rps_doc_word_count", "rps_doc_num_sentences"}:
                x = math.log1p(x) if x >= 0 else math.nan
            out.append(x)
    return out


def load(label: str, path: Path, domain_hint: str | None, limit: int | None):
    rows, meta = [], []
    with lzma.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if limit is not None and len(rows) >= limit:
                break
            record = json.loads(line)
            rows.append(scalarize(record))
            meta.append((label, record.get("id"), domain_hint or record.get("_source_domain", "unknown"),
                         str(record.get("content", "")).replace("\n", " ")[:240]))
    return np.asarray(rows, dtype=float), meta


def reference(values: np.ndarray, domains: list[str]):
    # Semantic signals share a pooled A1 scale so domain means remain comparable.
    # Only structural typicality signals use domain references.
    refs = {}
    for j, name in enumerate(RAW):
        if name not in INTERVAL:
            a = values[:, j]
            refs[("pooled", name)] = np.sort(a[np.isfinite(a)])
    for domain in sorted(set(domains)):
        v = values[np.array(domains) == domain]
        for j, name in enumerate(RAW):
            a = v[:, j]
            a = np.sort(a[np.isfinite(a)])
            if len(a) < 50:
                raise ValueError(f"insufficient A1 reference: {domain}/{name}: {len(a)}")
            refs[(domain, name)] = a
    return refs


def orient(values: np.ndarray, domains: list[str], refs: dict):
    out = np.full(values.shape, np.nan)
    for domain in sorted(set(domains)):
        mask = np.array(domains) == domain
        for j, name in enumerate(RAW):
            x = values[mask, j]
            ok = np.isfinite(x)
            a = refs[(domain if name in INTERVAL else "pooled", name)]
            q = np.empty(len(x)); q.fill(np.nan)
            if name in INTERVAL:
                lo1, lo10, hi90, hi99 = np.quantile(a, [0.01, 0.10, 0.90, 0.99])
                xx = x[ok]
                below = np.maximum(lo10 - xx, 0) / max(lo10 - lo1, 1e-12)
                above = np.maximum(xx - hi90, 0) / max(hi99 - hi90, 1e-12)
                q[ok] = np.clip(1 - below - above, 0, 1)
            else:
                q[ok] = np.searchsorted(a, x[ok], side="right") / len(a)
                if name in NEG:
                    q[ok] = 1 - q[ok]
            out[mask, j] = q
    return out


def collapse(u: np.ndarray):
    qstart = RAW.index("qurater_0")
    q = np.nanmean(u[:, qstart:qstart + 4], axis=1)
    cols = [u[:, RAW.index(name)] if name != "qurater" else q for name in FIELDS]
    return np.column_stack(cols)


def score(u: np.ndarray, a1_u: np.ndarray):
    x = collapse(u)
    coverage = np.isfinite(x).sum(axis=1)
    g = np.column_stack([np.nanmean(x[:, [FIELDS.index(n) for n in names]], axis=1)
                         for names in GROUPS.values()])
    eligible = (coverage >= 11) & np.isfinite(g).all(axis=1)
    equal_group = np.nanmean(g, axis=1)
    flat = np.nanmean(x, axis=1)
    median_group = np.nanmedian(g, axis=1)
    # A low relative percentile is not evidence of an advertisement or corruption.
    # Such flags need calibration against text before any hard score cap is used.
    for arr in (equal_group, flat, median_group):
        arr[~eligible] = np.nan
    # PCA is a structural comparator only: no external quality labels are used.
    a1 = collapse(a1_u)
    impute = np.nanmedian(a1, axis=0)
    a1 = np.where(np.isfinite(a1), a1, impute)
    xx = np.where(np.isfinite(x), x, impute)
    scaler = StandardScaler().fit(a1)
    pca = PCA(n_components=1, random_state=20260923).fit(scaler.transform(a1))
    pc = pca.transform(scaler.transform(xx))[:, 0]
    if np.corrcoef(pca.transform(scaler.transform(a1))[:, 0],
                   np.nanmean(a1, axis=1))[0, 1] < 0:
        pc = -pc
    a1_pc = pca.transform(scaler.transform(a1))[:, 0]
    if np.corrcoef(a1_pc, np.nanmean(a1, axis=1))[0, 1] < 0:
        a1_pc = -a1_pc
    pc_ref = np.sort(a1_pc)
    pc_score = np.searchsorted(pc_ref, pc, side="right") / len(pc_ref)
    pc_score[~eligible] = np.nan
    return {"group_equal": equal_group, "flat_equal": flat,
            "group_median": median_group, "pca_structural": pc_score}, g, eligible


def conflicts(u: np.ndarray, raw: np.ndarray, domains: list[str], refs: dict):
    pairs = [("fineweb_edu", "qurater_3"),
             ("fluency_en", "modernbert_readability"),
             ("modernbert_professionalism", "qurater_1"),
             ("rps_doc_frac_chars_top_2gram", "rps_doc_frac_chars_top_3gram")]
    flags = np.zeros(len(u), dtype=bool)
    by_pair = {}
    for a, b in pairs:
        x, y = u[:, RAW.index(a)], u[:, RAW.index(b)]
        reliable = np.zeros(len(u), dtype=bool)
        for domain in sorted(set(domains)):
            mask = np.array(domains) == domain
            a_ref, b_ref = refs[(domain, a)], refs[(domain, b)]
            alo, ahi = np.quantile(a_ref, [0.01, 0.99])
            blo, bhi = np.quantile(b_ref, [0.01, 0.99])
            ar, br = raw[mask, RAW.index(a)], raw[mask, RAW.index(b)]
            reliable[mask] = (np.isfinite(ar) & np.isfinite(br) &
                              (ar >= alo) & (ar <= ahi) & (br >= blo) & (br <= bhi))
        conflict = reliable & (((x >= .8) & (y <= .2)) | ((y >= .8) & (x <= .2)))
        flags |= conflict
        by_pair[f"{a}|{b}"] = int(conflict.sum())
    return flags, by_pair


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, default=ROOT / "问题一研究" / "quality_results")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 100:
        parser.error("--limit must be >=100 for domain references")
    paths = [("A1", DATA / "slimpajama_quality_signal_sample.jsonl.xz", None),
             ("A2", next((DATA / "slimpajama_quality_extended").glob("arxiv_*.xz")), "arxiv"),
             ("A3", next((DATA / "slimpajama_quality_extended").glob("github_*.xz")), "github")]
    arrays, metadata = [], []
    for label, path, hint in paths:
        x, m = load(label, path, hint, None if label == "A1" else args.limit)
        arrays.append(x); metadata.extend(m)
    all_x = np.concatenate(arrays)
    domains = [row[2] for row in metadata]
    a1_n = len(arrays[0])
    refs = reference(arrays[0], domains[:a1_n])
    u = orient(all_x, domains, refs)
    scores, groups, eligible = score(u, u[:a1_n])
    conflict, pair_counts = conflicts(u, all_x, domains, refs)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "sample_scores.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.writer(fh); wr.writerow(["source", "id", "domain", "eligible", "conflict", *scores, "text_excerpt"])
        for i, (source, ident, domain, excerpt) in enumerate(metadata):
            wr.writerow([source, ident, domain, int(eligible[i]), int(conflict[i]),
                         *[scores[k][i] for k in scores], excerpt if source == "A1" else ""])
    summary = []
    for source in ("A1", "A2", "A3"):
        for domain in sorted(set(domains)):
            ix = np.array([(s == source and d == domain) for s, _, d, _ in metadata])
            if not ix.any(): continue
            item = {"source": source, "domain": domain, "n": int(ix.sum()),
                    "eligible_n": int(eligible[ix].sum()), "conflict_n": int(conflict[ix].sum())}
            for name, arr in scores.items():
                v = arr[ix]; v = v[np.isfinite(v)]
                item[name + "_mean"] = float(v.mean()) if len(v) else math.nan
                item[name + "_median"] = float(np.median(v)) if len(v) else math.nan
            summary.append(item)
    with (args.output / "domain_summary.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(summary[0])); wr.writeheader(); wr.writerows(summary)
    diagnostics = {"counts": dict(Counter(s for s, _, _, _ in metadata)),
                   "nonfinite_raw": {name: int((~np.isfinite(all_x[:, j])).sum()) for j, name in enumerate(RAW)},
                   "pair_conflicts": pair_counts,
                   "score_correlations_A1": {f"{a}|{b}": float(np.corrcoef(scores[a][:a1_n], scores[b][:a1_n])[0, 1])
                                             for a in scores for b in scores if a < b},
                   "note": "PCA is structural, and no path has human quality ground truth here."}
    (args.output / "diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"counts": diagnostics["counts"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
