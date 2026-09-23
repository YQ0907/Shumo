"""问题一质量信号与冲突的可复现实验；输出仅供队伍校验建模假设。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import lzma
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import scipy
from scipy.special import expit, softmax


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "real_attachments" / "A_data_value"
PATHS = {
    "A1": DATA / "slimpajama_quality_signal_sample.jsonl.xz",
    "A2": DATA / "slimpajama_quality_extended" / "arxiv_part-6777d8857c6e-000486.jsonl.xz",
    "A3": DATA / "slimpajama_quality_extended" / "github_part-6777d8857c6e-000275.jsonl.xz",
}
FIELDS = [
    "fineweb_edu", "fluency_en", "modernbert_cleanliness",
    "modernbert_readability", "modernbert_reasoning", "modernbert_professionalism",
    "dsir_books", "dsir_wiki", "dsir_math", "qurater", "ad_en",
    "rps_doc_word_count", "rps_doc_num_sentences", "rps_doc_unigram_entropy",
    "rps_doc_frac_unique_words", "rps_doc_frac_no_alph_words",
    "rps_doc_frac_chars_top_2gram", "rps_doc_frac_chars_top_3gram",
    "rps_lines_uppercase_letter_fraction",
    "rps_lines_ending_with_terminal_punctution_mark",
    "rps_lines_numerical_chars_fraction", "rps_doc_mean_word_length",
]
EXTRA = ["qurater_style", "qurater_expertise", "qurater_facts", "qurater_education"]
COLS = FIELDS + EXTRA
INDEX = {name: i for i, name in enumerate(COLS)}
GROUPS = {
    "education": [0, 4, 5, 6, 7, 8, 9],
    "readability": [1, 3, 19],
    "cleanliness": [2, 10, 16, 17],
    "structure": [11, 12, 13, 14, 15, 18, 20, 21],
}
INTERVAL = {11, 12, 13, 14, 15, 18, 19, 20, 21}
NEGATIVE = {16, 17}
PAIR_NAMES = [
    ("fineweb_vs_qurater_education", 0, INDEX["qurater_education"]),
    ("fluency_vs_readability", 1, 3),
    ("professionalism_vs_expertise", 5, INDEX["qurater_expertise"]),
    ("two_vs_three_gram", 16, 17),
]


def number(value):
    try:
        out = float(value)
        return out if np.isfinite(out) else np.nan
    except (ValueError, TypeError):
        return np.nan


def vector(value, length):
    try:
        if isinstance(value, str):
            value = json.loads(value)
        arr = np.asarray(value, dtype=float)
        if arr.shape == (length,) and np.isfinite(arr).all():
            return arr
    except (ValueError, TypeError, json.JSONDecodeError):
        pass
    return None


def decode(row):
    x = np.full(len(COLS), np.nan)
    one = vector(row.get("fineweb_edu"), 1)
    if one is not None:
        x[0] = one[0]
    for j, name in [(1, "fluency_en"), (10, "ad_en")]:
        v = vector(row.get(name), 2)
        if v is not None:
            x[j] = expit(v[1] - v[0])
    for j in [2, 3, 4, 5]:
        v = vector(row.get(FIELDS[j]), 6)
        if v is not None:
            x[j] = np.dot(softmax(v), np.arange(6)) / 5
    for j in [6, 7, 8] + list(range(11, 22)):
        x[j] = number(row.get(FIELDS[j]))
    q = vector(row.get("qurater"), 4)
    if q is not None:
        x[22:26] = q
    for j in [11, 12]:
        if np.isfinite(x[j]) and x[j] >= 0:
            x[j] = np.log1p(x[j])
        else:
            x[j] = np.nan
    return x


def read_source(path, source, limit, limit_per_domain=None):
    ids, domains, rows = [], [], []
    domain_counts = Counter()
    with lzma.open(path, "rt", encoding="utf-8") as stream:
        for n, line in enumerate(stream):
            if limit is not None and len(ids) >= limit:
                break
            row = json.loads(line)
            domain = row.get("_source_domain") or ("arxiv" if source == "A2" else "github")
            if limit_per_domain is not None and domain_counts[domain] >= limit_per_domain:
                continue
            ids.append(row["id"])
            domains.append(domain)
            rows.append(decode(row))
            domain_counts[domain] += 1
    return {"source": source, "id": np.asarray(ids), "domain": np.asarray(domains),
            "raw": np.vstack(rows)}


def percentile_ranks(values, reference, reverse=False):
    """Midrank ECDF on A1 reference; ties receive the same score."""
    out = np.full(values.shape, np.nan)
    reference = np.sort(reference[np.isfinite(reference)])
    valid = np.isfinite(values)
    if len(reference) == 0:
        return out
    lo = np.searchsorted(reference, values[valid], "left")
    hi = np.searchsorted(reference, values[valid], "right")
    ranks = (lo + hi) / (2 * len(reference))
    out[valid] = 1 - ranks if reverse else ranks
    return out


def interval_scores(values, reference):
    """A1 same-domain 10–90% typical interval, linear fade to 1/99% tails."""
    out = np.full(values.shape, np.nan)
    reference = reference[np.isfinite(reference)]
    if len(reference) < 30:
        return out
    q01, q10, q90, q99 = np.quantile(reference, [.01, .10, .90, .99])
    good = np.isfinite(values)
    v = values[good]
    score = np.ones(len(v))
    below = v < q10
    above = v > q90
    score[below] = (v[below] - q01) / max(q10 - q01, 1e-12)
    score[above] = (q99 - v[above]) / max(q99 - q90, 1e-12)
    out[good] = np.clip(score, 0, 1)
    return out


def score_set(data, anchor):
    x, a = data["raw"], anchor["raw"]
    u = np.full_like(x, np.nan)
    for j in range(len(COLS)):
        if j == 9:
            continue
        if j in INTERVAL:
            for domain in np.unique(data["domain"]):
                sel = data["domain"] == domain
                ref = a[anchor["domain"] == domain, j]
                if np.isfinite(ref).sum() < 30:
                    ref = a[:, j]
                u[sel, j] = interval_scores(x[sel, j], ref)
        else:
            u[:, j] = percentile_ranks(x[:, j], a[:, j], reverse=j in NEGATIVE)
    # The predeclared model uses all four QuRating components equally. Its
    # expertise direction remains an explicit, unvalidated assumption.
    u[:, 9] = np.nanmean(u[:, 22:26], axis=1)
    groups = {}
    for name, idx in GROUPS.items():
        count = np.isfinite(u[:, idx]).sum(axis=1)
        groups[name] = np.divide(np.nansum(u[:, idx], axis=1), count,
                                 out=np.full(len(x), np.nan), where=count > 0)
    g = np.column_stack(list(groups.values()))
    coverage = np.isfinite(u[:, :22]).sum(axis=1)
    valid = (coverage >= 11) & np.isfinite(g).all(axis=1)
    q = np.where(valid, g.mean(axis=1), np.nan)
    q_median = np.where(valid, np.median(g, axis=1), np.nan)
    q_equal22 = np.where(valid, np.nanmean(u[:, :22], axis=1), np.nan)
    # Pairwise conflict is assessed only when BOTH raw values lie inside
    # the A1 reference domain's 1–99% range.
    reliable = np.isfinite(x)
    for j in sorted(set(j for _, aa, bb in PAIR_NAMES for j in (aa, bb))):
        for domain in np.unique(data["domain"]):
            sel = data["domain"] == domain
            ref = a[anchor["domain"] == domain, j]
            ref = ref[np.isfinite(ref)]
            if len(ref) < 30:
                ref = a[np.isfinite(a[:, j]), j]
            if len(ref):
                lo, hi = np.quantile(ref, [.01, .99])
                reliable[sel, j] &= (x[sel, j] >= lo) & (x[sel, j] <= hi)
            else:
                reliable[sel, j] = False
    pair_flags = {}
    for name, j, k in PAIR_NAMES:
        opposite = ((u[:, j] >= .8) & (u[:, k] <= .2)) | ((u[:, k] >= .8) & (u[:, j] <= .2))
        pair_flags[name] = opposite & reliable[:, j] & reliable[:, k]
    pair_any = np.column_stack(list(pair_flags.values())).any(axis=1)
    tradeoff = (groups["education"] >= .8) & (groups["cleanliness"] <= .2)
    return {"u": u, "groups": groups, "coverage": coverage, "valid": valid,
            "Q": q, "Q_median": q_median, "Q_equal22": q_equal22,
            "pairs": pair_flags, "pair_any": pair_any, "tradeoff": tradeoff}


def summarize(data, result, masks, output):
    rows = []
    for label, sel in masks.items():
        n = int(sel.sum())
        if n == 0:
            continue
        q = result["Q"][sel]
        good = q[np.isfinite(q)]
        row = {"subset": label, "n": n, "scored_n": len(good),
               "reject_rate": 1 - len(good) / n,
               "Q_mean": float(np.mean(good)) if len(good) else np.nan,
               "Q_median": float(np.median(good)) if len(good) else np.nan,
               "Q_p10": float(np.quantile(good, .1)) if len(good) else np.nan,
               "Q_p90": float(np.quantile(good, .9)) if len(good) else np.nan,
               "Q_group_median_mean": float(np.nanmean(result["Q_median"][sel])),
               "Q_equal22_mean": float(np.nanmean(result["Q_equal22"][sel])),
               "semantic_conflict_rate": float(np.mean(result["pair_any"][sel])),
               "tradeoff_rate": float(np.mean(result["tradeoff"][sel]))}
        for name, values in result["groups"].items():
            row[name + "_mean"] = float(np.nanmean(values[sel]))
        for name, values in result["pairs"].items():
            row[name + "_rate"] = float(np.mean(values[sel]))
        rows.append(row)
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="per-source rows for P1 smoke test")
    parser.add_argument("--limit-per-domain", type=int, default=None,
                        help="retain at most this many real records per A1 domain for P1")
    parser.add_argument("--output", type=Path, default=ROOT / "问题一质量实验")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.time()
    datasets = {name: read_source(path, name, args.limit, args.limit_per_domain if name == "A1" else None)
                for name, path in PATHS.items()}
    anchor = datasets["A1"]
    all_rows = []
    results = {}
    a1_ids = set(anchor["id"])
    for name, data in datasets.items():
        result = score_set(data, anchor)
        results[name] = result
        masks = {name + ":all": np.ones(len(data["id"]), dtype=bool)}
        for domain in np.unique(data["domain"]):
            masks[name + ":" + domain] = data["domain"] == domain
        if name != "A1":
            masks[name + ":exclusive"] = ~np.isin(data["id"], list(a1_ids))
            masks[name + ":overlap"] = np.isin(data["id"], list(a1_ids))
        all_rows.extend(summarize(data, result, masks, args.output / f"{name}_summary.csv"))
    # A1–extended identical IDs must yield identical scores and flags.
    lookup = {key: i for i, key in enumerate(anchor["id"])}
    overlap = {}
    for name in ("A2", "A3"):
        pairs = [(lookup[key], j) for j, key in enumerate(datasets[name]["id"]) if key in lookup]
        overlap[name] = len(pairs)
        if pairs:
            left, right = map(np.asarray, zip(*pairs))
            assert np.allclose(results["A1"]["Q"][left], results[name]["Q"][right], equal_nan=True)
            assert np.array_equal(results["A1"]["pair_any"][left], results[name]["pair_any"][right])
    for name, data in datasets.items():
        res = results[name]
        with (args.output / f"{name}_record_scores.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["id", "domain", "Q", "Q_group_median", "Q_equal22", "coverage",
                             "semantic_conflict", "education_cleanliness_tradeoff"]
                            + list(res["groups"]) + list(res["pairs"]))
            for i, id_ in enumerate(data["id"]):
                writer.writerow([id_, data["domain"][i], res["Q"][i], res["Q_median"][i],
                                 res["Q_equal22"][i], res["coverage"][i], int(res["pair_any"][i]),
                                 int(res["tradeoff"][i])]
                                + [v[i] for v in res["groups"].values()]
                                + [int(v[i]) for v in res["pairs"].values()])
    # Fixed per-domain IDs and text previews for human evaluation; no fabricated labels.
    picks = []
    r = results["A1"]
    for domain in np.unique(anchor["domain"]):
        idx = np.flatnonzero((anchor["domain"] == domain) & r["valid"])
        order = idx[np.argsort(r["Q"][idx])]
        for label, pos in [("low", 0), ("middle", len(order)//2), ("high", len(order)-1)]:
            if len(order):
                picks.append((str(anchor["id"][order[pos]]), domain, label, float(r["Q"][order[pos]])))
        conflicting = idx[r["pair_any"][idx]]
        if len(conflicting):
            picks.append((str(anchor["id"][conflicting[0]]), domain, "semantic_conflict", float(r["Q"][conflicting[0]])))
    chosen = {row[0]: row for row in picks}
    previews = {}
    with lzma.open(PATHS["A1"], "rt", encoding="utf-8") as stream:
        for line in stream:
            obj = json.loads(line)
            if obj["id"] in chosen:
                previews[obj["id"]] = obj.get("content", "")[:500].replace("\n", " ")
            if len(previews) == len(chosen):
                break
    with (args.output / "A1_原文核验样例.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "domain", "stratum", "Q", "content_preview", "human_information", "human_noise"])
        for id_, domain, label, q in picks:
            writer.writerow([id_, domain, label, q, previews.get(id_, ""), "", ""])
    # Disjoint calibration/evaluation sets for actual human ratings. Neither
    # displayed sheet reveals score, stratum, or domain to raters.
    rng = np.random.default_rng(20260923)
    blind_meta = []
    for domain in np.unique(anchor["domain"]):
        idx = np.flatnonzero((anchor["domain"] == domain) & r["valid"])
        low, midlo, midhi, high = np.quantile(r["Q"][idx], [.2, .4, .6, .8])
        eligible = {
            "low": idx[(r["Q"][idx] <= low) & ~r["pair_any"][idx]],
            "middle": idx[(r["Q"][idx] >= midlo) & (r["Q"][idx] <= midhi) & ~r["pair_any"][idx]],
            "high": idx[(r["Q"][idx] >= high) & ~r["pair_any"][idx]],
            "conflict": idx[r["pair_any"][idx]],
        }
        for stratum, candidates in eligible.items():
            chosen_idx = rng.choice(candidates, size=min(6, len(candidates)), replace=False)
            for pos, i in enumerate(chosen_idx):
                blind_meta.append({"id": str(anchor["id"][i]), "domain": domain,
                                   "stratum": stratum, "split": "calibration" if pos < 3 else "evaluation",
                                   "Q": float(r["Q"][i])})
    blind_ids = {item["id"] for item in blind_meta}
    blind_text = {}
    with lzma.open(PATHS["A1"], "rt", encoding="utf-8") as stream:
        for line in stream:
            obj = json.loads(line)
            if obj["id"] in blind_ids:
                blind_text[obj["id"]] = obj.get("content", "")[:2000].replace("\n", " ")
            if len(blind_text) == len(blind_ids):
                break
    with (args.output / "A1_人工评价索引_勿给评审.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(blind_meta[0]))
        writer.writeheader()
        writer.writerows(blind_meta)
    for split in ("calibration", "evaluation"):
        subset = [item for item in blind_meta if item["split"] == split]
        rng.shuffle(subset)
        with (args.output / f"A1_匿名人工评价_{split}.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["id", "content_preview", "information_value_0to5", "readability_0to5", "noise_0to5", "notes"])
            for item in subset:
                writer.writerow([item["id"], blind_text.get(item["id"], ""), "", "", "", ""])
    audits = {}
    for name, data in datasets.items():
        counts = Counter(data["id"])
        invalid = {col: int(np.isnan(data["raw"][:, j]).sum())
                   for j, col in enumerate(COLS) if j != 9}
        invalid["qurater"] = int(np.any(np.isnan(data["raw"][:, 22:26]), axis=1).sum())
        audits[name] = {"duplicate_id_rows": int(sum(v - 1 for v in counts.values())),
                        "missing_or_invalid_by_field": invalid,
                        "domain_counts": dict(Counter(data["domain"]))}
    manifest = {"run_seconds": round(time.time()-start, 2), "limit_per_source": args.limit,
                "limit_per_domain_A1": args.limit_per_domain, "audit": audits,
                "input_sha256": {name: sha256(path) for name, path in PATHS.items()},
                "n": {name: len(data["id"]) for name, data in datasets.items()},
                "overlap_with_A1": overlap, "python": sys.version, "platform": platform.platform(),
                "numpy": np.__version__, "scipy": scipy.__version__,
                "model": "A1 ECDF, four equal groups, semantic pair thresholds 0.8/0.2",
                "seed": 20260923, "command": "python 问题一_质量与冲突实验.py --output 问题一质量实验"}
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": manifest["n"], "overlap": overlap,
                      "A1_summary": [x for x in all_rows if x["subset"].startswith("A1:")]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
