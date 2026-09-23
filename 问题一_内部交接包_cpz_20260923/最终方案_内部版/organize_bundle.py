"""收集问题一代码、文献与小型结果的只读快照，并写机器可读清单。"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

GROUPS = {
    "problem": [
        "算力约束下提升大语言模型能力的资源配置建模.docx",
        "F题_精简重构版赛题.docx",
        "F题_精简重构版赛题.md",
        "F：数据说明-去除隐藏词.md",
        "题目分析报告.md",
        "使用指南.md",
    ],
    "code": [
        "问题一研究/quality_compare.py",
        "问题一研究/conflict_compare.py",
        "问题一研究/assess_ai_blind.py",
        "问题一研究/mixture_compare.py",
        "问题一研究/mixture_effects.py",
        "问题一研究/cross_session_audit.py",
        "问题一研究/cross_session_mixture.py",
        "问题一_质量与冲突实验.py",
        "20260923_cpz _问题一方案对比实验/对比实验_问题一.py",
    ],
    "literature": [
        "原题参考文献/01_Bahri_Explaining_neural_scaling_laws.pdf",
        "原题参考文献/02_Hoffmann_Training_compute_optimal_LLMs.pdf",
        "原题参考文献/03_Liu_RegMix.pdf",
        "原题参考文献/04_Biderman_Pythia.pdf",
        "原题参考文献/05_Asai_Synthesizing_scientific_literature.pdf",
        "原题参考文献/06_Schaeffer_Emergent_abilities_mirage.pdf",
        "原题参考文献/README_下载说明.md",
        "问题一研究/原题六篇论文与问题一的关系.md",
        "问题一_论文研读与模型路径比较.md",
    ],
    "evidence/quality": [
        "问题一研究/quality_results/domain_summary.csv",
        "问题一研究/quality_results/diagnostics.json",
        "问题一研究/quality_results/AI盲评核验摘要.json",
        "问题一研究/quality_results/原文盲评样本_子代理AI评分.csv",
    ],
    "evidence/conflict": [
        "问题一研究/conflict_results/policy_summary.json",
        "问题一研究/cross_session_results/quality_conflict_reconciliation.json",
    ],
    "evidence/mixture": [
        "问题一研究/mixture_results/model_comparison.json",
        "问题一研究/mixture_results/effect_protocol.json",
        "问题一研究/mixture_results/one_point_transfers.csv",
        "问题一研究/mixture_results/two_domain_interactions.csv",
        "问题一研究/mixture_results/test_1m_predictions.csv",
        "问题一研究/cross_session_results/harmonized_kernel_results.json",
    ],
    "evidence/other_session": [
        "20260923_cpz _问题一方案对比实验/问题一_三版方案比较与选择.md",
        "20260923_cpz _问题一方案对比实验/experiment_results.json",
        "问题一质量实验/A1_summary.csv",
        "问题一质量实验/A2_summary.csv",
        "问题一质量实验/A3_summary.csv",
    ],
    "notes": [
        "问题一研究/F_Q1-1与F_Q1-2综合比较.md",
        "问题一研究/问题一三阶段对比结果_供队伍复核.md",
        "问题一研究/质量阶段结果.md",
    ],
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    entries = []
    used = set()
    for group, paths in GROUPS.items():
        for relative in paths:
            src = ROOT / relative
            if not src.is_file():
                raise FileNotFoundError(src)
            dest = HERE / group / src.name
            if dest in used:
                raise ValueError(f"bundle name collision: {dest}")
            used.add(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            original_hash = digest(src)
            if digest(dest) != original_hash:
                raise ValueError(f"copy mismatch: {relative}")
            entries.append({"role": group, "source": relative.replace('\\','/'),
                            "bundle": dest.relative_to(HERE).as_posix(),
                            "bytes": src.stat().st_size, "sha256": original_hash})
    catalog = {"purpose": "Question 1 internal handoff; snapshots are copied, originals unchanged",
               "count": len(entries), "entries": entries,
               "not_copied": [
                   "real_attachments/A_data_value/* (raw source and 17-domain RegMix tables)",
                   "问题一研究/quality_results/sample_scores.csv (full rows in SQLite)",
                   "问题一研究/conflict_results/route_scores.csv (full row policy output)",
                   "问题一质量实验/A1_record_scores.csv / A2_record_scores.csv / A3_record_scores.csv (full rows in SQLite)",
               ]}
    (HERE / "bundle_manifest.json").write_text(json.dumps(catalog,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"bundled {len(entries)} files; manifest: {HERE/'bundle_manifest.json'}")


if __name__ == "__main__":
    main()
