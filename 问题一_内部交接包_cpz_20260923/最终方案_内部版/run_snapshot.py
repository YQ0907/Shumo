"""从 code/ 快照运行原问题一脚本，同时沿用原脚本的项目根目录语义。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGES = {
    "quality": "问题一研究/quality_compare.py",
    "conflict": "问题一研究/conflict_compare.py",
    "blind_audit": "问题一研究/assess_ai_blind.py",
    "mixture": "问题一研究/mixture_compare.py",
    "effects": "问题一研究/mixture_effects.py",
    "cross_quality": "问题一研究/cross_session_audit.py",
    "cross_mixture": "问题一研究/cross_session_mixture.py",
    "quality_alt": "问题一_质量与冲突实验.py",
    "alternative_compare": "20260923_cpz _问题一方案对比实验/对比实验_问题一.py",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("--help", "-h", "--list"):
        print("stages:")
        for stage, path in STAGES.items():
            print(f"  {stage:20s} {path}")
        print("usage: python run_snapshot.py STAGE [original script options]")
        return
    stage = sys.argv[1]
    if stage not in STAGES:
        raise SystemExit(f"unknown stage: {stage}; use --list")
    manifest = json.loads((HERE / "bundle_manifest.json").read_text(encoding="utf-8"))
    match = next(e for e in manifest["entries"] if e["source"] == STAGES[stage])
    snapshot = HERE / match["bundle"]
    original = HERE.parents[1] / STAGES[stage]
    assert snapshot.is_file() and original.is_file()
    # 被导入的同级模块取原路径；verify_bundle 保证它们与快照字节一致。
    sys.path.insert(0, str(original.parent))
    sys.argv = [str(snapshot), *sys.argv[2:]]
    namespace = {"__name__": "__main__", "__file__": str(original), "__package__": None}
    exec(compile(snapshot.read_bytes(), str(snapshot), "exec"), namespace)


if __name__ == "__main__":
    main()
