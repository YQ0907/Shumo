"""校验交接包快照、数据库核心约束和论文文件。无需训练模型。"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from organize_bundle import HERE, ROOT, digest


def main():
    catalog = json.loads((HERE / "bundle_manifest.json").read_text(encoding="utf-8"))
    entries = catalog["entries"]
    assert len(entries) == catalog["count"] and len(entries) >= 30
    for e in entries:
        src, dst = ROOT / e["source"], HERE / e["bundle"]
        assert src.is_file() and dst.is_file(), e
        assert src.stat().st_size == dst.stat().st_size == e["bytes"], e
        assert digest(src) == digest(dst) == e["sha256"], e
    db = HERE / "data/q1_evidence.sqlite"
    with sqlite3.connect(db) as con:
        assert con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert not con.execute("PRAGMA foreign_key_check").fetchall()
        expected = {"quality_record": 272505, "quality_recheck": 272505,
                    "ai_blind_rating": 135, "mixture_run": 1214,
                    "mixture_weight": 20638, "validation_loss": 15782,
                    "domain_mapping": 17, "bibliography": 6}
        for table, n in expected.items():
            assert con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == n, table
        assert con.execute("SELECT source,SUM(semantic_conflict_three) FROM quality_recheck GROUP BY source ORDER BY source").fetchall() == [
            ("A1",965),("A2",0),("A3",4173)]
        assert con.execute("SELECT split,scale,SUM(overlaps_train) FROM mixture_run GROUP BY split,scale ORDER BY split,scale").fetchall() == [
            ("estimated","10b",63),("estimated","70b",63),("test","1B",0),
            ("test","1m",0),("test","60m",0),("train","1m",512)]
        for ref, pdf in con.execute("SELECT ref_id,local_pdf FROM bibliography"):
            assert (ROOT / pdf).is_file() and (HERE / "literature" / Path(pdf).name).is_file(), ref
    for name in ("README.md","问题一_论文内部稿.md","章节_数据_文献映射.md","代码与运行说明.md"):
        assert (HERE / name).is_file(), name
    print(f"OK: {len(entries)} snapshots, {len(expected)} database tables, 6 PDFs, paper and documentation")


if __name__ == "__main__":
    main()
