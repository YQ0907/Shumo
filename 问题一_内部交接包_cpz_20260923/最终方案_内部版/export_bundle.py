"""导出不含临时检查文件的内部交接 ZIP，并验证压缩包 CRC。"""
from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE.parent / "问题一_内部交接包_20260923.zip"
SKIP_DIRS = {"__pycache__", "smoke_quality", "smoke_mixture"}
SKIP_SUFFIXES = {".pyc", ".building.sqlite"}


def main():
    paths = sorted(p for p in HERE.rglob("*") if p.is_file() and
                   not any(part in SKIP_DIRS for part in p.relative_to(HERE).parts) and
                   not any(p.name.endswith(s) for s in SKIP_SUFFIXES))
    temp = TARGET.with_suffix(".building.zip")
    with zipfile.ZipFile(temp,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in paths:
            z.write(path,arcname=(Path(HERE.name)/path.relative_to(HERE)).as_posix())
    with zipfile.ZipFile(temp) as z:
        bad = z.testzip()
        if bad:
            raise ValueError(f"bad ZIP member: {bad}")
        if len(z.namelist()) != len(paths):
            raise ValueError("ZIP member count differs")
    temp.replace(TARGET)
    h=hashlib.sha256()
    with TARGET.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):
            h.update(block)
    print(f"archive={TARGET}\nfiles={len(paths)}\nbytes={TARGET.stat().st_size}\nsha256={h.hexdigest()}")


if __name__=="__main__":
    main()
