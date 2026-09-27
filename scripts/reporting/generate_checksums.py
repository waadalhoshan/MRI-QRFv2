#!/usr/bin/env python3
"""Generate SHA256 checksums for public replication-package files."""

from pathlib import Path
import hashlib

ROOT=Path(__file__).resolve().parents[2]
EXCLUDE={".git","data","models","results"}
rows=[]
for p in sorted(ROOT.rglob("*")):
    if not p.is_file():
        continue
    rel=p.relative_to(ROOT)
    if rel.parts and rel.parts[0] in EXCLUDE:
        continue
    if rel.as_posix()=="CHECKSUMS.sha256":
        continue
    h=hashlib.sha256(p.read_bytes()).hexdigest()
    rows.append(f"{h}  {rel.as_posix()}")
(ROOT/"CHECKSUMS.sha256").write_text("\n".join(rows)+"\n",encoding="utf-8")
print(f"Wrote {len(rows)} checksums.")
