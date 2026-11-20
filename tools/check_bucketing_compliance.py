#!/usr/bin/env python3
"""Static compliance scanner & CI gate for FlagGems cross-backend shape bucketing.

Detects operator patterns that bypass runtime shape bucketing and trigger per-shape
JIT recompilation across backends:
  1. ``shape_stride_constexpr``: dynamic shape/stride/numel parameters annotated as ``tl.constexpr``
  2. ``wrapper_divisibility_kw``: wrapper-computed ``EVEN_*``/``ALIGNED_*``/``DIVISIBLE_*`` kwargs
  3. ``wrapper_next_pow2_block``: wrapper-computed ``BLOCK_* = triton.next_power_of_2(...)``
  4. ``op_level_bucketing_branch``: hardcoded ``if shape_bucketing:`` branches inside ops
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List


_PATTERNS = {
    "shape_stride_constexpr": re.compile(
        r"\b(M|N|K|L|S|B|SEQ_LEN|N_CTX|n_elements|numel|total|inner_size|outer_size|stride_[a-z0-9_]+|x_shape\d|shape_\d)\s*:\s*tl\.constexpr"
    ),
    "wrapper_divisibility_kw": re.compile(
        r"\b(EVEN_[A-Z0-9_]+|ALIGNED_[A-Z0-9_]+|DIVISIBLE_[A-Z0-9_]+)\s*="
    ),
    "wrapper_next_pow2_block": re.compile(
        r"\bBLOCK_[A-Z0-9_]*\s*=\s*triton\.next_power_of_2\("
    ),
    "op_level_bucketing_branch": re.compile(
        r"shape_bucketing"
    ),
}


def scan_file(path: Path) -> Dict[str, List[int]]:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    hits: Dict[str, List[int]] = {}
    for lineno, line in enumerate(txt.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for rule, pat in _PATTERNS.items():
            if pat.search(line):
                hits.setdefault(rule, []).append(lineno)
    return hits


def scan_flaggems(root: Path) -> Dict[str, Any]:
    src = root / "src/flag_gems"
    by_scope: Dict[str, Dict[str, Any]] = {}
    findings: List[Dict[str, Any]] = []

    scopes: List[tuple[str, List[Path]]] = [
        ("generic_ops", sorted((src / "ops").glob("*.py"))),
    ]
    bk_root = src / "runtime/backend"
    if bk_root.is_dir():
        for bk in sorted(bk_root.iterdir()):
            if bk.is_dir() and bk.name.startswith("_"):
                scopes.append((f"backend:{bk.name}", sorted(bk.rglob("*.py"))))

    for scope_name, files in scopes:
        rule_file_counts: Dict[str, int] = defaultdict(int)
        for fp in files:
            hits = scan_file(fp)
            if hits:
                rel = str(fp.relative_to(root))
                for rule in hits:
                    rule_file_counts[rule] += 1
                findings.append({"scope": scope_name, "file": rel, "rules": hits})
        by_scope[scope_name] = {
            "files_scanned": len(files),
            "rule_file_counts": dict(rule_file_counts),
        }

    return {
        "schema": "flaggems.bucketing_compliance.v1",
        "root": str(root),
        "by_scope": by_scope,
        "total_flagged_files": len(findings),
        "findings": findings,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FlagGems shape-bucketing compliance scanner")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json", dest="json_out", type=Path, default=None)
    parser.add_argument("--strict-paths", nargs="*", default=None, help="Fail if any listed relative paths have findings")
    args = parser.parse_args(argv)

    report = scan_flaggems(args.root)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(f"FlagGems Bucketing Compliance Report (flagged files: {report['total_flagged_files']})")
    for scope, info in report["by_scope"].items():
        rc = info["rule_file_counts"]
        if rc:
            print(f"  {scope:20s} (files={info['files_scanned']:4d}): {rc}")

    if args.strict_paths:
        strict_set = set(args.strict_paths)
        bad = [f for f in report["findings"] if f["file"] in strict_set]
        if bad:
            print(f"FAILED strict compliance check on {len(bad)} file(s):", [b["file"] for b in bad], file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
