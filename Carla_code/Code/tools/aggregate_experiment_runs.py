#!/usr/bin/env python3
"""Aggregate raw RecoverVLA episode logs without inventing missing results.

The script only reads ``decision.jsonl`` and the adjacent ``run_config.json``
written by the online runner.  It produces one row per episode/run, so RSR
cannot accidentally use the number of control or decision events as its
denominator.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _distance(a: Any, b: Any) -> float:
    if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)) or len(a) < 3 or len(b) < 3:
        return 0.0
    return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(3)))


def summarize(run_jsonl: Path, label: Optional[str] = None) -> Dict[str, Any]:
    rows = _read_jsonl(run_jsonl)
    summary_rows = [row for row in rows if row.get("type") == "episode_summary"]
    summary = summary_rows[-1] if summary_rows else {}
    run_dir = run_jsonl.parent
    config: Dict[str, Any] = {}
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))

    decision_rows = [row for row in rows if row.get("type") == "decision"]
    control_rows = [row for row in rows if row.get("type") == "control"]
    successful = summary.get("success")
    if successful is None:
        successful = any(
            bool((row.get("plan") or {}).get("target_found")) and row.get("observed") is not None
            for row in decision_rows + control_rows
        )
    false_reacq = sum(
        1
        for row in decision_rows + control_rows
        if bool((row.get("plan") or {}).get("target_found")) and row.get("observed") is None
    )
    positions = [row.get("drone", {}).get("position") for row in control_rows]
    path_length = sum(_distance(a, b) for a, b in zip(positions, positions[1:]))
    latencies = [float(row.get("latency_ms")) for row in decision_rows if row.get("latency_ms") is not None]
    timestamps = [float(row.get("timestamp")) for row in rows if row.get("timestamp") is not None]
    strategy = label or config.get("strategy") or (decision_rows[-1].get("plan", {}).get("source") if decision_rows else "unknown")
    return {
        "run_dir": str(run_dir),
        "strategy": strategy,
        "success": bool(successful),
        "rsr_numerator": int(bool(successful)),
        "decision_count": int(summary.get("decision_count", len(decision_rows))),
        "control_count": int(summary.get("control_count", len(control_rows))),
        "time_to_reacquisition_s": summary.get("time_to_reacquisition_s"),
        "episode_duration_s": summary.get("episode_duration_s") or (max(timestamps) - min(timestamps) if len(timestamps) > 1 else None),
        "path_length_m": summary.get("path_length_m") or path_length,
        "collision": bool(summary.get("collision", False)),
        "false_reacquisition_count": false_reacq,
        "remote_error_count": int(summary.get("remote_error_count", 0)),
        "mean_latency_ms": sum(latencies) / len(latencies) if latencies else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate RecoverVLA raw episode logs")
    parser.add_argument("--input", required=True, help="A decision_runs directory or one run directory")
    parser.add_argument("--output", required=True, help="Output directory for summary.csv/summary.json")
    parser.add_argument("--label", default="", help="Optional strategy label, e.g. E1 or E2")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    jsonl_paths = [input_path] if input_path.name == "decision.jsonl" else sorted(input_path.rglob("decision.jsonl"))
    if not jsonl_paths:
        raise SystemExit(f"No decision.jsonl found under {input_path}")
    rows = [summarize(path, args.label or None) for path in jsonl_paths]
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "episodes": len(rows),
        "successful_episodes": sum(int(row["success"]) for row in rows),
        "rsr": sum(int(row["success"]) for row in rows) / len(rows) if rows else None,
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
