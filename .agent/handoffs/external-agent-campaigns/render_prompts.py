#!/usr/bin/env python3
"""Render complete external-agent prompts from campaign briefs and contracts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def render_workload(workload: Path, *, check: bool) -> list[str]:
    manifest_path = workload / "campaign.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract_path = workload / manifest["prompt_contract"]
    contract = contract_path.read_text(encoding="utf-8").strip()
    errors: list[str] = []

    job_ids: set[str] = set()
    result_prefixes: set[str] = set()
    for job in manifest["jobs"]:
        job_id = job["id"]
        if job_id in job_ids:
            errors.append(f"{manifest_path}: duplicate job id {job_id}")
        job_ids.add(job_id)
        result_prefix = job["result_prefix"]
        if result_prefix in result_prefixes:
            errors.append(f"{manifest_path}: duplicate result prefix {result_prefix}")
        result_prefixes.add(result_prefix)
        brief_path = workload / job["brief"]
        prompt_path = workload / job["prompt"]
        brief = brief_path.read_text(encoding="utf-8").strip()
        if not brief.startswith('Title: "'):
            errors.append(f"{brief_path}: first line must be a literal Title")
            continue
        expected_zip = f"Result ZIP: `{result_prefix}-r01.zip`"
        if expected_zip not in brief:
            errors.append(f"{brief_path}: expected {expected_zip}")
            continue
        rendered = f"{brief}\n\n{contract}\n"
        if check:
            if not prompt_path.exists():
                errors.append(f"{prompt_path}: missing rendered prompt")
            elif prompt_path.read_text(encoding="utf-8") != rendered:
                errors.append(f"{prompt_path}: rendered prompt is stale")
        else:
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_path.write_text(rendered, encoding="utf-8")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("workloads", nargs="+", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    errors: list[str] = []
    for workload in args.workloads:
        errors.extend(render_workload(workload, check=args.check))
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
