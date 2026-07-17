#!/usr/bin/env python3
"""Lease dependency-ready missions and scarce resources to a single-machine agent swarm."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "control" / "mission-plan.yaml"
DEFAULT_STATE_DIR = ROOT / ".legibility-swarm"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_plan(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data


REQUIRED_HANDOFF_HEADINGS = (
    "mission",
    "base commit",
    "changed files",
    "verification",
    "known failures",
    "merge recommendation",
)


def parse_timestamp(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def lease_expired(row: dict[str, Any]) -> bool:
    expires = parse_timestamp(row.get("lease_expires_at"))
    return expires is not None and expires <= datetime.now(timezone.utc)


def validate_handoff(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8").lower()
    missing: list[str] = []
    for heading in REQUIRED_HANDOFF_HEADINGS:
        if not re.search(rf"^#+\s+{re.escape(heading)}\s*$", text, flags=re.MULTILINE):
            missing.append(heading)
    return missing


def default_state(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "polylogue-sinex-swarm-state/v1",
        "created_at": now(),
        "updated_at": now(),
        "missions": {
            mission["id"]: {
                "status": "pending",
                "agent": None,
                "worktree": None,
                "claimed_at": None,
                "heartbeat_at": None,
                "lease_expires_at": None,
                "finished_at": None,
                "handoff": None,
                "result": None,
                "notes": [],
            }
            for mission in plan["missions"]
        },
        "events": [],
    }


@contextmanager
def locked_state(state_dir: Path, plan: dict[str, Any]) -> Iterator[tuple[dict[str, Any], Path]]:
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = state_dir / "lock"
    state_path = state_dir / "state.json"
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else default_state(plan)
        yield state, state_path
        state["updated_at"] = now()
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, state_path)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def mission_map(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {m["id"]: m for m in plan["missions"]}


def root_key(repo: str, raw: str) -> str:
    if raw == "*":
        return f"{repo}:*"
    if ":" in raw and raw.split(":", 1)[0] in {"polylogue", "sinex"}:
        explicit_repo, path = raw.split(":", 1)
        return f"{explicit_repo}:{path.rstrip('/')}"
    return f"{repo}:{raw.rstrip('/')}"


def path_conflict(a: str, b: str) -> bool:
    repo_a, path_a = a.split(":", 1)
    repo_b, path_b = b.split(":", 1)
    if repo_a == "joint" or repo_b == "joint":
        # Joint only conflicts when explicit paths share a concrete repo prefix.
        pass
    if repo_a != repo_b and repo_a not in {"joint"} and repo_b not in {"joint"}:
        return False
    if path_a == "*" or path_b == "*":
        return True
    return path_a == path_b or path_a.startswith(path_b + "/") or path_b.startswith(path_a + "/")


def active_missions(plan: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
    by_id = mission_map(plan)
    return [by_id[mid] for mid, row in state["missions"].items() if row["status"] == "active"]


def resource_usage(plan: dict[str, Any], state: dict[str, Any]) -> dict[str, int]:
    usage: dict[str, int] = {}
    for mission in active_missions(plan, state):
        for resource, amount in mission.get("resources", {}).items():
            usage[resource] = usage.get(resource, 0) + amount
    return usage


def blockers(mission: dict[str, Any], plan: dict[str, Any], state: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    row = state["missions"][mission["id"]]
    if row["status"] != "pending":
        reasons.append(f"status={row['status']}")
        return reasons
    for dep in mission.get("depends_on", []):
        dep_status = state["missions"][dep]["status"]
        if dep_status != "complete":
            reasons.append(f"dependency {dep} is {dep_status}")
    usage = resource_usage(plan, state)
    for resource, amount in mission.get("resources", {}).items():
        capacity = plan["resource_capacities"][resource]["capacity"]
        if usage.get(resource, 0) + amount > capacity:
            reasons.append(f"resource {resource} would exceed {capacity}")
    target_roots = [root_key(mission["repo"], p) for p in mission.get("write_paths", [])]
    for active in active_missions(plan, state):
        active_roots = [root_key(active["repo"], p) for p in active.get("write_paths", [])]
        for target in target_roots:
            for occupied in active_roots:
                if path_conflict(target, occupied):
                    reasons.append(f"write path {target} conflicts with active {active['id']}:{occupied}")
                    break
            if reasons and reasons[-1].startswith("write path"):
                break
    return reasons


def append_event(state: dict[str, Any], event: str, mission: str, agent: str | None = None, **extra: Any) -> None:
    row = {"at": now(), "event": event, "mission": mission, "agent": agent}
    row.update(extra)
    state["events"].append(row)


def cmd_init(args: argparse.Namespace, plan: dict[str, Any]) -> int:
    with locked_state(args.state_dir, plan) as (state, _):
        if args.reset:
            state.clear()
            state.update(default_state(plan))
        print(f"initialized {len(state['missions'])} missions at {args.state_dir}")
    return 0


def cmd_status(args: argparse.Namespace, plan: dict[str, Any]) -> int:
    with locked_state(args.state_dir, plan) as (state, _):
        by_id = mission_map(plan)
        for horizon in ("72h", "7d", "30d"):
            print(f"\n[{horizon}]")
            for mid, row in state["missions"].items():
                m = by_id[mid]
                if m["horizon"] != horizon:
                    continue
                suffix = f" agent={row['agent']}" if row["agent"] else ""
                if row["status"] == "active" and lease_expired(row):
                    suffix += " LEASE-EXPIRED"
                print(f"{mid:10} {row['status']:9} {m['title']}{suffix}")
    return 0


def cmd_ready(args: argparse.Namespace, plan: dict[str, Any]) -> int:
    with locked_state(args.state_dir, plan) as (state, _):
        candidates = []
        for mission in plan["missions"]:
            reasons = blockers(mission, plan, state)
            if not reasons and (args.horizon is None or mission["horizon"] == args.horizon):
                candidates.append(mission)
        if args.json:
            print(json.dumps(candidates, indent=2))
        else:
            for mission in candidates:
                print(f"{mission['id']}\t{mission['horizon']}\t{mission['title']}\t{mission['prompt_file']}")
        return 0 if candidates else 2


def cmd_explain(args: argparse.Namespace, plan: dict[str, Any]) -> int:
    with locked_state(args.state_dir, plan) as (state, _):
        mission = mission_map(plan).get(args.mission)
        if mission is None:
            print(f"unknown mission {args.mission}", file=sys.stderr)
            return 1
        reasons = blockers(mission, plan, state)
        print(json.dumps({"mission": mission, "state": state["missions"][args.mission], "blockers": reasons}, indent=2))
        return 0 if not reasons else 2


def cmd_claim(args: argparse.Namespace, plan: dict[str, Any]) -> int:
    if args.lease_hours <= 0:
        print("--lease-hours must be positive", file=sys.stderr)
        return 2
    with locked_state(args.state_dir, plan) as (state, _):
        by_id = mission_map(plan)
        mission = by_id.get(args.mission)
        if mission is None:
            print(f"unknown mission {args.mission}", file=sys.stderr)
            return 1
        reasons = blockers(mission, plan, state)
        if reasons:
            print("cannot claim mission:", file=sys.stderr)
            for reason in reasons:
                print(f"- {reason}", file=sys.stderr)
            return 2
        row = state["missions"][args.mission]
        claimed_at = datetime.now(timezone.utc)
        expires_at = claimed_at + timedelta(hours=args.lease_hours)
        row.update(
            {
                "status": "active",
                "agent": args.agent,
                "worktree": args.worktree,
                "claimed_at": claimed_at.isoformat(),
                "heartbeat_at": claimed_at.isoformat(),
                "lease_expires_at": expires_at.isoformat(),
            }
        )
        append_event(
            state,
            "claimed",
            args.mission,
            args.agent,
            worktree=args.worktree,
            lease_hours=args.lease_hours,
            lease_expires_at=expires_at.isoformat(),
        )
        print(f"claimed {args.mission} for {args.agent}")
    return 0


def cmd_finish(args: argparse.Namespace, plan: dict[str, Any]) -> int:
    with locked_state(args.state_dir, plan) as (state, _):
        if args.mission not in state["missions"]:
            print(f"unknown mission {args.mission}", file=sys.stderr)
            return 1
        row = state["missions"][args.mission]
        if row["status"] != "active":
            print(f"mission {args.mission} is {row['status']}, not active", file=sys.stderr)
            return 2
        if row["agent"] != args.agent and not args.force:
            print(f"mission is leased to {row['agent']}; use --force only as coordinator", file=sys.stderr)
            return 2
        handoff = Path(args.handoff).resolve()
        if not handoff.is_file():
            print(f"handoff file not found: {handoff}", file=sys.stderr)
            return 2
        if mission_map(plan)[args.mission].get("handoff_required", False):
            missing = validate_handoff(handoff)
            if missing:
                print(
                    "handoff is missing required Markdown headings: " + ", ".join(missing),
                    file=sys.stderr,
                )
                return 2
        row.update(
            {
                "status": "complete" if args.result == "complete" else "failed",
                "finished_at": now(),
                "handoff": str(handoff),
                "result": args.result,
                "lease_expires_at": None,
            }
        )
        append_event(state, "finished", args.mission, args.agent, handoff=str(handoff), result=args.result)
        print(f"finished {args.mission} as {row['status']}")
    return 0


def cmd_release(args: argparse.Namespace, plan: dict[str, Any]) -> int:
    with locked_state(args.state_dir, plan) as (state, _):
        if args.mission not in state["missions"]:
            return 1
        row = state["missions"][args.mission]
        if row["status"] != "active":
            print(f"mission {args.mission} is not active", file=sys.stderr)
            return 2
        if row["agent"] != args.agent and not args.force:
            print(f"mission is leased to {row['agent']}", file=sys.stderr)
            return 2
        row.update(
            {
                "status": "pending",
                "agent": None,
                "worktree": None,
                "claimed_at": None,
                "heartbeat_at": None,
                "lease_expires_at": None,
            }
        )
        append_event(state, "released", args.mission, args.agent, reason=args.reason)
        print(f"released {args.mission}")
    return 0


def cmd_heartbeat(args: argparse.Namespace, plan: dict[str, Any]) -> int:
    if args.extend_hours <= 0:
        print("--extend-hours must be positive", file=sys.stderr)
        return 2
    with locked_state(args.state_dir, plan) as (state, _):
        if args.mission not in state["missions"]:
            print(f"unknown mission {args.mission}", file=sys.stderr)
            return 1
        row = state["missions"][args.mission]
        if row["status"] != "active":
            print(f"mission {args.mission} is {row['status']}, not active", file=sys.stderr)
            return 2
        if row["agent"] != args.agent and not args.force:
            print(f"mission is leased to {row['agent']}", file=sys.stderr)
            return 2
        beat = datetime.now(timezone.utc)
        expires_at = beat + timedelta(hours=args.extend_hours)
        row["heartbeat_at"] = beat.isoformat()
        row["lease_expires_at"] = expires_at.isoformat()
        if args.note:
            row["notes"].append({"at": beat.isoformat(), "agent": args.agent, "text": args.note})
        append_event(
            state,
            "heartbeat",
            args.mission,
            args.agent,
            lease_expires_at=expires_at.isoformat(),
            note=args.note,
        )
        print(f"heartbeat {args.mission}; lease extends to {expires_at.isoformat()}")
    return 0


def cmd_reap_stale(args: argparse.Namespace, plan: dict[str, Any]) -> int:
    reaped: list[str] = []
    with locked_state(args.state_dir, plan) as (state, _):
        for mid, row in state["missions"].items():
            if row["status"] != "active" or not lease_expired(row):
                continue
            reaped.append(mid)
            if args.dry_run:
                continue
            prior_agent = row["agent"]
            row.update(
                {
                    "status": "pending",
                    "agent": None,
                    "worktree": None,
                    "claimed_at": None,
                    "heartbeat_at": None,
                    "lease_expires_at": None,
                }
            )
            append_event(state, "reaped", mid, prior_agent, reason=args.reason)
        if args.json:
            print(json.dumps({"reaped": reaped, "dry_run": args.dry_run}, indent=2))
        else:
            action = "would reap" if args.dry_run else "reaped"
            print(f"{action} {len(reaped)} mission(s): {', '.join(reaped) if reaped else 'none'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--reset", action="store_true")
    sub.add_parser("status")
    ready = sub.add_parser("ready")
    ready.add_argument("--horizon", choices=["72h", "7d", "30d"])
    ready.add_argument("--json", action="store_true")
    explain = sub.add_parser("explain")
    explain.add_argument("mission")
    claim = sub.add_parser("claim")
    claim.add_argument("mission")
    claim.add_argument("--agent", required=True)
    claim.add_argument("--worktree", required=True)
    claim.add_argument("--lease-hours", type=float, default=12.0)
    finish = sub.add_parser("finish")
    finish.add_argument("mission")
    finish.add_argument("--agent", required=True)
    finish.add_argument("--handoff", required=True)
    finish.add_argument("--result", choices=["complete", "failed"], default="complete")
    finish.add_argument("--force", action="store_true")
    release = sub.add_parser("release")
    release.add_argument("mission")
    release.add_argument("--agent", required=True)
    release.add_argument("--reason", required=True)
    release.add_argument("--force", action="store_true")
    heartbeat = sub.add_parser("heartbeat")
    heartbeat.add_argument("mission")
    heartbeat.add_argument("--agent", required=True)
    heartbeat.add_argument("--extend-hours", type=float, default=12.0)
    heartbeat.add_argument("--note")
    heartbeat.add_argument("--force", action="store_true")
    reap = sub.add_parser("reap-stale")
    reap.add_argument("--reason", default="lease expired without heartbeat")
    reap.add_argument("--dry-run", action="store_true")
    reap.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.plan = args.plan.resolve()
    args.state_dir = args.state_dir.resolve()
    plan = load_plan(args.plan)
    commands = {
        "init": cmd_init,
        "status": cmd_status,
        "ready": cmd_ready,
        "explain": cmd_explain,
        "claim": cmd_claim,
        "finish": cmd_finish,
        "release": cmd_release,
        "heartbeat": cmd_heartbeat,
        "reap-stale": cmd_reap_stale,
    }
    return commands[args.command](args, plan)


if __name__ == "__main__":
    raise SystemExit(main())
