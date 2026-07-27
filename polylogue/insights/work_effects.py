"""Repository-effect observation adapters for work-evidence reconciliation.

``reconcile_work_effects`` (``work_reconciliation.py``) is a pure function: it
attaches whatever :class:`ObservedRepositoryEffect` and
:class:`ReconciliationJudgment` values a caller hands it. This module supplies
the missing production half -- adapters that read *independent* repository
evidence and a conservative matcher that turns shared identifiers into
judgments, so the reconciler is no longer fed by hand in tests only.

Three adapters are concrete and read-only:

- :class:`GitCommitEffectAdapter` reads ``git log`` for a local checkout.
- :class:`BeadsIssueEffectAdapter` reads a Beads ``interactions.jsonl``
  ledger (the same append-only shape ``sources/parsers/beads.py`` ingests).
- :class:`GitHubPullRequestEffectAdapter` shells out to the ``gh`` CLI (the
  same subprocess pattern ``insights.correlation_view._enrich_with_github_api``
  already uses) to read PR lifecycle state -- number, title, body, open/
  merged/closed state, and merge commit -- for one ``owner/name`` repo.

``gh`` access is genuinely unavailable in some lanes (no network egress,
missing binary, no ``gh auth login``). Rather than special-case those lanes,
every failure mode -- missing executable, auth/network error, a non-zero
``gh`` exit, unparseable output -- raises :class:`EffectAdapterUnavailableError`
explicitly, exactly like the other two adapters do for their own missing
inputs. ``collect_repository_effects`` already isolates one adapter's failure
from the others, so a lane with no GitHub access still gets git/Beads
effects and an honest "github: <reason>" entry in ``unavailable`` -- never a
silent empty result that would read as "no PRs found". PR
review/approval-level granularity (per-reviewer decisions, not just PR
lifecycle state) is deliberately out of scope here -- see
polylogue-1vpm.6.2 for that follow-on.

Judgment derivation (``derive_direct_identifier_judgments``) never uses time
or file-path proximity -- that heuristic already exists as a *candidate-only*
view in ``insights.session_commit``. Here, a claim is only evaluated against
an effect when they share an exact work-item id token (e.g. a Beads id such
as ``polylogue-1vpm.6.2``) in their text. A claim that names no id, or whose
id has no matching effect, is left out of the returned judgments entirely --
``reconcile_work_effects`` then leaves it unevaluated rather than guessing.
Every derived judgment is deliberately conservative and evaluates to
``"supported"``: a direct id match proves an independent effect exists for
that work item, not that its outcome semantically matches the claim's prose.
Distinguishing partial/contradicted/superseded needs field-level effect
semantics (e.g. reading a Beads ``status`` transition against what the claim
actually asserts) that is out of scope for identifier matching alone; a
richer evaluator can layer on top of these adapters without changing their
contract.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Final, Protocol

from polylogue.archive.artifact_taxonomy.support import looks_like_beads_interaction
from polylogue.core.errors import PolylogueError
from polylogue.core.refs import ObjectRef
from polylogue.insights.work_evidence import WorkEvidenceGraph
from polylogue.insights.work_reconciliation import (
    EffectAuthority,
    ObservedRepositoryEffect,
    ReconciliationJudgment,
    reconcile_work_effects,
)

# ── Identifier matching ──────────────────────────────────────────────

#: This repository's own Beads id convention (``<prefix>-<token>[.<token>]*``).
#: Callers archiving a different project's evidence pass their own pattern.
DEFAULT_WORK_ITEM_ID_PATTERN: Final = re.compile(r"\bpolylogue-[a-z0-9]+(?:\.[a-z0-9]+)*\b")


def referenced_work_item_ids(text: str, *, pattern: re.Pattern[str] = DEFAULT_WORK_ITEM_ID_PATTERN) -> frozenset[str]:
    """Return the set of work-item id tokens found verbatim in *text*."""

    return frozenset(pattern.findall(text))


# ── Errors ────────────────────────────────────────────────────────────


class EffectAdapterUnavailableError(PolylogueError):
    """Raised when an effect adapter cannot run given its current inputs.

    Covers both "the input doesn't exist" (bad repo/ledger path) and "this
    adapter isn't implemented yet" (the GitHub stub). Callers that want to
    keep collecting effects from other adapters should catch this per-adapter
    -- ``collect_repository_effects`` already does so.
    """


# ── Adapter protocol ─────────────────────────────────────────────────


class RepositoryEffectAdapter(Protocol):
    """One source of independently observed repository effects."""

    @property
    def authority(self) -> EffectAuthority:
        """The evidence authority this adapter's effects carry."""
        ...

    def collect(
        self,
        *,
        since_ms: int | None = None,
        until_ms: int | None = None,
    ) -> tuple[ObservedRepositoryEffect, ...]:
        """Return effects observed in ``[since_ms, until_ms]`` (either bound optional).

        Raises :class:`EffectAdapterUnavailableError` when the adapter cannot run
        at all (missing input, unimplemented backend) -- never returns an
        empty tuple to mean "unavailable".
        """
        ...


# ── Git commit adapter ───────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class GitCommitEffectAdapter:
    """Reads commit history from a local git checkout, read-only.

    Uses ``git log`` via subprocess against whatever ``repo_path`` the caller
    resolves (a fixture repo in tests; in production, whatever local checkout
    a session's repo identity resolves to -- this adapter does not itself
    discover that path). No writes, no network.
    """

    repo_path: Path
    ref: str = "HEAD"
    authority: EffectAuthority = "git"

    def collect(
        self,
        *,
        since_ms: int | None = None,
        until_ms: int | None = None,
    ) -> tuple[ObservedRepositoryEffect, ...]:
        head_sha = _run_git(self.repo_path, ["rev-parse", self.ref])
        if head_sha is None:
            raise EffectAdapterUnavailableError(
                f"{self.repo_path} is not a readable git repository (or {self.ref!r} does not resolve)"
            )
        snapshot_ref = ObjectRef(
            kind="context-snapshot",
            object_id=f"git:{_repo_identity(self.repo_path)}@{head_sha}",
        )
        args = ["log", self.ref, "--date=iso-strict", "--format=%H%x1f%ad%x1f%s%x1f%b%x1e"]
        if since_ms is not None:
            args += ["--since", _iso_ms(since_ms)]
        if until_ms is not None:
            args += ["--until", _iso_ms(until_ms)]
        output = _run_git(self.repo_path, args)
        if output is None:
            raise EffectAdapterUnavailableError(f"git log failed against {self.repo_path}")

        repo_id = _repo_identity(self.repo_path)
        effects: list[ObservedRepositoryEffect] = []
        for record in output.split("\x1e"):
            record = record.strip("\n")
            if not record:
                continue
            parts = record.split("\x1f")
            if len(parts) < 3:
                continue
            commit_hash, date_str, subject = parts[0], parts[1], parts[2]
            body = parts[3] if len(parts) > 3 else ""
            occurred_at_ms = _parse_iso_ms(date_str)
            label = f"{commit_hash[:12]} {subject}".strip()
            if body.strip():
                label = f"{label}\n{body.strip()}"
            effects.append(
                ObservedRepositoryEffect(
                    ref=ObjectRef(kind="commit", object_id=commit_hash),
                    label=label,
                    authority="git",
                    evidence_ref=ObjectRef(kind="artifact", object_id=f"git-commit:{repo_id}:{commit_hash}"),
                    repository_snapshot_ref=snapshot_ref,
                    occurred_at_ms=occurred_at_ms,
                )
            )
        return tuple(effects)


def _run_git(repo_path: Path, args: Sequence[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_path), *args],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip("\n")


def _repo_identity(repo_path: Path) -> str:
    return sha256(str(Path(repo_path).resolve()).encode("utf-8")).hexdigest()[:16]


# ── Beads issue-history adapter ──────────────────────────────────────


@dataclass(frozen=True, slots=True)
class BeadsIssueEffectAdapter:
    """Reads the append-only Beads interaction ledger, read-only.

    ``jsonl_path`` points at a ``.beads/interactions.jsonl``-shaped file (a
    fixture in tests; the committed ledger in production). Each valid
    interaction record becomes one effect -- this is the same record shape
    ``sources/parsers/beads.py`` already ingests as archived sessions, read
    here directly so reconciliation does not require the record to have been
    ingested first.
    """

    jsonl_path: Path
    authority: EffectAuthority = "beads"

    def collect(
        self,
        *,
        since_ms: int | None = None,
        until_ms: int | None = None,
    ) -> tuple[ObservedRepositoryEffect, ...]:
        if not self.jsonl_path.is_file():
            raise EffectAdapterUnavailableError(f"Beads interaction ledger not found: {self.jsonl_path}")
        snapshot_ref = ObjectRef(
            kind="context-snapshot",
            object_id=f"beads:{_file_identity(self.jsonl_path)}",
        )
        effects: list[ObservedRepositoryEffect] = []
        for line in self.jsonl_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not looks_like_beads_interaction(record):
                continue
            occurred_at_ms = _parse_iso_ms(str(record["created_at"]))
            if since_ms is not None and (occurred_at_ms is None or occurred_at_ms < since_ms):
                continue
            if until_ms is not None and (occurred_at_ms is None or occurred_at_ms > until_ms):
                continue
            issue_id = str(record["issue_id"])
            interaction_id = str(record["id"])
            effects.append(
                ObservedRepositoryEffect(
                    ref=ObjectRef(kind="beads-issue", object_id=f"{issue_id}:{interaction_id}"),
                    label=_beads_interaction_label(record),
                    authority="beads",
                    evidence_ref=ObjectRef(
                        kind="artifact",
                        object_id=f"beads-interaction:{issue_id}:{interaction_id}",
                    ),
                    repository_snapshot_ref=snapshot_ref,
                    occurred_at_ms=occurred_at_ms,
                )
            )
        return tuple(effects)


def _beads_interaction_label(record: dict[str, object]) -> str:
    issue_id = str(record["issue_id"])
    kind = str(record["kind"])
    extra = record.get("extra")
    field = extra.get("field") if isinstance(extra, dict) else None
    if isinstance(field, str):
        old_value = extra.get("old_value") if isinstance(extra, dict) else None
        new_value = extra.get("new_value") if isinstance(extra, dict) else None
        return f"{issue_id} {field}: {old_value!r} -> {new_value!r}"
    return f"{issue_id} {kind.replace('_', ' ')}"


def _file_identity(path: Path) -> str:
    return sha256(str(Path(path).resolve()).encode("utf-8")).hexdigest()[:16]


# ── GitHub PR adapter ─────────────────────────────────────────────────

#: `gh pr list --json` fields this adapter needs. Kept as an explicit tuple
#: (not `--json '*'`) so a `gh` version that drops/renames a field breaks
#: loudly at the `_run_gh` non-zero-exit / JSON-decode boundary instead of
#: silently starving one label of context.
_PR_JSON_FIELDS: Final = (
    "number",
    "title",
    "body",
    "state",
    "url",
    "createdAt",
    "updatedAt",
    "closedAt",
    "mergedAt",
    "mergeCommit",
)


@dataclass(frozen=True, slots=True)
class GitHubPullRequestEffectAdapter:
    """Reads pull-request lifecycle effects from GitHub via the ``gh`` CLI.

    Read-only: a single ``gh pr list --repo <repo> --state all`` call per
    ``collect()``, the same subprocess-shelling pattern already used by
    ``insights.correlation_view._enrich_with_github_api`` elsewhere in this
    repo. Every PR ``gh`` returns (open, merged, or closed -- ``--state all``)
    becomes one effect carrying its number, title, body, lifecycle state, and
    merge commit, so identifier matching against a PR's title/body works the
    same way it already does for git commit subjects and Beads interactions.

    ``limit`` bounds one ``gh`` call's page size (``gh`` itself paginates
    beyond it only if asked); ``since_ms``/``until_ms`` are applied
    client-side against each PR's most recent known lifecycle timestamp
    (merged, else closed, else updated, else created) after the page is
    fetched -- ``gh pr list`` has no native time-range filter for arbitrary
    bounds without changing the query shape into GitHub's search syntax.
    """

    repo: str  # "owner/name"
    authority: EffectAuthority = "github"
    limit: int = 200
    gh_path: str = "gh"
    timeout_s: int = 45

    def collect(
        self,
        *,
        since_ms: int | None = None,
        until_ms: int | None = None,
    ) -> tuple[ObservedRepositoryEffect, ...]:
        output = _run_gh(
            self.gh_path,
            [
                "pr",
                "list",
                "--repo",
                self.repo,
                "--state",
                "all",
                "--limit",
                str(self.limit),
                "--json",
                ",".join(_PR_JSON_FIELDS),
            ],
            timeout_s=self.timeout_s,
        )
        try:
            records = json.loads(output)
        except json.JSONDecodeError as exc:
            raise EffectAdapterUnavailableError(
                f"gh pr list for {self.repo!r} returned output that is not valid JSON"
            ) from exc
        if not isinstance(records, list):
            raise EffectAdapterUnavailableError(
                f"gh pr list for {self.repo!r} returned an unexpected JSON shape ({type(records).__name__})"
            )

        snapshot_ref = ObjectRef(
            kind="context-snapshot",
            object_id=f"github:{self.repo}@{sha256(output.encode('utf-8')).hexdigest()[:16]}",
        )
        effects: list[ObservedRepositoryEffect] = []
        for record in records:
            if not isinstance(record, dict) or record.get("number") is None:
                continue
            number = record["number"]
            occurred_at_ms = _first_present_ms(record, ("mergedAt", "closedAt", "updatedAt", "createdAt"))
            if since_ms is not None and (occurred_at_ms is None or occurred_at_ms < since_ms):
                continue
            if until_ms is not None and (occurred_at_ms is None or occurred_at_ms > until_ms):
                continue
            effects.append(
                ObservedRepositoryEffect(
                    ref=ObjectRef(kind="github-pr", object_id=f"{self.repo}#{number}"),
                    label=_pull_request_label(self.repo, record),
                    authority="github",
                    evidence_ref=ObjectRef(kind="artifact", object_id=f"github-pr:{self.repo}:{number}"),
                    repository_snapshot_ref=snapshot_ref,
                    occurred_at_ms=occurred_at_ms,
                )
            )
        return tuple(effects)


def _run_gh(gh_path: str, args: Sequence[str], *, timeout_s: int) -> str:
    """Run one ``gh`` subcommand and return its captured stdout.

    Every failure mode raises :class:`EffectAdapterUnavailableError` with a
    reason a caller can log or surface verbatim -- missing binary, auth/
    network failure (non-zero exit with ``gh``'s own stderr), and timeout are
    all distinguishable from "gh ran and found nothing".
    """
    command = [gh_path, *args]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError as exc:
        raise EffectAdapterUnavailableError(
            f"{gh_path!r} executable not found on PATH -- install the GitHub CLI to collect PR effects"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise EffectAdapterUnavailableError(f"{' '.join(command)} timed out after {timeout_s}s") from exc
    except OSError as exc:
        raise EffectAdapterUnavailableError(f"{' '.join(command)} failed to start: {exc}") from exc
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or "(no stderr output)"
        raise EffectAdapterUnavailableError(f"{' '.join(command)} exited {completed.returncode}: {stderr}")
    return completed.stdout


def _first_present_ms(record: dict[str, object], fields: Sequence[str]) -> int | None:
    for field in fields:
        value = record.get(field)
        if isinstance(value, str) and value:
            parsed = _parse_iso_ms(value)
            if parsed is not None:
                return parsed
    return None


def _pull_request_label(repo: str, record: dict[str, object]) -> str:
    number = record.get("number")
    title = str(record.get("title") or "").strip()
    state = str(record.get("state") or "").upper()
    label = f"{repo}#{number} [{state}] {title}".strip()
    merge_commit = record.get("mergeCommit")
    if isinstance(merge_commit, dict):
        merge_sha = merge_commit.get("oid")
        if isinstance(merge_sha, str) and merge_sha:
            label = f"{label} (merge {merge_sha[:12]})"
    body = str(record.get("body") or "").strip()
    if body:
        label = f"{label}\n{body}"
    return label


# ── Collection + judgment derivation ─────────────────────────────────


@dataclass(frozen=True, slots=True)
class EffectAdapterFailure:
    """One adapter's explicit, recorded unavailability."""

    authority: EffectAuthority
    reason: str


@dataclass(frozen=True, slots=True)
class EffectCollectionResult:
    """Effects gathered plus any adapters that failed explicitly."""

    effects: tuple[ObservedRepositoryEffect, ...]
    unavailable: tuple[EffectAdapterFailure, ...]


def collect_repository_effects(
    adapters: Sequence[RepositoryEffectAdapter],
    *,
    since_ms: int | None = None,
    until_ms: int | None = None,
) -> EffectCollectionResult:
    """Run every adapter, keeping one adapter's failure from losing the rest."""

    effects: list[ObservedRepositoryEffect] = []
    failures: list[EffectAdapterFailure] = []
    for adapter in adapters:
        try:
            effects.extend(adapter.collect(since_ms=since_ms, until_ms=until_ms))
        except EffectAdapterUnavailableError as exc:
            failures.append(EffectAdapterFailure(authority=adapter.authority, reason=str(exc)))
    return EffectCollectionResult(effects=tuple(effects), unavailable=tuple(failures))


def derive_direct_identifier_judgments(
    graph: WorkEvidenceGraph,
    effects: Iterable[ObservedRepositoryEffect],
    *,
    pattern: re.Pattern[str] = DEFAULT_WORK_ITEM_ID_PATTERN,
) -> tuple[ReconciliationJudgment, ...]:
    """Match claim nodes to effects via an explicit shared identifier.

    Conservative by construction: a claim is only linked to an effect when
    their texts share an exact work-item id token. Session presence, time
    proximity, and file overlap are never treated as a match here -- a claim
    naming no id, or whose id has no matching effect, is simply absent from
    the result and stays unevaluated when ``reconcile_work_effects`` applies
    these judgments.
    """

    effects = tuple(effects)
    judgments: list[ReconciliationJudgment] = []
    for node in graph.nodes:
        if node.kind != "claim" or not node.claim_text:
            continue
        claim_ids = referenced_work_item_ids(node.claim_text, pattern=pattern)
        if not claim_ids:
            continue
        for effect in effects:
            effect_ids = referenced_work_item_ids(effect.label, pattern=pattern)
            if not (claim_ids & effect_ids):
                continue
            judgments.append(
                ReconciliationJudgment(
                    claim_ref=node.ref,
                    effect_ref=effect.ref,
                    evaluation="supported",
                    evidence_ref=effect.evidence_ref,
                )
            )
    return tuple(judgments)


def reconcile_repository_effects(
    graph: WorkEvidenceGraph,
    *,
    adapters: Sequence[RepositoryEffectAdapter],
    since_ms: int | None = None,
    until_ms: int | None = None,
    id_pattern: re.Pattern[str] = DEFAULT_WORK_ITEM_ID_PATTERN,
) -> tuple[WorkEvidenceGraph, EffectCollectionResult]:
    """Collect effects, derive direct-identifier judgments, and reconcile.

    Pure with respect to storage -- callers own persisting the returned
    graph (see ``polylogue.operations.work_effect_reconciliation`` for the
    production read-modify-write wrapper against the archive).
    """

    collection = collect_repository_effects(adapters, since_ms=since_ms, until_ms=until_ms)
    judgments = derive_direct_identifier_judgments(graph, collection.effects, pattern=id_pattern)
    reconciled = reconcile_work_effects(graph, effects=collection.effects, judgments=judgments)
    return reconciled, collection


def _iso_ms(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def _parse_iso_ms(value: str) -> int | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


__all__ = [
    "DEFAULT_WORK_ITEM_ID_PATTERN",
    "BeadsIssueEffectAdapter",
    "EffectAdapterFailure",
    "EffectAdapterUnavailableError",
    "EffectCollectionResult",
    "GitCommitEffectAdapter",
    "GitHubPullRequestEffectAdapter",
    "RepositoryEffectAdapter",
    "collect_repository_effects",
    "derive_direct_identifier_judgments",
    "reconcile_repository_effects",
    "referenced_work_item_ids",
]
