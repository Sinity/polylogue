"""Declared on-disk layout of every canonical watch source.

A provider writes its material at exact positions below its root: Claude Code
puts a session transcript at ``<project>/<session>.jsonl`` and its subagents at
``<project>/<session>/subagents/agent-<id>.jsonl``, never three directories
deeper inside some copy of the tree. Discovery used to walk the whole root and
admit anything an unanchored artifact-rule regex or a suffix matched, so a
full copy of a provider tree nested under its own root (an agent git worktree
of ``~/.claude/projects``) was admitted file for file as if it were real.

A :class:`SourceLayout` states, relative to the source root, the depth and
position of every artifact kind the source holds. One path segment is one
pattern, matched in full. :data:`ANY_DEPTH` stands for zero or more
directories and is used only where the provider itself nests freely (memory
documents, the operator's inbox drops). Discovery descends only into a
directory some entry can still reach and admits only a file some entry
matches end to end; anything else under the root is an excluded entry,
reported and never walked or parsed.

An entry's ``kind`` is the owning ``OriginSpec`` artifact-rule kind when the
provider declares one for that family, so the layout anchors the rule: the
rule keeps classifying retained paths, and :func:`layout_declaration_defects`
checks that each entry's example path resolves to the same rule. Families
admitted by suffix alone (Codex rollouts, Hermes databases) have no rule and
carry a layout-only kind.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

from polylogue.core.enums import Provider

#: Zero or more directories. Never matches a hidden (dot-prefixed) directory:
#: a provider never nests its material under one, while ``.git`` and agent
#: worktree copies (``.claude/worktrees/...``) live exactly there.
ANY_DEPTH = "**"

#: One visible path segment of any name.
NAME = r"(?!\.)[^/]+"
_DATE = r"\d{4}-\d{2}-\d{2}"


@functools.lru_cache(maxsize=512)
def _segment(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern)


def _visible(part: str) -> bool:
    return not part.startswith(".")


def _full_match(pattern: tuple[str, ...], parts: tuple[str, ...]) -> bool:
    if not pattern:
        return not parts
    head = pattern[0]
    if head == ANY_DEPTH:
        if _full_match(pattern[1:], parts):
            return True
        return len(parts) > 1 and _visible(parts[0]) and _full_match(pattern, parts[1:])
    if not parts or _segment(head).fullmatch(parts[0]) is None:
        return False
    return _full_match(pattern[1:], parts[1:])


def _reaches(pattern: tuple[str, ...], parts: tuple[str, ...]) -> bool:
    """Whether a directory at ``parts`` can still contain a file ``pattern`` admits."""

    if not parts:
        # A pattern never ends in ANY_DEPTH, so a non-empty remainder always
        # holds the file segment still to come.
        return bool(pattern)
    if not pattern:
        return False
    head = pattern[0]
    if head == ANY_DEPTH:
        if _reaches(pattern[1:], parts):
            return True
        return _visible(parts[0]) and _reaches(pattern, parts[1:])
    if _segment(head).fullmatch(parts[0]) is None:
        return False
    return _reaches(pattern[1:], parts[1:])


@dataclass(frozen=True, slots=True)
class LayoutEntry:
    """One artifact kind at one declared position below a source root."""

    kind: str
    segments: tuple[str, ...]
    #: A synthetic root-relative path this entry admits. It documents the
    #: position and lets tests pin the entry against the owning artifact rule.
    example: str

    def __post_init__(self) -> None:
        if not self.segments or self.segments[-1] == ANY_DEPTH:
            raise ValueError(f"layout entry {self.kind!r} must end in a file segment")
        for first, second in zip(self.segments, self.segments[1:], strict=False):
            if first == second == ANY_DEPTH:
                raise ValueError(f"layout entry {self.kind!r} repeats {ANY_DEPTH}")
        for segment in self.segments:
            if segment != ANY_DEPTH:
                _segment(segment)

    def admits(self, parts: tuple[str, ...]) -> bool:
        return _full_match(self.segments, parts)

    def reaches(self, parts: tuple[str, ...]) -> bool:
        return _reaches(self.segments, parts)


@dataclass(frozen=True, slots=True)
class SourceLayout:
    """Every artifact position below one watch source's root."""

    #: The provider whose ``OriginSpec`` artifact rules the entry kinds name,
    #: or ``None`` for a Polylogue-owned spool or the operator's inbox.
    provider: Provider | None
    entries: tuple[LayoutEntry, ...]

    def artifact_kind(self, relative: Sequence[str]) -> str | None:
        """The kind declared at a root-relative file path, or ``None`` outside the layout."""

        parts = tuple(relative)
        for entry in self.entries:
            if entry.admits(parts):
                return entry.kind
        return None

    def admits_directory(self, relative: Sequence[str]) -> bool:
        """Whether discovery may descend into a root-relative directory."""

        parts = tuple(relative)
        return any(entry.reaches(parts) for entry in self.entries)

    def identity(self) -> tuple[object, ...]:
        """A stable value naming this declaration, for baseline signatures."""

        return (
            None if self.provider is None else self.provider.value,
            tuple((entry.kind, entry.segments) for entry in self.entries),
        )


def _claude_code_projects_layout() -> SourceLayout:
    # Claude Code names a project directory after its working directory with
    # every separator replaced by ``-``, so the name always begins with ``-``.
    project = r"-[^/]*"
    # A session directory is named by its session id; any visible name is
    # accepted at that depth, since only the subdirectories below it
    # (``subagents``, ``tool-results``, ``workflows``) carry artifacts.
    session = NAME
    return SourceLayout(
        Provider.CLAUDE_CODE,
        (
            LayoutEntry(
                "coordinator_session_stream",
                (project, r"[^/]+\.(?:jsonl|ndjson)"),
                "-home-user-repo/00000000-0000-4000-8000-000000000001.jsonl",
            ),
            LayoutEntry("session_index", (project, r"sessions-index\.json"), "-home-user-repo/sessions-index.json"),
            LayoutEntry(
                "agent_memory_document",
                (project, "memory", ANY_DEPTH, r"[^/]+\.md"),
                "-home-user-repo/memory/archive/note.md",
            ),
            LayoutEntry(
                "agent_transcript",
                (project, session, "subagents", r"agent-[^/]+\.(?:jsonl|ndjson)"),
                "-home-user-repo/00000000-0000-4000-8000-000000000001/subagents/agent-a1.jsonl",
            ),
            LayoutEntry(
                "agent_sidecar_meta",
                (project, session, "subagents", r"agent-[^/]+\.meta\.json"),
                "-home-user-repo/00000000-0000-4000-8000-000000000001/subagents/agent-a1.meta.json",
            ),
            LayoutEntry(
                "agent_transcript",
                (project, session, "subagents", "workflows", NAME, r"agent-[^/]+\.(?:jsonl|ndjson)"),
                "-home-user-repo/00000000-0000-4000-8000-000000000001/subagents/workflows/wf_1/agent-a1.jsonl",
            ),
            LayoutEntry(
                "agent_sidecar_meta",
                (project, session, "subagents", "workflows", NAME, r"agent-[^/]+\.meta\.json"),
                "-home-user-repo/00000000-0000-4000-8000-000000000001/subagents/workflows/wf_1/agent-a1.meta.json",
            ),
            LayoutEntry(
                "workflow_journal",
                (project, session, "subagents", "workflows", NAME, r"journal\.jsonl"),
                "-home-user-repo/00000000-0000-4000-8000-000000000001/subagents/workflows/wf_1/journal.jsonl",
            ),
            LayoutEntry(
                "workflow_run_snapshot",
                (project, session, "workflows", r"[^/]+\.json"),
                "-home-user-repo/00000000-0000-4000-8000-000000000001/workflows/wf_1.json",
            ),
            LayoutEntry(
                "tool_result_sidecar",
                (project, session, "tool-results", r"(?!hook-)(?!\.)[^/]+"),
                "-home-user-repo/00000000-0000-4000-8000-000000000001/tool-results/toolu_1.txt",
            ),
        ),
    )


def _codex_state_layout() -> SourceLayout:
    from polylogue.sources.origin_specs import database_capability_for_provider

    capability = database_capability_for_provider(Provider.CODEX)
    members = () if capability is None else tuple(member.filename for member in capability.members)
    # Codex writes its development database one directory down; every other
    # declared member sits at the install root.
    nested = {"codex-dev.db": ("sqlite",)}
    entries = [
        LayoutEntry(
            "database_member",
            (*nested.get(name, ()), re.escape(name)),
            "/".join((*nested.get(name, ()), name)),
        )
        for name in members
    ]
    entries.extend(
        (
            LayoutEntry("session_index", (r"session_index\.jsonl",), "session_index.jsonl"),
            LayoutEntry("prompt_history_log", (r"history\.jsonl",), "history.jsonl"),
        )
    )
    return SourceLayout(Provider.CODEX, tuple(entries))


def _hermes_layout() -> SourceLayout:
    from polylogue.sources.origin_specs import database_capability_for_provider

    capability = database_capability_for_provider(Provider.HERMES)
    members = () if capability is None else tuple(member.filename for member in capability.members)

    def home(prefix: tuple[str, ...], example_prefix: str) -> list[LayoutEntry]:
        rows = [
            LayoutEntry("database_member", (*prefix, re.escape(name)), f"{example_prefix}{name}") for name in members
        ]
        rows.extend(
            (
                LayoutEntry(
                    "session_snapshot",
                    (*prefix, "sessions", r"session_[^/]+\.json"),
                    f"{example_prefix}sessions/session_1.json",
                ),
                LayoutEntry(
                    "session_snapshot",
                    (*prefix, "sessions", "saved", r"[^/]+\.json"),
                    f"{example_prefix}sessions/saved/conversation_1.json",
                ),
                LayoutEntry(
                    "request_dump_sidecar",
                    (*prefix, "sessions", r"request_dump_[^/]+\.json"),
                    f"{example_prefix}sessions/request_dump_1.json",
                ),
                LayoutEntry(
                    "atif_document",
                    (*prefix, "observability", "nemo-relay", "atif", r"[^/]+\.json"),
                    f"{example_prefix}observability/nemo-relay/atif/trajectory-1.json",
                ),
                LayoutEntry(
                    "atof_stream",
                    (*prefix, "observability", "nemo-relay", "atof", r"[^/]+\.jsonl"),
                    f"{example_prefix}observability/nemo-relay/atof/events.jsonl",
                ),
            )
        )
        return rows

    # A Hermes profile is a complete Hermes home of its own below profiles/.
    return SourceLayout(Provider.HERMES, (*home((), ""), *home(("profiles", NAME), "profiles/work/")))


def hook_carrier_layout(provider: Provider) -> SourceLayout:
    """Hook carriers: one ``.ndjson`` file per writer under a day directory."""

    return SourceLayout(
        provider,
        (LayoutEntry("hook_event_carrier", (_DATE, r"[^/]+\.ndjson"), "2026-01-01/carrier-1.ndjson"),),
    )


def export_drop_layout(suffixes: Sequence[str]) -> SourceLayout:
    """An export drop: material whose structure is the export's own.

    The archive inbox holds operator drops, and a directory declared for an
    export-only origin (ChatGPT, Claude.ai, Gemini, ...) is the same kind of
    material. Neither has a provider-written position, so the admission
    boundary is the file format; ``polylogue import`` and the bound
    acquisition boundary then hold each file to its declared origin.
    """

    suffix = "|".join(re.escape(item) for item in suffixes)
    return SourceLayout(
        None,
        (LayoutEntry("export_drop", (ANY_DEPTH, rf"[^/]+(?:{suffix})"), "export/conversations.json"),),
    )


def _browser_capture_layout() -> SourceLayout:
    # ``capture_artifact_path``: <provider token>/<session>-<hash>.json. The
    # receiver's own bookkeeping (browser-actions/, backfill-checkpoints/,
    # .staging/) sits beside the provider directories and is never admitted.
    providers = "|".join(re.escape(provider.value) for provider in Provider)
    return SourceLayout(
        None,
        (
            LayoutEntry(
                "browser_capture_envelope",
                (rf"(?:{providers})", r"[^/]+\.json"),
                "chatgpt/session-0123456789ab.json",
            ),
        ),
    )


@functools.cache
def declared_source_layouts() -> Mapping[str, SourceLayout]:
    """The layout of every canonical watch source, by watch-source name."""

    from polylogue.sources.live.watcher import HOOK_CARRIER_PROVIDERS, INBOX_SOURCE_SUFFIXES

    layouts: dict[str, SourceLayout] = {
        "claude-code": _claude_code_projects_layout(),
        "claude-code-todos": SourceLayout(
            Provider.CLAUDE_CODE,
            (LayoutEntry("todo_snapshot", (r"[^/]+\.json",), "00000000-0000-4000-8000-000000000001.json"),),
        ),
        "claude-code-history": SourceLayout(
            Provider.CLAUDE_CODE,
            (LayoutEntry("prompt_history_log", (r"history\.jsonl",), "history.jsonl"),),
        ),
        "codex": SourceLayout(
            Provider.CODEX,
            (
                LayoutEntry(
                    "session_stream",
                    (r"\d{4}", r"\d{2}", r"\d{2}", r"rollout-[^/]+\.jsonl"),
                    "2026/01/02/rollout-2026-01-02T00-00-00-1.jsonl",
                ),
            ),
        ),
        "codex-state": _codex_state_layout(),
        "codex-memories": SourceLayout(
            Provider.CODEX,
            (LayoutEntry("agent_memory_document", (ANY_DEPTH, r"[^/]+\.md"), "archive/MEMORY.md"),),
        ),
        "gemini-cli": SourceLayout(
            Provider.GEMINI_CLI,
            (
                LayoutEntry(
                    "session_document",
                    (NAME, "chats", r"session-[^/]+\.(?:json|jsonl)"),
                    "project/chats/session-2026-01-01T00-00-1.json",
                ),
                LayoutEntry(
                    "subagent_session_document",
                    (NAME, "chats", NAME, r"[^/]+\.json"),
                    "project/chats/00000000-0000-4000-8000-000000000001/sub1.json",
                ),
                LayoutEntry("prompt_history_log", (NAME, r"logs\.json"), "project/logs.json"),
                LayoutEntry(
                    "tool_result_sidecar",
                    (NAME, "tool-outputs", r"session-[^/]+", r"(?!\.)[^/]+"),
                    "project/tool-outputs/session-1/run_shell_command_1.txt",
                ),
            ),
        ),
        "hermes": _hermes_layout(),
        "antigravity": SourceLayout(
            Provider.ANTIGRAVITY,
            (
                LayoutEntry("session_document", ("conversations", r"[^/]+\.pb"), "conversations/c1.pb"),
                LayoutEntry("metadata_document", ("brain", NAME, r"[^/]+\.md"), "brain/c1/task.md"),
                LayoutEntry(
                    "agent_sidecar_meta",
                    ("brain", NAME, r"[^/]+\.metadata\.json"),
                    "brain/c1/task.md.metadata.json",
                ),
            ),
        ),
        # Antigravity's CLI keeps one trajectory SQLite store per conversation
        # beside the IDE state directory; ``conversation_summaries.db`` next to
        # it is a summary index the parser does not consume.
        "antigravity-cli": SourceLayout(
            Provider.ANTIGRAVITY,
            (LayoutEntry("trajectory_store", ("conversations", r"[^/]+\.db"), "conversations/c1.db"),),
        ),
        "browser-capture": _browser_capture_layout(),
        "inbox": export_drop_layout(INBOX_SOURCE_SUFFIXES),
    }
    for provider_token in HOOK_CARRIER_PROVIDERS:
        layouts[f"{provider_token}-hooks"] = hook_carrier_layout(Provider.from_string(provider_token))
    return layouts


def declared_source_layout(name: str) -> SourceLayout:
    """The declared layout of a canonical watch source; a missing one is a defect."""

    try:
        return declared_source_layouts()[name]
    except KeyError:
        raise KeyError(f"watch source {name!r} declares no layout") from None


def source_layout_for(name: str) -> SourceLayout:
    """The layout a source named ``name`` is walked by, on every route.

    A canonical watch-source name resolves to its declared layout, wherever
    its root is: a one-shot source named ``codex`` is walked as
    ``~/.codex/sessions`` is. Any other name labels an explicitly
    declared input -- an export-only origin's directory, an operator drop --
    which has no provider-written position and is walked as an export drop.
    """

    layouts = declared_source_layouts()
    if name in layouts:
        return layouts[name]
    return explicit_input_layout()


def explicit_input_layout() -> SourceLayout:
    """The layout of an explicitly named input directory: an export drop.

    An operator-named root (``polylogue import``, a schema-inference
    ``provider=path`` argument) is input, not discovery of a provider-written
    location, so it is bounded by file format, as the inbox is.
    """

    from polylogue.sources.live.watcher import INBOX_SOURCE_SUFFIXES

    return _export_drop(tuple(INBOX_SOURCE_SUFFIXES))


@functools.cache
def _export_drop(suffixes: tuple[str, ...]) -> SourceLayout:
    return export_drop_layout(suffixes)


#: The project directory synthetic and demo Claude Code sessions live in.
SYNTHETIC_CLAUDE_CODE_PROJECT = "-synthetic-project"


def canonical_session_position(source_name: str, stem: str, suffix: str) -> PurePosixPath:
    """Where a source holds a session document named ``stem``, relative to its root.

    Writers of synthetic and demo source trees place each session where the
    provider itself would, so every route admits it through the same
    declared layout as real material. An export label keeps the file at the
    root of its drop.
    """

    if source_name == "claude-code":
        return PurePosixPath(SYNTHETIC_CLAUDE_CODE_PROJECT, f"{stem}.jsonl")
    if source_name == "codex":
        return PurePosixPath("2026", "01", "01", f"rollout-{stem}.jsonl")
    if source_name == "gemini-cli":
        return PurePosixPath("synthetic", "chats", f"session-{stem}{suffix}")
    if source_name == "hermes":
        if suffix == ".jsonl":
            return PurePosixPath("observability", "nemo-relay", "atof", f"{stem}.jsonl")
        return PurePosixPath("sessions", f"session_{stem}{suffix}")
    if source_name == "antigravity":
        return PurePosixPath("conversations", f"{stem}.pb")
    return PurePosixPath(f"{stem}{suffix}")


def layout_declaration_defects(root_for: Mapping[str, str]) -> tuple[str, ...]:
    """Check every declared entry against its own example and the owning rule.

    ``root_for`` maps a watch-source name to a synthetic absolute root (rule
    patterns are written against canonical install paths). An entry must
    admit its example, and the provider's artifact rule must classify the
    example as the same kind, or claim no rule when the kind is layout-only.
    """

    from polylogue.sources.origin_specs import ORIGIN_SPECS, artifact_rule_for_path

    defects: list[str] = []
    for name, layout in declared_source_layouts().items():
        rule_kinds: set[str] = set()
        if layout.provider is not None:
            for spec in ORIGIN_SPECS:
                if layout.provider in spec.provider_wires:
                    rule_kinds.update(rule.kind for rule in spec.artifact_rules)
        for entry in layout.entries:
            parts = PurePosixPath(entry.example).parts
            if layout.artifact_kind(parts) != entry.kind:
                defects.append(f"{name}: {entry.kind} example {entry.example!r} resolves elsewhere in its layout")
            if layout.provider is None or name not in root_for:
                continue
            rule = artifact_rule_for_path(layout.provider, f"{root_for[name]}/{entry.example}")
            if entry.kind in rule_kinds:
                if rule is None or rule.kind != entry.kind:
                    found = None if rule is None else rule.kind
                    defects.append(f"{name}: {entry.kind} example {entry.example!r} matches rule {found!r}")
            elif rule is not None:
                defects.append(
                    f"{name}: layout-only kind {entry.kind} example {entry.example!r} is claimed by rule {rule.kind!r}"
                )
    return tuple(defects)


__all__ = [
    "ANY_DEPTH",
    "SYNTHETIC_CLAUDE_CODE_PROJECT",
    "canonical_session_position",
    "NAME",
    "LayoutEntry",
    "SourceLayout",
    "declared_source_layout",
    "declared_source_layouts",
    "explicit_input_layout",
    "export_drop_layout",
    "hook_carrier_layout",
    "layout_declaration_defects",
    "source_layout_for",
]
