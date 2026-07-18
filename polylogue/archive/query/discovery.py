"""Executable, parser-gated query discovery declarations.

The rows in this module are the declaration source for MCP discovery,
query-completion examples, generated search documentation, and cookbook
prompts. Tests pass every positive row through the production parser and pin
every negative row to the production diagnostic class and text.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal, cast

from polylogue.archive.query.transaction import (
    QueryCoverageClass,
    QueryResultSemanticsContract,
)

QueryDiscoveryParser = Literal["session", "unit"]
QueryDiscoveryCostClass = Literal["selective", "corpus-scale"]
QueryDiscoveryCatalogView = Literal["full", "featured", "capability"]
QueryDiscoveryUnitSource = Literal[
    "sessions",
    "messages",
    "actions",
    "blocks",
    "assertions",
    "files",
    "runs",
    "observed-events",
    "context-snapshots",
    "delegations",
]
QueryDiscoveryRoute = Literal[
    "query",
    "ranked-search",
    "sampled-query",
    "aggregate-query",
    "context-builder",
    "recursive-graph",
]
QueryExampleParameterKind = Literal["text", "value", "date"]

QUERY_DISCOVERY_GRAMMAR: dict[str, str] = {
    "compact_form": "<field>:<value> [AND|OR|NOT ...]",
    "session_form": "sessions where <session-predicate>",
    "terminal_form": "<terminal-source> where <predicate>",
    "pipeline_form": "<terminal-source> where <predicate> | group by <field> | count",
    "scope_form": "sessions where <predicate> | <terminal-source> where <predicate>",
    "projection_form": "<session-expression> with <unit>(<columns>)",
}


@dataclass(frozen=True, slots=True)
class QueryExampleParameter:
    """One named, typed substitution accepted by a discovery example template."""

    name: str
    kind: QueryExampleParameterKind


@dataclass(frozen=True, slots=True)
class QueryDiscoveryExample:
    """One positive query example with explicit execution/result semantics."""

    key: str
    expression: str
    parser: QueryDiscoveryParser
    unit_source: QueryDiscoveryUnitSource
    answers: str
    result_semantics: QueryCoverageClass
    projection_columns: tuple[str, ...]
    cost_class: QueryDiscoveryCostClass
    route: QueryDiscoveryRoute
    featured: bool = False
    template: str | None = None
    parameters: tuple[QueryExampleParameter, ...] = ()

    def to_payload(self) -> dict[str, object]:
        return {
            "key": self.key,
            "expression": self.expression,
            "parser": self.parser,
            "unit_source": self.unit_source,
            "answers": self.answers,
            "result_semantics": self.result_semantics,
            "projection_columns": list(self.projection_columns),
            "cost_class": self.cost_class,
            "route": self.route,
            "featured": self.featured,
            "template": self.template,
            "parameters": [{"name": parameter.name, "kind": parameter.kind} for parameter in self.parameters],
        }


@dataclass(frozen=True, slots=True)
class QueryDiscoveryNegativeExample:
    """One common rejected form and the real parser diagnostic it must retain."""

    key: str
    expression: str
    parser: QueryDiscoveryParser
    diagnostic_class: str
    diagnostic: str
    corrected_form: str
    field: str | None = None
    shipped_at: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, object]:
        return {
            "key": self.key,
            "expression": self.expression,
            "parser": self.parser,
            "diagnostic_class": self.diagnostic_class,
            "diagnostic": self.diagnostic,
            "field": self.field,
            "corrected_form": self.corrected_form,
            "shipped_at": list(self.shipped_at),
        }


SESSION_COLUMNS = (
    "id",
    "origin",
    "title",
    "target_ref",
    "anchor",
    "actions",
    "created_at",
    "updated_at",
    "message_count",
    "tags",
    "summary",
    "words",
    "repo",
    "cwd_display",
    "flags",
)
RANKED_SESSION_COLUMNS = (
    "session.id",
    "session.origin",
    "session.title",
    "session.message_count",
    "session.created_at",
    "session.updated_at",
    "match.rank",
    "match.retrieval_lane",
    "match.match_surface",
    "match.target_ref",
    "match.message_id",
    "match.snippet",
    "match.score",
    "match.score_kind",
    "match.matched_terms",
)
MESSAGE_COLUMNS = (
    "unit",
    "message_id",
    "session_id",
    "origin",
    "title",
    "role",
    "message_type",
    "material_origin",
    "occurred_at_ms",
    "position",
    "word_count",
    "text",
)
ACTION_COLUMNS = (
    "unit",
    "session_id",
    "message_id",
    "origin",
    "title",
    "tool_use_block_id",
    "tool_result_block_id",
    "tool_name",
    "semantic_type",
    "tool_command",
    "tool_path",
    "occurred_at_ms",
    "output_text",
    "is_error",
    "exit_code",
    "followup_class",
    "followup_message_ref",
)
BLOCK_COLUMNS = (
    "unit",
    "block_id",
    "message_id",
    "session_id",
    "origin",
    "title",
    "block_type",
    "position",
    "text",
    "tool_name",
    "semantic_type",
    "tool_command",
    "tool_path",
)
FILE_COLUMNS = (
    "unit",
    "session_id",
    "origin",
    "title",
    "path",
    "action_count",
    "first_message_id",
    "first_tool_use_block_id",
    "last_tool_use_block_id",
    "first_seen_ms",
    "last_seen_ms",
)
ASSERTION_COLUMNS = (
    "unit",
    "assertion_id",
    "target_ref",
    "scope_ref",
    "kind",
    "key",
    "body_text",
    "value",
    "author_ref",
    "author_kind",
    "status",
    "visibility",
    "evidence_refs",
    "staleness",
    "context_policy",
    "created_at_ms",
    "updated_at_ms",
)
RUN_COLUMNS = (
    "unit",
    "run_ref",
    "session_id",
    "origin",
    "title",
    "native_session_id",
    "native_parent_session_id",
    "parent_run_ref",
    "agent_ref",
    "lineage_refs",
    "provider_origin",
    "harness",
    "role",
    "cwd",
    "git_branch",
    "status",
    "confidence",
    "transcript_ref",
    "evidence_refs",
    "context_snapshot_ref",
)
OBSERVED_EVENT_COLUMNS = (
    "unit",
    "event_ref",
    "session_id",
    "origin",
    "title",
    "kind",
    "summary",
    "delivery_state",
    "subject_ref",
    "object_refs",
    "evidence_refs",
)
CONTEXT_SNAPSHOT_COLUMNS = (
    "unit",
    "snapshot_ref",
    "session_id",
    "origin",
    "title",
    "run_ref",
    "boundary",
    "inheritance_mode",
    "segment_refs",
    "evidence_refs",
    "metadata",
)
DELEGATION_COLUMNS = (
    "unit",
    "delegation_ref",
    "parent_session_id",
    "child_session_id",
    "parent_origin",
    "mapping_state",
    "evidence_basis",
    "instruction_message_id",
    "instruction_tool_use_block_id",
    "instruction_preview",
    "instruction_sha256",
    "instruction_truncated",
    "artifact_block_id",
    "artifact_preview",
    "artifact_sha256",
    "artifact_truncated",
    "dispatch_turn_model",
    "requested_model",
    "child_session_dominant_model",
    "result_is_error",
    "result_exit_code",
    "result_status",
    "link_confidence",
    "link_method",
    "inheritance",
    "evidence_refs",
)
AGGREGATE_COLUMNS = ("unit", "group_by", "group_key", "count")
RECURSIVE_COLUMNS = ("session_id", "parent_refs", "child_refs", "continuation")


RESULT_SEMANTICS_TEACHING: tuple[QueryResultSemanticsContract, ...] = (
    QueryResultSemanticsContract(
        coverage="exhaustive",
        total="qualified",
        continuation="cursor-or-offset",
        phrase=("Exhaustive relation is paged: total is page-local; follow continuation until absent."),
    ),
    QueryResultSemanticsContract(
        coverage="top-k",
        total="qualified",
        continuation="ranked-frontier",
        phrase=(
            "Top-k by relevance, not exhaustive; total is qualified and must not be read as all matching archive rows."
        ),
    ),
    QueryResultSemanticsContract(
        coverage="sample",
        total="qualified",
        continuation="none",
        phrase="Sample, not exhaustive; total is qualified and does not establish archive-wide coverage.",
    ),
    QueryResultSemanticsContract(
        coverage="aggregate",
        total="aggregate",
        continuation="cursor-or-offset",
        phrase=(
            "Aggregate over the declared input relation; totals describe aggregate buckets, while input coverage is "
            "reported separately."
        ),
    ),
    QueryResultSemanticsContract(
        coverage="bounded-context",
        total="qualified",
        continuation="none",
        phrase=(
            "Bounded context for orientation, not exhaustive; omitted evidence is expected and total is qualified."
        ),
    ),
    QueryResultSemanticsContract(
        coverage="recursive-page",
        total="qualified",
        continuation="recursive-cursor",
        phrase=(
            "Recursive relation, physically paged; node and edge totals remain qualified until every continuation ends."
        ),
    ),
)

_RESULT_SEMANTICS_BY_CLASS = {contract.coverage: contract for contract in RESULT_SEMANTICS_TEACHING}


def result_semantics_teaching(coverage: QueryCoverageClass) -> QueryResultSemanticsContract:
    """Return the shared truthful wording for one coverage class."""

    return _RESULT_SEMANTICS_BY_CLASS[coverage]


def _example(
    key: str,
    expression: str,
    unit_source: QueryDiscoveryUnitSource,
    answers: str,
    *,
    result_semantics: QueryCoverageClass = "exhaustive",
    projection_columns: tuple[str, ...] | None = None,
    cost_class: QueryDiscoveryCostClass = "selective",
    route: QueryDiscoveryRoute = "query",
    featured: bool = False,
    template: str | None = None,
    parameters: tuple[QueryExampleParameter, ...] = (),
) -> QueryDiscoveryExample:
    parser: QueryDiscoveryParser = "session" if unit_source == "sessions" else "unit"
    if projection_columns is None:
        projection_columns = RANKED_SESSION_COLUMNS if result_semantics == "top-k" else SESSION_COLUMNS
    return QueryDiscoveryExample(
        key=key,
        expression=expression,
        parser=parser,
        unit_source=unit_source,
        answers=answers,
        result_semantics=result_semantics,
        projection_columns=projection_columns,
        cost_class=cost_class,
        route=route,
        featured=featured,
        template=template,
        parameters=parameters,
    )


# 106 provider-neutral, privacy-safe positive declarations. The corpus is
# intentionally broad enough to exercise compact selectors, Boolean session
# predicates, every terminal source, session-scoped pipelines, aggregates,
# projections, ranked search, sampling, bounded context, and recursive seeds.
QUERY_DISCOVERY_EXAMPLES: tuple[QueryDiscoveryExample, ...] = (
    _example(
        "session-repository",
        "repo:example-repo",
        "sessions",
        "Finds sessions associated with one repository.",
        featured=True,
    ),
    _example(
        "session-origin", "origin:unknown-export", "sessions", "Finds sessions from the neutral unknown-export origin."
    ),
    _example("session-tag", "tag:review", "sessions", "Finds sessions carrying the review tag."),
    _example("session-path", "path:src/query", "sessions", "Finds sessions that reference a query-source path."),
    _example(
        "session-cwd", "cwd:/workspace/example-repo", "sessions", "Finds sessions rooted below one workspace directory."
    ),
    _example("session-tool", "tool:shell", "sessions", "Finds sessions that used a shell tool."),
    _example("session-action", "action:file_edit", "sessions", "Finds sessions containing file-edit actions."),
    _example("session-has-tools", "has:tools", "sessions", "Finds sessions with tool-use evidence."),
    _example(
        "session-exact-ref", "session:example-origin:session-001", "sessions", "Finds one exact session reference."
    ),
    _example(
        "session-title", 'title:"query compiler"', "sessions", "Finds sessions whose title mentions the query compiler."
    ),
    _example("session-since", "since:7d", "sessions", "Finds sessions from the last seven days."),
    _example("session-until", "until:2026-06-30", "sessions", "Finds sessions at or before a fixed date."),
    _example(
        "session-contains",
        'contains:"schema migration"',
        "sessions",
        "Finds sessions containing an exact schema-migration phrase.",
        cost_class="corpus-scale",
    ),
    _example("session-message-count", "messages:>=10", "sessions", "Finds sessions with at least ten messages."),
    _example("session-word-count", "words:<=2000", "sessions", "Finds sessions with no more than two thousand words."),
    _example(
        "session-negated-tag",
        "repo:example-repo AND NOT tag:stale",
        "sessions",
        "Finds repository sessions that are not tagged stale.",
    ),
    _example(
        "session-boolean-repositories",
        "sessions where (repo:example-repo OR repo:example-library) AND NOT tag:stale",
        "sessions",
        "Finds non-stale sessions from either of two repositories.",
        featured=True,
    ),
    _example(
        "session-duration",
        "sessions where duration_ms >= 60000",
        "sessions",
        "Finds sessions lasting at least one minute.",
    ),
    _example(
        "session-authored-volume",
        "sessions where authored_user_messages >= 2 AND authored_user_words >= 100",
        "sessions",
        "Finds sessions with at least two substantial human-authored prompts.",
    ),
    _example(
        "session-no-system-or-tool-messages",
        "sessions where system_messages = 0 AND tool_messages = 0",
        "sessions",
        "Finds sessions containing neither system nor tool-role messages.",
    ),
    _example(
        "session-tool-without-paste",
        "sessions where tool_use_messages >= 1 AND paste_messages = 0",
        "sessions",
        "Finds tool-using sessions without paste evidence.",
    ),
    _example(
        "session-exists-message",
        "sessions where exists message(role:assistant AND text:timeout)",
        "sessions",
        "Finds sessions with an assistant message mentioning a timeout.",
        cost_class="corpus-scale",
        featured=True,
    ),
    _example(
        "session-exists-action",
        "sessions where exists action(tool:shell AND command:pytest)",
        "sessions",
        "Finds sessions with a shell action whose command mentions pytest.",
        cost_class="corpus-scale",
    ),
    _example(
        "session-exists-file",
        "sessions where exists file(action:file_edit AND path:src/query)",
        "sessions",
        "Finds sessions that edited a file below the query source path.",
    ),
    _example(
        "session-sequence",
        "sessions where seq(action:file_edit -> action:shell)",
        "sessions",
        "Finds sessions where a file edit precedes a shell action.",
        cost_class="corpus-scale",
        featured=True,
    ),
    _example(
        "session-date-range",
        "sessions where date between 2026-06-01 and 2026-06-30",
        "sessions",
        "Finds sessions inside a fixed June date range.",
    ),
    _example(
        "ranked-semantic-text",
        'near:"semantic search"',
        "sessions",
        "Ranks the sessions most relevant to a semantic-search phrase.",
        result_semantics="top-k",
        cost_class="corpus-scale",
        route="ranked-search",
        featured=True,
        template="near:{topic}",
        parameters=(QueryExampleParameter("topic", "text"),),
    ),
    _example(
        "ranked-similar-session",
        "near:id:example-origin:session-001",
        "sessions",
        "Ranks sessions most similar to one stored session.",
        result_semantics="top-k",
        cost_class="corpus-scale",
        route="ranked-search",
    ),
    _example(
        "ranked-boolean-semantic",
        'sessions where semantic:"query compiler failure"',
        "sessions",
        "Ranks sessions relevant to a query-compiler failure.",
        result_semantics="top-k",
        cost_class="corpus-scale",
        route="ranked-search",
    ),
    _example(
        "ranked-near-text-alias",
        "sessions where near:text:timeout",
        "sessions",
        "Ranks sessions relevant to timeout text through the near-text alias.",
        result_semantics="top-k",
        cost_class="corpus-scale",
        route="ranked-search",
    ),
    _example(
        "ranked-fts-leaf",
        '~"sqlite locking"',
        "sessions",
        "Ranks sessions matching a SQLite-locking full-text phrase.",
        result_semantics="top-k",
        cost_class="corpus-scale",
        route="ranked-search",
    ),
    _example(
        "ranked-fts-repository",
        'sessions where ~"null pointer" AND repo:example-repo',
        "sessions",
        "Ranks repository sessions matching a null-pointer phrase.",
        result_semantics="top-k",
        cost_class="corpus-scale",
        route="ranked-search",
    ),
    _example(
        "ranked-rollback",
        'near:"failed migration rollback"',
        "sessions",
        "Ranks sessions relevant to a failed migration rollback.",
        result_semantics="top-k",
        cost_class="corpus-scale",
        route="ranked-search",
    ),
    _example(
        "ranked-reviewed-exceptions",
        'sessions where semantic:"unhandled exception" AND tag:review',
        "sessions",
        "Ranks reviewed sessions relevant to unhandled exceptions.",
        result_semantics="top-k",
        cost_class="corpus-scale",
        route="ranked-search",
    ),
    _example(
        "ranked-repository-window",
        "repo:example-repo AND since:30d",
        "sessions",
        "Ranks recent sessions from one repository through the search route.",
        result_semantics="top-k",
        cost_class="corpus-scale",
        route="ranked-search",
        template="repo:{repo} AND since:{since}",
        parameters=(QueryExampleParameter("repo", "value"), QueryExampleParameter("since", "date")),
    ),
    _example(
        "sample-repository-window",
        "repo:example-repo since:30d",
        "sessions",
        "Samples recent sessions from one repository.",
        result_semantics="sample",
        route="sampled-query",
        featured=True,
    ),
    _example(
        "sample-origin-family",
        "origin:(unknown-export|beads-issue)",
        "sessions",
        "Samples sessions from two neutral origin families.",
        result_semantics="sample",
        route="sampled-query",
    ),
    _example(
        "sample-tool-using",
        "sessions where tool_use_messages >= 1",
        "sessions",
        "Samples sessions containing tool-use evidence.",
        result_semantics="sample",
        route="sampled-query",
    ),
    _example(
        "sample-dialogue",
        "sessions where authored_user_messages >= 1 AND assistant_messages >= 1",
        "sessions",
        "Samples sessions containing both authored prompts and assistant replies.",
        result_semantics="sample",
        route="sampled-query",
    ),
    _example(
        "sample-error-actions",
        "sessions where exists action(is_error:true)",
        "sessions",
        "Samples sessions containing error-marked actions.",
        result_semantics="sample",
        cost_class="corpus-scale",
        route="sampled-query",
    ),
    _example(
        "sample-candidate-assertions",
        "sessions where exists assertion(status:candidate)",
        "sessions",
        "Samples sessions with candidate assertions.",
        result_semantics="sample",
        cost_class="corpus-scale",
        route="sampled-query",
    ),
    _example(
        "messages-assistant-timeout",
        "messages where role:assistant AND text:timeout",
        "messages",
        "Returns assistant messages mentioning a timeout.",
        projection_columns=MESSAGE_COLUMNS,
        cost_class="corpus-scale",
        featured=True,
    ),
    _example(
        "messages-repository-text",
        "messages where session.repo:example-repo AND type:text",
        "messages",
        "Returns text messages owned by sessions in one repository.",
        projection_columns=MESSAGE_COLUMNS,
    ),
    _example(
        "messages-token-duration",
        "messages where input_tokens >= 1000 AND duration_ms between 2000 and 3000",
        "messages",
        "Returns high-input messages whose duration is between two and three seconds.",
        projection_columns=MESSAGE_COLUMNS,
    ),
    _example(
        "actions-file-edits",
        "actions where session.repo:example-repo AND action:file_edit AND path:src/query",
        "actions",
        "Returns file-edit actions under the query source path for one repository.",
        projection_columns=ACTION_COLUMNS,
        featured=True,
    ),
    _example(
        "actions-shell-pytest",
        "actions where tool:shell AND command:pytest",
        "actions",
        "Returns shell actions whose command mentions pytest.",
        projection_columns=ACTION_COLUMNS,
        cost_class="corpus-scale",
        featured=True,
    ),
    _example(
        "actions-failed-output",
        "actions where is_error:true AND output:failed",
        "actions",
        "Returns error-marked actions whose output mentions failure.",
        projection_columns=ACTION_COLUMNS,
        cost_class="corpus-scale",
    ),
    _example(
        "actions-unacknowledged-failures",
        "actions where session.repo:example-repo AND session.since:7d AND output:failed",
        "actions",
        "Returns recent failed action evidence for one repository.",
        projection_columns=ACTION_COLUMNS,
        cost_class="corpus-scale",
        featured=True,
        template="actions where session.repo:{repo} AND session.since:{since} AND output:failed",
        parameters=(QueryExampleParameter("repo", "value"), QueryExampleParameter("since", "date")),
    ),
    _example(
        "blocks-code-sqlite",
        "blocks where type:code AND text:sqlite",
        "blocks",
        "Returns code blocks mentioning SQLite.",
        projection_columns=BLOCK_COLUMNS,
        cost_class="corpus-scale",
    ),
    _example(
        "blocks-recent-short-sessions",
        "blocks where session.since:7d AND session.words:<=500 AND type:code",
        "blocks",
        "Returns code blocks from recent sessions with at most five hundred words.",
        projection_columns=BLOCK_COLUMNS,
    ),
    _example(
        "blocks-editor-readme",
        "blocks where tool:editor AND path:README.md",
        "blocks",
        "Returns editor blocks addressing the README path.",
        projection_columns=BLOCK_COLUMNS,
    ),
    _example(
        "files-query-parser",
        "files where path:src/query/parser.py",
        "files",
        "Returns file evidence for the query parser path.",
        projection_columns=FILE_COLUMNS,
        featured=True,
        template="files where path:{path}",
        parameters=(QueryExampleParameter("path", "value"),),
    ),
    _example(
        "files-repository-path",
        "files where session.repo:example-repo AND path:src/mcp/server.py",
        "files",
        "Returns file evidence for one path within one repository.",
        projection_columns=FILE_COLUMNS,
        featured=True,
        template="files where session.repo:{repo} AND path:{path}",
        parameters=(QueryExampleParameter("repo", "value"), QueryExampleParameter("path", "value")),
    ),
    _example(
        "files-read-search-doc",
        "files where action:file_read AND path:docs/search.md",
        "files",
        "Returns file-read evidence for the search documentation path.",
        projection_columns=FILE_COLUMNS,
    ),
    _example(
        "assertions-active-decisions",
        "assertions where kind:decision AND status:active AND text:review",
        "assertions",
        "Returns active decision assertions mentioning review.",
        projection_columns=ASSERTION_COLUMNS,
        cost_class="corpus-scale",
    ),
    _example(
        "assertions-decisions-about-topic",
        'assertions where kind:decision AND text:"schema migration"',
        "assertions",
        "Returns decision assertions about a named topic.",
        projection_columns=ASSERTION_COLUMNS,
        cost_class="corpus-scale",
        featured=True,
        template="assertions where kind:decision AND text:{topic}",
        parameters=(QueryExampleParameter("topic", "text"),),
    ),
    _example(
        "assertions-repository-caveats",
        "assertions where session.repo:example-repo AND kind:caveat",
        "assertions",
        "Returns caveat assertions associated with one repository.",
        projection_columns=ASSERTION_COLUMNS,
    ),
    _example(
        "assertions-candidate-target",
        "assertions where status:candidate AND target:session:example-origin:session-001",
        "assertions",
        "Returns candidate assertions targeting one session.",
        projection_columns=ASSERTION_COLUMNS,
    ),
    _example(
        "runs-completed-subagents",
        "runs where role:subagent AND status:completed AND agent:worker",
        "runs",
        "Returns completed subagent runs attributed to a worker agent.",
        projection_columns=RUN_COLUMNS,
    ),
    _example(
        "runs-repository-main",
        "runs where session.repo:example-repo AND role:main",
        "runs",
        "Returns main runs associated with one repository.",
        projection_columns=RUN_COLUMNS,
    ),
    _example(
        "runs-provider-branch",
        "runs where provider_origin:example-provider AND git_branch:main",
        "runs",
        "Returns runs recorded on the main branch for one neutral provider label.",
        projection_columns=RUN_COLUMNS,
    ),
    _example(
        "events-acted-on",
        "observed-events where delivery_state:acted_on AND text:issue-2100",
        "observed-events",
        "Returns acted-on observed events mentioning one issue token.",
        projection_columns=OBSERVED_EVENT_COLUMNS,
        cost_class="corpus-scale",
    ),
    _example(
        "events-origin-object",
        "observed-events where session.origin:unknown-export AND object_ref:review:example",
        "observed-events",
        "Returns observed events for one origin and object reference.",
        projection_columns=OBSERVED_EVENT_COLUMNS,
    ),
    _example(
        "events-tool-handler",
        "observed-events where kind:tool_finished AND handler:mcp",
        "observed-events",
        "Returns tool-finished events handled through MCP.",
        projection_columns=OBSERVED_EVENT_COLUMNS,
    ),
    _example(
        "snapshots-session-start",
        "context-snapshots where boundary:session_start AND session.repo:example-repo",
        "context-snapshots",
        "Returns session-start context snapshots for one repository.",
        projection_columns=CONTEXT_SNAPSHOT_COLUMNS,
    ),
    _example(
        "snapshots-subagent-date",
        "context-snapshots where session.messages:>=2 AND session.date:>=2026-01-02 AND boundary:subagent_start",
        "context-snapshots",
        "Returns subagent-start snapshots owned by multi-message sessions after a date.",
        projection_columns=CONTEXT_SNAPSHOT_COLUMNS,
    ),
    _example(
        "snapshots-isolated-handoff",
        "context-snapshots where inheritance_mode:isolated AND text:handoff",
        "context-snapshots",
        "Returns isolated context snapshots mentioning a handoff.",
        projection_columns=CONTEXT_SNAPSHOT_COLUMNS,
        cost_class="corpus-scale",
    ),
    _example(
        "delegations-resolved-review",
        "delegations where mapping_state:resolved AND instruction:review",
        "delegations",
        "Returns resolved delegations whose instruction mentions review.",
        projection_columns=DELEGATION_COLUMNS,
        cost_class="corpus-scale",
    ),
    _example(
        "delegations-completed-model",
        "delegations where result_status:completed AND child_model:example-model",
        "delegations",
        "Returns completed delegations attributed to one neutral child-model label.",
        projection_columns=DELEGATION_COLUMNS,
    ),
    _example(
        "delegations-repository-success",
        "delegations where session.repo:example-repo AND is_error:false",
        "delegations",
        "Returns non-error delegations associated with one repository.",
        projection_columns=DELEGATION_COLUMNS,
    ),
    _example(
        "scoped-repository-messages",
        "sessions where repo:example-repo AND origin:unknown-export | messages where role:assistant",
        "messages",
        "Returns assistant messages scoped to repository sessions from one origin.",
        projection_columns=MESSAGE_COLUMNS,
        featured=True,
    ),
    _example(
        "scoped-origin-actions",
        "sessions where origin:(unknown-export|beads-issue) | actions where action:file_edit",
        "actions",
        "Returns file-edit actions scoped to two neutral origin families.",
        projection_columns=ACTION_COLUMNS,
    ),
    _example(
        "scoped-repository-files",
        "sessions where repo:example-repo | files where action:file_edit",
        "files",
        "Returns edited-file evidence scoped to one repository.",
        projection_columns=FILE_COLUMNS,
    ),
    _example(
        "scoped-fts-user-messages",
        'sessions where ~"timeout failure" | messages where role:user',
        "messages",
        "Returns user messages from sessions matching a timeout-failure phrase.",
        projection_columns=MESSAGE_COLUMNS,
        cost_class="corpus-scale",
    ),
    _example(
        "scoped-code-user-messages",
        "sessions where exists block(type:code) | messages where role:user",
        "messages",
        "Returns user messages from sessions containing code blocks.",
        projection_columns=MESSAGE_COLUMNS,
        cost_class="corpus-scale",
    ),
    _example(
        "scoped-sequence-user-messages",
        "sessions where seq(action:file_edit -> action:shell) | messages where role:user",
        "messages",
        "Returns user messages from sessions with an edit-then-shell sequence.",
        projection_columns=MESSAGE_COLUMNS,
        cost_class="corpus-scale",
    ),
    _example(
        "scoped-repository-assertions",
        "sessions where repo:example-repo | assertions where status:active",
        "assertions",
        "Returns active assertions scoped to one repository.",
        projection_columns=ASSERTION_COLUMNS,
    ),
    _example(
        "scoped-repository-events",
        "sessions where repo:example-repo | observed-events where kind:tool_finished",
        "observed-events",
        "Returns tool-finished observed events scoped to one repository.",
        projection_columns=OBSERVED_EVENT_COLUMNS,
    ),
    _example(
        "aggregate-messages-by-role",
        "messages where text:timeout | group by role | count",
        "messages",
        "Counts timeout-matching messages by role.",
        result_semantics="aggregate",
        projection_columns=AGGREGATE_COLUMNS,
        cost_class="corpus-scale",
        route="aggregate-query",
        featured=True,
    ),
    _example(
        "aggregate-assistant-messages-by-type",
        "messages where role:assistant | group by type | count | sort by count desc",
        "messages",
        "Counts assistant messages by type in descending frequency order.",
        result_semantics="aggregate",
        projection_columns=AGGREGATE_COLUMNS,
        cost_class="corpus-scale",
        route="aggregate-query",
    ),
    _example(
        "aggregate-errors-by-tool",
        "actions where is_error:true | group by tool | count",
        "actions",
        "Counts error-marked actions by tool.",
        result_semantics="aggregate",
        projection_columns=AGGREGATE_COLUMNS,
        cost_class="corpus-scale",
        route="aggregate-query",
        featured=True,
    ),
    _example(
        "aggregate-followups-by-action",
        "actions where followup_class:silent_proceed | group by action | count",
        "actions",
        "Counts silent-proceed follow-ups by action category.",
        result_semantics="aggregate",
        projection_columns=AGGREGATE_COLUMNS,
        cost_class="corpus-scale",
        route="aggregate-query",
    ),
    _example(
        "aggregate-code-blocks-by-tool",
        "blocks where type:code | group by tool | count",
        "blocks",
        "Counts code blocks by tool.",
        result_semantics="aggregate",
        projection_columns=AGGREGATE_COLUMNS,
        cost_class="corpus-scale",
        route="aggregate-query",
    ),
    _example(
        "aggregate-edited-files-by-path",
        "files where action:file_edit | group by path | count",
        "files",
        "Counts file-edit evidence by path.",
        result_semantics="aggregate",
        projection_columns=AGGREGATE_COLUMNS,
        cost_class="corpus-scale",
        route="aggregate-query",
    ),
    _example(
        "aggregate-active-assertions-by-kind",
        "assertions where status:active | group by kind | count",
        "assertions",
        "Counts active assertions by kind.",
        result_semantics="aggregate",
        projection_columns=AGGREGATE_COLUMNS,
        cost_class="corpus-scale",
        route="aggregate-query",
    ),
    _example(
        "aggregate-events-by-status",
        "observed-events where kind:tool_finished | group by status | count",
        "observed-events",
        "Counts tool-finished observed events by status.",
        result_semantics="aggregate",
        projection_columns=AGGREGATE_COLUMNS,
        cost_class="corpus-scale",
        route="aggregate-query",
    ),
    _example(
        "aggregate-delegations-by-result",
        "delegations where mapping_state:resolved | group by result_status | count",
        "delegations",
        "Counts resolved delegations by result status.",
        result_semantics="aggregate",
        projection_columns=AGGREGATE_COLUMNS,
        cost_class="corpus-scale",
        route="aggregate-query",
    ),
    _example(
        "aggregate-scoped-files-by-path",
        "sessions where repo:example-repo | files where action:file_edit | group by path | count",
        "files",
        "Counts edited-file evidence by path inside one repository scope.",
        result_semantics="aggregate",
        projection_columns=AGGREGATE_COLUMNS,
        cost_class="corpus-scale",
        route="aggregate-query",
    ),
    _example(
        "aggregate-scoped-messages-by-role",
        "sessions where repo:example-repo | messages where role:assistant | group by role | count",
        "messages",
        "Counts assistant messages by role inside one repository scope.",
        result_semantics="aggregate",
        projection_columns=AGGREGATE_COLUMNS,
        cost_class="corpus-scale",
        route="aggregate-query",
    ),
    _example(
        "aggregate-message-keys",
        "messages where text:error | group by role | count | sort by key asc | limit 10",
        "messages",
        "Returns the first ten role buckets for error-matching messages in key order.",
        result_semantics="aggregate",
        projection_columns=AGGREGATE_COLUMNS,
        cost_class="corpus-scale",
        route="aggregate-query",
    ),
    _example(
        "aggregate-shell-exit-codes",
        "actions where action:shell | group by exit_code | count | sort by count desc | limit 5",
        "actions",
        "Returns the five most frequent exit-code buckets for shell actions.",
        result_semantics="aggregate",
        projection_columns=AGGREGATE_COLUMNS,
        cost_class="corpus-scale",
        route="aggregate-query",
    ),
    _example(
        "aggregate-warning-block-types",
        "blocks where text:warning | group by type | count | sort by count desc",
        "blocks",
        "Counts warning-matching blocks by type in descending frequency order.",
        result_semantics="aggregate",
        projection_columns=AGGREGATE_COLUMNS,
        cost_class="corpus-scale",
        route="aggregate-query",
    ),
    _example(
        "aggregate-review-assertion-status",
        "assertions where text:review | group by status | count | sort by key asc",
        "assertions",
        "Counts review-matching assertions by status in key order.",
        result_semantics="aggregate",
        projection_columns=AGGREGATE_COLUMNS,
        cost_class="corpus-scale",
        route="aggregate-query",
    ),
    _example(
        "aggregate-handler-delivery",
        "observed-events where handler:mcp | group by delivery_state | count | sort by count desc",
        "observed-events",
        "Counts MCP-handled observed events by delivery state.",
        result_semantics="aggregate",
        projection_columns=AGGREGATE_COLUMNS,
        cost_class="corpus-scale",
        route="aggregate-query",
    ),
    _example(
        "context-session-messages",
        "repo:example-repo with messages(message_id,role,text)",
        "sessions",
        "Builds bounded session orientation with selected message columns.",
        result_semantics="bounded-context",
        projection_columns=("session_id", "message_id", "role", "text"),
        route="context-builder",
        featured=True,
    ),
    _example(
        "context-session-actions",
        "sessions where repo:example-repo with actions(tool_name,semantic_type,is_error)",
        "sessions",
        "Builds bounded session orientation with selected action columns.",
        result_semantics="bounded-context",
        projection_columns=("session_id", "tool_name", "semantic_type", "is_error"),
        route="context-builder",
    ),
    _example(
        "context-session-assertions",
        "sessions where tag:review with assertions(assertion_id,kind,status)",
        "sessions",
        "Builds bounded orientation for reviewed sessions with assertion columns.",
        result_semantics="bounded-context",
        projection_columns=("session_id", "assertion_id", "kind", "status"),
        route="context-builder",
    ),
    _example(
        "context-session-files",
        "sessions where path:src/query with files(path,action_count,last_seen_ms)",
        "sessions",
        "Builds bounded orientation for path-matching sessions with file columns.",
        result_semantics="bounded-context",
        projection_columns=("session_id", "path", "action_count", "last_seen_ms"),
        route="context-builder",
    ),
    _example(
        "context-session-mixed",
        'sessions where title:"query compiler" with messages(message_id,role), actions(tool_name,semantic_type), files(path)',
        "sessions",
        "Builds bounded mixed evidence for sessions whose title mentions the query compiler.",
        result_semantics="bounded-context",
        projection_columns=("session_id", "message_id", "role", "tool_name", "semantic_type", "path"),
        route="context-builder",
        featured=True,
    ),
    _example(
        "context-latest-assistant-messages",
        "messages where role:assistant | sort by time desc | limit 10",
        "messages",
        "Returns a bounded orientation slice of the ten latest assistant messages.",
        result_semantics="bounded-context",
        projection_columns=MESSAGE_COLUMNS,
        route="context-builder",
    ),
    _example(
        "context-latest-repository-messages",
        "sessions where repo:example-repo | messages where role:assistant | sort by time desc | limit 10",
        "messages",
        "Returns a bounded orientation slice of recent assistant messages for one repository.",
        result_semantics="bounded-context",
        projection_columns=MESSAGE_COLUMNS,
        route="context-builder",
    ),
    _example(
        "recursive-lineage-seed",
        "lineage:id:example-origin:session-child",
        "sessions",
        "Seeds a recursive lineage walk from one session.",
        result_semantics="recursive-page",
        projection_columns=RECURSIVE_COLUMNS,
        route="recursive-graph",
        featured=True,
    ),
    _example(
        "recursive-lineage-session-stage",
        "sessions where lineage:id:example-origin:session-child",
        "sessions",
        "Selects the session lineage used as a recursive graph seed.",
        result_semantics="recursive-page",
        projection_columns=RECURSIVE_COLUMNS,
        route="recursive-graph",
    ),
    _example(
        "recursive-lineage-messages",
        "sessions where lineage:id:example-origin:session-child | messages where role:user",
        "messages",
        "Pages user-message evidence across a recursively selected session lineage.",
        result_semantics="recursive-page",
        projection_columns=MESSAGE_COLUMNS + ("continuation",),
        route="recursive-graph",
    ),
    _example(
        "recursive-lineage-runs",
        "sessions where lineage:id:example-origin:session-child | runs where role:subagent",
        "runs",
        "Pages subagent-run evidence across a recursively selected session lineage.",
        result_semantics="recursive-page",
        projection_columns=RUN_COLUMNS + ("continuation",),
        route="recursive-graph",
    ),
    _example(
        "recursive-lineage-delegations",
        "sessions where lineage:id:example-origin:session-child | delegations where mapping_state:resolved",
        "delegations",
        "Pages resolved delegation evidence across a recursively selected session lineage.",
        result_semantics="recursive-page",
        projection_columns=DELEGATION_COLUMNS + ("continuation",),
        route="recursive-graph",
    ),
)


QUERY_DISCOVERY_NEGATIVE_EXAMPLES: tuple[QueryDiscoveryNegativeExample, ...] = (
    QueryDiscoveryNegativeExample(
        key="missing-terminal-conjunction-and-session-date-prefix",
        expression="actions where session.repo:example-repo since:7d AND output:failed",
        parser="unit",
        diagnostic_class="ExpressionCompileError",
        diagnostic="invalid query expression near column 27",
        corrected_form="actions where session.repo:example-repo AND session.since:7d AND output:failed",
        shipped_at=("polylogue/mcp/server_prompts.py:509",),
    ),
    QueryDiscoveryNegativeExample(
        key="unscoped-terminal-repository-field",
        expression="files where repo:example-repo AND path:src/mcp/server.py",
        parser="unit",
        diagnostic_class="ExpressionCompileError",
        diagnostic="field 'repo' is not supported for file predicates",
        field="repo",
        corrected_form="files where session.repo:example-repo AND path:src/mcp/server.py",
        shipped_at=("polylogue/mcp/server_prompts.py:524",),
    ),
    QueryDiscoveryNegativeExample(
        key="sessions-direct-count",
        expression="sessions where repo:example-repo | count",
        parser="unit",
        diagnostic_class="ExpressionCompileError",
        diagnostic=(
            "pipeline `count` cannot follow `sessions where ...` directly; `sessions` has no terminal row/aggregate "
            "lowerer of its own, only a session-scoping stage. To count matching sessions, use `find <predicate> then "
            "analyze --count` instead of a pipeline `| count`. To count/group/sort/limit rows of a specific terminal "
            "unit scoped to these sessions, pipe into an executable `<unit>s where ...` terminal stage first, e.g. "
            "`sessions where ... | actions where ... | group by FIELD | count`. Supported terminal units: "
            "messages/actions/blocks/assertions/files/runs/observed-events/context-snapshots/delegations."
        ),
        corrected_form="sessions where repo:example-repo | actions where is_error:true | count",
    ),
    QueryDiscoveryNegativeExample(
        key="empty-terminal-predicate",
        expression="messages where",
        parser="unit",
        diagnostic_class="ExpressionCompileError",
        diagnostic="messages where requires a predicate",
        corrected_form="messages where role:assistant",
    ),
    QueryDiscoveryNegativeExample(
        key="equals-instead-of-field-colon",
        expression="sessions where repo=example-repo",
        parser="session",
        diagnostic_class="ExpressionCompileError",
        diagnostic="invalid query expression near column 16",
        corrected_form="sessions where repo:example-repo",
    ),
    QueryDiscoveryNegativeExample(
        key="missing-group-by-keyword",
        expression="messages where role:assistant | group role | count",
        parser="unit",
        diagnostic_class="ExpressionCompileError",
        diagnostic=(
            "unsupported pipeline stage 'group role'; supported terminal stages are `sort by time|count|key "
            "[asc|desc]`, `group by FIELD`, `count`, `limit N`, and `offset N`"
        ),
        corrected_form="messages where role:assistant | group by role | count",
    ),
    QueryDiscoveryNegativeExample(
        key="missing-terminal-where",
        expression="sessions where repo:example-repo | messages role:assistant",
        parser="unit",
        diagnostic_class="ExpressionCompileError",
        diagnostic=(
            "pipeline terminal stage must be an executable `<unit>s where ...` query (got 'messages role:assistant'); "
            "supported terminal units: messages/actions/blocks/assertions/files/runs/observed-events/"
            "context-snapshots/delegations."
        ),
        corrected_form="sessions where repo:example-repo | messages where role:assistant",
    ),
    QueryDiscoveryNegativeExample(
        key="unsupported-run-aggregate",
        expression="runs where status:completed | group by status | count",
        parser="unit",
        diagnostic_class="ExpressionCompileError",
        diagnostic="pipeline `group by` is not supported for run rows; this terminal unit has no aggregate lowerer",
        field="group",
        corrected_form="runs where status:completed | sort by time desc | limit 10",
    ),
    QueryDiscoveryNegativeExample(
        key="unsupported-context-snapshot-count",
        expression="context-snapshots where boundary:session_start | count",
        parser="unit",
        diagnostic_class="ExpressionCompileError",
        diagnostic=(
            "pipeline `count` is not supported for context-snapshot rows; this terminal unit has no aggregate lowerer"
        ),
        field="count",
        corrected_form="context-snapshots where boundary:session_start | sort by time desc | limit 10",
    ),
    QueryDiscoveryNegativeExample(
        key="plural-structural-unit",
        expression="sessions where exists messages(role:assistant)",
        parser="session",
        diagnostic_class="ExpressionCompileError",
        diagnostic="invalid query expression near column 30",
        corrected_form="sessions where exists message(role:assistant)",
    ),
    QueryDiscoveryNegativeExample(
        key="unclosed-boolean-group",
        expression="sessions where (repo:example-repo",
        parser="session",
        diagnostic_class="ExpressionCompileError",
        diagnostic="invalid query expression near column 17",
        corrected_form="sessions where (repo:example-repo)",
    ),
    QueryDiscoveryNegativeExample(
        key="negated-ranked-leaf",
        expression='sessions where NOT near:"semantic search"',
        parser="session",
        diagnostic_class="ExpressionCompileError",
        diagnostic="field 'near' is not supported inside Boolean SQL predicates yet",
        field="near",
        corrected_form='sessions where semantic:"semantic search"',
    ),
    QueryDiscoveryNegativeExample(
        key="equals-on-string-field",
        expression="assertions where kind = decision",
        parser="unit",
        diagnostic_class="ExpressionCompileError",
        diagnostic="invalid query expression near column 1",
        corrected_form="assertions where kind:decision",
    ),
    QueryDiscoveryNegativeExample(
        key="missing-field-value",
        expression="files where path",
        parser="unit",
        diagnostic_class="ExpressionCompileError",
        diagnostic="invalid query expression near column 1",
        corrected_form="files where path:src/query/parser.py",
    ),
    QueryDiscoveryNegativeExample(
        key="session-only-unknown-field",
        expression="sessions where status:active",
        parser="session",
        diagnostic_class="ExpressionCompileError",
        diagnostic="field 'status' is not supported for session predicates",
        field="status",
        corrected_form="sessions where exists assertion(status:active)",
    ),
    QueryDiscoveryNegativeExample(
        key="terminal-session-field-without-prefix",
        expression="messages where repo:example-repo",
        parser="unit",
        diagnostic_class="ExpressionCompileError",
        diagnostic="field 'repo' is not supported for message predicates",
        field="repo",
        corrected_form="messages where session.repo:example-repo",
    ),
    QueryDiscoveryNegativeExample(
        key="non-integer-limit",
        expression="messages where role:assistant | limit many",
        parser="unit",
        diagnostic_class="ExpressionCompileError",
        diagnostic="pipeline `limit` stage requires an integer",
        field="limit",
        corrected_form="messages where role:assistant | limit 10",
    ),
    QueryDiscoveryNegativeExample(
        key="fts-column-query-at-strict-command-floor",
        expression="text:css {session_id example}: refactor",
        parser="session",
        diagnostic_class="ExpressionCompileError",
        diagnostic=(
            "unknown query field 'text'; recognized fields: action, assistant_messages, assistant_words, "
            "authored_user_messages, authored_user_words, contains, cwd, duration_ms, has, id, lane, lineage, "
            "messages, near, origin, paste_messages, path, project, repo, session, since, system_messages, tag, "
            "thinking_messages, title, tool, tool_messages, tool_use_messages, until, user_messages, user_words, words"
        ),
        field="text",
        corrected_form='contains:"css refactor"',
        shipped_at=("docs/search.md:924",),
    ),
)

_EXAMPLE_BY_KEY = {example.key: example for example in QUERY_DISCOVERY_EXAMPLES}


def query_discovery_example(key: str) -> QueryDiscoveryExample:
    """Return one declared example by stable key."""

    try:
        return _EXAMPLE_BY_KEY[key]
    except KeyError as exc:
        raise KeyError(f"unknown query discovery example: {key}") from exc


def query_discovery_examples(
    *,
    unit_source: str | None = None,
    result_semantics: QueryCoverageClass | None = None,
    featured: bool | None = None,
) -> tuple[QueryDiscoveryExample, ...]:
    """Filter the positive corpus without copying declaration authority."""

    rows = QUERY_DISCOVERY_EXAMPLES
    if unit_source is not None:
        rows = tuple(row for row in rows if row.unit_source == unit_source)
    if result_semantics is not None:
        rows = tuple(row for row in rows if row.result_semantics == result_semantics)
    if featured is not None:
        rows = tuple(row for row in rows if row.featured is featured)
    return rows


_SAFE_VALUE = re.compile(r"[A-Za-z0-9_./:@#-]+")
_SAFE_DATE = re.compile(r"[A-Za-z0-9_.:+-]+")


def _render_parameter(parameter: QueryExampleParameter, value: object) -> str:
    rendered = str(value)
    if not rendered or "\x00" in rendered:
        raise ValueError(f"query example parameter {parameter.name!r} must be non-empty and contain no NUL byte")
    if parameter.kind == "text":
        return json.dumps(rendered, ensure_ascii=False)
    if any(character in rendered for character in "\r\n"):
        raise ValueError(f"query example parameter {parameter.name!r} must be a single-line value")
    if parameter.kind == "date":
        if _SAFE_DATE.fullmatch(rendered) is not None:
            return rendered
        return json.dumps(rendered, ensure_ascii=False)
    if _SAFE_VALUE.fullmatch(rendered) is not None:
        return rendered
    return json.dumps(rendered, ensure_ascii=False)


def render_query_discovery_example(key: str, **values: object) -> str:
    """Render a parameterized example while preserving parser-valid quoting."""

    example = query_discovery_example(key)
    if example.template is None:
        if values:
            raise ValueError(f"query discovery example {key!r} has no parameters")
        return example.expression
    expected = {parameter.name for parameter in example.parameters}
    actual = set(values)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        detail = []
        if missing:
            detail.append(f"missing={','.join(missing)}")
        if unexpected:
            detail.append(f"unexpected={','.join(unexpected)}")
        raise ValueError(f"query discovery example {key!r} parameter mismatch: {'; '.join(detail)}")
    rendered = {
        parameter.name: _render_parameter(parameter, values[parameter.name]) for parameter in example.parameters
    }
    return example.template.format_map(rendered)


def _capability_examples() -> tuple[QueryDiscoveryExample, ...]:
    """Return one deterministic exemplar for every result-semantics class."""

    rows: list[QueryDiscoveryExample] = []
    seen: set[QueryCoverageClass] = set()
    for example in QUERY_DISCOVERY_EXAMPLES:
        if example.result_semantics in seen:
            continue
        rows.append(example)
        seen.add(example.result_semantics)
    return tuple(rows)


def _capability_example_payload(example: QueryDiscoveryExample) -> dict[str, object]:
    """Project the required corpus columns without template-only registry detail."""

    return {
        "key": example.key,
        "expression": example.expression,
        "unit_source": example.unit_source,
        "answers": example.answers,
        "result_semantics": example.result_semantics,
        "projection_columns": list(example.projection_columns),
        "cost_class": example.cost_class,
        "route": example.route,
    }


def query_discovery_catalog_payload(*, view: QueryDiscoveryCatalogView = "full") -> dict[str, object]:
    """Return a stable full, featured, or MCP-budgeted discovery catalog."""

    if view == "full":
        examples = QUERY_DISCOVERY_EXAMPLES
        negative_examples = QUERY_DISCOVERY_NEGATIVE_EXAMPLES
        example_payloads = [example.to_payload() for example in examples]
    elif view == "featured":
        examples = query_discovery_examples(featured=True)
        negative_examples = QUERY_DISCOVERY_NEGATIVE_EXAMPLES
        example_payloads = [example.to_payload() for example in examples]
    elif view == "capability":
        examples = _capability_examples()
        negative_examples = tuple(example for example in QUERY_DISCOVERY_NEGATIVE_EXAMPLES if example.shipped_at)
        example_payloads = [_capability_example_payload(example) for example in examples]
    else:
        raise ValueError(f"unknown query discovery catalog view: {view!r}")

    return {
        "version": 1,
        "view": view,
        "positive_count": len(QUERY_DISCOVERY_EXAMPLES),
        "negative_count": len(QUERY_DISCOVERY_NEGATIVE_EXAMPLES),
        "result_semantics": [
            {
                "class": contract.coverage,
                "coverage": contract.coverage,
                "total": contract.total,
                "continuation": contract.continuation,
                "teaching": contract.phrase,
            }
            for contract in RESULT_SEMANTICS_TEACHING
        ],
        "examples": example_payloads,
        "negative_examples": [example.to_payload() for example in negative_examples],
    }


def query_coverage_classes() -> tuple[QueryCoverageClass, ...]:
    """Return result classes in declaration order."""

    return cast(tuple[QueryCoverageClass, ...], tuple(contract.coverage for contract in RESULT_SEMANTICS_TEACHING))


__all__ = [
    "QUERY_DISCOVERY_EXAMPLES",
    "QUERY_DISCOVERY_GRAMMAR",
    "QUERY_DISCOVERY_NEGATIVE_EXAMPLES",
    "RESULT_SEMANTICS_TEACHING",
    "QueryDiscoveryCatalogView",
    "QueryDiscoveryCostClass",
    "QueryDiscoveryExample",
    "QueryDiscoveryNegativeExample",
    "QueryDiscoveryParser",
    "QueryDiscoveryRoute",
    "QueryDiscoveryUnitSource",
    "QueryExampleParameter",
    "query_coverage_classes",
    "query_discovery_catalog_payload",
    "query_discovery_example",
    "query_discovery_examples",
    "render_query_discovery_example",
    "result_semantics_teaching",
]
