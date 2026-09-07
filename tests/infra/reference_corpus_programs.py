"""Generated corpus programs and requests for the reference-model differential.

A corpus program is a deterministic, seed-addressed description of an archive:
sessions, messages, blocks, tags and lineage edges, drawn from a closed
vocabulary the reference model declares.  The same program admits to a real
archive through the production writer and answers from arithmetic, so a
differential compares two independent computations of the same declared facts.

Determinism is by construction: every draw comes from one
:class:`random.Random` seeded by the program's own seed.  A failing seed is
the whole reproduction — record it and re-run.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from tests.infra.reference_model import (
    ModelBlock,
    ModelCorpus,
    ModelMessage,
    ModelRequest,
    ModelRole,
    ModelSession,
    assert_model_vocabulary,
)

#: Provider tokens and the public origin each maps to.  Stated as a pair
#: because the mapping is not reversible in general and the model must not
#: guess it (``docs/provider-origin-identity.md``).
PROVIDER_ORIGINS: tuple[tuple[str, str], ...] = (
    ("claude-code", "claude-code-session"),
    ("codex", "codex-session"),
    ("gemini-cli", "gemini-cli-session"),
)

#: ASCII word tokens the generators draw message text from.  Distinct stems
#: keep a term from matching a session by accident, which is what makes a
#: text-filter differential meaningful rather than trivially total.
VOCABULARY: tuple[str, ...] = (
    "widget",
    "gadget",
    "sprocket",
    "flange",
    "grommet",
    "ratchet",
    "spindle",
    "trunnion",
)

#: The tokens :func:`generate_requests` builds ``contains:`` clauses from.
#: A text perturbation has to move one of these or no request observes it.
QUERY_TOKENS: tuple[str, ...] = ("widget", "gadget", "sprocket")

TOOL_NAMES: tuple[str, ...] = ("Bash", "Read", "Edit")
SEMANTIC_TYPES: tuple[str, ...] = ("shell", "file_read", "file_edit")
TAGS: tuple[str, ...] = ("review", "archive", "followup")

ROLES: tuple[ModelRole, ...] = ("user", "assistant", "system", "tool")

_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


def _timestamp(offset_minutes: int) -> str:
    return (_EPOCH + timedelta(minutes=offset_minutes)).isoformat()


def _text(rng: random.Random) -> str:
    return " ".join(rng.sample(VOCABULARY, k=rng.randint(1, 4)))


def _blocks(rng: random.Random, role: ModelRole, text: str) -> tuple[ModelBlock, ...]:
    blocks: list[ModelBlock] = [ModelBlock(type="text", text=text)]
    if role == "assistant" and rng.random() < 0.5:
        blocks.append(ModelBlock(type="thinking", text=_text(rng)))
    if role == "assistant" and rng.random() < 0.5:
        index = rng.randrange(len(TOOL_NAMES))
        blocks.append(
            ModelBlock(
                type="tool_use",
                tool_name=TOOL_NAMES[index],
                tool_id=f"tool-{rng.randrange(1000)}",
                semantic_type=SEMANTIC_TYPES[index],
            )
        )
    if role == "tool":
        blocks.append(ModelBlock(type="tool_result", text=text, tool_name=rng.choice(TOOL_NAMES)))
    return tuple(blocks)


def _session(rng: random.Random, index: int, *, parent_key: str | None) -> ModelSession:
    provider, origin = rng.choice(PROVIDER_ORIGINS)
    message_count = rng.randint(1, 5)
    messages: list[ModelMessage] = []
    for position in range(message_count):
        role: ModelRole = "user" if position == 0 else rng.choice(ROLES)
        text = _text(rng)
        messages.append(
            ModelMessage(
                message_id=f"m{position + 1}",
                role=role,
                text=text,
                timestamp=_timestamp(index * 60 + position),
                blocks=_blocks(rng, role, text),
                # Only user-role messages are ever human-authored; the counter
                # is independent of role in production, so the generator gives
                # the differential a case where the two disagree.
                material_origin="human_authored" if role == "user" and rng.random() < 0.7 else None,
            )
        )
    tags = tuple(sorted(rng.sample(TAGS, k=rng.randint(0, 2))))
    created = _timestamp(index * 60)
    return ModelSession(
        key=f"s{index:02d}",
        origin=origin,
        provider=provider,
        title=f"Session {index} {rng.choice(VOCABULARY)}",
        created_at=created,
        updated_at=_timestamp(index * 60 + message_count),
        messages=tuple(messages),
        tags=tags,
        parent_key=parent_key,
    )


def generate_corpus(seed: int, *, session_count: int = 8, with_lineage: bool = False) -> ModelCorpus:
    """Return the corpus program addressed by ``seed``.

    ``with_lineage`` links a later session to an earlier one.  Lineage edges
    are a parent reference only: the generator never declares an inherited
    prefix, because a declared prefix is a parser assertion and this program
    admits through the writer, not a parser.
    """
    rng = random.Random(seed)
    sessions: list[ModelSession] = []
    for index in range(session_count):
        parent_key: str | None = None
        if with_lineage and index > 0 and rng.random() < 0.4:
            parent_key = sessions[rng.randrange(len(sessions))].key
        sessions.append(_session(rng, index, parent_key=parent_key))
    corpus = ModelCorpus(sessions=tuple(sessions))
    assert_model_vocabulary(corpus)
    return corpus


def is_executable(expression: str) -> bool:
    """Whether production both parses and lowers ``expression``.

    Not every clause the compact grammar accepts is supported inside a
    Boolean predicate (``contains:`` is one), so a generator that emitted
    every combination would spend its budget on expressions the product
    refuses.  Asking production settles it without a second vocabulary table
    here that could drift from the compiler.
    """
    from polylogue.archive.query.expression import ExpressionCompileError, compile_expression

    try:
        compile_expression(expression)
    except ExpressionCompileError:
        return False
    return True


def _uses_text_terms(expression: str) -> bool:
    """Whether the compiled request reaches the ranked-search route."""
    from polylogue.archive.query.expression import compile_expression

    spec = compile_expression(expression)
    return bool(spec.query_terms or spec.contains_terms)


def generate_requests(corpus: ModelCorpus, seed: int, *, count: int | None = None) -> tuple[ModelRequest, ...]:
    """Return executable requests drawn from the model's declared vocabulary.

    Coverage is systematic, not sampled: the clause vocabulary is shuffled by
    the seed and then *every* clause is asked in the compact spelling, in the
    Boolean spelling, and under one connective.  A lowering that is right for
    ``tag:x`` and wrong for ``sessions where tag:x`` — or right for ``AND``
    and wrong for ``AND NOT`` — cannot then survive by never being drawn, and
    a perturbation of the corpus cannot go unobserved because the clause that
    would have shown it was left out.  ``count`` bounds the set for a caller
    that wants a cheaper run; the default asks everything.
    """
    rng = random.Random(seed ^ 0x5EED)
    sessions = tuple(corpus)
    origins = sorted({session.origin for session in sessions})
    tags = sorted({tag for session in sessions for tag in session.tags})
    tools = sorted({name for session in sessions for name in session.tool_names})

    clauses: list[str] = [
        *(f"origin:{origin}" for origin in origins),
        *(f"tag:{tag}" for tag in tags),
        *(f"tool:{name}" for name in tools),
        *(f"contains:{token}" for token in QUERY_TOKENS),
        "has:thinking",
        "has:tool_use",
        "messages:>=2",
        "messages:<=2",
        "words:>=6",
        "user_messages:>=1",
        "assistant_messages:>=1",
        "system_messages:=0",
        "tool_use_messages:>=1",
        "thinking_messages:>=1",
        "paste_messages:=0",
        "authored_user_messages:>=1",
        "authored_user_words:>=3",
        "assistant_words:>=1",
        "user_words:>=2",
        f"id:{sessions[0].session_id}",
        f"title:{sessions[0].title.split()[-1] if sessions[0].title else 'Session'}",
        "since:2026-01-01T00:00:00+00:00",
        "until:2026-12-31T00:00:00+00:00",
    ]
    rng.shuffle(clauses)

    #: The connective each clause is additionally asked under, rotating so a
    #: run covers all three rather than whichever a draw happened to pick.
    connectives = ("OR", "AND", "AND NOT")

    requests: list[ModelRequest] = []
    for index, clause in enumerate(clauses):
        other = clauses[(index + 1) % len(clauses)]
        spellings = (
            clause,
            f"sessions where {clause}",
            f"sessions where {clause} {connectives[index % len(connectives)]} {other}",
        )
        for expression in spellings:
            if count is not None and len(requests) >= count:
                return tuple(requests)
            if not is_executable(expression):
                continue
            # A window is only cut for structured requests.  A request
            # carrying a text term is answered on the ranked-search route,
            # whose page order is relevance, and asking the model to predict a
            # relevance-ordered page would test the ranker, not the filter.
            paginate = not _uses_text_terms(expression)
            requests.append(
                ModelRequest(
                    name=f"r{len(requests):02d}",
                    expression=expression,
                    limit=rng.choice((None, 2, 3)) if paginate else None,
                    offset=rng.choice((0, 0, 1)) if paginate else 0,
                )
            )
    if not requests:
        raise RuntimeError(f"generated no executable requests for seed {seed}")
    return tuple(requests)


def divergent_corpus(corpus: ModelCorpus, *, kind: str) -> ModelCorpus:
    """Return ``corpus`` perturbed so the model no longer describes the archive.

    This is the differential's anti-vacuity control: seeding the model with a
    corpus that is semantically one step away from what the archive holds must
    make the comparison fail.  A differential that stays green under every
    perturbation is comparing nothing.
    """
    sessions = list(corpus)
    if kind == "drop-message":
        # Down to a single message, not merely one fewer: the perturbation has
        # to cross a threshold a generated clause asks about, or the model and
        # the archive would still agree on every request and the control would
        # prove nothing.
        target = next(session for session in sessions if len(session.messages) > 1)
        replacement = ModelSession(
            **{
                **{key: getattr(target, key) for key in _SESSION_FIELDS},
                "messages": target.messages[:1],
            }
        )
    elif kind == "swap-origin":
        target = sessions[0]
        other = next(origin for _, origin in PROVIDER_ORIGINS if origin != target.origin)
        replacement = ModelSession(**{**{key: getattr(target, key) for key in _SESSION_FIELDS}, "origin": other})
    elif kind == "retag":
        # A tagged session, so the declared tag a generated clause asks about
        # is the one that disappears.
        target = next(session for session in sessions if session.tags)
        replacement = ModelSession(
            **{**{key: getattr(target, key) for key in _SESSION_FIELDS}, "tags": ("nonexistent-tag",)}
        )
    elif kind == "reword":
        # A token the corpus queries is added to a session that does not carry
        # it, so the model claims a text match the archive cannot make.  Both
        # the message text and its block are reworded, because production
        # counts the message and indexes the block from the same content.
        token, target = _absent_query_token(sessions)
        head = target.messages[0]
        reworded = ModelMessage(
            message_id=head.message_id,
            role=head.role,
            text=f"{head.text} {token}",
            timestamp=head.timestamp,
            blocks=tuple(
                ModelBlock(
                    type=block.type,
                    text=f"{block.text} {token}",
                    tool_name=block.tool_name,
                    tool_id=block.tool_id,
                    semantic_type=block.semantic_type,
                )
                if block.type == "text" and block.text is not None
                else block
                for block in head.blocks
            ),
            material_origin=head.material_origin,
        )
        replacement = ModelSession(
            **{
                **{key: getattr(target, key) for key in _SESSION_FIELDS},
                "messages": (reworded, *target.messages[1:]),
            }
        )
    else:
        raise ValueError(f"unknown divergence kind: {kind!r}")
    return ModelCorpus(tuple(replacement if session.key == target.key else session for session in sessions))


def _absent_query_token(sessions: Sequence[ModelSession]) -> tuple[str, ModelSession]:
    """A queried token and a session whose messages do not carry it."""
    for token in QUERY_TOKENS:
        for session in sessions:
            # ``search_tokens`` is the model's own text-match set: a token
            # present only in a thinking block would already match, and adding
            # it to the message text would change nothing.
            if session.messages and token not in session.search_tokens:
                return token, session
    raise RuntimeError("every session carries every queried token; the corpus cannot be perturbed by text")


_SESSION_FIELDS: Sequence[str] = (
    "key",
    "origin",
    "provider",
    "title",
    "created_at",
    "updated_at",
    "messages",
    "tags",
    "parent_key",
    "git_repository_url",
    "provider_project_ref",
    "working_directories",
)

#: The perturbations :func:`divergent_corpus` knows how to apply.
DIVERGENCE_KINDS: tuple[str, ...] = ("drop-message", "swap-origin", "retag", "reword")


__all__ = [
    "DIVERGENCE_KINDS",
    "PROVIDER_ORIGINS",
    "QUERY_TOKENS",
    "SEMANTIC_TYPES",
    "TAGS",
    "TOOL_NAMES",
    "VOCABULARY",
    "divergent_corpus",
    "generate_corpus",
    "generate_requests",
]
