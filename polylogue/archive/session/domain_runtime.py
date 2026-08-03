"""Runtime behavior helpers for ``Session`` models."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping
from typing import TYPE_CHECKING, Self, cast

from polylogue.archive.message.messages import MessageCollection
from polylogue.archive.message.models import DialoguePair, Message
from polylogue.archive.message.roles import Role, normalize_message_roles
from polylogue.archive.session.display_mixin import DisplayTitleTagsMixin
from polylogue.core.enums import MaterialOrigin

if TYPE_CHECKING:
    from polylogue.archive.projection.projections import SessionProjection
    from polylogue.archive.semantic.content_projection import ContentProjectionSpec
    from polylogue.archive.session.domain_models import Session


class SessionRuntimeMixin(DisplayTitleTagsMixin):
    messages: MessageCollection

    if TYPE_CHECKING:

        def model_copy(self, *, update: Mapping[str, object] | None = None, deep: bool = False) -> Self: ...

    def filter(self, predicate: Callable[[Message], bool]) -> Self:
        filtered_messages = [message for message in self.messages if predicate(message)]
        return self.model_copy(update={"messages": MessageCollection(messages=filtered_messages)})

    def with_roles(self, roles: object) -> Self:
        selected_roles = normalize_message_roles(roles)
        return self.filter(lambda message: message.role in selected_roles)

    def with_material_origins(self, origins: object) -> Self:
        raw_origins = origins if isinstance(origins, (tuple, list, set, frozenset)) else (origins,)
        selected_origins = tuple(MaterialOrigin.validate_filter_token(origin) for origin in raw_origins)
        return self.filter(lambda message: message.material_origin in selected_origins)

    def with_content_projection(
        self,
        projection: ContentProjectionSpec | Mapping[str, object] | None,
    ) -> Self:
        from polylogue.archive.semantic.content_projection import project_session_content

        return cast(Self, project_session_content(cast("Session", self), projection))

    def authored_dialogue(self) -> Self:
        return self.with_roles((Role.USER, Role.ASSISTANT))

    def without_noise(self) -> Self:
        return self.filter(lambda message: not message.is_noise)

    def substantive_only(self) -> Self:
        return self.filter(lambda message: message.is_substantive)

    def mainline_messages(self) -> list[Message]:
        """Return the currently-displayed conversation, one message per branch point.

        ``is_active_path`` (not ``branch_index``) is the display-state signal
        (polylogue-9qq7): ``branch_index`` is creation order, so for an
        edited/regenerated turn ``branch_index == 0`` is the FIRST attempt,
        not necessarily the accepted sibling. When ``is_active_path`` is
        unknown (``None`` -- e.g. a provider with no branch concept, or a read
        path that hasn't resolved it), fall back to the historical
        ``branch_index == 0`` rule rather than hiding the message: unknown
        must never collapse to "not active".
        """
        return [
            message
            for message in self.messages
            if (message.is_active_path if message.is_active_path is not None else message.branch_index == 0)
        ]

    def iter_dialogue(self) -> Iterator[Message]:
        for message in self.messages:
            if message.is_dialogue:
                yield message

    def iter_substantive(self) -> Iterator[Message]:
        for message in self.messages:
            if message.is_substantive:
                yield message

    def iter_pairs(self) -> Iterator[DialoguePair]:
        substantive_messages = [message for message in self.messages if message.is_substantive]
        index = 0
        while index < len(substantive_messages) - 1:
            current = substantive_messages[index]
            next_message = substantive_messages[index + 1]
            if current.is_user and next_message.is_assistant:
                yield DialoguePair(user=current, assistant=next_message)
                index += 2
            else:
                index += 1

    def iter_thinking(self) -> Iterator[str]:
        for message in self.messages:
            if not message.is_thinking:
                continue
            thinking = message.extract_thinking()
            if thinking:
                yield thinking

    def iter_branches(self) -> Iterator[tuple[str, list[Message]]]:
        by_parent: dict[str, list[Message]] = defaultdict(list)
        for message in self.messages:
            if message.parent_id:
                by_parent[message.parent_id].append(message)

        for parent_id, children in by_parent.items():
            if len(children) > 1:
                yield parent_id, sorted(children, key=lambda message: message.branch_index)

    def to_text(self, include_role: bool = True) -> str:
        lines: list[str] = []
        for message in self.messages:
            if not message.text:
                continue
            lines.append(f"{message.role}: {message.text}" if include_role else message.text)
        return "\n\n".join(lines)

    def to_clean_text(self) -> str:
        return self.substantive_only().to_text()

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def user_message_count(self) -> int:
        return sum(1 for message in self.messages if message.is_user)

    @property
    def assistant_message_count(self) -> int:
        return sum(1 for message in self.messages if message.is_assistant)

    @property
    def word_count(self) -> int:
        return sum(message.word_count for message in self.messages)

    @property
    def total_cost_usd(self) -> float:
        """Sum of per-message ``cost_usd`` values.

        Message-level cost/duration is sourced through typed message and
        insight projections (``polylogue.archive.semantic.pricing``) per
        #803/#1139. This property is retained for callers that still expect
        a scalar on the hydrated session and always returns ``0.0``;
        downstream readers consume ``CostEstimatePayload`` from the typed
        insight layer instead.
        """

        return 0.0

    @property
    def total_duration_ms(self) -> int:
        """Total duration belongs to typed insight/session projections."""
        return 0

    def project(self) -> SessionProjection:
        from polylogue.archive.projection.projections import SessionProjection
        from polylogue.archive.session.domain_models import Session

        if not isinstance(self, Session):
            raise TypeError(f"projection requires Session, got {type(self).__name__}")
        return SessionProjection(self)


__all__ = ["SessionRuntimeMixin"]
