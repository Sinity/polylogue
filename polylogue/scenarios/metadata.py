"""Shared scenario metadata used across compiled verification artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from polylogue.insights.authored_payloads import (
    PayloadDict,
    merge_unique_string_tuples,
    payload_string,
    payload_string_tuple,
)


def _object_string_attribute(obj: object, name: str, default: str) -> str:
    value = getattr(obj, name, None)
    if isinstance(value, str):
        return value
    return default


def _object_string_tuple_attribute(obj: object, name: str) -> tuple[str, ...]:
    value = getattr(obj, name, None)
    return payload_string_tuple(value) if isinstance(value, list | tuple) else ()


def _object_int_attribute(obj: object, name: str) -> int | None:
    value = getattr(obj, name, None)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _payload_int(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _first_present_string(current: str, others: tuple[ScenarioMetadata, ...], name: str) -> str:
    if current:
        return current
    for other in others:
        value = getattr(other, name)
        if isinstance(value, str) and value:
            return value
    return ""


def _first_present_int(current: int | None, others: tuple[ScenarioMetadata, ...], name: str) -> int | None:
    if current is not None:
        return current
    for other in others:
        value = getattr(other, name)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


@dataclass(frozen=True, kw_only=True)
class ScenarioMetadata:
    """Portable scenario metadata shared by checks, campaigns, and reports."""

    origin: str = "authored"
    tags: tuple[str, ...] = ()
    docs_role: str = ""
    caption: str = ""
    narrative_order: int | None = None
    audience: tuple[str, ...] = ()
    demonstrates: tuple[str, ...] = ()
    privacy_level: str = ""
    media: tuple[str, ...] = ()
    visual_style: str = ""

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> ScenarioMetadata:
        return cls(
            origin=payload_string(payload.get("origin"), "authored"),
            tags=payload_string_tuple(payload.get("tags")),
            docs_role=payload_string(payload.get("docs_role"), ""),
            caption=payload_string(payload.get("caption"), ""),
            narrative_order=_payload_int(payload.get("narrative_order")),
            audience=payload_string_tuple(payload.get("audience")),
            demonstrates=payload_string_tuple(payload.get("demonstrates")),
            privacy_level=payload_string(payload.get("privacy_level"), ""),
            media=payload_string_tuple(payload.get("media")),
            visual_style=payload_string(payload.get("visual_style"), ""),
        )

    @classmethod
    def from_object(cls, obj: object) -> ScenarioMetadata:
        return cls(
            origin=_object_string_attribute(obj, "origin", "authored"),
            tags=_object_string_tuple_attribute(obj, "tags"),
            docs_role=_object_string_attribute(obj, "docs_role", ""),
            caption=_object_string_attribute(obj, "caption", ""),
            narrative_order=_object_int_attribute(obj, "narrative_order"),
            audience=_object_string_tuple_attribute(obj, "audience"),
            demonstrates=_object_string_tuple_attribute(obj, "demonstrates"),
            privacy_level=_object_string_attribute(obj, "privacy_level", ""),
            media=_object_string_tuple_attribute(obj, "media"),
            visual_style=_object_string_attribute(obj, "visual_style", ""),
        )

    def to_payload(self) -> PayloadDict:
        payload: PayloadDict = {"origin": self.origin}
        if self.tags:
            payload["tags"] = list(self.tags)
        if self.docs_role:
            payload["docs_role"] = self.docs_role
        if self.caption:
            payload["caption"] = self.caption
        if self.narrative_order is not None:
            payload["narrative_order"] = self.narrative_order
        if self.audience:
            payload["audience"] = list(self.audience)
        if self.demonstrates:
            payload["demonstrates"] = list(self.demonstrates)
        if self.privacy_level:
            payload["privacy_level"] = self.privacy_level
        if self.media:
            payload["media"] = list(self.media)
        if self.visual_style:
            payload["visual_style"] = self.visual_style
        return payload

    def merged(self, *others: ScenarioMetadata) -> ScenarioMetadata:
        if not others:
            return self
        return ScenarioMetadata(
            origin=self.origin,
            tags=merge_unique_string_tuples(self.tags, *(other.tags for other in others)),
            docs_role=_first_present_string(self.docs_role, others, "docs_role"),
            caption=_first_present_string(self.caption, others, "caption"),
            narrative_order=_first_present_int(self.narrative_order, others, "narrative_order"),
            audience=merge_unique_string_tuples(self.audience, *(other.audience for other in others)),
            demonstrates=merge_unique_string_tuples(self.demonstrates, *(other.demonstrates for other in others)),
            privacy_level=_first_present_string(self.privacy_level, others, "privacy_level"),
            media=merge_unique_string_tuples(self.media, *(other.media for other in others)),
            visual_style=_first_present_string(self.visual_style, others, "visual_style"),
        )

    def with_defaults(self, defaults: ScenarioMetadata) -> ScenarioMetadata:
        """Fill missing presentation metadata from an execution default."""

        return ScenarioMetadata(
            origin=self.origin,
            tags=merge_unique_string_tuples(self.tags, defaults.tags),
            docs_role=self.docs_role or defaults.docs_role,
            caption=self.caption or defaults.caption,
            narrative_order=self.narrative_order if self.narrative_order is not None else defaults.narrative_order,
            audience=self.audience or defaults.audience,
            demonstrates=self.demonstrates or defaults.demonstrates,
            privacy_level=self.privacy_level or defaults.privacy_level,
            media=self.media or defaults.media,
            visual_style=self.visual_style or defaults.visual_style,
        )


__all__ = ["ScenarioMetadata"]
