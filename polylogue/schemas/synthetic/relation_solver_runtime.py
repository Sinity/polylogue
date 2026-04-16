"""Runtime behavior mixin for synthetic relation solving."""

from __future__ import annotations


class RelationConstraintSolverRuntimeMixin:
    def _parse_foreign_keys(self, schema: dict[str, object]) -> None:
        for fk in schema.get("x-polylogue-foreign-keys", []):
            source = fk.get("source", "")
            target = fk.get("target", "")
            if source and target:
                self.fk_graph.references[source] = target

    def _parse_time_deltas(self, schema: dict[str, object]) -> None:
        for td in schema.get("x-polylogue-time-deltas", []):
            self.time_deltas.append(
                self._time_delta_cls(
                    field_a=td.get("field_a", ""),
                    field_b=td.get("field_b", ""),
                    min_delta=td.get("min_delta", 0.0),
                    max_delta=td.get("max_delta", 0.0),
                    avg_delta=td.get("avg_delta", 0.0),
                )
            )

    def _parse_mutual_exclusions(self, schema: dict[str, object]) -> None:
        for me in schema.get("x-polylogue-mutually-exclusive", []):
            parent = me.get("parent", "")
            fields = me.get("fields", [])
            if parent and len(fields) >= 2:
                self.mutual_exclusions.append(
                    self._mutual_exclusion_cls(
                        parent_path=parent,
                        field_names=frozenset(fields),
                    )
                )

    def _parse_string_lengths(self, schema: dict[str, object]) -> None:
        for sl in schema.get("x-polylogue-string-lengths", []):
            path = sl.get("path", "")
            if path:
                self.string_lengths[path] = self._string_length_cls(
                    path=path,
                    min_length=sl.get("min", 0),
                    max_length=sl.get("max", 100),
                    avg_length=sl.get("avg", 50.0),
                    stddev=sl.get("stddev", 10.0),
                )

    def register_generated_id(self, path: str, value: str) -> None:
        self.fk_graph.register_id(path, value)

    def resolve_foreign_key(self, path: str, rng) -> str | None:
        return self.fk_graph.resolve_reference(path, rng)

    def get_time_delta(
        self,
        field_a: str,
        field_b: str,
        rng,
    ) -> float | None:
        for td in self.time_deltas:
            if (td.field_a == field_a and td.field_b == field_b) or (td.field_a == field_b and td.field_b == field_a):
                if td.stddev_approx > 0:
                    val = rng.gauss(td.avg_delta, td.stddev_approx)
                else:
                    val = rng.uniform(td.min_delta, td.max_delta)
                return max(td.min_delta, min(td.max_delta, val))
        return None

    def filter_mutually_exclusive(
        self,
        parent_path: str,
        field_names: set[str],
        rng,
    ) -> set[str]:
        result = set(field_names)
        for group in self.mutual_exclusions:
            if group.parent_path != parent_path:
                continue
            overlap = result & group.field_names
            if len(overlap) > 1:
                keeper = rng.choice(sorted(overlap))
                result -= overlap
                result.add(keeper)
        return result

    def generate_string_with_length(
        self,
        path: str,
        rng,
        base_text: str,
    ) -> str:
        constraint = self.string_lengths.get(path)
        if constraint is None:
            return base_text

        target = int(rng.gauss(constraint.avg_length, constraint.stddev))
        target = max(constraint.min_length, min(constraint.max_length, target))

        if len(base_text) == 0:
            return base_text

        if len(base_text) >= target:
            if target <= 3:
                return base_text[:target]
            truncated = base_text[:target]
            last_space = truncated.rfind(" ")
            if last_space > target // 2:
                return truncated[:last_space]
            return truncated
        repetitions = (target // len(base_text)) + 1
        extended = (base_text + " ") * repetitions
        return extended[:target].rstrip()

    def path_matches(self, schema_path: str, annotation_path: str) -> bool:
        return schema_path == annotation_path


__all__ = ["RelationConstraintSolverRuntimeMixin"]
