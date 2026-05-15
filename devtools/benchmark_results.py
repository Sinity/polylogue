"""Shared pytest-benchmark result parsing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from polylogue.core.json import JSONDocument, json_document, json_document_list


class BenchmarkStatRecord(TypedDict):
    name: str
    fullname: str
    group: str
    mean: float
    median: float
    minimum: float
    maximum: float
    stddev: float
    rounds: int
    ops: float | None


@dataclass(frozen=True)
class BenchmarkStat:
    name: str
    fullname: str
    group: str
    mean: float
    median: float
    minimum: float
    maximum: float
    stddev: float
    rounds: int
    ops: float | None


def benchmark_key(entry: JSONDocument) -> str:
    """Return the most stable pytest-benchmark identity for an entry."""
    return str(entry.get("fullname") or entry.get("fullfunc") or entry.get("name"))


def parse_benchmark_stat(raw: JSONDocument) -> BenchmarkStat:
    """Parse one pytest-benchmark JSON entry."""
    stats = json_document(raw.get("stats"))
    mean = _float_field(stats, "mean")
    raw_group = raw.get("group")
    return BenchmarkStat(
        name=str(raw.get("name", "unknown")),
        fullname=benchmark_key(raw),
        group=str(raw_group) if raw_group is not None else "benchmark",
        mean=mean,
        median=_float_field(stats, "median"),
        minimum=_float_field(stats, "min"),
        maximum=_float_field(stats, "max"),
        stddev=_float_field(stats, "stddev") if "stddev" in stats else 0.0,
        rounds=_int_field(stats, "rounds") if "rounds" in stats else 0,
        ops=(1.0 / mean) if mean > 0 else None,
    )


def parse_pytest_benchmark_stats(payload: object) -> list[BenchmarkStat]:
    """Parse all benchmark entries from a pytest-benchmark JSON payload."""
    document = json_document(payload)
    return [parse_benchmark_stat(entry) for entry in json_document_list(document.get("benchmarks"))]


def benchmark_stat_record(stat: BenchmarkStat) -> BenchmarkStatRecord:
    """Convert a parsed benchmark stat to a JSON-serializable campaign record."""
    return {
        "name": stat.name,
        "fullname": stat.fullname,
        "group": stat.group,
        "mean": stat.mean,
        "median": stat.median,
        "minimum": stat.minimum,
        "maximum": stat.maximum,
        "stddev": stat.stddev,
        "rounds": stat.rounds,
        "ops": stat.ops,
    }


def _number_value(value: object, *, field: str) -> str | int | float:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(f"Benchmark payload field {field!r} is not numeric: {value!r}")
    return value


def _number_field(payload: JSONDocument, field: str) -> str | int | float:
    return _number_value(payload[field], field=field)


def _float_field(payload: JSONDocument, field: str) -> float:
    return float(_number_field(payload, field))


def _int_field(payload: JSONDocument, field: str) -> int:
    return int(_number_field(payload, field))


__all__ = [
    "BenchmarkStat",
    "BenchmarkStatRecord",
    "benchmark_key",
    "benchmark_stat_record",
    "parse_benchmark_stat",
    "parse_pytest_benchmark_stats",
]
