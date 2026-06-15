"""Recovery digest transform benchmark tests.

Run with:
    pytest tests/benchmarks/test_recovery_digest.py --benchmark-enable -p no:xdist -v
"""

from __future__ import annotations

import pytest

from polylogue.archive.message.messages import MessageCollection
from polylogue.archive.message.models import Message
from polylogue.archive.message.roles import Role
from polylogue.archive.session.domain_models import Session
from polylogue.core.enums import Origin
from polylogue.insights.transforms import compile_recovery_digest
from polylogue.types import SessionId
from tests.benchmarks.helpers import BenchmarkFixture


def _session(message_count: int) -> Session:
    messages: list[Message] = []
    for index in range(message_count):
        messages.append(
            Message(
                id=f"m{index}",
                role=Role.USER if index % 4 == 0 else Role.ASSISTANT,
                text=(
                    f"Decision: keep recovery digest benchmark fixture {index} deterministic.\n"
                    f"Done:\n- PR #{1900 + index} merged\n"
                    "Blockers:\n- none\n"
                    f"Next: verify artifact #{index}"
                ),
                blocks=[
                    {
                        "type": "tool_use",
                        "id": f"tool-{index}",
                        "name": "Bash",
                        "tool_input": {"command": f"devtools verify --quick --slice {index}"},
                    },
                    {
                        "type": "tool_result",
                        "tool_id": f"tool-{index}",
                        "text": (
                            "ruff check ... ok\n"
                            f"{20 + index % 7} passed in {0.7 + (index % 5) / 10:.2f}s\n"
                            f"https://github.com/Sinity/polylogue/pull/{1900 + index}"
                        ),
                    },
                    {
                        "type": "tool_use",
                        "id": f"agent-{index}",
                        "name": "Task",
                        "tool_input": {
                            "subagent_type": "Explore",
                            "prompt": f"Map recovery evidence lane {index}.",
                        },
                    },
                    {
                        "type": "tool_result",
                        "tool_id": f"agent-{index}",
                        "text": f"Subagent done: evidence lane {index} is bounded.",
                    },
                ],
            )
        )

    return Session(
        id=SessionId("codex-session:recovery-benchmark"),
        origin=Origin.CODEX_SESSION,
        title="Recovery digest benchmark",
        git_branch="feature/test/recovery-digest-benchmark",
        working_directories=("/realm/project/polylogue",),
        messages=MessageCollection(messages=messages),
    )


@pytest.mark.benchmark
@pytest.mark.parametrize("message_count", [12, 120])
def test_bench_compile_recovery_digest(benchmark: BenchmarkFixture, message_count: int) -> None:
    """compile_recovery_digest() over tool-heavy session transcripts."""
    session = _session(message_count)

    digest = benchmark(lambda: compile_recovery_digest(session))

    assert digest.size_metrics.message_count == message_count
    assert digest.size_metrics.tool_summary_count == message_count
    assert digest.size_metrics.subagent_report_count == message_count
    assert digest.transform.transform_id == "recovery_digest_v0"
