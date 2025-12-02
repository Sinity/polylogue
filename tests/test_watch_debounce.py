from pathlib import Path

from polylogue.cli.watch import _run_watch_sessions
from polylogue.commands import CommandEnv
from polylogue.ui import create_ui


class DummyProvider:
    name = "dummy"
    title = "Dummy"
    watch_banner = ""
    watch_log_title = "Dummy Watch"
    watch_suffixes = None
    supports_watch = True
    default_base: Path
    default_output: Path

    def __init__(self, base: Path, output: Path):
        self.default_base = base
        self.default_output = output

    def sync_fn(self, **_kwargs):
        # simple stub result with required attributes
        return type(
            "Result",
            (),
            {
                "written": [],
                "skipped": [],
                "pruned": 0,
                "output_dir": self.default_output,
                "attachments": 0,
                "attachment_bytes": 0,
                "tokens": 0,
                "words": 0,
                "diffs": 0,
                "duration": 0.0,
            },
        )


def test_watch_debounce_accumulates(tmp_path: Path):
    ui = create_ui(plain=True)
    env = CommandEnv(ui=ui)
    base = tmp_path / "base"
    out = tmp_path / "out"
    provider = DummyProvider(base, out)

    # just ensure it doesn't raise when called with once flag (no watch loop)
    class Args:
        provider = None
        base_dir = None
        out = None
        debounce = 0.5
        once = True
        collapse_threshold = 25

    _run_watch_sessions(Args(), env, provider)
