from types import SimpleNamespace

from polylogue.pipeline_runner import Pipeline, PipelineContext


def test_pipeline_runs_stages_in_order():
    order = []

    class Stage:
        def __init__(self, name):
            self.name = name

        def run(self, ctx):
            order.append(self.name)

    pipeline = Pipeline([Stage("first"), Stage("second")])
    context = PipelineContext(env=SimpleNamespace(), options=None)
    pipeline.run(context)

    assert order == ["first", "second"]
    assert context.aborted is False


def test_pipeline_abort_stops_subsequent_stages():
    order = []

    class AbortStage:
        def run(self, ctx):
            order.append("abort")
            ctx.abort()

    class ShouldNotRun:
        def run(self, ctx):
            order.append("safety")

    pipeline = Pipeline([AbortStage(), ShouldNotRun()])
    context = PipelineContext(env=SimpleNamespace(), options=None)
    pipeline.run(context)

    assert order == ["abort"]
    assert context.aborted is True


def test_pipeline_records_stage_history():
    class Stage:
        def run(self, ctx):
            ctx.set("value", 42)

    pipeline = Pipeline([Stage()])
    context = PipelineContext(env=SimpleNamespace(), options=None)
    pipeline.run(context)

    assert context.history
    entry = context.history[0]
    assert entry["name"] == "Stage"
    assert entry["status"] == "ok"
    assert entry["duration"] >= 0.0


def test_pipeline_records_errors_and_aborts():
    class FailingStage:
        def run(self, ctx):
            raise ValueError("boom")

    pipeline = Pipeline([FailingStage()])
    context = PipelineContext(env=SimpleNamespace(), options=None)

    try:
        pipeline.run(context)
    except ValueError:
        pass
    else:  # pragma: no cover - defensive
        assert False, "expected ValueError"

    assert context.aborted is True
    assert context.history and context.history[0]["status"] == "error"
    assert context.errors and "FailingStage" in context.errors[0]


def test_pipeline_context_require():
    ctx = PipelineContext(env=SimpleNamespace(), options=None)
    ctx.set("key", "value")
    assert ctx.require("key") == "value"
    try:
        ctx.require("missing")
    except KeyError as exc:
        assert "missing" in str(exc)
    else:  # pragma: no cover - defensive
        assert False, "expected KeyError"
