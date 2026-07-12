"""Product-grade deterministic demo archive helpers."""

from .constructs import DEMO_CONSTRUCTS, DemoConstruct, DemoConstructCoverage, evaluate_demo_constructs
from .models import DemoSeedResult, DemoTourResult, DemoTourStep, DemoVerifyResult
from .receipts import (
    CompletionClaimEvidence,
    CompletionClaimExperimentResult,
    DemoActionReceipt,
    DemoReceiptsResult,
    inspect_completion_claims,
    inspect_demo_receipts,
    render_demo_receipts,
)
from .script import render_demo_script
from .seed import DEMO_SOURCE_DIRNAME, demo_source_specs, materialize_demo_source, seed_demo_archive
from .tour import run_demo_tour
from .verify import verify_demo_archive

__all__ = [
    "DEMO_CONSTRUCTS",
    "DEMO_SOURCE_DIRNAME",
    "DemoConstruct",
    "DemoConstructCoverage",
    "DemoActionReceipt",
    "CompletionClaimEvidence",
    "CompletionClaimExperimentResult",
    "DemoReceiptsResult",
    "DemoSeedResult",
    "DemoTourResult",
    "DemoTourStep",
    "DemoVerifyResult",
    "demo_source_specs",
    "evaluate_demo_constructs",
    "inspect_demo_receipts",
    "inspect_completion_claims",
    "materialize_demo_source",
    "render_demo_receipts",
    "render_demo_script",
    "run_demo_tour",
    "seed_demo_archive",
    "verify_demo_archive",
]
