# Import, demo, and terminology floor

Public command: `import`.
Public object: session.
Public source family label: origin.

Internal daemon and pipeline code may still use ingestion words for background processing. Raw third-party export terms may remain inside parser modules, fixtures, and format-specific tests, but normalized surfaces should use Polylogue terms.

Current floor: `build_demo_corpus_specs()` declares the deterministic approved
demo fixture-world contract used by the release gate. It covers ChatGPT normal
chat plus Claude Code and Codex coding-agent-style sessions with fixed seeds.
`polylogue import --demo` materializes that fixture world and schedules it
through the daemon-backed import path.

Remaining work: end-to-end demo archive convergence evidence, README-ready
commands against the generated demo archive, parser regression pack for
advertised origins, truthful import operation status, early unsupported-shape
classification.

Acceptance: public examples use import/session/origin; demo mode never touches the real archive or configured source roots; parser outputs expose normalized terms; import status is visible in CLI status and web home.
