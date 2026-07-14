# Fuzz Targets

Atheris-based fuzz targets covering the highest-leverage parser and
sanitizer boundaries. Each target has two modes:

- **pytest mode** (default in CI): each fuzz module exposes pytest tests that
  iterate the target over a stable, seed-deterministic random byte stream.
  These run in the normal test suite and guarantee that the target survives
  arbitrary input on every commit.
- **libFuzzer mode** (campaign): the module's `main()` invokes
  `atheris.Setup(...); atheris.Fuzz()` against libFuzzer for extended
  coverage-guided fuzzing. This is opt-in and intended for periodic
  campaigns or local bug-hunting.

## Targets

| Module | Target function(s) | Surface under test |
| --- | --- | --- |
| `fuzz_fts5_escape.py` | `fuzz_fts5_escape` | `polylogue.storage.search.escape_fts5_query` — SQLite FTS5 MATCH escaping. |
| `fuzz_json_parsers.py` | `fuzz_chatgpt_parser`, `fuzz_codex_parser`, `fuzz_claude_code_parser`, `fuzz_claude_ai_parser`, `fuzz_drive_parser`, `fuzz_antigravity_parser`, `fuzz_browser_capture_parser`, `fuzz_local_agent_parser`, `fuzz_all_parsers` | `polylogue.sources.parsers.*` — provider record dispatch (chatgpt, codex, claude-code/ai, drive/gemini, antigravity, browser_capture, local_agent). |
| `fuzz_path_sanitizer.py` | `fuzz_path_sanitizer`, `fuzz_name_sanitizer` | `polylogue.core.security.sanitize_path` and filename sanitization. |
| `fuzz_timestamp.py` | `fuzz_parse_timestamp`, `fuzz_normalize_timestamp`, `fuzz_format_timestamp`, `fuzz_all_timestamps` | `polylogue.core.timestamps` — ISO/epoch parsing and normalization. |

Every public target above is asserted to be present and structurally callable
by `tests/unit/sources/test_fuzz_targets_executable.py`. That test does not
run the target — it only imports the module and confirms the entrypoint is
defined, so a deleted or renamed target fails the unit suite immediately.

## Invocation

### Pytest mode (always available)

```bash
nix develop -c pytest tests/fuzz -q
```

### libFuzzer mode (extended campaign)

Atheris is a dev dependency (`pyproject.toml` `[project.optional-dependencies]
dev`) and is installed in the project devshell. If running outside the
devshell, install with `uv pip install atheris>=2.3.0`.

```bash
# 10k iterations against one target (seconds to a few minutes)
nix develop -c python tests/fuzz/fuzz_fts5_escape.py

# Long campaign — raise iteration count via env var
FUZZ_ITERATIONS=1000000 nix develop -c python tests/fuzz/fuzz_json_parsers.py

# Bounded by wall-clock instead of iterations
nix develop -c python tests/fuzz/fuzz_path_sanitizer.py -max_total_time=600
```

`atheris.Setup` accepts libFuzzer CLI flags after `sys.argv`. The targets pass
`-max_total_time=300` and `-runs=$FUZZ_ITERATIONS` by default. Override via
extra args on the command line.

## Seed Corpus

Atheris uses libFuzzer-compatible corpora. A local seed corpus, when present,
lives at `.local/fuzz-corpus/<target-name>/` and is gitignored. Bootstrap a
corpus by running a target once with an empty directory:

```bash
mkdir -p .local/fuzz-corpus/fts5
nix develop -c python tests/fuzz/fuzz_fts5_escape.py .local/fuzz-corpus/fts5
```

libFuzzer will populate the corpus with discovered inputs that expand
coverage. Subsequent runs reuse the corpus for faster convergence.

## Failure Capture Policy

When a fuzz target finds a crash, libFuzzer writes the minimized triggering
input to `crash-<sha1>` in the current working directory.

Promote the crash into a durable pytest regression case:

1. Save the failing bytes to `tests/witnesses/<target>/<short-slug>.bin` (or
   `.json` if textual). Inputs are committed to the repo only after privacy
   review — strip personal data first using `devtools witness-minimize`.
2. Add a parametrized pytest case in the same fuzz module (or a new
   `tests/unit/sources/test_<target>_witnesses.py`) that loads the bytes and
   asserts the target either returns cleanly or raises an expected error
   type. Crashes (`AssertionError`, segfault, unhandled exception) must
   never recur.
3. Fix the underlying bug in the production code in the same PR. A fuzz
   crash without an accompanying fix is not promotable.

See the existing parser property tests in
`tests/unit/sources/test_parsers_props.py` for the pattern that pytest cases
should follow.
