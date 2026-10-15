# First-path demonstrations: Morrow and Kestrel

These inputs and answer checks are the reusable start of M1 acceptance in the [realization plan](../../REALIZATION-PLAN.md). They exercise a narrow independent path per product, not all of slice 1. **Production demonstration status: unrun.** Neither composition skeleton has the daemon/storage/client route required below. Preparing inputs or passing oracle tests does not satisfy M1.

## Prepare independent inputs

From the spec root, into a new directory:

```sh
python3 polylogue-sinex-joint/realization/demos/prepare.py /realm/tmp/work/my-m1-inputs
python3 polylogue-sinex-joint/realization/demos/test_demos.py
```

The preparer refuses existing directories, reads the existing synthetic archives read-only and calls no product code. `expected.json` records source coordinates/hashes, prepared input coordinates and independently derived facts. SQLite expectations depend on rows, not version-dependent file bytes. Generated files stay in the owning run's scratch; do not edit the authoritative Morrow/Kestrel inputs. Each case runs in its own empty product state with only the selected source configured.

| Case | Polylogue: one Morrow session | Sinex: Atuin only |
|---|---|---|
| base | 13 records, 12 messages, 3 human prompts, 1 session | 21 rows read, 20 live commands; the soft-deleted row is not a command |
| append | One new typed prompt: 13 messages, 4 human prompts | One new row: 21 live commands |
| empty | No session; completed fresh search is empty, exit 1 | Valid empty database; completed read, zero commands, no record error |
| malformed | One complete invalid JSON line between valid records; 12 messages survive and 1 record error is visible | One invalid timestamp type alongside valid rows; 20 live commands survive and 1 record error is visible |

The malformed timestamp is text where Atuin declares integer nanoseconds. It is not an unusual valid date or a legitimately unknown clock. The Polylogue malformed line ends in a newline; it is not an incomplete tail that should wait for more bytes.

The Atuin case explicitly configures the fixture's privacy rule `tok_[a-z0-9_]+` → `[withheld]` before acquisition, using the product's attributed rule/config path. Source identity is computed from the unmasked source evidence as specified. The synthetic token must not appear in stored payloads or client output; retained raw bytes follow the declared privacy class. This demonstration does not authorize retention of private inputs or a deployment change.

## Run the real paths when bootstrap exists

Use packaged candidate binaries, isolated state and actual clients. Record spec/candidate SHAs, executable hashes/revision, schema identity, exact argument arrays, configuration and exit statuses in the existing AgentCTL/xtask run store. No script may fall back to an installed operator service. Polylogue's explicit `--archive DIR` selects its test archive; Sinex requires isolated database/store/bus plus a client configured to reach that test daemon. Do not rely on ambient routing or default sockets. Bootstrap must supply these launch/endpoint bindings before an unattended run is possible.

1. Start the candidate daemon with just the selected fixture source. Wait for an actual completed read/admission cycle and read-model position, not a generic health boolean. Retain source accounting and raw diagnostic output.
2. **Polylogue:** import the selected directory with `polylogue --archive DIR import INPUT --wait`; read the full session ID with `read --format json`; search/list that reference with `--json`; export it and compare its semantic contents with the read document. A configured Claude Code collection route should later repeat this against the growing input, rather than treating repeated import as proof of live collection. **Sinex:** configure `terminal.atuin-history` to the case's `history.db` and complete a poll; run each exact `query_checks[].query` via `sinexctl query QUERY --json`. These queries use raw `command.executed` events, not the later `commands` alias that requires fusion.
3. Compare actual JSON answers with `check_answer.py` below. Inspect `trace` through the real client: references reach retained bytes; timestamps and native keys agree; replay may change interpretation IDs but not occurrence coordinates. For Sinex the query set checks total, command, original time and exit code, including the absent deleted row and masked token. It does not yet test every payload field or provenance coordinate.
4. Repeat the same import/poll without changing input; semantic answers stay equal with no duplicate live facts. Append **to the same configured path** using the prepared append variant; wait for completion and check append expectations. Changing paths tests re-import, not append. Preserve logs identifying which action happened. For the supported Polylogue JSONL path, also insert a separator-only append between content appends: verify the full retained window and contiguous predecessor, observe durable byte advancement with zero new normalized records/sessions, then restart/retry and append the next record. Raw reconstruction, normalized answers, markers and attachment dependencies must equal the uninterrupted baseline. A gap or nonblank unfinished tail remains unavailable/partial rather than completed zero; no empty session is fabricated. This consumer-chain scenario remains unrun until the production path demonstrates it.
5. Run empty and malformed cases in separate state. Source accounting must distinguish completed zero from unread, failed or behind; list the malformed form, exact error and the valid records retained. The answer comparator checks returned data, not these source-accounting claims.
6. Run an uninterrupted baseline, then a second isolated daemon killed between the relevant retention/admission/index/publication boundaries. The run harness must kill only its recorded child PID, restart the same candidate against the same state and retry the same intent. Require semantic equivalence with the baseline and inspect durable progress. Waiting a fixed sleep and killing without evidence of the boundary is only a smoke test. M0 must provide the fault-injection/test hook or observable boundary before this criterion can close.
7. Polylogue M1 also calls real MCP `search`/`read` and compares the same facts, including empty and failed outcomes; do not call handler functions directly. Sinex's narrow first CLI path leaves confirmed delivery and MCP obligations explicitly open until delivered. Full slice acceptance, personal-corpus usefulness and installed-client/cutover verification remain separate.

**Publish the result even when a run fails.** The production driver must capture each command's stdout/stderr and exact exit status into its existing run store before invoking the comparator. A launch error, timeout, malformed response or answer mismatch still produces the same readable milestone result: show independent expectations, actual output or explicitly no answer, the failure reason, raw-log links and the repeat command. Later scenarios that could not run remain unrun with that reason; never synthesize an empty answer or completed progress. Clean up only the run's owned services, preserving its outputs. Rendering a report never clears an unexpected command failure or failed comparison; the run exits nonzero.

Preserve the exact exercised synthetic inputs and their digests with that result before temporary-state cleanup, including generated or mutated variants that triggered a failure. Record the seed and fixture recipe too. This keeps the failing bytes inspectable when generators or fixtures later change.

**Failed-run acceptance:** deliberately corrupt one actual answer while leaving the expected fixture unchanged; require a nonzero comparison/run exit and an inspectable expected-versus-actual mismatch in the result. Separately make the candidate fail to launch; require a nonzero run exit, launch diagnostics and an explicitly absent actual answer, with dependent scenarios unrun. These cases qualify failure reporting only when the real driver performs them. The production driver/reporter is still unimplemented; the current comparator and fixture tests alone do not satisfy this acceptance.

## Compare captured answers

```sh
python3 polylogue-sinex-joint/realization/demos/check_answer.py polylogue base EXPECTED.json READ.json
python3 polylogue-sinex-joint/realization/demos/check_answer.py sinex base EXPECTED.json QUERY-0.json --query-index 0
```

For nonempty Polylogue cases, provide the export-format document from `read --format json`; for empty, provide the completed query envelope and separately assert exit 1. For Sinex, run **every** generated query check and provide its query result envelope. The comparator rejects wrong prompt text/attribution, wrong query/total, incomplete pages and the synthetic token. Its tests deliberately corrupt answers; their passing demonstrates the oracle rejects those mistakes, not that a product works.

Preparing the oracle exposed two omissions in [Sinex's result contract](../../../sinex-spec/spec/query.md): the read-position field is now named `resolved.commit_seq`, and `page` remains an object with a null `cursor` on the last page, including empty results. The comparator checks those fields and semantic event rows, not merely totals. Source-accounting serialization, launch/endpoint binding, subprocess ownership, durability fault control and real MCP transport remain necessary work. Do not invent a parallel demo API to bypass those product contracts.

The fixture/oracle tests also exercise the existing reference MCP adapter: it now preserves `time_quality` on serialized event rows, including captured and unknown clocks. Its scripted transcript was regenerated from the adapter; each result is complete parseable JSON, removing the old 2,500-character display cut that hid rows. Fixture readers close their files and SQLite handles explicitly. This is a reference-handler check, not an MCP transport or product run.

One operator-readable result should show the question, independently expected and actual answer, commands, diagnostics, retry sequence and remaining failures, linking raw logs in the owning run store. Summarize actual outputs; do not turn these instructions into a green readiness report. This batch verifies fixture/oracle preparation only; executable orchestration and product E2E acceptance remain open.
