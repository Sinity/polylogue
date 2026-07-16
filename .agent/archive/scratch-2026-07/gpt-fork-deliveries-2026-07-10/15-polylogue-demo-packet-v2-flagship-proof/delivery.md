## Turn 114 (role=assistant)

# Demo Packet v2 + flagship proof package

I treated the attached mission as the newer execution authority, including its explicit requirement that these remain deterministic product-contract proofs rather than simulated field or private-archive findings. fileciteturn0file0

The complete package is here:

**[Download the Polylogue Demo Packet v2 and flagship proof kit](sandbox:/mnt/data/Polylogue-Demo-Packet-v2-Flagships-kit.zip)**  
[SHA-256 checksum file](sandbox:/mnt/data/Polylogue-Demo-Packet-v2-Flagships-kit.zip.sha256)

```text
40fa31aeccd41a8c61e3bbe5d721d1f5395cc4f14c7411f94663b777a23eef77
```

The ZIP was integrity-tested after creation, extracted into a fresh directory, and its internal manifest was reverified. The manifest covers 114 payload files, with no unlisted files or symlinks.

Start with:

- [00-START-HERE.md](sandbox:/mnt/data/Polylogue-Demo-Packet-v2-Flagships-kit/00-START-HERE.md)
- [Interactive static evidence viewer](sandbox:/mnt/data/Polylogue-Demo-Packet-v2-Flagships-kit/visuals/evidence-viewer.html)
- [Executive summary](sandbox:/mnt/data/Polylogue-Demo-Packet-v2-Flagships-kit/EXECUTIVE-SUMMARY.md)
- [Flagship-results visual](sandbox:/mnt/data/Polylogue-Demo-Packet-v2-Flagships-kit/visuals/flagship-results.png)

## What was implemented

The patch series introduces Demo Packet v2 as a genuine epistemic contract rather than a file-presence convention.

A conforming packet now requires:

- exactly one primary construct;
- a claim declared before execution and citing its predeclaration receipt;
- an independent oracle with expected values and evidence;
- a comparative baseline;
- at least one negative control;
- at least one missing-evidence control;
- an explicit falsifier and its evaluated state;
- measured results linked to evidence;
- typed, packet-local receipts;
- mandatory SHA-256 binding for every receipt artifact;
- reproduction and provenance metadata;
- explicit non-claims.

The validator also enforces properties that JSON Schema alone does not conveniently express:

- exact report headings, so `Claimant` cannot satisfy a required `Claim` section;
- receipt-path confinement;
- exact ref occurrence in the declared artifact;
- ref namespace and receipt-kind agreement;
- nested citation closure;
- duplicate receipt-ref rejection;
- unique control IDs;
- unique measurement names;
- consistency between `falsifier.triggered` and `falsifier.result`;
- rejection of conforming packet directories omitted from the registry.

The normative files are:

- [Demo Packet v2 JSON Schema](sandbox:/mnt/data/Polylogue-Demo-Packet-v2-Flagships-kit/contracts/demo-packet-v2.schema.json)
- [Demo doctrine](sandbox:/mnt/data/Polylogue-Demo-Packet-v2-Flagships-kit/contracts/demo-doctrine.md)
- [Code-surface map](sandbox:/mnt/data/Polylogue-Demo-Packet-v2-Flagships-kit/CODE-SURFACE-MAP.md)

All previously registered packets were migrated. The deterministic tour’s report JSON and human report now use the v2 claim, oracle, controls, falsifier, results, and non-claims vocabulary.

## Flagship 1: The Receipts

The demonstration uses provider-native Codex records that pass through the production parser and archive writer. It does not fabricate a completed packet directly.

The planted agent sentence is:

```text
Tests pass.
```

The immediately preceding structural verifier evidence says:

```text
invocation: pytest -q tests/test_receipts.py
exit code: 1
duration: 1750 ms
outcome: failed
```

The packet resolves the relationship through:

- the exact authored claim block;
- the normalized paired action;
- the normalized raw tool-result block;
- the provider-native result record;
- the pre-normalization experiment plan.

The full artifacts are:

- [The Receipts human report](sandbox:/mnt/data/Polylogue-Demo-Packet-v2-Flagships-kit/flagship-run/the-receipts/report.md)
- [The Receipts machine packet](sandbox:/mnt/data/Polylogue-Demo-Packet-v2-Flagships-kit/flagship-run/the-receipts/packet.json)
- [Exact claim block](sandbox:/mnt/data/Polylogue-Demo-Packet-v2-Flagships-kit/flagship-run/the-receipts/receipts/block-codex-session-demo-receipts-receipts-a1-0.json)
- [Paired verifier action](sandbox:/mnt/data/Polylogue-Demo-Packet-v2-Flagships-kit/flagship-run/the-receipts/receipts/action-codex-session-demo-receipts-fc-receipts-verifier-0.json)
- [Raw normalized result block](sandbox:/mnt/data/Polylogue-Demo-Packet-v2-Flagships-kit/flagship-run/the-receipts/receipts/block-codex-session-demo-receipts-result-receipts-verifier-0.json)
- [Provider-native oracle](sandbox:/mnt/data/Polylogue-Demo-Packet-v2-Flagships-kit/flagship-run/the-receipts/receipts/artifact-the-receipts-provider-oracle.json)

### Comparative baseline

The naive-grep arm finds:

```text
pass hits:  2
error hits: 1
```

This is misleading because it cannot identify which text is an operation outcome.

The anti-grep controls deliberately include:

- prose containing `error` without any failed operation;
- a real verifier failure whose result does not contain `error`;
- result-text variants `all done`, the original failure output, and `ERROR ERROR ERROR`.

All three result-text variants remain structurally `failed`. The outcome therefore does not depend on vocabulary.

A second verifier invocation has no result. Polylogue returns:

```text
missing_result
```

It does not infer success from absence.

The complete baseline and controls are in:

- [Naive-grep receipt](sandbox:/mnt/data/Polylogue-Demo-Packet-v2-Flagships-kit/flagship-run/the-receipts/receipts/query-the-receipts-naive-grep.json)
- [Vocabulary-invariance receipt](sandbox:/mnt/data/Polylogue-Demo-Packet-v2-Flagships-kit/flagship-run/the-receipts/receipts/artifact-the-receipts-vocabulary-invariance.json)
- [Missing-result action](sandbox:/mnt/data/Polylogue-Demo-Packet-v2-Flagships-kit/flagship-run/the-receipts/receipts/action-codex-session-demo-receipts-fc-receipts-missing-0.json)

## Flagship 2: Count It Once

The second flagship distinguishes three quantities that transcript systems commonly conflate:

| View | Rows | Tokens |
|---|---:|---:|
| Stored own rows | 6 | 210 |
| Sum of readable composed session views | 8 | 240 |
| Logical unique high-water | 6 | 210 |
| Copied-prefix difference | 2 | 30 |

The physical parent and fork sessions remain fully inspectable. The fork’s two copied-prefix messages appear in both composed views, but stable-ref logical accounting charges them once.

The exact duplicated refs are preserved rather than hidden behind an aggregate.

A fresh subagent contributes one independent 60-token row. It remains in the logical set, demonstrating that the implementation is not merely deduplicating everything attached to the same session tree.

A nonexistent selected session is also exercised and returned explicitly as missing.

The complete artifacts are:

- [Count It Once human report](sandbox:/mnt/data/Polylogue-Demo-Packet-v2-Flagships-kit/flagship-run/count-it-once/report.md)
- [Count It Once machine packet](sandbox:/mnt/data/Polylogue-Demo-Packet-v2-Flagships-kit/flagship-run/count-it-once/packet.json)
- [Stored, composed, and logical accounting rows](sandbox:/mnt/data/Polylogue-Demo-Packet-v2-Flagships-kit/flagship-run/count-it-once/receipts/query-count-it-once-accounting.json)
- [Exact differing rows](sandbox:/mnt/data/Polylogue-Demo-Packet-v2-Flagships-kit/flagship-run/count-it-once/receipts/artifact-count-it-once-differing-rows.json)
- [Parent session receipt](sandbox:/mnt/data/Polylogue-Demo-Packet-v2-Flagships-kit/flagship-run/count-it-once/receipts/session-codex-session-demo-lineage-parent.json)
- [Fork session receipt](sandbox:/mnt/data/Polylogue-Demo-Packet-v2-Flagships-kit/flagship-run/count-it-once/receipts/session-codex-session-demo-lineage-fork.json)
- [Fresh-subagent receipt](sandbox:/mnt/data/Polylogue-Demo-Packet-v2-Flagships-kit/flagship-run/count-it-once/receipts/session-codex-session-demo-lineage-subagent.json)
- [Missing-session control](sandbox:/mnt/data/Polylogue-Demo-Packet-v2-Flagships-kit/flagship-run/count-it-once/receipts/query-count-it-once-missing-session-control.json)

## Stronger-than-requested gate audit

Beyond the required missing-falsifier and missing-controls tests, I executed a 15-case adversarial mutation matrix.

Starting from the same conforming packet, the audit independently challenged:

1. missing falsifier;
2. missing controls;
3. missing claim predeclaration receipt;
4. missing receipt digest;
5. incorrect receipt digest;
6. receipt path escaping the packet;
7. declared ref absent from the artifact;
8. duplicate receipt ref;
9. undeclared claim citation;
10. deceptive `Claimant` report heading;
11. inconsistent falsifier state;
12. duplicate control ID;
13. duplicate measurement name;
14. conforming but unregistered packet;
15. the untouched valid control packet.

All 15 produced the expected accept or reject result.

- [Human-readable adversarial audit](sandbox:/mnt/data/Polylogue-Demo-Packet-v2-Flagships-kit/audits/GATE-ADVERSARIAL-AUDIT.md)
- [Machine-readable adversarial audit](sandbox:/mnt/data/Polylogue-Demo-Packet-v2-Flagships-kit/audits/gate-adversarial-audit.json)
- [Required negative-test process transcript](sandbox:/mnt/data/Polylogue-Demo-Packet-v2-Flagships-kit/logs/negative-gate-transcript.txt)

The required subprocess results were:

```text
complete packet        exit 0
missing falsifier      exit 1
missing controls       exit 1
```

## Predeclaration and deterministic execution

Before any archive database was initialized or any normalization occurred, the fixture wrote:

- the two claims;
- their expected measurements;
- explicit falsifiers;
- a manifest of provider-native source files;
- SHA-256 hashes and byte lengths for those files.

That plan is available here:

[Pre-normalization experiment plan](sandbox:/mnt/data/Polylogue-Demo-Packet-v2-Flagships-kit/flagship-run/archive/source/experiment-plan.json)

Two independent executions produced:

- byte-identical The Receipts packet directories;
- byte-identical Count It Once packet directories;
- byte-identical provider-native source directories.

The committed packet directories in the patch series are also byte-identical to the executed packet directories in the kit.

## Patch series

The implementation is an eight-commit series against:

```text
base: f6c1da997bea64bc6cd9670d9cbb8f7e7439ec51
head: 55c7833cc8bd3390e767114982b7145453f8d572
tree: 06eaec8003ae2486240f396ab0ddbdde4eb748fc
```

Artifacts:

- [Combined patch](sandbox:/mnt/data/Polylogue-Demo-Packet-v2-Flagships-kit/patches/polylogue-demo-packet-v2-combined.patch)
- [Incremental Git bundle](sandbox:/mnt/data/Polylogue-Demo-Packet-v2-Flagships-kit/patches/polylogue-demo-packet-v2.bundle)
- [Patch review guide](sandbox:/mnt/data/Polylogue-Demo-Packet-v2-Flagships-kit/PATCH-REVIEW.md)
- [Changed-file inventory](sandbox:/mnt/data/Polylogue-Demo-Packet-v2-Flagships-kit/CHANGED-FILES.txt)
- [Diff statistics](sandbox:/mnt/data/Polylogue-Demo-Packet-v2-Flagships-kit/DIFFSTAT.txt)

The series changes 82 paths with 4,166 insertions and 95 deletions.

Applying all eight patches with `git am` in a clean worktree at the exact base produced the exact expected Git tree. The packaged apply helper verifies that tree identity rather than comparing commit hashes, since `git am` may produce different committer metadata.

## Validation summary

The complete report is [VALIDATION-REPORT.md](sandbox:/mnt/data/Polylogue-Demo-Packet-v2-Flagships-kit/VALIDATION-REPORT.md). The machine summary is [EXECUTION-RECEIPT.json](sandbox:/mnt/data/Polylogue-Demo-Packet-v2-Flagships-kit/EXECUTION-RECEIPT.json).

Passed:

```text
focused tests on implementation branch             41
focused tests after clean patch application        41
registered packets                                  5 / 5
flagship receipt hashes and refs                   16 / 16
adversarial contract expectations                  15 / 15
independent deterministic executions                2
patches applied                                     8 / 8
exact applied Git-tree identity                     PASS
Git bundle verification                             PASS
SQLite integrity                                    PASS
archive counts                                      4 sessions / 12 messages / 12 blocks
package manifest                                    114 payload files
ZIP integrity                                       PASS
```

## Beads handoff

I did not modify the authoritative Beads database. The supplied snapshot lacked `polylogue-212.12`, so the package includes closure evidence and proposed notes for the operator or sole Beads captain.

[Beads handoff and proposed closure evidence](sandbox:/mnt/data/Polylogue-Demo-Packet-v2-Flagships-kit/BEADS-HANDOFF.md)

It covers:

- recommended closure evidence for `polylogue-212.12`;
- recommended closure evidence for `polylogue-212.2`;
- ownership advice for the Count It Once flagship;
- eight concrete follow-on Beads for field validation, generated lineage tests, independent conformance, public rendering, privacy linting, provider parity, and tour dependency separation.

## Qualification and non-claims

The execution environment was offline and lacked Ruff, MyPy, Hypothesis, `sqlite-vec`, and several normal runtime dependencies. Small reviewer-only compatibility shims were used to execute the focused suite and public CLI; none is committed or packaged.

Accordingly, the package does not claim:

- full test-suite success;
- Ruff or MyPy success;
- execution under the repository’s locked development environment;
- completion of the legacy vector-dependent full tour;
- field prevalence;
- production-scale performance;
- provider parity;
- model quality;
- agent-memory uplift;
- the private-archive Receipts benchmark.

The private benchmark remains a separate local-only lane and was not simulated.

[Complete public non-claims](sandbox:/mnt/data/Polylogue-Demo-Packet-v2-Flagships-kit/NON-CLAIMS.md)

The fastest rigorous review sequence is the static evidence viewer, the two packet reports, the adversarial audit, the construct-validity audit, and then the eight patches in order.

---

