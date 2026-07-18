I completed the implementation patch and produced the four substantive delivery artifacts against snapshot commit `536a53efac0cbe4a2473ad379e4db49ef3fce74d`. The patch preserves PR #2791’s judgment lifecycle as the sole transition authority while adding the requested transaction safety, evidence disclosure, queue health, workflow consolidation, and prepared operator canary. This work follows the supplied mission and Bead authority. 

The implementation changes 36 files with 1,744 insertions and 601 deletions. The central `polylogue-41ow` fix reserves the SQLite writer slot before reading existing judgment state: standalone writes use `BEGIN IMMEDIATE`, while writes nested inside caller-owned transactions use a SAVEPOINT plus a zero-row write to upgrade the deferred transaction before the preservation read. The real two-connection regression proves that an automated replay cannot interleave between the preservation read and an operator acceptance and silently revert the judgment.

The remaining work includes bounded evidence previews shared by CLI and MCP, non-destructive candidate queue health integrated into root `judge`, direct status, and daemon status, removal of the duplicate `mark candidates` public command, stable actor-scoped capture idempotency, and an exact operator canary script. The canary script now checks initial capture, bounded review disclosure, acceptance, resulting assertion and judgment receipts, exact post-judgment replay, changed-content conflict, accepted-review visibility, reference resolution, context non-injection, and durable SQLite records. Its shell syntax passes `bash -n`.

Verification completed includes:

* 63 storage, lifecycle, bulk, evidence, and queue-health tests passing.
* 4 modified facade and cross-tier debt tests passing.
* 307 canonical CLI, action-contract, completion, note-canary, and deterministic-output tests passing.
* 29 MCP, product-workflow, and search-envelope tests passing.
* 92 direct-status tests passing.
* 77 status-routing and diagnostic tests passing.
* 59 daemon-status tests passing.
* Ruff passing over all changed Python files.
* Mypy passing over 14 changed production modules.
* Compileall and both generated-document synchronization checks passing.
* `git diff --check` passing.
* `PATCH.diff` successfully passing `git apply --check` in a clean detached worktree at the exact snapshot commit.

`PATCH.diff` is 189,695 bytes with SHA-256:

```text
a72c7e066521721cec2de5784d3e8071f2604496c5bb27ea2bb90b7699e7ec76
```

The complete repository suite remains unverified because two broad invocations exceeded the command ceiling without reporting a failure. The latest terminal PTY snapshot rerun also remains environmentally inconclusive: an unrelated Tokio worker intermittently injected a `Bad file descriptor` panic before otherwise valid CLI output. No live archive, daemon, MCP transport, browser surface, deployment, credentials, or genuine operator event was available, so the live canary was correctly prepared rather than fabricated.

The exact ZIP was not constructed before the execution environment stopped further filesystem operations. The completed package members are available individually:

[Download HANDOFF.md](sandbox:/mnt/data/ann04-package/HANDOFF.md)

[Download PATCH.diff](sandbox:/mnt/data/ann04-package/PATCH.diff)

[Download TESTS.md](sandbox:/mnt/data/ann04-package/TESTS.md)

[Download EVIDENCE.md](sandbox:/mnt/data/ann04-package/EVIDENCE.md)

The missing ZIP assembly and final ZIP-member/hash validation are a small packaging repair. The implementation itself does not presently indicate a need for a substantial second engineering pass.

