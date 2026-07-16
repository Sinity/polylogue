# A5 — Prior art for warm-process / local-daemon CLIs

Research deliverable for the swarm2 thin-client + hot-daemon + composer design.
Read-only research; every "steal this" recommendation is grounded in a shipped
system and cross-referenced to Polylogue's target frame (SWARM_BRIEF.md).

**Scope reminder (the frame we design within):** powerful resident daemon
(feature-rich, memory-hungry is fine), thin client over UDS speaking a wire
protocol (not FFI), and a **composer / live-preview** as the headline UX needing
`complete(partial)` and `preview(spec)` in single-digit ms. Daemon may be
REQUIRED (`--no-daemon` is break-glass only). No bare-query mode.

---

## Survey: the systems, cold

| System | Transport | Discovery / autostart | Single-instance key | Cold-start UX | Idle/lifetime | Most relevant to |
|---|---|---|---|---|---|---|
| **LSP (generic)** | JSON-RPC 2.0 over stdio (or socket) | Editor spawns + owns the process; `initialize`/`initialized` handshake negotiates capabilities | per-editor process (1:1) | editor shows "indexing"; requests queue behind `initialize` | dies with client | protocol shape, capability negotiation, lazy resolve |
| **gopls daemon** (`-remote=auto`) | JSON-RPC; thin **forwarder** sidecar → shared daemon over socket | forwarder auto-starts daemon if absent, connects, forwards | `auto;<id>` selects a named shared daemon on POSIX | forwarder is instant; daemon warms cache once, shared across clients | `-remote.listen.timeout=30m` after last client | **thin-forwarder + shared warm cache** (our exact shape) |
| **Bazel server** | gRPC over loopback; `command_port` file | client reads `$OUTPUT_BASE/server/{command_port,request_cookie,response_cookie,server.pid.txt}`; starts JVM if stale | `output_base` = `md5(workspace_root)`; server lockfile | first build eats JVM+graph load; later builds are warm | `--max_idle_secs` (0 = never) | **socket-handshake files + cookies + per-workspace identity** |
| **Watchman** | UDS; request/response PDUs in JSON **or BSER** binary; subscriptions push | `watchman get-sockname` returns socket path; client autostarts daemon | one daemon per user; socket path derived from user/state | first watch crawls tree; queries block up to `sync_timeout` | long-lived; explicit `shutdown-server` | **binary wire format, subscriptions, freshness via cookies** |
| **git credential-cache** | UDS, `$XDG_CACHE_HOME/git/credential/socket` | client connects; daemon spawned on demand | socket path; daemon **exits when last credential expires** | negligible | self-terminating on empty state | **minimal UDS + self-reaping daemon + XDG path** |
| **Emacs daemon** | UDS in `$XDG_RUNTIME_DIR/emacs/` (fallback `$TMPDIR/emacs$UID`) | `emacsclient --alternate-editor=''` starts `emacs --daemon` then reconnects | named socket in runtime dir; `--socket-name` for multiples | daemon boot is slow once; every `emacsclient` after is instant | persistent until `kill-emacs` | **autostart-on-miss + runtime-dir socket discovery** |
| **fzf `--listen`** | HTTP over TCP **or UDS**; POST actions, GET state | port 0 → auto-pick, exported via `$FZF_PORT`; `FZF_API_KEY` header auth | the running fzf process itself | n/a (already interactive) | dies with the picker | **live remote control of an interactive surface (the composer)** |
| **Fig/Amazon Q autocomplete** | local socket IPC; PTY proxy (`figterm`) → Rust client → Node spec engine | shell integration streams keystrokes to daemon | per-user daemon + per-terminal PTY proxy | spec load once | resident | **declarative completion-spec engine + keystroke streaming** |

Everything below distills these into the patterns worth stealing.

---

## The 5–7 patterns to steal (opinionated, ranked)

### Pattern 1 — Thin forwarder over a shared warm daemon (gopls `-remote=auto`)

This is Polylogue's target shape almost exactly, already shipped and battle-hardened.
gopls splits into a **forwarder** (thin, per-client, instant to spawn) that owns
nothing but the connection, and a **daemon** (one process) that holds the expensive
cache. `-remote=auto` makes the forwarder *auto-start the daemon if it is not
already up*, connect, and forward the protocol; all clients share the daemon's
cache but keep their own session/view state
([go.dev/gopls/daemon](https://go.dev/gopls/daemon)).

The critical design lesson: gopls separates **cache** (shared, session-independent
— the warm archive/index/embeddings/query planner state) from **session/view**
(per-client — the composer's in-progress query, cursor, selection, output prefs).

**Steal for Polylogue:**
- The thin client's *only* job is: find socket → if absent, fork-exec `polylogued`
  and wait for readiness → connect → send spec → stream result. No substrate import.
  This is what lets the client later be rewritten in Go/Rust — the forwarder role
  is trivial and language-agnostic. gopls proves the split works.
- Keep a **shared daemon cache** (the loaded index.db handles, FTS readiness state,
  embedding index, compiled query grammar/Lark parser) and **per-connection session
  state** (composer partial query, view/render prefs, pagination cursor). Do not
  conflate them; that separation is why one warm daemon serves many CLI/TUI/MCP
  clients without re-warming.
- Adopt gopls's `-remote.listen.timeout` idea: daemon stays alive N minutes after
  the last client disconnects, so a burst of `polylogue find … | read …` invocations
  reuse one warm process instead of re-warming per command.

### Pattern 2 — Handshake files + cookies for discovery, liveness, and identity (Bazel)

Bazel is the reference for *robust* connect-or-start. In `$OUTPUT_BASE/server/` it
writes `command_port`, `request_cookie`, `response_cookie`, and `server.pid.txt`
([bazel.build command-line-reference](https://bazel.build/reference/command-line-reference)).
The client reads the port, checks the pid is alive, and includes the request cookie
in every call so a *stale* server (wrong version, different workspace, zombie) is
rejected rather than silently misused. The `output_base` is `md5(workspace_root)`,
so each workspace deterministically maps to its own server — no cross-talk.

**Steal for Polylogue:**
- A daemon **statefile** next to the archive (e.g. `ops.db`-adjacent
  `polylogued.json`) carrying: `{pid, socket_path, protocol_version,
  daemon_version, archive_fingerprint, request_cookie, started_at}`. The client's
  connect sequence: read statefile → pid alive? → `protocol_version` compatible? →
  `archive_fingerprint` matches configured archive? → cookie matches. Any mismatch
  → treat as no daemon and (re)start. This kills the entire class of "connected to
  a daemon serving a different/old archive" bugs.
- **Per-archive identity**, Bazel-style: derive the socket name / statefile location
  from a hash of the resolved archive root, so two archives (prod 38 GB vs a test
  `/tmp/polylogue-archive`) get independent daemons with zero config. This composes
  cleanly with Polylogue's existing `POLYLOGUE_ARCHIVE_ROOT` override.
- **`--max_idle_secs` equivalent** as a config knob, `0` = never (the operator wants
  a resident daemon; default should lean long or infinite, unlike Bazel's default).

### Pattern 3 — Capability negotiation + lazy resolve = the composer's speed budget (LSP)

LSP's `initialize` handshake exchanges `ClientCapabilities`/`ServerCapabilities`
before any real request, so client and server agree on what's supported instead of
probing. But the *directly load-bearing* pattern for the composer is LSP's
**two-phase completion**:

1. `textDocument/completion` returns a list of items **cheaply** — labels/kinds
   only. If the list is provisional it sets `isIncomplete: true`, telling the client
   to re-query as the user types more.
2. `completionItem/resolve` lazily fills the **expensive** fields (documentation,
   `additionalTextEdits`) for *only the item the user actually highlights*
   ([lsp-types CompletionOptions](https://docs.rs/lsp-types/latest/lsp_types/struct.CompletionOptions.html)).

This is exactly how you keep a live composer under budget: enumerating candidate
field names / values / operators / lanes must be near-free; the *expensive* work
(a real preview, a count, a cost estimate) is deferred to the one thing in focus.

**Steal for Polylogue:**
- Split the composer protocol into a cheap enumeration op and an expensive resolve
  op. `complete(partial) -> [candidate]` returns structural completions (fields,
  enum values, pipeline stages, set-ops, view names, saved macros) from static
  grammar knowledge + cheap `index.db` cardinality — target sub-ms, no query
  execution. `preview(spec)` / `resolve(candidate)` runs the actual bounded query
  only for the current focus.
- Carry an `isIncomplete`/`is_partial` flag on completion results so the composer
  knows to re-request as the query narrows, and a `data`/opaque token on each
  candidate that the resolve call echoes back (LSP requires `data` be preserved
  verbatim between completion and resolve — Zed shipped bugs by dropping it:
  [zed #21185](https://github.com/zed-industries/zed/issues/21185)). Design the
  token so resolve needs no re-parse of the partial.
- **Version negotiation up front**: the connect handshake exchanges
  `{protocol_version, supported_ops, supported_views, retrieval_lanes}` so a newer
  daemon and older client (or Go client) degrade gracefully instead of erroring
  mid-composer.

### Pattern 4 — Debounce + cancellation, or the composer melts the daemon (LSP clients)

Every LSP client learned the same lesson the hard way: firing a completion/preview
request **per keystroke** is a performance disaster; you must **debounce** and
**cancel superseded requests**. Naive clients cause dramatically higher server CPU;
common practice is a debounce window and `$/cancelRequest` for in-flight work that a
newer keystroke obsoletes ([nvim-compe #388](https://github.com/hrsh7th/nvim-compe/issues/388),
[helix #8351](https://github.com/helix-editor/helix/discussions/8351)).

**Steal for Polylogue:**
- The composer client must **debounce** `preview(spec)` (cheap `complete` can fire
  more eagerly) and stamp each request with a monotonic **generation id**; the
  daemon must support **cancellation** — when preview N+1 arrives, preview N is
  abandoned server-side (don't finish a 300 ms query nobody will read).
- Because the daemon is a shared warm process (Pattern 1), one runaway composer must
  not starve concurrent `read`/MCP clients: preview queries run under a **bounded
  budget** (row/time cap) and are the *first* thing cancelled. This dovetails with
  watchman's `sync_timeout`/`query blocks up to 60 s` model — always bound the wait.

### Pattern 5 — Binary wire + push subscriptions for streaming/live results (Watchman)

Watchman's UDS speaks a request/response protocol with PDUs in **JSON or BSER**
(a length-prefixed binary encoding — the client opts in by sending a magic byte
sequence), and `subscribe` lets the daemon **unilaterally push** any number of PDUs
to the client for stateful, live updates
([watchman socket-interface](https://facebook.github.io/watchman/docs/socket-interface),
[BSER](https://facebook.github.io/watchman/docs/bser.html)). Freshness is guaranteed
not by polling but by **cookies**: watchman drops a unique marker file and waits to
observe it, proving the view is current before answering
([watchman cookies](https://facebook.github.io/watchman/docs/cookies)).

**Steal for Polylogue:**
- Length-prefixed framing from day one (JSON body is fine to start; keep a binary
  path open for large result sets — 38 GB archive, big transcripts). Length prefix
  lets the thin/Go client read exactly one PDU without a JSON streaming parser.
- **Subscription/streaming** ops for the two places it matters: (a) the composer's
  live preview can be a subscription that re-pushes as the daemon finishes deeper
  ranking, and (b) `read`/`analyze` of a huge session streams message pages as PDUs
  rather than one giant blob. This is also how the daemon reports "index still
  warming" progressively instead of blocking.
- **Freshness contract** analogous to cookies: Polylogue already treats FTS/index
  readiness as a freshness invariant (docs/internals.md). Surface that in the
  protocol — a preview result carries a `freshness`/`index_ready` marker so the
  composer can show "results may be stale, index rebuilding" instead of lying, and
  a client can opt to block until fresh (watchman's `sync_timeout`).

### Pattern 6 — Autostart-on-miss with a per-user runtime socket (Emacs + git-credential-cache)

Two small tools nail the *ergonomics* of "the daemon should just be there":

- **Emacs**: `emacsclient --alternate-editor=''` — if no server answers, it runs
  `emacs --daemon` and reconnects, transparently. The socket lives in
  `$XDG_RUNTIME_DIR/emacs/` (falling back to `$TMPDIR/emacs$UID`), and `emacsclient`
  is written to *look in both* to survive the daemon being started in a different
  env ([emacsclient options](https://www.gnu.org/software/emacs/manual/html_node/emacs/emacsclient-Options.html)).
- **git-credential-cache**: socket at `$XDG_CACHE_HOME/git/credential/socket`;
  the daemon is spawned on demand and **exits when it has no state left**
  ([git-credential-cache--daemon](https://man7.org/linux/man-pages/man1/git-credential-cache--daemon.1.html)).

**Steal for Polylogue:**
- **Socket in `$XDG_RUNTIME_DIR/polylogue/<archive-hash>.sock`** (fallback `$TMPDIR`),
  not in the archive dir — runtime dir is tmpfs, per-user, auto-cleaned on logout,
  and correctly scoped. Bazel's per-workspace identity (Pattern 2) becomes the
  filename component here.
- **Transparent autostart** is the default client behavior (Emacs model): a plain
  `polylogue …` invocation with no live daemon forks `polylogued`, waits for a
  readiness ping (bounded, with a spinner), then proceeds. `--no-daemon` is the
  break-glass that imports the substrate in-process instead. The operator never runs
  a "start the daemon" command in the normal path.
- Robust socket discovery like emacsclient: check the statefile's `socket_path`
  first, then the conventional runtime-dir path, so a daemon started under systemd
  (different `$XDG_RUNTIME_DIR` semantics) is still found.
- Note the **anti-pattern to avoid**: git-credential-cache self-reaps when empty —
  Polylogue's operator explicitly wants a *resident, warm* daemon, so we invert this
  (long/infinite idle timeout, Pattern 2). Self-reaping is wrong here; it throws away
  the warm cache we paid to build.

### Pattern 7 — Declarative completion-spec engine (Fig) for the composer's brain

Fig/Amazon Q autocomplete is the closest prior art to Polylogue's *composer*
specifically. Its architecture: a **PTY proxy** (`figterm`) streams the user's
in-progress command line to a local daemon over socket IPC; the daemon maps the
partial input to a **completion spec** — a declarative schema of subcommands,
options, and args — and renders IDE-style suggestions
([withfig/autocomplete](https://github.com/withfig/autocomplete),
[fig.io user-manual](https://fig.io/user-manual/autocomplete)). The key idea: the
grammar of what-can-come-next is **data, not code**, so completions are generated by
walking a declarative spec against the parsed partial rather than bespoke per-command
logic.

**Steal for Polylogue:**
- Polylogue *already has* the spec: the Lark LALR grammar
  (`archive/query/expression.py:_QUERY_GRAMMAR`) plus the pipeline stage vocabulary
  and the `ProjectionSpec`/`RenderSpec` enums. The composer's `complete(partial)`
  should be driven by walking that grammar/enum set at the parse frontier — "given
  the partial parses to this state, the legal next tokens are these fields / these
  operators / these stage names / these view names / these set-ops." This is the
  Fig completion-spec pattern applied to a real formal grammar, which is *stronger*
  than Fig's hand-authored specs. It also directly generalizes the `fnm.10`
  fields/select and `fnm.12` macros beads.
- Value completions (enum values for a field, distinct origins, tag names) come from
  cheap `index.db` lookups — the daemon's warm cache makes this the free half of
  Pattern 3's split.
- fzf's `--listen` (Pattern-adjacent) shows the *other* half: a running interactive
  surface that accepts POST actions and exposes state over GET
  ([fzf man](https://man.archlinux.org/man/fzf.1.en)). If the composer is a TUI, the
  same daemon protocol *is* its control channel — `reload`/`change-prompt`-style
  actions become `preview(spec)`/`complete(partial)` calls. Don't invent a second
  IPC channel for the TUI; the composer speaks the same UDS protocol as the CLI.

---

## Synthesis: the recommended shape (one paragraph)

Build the **gopls forwarder split** (Pattern 1): a trivial, language-agnostic thin
client that finds-or-autostarts (Emacs model, Pattern 6) a resident warm daemon whose
socket lives in `$XDG_RUNTIME_DIR/polylogue/<archive-hash>.sock` and whose identity
is pinned by a **Bazel-style statefile with a request cookie + protocol/archive
version** (Pattern 2). The wire protocol is **length-prefixed PDUs with capability
negotiation up front and push subscriptions** (Watchman + LSP `initialize`,
Patterns 3/5). The composer is the headline: a **two-phase completion protocol** —
near-free `complete(partial)` driven by walking the *existing Lark grammar +
Projection/Render enums* (Fig completion-spec pattern over a real grammar, Pattern 7),
and deferred, **debounced, cancellable, budget-bounded** `preview(spec)` (LSP lazy
resolve + client debounce + `$/cancelRequest`, Patterns 3/4). Keep the daemon cache
shared and the composer session per-connection (gopls), and never let a runaway
preview starve concurrent `read`/MCP clients.

## Open questions for the operator

1. **Idle policy**: resident-forever (systemd user unit, warm always) vs a long
   `max_idle_secs`? Prior art splits — the operator's "hot daemon" language implies
   forever; confirm it should be a **systemd user service**, which also solves the
   `$XDG_RUNTIME_DIR` discovery robustness (emacsclient's dual-path hack exists
   precisely because of env skew between daemon-start and client env).
2. **One daemon per archive, or one multiplexing daemon?** Bazel = per-workspace;
   gopls `auto;<id>` = named shared. Per-archive (hash-keyed socket) is simpler and
   matches `POLYLOGUE_ARCHIVE_ROOT`; a multiplexing daemon is more memory-efficient
   if the operator routinely hits several archives.
3. **Binary wire (BSER-like) now or later?** JSON is enough to ship; the length-prefix
   framing must exist from day one so a future Go/Rust client and large streamed
   results don't force a protocol break.
4. **Composer transport for TUI**: reuse the UDS protocol as the TUI's control channel
   (fzf `--listen` model) — confirm we do *not* want a separate in-process TUI path.
5. **Cancellation semantics**: is best-effort server-side abandonment of superseded
   previews acceptable, or do we need hard `$/cancelRequest` acknowledgement before
   the next preview runs? Affects the daemon's request loop design.

## Sources

- gopls daemon: <https://go.dev/gopls/daemon>, <https://github.com/golang/tools/blob/master/gopls/doc/daemon.md>, <https://go.dev/gopls/design/design>
- Bazel server / command_port / cookies / output_base: <https://bazel.build/reference/command-line-reference>
- Watchman socket / BSER / cookies: <https://facebook.github.io/watchman/docs/socket-interface>, <https://facebook.github.io/watchman/docs/bser.html>, <https://facebook.github.io/watchman/docs/cookies>
- git-credential-cache--daemon: <https://man7.org/linux/man-pages/man1/git-credential-cache--daemon.1.html>, <https://git-scm.com/docs/git-credential-cache--daemon>
- Emacs daemon / emacsclient socket discovery: <https://www.gnu.org/software/emacs/manual/html_node/emacs/emacsclient-Options.html>
- fzf `--listen`: <https://man.archlinux.org/man/fzf.1.en>, <https://github.com/junegunn/fzf/blob/master/CHANGELOG.md>
- Fig autocomplete / completion specs: <https://github.com/withfig/autocomplete>, <https://fig.io/user-manual/autocomplete>
- LSP completion resolve / debounce / capabilities: <https://docs.rs/lsp-types/latest/lsp_types/struct.CompletionOptions.html>, <https://github.com/hrsh7th/nvim-compe/issues/388>, <https://github.com/zed-industries/zed/issues/21185>
