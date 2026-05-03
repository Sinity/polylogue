// Implementation handoff — first vertical slice for the daemon split.
// Renders as a structured doc: goal · slice · files · endpoints · models · tests · verify · risks · defer.

const Handoff = () => (
  <div className="handoff">
    <header className="handoff__head">
      <div className="handoff__crumb">
        <Mono>HANDOFF</Mono><span>›</span>
        <Mono>polylogued · vertical slice 1</Mono><span>›</span>
        <Mono>2026-05-02</Mono>
      </div>
      <h2>Move long-running modes out of <Mono>polylogue</Mono>; serve the local web reader from <Mono>polylogued</Mono>.</h2>
      <p>Smallest change that lets a user run <Mono>polylogued run</Mono> and open the local archive at <Mono>http://127.0.0.1:8765</Mono>. No new query semantics. Existing query/read kernel reused.</p>
      <div className="handoff__pills">
        <Pill state="ok">3–5 days</Pill>
        <Pill state="ok">no schema migration</Pill>
        <Pill state="warn">touches click_app</Pill>
        <Pill state="ok">behind feature flag <Mono>POLYLOGUE_DAEMON=1</Mono></Pill>
      </div>
    </header>

    <div className="handoff__grid">
      <Card title="goal">
        <ul className="checks">
          <li><Check ok /> <Mono>polylogued run</Mono> starts; logs <Mono>http://127.0.0.1:8765</Mono>.</li>
          <li><Check ok /> Web reader serves <Mono>/c/&lt;id&gt;</Mono> with header + paginated messages.</li>
          <li><Check ok /> Local API exposes the existing query/read kernel — no new semantics.</li>
          <li><Check ok /> <Mono>polylogue --latest open</Mono> opens the daemon URL when running, falls back to static HTML otherwise.</li>
          <li><Check ok /> <Mono>polylogue doctor --target daemon</Mono> reports up/lag/uptime.</li>
        </ul>
      </Card>

      <Card title="non-goal (defer)">
        <ul className="checks">
          <li><Check muted /> annotations · stars · pins · saved views (slice 2).</li>
          <li><Check muted /> recall pack builder (slice 3).</li>
          <li><Check muted /> moving <Mono>browser-capture serve</Mono> into daemon (slice 2).</li>
          <li><Check muted /> <Mono>polylogue-mcp</Mono> as separate executable (slice 4).</li>
          <li><Check muted /> retiring <Mono>products</Mono> command (slice 5, parallel).</li>
        </ul>
      </Card>

      <Card title="files likely touched" wide>
        <ul className="files">
          <li><Mono>polylogue/daemon/__init__.py</Mono>            <span>NEW · package init</span></li>
          <li><Mono>polylogue/daemon/app.py</Mono>                 <span>NEW · ASGI app · query/read endpoints</span></li>
          <li><Mono>polylogue/daemon/serve.py</Mono>               <span>NEW · uvicorn launcher · pid file · log</span></li>
          <li><Mono>polylogue/daemon/status.py</Mono>              <span>NEW · DaemonStatus model · lag/uptime/pid</span></li>
          <li><Mono>polylogue/daemon/web/</Mono>                   <span>NEW · static reader bundle (vendored)</span></li>
          <li><Mono>polylogue/cli/click_command_registration.py</Mono>  <span>edit · keep stub <Mono>browser-capture</Mono>; do not register <Mono>watch</Mono></span></li>
          <li><Mono>polylogue/cli/commands/doctor.py</Mono>        <span>edit · add <Mono>--target daemon</Mono></span></li>
          <li><Mono>polylogue/cli/query_actions.py</Mono>          <span>edit · <Mono>open</Mono> probes daemon, falls back</span></li>
          <li><Mono>polylogue/__main__.py</Mono>                   <span>+ <Mono>polylogued</Mono> entrypoint</span></li>
          <li><Mono>pyproject.toml</Mono>                          <span>+ console_script <Mono>polylogued = polylogue.daemon.serve:main</Mono></span></li>
          <li><Mono>tests/daemon/</Mono>                           <span>NEW · smoke + endpoint tests</span></li>
        </ul>
      </Card>

      <Card title="local api endpoints" wide>
        <table className="api">
          <thead><tr><th>method</th><th>path</th><th>shape</th><th>backed by</th></tr></thead>
          <tbody>
            <tr><td>GET</td><td><Mono>/api/health</Mono></td><td><Mono>{`{ status, uptime_s, pid, components[] }`}</Mono></td><td><Mono>DaemonStatus</Mono></td></tr>
            <tr><td>GET</td><td><Mono>/api/conversations</Mono></td><td>list (filters mirror CLI root flags)</td><td><Mono>ConversationQuerySpec</Mono></td></tr>
            <tr><td>GET</td><td><Mono>/api/conversations/&#123;id&#125;</Mono></td><td>header only</td><td><Mono>get_conversation_header</Mono> (#570)</td></tr>
            <tr><td>GET</td><td><Mono>/api/conversations/&#123;id&#125;/messages</Mono></td><td>page, <Mono>?limit&amp;offset&amp;role&amp;type</Mono></td><td><Mono>messages</Mono> read surface</td></tr>
            <tr><td>GET</td><td><Mono>/api/conversations/&#123;id&#125;/raw</Mono></td><td>raw artifact page</td><td><Mono>raw</Mono> read surface</td></tr>
            <tr><td>GET</td><td><Mono>/api/facets</Mono></td><td><Mono>{`{ providers, repos, tags, has[] }`}</Mono></td><td><Mono>completion descriptors</Mono></td></tr>
            <tr><td>GET</td><td><Mono>/c/&#123;id&#125;</Mono></td><td>web reader (server-rendered shell, hydrated)</td><td>same kernel</td></tr>
          </tbody>
        </table>
      </Card>

      <Card title="state · DaemonStatus model">
<pre className="codeblock">{`@dataclass(frozen=True)
class ComponentHealth:
    name: str           # "live:claude-code", "fts:index", ...
    state: Literal["ok","warn","stale","err","idle"]
    lag_s: float | None
    detail: str

@dataclass(frozen=True)
class DaemonStatus:
    pid: int
    uptime_s: float
    components: tuple[ComponentHealth, ...]
    archive_path: Path

    @property
    def overall(self) -> str:
        if any(c.state == "err"  for c in self.components): return "err"
        if any(c.state == "warn" for c in self.components): return "warn"
        return "ok"`}</pre>
      </Card>

      <Card title="acceptance tests">
        <ul className="checks">
          <li><Check ok /> <Mono>tests/daemon/test_smoke.py::test_polylogued_starts_and_serves</Mono></li>
          <li><Check ok /> <Mono>::test_health_reports_archive_path</Mono></li>
          <li><Check ok /> <Mono>::test_conversation_list_mirrors_cli_filters</Mono> (parametrized over <Mono>--provider</Mono>, <Mono>--repo</Mono>, <Mono>--has</Mono>)</li>
          <li><Check ok /> <Mono>::test_messages_pagination_matches_cli</Mono> (CLI vs API parity)</li>
          <li><Check ok /> <Mono>::test_open_falls_back_to_static_when_daemon_down</Mono></li>
          <li><Check ok /> <Mono>tests/cli/test_no_watch_alias.py</Mono> — asserts <Mono>polylogue watch</Mono> exits non-zero (clean break)</li>
        </ul>
      </Card>

      <Card title="verify · smoke commands">
<pre className="codeblock">{`# 1. start daemon, hit health
polylogued run &
curl -s http://127.0.0.1:8765/api/health | jq .overall   # → "ok"

# 2. parity vs CLI
diff <(polylogue --provider claude-code list --format json) \\
     <(curl -s 'http://127.0.0.1:8765/api/conversations?provider=claude-code')

# 3. doctor sees daemon
polylogue doctor --target daemon
# → daemon  ok  uptime 0:00:14  pid 12345  archive ~/.local/share/polylogue/archive.sqlite

# 4. open reuses daemon
polylogue --latest open
# → opens http://127.0.0.1:8765/c/<id>

# 5. clean break confirmed
polylogue watch                                          # → exit 2, "no such command"`}</pre>
      </Card>

      <Card title="risks &amp; unknowns">
        <ul className="risks">
          <li><Mark warn /> <b>port collision</b> — 8765 is already used by browser-capture receiver. Make port configurable; receiver moves into the same daemon in slice 2.</li>
          <li><Mark warn /> <b>cli/api filter parity</b> — root-flag → query-string mapping must come from a single descriptor table (#621). Don't duplicate.</li>
          <li><Mark warn /> <b>process management</b> — keep daemon as a foreground process; users wrap with systemd/launchd/runit themselves. No service-manager code in slice 1.</li>
          <li><Mark err /> <b>do not</b> reintroduce <Mono>polylogue watch</Mono> as a thin wrapper. Clean break per project policy.</li>
          <li><Mark info /> <b>web reader bundle</b> — vendor a static build under <Mono>polylogue/daemon/web/</Mono>; no node toolchain on the user's machine.</li>
        </ul>
      </Card>

      <Card title="follow-up slices" wide>
        <ol className="slices">
          <li><b>slice 2</b> · move <Mono>browser-capture serve</Mono> into <Mono>polylogued</Mono>; <Mono>doctor --target browser-capture</Mono> reads daemon status; extension popup mirrors. Annotations/stars CRUD on conversations.</li>
          <li><b>slice 3</b> · recall-pack builder in web reader (multi-select → bundle). <Mono>polylogue select conversation --print id</Mono> wired to the same builder.</li>
          <li><b>slice 4</b> · split <Mono>polylogue-mcp</Mono> as a separate console_script. MCP talks to daemon if running, falls back direct.</li>
          <li><b>slice 5</b> · retire <Mono>polylogue products</Mono>; expose concrete derived nouns (<Mono>sessions</Mono>, <Mono>threads</Mono>, <Mono>costs</Mono>, <Mono>debt</Mono>) and a compact <Mono>derived status|export</Mono> admin group.</li>
          <li><b>slice 6</b> · upgrade completion: archive-backed <Mono>--repo</Mono>, <Mono>--cwd-prefix</Mono> path-aware, descriptors expose preview fields. Wire <Mono>polylogue select</Mono> to fzf when present.</li>
        </ol>
      </Card>
    </div>
  </div>
);

const Card = ({ title, wide, children }) => (
  <section className={`hcard ${wide ? 'hcard--wide' : ''}`}>
    <header className="hcard__head"><Mono>{title}</Mono></header>
    <div className="hcard__body">{children}</div>
  </section>
);

const Pill = ({ state, children }) => (
  <span className={`pill pill--${state}`}><StateDot state={state} />{children}</span>
);

const Check = ({ ok, muted }) => (
  <span className={`check ${ok ? 'check--ok' : ''} ${muted ? 'check--muted' : ''}`}>
    {ok ? '✓' : muted ? '·' : '○'}
  </span>
);

const Mark = ({ warn, err, info }) => {
  const s = err ? 'err' : warn ? 'warn' : 'info';
  return <StateDot state={s} />;
};

window.Handoff = Handoff;
