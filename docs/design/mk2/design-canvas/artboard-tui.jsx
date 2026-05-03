// TUI cockpit — Textual-style operator dashboard. Rendered as a faux-terminal grid.

const TUICockpit = () => {
  return (
    <div className="tui">
      <div className="tui__chrome">
        <Mono className="t-title">polylogue dashboard · operator cockpit</Mono>
        <Mono className="tui__crumb">F1 help · F2 panel · / search · : palette</Mono>
      </div>
      <div className="tui__grid">
        {/* archive */}
        <Panel title="ARCHIVE" k="1">
          <KV k="conversations" v="412" />
          <KV k="messages"      v="38,212" />
          <KV k="attachments"   v="1,840" />
          <KV k="embeddings"    v="34,901  84.2%" />
          <KV k="schema"        v="v17 · ok" />
          <KV k="fts"           v={<><StateDot state="warn" /> 1,210 pending backfill</>} />
        </Panel>

        {/* live */}
        <Panel title="LIVE" k="2">
          <KV k="claude-code" v={<><StateDot state="ok" /> lag 0.4s · cursor up</>} />
          <KV k="codex"       v={<><StateDot state="ok" /> lag 1.1s · cursor up</>} />
          <KV k="gemini"      v={<><StateDot state="idle" /> not configured</>} />
          <Spark label="ingest 60s" data={[1,2,1,3,4,2,5,6,4,3,8,7,5,4,3,5,6,4,2,3,4,5,4,3,2,4,5,6,7,4]} />
        </Panel>

        {/* capture */}
        <Panel title="CAPTURE" k="3">
          <KV k="receiver"    v={<><StateDot state="ok" /> 127.0.0.1:8765</>} />
          <KV k="extension"   v={<><StateDot state="ok" /> connected · v1.4</>} />
          <KV k="scope"       v="claude.ai · chatgpt.com" />
          <KV k="last"        v={<Mono>12:04:08 · 2c41ed09</Mono>} />
          <KV k="privacy"     v={<><StateDot state="ok" /> page allow-list · no cookies</>} />
        </Panel>

        {/* recent */}
        <Panel title="RECENT" k="4">
          {window.CONVERSATIONS.slice(0, 6).map(c => (
            <div key={c.id} className="recent-row">
              <Mono className="recent-row__t">{fmtTime(c.since)}</Mono>
              <ProviderChip name={c.provider} />
              <Mono className="recent-row__id">{c.id}</Mono>
              <span className="recent-row__title">{c.title}</span>
            </div>
          ))}
        </Panel>

        {/* diagnostics */}
        <Panel title="DIAGNOSTICS" k="5">
          <Diag state="ok"    label="archive integrity"   detail="sha256 manifest match" />
          <Diag state="warn"  label="fts trigger drift"   detail="UPDATE OF body backfill needed" />
          <Diag state="stale" label="derived:costs"        detail="last roll-up 14m ago" />
          <Diag state="ok"    label="schema migrations"   detail="v17 · 0 pending" />
          <Diag state="ok"    label="receiver reachable"  detail="127.0.0.1:8765 200 OK" />
        </Panel>

        {/* next */}
        <Panel title="NEXT" k="6">
          <div className="next">
            <Mono>›</Mono>
            <span>backfill fts triggers — 1,210 rows · ~12s</span>
            <Mono className="muted">polylogue doctor --target fts --fix</Mono>
          </div>
          <div className="next next--muted">
            <Mono>·</Mono>
            <span>roll up costs — 14m stale</span>
            <Mono className="muted">polylogue derived export costs</Mono>
          </div>
          <div className="next next--muted">
            <Mono>·</Mono>
            <span>review #635 daemon split</span>
            <Mono className="muted">polylogue --tag '#635' list</Mono>
          </div>
        </Panel>
      </div>

      <footer className="tui__bar">
        <span><StateDot state="ok" /> polylogued · 4h 12m · pid 184220</span>
        <span>archive 412/38,212</span>
        <span>14 reqs/min</span>
        <span className="tui__bar__right"><Mono>q</Mono> quit · <Mono>?</Mono> help · <Mono>r</Mono> refresh · <Mono>o</Mono> open in web</span>
      </footer>
    </div>
  );
};

const Panel = ({ title, k, children }) => (
  <section className="tpanel">
    <header className="tpanel__head">
      <Mono className="tpanel__k">[{k}]</Mono>
      <Mono className="tpanel__t">{title}</Mono>
    </header>
    <div className="tpanel__body">{children}</div>
  </section>
);

const KV = ({ k, v }) => (
  <div className="kv">
    <Mono className="kv__k">{k}</Mono>
    <span className="kv__v">{v}</span>
  </div>
);

const Diag = ({ state, label, detail }) => (
  <div className="diag">
    <StateDot state={state} />
    <Mono className="diag__l">{label}</Mono>
    <span className="diag__d">{detail}</span>
  </div>
);

const Spark = ({ label, data }) => {
  const max = Math.max(...data);
  const blocks = ' ▁▂▃▄▅▆▇█';
  const s = data.map(v => blocks[Math.round((v / max) * 8)]).join('');
  return (
    <div className="spark">
      <Mono className="spark__l">{label}</Mono>
      <Mono className="spark__d">{s}</Mono>
    </div>
  );
};

window.TUICockpit = TUICockpit;
