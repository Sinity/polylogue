// States artboard — empty / loading / degraded / failure / privacy across surfaces.
// Compact 4-col grid showing each state's microcopy, affordance, and recovery.

const States = () => (
  <div className="states">
    <header className="states__head">
      <Mono>STATES</Mono>
      <h2>Empty · loading · degraded · failure · privacy</h2>
      <p>Every surface must render these. Calm persistent status; never a blocking modal. Recovery affordance is always a CLI command the user could also run.</p>
    </header>

    <div className="states__grid">
      {/* WEB READER */}
      <StateCol surface="web reader · /">
        <StateRow tone="empty" title="no archive yet">
          <p>No conversations. The daemon is running, the archive is just empty.</p>
          <Mono className="cta">polylogue run --input ~/Downloads/conversations.json</Mono>
        </StateRow>
        <StateRow tone="loading" title="indexing 2,140 messages">
          <Bar pct={62} />
          <p className="muted">fts backfill · ~14s remaining · safe to keep browsing</p>
        </StateRow>
        <StateRow tone="degraded" title="fts trigger drift">
          <p>Title search is current; body search may miss recent edits.</p>
          <Mono className="cta">polylogue doctor --target fts --fix</Mono>
        </StateRow>
        <StateRow tone="failure" title="archive locked">
          <p>Another writer holds the SQLite write lock.</p>
          <Mono className="cta">polylogue doctor --target lock</Mono>
        </StateRow>
        <StateRow tone="privacy" title="private window">
          <p>Capture paused while a private browser window is foregrounded.</p>
          <Mono className="cta muted">extension policy · page allow-list</Mono>
        </StateRow>
      </StateCol>

      {/* CLI */}
      <StateCol surface="polylogue · CLI">
        <StateRow tone="empty" title="no matches">
<pre>{`polylogue "fts trigger" --provider chatgpt list
0 conversations matched.
hint: drop --provider, or try \`polylogue select conversation\`.`}</pre>
        </StateRow>
        <StateRow tone="loading" title="streaming">
<pre>{`polylogue --tail --provider claude-code list
[live] 1 new · 0.4s lag · ⌃C to stop`}</pre>
        </StateRow>
        <StateRow tone="degraded" title="daemon down">
<pre>{`polylogue --latest open
warn: polylogued not running · falling back to static archive
opening file:///.../site/c/7b3c2e91.html`}</pre>
        </StateRow>
        <StateRow tone="failure" title="strict validator">
<pre>{`polylogue --has bogus list
error: --has  invalid value 'bogus'
choices: tool_use thinking paste
exit 2`}</pre>
        </StateRow>
        <StateRow tone="privacy" title="machine output">
<pre className="muted">{`polylogue list --format json
# stdout: structured json only · no banner · safe for piping`}</pre>
        </StateRow>
      </StateCol>

      {/* TUI */}
      <StateCol surface="dashboard · TUI">
        <StateRow tone="empty" title="no live source">
          <KV k="claude-code" v={<><StateDot state="idle" /> not configured</>} />
          <p className="muted">enable from <Mono>polylogued enable-source claude-code</Mono></p>
        </StateRow>
        <StateRow tone="loading" title="seeding">
          <KV k="archive" v={<><StateDot state="info" /> seeding · 1,840 / 12,400</>} />
          <Bar pct={14} />
        </StateRow>
        <StateRow tone="degraded" title="receiver unreachable">
          <KV k="receiver" v={<><StateDot state="warn" /> 127.0.0.1:8765 · refused</>} />
          <p className="muted">port collision · check <Mono>polylogued status</Mono></p>
        </StateRow>
        <StateRow tone="failure" title="schema drift">
          <KV k="schema" v={<><StateDot state="err" /> v17 archive · v18 expected</>} />
          <Mono className="cta">polylogue schema migrate</Mono>
        </StateRow>
        <StateRow tone="privacy" title="capture paused">
          <KV k="capture" v={<><StateDot state="idle" /> paused · private window</>} />
          <p className="muted">no events buffered · no leakage</p>
        </StateRow>
      </StateCol>

      {/* DAEMON */}
      <StateCol surface="polylogued · daemon">
        <StateRow tone="empty" title="cold start">
<pre>{`polylogued run
[boot] archive ~/.local/share/polylogue/archive.sqlite
[boot] no live sources configured · serving read-only
listening on http://127.0.0.1:8765`}</pre>
        </StateRow>
        <StateRow tone="loading" title="catching up">
<pre>{`[live:claude-code] catching up · 412 events · 1.8s lag`}</pre>
        </StateRow>
        <StateRow tone="degraded" title="one source stale">
<pre>{`[live:codex] WARN cursor stale 14m · holding · /api/health → warn`}</pre>
        </StateRow>
        <StateRow tone="failure" title="port in use">
<pre>{`[serve] FATAL :8765 already in use (pid 7144 · receiver)
hint: stop the standalone receiver; daemon owns the port now`}</pre>
        </StateRow>
        <StateRow tone="privacy" title="bind">
<pre className="muted">{`[serve] bind 127.0.0.1 only · no 0.0.0.0 · no token in URL`}</pre>
        </StateRow>
      </StateCol>
    </div>
  </div>
);

const StateCol = ({ surface, children }) => (
  <div className="scol">
    <header className="scol__head"><Mono>{surface}</Mono></header>
    <div className="scol__body">{children}</div>
  </div>
);

const StateRow = ({ tone, title, children }) => (
  <div className={`srow srow--${tone}`}>
    <div className="srow__head">
      <span className={`srow__tone srow__tone--${tone}`}>{tone}</span>
      <span className="srow__title">{title}</span>
    </div>
    <div className="srow__body">{children}</div>
  </div>
);

const Bar = ({ pct }) => (
  <div className="bar"><span style={{ width: `${pct}%` }} /></div>
);

window.States = States;
