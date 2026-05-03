// Web reader artboard — local /127.0.0.1 polylogued web app.
// Search + facets on the left, conversation list, reader pane, raw/provenance side panel.

const WebReader = () => {
  const [q, setQ] = React.useState('');
  const [provider, setProvider] = React.useState(null);
  const [repo, setRepo] = React.useState(null);
  const [hasFlag, setHasFlag] = React.useState(null);
  const [selected, setSelected] = React.useState('7b3c2e91');
  const [side, setSide] = React.useState('messages'); // messages | raw | provenance

  const filtered = window.CONVERSATIONS.filter(c => {
    if (q && !(c.title.toLowerCase().includes(q.toLowerCase()) ||
               c.lastTurn.toLowerCase().includes(q.toLowerCase()) ||
               (c.tags||[]).some(t => t.includes(q.toLowerCase())))) return false;
    if (provider && c.provider !== provider) return false;
    if (repo && c.repo !== repo) return false;
    if (hasFlag && !(c.has||[]).includes(hasFlag)) return false;
    return true;
  });

  const conv = window.CONV_DETAIL[selected];

  const repos = Array.from(new Set(window.CONVERSATIONS.map(c => c.repo).filter(Boolean)));

  return (
    <div className="webreader">
      {/* address bar / chrome */}
      <div className="webreader__chrome">
        <div className="wr-traffic">
          <span /><span /><span />
        </div>
        <div className="wr-url">
          <Mono>http://127.0.0.1:8765/</Mono>
          <span className="wr-url__cwd">— polylogued · local-only</span>
        </div>
        <div className="wr-actions">
          <Mono>⌘K</Mono>
        </div>
      </div>

      <div className="webreader__body">
        {/* facets */}
        <aside className="wr-facets">
          <div className="wr-search">
            <Mono>/</Mono>
            <input
              value={q} onChange={e => setQ(e.target.value)}
              placeholder="query terms"
              spellCheck={false}
            />
          </div>

          <SectionLabel>provider</SectionLabel>
          <ul className="facet-list">
            {Object.keys(window.PROVIDERS).map(p => {
              const n = window.CONVERSATIONS.filter(c => c.provider === p).length;
              if (!n) return null;
              return (
                <li key={p}
                    className={provider === p ? 'on' : ''}
                    onClick={() => setProvider(provider === p ? null : p)}>
                  <ProviderChip name={p} />
                  <Mono className="facet-n">{n}</Mono>
                </li>
              );
            })}
          </ul>

          <SectionLabel>repo</SectionLabel>
          <ul className="facet-list">
            {repos.map(r => {
              const n = window.CONVERSATIONS.filter(c => c.repo === r).length;
              return (
                <li key={r}
                    className={repo === r ? 'on' : ''}
                    onClick={() => setRepo(repo === r ? null : r)}>
                  <Mono>{r}</Mono><Mono className="facet-n">{n}</Mono>
                </li>
              );
            })}
          </ul>

          <SectionLabel>has</SectionLabel>
          <ul className="facet-list">
            {['tool_use','thinking','paste'].map(h => {
              const n = window.CONVERSATIONS.filter(c => (c.has||[]).includes(h)).length;
              return (
                <li key={h}
                    className={hasFlag === h ? 'on' : ''}
                    onClick={() => setHasFlag(hasFlag === h ? null : h)}>
                  <Mono>{h}</Mono><Mono className="facet-n">{n}</Mono>
                </li>
              );
            })}
          </ul>

          <Hr />
          <SectionLabel>saved views</SectionLabel>
          <ul className="facet-list saved">
            <li><Mono>★ debt &amp; #621 thread</Mono></li>
            <li><Mono>★ daemon design</Mono></li>
            <li><Mono>★ fts incidents</Mono></li>
          </ul>
        </aside>

        {/* result list */}
        <section className="wr-list">
          <header>
            <Mono>{filtered.length}</Mono>
            <span> conversations · sorted by recency</span>
            {(provider || repo || hasFlag || q) && (
              <button className="clear" onClick={() => { setProvider(null); setRepo(null); setHasFlag(null); setQ(''); }}>
                clear filters
              </button>
            )}
          </header>
          <ul>
            {filtered.map(c => (
              <li key={c.id}
                  className={selected === c.id ? 'on' : ''}
                  onClick={() => setSelected(c.id)}>
                <div className="wr-list__top">
                  <ProviderChip name={c.provider} />
                  <Mono className="wr-list__id">{c.id}</Mono>
                  <span className="wr-list__since">{fmtTime(c.since)}</span>
                </div>
                <div className="wr-list__title">{c.title}</div>
                <div className="wr-list__meta">
                  {c.repo && <Mono className="wr-list__repo">{c.repo}</Mono>}
                  <Mono>{c.msgs} msgs</Mono>
                  <Mono>{(c.tokens/1000).toFixed(1)}k tok</Mono>
                  {(c.has||[]).map(h => <span key={h} className="has-tag">{h}</span>)}
                </div>
              </li>
            ))}
          </ul>
        </section>

        {/* reader */}
        <section className="wr-reader">
          {conv ? <ReaderBody conv={conv} /> : <ReaderEmpty id={selected} />}
        </section>

        {/* side panel */}
        <aside className="wr-side">
          <div className="wr-side__tabs">
            {['messages','raw','provenance'].map(t => (
              <button key={t}
                      className={side === t ? 'on' : ''}
                      onClick={() => setSide(t)}>
                <Mono>{t}</Mono>
              </button>
            ))}
          </div>
          <div className="wr-side__body">
            {conv && side === 'messages' && <MessageOutline conv={conv} />}
            {conv && side === 'raw'      && <RawList conv={conv} />}
            {conv && side === 'provenance' && <Provenance conv={conv} />}
          </div>
        </aside>
      </div>

      {/* status strip */}
      <footer className="wr-status">
        <span><StateDot state="ok" /> <Mono>polylogued</Mono> · 4h 12m</span>
        <span><StateDot state="ok" /> live:claude-code · 0.4s lag</span>
        <span><StateDot state="ok" /> capture · ext connected</span>
        <span><StateDot state="warn" /> fts · 1,210 pending</span>
        <span className="wr-status__right"><Mono>⌘K</Mono> palette · <Mono>/</Mono> search · <Mono>g a</Mono> annotations</span>
      </footer>
    </div>
  );
};

const ReaderEmpty = ({ id }) => (
  <div className="reader-empty">
    <Mono>conversation {id}</Mono>
    <p>full reader for this id is rendered on demand by the web app — header, messages, raw artifacts come from the same query/read kernel the CLI uses.</p>
    <p className="muted">switch to <Mono>7b3c2e91</Mono> to see a populated example.</p>
  </div>
);

const ReaderBody = ({ conv }) => {
  const h = conv.header;
  return (
    <article className="reader">
      <header className="reader__head">
        <div className="reader__crumb">
          <Mono>{h.id.slice(0,8)}</Mono><span>›</span>
          <ProviderChip name={h.provider} /><span>›</span>
          <Mono>{h.repo}</Mono>
        </div>
        <h1>polylogued daemon split: live + receiver + serve</h1>
        <div className="reader__meta">
          <Meta k="cwd"     v={<Mono>{h.cwd}</Mono>} />
          <Meta k="model"   v={<Mono>{h.model}</Mono>} />
          <Meta k="started" v={<Mono>{h.started}</Mono>} />
          <Meta k="msgs"    v={<Mono>{h.msgs}</Mono>} />
          <Meta k="tokens"  v={<Mono>{fmtNum(h.tokens)}</Mono>} />
          <Meta k="cost"    v={<Mono>{h.cost}</Mono>} />
        </div>
        <div className="reader__tags">
          {h.tags.map(t => <span key={t} className="tag">{t}</span>)}
          {h.has.map(t => <span key={t} className="has-tag">{t}</span>)}
        </div>
        <div className="reader__actions">
          <button><Key>S</Key> star</button>
          <button><Key>P</Key> pin</button>
          <button><Key>A</Key> annotate</button>
          <button><Key>R</Key> recall pack</button>
          <button className="ghost"><Mono>polylogue messages {h.id.slice(0,8)} --limit 50</Mono></button>
        </div>
      </header>

      <div className="reader__messages">
        {conv.messages.map(m => <Message key={m.i} m={m} />)}
        <div className="reader__more">
          <Mono>↓ messages 8–71</Mono> · paginated · <Mono>polylogue messages {h.id.slice(0,8)} --offset 7</Mono>
        </div>
      </div>
    </article>
  );
};

const Message = ({ m }) => (
  <div className={`msg msg--${m.role}`}>
    <div className="msg__gutter">
      <Mono className="msg__i">{String(m.i).padStart(2,'0')}</Mono>
      <Mono className="msg__t">{m.t}</Mono>
    </div>
    <div className="msg__body">
      <div className="msg__role">
        <Mono>{m.role}</Mono>
        {(m.has||[]).map(h => <span key={h} className="has-tag">{h}</span>)}
      </div>
      <pre className="msg__text">{m.body}</pre>
    </div>
  </div>
);

const MessageOutline = ({ conv }) => (
  <ul className="outline">
    {conv.messages.map(m => (
      <li key={m.i}>
        <Mono className="outline__i">{String(m.i).padStart(2,'0')}</Mono>
        <Mono className={`outline__role outline__role--${m.role}`}>{m.role}</Mono>
        <span className="outline__t">{m.body.slice(0, 56)}{m.body.length > 56 ? '…' : ''}</span>
      </li>
    ))}
  </ul>
);

const RawList = ({ conv }) => (
  <ul className="raw">
    {conv.raw.map((r, i) => (
      <li key={i}>
        <Mono className="raw__t">{r.t}</Mono>
        <Mono className="raw__k">{r.kind}</Mono>
        <span className="raw__s">{r.summary}</span>
      </li>
    ))}
    <li className="raw__hint muted">
      <Mono>polylogue raw {conv.header.id.slice(0,8)} --format json</Mono>
    </li>
  </ul>
);

const Provenance = ({ conv }) => (
  <div className="prov">
    <SectionLabel>source</SectionLabel>
    <Meta k="ingest"  v={<Mono>live:claude-code</Mono>} />
    <Meta k="path"    v={<Mono>~/.claude/projects/polylogue/sessions/7b3c…jsonl</Mono>} />
    <Meta k="cursor"  v={<Mono>byte 144,112 / 144,112</Mono>} />
    <Meta k="hashed"  v={<Mono>sha256:9f8e…</Mono>} />
    <Hr />
    <SectionLabel>derived</SectionLabel>
    <Meta k="session" v={<Mono>412 events · profile ok</Mono>} />
    <Meta k="phases"  v={<Mono>3 (design · cleanup · plan)</Mono>} />
    <Meta k="thread"  v={<Mono>↗ 8d821a4f, 904ee18f</Mono>} />
    <Meta k="costs"   v={<Mono>$0.91 · model claude-sonnet-4.5</Mono>} />
    <Hr />
    <SectionLabel>privacy</SectionLabel>
    <div className="prov__priv">
      <StateDot state="ok" /> <span>local archive only · no remote sync</span>
    </div>
  </div>
);

window.WebReader = WebReader;
