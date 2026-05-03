// Surface-boundary diagram — shows the corrected architecture as a single map.
// Top: four executables (polylogue / polylogued / polylogue-mcp / devtools)
// Middle: shared kernel (query · read · status · derived)
// Bottom: storage

const Boundary = () => (
  <div className="boundary">
    <header className="boundary__head">
      <h2>SURFACE BOUNDARIES</h2>
      <p>One archive · one query/read/status/derived kernel · four renderers. No duplicated semantics.</p>
    </header>

    <div className="boundary__top">
      <Surface name="polylogue"
               kind="interactive CLI"
               role="query-first archive grammar"
               examples={[
                 'polylogue [query] [filters] list',
                 'polylogue messages <id>  ·  polylogue raw <id>',
                 'polylogue doctor  ·  polylogue select conversation',
               ]} />
      <Surface name="polylogued"
               kind="long-running daemon"
               role="lives at 127.0.0.1:8765"
               examples={[
                 'live ingestion (claude-code, codex)',
                 'browser-capture receiver + extension API',
                 'local API + web reader',
               ]} />
      <Surface name="polylogue-mcp"
               kind="MCP stdio adapter"
               role="launched by MCP client"
               examples={[
                 '--role read | write | admin',
                 'talks to polylogued if running',
                 'falls back to direct archive access',
               ]} />
      <Surface name="devtools"
               kind="source-checkout only"
               role="repo control plane / proof"
               examples={[
                 'devtools verify  ·  render-all',
                 'lab-corpus seed  ·  affected-obligations',
                 'not on user PATH',
               ]} />
    </div>

    <div className="boundary__bus">
      <span className="boundary__bus-l">SHARED KERNEL</span>
      <div className="boundary__bus-cols">
        <Bus title="query" lines={['ConversationQuerySpec','RootModeRequest','query descriptors','strict validators','completion descriptors']} />
        <Bus title="read"  lines={['conversation header','message page','raw artifact page','session tree']} />
        <Bus title="status" lines={['archive doctor','live ingestion','browser-capture','derived readiness','daemon status']} />
        <Bus title="derived" lines={['session profile','phases','threads','costs','debt']} />
      </div>
    </div>

    <div className="boundary__store">
      <Mono>~/.local/share/polylogue/archive.sqlite</Mono>
      <span>·</span>
      <Mono>fts5</Mono>
      <span>·</span>
      <Mono>content-hash dedupe</Mono>
      <span>·</span>
      <Mono>local-only · no remote sync</Mono>
    </div>

    <div className="boundary__notes">
      <Note title="removed">
        <Mono>polylogue watch</Mono> · <Mono>polylogue browser-capture serve</Mono> ·
        <Mono>polylogue mcp</Mono> · <Mono>polylogue products</Mono>
      </Note>
      <Note title="renamed">
        derived read models surface inline in <Mono>show</Mono>/web reader, plus concrete nouns
        <Mono>sessions</Mono> <Mono>threads</Mono> <Mono>costs</Mono> <Mono>debt</Mono>.
        Compact admin group <Mono>derived status|export</Mono>.
      </Note>
      <Note title="preserved">
        Query-first grammar: <Mono>polylogue [query] [filters] [verb]</Mono>.
        Verbs: <Mono>list count stats show open messages raw bulk-export delete</Mono>.
      </Note>
    </div>
  </div>
);

const Surface = ({ name, kind, role, examples }) => (
  <article className="surface">
    <header>
      <Mono className="surface__name">{name}</Mono>
      <span className="surface__kind">{kind}</span>
    </header>
    <p className="surface__role">{role}</p>
    <ul>
      {examples.map((e, i) => <li key={i}><Mono>{e}</Mono></li>)}
    </ul>
  </article>
);

const Bus = ({ title, lines }) => (
  <div className="bus">
    <SectionLabel>{title}</SectionLabel>
    <ul>{lines.map((l, i) => <li key={i}><Mono>{l}</Mono></li>)}</ul>
  </div>
);

const Note = ({ title, children }) => (
  <div className="boundary__note">
    <SectionLabel>{title}</SectionLabel>
    <div>{children}</div>
  </div>
);

window.Boundary = Boundary;
