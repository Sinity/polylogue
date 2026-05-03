// CLI artboard — query-first terminal session with dynamic completion popup.

const TerminalCLI = () => {
  // Show a sequence of "frozen" terminal sessions side-by-side, plus one live one with a completion popup.
  return (
    <div className="terminal">
      <div className="terminal__chrome">
        <div className="t-traffic"><span /><span /><span /></div>
        <Mono className="t-title">~/src/polylogue · zsh</Mono>
      </div>
      <div className="terminal__body">

        <Block prompt={'~/src/polylogue ❯'} cmd={'polylogue'} />
        <Pre>
{`polylogue · 412 conversations · 4 providers · ~38.2k messages · $74.10 estimated
  claude-code  ████████████░░░░░░░  248
  chatgpt      ████░░░░░░░░░░░░░░░   72
  claude-ai    ███░░░░░░░░░░░░░░░░   58
  codex        ██░░░░░░░░░░░░░░░░░   34

next: \`polylogue --latest\`  ·  \`polylogue --provider claude-code list\`
`}
        </Pre>

        <Block prompt={'~/src/polylogue ❯'}
               cmd={<><Mono>polylogue</Mono> <span className="cli-q">"fts trigger"</span> <Mono>--repo polylogue --since "last week" list</Mono></>} />
        <Table headers={['id','provider','repo','msgs','tok','since','title']}
               rows={[
                 ['f0a2bb73','claude-code','polylogue', '53', '31.9k', '4d ago', 'fts trigger drift on UPDATE OF title'],
                 ['8d821a4f','claude-code','polylogue', '38', '24.2k', '2d ago', 'query descriptor cleanup — strict validators'],
                 ['904ee18f','claude-code','polylogue', '24', '11.4k', '4d ago', 'doctor --target browser-capture'],
               ]} />

        <Block prompt={'~/src/polylogue ❯'}
               cmd={<><Mono>polylogue --has thinking --sort tokens list --limit 3 --format json</Mono></>} />
        <Pre className="json">
{`[
  {"id":"7b3c2e91","provider":"claude-code","msgs":71,"tokens":52404,
   "title":"polylogued daemon split: live + receiver + serve",
   "has":["tool_use","thinking","paste"], "since":"2026-04-30T14:02Z"},
  {"id":"f0a2bb73","provider":"claude-code","msgs":53,"tokens":31900,
   "title":"fts trigger drift on UPDATE OF title",
   "has":["tool_use","thinking"], "since":"2026-04-27T22:01Z"},
  {"id":"8d821a4f","provider":"claude-code","msgs":38,"tokens":24180,
   "title":"query descriptor cleanup — strict validators",
   "has":["tool_use","thinking"], "since":"2026-04-30T18:21Z"}
]`}
        </Pre>

        {/* Live session with completion popup */}
        <Block prompt={'~/src/polylogue ❯'}
               cmd={<><Mono>polylogue --provider </Mono><span className="cli-cursor">claude-c</span><span className="cli-blink" /></>} />
        <CompletionPopup
          values={[
            { v: 'claude-code', d: '248 conversations · live ingestion ok' },
            { v: 'claude-ai',   d: '58 conversations · browser capture' },
          ]}
          highlight={0}
          source={'archive-backed (--provider :: ProviderDescriptor)'}
        />

        <Hr />

        <Block prompt={'~/src/polylogue ❯'}
               cmd={<><Mono>polylogue messages 7b3c2e91 --message-role user --limit 3</Mono></>} />
        <Pre>
{`#01 user · 14:02:11 · 84 tok
  split browser-capture serve out of click_app — should it move to polylogued
  or stay reachable as \`polylogue browser-capture serve\`?

#03 user · 14:03:02 · 12 tok
  and \`polylogue mcp\`?

#05 user · 14:05:44 · 22 tok
  show me the click_app diff for removing the serve command

next: polylogue messages 7b3c2e91 --limit 3 --offset 3`}
        </Pre>

        <Block prompt={'~/src/polylogue ❯'}
               cmd={<><Mono>polylogue --latest open</Mono></>} />
        <Pre className="muted">
{`opening 7b3c2e91 in http://127.0.0.1:8765/c/7b3c2e91 (polylogued)`}
        </Pre>

      </div>
    </div>
  );
};

const Block = ({ prompt, cmd }) => (
  <div className="t-line">
    <Mono className="t-prompt">{prompt}</Mono>
    <span className="t-cmd">{typeof cmd === 'string' ? <Mono>{cmd}</Mono> : cmd}</span>
  </div>
);

const Pre = ({ children, className = '' }) => (
  <pre className={`t-pre ${className}`}>{children}</pre>
);

const Table = ({ headers, rows }) => (
  <div className="t-table">
    <div className="t-table__row t-table__row--head">
      {headers.map((h, i) => <Mono key={i}>{h}</Mono>)}
    </div>
    {rows.map((r, i) => (
      <div key={i} className="t-table__row">
        {r.map((c, j) => <Mono key={j}>{c}</Mono>)}
      </div>
    ))}
  </div>
);

const CompletionPopup = ({ values, highlight, source }) => (
  <div className="completion">
    <div className="completion__inner">
      {values.map((v, i) => (
        <div key={v.v} className={`completion__row ${i === highlight ? 'on' : ''}`}>
          <Mono className="completion__v">{v.v}</Mono>
          <span className="completion__d">{v.d}</span>
        </div>
      ))}
      <div className="completion__src">
        <Mono>↑↓</Mono> select · <Mono>↵</Mono> accept · source: <Mono>{source}</Mono>
      </div>
    </div>
  </div>
);

window.TerminalCLI = TerminalCLI;
