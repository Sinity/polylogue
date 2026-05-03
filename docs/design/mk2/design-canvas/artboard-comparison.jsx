// Comparison artboard — Polylogue vs the alternatives a user has today.

const Comparison = () => {
  const cols = [
    { id: 'polylogue', name: 'Polylogue',         sub: 'local-first, query-first' },
    { id: 'native',    name: 'Native exports',    sub: 'Claude.ai · ChatGPT · Cursor' },
    { id: 'wrapper',   name: 'Web wrappers',      sub: 'TypingMind · Msty · LibreChat' },
    { id: 'notes',     name: 'Notes apps',        sub: 'Obsidian · Notion · folders' },
  ];

  const rows = [
    { area: 'archive',
      cells: [
        { v: 'unified across providers · sqlite + raw blobs', t: 'ok' },
        { v: 'one provider per export · manual merge', t: 'no' },
        { v: 'per-app account · cloud-bound', t: 'no' },
        { v: 'manual paste · no schema', t: 'no' },
      ]},
    { area: 'capture',
      cells: [
        { v: 'live tail (claude-code, codex) + browser receiver (slice 2)', t: 'ok' },
        { v: 'on-demand zip download', t: 'mid' },
        { v: 'only what flowed through the wrapper', t: 'mid' },
        { v: 'whatever you remembered to paste', t: 'no' },
      ]},
    { area: 'query',
      cells: [
        { v: 'descriptor table · CLI + facets + palette in lockstep', t: 'ok' },
        { v: 'browser search per export', t: 'no' },
        { v: 'per-app search box · no facets', t: 'mid' },
        { v: 'full-text only · no provider/repo/tool axes', t: 'mid' },
      ]},
    { area: 'reading',
      cells: [
        { v: 'reader + raw + provenance · message-anchored notes', t: 'ok' },
        { v: 'static markdown', t: 'no' },
        { v: 'reading lives inside the wrapper', t: 'mid' },
        { v: 'reading is the strong suit', t: 'ok' },
      ]},
    { area: 'agents',
      cells: [
        { v: 'MCP adapter · read role default · sanitized errors', t: 'ok' },
        { v: 'none', t: 'no' },
        { v: 'wrapper-specific plugins', t: 'mid' },
        { v: 'community plugins', t: 'mid' },
      ]},
    { area: 'privacy',
      cells: [
        { v: 'loopback only · no telemetry leaves disk', t: 'ok' },
        { v: 'cloud-default', t: 'no' },
        { v: 'cloud-default · third-party hosting', t: 'no' },
        { v: 'depends on app · usually cloud', t: 'mid' },
      ]},
    { area: 'cost',
      cells: [
        { v: 'free · open source · uses your existing keys', t: 'ok' },
        { v: 'free with provider account', t: 'ok' },
        { v: '$10–25/mo · plus tokens', t: 'mid' },
        { v: 'subscription model', t: 'mid' },
      ]},
    { area: 'when this is the right tool',
      cells: [
        { v: 'you already have logs from many providers and want one cockpit', t: 'ok' },
        { v: 'you only use one provider and don\'t mind their UI', t: 'mid' },
        { v: 'you want a slick chat UI more than an archive', t: 'mid' },
        { v: 'you mostly write, with light AI use', t: 'mid' },
      ]},
  ];

  return (
    <div className="cmp">
      <header className="cmp__head">
        <Mono>HONEST COMPARISON</Mono>
        <h2>Where Polylogue earns its keep — and where it doesn\u2019t</h2>
        <p>This is a power tool for people whose conversations already live in many places. If you\u2019re happy in one provider\u2019s UI, you don\u2019t need it.</p>
      </header>

      <table className="cmp__table">
        <thead>
          <tr>
            <th></th>
            {cols.map(c => (
              <th key={c.id}>
                <div className="cmp__name">{c.name}</div>
                <div className="cmp__sub"><Mono>{c.sub}</Mono></div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map(r => (
            <tr key={r.area}>
              <th scope="row"><Mono>{r.area}</Mono></th>
              {r.cells.map((c,i) => (
                <td key={i} className={`cmp__cell cmp__cell--${c.t}`}>
                  <span className="cmp__dot"></span>
                  <span>{c.v}</span>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

window.Comparison = Comparison;
