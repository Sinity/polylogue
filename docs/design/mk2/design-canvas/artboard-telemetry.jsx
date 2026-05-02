// Telemetry artboard — what the local daemon counts and how each surface reads it.
// Privacy: counters are sqlite-local, never phoned home.

const Telemetry = () => {
  const counters = [
    ['ingest.live.events',           'counter', 'live tail (claude-code, codex)',     '12,431 / 24h'],
    ['ingest.capture.posts',         'counter', 'browser receiver',                    '208 / 24h'],
    ['ingest.import.runs',           'counter', 'manual + scheduled imports',          '6 / 24h'],
    ['query.descriptor.hits{name=}', 'counter', 'how often each filter is used',       'top: provider · since · repo'],
    ['query.latency.p50',            'gauge',   'archive query response (ms)',         '14ms'],
    ['query.latency.p95',            'gauge',   '',                                    '62ms'],
    ['reader.open',                  'counter', 'reader page opens by source',         'web 71% · cli 22% · tui 7%'],
    ['reader.dwell',                 'histogram','seconds per conversation',           'p50 38s · p95 6m'],
    ['recall.packs.created',         'counter', '#558 surface',                        '— pending slice 3'],
    ['mcp.calls{tool=,role=}',       'counter', '#570 surface',                        '— pending slice 4'],
    ['daemon.uptime',                'gauge',   'seconds since boot',                  '2d 04:11:23'],
    ['daemon.lag{component=}',       'gauge',   'per-component freshness',             'live 1.8s · capture 4.6s'],
  ];

  const policy = [
    'sqlite at $XDG_STATE_HOME/polylogue/telemetry.db — same disk as the archive',
    'no network egress — daemon binds 127.0.0.1 only · same posture as the reader',
    '`polylogue doctor --target=daemon` prints the table; `--json` for piping',
    '`polylogue telemetry purge --since=30d` is the only export-shaped command',
    'opt-in flag for sharing: prints to stdout, never auto-uploads',
  ];

  return (
    <div className="telemetry">
      <header className="tele__head">
        <Mono>LOCAL TELEMETRY</Mono>
        <h2>The daemon counts; nothing leaves the machine</h2>
        <p>Every gauge here is readable by the CLI, web reader, and TUI through the same daemon endpoint. Counters live in sqlite next to the archive — local-first, like the rest of the system.</p>
      </header>

      <div className="tele__grid">
        <section className="tele__col">
          <header><Mono>counters · gauges</Mono></header>
          <table className="tele__table">
            <thead><tr><th>name</th><th>kind</th><th>source</th><th>last</th></tr></thead>
            <tbody>
              {counters.map(([n,k,s,l]) => (
                <tr key={n}>
                  <td><Mono>{n}</Mono></td>
                  <td className="tele__kind">{k}</td>
                  <td>{s}</td>
                  <td className="tele__last">{l}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>

        <section className="tele__col tele__col--policy">
          <header><Mono>privacy posture</Mono></header>
          <ul className="tele__policy">
            {policy.map((p,i) => <li key={i}>{p}</li>)}
          </ul>
          <header style={{ marginTop: 16 }}><Mono>example · doctor output</Mono></header>
          <pre className="codeblock">$ polylogue doctor --target=daemon
[ok]   daemon         pid 7842   up 2d 04:11
[ok]   archive        14,031 conversations · 2.1 GB
[warn] capture        lag 4.6s  (extension v0.3.1)
[ok]   live tail      claude-code, codex  lag &lt; 2s
[ok]   reader         http://127.0.0.1:8765
[ok]   mcp            disabled (slice 4)

query.latency      p50 14ms  p95 62ms
top descriptors    provider · since · repo · has-tool
</pre>
        </section>
      </div>
    </div>
  );
};

window.Telemetry = Telemetry;
