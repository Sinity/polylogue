// Roadmap artboard — slices through the open issue stack, ordered by leverage.

const Roadmap = () => {
  const slices = [
    { n: 1, title: 'daemon ground truth', weeks: 'wk 1–2', issues: ['#621','#570'],
      ships: [
        'split `polylogued` console_script + pid file',
        'serve vendored web reader at 127.0.0.1:8765',
        'unify status under DaemonStatus → CLI doctor + TUI both consume it',
        'collapse query parsing into one descriptor table (#621)',
      ],
      proves: 'one source of truth for "is the archive healthy?" — every other surface reads from it',
      risk: 'low — moves existing code behind a port',
    },
    { n: 2, title: 'browser capture · receiver', weeks: 'wk 3–4', issues: ['#566'],
      ships: [
        'browser-extension HAR-style capture posting to daemon',
        'receiver normalizes Claude.ai + ChatGPT web traffic into stored conversations',
        'web reader gains <CaptureBadge> on conversations sourced via capture',
      ],
      proves: 'covers the Claude.ai gap before official export catches up',
      risk: 'med — DOM/API drift on provider sites; receiver is the seam',
    },
    { n: 3, title: 'recall packs', weeks: 'wk 5', issues: ['#558'],
      ships: [
        '<RecallBuilder> in web reader · multi-select → bundle',
        '`polylogue recall new --from <spec>` parity command',
        'pack format = signed manifest + message refs (no payload duplication)',
      ],
      proves: 'archive becomes a reading list, not a junk drawer',
      risk: 'low — additive feature on top of slice 1',
    },
    { n: 4, title: 'mcp adapter', weeks: 'wk 6', issues: ['#570'],
      ships: [
        '`polylogue-mcp` console_script (separate binary)',
        'tools: search · read · annotate (gated behind read role)',
        'sanitized errors per #570 — no raw stack traces over MCP',
      ],
      proves: 'agents can read the archive without re-implementing query',
      risk: 'med — role gating must be airtight',
    },
    { n: 5, title: 'annotations + saved views', weeks: 'wk 7', issues: ['—'],
      ships: [
        'message-anchored notes (markdown, sqlite-backed)',
        'saved view = filter spec + sort, not a copy of results',
        'fuzzy selector and palette gain `view:` prefix',
      ],
      proves: 'durable second-brain layer over the archive',
      risk: 'low',
    },
    { n: 6, title: 'time-series + sparkline backfill', weeks: 'wk 8', issues: ['—'],
      ships: [
        'rollups: per-day token + cost + provider counts',
        'TUI StatCards switch from synthetic to real sparklines',
        'web reader gains a quiet activity strip',
      ],
      proves: 'the cockpit stops being decorative',
      risk: 'low — pure read-side rollup',
    },
  ];

  return (
    <div className="roadmap">
      <header className="roadmap__head">
        <Mono>ROADMAP · 8 weeks · single maintainer</Mono>
        <h2>Sequenced for compounding leverage, not feature breadth</h2>
        <p>Each slice ends in a runnable artifact a user can poke at. Order is fixed: ground truth before capture, capture before bundling, bundling before agents.</p>
      </header>

      <ol className="slices2">
        {slices.map(s => (
          <li key={s.n}>
            <div className="slice2__rail">
              <div className="slice2__num">{String(s.n).padStart(2,'0')}</div>
              <div className="slice2__line"></div>
            </div>
            <div className="slice2__body">
              <header>
                <Mono className="slice2__weeks">{s.weeks}</Mono>
                <h3>{s.title}</h3>
                <span className="slice2__issues">
                  {s.issues.map(i => <span key={i} className="pill">{i}</span>)}
                </span>
              </header>
              <ul className="slice2__ships">
                {s.ships.map((line,i) => <li key={i}>{line}</li>)}
              </ul>
              <footer>
                <span><Mono>proves</Mono> {s.proves}</span>
                <span><Mono>risk</Mono> {s.risk}</span>
              </footer>
            </div>
          </li>
        ))}
      </ol>
    </div>
  );
};

window.Roadmap = Roadmap;
