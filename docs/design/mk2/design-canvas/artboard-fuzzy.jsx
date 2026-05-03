// Fuzzy selector — `polylogue select conversation` — fzf-style with descriptors.

const FuzzySelect = () => {
  const [q, setQ] = React.useState('daemon');
  const [idx, setIdx] = React.useState(0);

  const score = (s, q) => {
    s = s.toLowerCase(); q = q.toLowerCase();
    if (!q) return 0;
    let si = 0, sc = 0, last = -1;
    for (const ch of q) {
      const f = s.indexOf(ch, si);
      if (f < 0) return -1;
      sc += (last >= 0 && f === last + 1) ? 3 : 1;
      last = f; si = f + 1;
    }
    return sc;
  };

  const ranked = window.CONVERSATIONS
    .map(c => ({ c, s: Math.max(score(c.title, q), score(c.lastTurn, q), score(c.repo || '', q)) }))
    .filter(x => x.s > 0)
    .sort((a, b) => b.s - a.s)
    .slice(0, 10);

  const sel = ranked[Math.min(idx, ranked.length - 1)]?.c;

  return (
    <div className="fuzzy">
      <div className="fuzzy__chrome">
        <Mono className="t-title">polylogue select conversation · zsh</Mono>
      </div>
      <div className="fuzzy__body">
        <div className="fuzzy__cmd">
          <Mono className="t-prompt">~ ❯</Mono>
          <Mono>polylogue --repo polylogue select conversation --recent</Mono>
        </div>

        <div className="fuzzy__panel">
          {/* left: result list */}
          <div className="fuzzy__list">
            <div className="fuzzy__query">
              <Mono>›</Mono>
              <input value={q} onChange={e => { setQ(e.target.value); setIdx(0); }}
                     spellCheck={false} autoFocus />
              <Mono className="fuzzy__count">{ranked.length}/{window.CONVERSATIONS.length}</Mono>
            </div>
            <ul>
              {ranked.map((r, i) => (
                <li key={r.c.id} className={i === idx ? 'on' : ''} onClick={() => setIdx(i)}>
                  <Mono className="fuzzy__pre">{i === idx ? '▌' : ' '}</Mono>
                  <ProviderChip name={r.c.provider} />
                  <Mono className="fuzzy__id">{r.c.id}</Mono>
                  <span className="fuzzy__title">{r.c.title}</span>
                  <Mono className="fuzzy__repo">{r.c.repo || '—'}</Mono>
                </li>
              ))}
            </ul>
            <div className="fuzzy__keys">
              <Mono>↑↓</Mono> nav · <Mono>↵</Mono> open · <Mono>tab</Mono> multi-select · <Mono>^p</Mono> print id · <Mono>esc</Mono> cancel
            </div>
          </div>

          {/* right: preview */}
          <aside className="fuzzy__preview">
            {sel ? (
              <>
                <div className="fuzzy__phead">
                  <Mono>{sel.id}</Mono>
                  <span>· {fmtTime(sel.since)}</span>
                </div>
                <div className="fuzzy__ptitle">{sel.title}</div>
                <div className="fuzzy__pmeta">
                  <Meta k="provider" v={<ProviderChip name={sel.provider} />} />
                  <Meta k="repo"     v={<Mono>{sel.repo || '—'}</Mono>} />
                  <Meta k="cwd"      v={<Mono>{sel.cwd || '—'}</Mono>} />
                  <Meta k="msgs"     v={<Mono>{sel.msgs}</Mono>} />
                  <Meta k="tokens"   v={<Mono>{fmtNum(sel.tokens)}</Mono>} />
                </div>
                <Hr />
                <div className="fuzzy__plast">
                  <SectionLabel>last turn</SectionLabel>
                  <p>{sel.lastTurn}</p>
                </div>
                <div className="fuzzy__pwhere">
                  <SectionLabel>completion source</SectionLabel>
                  <Mono>conversation_id :: ConversationDescriptor</Mono>
                  <Mono className="muted">archive-backed · provider/title hinted · &lt;5ms</Mono>
                </div>
              </>
            ) : <p className="muted">no match</p>}
          </aside>
        </div>
      </div>
    </div>
  );
};

window.FuzzySelect = FuzzySelect;
