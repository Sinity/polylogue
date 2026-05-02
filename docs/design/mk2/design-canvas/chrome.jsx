// Shared chrome for the cockpit — color tokens, small primitives, mono/sans fonts.
// All artboards import these styles from a single <style id="cockpit-tokens"> block in index.html.

const Tok = ({ children }) => <span className="tok">{children}</span>;
const Mono = ({ children, className = '' }) => <span className={`mono ${className}`}>{children}</span>;

// status dot — desaturated cybernetic palette
const StateDot = ({ state }) => {
  const map = {
    ok:    { c: '#6fae8f', label: 'ok'    },
    warn:  { c: '#caa15a', label: 'warn'  },
    stale: { c: '#caa15a', label: 'stale' },
    err:   { c: '#c97264', label: 'err'   },
    idle:  { c: '#56606a', label: 'idle'  },
    info:  { c: '#7fa3b5', label: 'info'  },
  };
  const s = map[state] ?? map.idle;
  return <span className="state-dot" style={{ '--c': s.c }} title={s.label} />;
};

// provider chip — uses provider hex but desaturated bg via low alpha
const ProviderChip = ({ name }) => {
  const p = (window.PROVIDERS && window.PROVIDERS[name]) || { hex: '#94a3b8', label: name };
  return (
    <span className="provider-chip" style={{ '--p': p.hex }}>
      <span className="provider-chip__dot" />
      <Mono>{p.label}</Mono>
    </span>
  );
};

// keycap
const Key = ({ children }) => <kbd className="key">{children}</kbd>;

// section label — small caps, slate
const SectionLabel = ({ children }) => <div className="section-label">{children}</div>;

// thin horizontal rule
const Hr = () => <div className="hr" />;

// tiny inline meta
const Meta = ({ k, v }) => (
  <span className="meta"><span className="meta-k">{k}</span><span className="meta-v">{v}</span></span>
);

// fmt helpers
const fmtTime = (iso) => {
  const d = new Date(iso);
  const now = new Date('2026-05-02T12:05:00Z');
  const diff = (now - d) / 1000;
  if (diff < 60)        return `${Math.round(diff)}s ago`;
  if (diff < 3600)      return `${Math.round(diff/60)}m ago`;
  if (diff < 86_400)    return `${Math.round(diff/3600)}h ago`;
  return `${Math.round(diff/86_400)}d ago`;
};
const fmtNum = (n) => n.toLocaleString('en-US');

Object.assign(window, { Tok, Mono, StateDot, ProviderChip, Key, SectionLabel, Hr, Meta, fmtTime, fmtNum });
