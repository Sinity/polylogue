// Synthetic Polylogue corpus and shared kernel state.
// All artboards read from this — same query/read/status models, different renderers.

const PROVIDERS = {
  'claude-code':   { hex: '#d97757', label: 'claude-code' },
  'claude-ai':     { hex: '#d97757', label: 'claude-ai' },
  'chatgpt':       { hex: '#10a37f', label: 'chatgpt' },
  'codex':         { hex: '#00bcd4', label: 'codex' },
  'gemini':        { hex: '#4285f4', label: 'gemini' },
};

const CONVERSATIONS = [
  { id: '8d821a4f', provider: 'claude-code', repo: 'polylogue', cwd: '~/src/polylogue', title: 'query descriptor cleanup — strict validators',
    msgs: 38, tokens: 24_180, cost: 0.42, since: '2026-04-30T18:21Z', tags: ['descriptors','cleanup'],
    has: ['tool_use','thinking'], lastTurn: 'fix(query): tighten --has-type validator against descriptor enum', actions: ['file_edit','shell','search'] },
  { id: '7b3c2e91', provider: 'claude-code', repo: 'polylogue', cwd: '~/src/polylogue/cli', title: 'polylogued daemon split: live + receiver + serve',
    msgs: 71, tokens: 52_404, cost: 0.91, since: '2026-04-30T14:02Z', tags: ['daemon','architecture'],
    has: ['tool_use','thinking','paste'], lastTurn: 'split browser-capture serve out of click_app; add polylogued run', actions: ['file_edit','file_read','shell'] },
  { id: '4f10aa2d', provider: 'claude-code', repo: 'sinex',     cwd: '~/src/sinex',     title: 'session-unit provenance: shell + browser join',
    msgs: 22, tokens: 9_870, cost: 0.18, since: '2026-04-30T10:15Z', tags: ['provenance'],
    has: ['tool_use'], lastTurn: 'verify shell-cwd ↔ browser-tab join key on synthetic events', actions: ['file_read','search'] },
  { id: 'a90bf311', provider: 'codex',       repo: 'polylogue', cwd: '~/src/polylogue', title: 'raw artifacts vs provider records — naming truth',
    msgs: 14, tokens: 5_201, cost: 0.07, since: '2026-04-29T22:48Z', tags: ['naming','#621'],
    has: [], lastTurn: 'rename internal raw_records → raw_artifacts; CLI keeps `raw`', actions: ['file_edit'] },
  { id: '2c41ed09', provider: 'claude-ai',   repo: null,         cwd: null,              title: 'cold monochrome cockpit — palette discussion',
    msgs: 11, tokens: 4_600, cost: 0.05, since: '2026-04-29T20:12Z', tags: ['design'],
    has: ['paste'], lastTurn: 'graphite #0e0f12 + slate-300 fg + amber alert; no purple', actions: [] },
  { id: 'e5d7102b', provider: 'chatgpt',     repo: null,         cwd: null,              title: 'rsync incantation for archive snapshot',
    msgs: 6, tokens: 1_840, cost: 0.02, since: '2026-04-29T16:04Z', tags: [],
    has: [], lastTurn: 'rsync -aHAX --delete-after with sentinel manifest', actions: [] },
  { id: '3a6e8f44', provider: 'claude-code', repo: 'thoughtspace', cwd: '~/notes/thoughtspace', title: 'roadmap: collapse products → derived views',
    msgs: 29, tokens: 14_220, cost: 0.31, since: '2026-04-28T23:55Z', tags: ['roadmap','#624'],
    has: ['thinking'], lastTurn: 'derived sessions/threads/costs/debt; products gone', actions: ['file_edit'] },
  { id: 'b1029ccd', provider: 'claude-code', repo: 'polylogue', cwd: '~/src/polylogue', title: 'completion descriptors: --repo from session profile',
    msgs: 17, tokens: 7_010, cost: 0.13, since: '2026-04-28T19:33Z', tags: ['completion'],
    has: ['tool_use'], lastTurn: 'archive-backed --repo completion; not tag-based', actions: ['file_edit','shell'] },
  { id: 'cc44918a', provider: 'gemini',      repo: null,         cwd: null,              title: 'sqlite WAL on tmpfs — durability tradeoff',
    msgs: 9, tokens: 3_120, cost: 0.04, since: '2026-04-28T11:12Z', tags: [],
    has: [], lastTurn: 'don\'t. checkpoint to disk; cache hot pages instead', actions: [] },
  { id: 'f0a2bb73', provider: 'claude-code', repo: 'polylogue', cwd: '~/src/polylogue', title: 'fts trigger drift on UPDATE OF title',
    msgs: 53, tokens: 31_900, cost: 0.66, since: '2026-04-27T22:01Z', tags: ['fts','bug'],
    has: ['tool_use','thinking'], lastTurn: 'add UPDATE OF body trigger; backfill 1.2k rows', actions: ['file_edit','shell','search'] },
  { id: '6da13ee2', provider: 'claude-code', repo: 'sinnix',    cwd: '~/sys/sinnix',     title: 'hyprland scratchpad: capture-quick keybind',
    msgs: 12, tokens: 4_080, cost: 0.06, since: '2026-04-27T17:40Z', tags: ['hyprland'],
    has: [], lastTurn: 'super+space → polylogue capture; pipe to receiver', actions: ['file_edit'] },
  { id: '904ee18f', provider: 'claude-code', repo: 'polylogue', cwd: '~/src/polylogue', title: 'doctor --target browser-capture: receiver health',
    msgs: 24, tokens: 11_440, cost: 0.21, since: '2026-04-27T09:18Z', tags: ['doctor','capture'],
    has: ['tool_use'], lastTurn: 'absorb browser-capture status; extension popup mirrors', actions: ['file_read','shell'] },
];

// Daemon component status (cybernetic calm — desaturated greens/amber/coral)
const DAEMON = {
  status: 'running',
  pid: 184_220,
  uptime: '4h 12m',
  components: [
    { name: 'live:claude-code',     state: 'ok',     lag: '0.4s', detail: '~/.claude/projects · cursor up' },
    { name: 'live:codex',           state: 'ok',     lag: '1.1s', detail: '~/.codex/sessions · cursor up' },
    { name: 'browser-capture',      state: 'ok',     lag: null,   detail: '127.0.0.1:8765 · ext connected' },
    { name: 'local-api',            state: 'ok',     lag: null,   detail: '127.0.0.1:8765/api · 14 reqs/min' },
    { name: 'web-app',              state: 'ok',     lag: null,   detail: 'http://127.0.0.1:8765/' },
    { name: 'fts:index',            state: 'warn',   lag: null,   detail: '1,210 rows pending backfill' },
    { name: 'derived:sessions',     state: 'ok',     lag: '12s',  detail: 'incremental · 412 sessions' },
    { name: 'derived:costs',        state: 'stale',  lag: '14m',  detail: 'last roll-up 14m ago' },
  ],
};

const RECENT_CAPTURE = [
  { t: '12:04:11', kind: 'claude-code', label: 'message → 7b3c2e91 (assistant, 412 tok)' },
  { t: '12:04:08', kind: 'browser',     label: 'claude.ai/chat/abc… → 2c41ed09 (user, 84 tok)' },
  { t: '12:03:52', kind: 'codex',       label: 'message → a90bf311 (tool_use, exec)' },
  { t: '12:03:21', kind: 'claude-code', label: 'message → 8d821a4f (user, 38 tok)' },
  { t: '12:02:55', kind: 'claude-code', label: 'message → 7b3c2e91 (user, 290 tok)' },
];

// Synthetic conversation body — used by the reader artboard
const CONV_DETAIL = {
  '7b3c2e91': {
    header: { id: '7b3c2e91-2c4a-4f9d-9e10-a2b8c0d11e23', provider: 'claude-code', repo: 'polylogue',
              cwd: '~/src/polylogue/cli', model: 'claude-sonnet-4.5', started: '2026-04-30 14:02:11',
              ended: '2026-04-30 17:48:02', msgs: 71, tokens: 52_404, cost: '$0.91',
              tags: ['daemon','architecture','#635'], has: ['tool_use','thinking','paste'] },
    messages: [
      { i: 1,  role: 'user', t: '14:02:11', body: 'split browser-capture serve out of click_app — should it move to polylogued or stay reachable as `polylogue browser-capture serve`?', has: ['paste'] },
      { i: 2,  role: 'assistant', t: '14:02:18', body: 'Move it. Long-running serve is daemon work; the interactive CLI should not own a port. Concretely:\n  • polylogued run → starts live ingestion + receiver + local API + web app\n  • polylogue doctor --target browser-capture → reads daemon status\n  • extension popup mirrors the same status\nKeep no `polylogue watch` alias — clean break.', has: ['thinking'] },
      { i: 3,  role: 'user', t: '14:03:02', body: 'and `polylogue mcp`?' },
      { i: 4,  role: 'assistant', t: '14:03:09', body: 'Separate executable: `polylogue-mcp`. MCP clients launch it directly with --role read|write|admin. It can talk to polylogued when present, falls back to direct archive access otherwise. Don\'t require every MCP client to spin up live ingestion.' },
      { i: 5,  role: 'user', t: '14:05:44', body: 'show me the click_app diff for removing the serve command', has: [] },
      { i: 6,  role: 'tool', t: '14:05:46', body: '$ rg -n "browser-capture serve" polylogue/cli/\npolylogue/cli/click_command_registration.py:42:    register("browser-capture", browser_capture_group)', has: ['tool_use'] },
      { i: 7,  role: 'assistant', t: '14:05:51', body: 'Drop the `serve` subcommand from `browser_capture_group`; keep `status` as a thin reader of daemon status. Move the receiver app into `polylogue/daemon/receiver.py`.' },
    ],
    raw: [
      { kind: 'message',     t: '14:02:11', summary: 'user msg, 1 part, 84 tok' },
      { kind: 'message',     t: '14:02:18', summary: 'assistant msg, 1 part, 412 tok, thinking_block' },
      { kind: 'tool_call',   t: '14:05:46', summary: 'Bash(rg -n …) → 1 line stdout' },
      { kind: 'session_end', t: '17:48:02', summary: 'cursor closed; total 71 msgs' },
    ],
  },
};

const KERNEL = { PROVIDERS, CONVERSATIONS, DAEMON, RECENT_CAPTURE, CONV_DETAIL };
Object.assign(window, { KERNEL, PROVIDERS, CONVERSATIONS, DAEMON, RECENT_CAPTURE, CONV_DETAIL });
