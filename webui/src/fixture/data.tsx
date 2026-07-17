import type { EvidenceState, OriginToken } from '../generated/contracts';
import type { TranscriptMessage } from '../design-system';

export interface FixtureSession {
  id: string;
  title: string;
  summary: string;
  origin: OriginToken | null;
  evidence: EvidenceState;
  updatedAt: string;
  messageCount: number;
  transcript: ReadonlyArray<TranscriptMessage>;
  code: string;
  diff: string;
}

export const SESSIONS: ReadonlyArray<FixtureSession> = [
  {
    id: 'sanitized-alpha',
    title: 'Deterministic release-note synthesis',
    summary: 'A sanitized conversation about turning verified changes into a concise release note.',
    origin: 'claude-code-session',
    evidence: 'exact',
    updatedAt: '2026-06-12T09:30:00Z',
    messageCount: 4,
    transcript: [
      {
        id: 'alpha-1',
        role: 'system',
        author: 'System',
        timestamp: '2026-06-12T09:24:00Z',
        body: <p>Use only the supplied fixture facts. Do not infer operator state.</p>,
        evidence: 'exact',
      },
      {
        id: 'alpha-2',
        role: 'user',
        author: 'Operator',
        timestamp: '2026-06-12T09:25:00Z',
        body: <p>Summarize the validated changes and keep unknown verification explicit.</p>,
        evidence: 'exact',
      },
      {
        id: 'alpha-3',
        role: 'assistant',
        author: 'Assistant',
        timestamp: '2026-06-12T09:27:00Z',
        body: <p>The release note separates executed checks from browser and deployment checks that remain unverified.</p>,
        evidence: 'qualified',
      },
      {
        id: 'alpha-4',
        role: 'tool',
        author: 'Fixture tool',
        timestamp: '2026-06-12T09:30:00Z',
        body: <p>Four sanitized records were read from the in-memory fixture.</p>,
        evidence: 'exact',
      },
    ],
    code: 'const verification = { unit: "passed", deployment: "unverified" };\n',
    diff: '- status: assumed\n+ status: unverified\n+ evidence: deterministic fixture',
  },
  {
    id: 'sanitized-beta',
    title: 'Search indexing regression',
    summary: 'A bounded debugging trace for a stale index cursor and its reset behavior.',
    origin: 'codex-session',
    evidence: 'qualified',
    updatedAt: '2026-06-10T14:15:00Z',
    messageCount: 3,
    transcript: [
      {
        id: 'beta-1',
        role: 'user',
        author: 'Operator',
        timestamp: '2026-06-10T14:10:00Z',
        body: <p>The second page repeats after the query changes.</p>,
        evidence: 'exact',
      },
      {
        id: 'beta-2',
        role: 'assistant',
        author: 'Assistant',
        timestamp: '2026-06-10T14:12:00Z',
        body: <p>Abort the active request and increment the paging generation when filters reset.</p>,
        evidence: 'qualified',
      },
      {
        id: 'beta-3',
        role: 'tool',
        author: 'Fixture tool',
        timestamp: '2026-06-10T14:15:00Z',
        body: <p>The stale response was ignored after reset.</p>,
        evidence: 'exact',
      },
    ],
    code: 'generation.current += 1;\ncontroller.current?.abort();\n',
    diff: '- append(stalePage.items)\n+ if (generation === current) append(page.items)',
  },
  {
    id: 'sanitized-gamma',
    title: 'Accessibility audit handoff',
    summary: 'Keyboard order, contrast evidence, semantic tables, and reduced-motion requirements.',
    origin: 'chatgpt-export',
    evidence: 'stale',
    updatedAt: '2026-05-29T18:00:00Z',
    messageCount: 2,
    transcript: [
      {
        id: 'gamma-1',
        role: 'user',
        author: 'Reviewer',
        timestamp: '2026-05-29T17:52:00Z',
        body: <p>Keep the server-rendered route usable when the enhancement bundle is unavailable.</p>,
        evidence: 'exact',
      },
      {
        id: 'gamma-2',
        role: 'assistant',
        author: 'Assistant',
        timestamp: '2026-05-29T18:00:00Z',
        body: <p>Links, forms, native disclosures, headings, and table captions remain functional without JavaScript.</p>,
        evidence: 'stale',
      },
    ],
    code: '<main id="main-content" tabindex="-1"><h1>Session</h1></main>\n',
    diff: '- <div onclick="openRow()">\n+ <a href="/sessions/sanitized-gamma">',
  },
  {
    id: 'sanitized-delta',
    title: 'Unknown source normalization',
    summary: 'A fixture demonstrating that unknown provenance is not silently mapped to a public provider badge.',
    origin: null,
    evidence: 'unknown',
    updatedAt: '2026-05-17T11:20:00Z',
    messageCount: 1,
    transcript: [
      {
        id: 'delta-1',
        role: 'unknown',
        author: 'Unattributed record',
        timestamp: '2026-05-17T11:20:00Z',
        body: <p>The content is readable, but its source token was not part of the public contract.</p>,
        evidence: 'unknown',
      },
    ],
    code: 'origin: null,\nevidence: "unknown"\n',
    diff: '- origin: "best-guess"\n+ origin: null',
  },
];

export function searchSessions(query: string): ReadonlyArray<FixtureSession> {
  const normalized = query.trim().toLocaleLowerCase();
  if (!normalized) return [];
  return SESSIONS.filter((session) =>
    `${session.title} ${session.summary}`.toLocaleLowerCase().includes(normalized),
  );
}
