import { useMemo, useState } from 'preact/hooks';

import {
  Cluster,
  CodeBlock,
  DataTable,
  DegradedState,
  DiffBlock,
  Disclosure,
  EmptyState,
  EvidenceStateBadge,
  FacetChipGroup,
  Grid,
  HonestState,
  OriginBadge,
  PUBLIC_ORIGINS,
  SearchField,
  Skeleton,
  SkipLink,
  Sparkline,
  Stack,
  Surface,
  ThemeToggle,
  Timeline,
  TranscriptBlock,
  UnknownOriginBadge,
  UnknownState,
  VerticalFrame,
  type DataColumn,
  type EvidenceState,
  type OriginToken,
  useContinuationPaging,
} from '../design-system';
import { SESSIONS, searchSessions, type FixtureSession } from './data';
import type { FixtureRoute } from './routes';

const EVIDENCE_OPTIONS: ReadonlyArray<{ value: EvidenceState; label: string; count: number }> = [
  { value: 'exact', label: 'Exact', count: 1 },
  { value: 'qualified', label: 'Qualified', count: 1 },
  { value: 'stale', label: 'Stale', count: 1 },
  { value: 'unknown', label: 'Unknown', count: 1 },
  { value: 'degraded', label: 'Degraded', count: 0 },
];

function currentPath(route: FixtureRoute): string {
  switch (route.kind) {
    case 'list': return '/';
    case 'reader': return `/sessions/${encodeURIComponent(route.sessionId)}`;
    case 'search': return '/search';
    case 'evidence': return '/evidence';
    case 'timeline': return '/timeline';
    case 'not-found': return route.path;
  }
}

function Shell({ route, children }: { route: FixtureRoute; children: preact.ComponentChildren }) {
  const path = currentPath(route);
  const nav = [
    { href: '/', label: 'Sessions' },
    { href: '/search?q=evidence', label: 'Search' },
    { href: '/evidence', label: 'Evidence states' },
    { href: '/timeline', label: 'Timeline' },
  ];
  return (
    <div class="pl-shell">
      <SkipLink />
      <header class="pl-site-header">
        <a class="pl-brand" href="/">Polylogue fixture</a>
        <nav class="pl-site-nav" aria-label="Fixture routes">
          {nav.map((item) => (
            <a key={item.href} href={item.href} aria-current={path === item.href.split('?')[0] ? 'page' : undefined}>
              {item.label}
            </a>
          ))}
        </nav>
        <ThemeToggle />
      </header>
      {children}
      <footer class="pl-site-footer">
        Sanitized deterministic fixture · no archive, daemon credential, or external network required.
      </footer>
    </div>
  );
}

async function loadFixtureSessions(cursor: string, signal: AbortSignal) {
  await Promise.resolve();
  if (signal.aborted) throw new DOMException('Fixture continuation aborted', 'AbortError');
  const offset = Number.parseInt(cursor, 10);
  const items = SESSIONS.slice(offset);
  return { items, nextCursor: null };
}

function ListView() {
  const [selectedEvidence, setSelectedEvidence] = useState<ReadonlySet<EvidenceState>>(new Set());
  const paging = useContinuationPaging({
    initialItems: SESSIONS.slice(0, 3),
    initialCursor: '3',
    loadPage: loadFixtureSessions,
  });
  const rows = useMemo(() => selectedEvidence.size === 0
    ? paging.items
    : paging.items.filter((session) => selectedEvidence.has(session.evidence)),
  [paging.items, selectedEvidence]);

  const columns: ReadonlyArray<DataColumn<FixtureSession>> = [
    {
      id: 'session',
      header: 'Session',
      cell: (session) => <a href={`/sessions/${encodeURIComponent(session.id)}`}>{session.title}</a>,
    },
    {
      id: 'origin',
      header: 'Origin',
      cell: (session) => session.origin ? <OriginBadge origin={session.origin} /> : <UnknownOriginBadge />,
    },
    {
      id: 'evidence',
      header: 'Evidence',
      cell: (session) => <EvidenceStateBadge state={session.evidence} />,
    },
    {
      id: 'updated',
      header: 'Updated',
      priority: 'secondary',
      cell: (session) => <time dateTime={session.updatedAt}>{session.updatedAt.slice(0, 10)}</time>,
    },
    {
      id: 'messages',
      header: 'Messages',
      align: 'end',
      cell: (session) => session.messageCount,
    },
  ];

  return (
    <VerticalFrame
      id="webui-02"
      state={rows.length === 0 ? 'empty' : 'ready'}
      title="Session inventory"
      description="A dense, semantic table with generated origin badges, honest evidence states, facets, and continuation controls."
      actions={<EvidenceStateBadge state="exact" qualifiedBy="sanitized fixture" />}
    >
      <Stack space={5}>
        <SearchField label="Search sessions" placeholder="Try evidence or indexing" />
        <FacetChipGroup
          label="Evidence state"
          options={EVIDENCE_OPTIONS}
          selected={selectedEvidence}
          onChange={setSelectedEvidence}
        />
        <DataTable
          caption={`${rows.length} visible sanitized sessions`}
          rows={rows}
          columns={columns}
          rowKey={(session) => session.id}
          density="compact"
          onRowActivate={(session) => {
            if (typeof window !== 'undefined') window.location.assign(`/sessions/${encodeURIComponent(session.id)}`);
          }}
          absence="empty"
          absenceDescription="No loaded sessions match the selected evidence facets. This is a known empty result, not an unknown archive state."
          continuation={{
            hasMore: paging.hasMore,
            loading: paging.loading,
            error: paging.error,
            label: 'Load final fixture row',
            onLoadMore: paging.loadMore,
          }}
        />
      </Stack>
    </VerticalFrame>
  );
}

function ReaderView({ sessionId }: { sessionId: string }) {
  const session = SESSIONS.find((candidate) => candidate.id === sessionId);
  if (!session) {
    return (
      <VerticalFrame
        id="webui-03"
        state="unknown"
        title="Session unavailable"
        description="The fixture route is valid, but this identifier is not present in the sanitized data set."
      >
        <UnknownState description="No source record was consulted, so absence cannot be claimed beyond this fixture." />
      </VerticalFrame>
    );
  }

  return (
    <VerticalFrame
      id="webui-03"
      state={session.evidence === 'unknown' ? 'unknown' : 'ready'}
      title={session.title}
      description={session.summary}
      actions={
        <Cluster space={2}>
          {session.origin ? <OriginBadge origin={session.origin} /> : <UnknownOriginBadge />}
          <EvidenceStateBadge state={session.evidence} />
        </Cluster>
      }
    >
      <Stack space={5}>
        <SearchField label="Search from reader" placeholder="Search the fixture" />
        <Surface>
          <dl class="pl-reader-meta">
            <div><dt>Fixture ID</dt><dd>{session.id}</dd></div>
            <div><dt>Updated</dt><dd><time dateTime={session.updatedAt}>{session.updatedAt}</time></dd></div>
            <div><dt>Messages</dt><dd>{session.messageCount}</dd></div>
            <div><dt>Source status</dt><dd>{session.origin ? 'Public origin token' : 'Unknown origin'}</dd></div>
          </dl>
        </Surface>
        <TranscriptBlock messages={session.transcript} />
        <Disclosure summary="Implementation evidence" open>
          <Stack space={4}>
            <CodeBlock code={session.code} language="typescript" caption="Sanitized code excerpt" />
            <DiffBlock diff={session.diff} caption="Behavioral correction" />
          </Stack>
        </Disclosure>
        <a href="/">Back to session inventory</a>
      </Stack>
    </VerticalFrame>
  );
}

function SearchView({ query }: { query: string }) {
  const results = searchSessions(query);
  const columns: ReadonlyArray<DataColumn<FixtureSession>> = [
    {
      id: 'session',
      header: 'Match',
      cell: (session) => (
        <Stack space={1}>
          <a href={`/sessions/${encodeURIComponent(session.id)}`}>{session.title}</a>
          <span>{session.summary}</span>
        </Stack>
      ),
    },
    {
      id: 'origin',
      header: 'Origin',
      cell: (session) => session.origin ? <OriginBadge origin={session.origin} /> : <UnknownOriginBadge />,
    },
    {
      id: 'evidence',
      header: 'Evidence',
      cell: (session) => <EvidenceStateBadge state={session.evidence} />,
    },
  ];
  const state = query.trim() && results.length === 0 ? 'empty' : 'ready';

  return (
    <VerticalFrame
      id="webui-04"
      state={state}
      title="Search sanitized sessions"
      description="The GET form and result links are fully server-rendered and remain usable without JavaScript."
    >
      <Stack space={5}>
        <SearchField label="Search query" defaultValue={query} placeholder="Try accessibility" />
        {!query.trim() ? (
          <EmptyState description="No query was submitted. This is an empty input state, not an unknown result set." />
        ) : (
          <DataTable
            caption={`${results.length} results for “${query}”`}
            rows={results}
            columns={columns}
            rowKey={(session) => session.id}
            absence="empty"
            absenceDescription={`The deterministic fixture contains no matches for “${query}”.`}
          />
        )}
      </Stack>
    </VerticalFrame>
  );
}

function EvidenceView() {
  return (
    <VerticalFrame
      id="webui-05"
      state="degraded"
      title="Evidence and origin vocabulary"
      description="Public tokens are generated from Python authority. Unknown provenance remains visibly separate."
    >
      <Stack space={6}>
        <Surface>
          <Stack space={4}>
            <h2>Ten public origin tokens</h2>
            <Cluster>
              {PUBLIC_ORIGINS.map((origin: OriginToken) => <OriginBadge key={origin} origin={origin} />)}
              <UnknownOriginBadge />
            </Cluster>
          </Stack>
        </Surface>
        <Surface>
          <Stack space={4}>
            <h2>Evidence-state badges</h2>
            <Cluster>
              {(['exact', 'qualified', 'stale', 'unknown', 'degraded'] as const).map((state) => (
                <EvidenceStateBadge key={state} state={state} />
              ))}
            </Cluster>
          </Stack>
        </Surface>
        <Grid className="pl-state-grid" min="16rem">
          <HonestState kind="empty" title="Known empty" description="The query completed and returned zero matching rows." />
          <HonestState kind="unknown" title="Unknown" description="The source was not consulted or could not establish absence." />
          <DegradedState description="Some evidence is readable, but one or more dependencies are incomplete." />
          <Stack space={3}>
            <HonestState kind="loading" description="The current request has not completed." />
            <Skeleton lines={2} />
          </Stack>
        </Grid>
      </Stack>
    </VerticalFrame>
  );
}

function TimelineView() {
  return (
    <VerticalFrame
      id="webui-06"
      state="ready"
      title="Timeline and trend primitives"
      description="Semantic event order and a labelled SVG sparkline, with no chart runtime or motion dependency."
    >
      <Grid min="20rem">
        <Surface>
          <Timeline
            items={[
              { id: 'capture', at: '2026-06-12T09:24:00Z', label: 'Fixture captured', description: 'Sanitized source rows were fixed.', state: 'exact' },
              { id: 'render', at: '2026-06-12T09:27:00Z', label: 'SSR rendered', description: 'Semantic markup was produced before enhancement.', state: 'qualified' },
              { id: 'verify', at: '2026-06-12T09:30:00Z', label: 'Harness verified', description: 'Unit, keyboard, a11y, and visual paths share the same fixture.', state: 'exact' },
            ]}
          />
        </Surface>
        <Surface>
          <Stack space={4}>
            <div class="pl-metric">
              <span>Evidence rows over fixed intervals</span>
              <strong>24</strong>
            </div>
            <Sparkline values={[4, 7, 6, 11, 13, 18, 17, 24]} label="Evidence rows rise from 4 to 24" />
            <p>The accessible name and description expose the same values without relying on shape or color.</p>
          </Stack>
        </Surface>
      </Grid>
    </VerticalFrame>
  );
}

function NotFoundView({ path }: { path: string }) {
  return (
    <VerticalFrame
      id="webui-02"
      state="empty"
      title="Route not found"
      description={`The deterministic fixture has no route for ${path}.`}
    >
      <EmptyState description="Use the session inventory, search, evidence, or timeline route." />
    </VerticalFrame>
  );
}

export function App({ route }: { route: FixtureRoute }) {
  let view;
  switch (route.kind) {
    case 'list': view = <ListView />; break;
    case 'reader': view = <ReaderView sessionId={route.sessionId} />; break;
    case 'search': view = <SearchView query={route.query} />; break;
    case 'evidence': view = <EvidenceView />; break;
    case 'timeline': view = <TimelineView />; break;
    case 'not-found': view = <NotFoundView path={route.path} />; break;
  }
  return <Shell route={route}>{view}</Shell>;
}
