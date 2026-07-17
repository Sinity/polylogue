export type FixtureRoute =
  | { kind: 'list' }
  | { kind: 'reader'; sessionId: string }
  | { kind: 'search'; query: string }
  | { kind: 'evidence' }
  | { kind: 'timeline' }
  | { kind: 'not-found'; path: string };

export function routeFromUrl(url: URL): FixtureRoute {
  const path = url.pathname.replace(/\/+$/, '') || '/';
  if (path === '/') return { kind: 'list' };
  if (path === '/search') return { kind: 'search', query: url.searchParams.get('q') ?? '' };
  if (path === '/evidence') return { kind: 'evidence' };
  if (path === '/timeline') return { kind: 'timeline' };
  const readerMatch = /^\/sessions\/([^/]+)$/.exec(path);
  if (readerMatch?.[1]) return { kind: 'reader', sessionId: decodeURIComponent(readerMatch[1]) };
  return { kind: 'not-found', path };
}

export function routeFromRequestTarget(target: string): FixtureRoute {
  return routeFromUrl(new URL(target, 'http://fixture.invalid'));
}
