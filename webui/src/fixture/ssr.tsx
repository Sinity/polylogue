import renderToString from 'preact-render-to-string';

import { App } from './app';
import { SESSIONS } from './data';
import { routeFromRequestTarget } from './routes';

export interface RenderedFixtureDocument {
  html: string;
  status: number;
}

export function renderDocument(requestTarget: string): RenderedFixtureDocument {
  const route = routeFromRequestTarget(requestTarget);
  const content = renderToString(<App route={route} />);
  const status = route.kind === 'not-found'
    || (route.kind === 'reader' && !SESSIONS.some((session) => session.id === route.sessionId))
    ? 404
    : 200;
  const html = `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light dark">
  <meta name="description" content="Sanitized deterministic Polylogue WebUI design-system fixture">
  <title>Polylogue WebUI design-system fixture</title>
  <link rel="stylesheet" href="/assets/webui.css">
  <script type="module" src="/assets/webui.js"></script>
</head>
<body>
  <div id="app">${content}</div>
</body>
</html>`;
  return { html, status };
}
