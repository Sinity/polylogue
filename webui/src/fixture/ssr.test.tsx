import { renderDocument } from './ssr';

describe('deterministic SSR fixture', () => {
  it('publishes readable list semantics before client enhancement', () => {
    const rendered = renderDocument('/');

    expect(rendered.status).toBe(200);
    expect(rendered.html).toContain('data-webui-vertical="webui-02"');
    expect(rendered.html).toContain('<table');
    expect(rendered.html).toContain('<caption>');
    expect(rendered.html).toContain('href="/sessions/sanitized-alpha"');
    expect(rendered.html).toContain('method="get"');
    expect(rendered.html).toContain('src="/assets/webui.js"');
    expect(rendered.html).not.toContain('onclick=');
  });

  it('renders reader and search routes from the URL alone', () => {
    const reader = renderDocument('/sessions/sanitized-alpha');
    const search = renderDocument('/search?q=accessibility');

    expect(reader.status).toBe(200);
    expect(reader.html).toContain('data-webui-vertical="webui-03"');
    expect(reader.html).toContain('Deterministic release-note synthesis');
    expect(search.status).toBe(200);
    expect(search.html).toContain('data-webui-vertical="webui-04"');
    expect(search.html).toContain('Accessibility audit handoff');
  });

  it('distinguishes an unknown session from a known empty search', () => {
    const unknown = renderDocument('/sessions/not-in-fixture');
    const empty = renderDocument('/search?q=definitely-absent');

    expect(unknown.status).toBe(404);
    expect(unknown.html).toContain('data-honest-state="unknown"');
    expect(empty.status).toBe(200);
    expect(empty.html).toContain('data-honest-state="empty"');
  });
});
