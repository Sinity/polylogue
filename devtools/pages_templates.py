"""Jinja templates for the Polylogue documentation site."""

from __future__ import annotations

BASE_TEMPLATE = """<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="Polylogue documentation — local, evidence-addressable archives for AI work.">
    <title>{{ title }} — Polylogue</title>
    <style>
{{ style | safe }}
    </style>
    {% block head %}{% endblock %}
</head>
<body class="{{ body_class }}">
    <a class="skip-link" href="#main-content">Skip to content</a>
    <header class="site-header">
        <div class="header-inner">
            <button class="nav-toggle" type="button" aria-label="Open documentation navigation" aria-expanded="false" onclick="toggleNav(this)">☰</button>
            <a href="{{ site_root }}" class="logo">poly<span class="logo-mark">logue</span></a>
            <nav class="primary-nav" aria-label="Primary navigation">
                <a class="nav-essential" href="{{ get_started_href }}">Get started</a>
                <a href="{{ docs_href }}">Documentation</a>
                <a class="nav-essential" href="{{ board_href }}">Roadmap</a>
                <a href="https://github.com/Sinity/polylogue">GitHub</a>
            </nav>
            <button class="theme-toggle" type="button" onclick="toggleTheme(this)" title="Toggle color theme" aria-label="Toggle color theme">◐</button>
        </div>
    </header>
    <div class="page-layout">
        <nav class="site-nav" id="site-nav" aria-label="Documentation navigation">
            {% for section in nav %}
            <section class="nav-section">
                <div class="nav-section-title">{{ section.title }}</div>
                {% for item in section.entries %}
                <a href="{{ item.href }}" {% if item.path == current_path %}class="active" aria-current="page"{% endif %}>{{ item.label }}</a>
                {% endfor %}
            </section>
            {% endfor %}
        </nav>
        <main class="content" id="main-content">
            {% block content %}{% endblock %}
        </main>
    </div>
    <footer class="site-footer">
        <div class="footer-inner">
            <span>Polylogue · local evidence for AI work</span>
            <span><a href="{{ docs_href }}">Documentation</a> · <a href="{{ board_href }}">Roadmap</a> · <a href="https://github.com/Sinity/polylogue">Source</a></span>
        </div>
    </footer>
    <script>
        (function () {
            var saved = localStorage.getItem('polylogue-theme');
            var preferred = window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', saved || preferred);
        })();
        function toggleTheme() {
            var html = document.documentElement;
            var next = html.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
            html.setAttribute('data-theme', next);
            localStorage.setItem('polylogue-theme', next);
        }
        function toggleNav(button) {
            var open = document.body.classList.toggle('nav-open');
            button.setAttribute('aria-expanded', String(open));
            button.setAttribute('aria-label', open ? 'Close documentation navigation' : 'Open documentation navigation');
        }
        document.addEventListener('keydown', function (event) {
            if (event.key === 'Escape') {
                document.body.classList.remove('nav-open');
                var button = document.querySelector('.nav-toggle');
                if (button) button.setAttribute('aria-expanded', 'false');
            }
        });
    </script>
    {% block scripts %}{% endblock %}
</body>
</html>
"""

HOME_TEMPLATE = """{% extends "base.html" %}
{% block content %}
<section class="home-hero">
    <p class="eyebrow">Local-first · cross-provider · evidence-addressable</p>
    <h1>Know what the agents actually did.</h1>
    <p class="tagline">Polylogue turns AI chats, coding-agent sessions, tool calls, results, forks, costs, and reviewed notes into one local evidence system—so work can be searched, audited, and resumed without trusting a transcript summary.</p>
    <div class="hero-actions">
        <a class="button button-primary" href="{{ get_started_href }}">Get started <span aria-hidden="true">→</span></a>
        <a class="button" href="{{ demos_href }}">Run a private-data-free proof</a>
        <a class="button" href="{{ board_href }}">Browse the roadmap</a>
    </div>
    <div class="hero-command"><pre><code>nix run github:Sinity/polylogue -- demo receipts --compact</code></pre></div>
</section>

<section class="home-section">
    <p class="section-kicker">Evidence pipeline</p>
    <h2>From provider-shaped artifacts to defensible answers</h2>
    <p class="section-intro">Raw evidence remains durable. Normalized views and analytics are rebuildable. Human judgment is explicit rather than blended into machine inference.</p>
    <div class="pipeline" aria-label="Polylogue evidence pipeline">
        <div class="pipeline-step"><span>01 · Acquire</span><strong>Preserve source evidence</strong><p>Exports, agent files, hooks, browser capture, and telemetry.</p></div>
        <div class="pipeline-step"><span>02 · Normalize</span><strong>Recover work structure</strong><p>Sessions, messages, actions, results, lineage, and authoredness.</p></div>
        <div class="pipeline-step"><span>03 · Derive</span><strong>Build replaceable views</strong><p>Search, costs, phases, profiles, claims, and optional vectors.</p></div>
        <div class="pipeline-step"><span>04 · Use</span><strong>Search, audit, resume</strong><p>CLI, Python, MCP, daemon API, and local web reader.</p></div>
    </div>
</section>

<section class="home-section">
    <p class="section-kicker">What changes</p>
    <h2>A work archive, not another transcript folder</h2>
    <div class="capability-grid">
        <article class="capability-card"><span class="cap-index">CLAIMS → OUTCOMES</span><h3>Audit the work behind the prose</h3><p>Resolve “tests pass” to the tool call, exit status, duration, and raw result instead of treating an assistant sentence as evidence.</p></article>
        <article class="capability-card"><span class="cap-index">PHYSICAL → LOGICAL</span><h3>Keep lineage without double-counting</h3><p>Preserve forks, resumptions, copied prefixes, and subagents while composing the logical work they represent.</p></article>
        <article class="capability-card"><span class="cap-index">HISTORY → HANDOFF</span><h3>Resume from bounded context</h3><p>Compile reviewed evidence, omissions, caveats, and user judgments into a reproducible context package for the next agent.</p></article>
        <article class="capability-card"><span class="cap-index">USAGE → ACCOUNTING</span><h3>Keep cost semantics honest</h3><p>Separate provider-reported usage, cache lanes, reasoning tokens, catalog estimates, and subscription-credit views.</p></article>
    </div>
</section>

<section class="home-section">
    <p class="section-kicker">One archive, several interfaces</p>
    <h2>Meet the workflow where it already lives</h2>
    <p class="section-intro">Polylogue ingests provider-specific evidence, but its query and reading model is provider-independent.</p>
    <div class="surface-row" aria-label="Supported evidence origins">
        <span class="surface-pill">Claude Code</span><span class="surface-pill">Codex</span><span class="surface-pill">ChatGPT</span><span class="surface-pill">Claude.ai</span><span class="surface-pill">Gemini</span><span class="surface-pill">Hermes</span><span class="surface-pill">Antigravity</span><span class="surface-pill">Hooks</span><span class="surface-pill">Browser capture</span><span class="surface-pill">OTLP-shaped events</span>
    </div>
    <div class="surface-row" aria-label="Interfaces">
        <span class="surface-pill">CLI</span><span class="surface-pill">Python API</span><span class="surface-pill">MCP</span><span class="surface-pill">Local daemon</span><span class="surface-pill">Web reader</span><span class="surface-pill">Markdown and JSON exports</span>
    </div>
    <div class="home-callout">
        <div><h2>See the project as it evolves</h2><p>The committed Beads graph is rendered as a searchable roadmap with designs, acceptance criteria, dependencies, and closure records.</p></div>
        <a class="button" href="{{ board_href }}">Open roadmap <span aria-hidden="true">→</span></a>
    </div>
</section>
{% endblock %}
"""

DOC_TEMPLATE = """{% extends "base.html" %}
{% block content %}
<article class="doc">{{ content | safe }}</article>
<nav class="doc-pager" aria-label="Adjacent documentation">
    {% if prev %}<a href="{{ prev.href }}">← {{ prev.label }}</a>{% else %}<span></span>{% endif %}
    {% if next %}<a href="{{ next.href }}">{{ next.label }} →</a>{% else %}<span></span>{% endif %}
</nav>
{% endblock %}
"""

BEADS_TEMPLATE = """{% extends "base.html" %}
{% block content %}
<header class="board-hero">
    <div><p class="eyebrow">Committed project record</p><h1>Roadmap and work graph</h1><p>Search the project’s Beads issues, inspect designs and acceptance criteria, and follow dependencies through delivered and active work. This view is rebuilt from the repository on every documentation deployment.</p></div>
    <div class="board-meta"><a href="https://github.com/Sinity/polylogue/blob/master/.beads/issues.jsonl">View source JSONL</a></div>
</header>
<div class="board-stats" id="board-stats" aria-label="Issue summary"></div>
<form class="board-toolbar" id="board-toolbar">
    <input type="search" id="board-search" placeholder="Search titles, descriptions, designs, criteria…" aria-label="Search issues">
    <select id="board-status" aria-label="Filter by status"><option value="active">Active work</option><option value="ready">Ready</option><option value="blocked">Blocked</option><option value="all">All statuses</option><option value="closed">Closed</option></select>
    <select id="board-priority" aria-label="Filter by priority"><option value="all">All priorities</option><option value="0">P0</option><option value="1">P1</option><option value="2">P2</option><option value="3">P3</option><option value="4">P4</option></select>
    <select id="board-type" aria-label="Filter by type"><option value="all">All types</option></select>
</form>
<div class="board-results-meta"><span id="board-result-count">Loading issues…</span><span>Open an item for its full record</span></div>
<div class="issue-list" id="issue-list" aria-live="polite"></div>
<button class="button load-more" id="load-more" type="button" hidden>Show more</button>
{% endblock %}
{% block scripts %}
<script>
(() => {
    const pageSize = 100;
    const state = { issues: [], visible: pageSize };
    const els = Object.fromEntries(['board-stats','board-search','board-status','board-priority','board-type','board-result-count','issue-list','load-more'].map(id => [id, document.getElementById(id)]));
    const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    const text = value => esc(value).replace(/(https?:[/][/][^ <]+)/g, '<a href="$1">$1</a>');
    const isClosed = issue => issue.status === 'closed';
    const blockingDeps = issue => (issue.dependencies || []).filter(dep => dep.type === 'blocks' && !isClosed(state.byId.get(dep.depends_on_id) || {}));
    const bucket = issue => isClosed(issue) ? 'closed' : issue.status === 'in_progress' ? 'in_progress' : blockingDeps(issue).length ? 'blocked' : 'ready';
    const statusBadge = issue => {
        const value = bucket(issue);
        const cls = value === 'closed' ? 'badge-green' : value === 'blocked' ? 'badge-red' : value === 'in_progress' ? 'badge-yellow' : '';
        return `<span class="badge ${cls}">${esc(value.replace('_', ' '))}</span>`;
    };
    const field = (title, value) => value ? `<section class="issue-section"><h3>${title}</h3><div class="issue-prose">${text(value)}</div></section>` : '';
    const dependencyLinks = issue => {
        const deps = issue.dependencies || [];
        if (!deps.length) return '';
        return `<section class="issue-section"><h3>Relationships</h3><div class="issue-links">${deps.map(dep => `<button class="issue-link" type="button" data-issue="${esc(dep.depends_on_id)}">${esc(dep.type)} · ${esc(dep.depends_on_id)}</button>`).join('')}</div></section>`;
    };
    const card = issue => {
        const labels = (issue.labels || []).map(label => `<span class="badge">${esc(label)}</span>`).join('');
        const date = issue.updated_at ? new Date(issue.updated_at).toLocaleDateString(undefined, {year:'numeric',month:'short',day:'numeric'}) : '';
        return `<details class="issue-card" id="${esc(issue.id)}"><summary class="issue-summary"><span class="priority priority-${esc(issue.priority)}">P${esc(issue.priority)}</span><span><span class="issue-title"><span class="issue-id">${esc(issue.id)}</span>${esc(issue.title)}</span><span class="issue-subline">${statusBadge(issue)}<span class="badge">${esc(issue.issue_type || 'task')}</span>${labels}</span></span><span class="issue-updated">${esc(date)}</span></summary><div class="issue-body">${field('Description', issue.description)}${field('Design', issue.design)}${field('Acceptance criteria', issue.acceptance_criteria)}${field('Notes', issue.notes)}${dependencyLinks(issue)}${field('Closure', issue.close_reason)}</div></details>`;
    };
    function matches(issue) {
        const query = els['board-search'].value.trim().toLowerCase();
        const haystack = [issue.id, issue.title, issue.description, issue.design, issue.acceptance_criteria, issue.notes, issue.close_reason, ...(issue.labels || [])].join(String.fromCharCode(10)).toLowerCase();
        const status = els['board-status'].value;
        const issueBucket = bucket(issue);
        const statusOk = status === 'all' || (status === 'active' && issueBucket !== 'closed') || status === issueBucket;
        return statusOk && (els['board-priority'].value === 'all' || String(issue.priority) === els['board-priority'].value) && (els['board-type'].value === 'all' || issue.issue_type === els['board-type'].value) && (!query || query.split(' ').filter(Boolean).every(term => haystack.includes(term)));
    }
    function updateUrl() {
        const url = new URL(location.href);
        [['q','board-search'],['status','board-status'],['priority','board-priority'],['type','board-type']].forEach(([key,id]) => {
            const value = els[id].value;
            if (value && !(['status','priority','type'].includes(key) && value === (key === 'status' ? 'active' : 'all'))) url.searchParams.set(key, value); else url.searchParams.delete(key);
        });
        history.replaceState(null, '', url);
    }
    function render() {
        const filtered = state.issues.filter(matches).sort((a,b) => (a.priority - b.priority) || String(b.updated_at).localeCompare(String(a.updated_at)));
        els['issue-list'].innerHTML = filtered.slice(0, state.visible).map(card).join('') || '<div class="empty-state">No issues match these filters.</div>';
        els['board-result-count'].textContent = `${filtered.length.toLocaleString()} matching issue${filtered.length === 1 ? '' : 's'}`;
        els['load-more'].hidden = filtered.length <= state.visible;
        updateUrl();
        if (location.hash) {
            const target = document.getElementById(decodeURIComponent(location.hash.slice(1)));
            if (target) { target.open = true; requestAnimationFrame(() => target.scrollIntoView({block:'start'})); }
        }
    }
    function openIssue(id) {
        const issue = state.byId.get(id);
        if (!issue) return;
        els['board-search'].value = id;
        els['board-status'].value = 'all';
        state.visible = pageSize;
        location.hash = encodeURIComponent(id);
        render();
    }
    fetch('issues.jsonl').then(response => { if (!response.ok) throw new Error(`HTTP ${response.status}`); return response.text(); }).then(raw => {
        state.issues = raw.split(String.fromCharCode(10)).map(line => line.endsWith(String.fromCharCode(13)) ? line.slice(0, -1) : line).filter(Boolean).map(line => JSON.parse(line)).filter(row => row._type === 'issue');
        state.byId = new Map(state.issues.map(issue => [issue.id, issue]));
        const params = new URLSearchParams(location.search);
        els['board-search'].value = params.get('q') || '';
        els['board-status'].value = params.get('status') || 'active';
        els['board-priority'].value = params.get('priority') || 'all';
        const types = [...new Set(state.issues.map(issue => issue.issue_type).filter(Boolean))].sort();
        els['board-type'].insertAdjacentHTML('beforeend', types.map(type => `<option value="${esc(type)}">${esc(type)}</option>`).join(''));
        els['board-type'].value = params.get('type') || 'all';
        const counts = { total: state.issues.length, active: 0, ready: 0, blocked: 0, closed: 0 };
        state.issues.forEach(issue => { const b = bucket(issue); counts[b] = (counts[b] || 0) + 1; if (b !== 'closed') counts.active++; });
        els['board-stats'].innerHTML = [['Active',counts.active],['Ready',counts.ready],['Blocked',counts.blocked],['In progress',counts.in_progress || 0],['Delivered',counts.closed]].map(([label,value]) => `<div class="board-stat"><strong>${value.toLocaleString()}</strong><span>${label}</span></div>`).join('');
        render();
    }).catch(error => { els['issue-list'].innerHTML = `<div class="empty-state">Could not load the committed Beads records: ${esc(error.message)}</div>`; els['board-result-count'].textContent = 'Board unavailable'; });
    ['board-search','board-status','board-priority','board-type'].forEach(id => els[id].addEventListener(id === 'board-search' ? 'input' : 'change', () => { state.visible = pageSize; render(); }));
    els['load-more'].addEventListener('click', () => { state.visible += pageSize; render(); });
    els['issue-list'].addEventListener('click', event => { const button = event.target.closest('[data-issue]'); if (button) openIssue(button.dataset.issue); });
})();
</script>
{% endblock %}
"""

VERIFIABILITY_CATALOG_TEMPLATE = """{% extends "base.html" %}{% block content %}<h1>Evidence Catalog</h1><p>Last updated: {{ updated_at }}</p><div class="card"><strong>{{ claims|length }} claims</strong> · <span class="badge badge-green">{{ fresh_count }} fresh</span> <span class="badge badge-yellow">{{ stale_count }} stale</span> <span class="badge badge-red">{{ overridden_count }} overridden</span></div><table><thead><tr><th>Claim</th><th>Oracle</th><th>Domain</th><th>Last evidence</th><th>Status</th></tr></thead><tbody>{% for claim in claims %}<tr><td>{{ claim.description }}</td><td><code>{{ claim.oracle }}</code></td><td>{{ claim.assurance_domain }}</td><td>{{ claim.last_evidence_at or "—" }}</td><td><span class="badge badge-{{ 'green' if claim.status == 'fresh' else 'yellow' if claim.status == 'stale' else 'red' if claim.status == 'overridden' else '' }}">{{ claim.status }}</span></td></tr>{% endfor %}</tbody></table>{% endblock %}"""

COVERAGE_TEMPLATE = """{% extends "base.html" %}{% block content %}<h1>Coverage Map</h1><p>Line coverage: {{ coverage_pct }}% · {{ modules_tested }}/{{ modules_total }} modules tested</p><table><thead><tr><th>Module</th><th>Line %</th><th>Branch %</th><th>Status</th></tr></thead><tbody>{% for mod in modules %}<tr><td><code>{{ mod.name }}</code></td><td>{{ mod.line_pct }}%</td><td>{{ mod.branch_pct }}%</td><td><span class="badge badge-{{ 'green' if mod.line_pct >= 85 else 'yellow' if mod.line_pct >= 70 else 'red' }}">{{ 'good' if mod.line_pct >= 85 else 'ok' if mod.line_pct >= 70 else 'low' }}</span></td></tr>{% endfor %}</tbody></table>{% endblock %}"""

MUTATION_TEMPLATE = """{% extends "base.html" %}{% block content %}<h1>Mutation Scores</h1><p>Committed index from the most recent campaign run.</p><table><thead><tr><th>Campaign</th><th>Score</th><th>Trend</th><th>Survivors</th></tr></thead><tbody>{% for camp in campaigns %}<tr><td>{{ camp.name }}</td><td>{{ camp.score_pct }}%</td><td>{{ '↑' if camp.trend > 0 else '↓' if camp.trend < 0 else '→' }} {{ camp.trend|abs if camp.trend else '' }}</td><td>{{ camp.survivor_count }}</td></tr>{% endfor %}</tbody></table>{% endblock %}"""

DRIFT_TEMPLATE = """{% extends "base.html" %}{% block content %}<h1>Schema Drift</h1><p>Per-provider schema freshness. A provider is stale after 30 days without regeneration.</p><div class="card-grid">{% for prov in providers %}<div class="card"><h3>{{ prov.name }}</h3><p>Version: <code>{{ prov.version }}</code><br>Last regenerated: {{ prov.last_regenerated }}<br>Age: {{ prov.age_days }} days</p>{% if prov.drift_detected %}<span class="badge badge-red">drift detected</span>{% endif %}</div>{% endfor %}</div>{% endblock %}"""

FRESHNESS_TEMPLATE = """{% extends "base.html" %}{% block content %}<h1>Evidence Freshness</h1><p>Per-claim staleness status.</p><div class="card"><span class="badge badge-green">{{ fresh_count }} fresh</span> <span class="badge badge-yellow">{{ stale_count }} stale</span> <span class="badge badge-red">{{ overridden_count }} overridden</span> <span class="badge">{{ uncollected_count }} uncollected</span></div><table><thead><tr><th>Claim</th><th>Policy</th><th>Last collected</th><th>Status</th></tr></thead><tbody>{% for claim in claims %}<tr><td>{{ claim.description }}</td><td>{{ claim.staleness_policy }}</td><td>{{ claim.last_collected_at or "—" }}</td><td><span class="badge badge-{{ 'green' if claim.staleness == 'fresh' else 'yellow' if claim.staleness == 'approaching' else 'red' if claim.staleness in ('stale','overridden') else '' }}">{{ claim.staleness }}</span></td></tr>{% endfor %}</tbody></table>{% endblock %}"""

PAGES_TEMPLATES = {
    "base.html": BASE_TEMPLATE,
    "home.html": HOME_TEMPLATE,
    "doc.html": DOC_TEMPLATE,
    "beads.html": BEADS_TEMPLATE,
    "verifiability_catalog.html": VERIFIABILITY_CATALOG_TEMPLATE,
    "verifiability_coverage.html": COVERAGE_TEMPLATE,
    "verifiability_mutation.html": MUTATION_TEMPLATE,
    "verifiability_drift.html": DRIFT_TEMPLATE,
    "verifiability_freshness.html": FRESHNESS_TEMPLATE,
}

__all__ = ["PAGES_TEMPLATES"]
