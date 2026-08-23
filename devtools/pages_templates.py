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
            <span><a href="{{ docs_href }}">Documentation</a> · <a href="https://github.com/Sinity/polylogue">Source</a></span>
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

VERIFIABILITY_CATALOG_TEMPLATE = """{% extends "base.html" %}{% block content %}<h1>Evidence Catalog</h1><p>Last updated: {{ updated_at }}</p><div class="card"><strong>{{ claims|length }} claims</strong> · <span class="badge badge-green">{{ fresh_count }} fresh</span> <span class="badge badge-yellow">{{ stale_count }} stale</span> <span class="badge badge-red">{{ overridden_count }} overridden</span></div><table><thead><tr><th>Claim</th><th>Oracle</th><th>Domain</th><th>Last evidence</th><th>Status</th></tr></thead><tbody>{% for claim in claims %}<tr><td>{{ claim.description }}</td><td><code>{{ claim.oracle }}</code></td><td>{{ claim.assurance_domain }}</td><td>{{ claim.last_evidence_at or "—" }}</td><td><span class="badge badge-{{ 'green' if claim.status == 'fresh' else 'yellow' if claim.status == 'stale' else 'red' if claim.status == 'overridden' else '' }}">{{ claim.status }}</span></td></tr>{% endfor %}</tbody></table>{% endblock %}"""

COVERAGE_TEMPLATE = """{% extends "base.html" %}{% block content %}<h1>Coverage Map</h1><p>Line coverage: {{ coverage_pct }}% · {{ modules_tested }}/{{ modules_total }} modules tested</p><table><thead><tr><th>Module</th><th>Line %</th><th>Branch %</th><th>Status</th></tr></thead><tbody>{% for mod in modules %}<tr><td><code>{{ mod.name }}</code></td><td>{{ mod.line_pct }}%</td><td>{{ mod.branch_pct }}%</td><td><span class="badge badge-{{ 'green' if mod.line_pct >= 85 else 'yellow' if mod.line_pct >= 70 else 'red' }}">{{ 'good' if mod.line_pct >= 85 else 'ok' if mod.line_pct >= 70 else 'low' }}</span></td></tr>{% endfor %}</tbody></table>{% endblock %}"""

MUTATION_TEMPLATE = """{% extends "base.html" %}{% block content %}<h1>Mutation Scores</h1><p>Committed index from the most recent campaign run.</p><table><thead><tr><th>Campaign</th><th>Score</th><th>Trend</th><th>Survivors</th></tr></thead><tbody>{% for camp in campaigns %}<tr><td>{{ camp.name }}</td><td>{{ camp.score_pct }}%</td><td>{{ '↑' if camp.trend > 0 else '↓' if camp.trend < 0 else '→' }} {{ camp.trend|abs if camp.trend else '' }}</td><td>{{ camp.survivor_count }}</td></tr>{% endfor %}</tbody></table>{% endblock %}"""

DRIFT_TEMPLATE = """{% extends "base.html" %}{% block content %}<h1>Schema Drift</h1><p>Per-provider schema freshness. A provider is stale after 30 days without regeneration.</p><div class="card-grid">{% for prov in providers %}<div class="card"><h3>{{ prov.name }}</h3><p>Version: <code>{{ prov.version }}</code><br>Last regenerated: {{ prov.last_regenerated }}<br>Age: {{ prov.age_days }} days</p>{% if prov.drift_detected %}<span class="badge badge-red">drift detected</span>{% endif %}</div>{% endfor %}</div>{% endblock %}"""

FRESHNESS_TEMPLATE = """{% extends "base.html" %}{% block content %}<h1>Evidence Freshness</h1><p>Per-claim staleness status.</p><div class="card"><span class="badge badge-green">{{ fresh_count }} fresh</span> <span class="badge badge-yellow">{{ stale_count }} stale</span> <span class="badge badge-red">{{ overridden_count }} overridden</span> <span class="badge">{{ uncollected_count }} uncollected</span></div><table><thead><tr><th>Claim</th><th>Policy</th><th>Last collected</th><th>Status</th></tr></thead><tbody>{% for claim in claims %}<tr><td>{{ claim.description }}</td><td>{{ claim.staleness_policy }}</td><td>{{ claim.last_collected_at or "—" }}</td><td><span class="badge badge-{{ 'green' if claim.staleness == 'fresh' else 'yellow' if claim.staleness == 'approaching' else 'red' if claim.staleness in ('stale','overridden') else '' }}">{{ claim.staleness }}</span></td></tr>{% endfor %}</tbody></table>{% endblock %}"""

PAGES_TEMPLATES = {
    "base.html": BASE_TEMPLATE,
    "home.html": HOME_TEMPLATE,
    "doc.html": DOC_TEMPLATE,
    "verifiability_catalog.html": VERIFIABILITY_CATALOG_TEMPLATE,
    "verifiability_coverage.html": COVERAGE_TEMPLATE,
    "verifiability_mutation.html": MUTATION_TEMPLATE,
    "verifiability_drift.html": DRIFT_TEMPLATE,
    "verifiability_freshness.html": FRESHNESS_TEMPLATE,
}

__all__ = ["PAGES_TEMPLATES"]
