"""Jinja2 templates for the documentation site."""

from __future__ import annotations

BASE_TEMPLATE = """<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }} — Polylogue</title>
    <style>
{{ style }}
    </style>
    {% block head %}{% endblock %}
</head>
<body>
    <header class="site-header">
        <a href="/polylogue/" class="logo">poly<span>logue</span></a>
        <div class="search-bar">
            <input type="search" placeholder="Search docs..." id="search-input">
        </div>
        <button class="theme-toggle" onclick="toggleTheme()" title="Toggle theme">
            <span class="light-icon">\u2600</span>
            <span class="dark-icon">\u263e</span>
        </button>
    </header>
    <div class="page-layout">
        <nav class="site-nav" id="site-nav">
            {% for section in nav %}
            <div class="nav-section">
                <div class="nav-section-title">{{ section.title }}</div>
                {% for item in section.entries %}
                <a href="/polylogue{{ item.path }}" {% if item.path == current_path %}class="active"{% endif %}>
                    {{ item.label }}
                </a>
                {% endfor %}
            </div>
            {% endfor %}
        </nav>
        <main class="content">
            {% block content %}{% endblock %}
        </main>
    </div>
    <footer class="site-footer">
        <span>Polylogue — your AI memory</span>
        <a href="https://github.com/Sinity/polylogue">GitHub</a>
    </footer>
    <script>
        function toggleTheme() {
            var html = document.documentElement;
            var current = html.getAttribute('data-theme');
            html.setAttribute('data-theme', current === 'dark' ? 'light' : 'dark');
            localStorage.setItem('theme', current === 'dark' ? 'light' : 'dark');
        }
        (function() {
            var saved = localStorage.getItem('theme');
            if (saved) document.documentElement.setAttribute('data-theme', saved);
        })();
    </script>
    {% block scripts %}{% endblock %}
</body>
</html>
"""

HOME_TEMPLATE = """{% extends "base.html" %}
{% block content %}
<div class="home-hero">
    <h1>Polylogue</h1>
    <p class="tagline">Your AI memory</p>
    <div class="hero-search">
        <input type="search" placeholder="Search your archive..."
               onkeydown="if(event.key==='Enter'){window.location='/polylogue/docs/search/?q='+encodeURIComponent(this.value)}">
    </div>
    <div class="card-grid">
        <div class="stat-card">
            <div class="stat-value">{{ stats.session_count }}</div>
            <div class="stat-label">sessions</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{{ stats.message_count }}</div>
            <div class="stat-label">messages</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{{ stats.provider_count }}</div>
            <div class="stat-label">providers</div>
        </div>
    </div>
    <div class="home-links">
        <a href="/polylogue/docs/getting-started/">Get started \u2192</a>
        <a href="/polylogue/docs/">Documentation \u2192</a>
        <a href="/polylogue/architecture/">Architecture \u2192</a>
        <a href="/polylogue/verifiability/">Verifiability \u2192</a>
    </div>
</div>
{% endblock %}
"""

DOC_TEMPLATE = """{% extends "base.html" %}
{% block content %}
{{ content | safe }}
<hr style="margin: 3rem 0 1rem; border-color: var(--border);">
<nav style="display: flex; justify-content: space-between; font-size: 0.9rem;">
    {% if prev %}<a href="/polylogue{{ prev.path }}">\u2190 {{ prev.label }}</a>{% else %}<span></span>{% endif %}
    {% if next %}<a href="/polylogue{{ next.path }}">{{ next.label }} \u2192</a>{% else %}<span></span>{% endif %}
</nav>
{% endblock %}
"""

VERIFIABILITY_CATALOG_TEMPLATE = """{% extends "base.html" %}
{% block content %}
<h1>Evidence Catalog</h1>
<p class="caption">Last updated: {{ updated_at }}</p>
<div class="card" style="margin-bottom: 2rem;">
    <strong>{{ claims|length }} claims</strong> &middot;
    <span class="badge badge-green">{{ fresh_count }} fresh</span>
    <span class="badge badge-yellow">{{ stale_count }} stale</span>
    <span class="badge badge-red">{{ overridden_count }} overridden</span>
</div>
<table>
<thead><tr><th>Claim</th><th>Oracle</th><th>Domain</th><th>Last Evidence</th><th>Status</th></tr></thead>
<tbody>
{% for claim in claims %}
<tr>
    <td>{{ claim.description }}</td>
    <td><code>{{ claim.oracle }}</code></td>
    <td>{{ claim.assurance_domain }}</td>
    <td>{{ claim.last_evidence_at or "\u2014" }}</td>
    <td>{% if claim.status == "fresh" %}<span class="badge badge-green">fresh</span>{% elif claim.status == "stale" %}<span class="badge badge-yellow">stale</span>{% elif claim.status == "overridden" %}<span class="badge badge-red">overridden</span>{% else %}<span class="badge">{{ claim.status }}</span>{% endif %}</td>
</tr>
{% endfor %}
</tbody>
</table>
{% endblock %}
"""

COVERAGE_TEMPLATE = """{% extends "base.html" %}
{% block content %}
<h1>Coverage Map</h1>
<p class="caption">Line coverage: {{ coverage_pct }}% &middot; {{ modules_tested }}/{{ modules_total }} modules tested</p>
<table>
<thead><tr><th>Module</th><th>Line %</th><th>Branch %</th><th>Status</th></tr></thead>
<tbody>
{% for mod in modules %}
<tr>
    <td><code>{{ mod.name }}</code></td>
    <td>{{ mod.line_pct }}%</td>
    <td>{{ mod.branch_pct }}%</td>
    <td>{% if mod.line_pct >= 85 %}<span class="badge badge-green">good</span>{% elif mod.line_pct >= 70 %}<span class="badge badge-yellow">ok</span>{% else %}<span class="badge badge-red">low</span>{% endif %}</td>
</tr>
{% endfor %}
</tbody>
</table>
{% endblock %}
"""

MUTATION_TEMPLATE = """{% extends "base.html" %}
{% block content %}
<h1>Mutation Scores</h1>
<p class="caption">Committed index from most recent campaign run</p>
<table>
<thead><tr><th>Campaign</th><th>Score</th><th>Trend</th><th>Survivors</th></tr></thead>
<tbody>
{% for camp in campaigns %}
<tr>
    <td>{{ camp.name }}</td>
    <td><div style="display: flex; align-items: center; gap: 0.5rem;">
        <div style="flex: 1; height: 8px; background: var(--bg-tertiary); border-radius: 4px;">
            <div style="width: {{ camp.score_pct }}%; height: 100%; background: {% if camp.score_pct >= 80 %}var(--green){% elif camp.score_pct >= 60 %}var(--yellow){% else %}var(--red){% endif %}; border-radius: 4px;"></div>
        </div>
        <span style="font-size: 0.85rem; font-weight: 600;">{{ camp.score_pct }}%</span>
    </div></td>
    <td>{% if camp.trend > 0 %}<span style="color: var(--green);">\u2191{{ camp.trend }}%</span>{% elif camp.trend < 0 %}<span style="color: var(--red);">\u2193{{ camp.trend|abs }}%</span>{% else %}<span style="color: var(--text-secondary);">\u2192</span>{% endif %}</td>
    <td>{{ camp.survivor_count }}</td>
</tr>
{% endfor %}
</tbody>
</table>
{% endblock %}
"""

DRIFT_TEMPLATE = """{% extends "base.html" %}
{% block content %}
<h1>Schema Drift</h1>
<p class="caption">Per-provider schema freshness. Red if &gt;30 days since last regeneration.</p>
<div class="card-grid">
{% for prov in providers %}
<div class="card" style="border-left: 3px solid {% if prov.age_days > 30 %}var(--red){% elif prov.age_days > 14 %}var(--yellow){% else %}var(--green){% endif %};">
    <h3 style="margin-top: 0;">{{ prov.name }}</h3>
    <p style="margin-bottom: 0.25rem;">Version: <code>{{ prov.version }}</code></p>
    <p style="margin-bottom: 0.25rem;">Last regenerated: {{ prov.last_regenerated }}</p>
    <p style="margin-bottom: 0;">Age: {{ prov.age_days }} days
        {% if prov.age_days > 30 %}<span class="badge badge-red">stale</span>
        {% elif prov.age_days > 14 %}<span class="badge badge-yellow">aging</span>
        {% else %}<span class="badge badge-green">fresh</span>{% endif %}
    </p>
    {% if prov.drift_detected %}<p style="margin-top: 0.5rem; color: var(--red);">Drift detected</p>{% endif %}
</div>
{% endfor %}
</div>
{% endblock %}
"""

FRESHNESS_TEMPLATE = """{% extends "base.html" %}
{% block content %}
<h1>Evidence Freshness</h1>
<p class="caption">Per-claim staleness status.</p>
<div class="card" style="margin-bottom: 2rem;">
    <span class="badge badge-green">{{ fresh_count }} fresh</span>
    <span class="badge badge-yellow">{{ stale_count }} stale</span>
    <span class="badge badge-red">{{ overridden_count }} overridden</span>
    <span>{{ uncollected_count }} uncollected</span>
</div>
<table>
<thead><tr><th>Claim</th><th>Policy</th><th>Last Collected</th><th>Status</th></tr></thead>
<tbody>
{% for claim in claims %}
<tr>
    <td>{{ claim.description }}</td>
    <td>{{ claim.staleness_policy }}</td>
    <td>{{ claim.last_collected_at or "\u2014" }}</td>
    <td>{% if claim.staleness == "fresh" %}<span class="badge badge-green">fresh</span>{% elif claim.staleness == "approaching" %}<span class="badge badge-yellow">approaching</span>{% elif claim.staleness == "stale" %}<span class="badge badge-red">stale</span>{% elif claim.staleness == "overridden" %}<span class="badge badge-red">overridden</span>{% else %}<span class="badge">{{ claim.staleness }}</span>{% endif %}</td>
</tr>
{% endfor %}
</tbody>
</table>
{% endblock %}
"""

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
