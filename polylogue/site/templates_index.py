"""Index page templates for the static site."""

from __future__ import annotations

INDEX_TEMPLATE = """<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        :root {
            --bg-primary: #0a0a0c;
            --bg-secondary: #16161a;
            --bg-elevated: #1e1e24;
            --text-primary: #f8f9fa;
            --text-secondary: #94a3b8;
            --text-muted: #6b7280;
            --accent: #6366f1;
            --accent-glow: rgba(99, 102, 241, 0.4);
            --border: #2d2d35;
        }

        @media (prefers-color-scheme: light) {
            :root {
                --bg-primary: #ffffff;
                --bg-secondary: #f9fafb;
                --bg-elevated: #ffffff;
                --text-primary: #111827;
                --text-secondary: #4b5563;
                --text-muted: #9ca3af;
                --border: #e5e7eb;
            }
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            -webkit-font-smoothing: antialiased;
        }

        .container {
            max-width: 1000px;
            margin: 0 auto;
            padding: 3rem 1.5rem;
        }

        header {
            margin-bottom: 3rem;
            text-align: center;
        }

        h1 {
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(to bottom right, #fff, #94a3b8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        .subtitle {
            color: var(--text-secondary);
            font-size: 1rem;
        }

        .stats {
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin: 2rem 0;
            flex-wrap: wrap;
        }

        .stat {
            text-align: center;
        }

        .stat-value {
            font-size: 2rem;
            font-weight: 700;
            color: var(--accent);
        }

        .stat-label {
            font-size: 0.875rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .conversation-list {
            display: grid;
            gap: 1.5rem;
            list-style: none;
        }

        .conversation-card {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
            transition: all 0.2s;
        }

        .conversation-card:hover {
            border-color: var(--accent);
            box-shadow: 0 0 20px var(--accent-glow);
            transform: translateY(-2px);
        }

        .conversation-link {
            text-decoration: none;
            color: inherit;
        }

        .conversation-title {
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 0.5rem;
        }

        .conversation-meta {
            display: flex;
            gap: 1rem;
            font-size: 0.875rem;
            color: var(--text-secondary);
            flex-wrap: wrap;
        }

        .badge {
            padding: 0.25rem 0.5rem;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            background: var(--bg-elevated);
            border: 1px solid var(--border);
        }

        .sidebar {
            margin-bottom: 2rem;
        }

        .provider-list {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            list-style: none;
        }

        .provider-list a {
            padding: 0.5rem 1rem;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text-secondary);
            text-decoration: none;
            font-size: 0.875rem;
            transition: all 0.2s;
        }

        .provider-list a:hover {
            border-color: var(--accent);
            color: var(--text-primary);
        }

        nav.nav-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 0;
            border-bottom: 1px solid var(--border);
            margin-bottom: 2rem;
        }

        nav a {
            color: var(--accent);
            text-decoration: none;
        }

        .footer {
            margin-top: 3rem;
            padding-top: 2rem;
            border-top: 1px solid var(--border);
            text-align: center;
            color: var(--text-muted);
            font-size: 0.875rem;
        }

        @media (max-width: 768px) {
            .container { padding: 2rem 1rem; }
            h1 { font-size: 2rem; }
            .stats { gap: 1rem; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{{ title }}</h1>
            <p class="subtitle">{{ description }}</p>
            <div class="stats">
                <div class="stat">
                    <div class="stat-value">{{ total_conversations }}</div>
                    <div class="stat-label">Conversations</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{{ total_messages }}</div>
                    <div class="stat-label">Messages</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{{ provider_count }}</div>
                    <div class="stat-label">Providers</div>
                </div>
            </div>
        </header>

        {{ search_markup | safe }}

        {% if providers %}
        <div class="sidebar">
            <ul class="provider-list">
                {% for provider, count in providers.items() %}
                <li><a href="{{ provider }}/index.html">{{ provider }} ({{ count }})</a></li>
                {% endfor %}
                <li><a href="dashboard.html">Dashboard</a></li>
            </ul>
        </div>
        {% endif %}

        <ul class="conversation-list">
            {% for conv in conversations %}
            <li class="conversation-card">
                <a href="{{ conv.path }}" class="conversation-link">
                    <h2 class="conversation-title">{{ conv.title or conv.id[:12] }}</h2>
                    <div class="conversation-meta">
                        <span class="badge">{{ conv.provider }}</span>
                        <span>{{ conv.message_count }} messages</span>
                        {% if conv.created_at %}
                        <span>{{ conv.created_at }}</span>
                        {% endif %}
                    </div>
                </a>
            </li>
            {% endfor %}
        </ul>

        <div class="footer">
            <p>Generated {{ generated_at }} by <a href="https://github.com/anthropics/polylogue">Polylogue</a></p>
        </div>
    </div>
</body>
</html>
"""

__all__ = ["INDEX_TEMPLATE"]
