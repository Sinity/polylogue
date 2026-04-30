"""Conversation page template for the static site."""

from __future__ import annotations

CONVERSATION_STYLE = """
        :root {
            --bg-primary: #0a0a0c;
            --bg-secondary: #121418;
            --bg-elevated: #1a1d24;
            --bg-code: #1e2129;
            --text-primary: #e4e4e7;
            --text-secondary: #8b8fa3;
            --text-muted: #5c6072;
            --accent: #6366f1;
            --accent-soft: rgba(99, 102, 241, 0.08);
            --border: #23262e;
            --user-color: #6366f1;
            --assistant-color: #22c55e;
            --system-color: #f59e0b;
            --tool-color: #8b5cf6;
        }

        @media (prefers-color-scheme: light) {
            :root {
                --bg-primary: #ffffff;
                --bg-secondary: #f8f9fb;
                --bg-elevated: #ffffff;
                --bg-code: #f3f4f6;
                --text-primary: #18181b;
                --text-secondary: #52525b;
                --text-muted: #a1a1aa;
                --accent: #4f46e5;
                --accent-soft: rgba(79, 70, 229, 0.06);
                --border: #e4e4e7;
                --user-color: #4f46e5;
                --assistant-color: #16a34a;
            }
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            -webkit-font-smoothing: antialiased;
            font-size: 16px;
        }

        .container { max-width: 860px; margin: 0 auto; padding: 2rem 1.25rem 4rem; }

        .nav-back {
            color: var(--text-secondary);
            text-decoration: none;
            display: inline-block;
            margin-bottom: 1.5rem;
            font-size: 0.875rem;
            transition: color 0.2s;
        }
        .nav-back:hover { color: var(--accent); }

        .conv-header {
            margin-bottom: 2.5rem;
            padding-bottom: 1.5rem;
            border-bottom: 1px solid var(--border);
        }

        h1 {
            font-size: 1.6rem;
            font-weight: 700;
            line-height: 1.3;
            margin-bottom: 0.625rem;
            color: var(--text-primary);
        }

        .conv-meta {
            display: flex;
            gap: 1.25rem;
            font-size: 0.8125rem;
            color: var(--text-muted);
            flex-wrap: wrap;
            align-items: center;
        }

        .conv-meta .badge {
            background: var(--accent-soft);
            color: var(--accent);
            padding: 0.15rem 0.6rem;
            border-radius: 4px;
            font-weight: 600;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .message {
            padding: 1.125rem 1.25rem;
            margin: 0.5rem 0;
            border-radius: 6px;
            border-left: 3px solid var(--border);
            background: var(--bg-secondary);
            transition: background 0.15s;
        }

        .message-user {
            border-left-color: var(--user-color);
            background: color-mix(in srgb, var(--user-color) 4%, var(--bg-primary));
        }

        .message-assistant {
            border-left-color: var(--assistant-color);
        }

        .message-system {
            border-left-color: var(--system-color);
            font-style: italic;
        }

        .message-tool {
            border-left-color: var(--tool-color);
            font-size: 0.875rem;
            opacity: 0.85;
        }

        .message-role {
            font-weight: 600;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--text-muted);
            margin-bottom: 0.375rem;
        }

        .message-body {
            word-break: break-word;
            line-height: 1.72;
            font-size: 0.9375rem;
            color: var(--text-primary);
        }

        .message-body p { margin-bottom: 0.75rem; }
        .message-body p:last-child { margin-bottom: 0; }

        .message-body pre {
            background: var(--bg-code);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 1rem 1.125rem;
            overflow-x: auto;
            font-size: 0.8125rem;
            line-height: 1.55;
            margin: 0.75rem 0;
        }

        .message-body code {
            font-family: "JetBrains Mono", "Fira Code", "Cascadia Code", "SF Mono", Consolas, monospace;
            font-size: 0.8125rem;
        }

        .message-body :not(pre) > code {
            background: var(--bg-code);
            border: 1px solid var(--border);
            border-radius: 3px;
            padding: 0.125rem 0.375rem;
            font-size: 0.85em;
        }

        .message-body blockquote {
            border-left: 2px solid var(--accent);
            margin: 0.75rem 0;
            padding: 0.25rem 0 0.25rem 1rem;
            color: var(--text-secondary);
        }

        .message-body ul, .message-body ol {
            margin: 0.5rem 0;
            padding-left: 1.5rem;
        }

        .message-body li { margin-bottom: 0.25rem; }

        .message-body table {
            border-collapse: collapse;
            width: 100%;
            margin: 0.75rem 0;
            font-size: 0.875rem;
        }

        .message-body th, .message-body td {
            border: 1px solid var(--border);
            padding: 0.5rem 0.75rem;
            text-align: left;
        }

        .message-body th {
            background: var(--bg-elevated);
            font-weight: 600;
        }

        footer {
            text-align: center;
            color: var(--text-muted);
            font-size: 0.75rem;
            margin-top: 3rem;
            padding-top: 1.5rem;
            border-top: 1px solid var(--border);
        }

        footer a { color: var(--text-secondary); text-decoration: none; }
        footer a:hover { color: var(--accent); }

        @media (max-width: 640px) {
            .container { padding: 1rem 1rem 3rem; }
            h1 { font-size: 1.35rem; }
            .message { padding: 0.875rem 1rem; }
        }

        .message-body a { color: var(--accent); }

        .message-body h1, .message-body h2, .message-body h3,
        .message-body h4, .message-body h5, .message-body h6 {
            margin: 1rem 0 0.5rem;
        }

        .highlight {
            background: var(--bg-code) !important;
            border-radius: 6px;
            overflow-x: auto;
            margin: 0.5rem 0;
        }

        .highlight pre {
            margin: 0;
            padding: 0.75rem;
            background: transparent !important;
            border: none;
        }

        {{ highlight_css }}
"""


def _build_conversation_template() -> str:
    return f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{{{ title }}}} | Polylogue</title>
    <style>
{CONVERSATION_STYLE}
    </style>
</head>
<body>
    <div class="container">
        <a href="../../index.html" class="nav-back">← Back to Index</a>
        <h1>{{{{ title }}}}</h1>
        <div class="conv-meta">
            <span class="badge">{{{{ provider }}}}</span>
            <span>{{{{ message_count }}}} messages</span>
            {{% if updated_at %}}<span>{{{{ updated_at }}}}</span>{{% endif %}}
        </div>

        <div data-pagefind-body>
        {{% for msg in messages %}}
        <div class="message message-{{{{ msg.role or 'unknown' }}}}">
            <div class="message-role">{{{{ msg.role or 'unknown' }}}}</div>
            <div class="message-body">{{{{ msg.html_content | sanitize_html }}}}</div>
        </div>
        {{% endfor %}}
        </div>

        <div class="footer">
            <p>Generated by <a href="https://github.com/anthropics/polylogue">Polylogue</a></p>
        </div>
    </div>
</body>
</html>
"""


CONVERSATION_TEMPLATE = _build_conversation_template()

__all__ = ["CONVERSATION_STYLE", "CONVERSATION_TEMPLATE"]
