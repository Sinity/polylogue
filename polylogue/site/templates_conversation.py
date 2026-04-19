"""Conversation page template for the static site."""

from __future__ import annotations

CONVERSATION_STYLE = """
        :root {
            --bg-primary: #0a0a0c;
            --bg-secondary: #16161a;
            --bg-elevated: #1e1e24;
            --bg-code: #282c34;
            --text-primary: #f8f9fa;
            --text-secondary: #94a3b8;
            --text-muted: #6b7280;
            --accent: #6366f1;
            --border: #2d2d35;
        }

        @media (prefers-color-scheme: light) {
            :root {
                --bg-primary: #ffffff;
                --bg-secondary: #f9fafb;
                --bg-elevated: #ffffff;
                --bg-code: #f6f8fa;
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
        }

        .container { max-width: 900px; margin: 0 auto; padding: 2rem 1.5rem; }

        .nav-back {
            color: var(--accent);
            text-decoration: none;
            display: inline-block;
            margin-bottom: 1.5rem;
        }

        h1 { font-size: 1.75rem; margin-bottom: 0.5rem; }

        .conv-meta {
            display: flex;
            gap: 1rem;
            font-size: 0.875rem;
            color: var(--text-secondary);
            margin-bottom: 2rem;
            flex-wrap: wrap;
        }

        .message {
            padding: 1rem 1.25rem;
            margin: 0.75rem 0;
            border-radius: 8px;
            border-left: 3px solid var(--border);
        }

        .message-user {
            background: rgba(99, 102, 241, 0.06);
            border-left-color: rgba(99, 102, 241, 0.5);
        }

        .message-assistant {
            background: rgba(16, 185, 129, 0.06);
            border-left-color: rgba(16, 185, 129, 0.5);
        }

        .message-system {
            background: rgba(245, 158, 11, 0.06);
            border-left-color: rgba(245, 158, 11, 0.5);
        }

        .message-tool {
            background: rgba(139, 92, 246, 0.06);
            border-left-color: rgba(139, 92, 246, 0.5);
            font-size: 0.85rem;
        }

        .message-role {
            font-weight: 600;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }

        .message-body {
            word-break: break-word;
            line-height: 1.7;
        }

        .message-body p { margin-bottom: 0.75rem; }
        .message-body p:last-child { margin-bottom: 0; }

        .message-body ul, .message-body ol {
            margin: 0.5rem 0 0.75rem 1.5rem;
        }

        .message-body li { margin-bottom: 0.25rem; }

        .message-body blockquote {
            border-left: 3px solid var(--accent);
            padding-left: 1rem;
            margin: 0.5rem 0;
            color: var(--text-secondary);
        }

        .message-body a { color: var(--accent); }

        .message-body h1, .message-body h2, .message-body h3,
        .message-body h4, .message-body h5, .message-body h6 {
            margin: 1rem 0 0.5rem;
        }

        .message-body table {
            border-collapse: collapse;
            margin: 0.75rem 0;
            width: 100%;
        }

        .message-body th, .message-body td {
            border: 1px solid var(--border);
            padding: 0.4rem 0.75rem;
            text-align: left;
        }

        .message-body th {
            background: var(--bg-elevated);
            font-weight: 600;
        }

        pre {
            background: var(--bg-code, #282c34);
            padding: 0.75rem;
            border-radius: 6px;
            overflow-x: auto;
            font-size: 0.85em;
            margin: 0.5rem 0;
            border: 1px solid var(--border);
        }

        code {
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            font-size: 0.9em;
        }

        p code {
            background: var(--bg-elevated, #1e1e24);
            padding: 0.125rem 0.375rem;
            border-radius: 4px;
        }

        .highlight {
            background: var(--bg-code, #282c34) !important;
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

        .footer {
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid var(--border);
            text-align: center;
            color: var(--text-muted);
            font-size: 0.8rem;
        }
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
            <div class="message-body">{{{{ msg.html_content | safe }}}}</div>
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
