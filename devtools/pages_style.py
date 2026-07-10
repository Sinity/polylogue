"""CSS design tokens and base styles for the documentation site."""

from __future__ import annotations

PAGES_DESIGN_TOKENS = """
        :root {
            --bg-primary: #0B0E14;
            --bg-secondary: #13171F;
            --bg-tertiary: #1A1F2A;
            --text-primary: #E6E8EC;
            --text-secondary: #8B919A;
            --text-tertiary: #555B63;
            --accent: #3B82F6;
            --accent-hover: #2563EB;
            --border: #1E2430;
            --shadow: rgba(0, 0, 0, 0.4);
            --green: #22C55E;
            --yellow: #EAB308;
            --red: #EF4444;
            --provider-claude-code: #D97706;
            --provider-codex: #3B82F6;
            --provider-chatgpt: #10B981;
            --provider-gemini: #8B5CF6;
            --provider-claude-ai: #F59E0B;
            --header-height: 56px;
            --nav-width: 240px;
            --content-max-width: 720px;
            --footer-height: 40px;
            --font-body: 'Inter', system-ui, -apple-system, sans-serif;
            --font-code: 'JetBrains Mono', 'Fira Code', monospace;
        }
        @media (prefers-color-scheme: light) {
            :root {
                --bg-primary: #FAFAFA;
                --bg-secondary: #FFFFFF;
                --bg-tertiary: #F3F4F6;
                --text-primary: #111827;
                --text-secondary: #6B7280;
                --text-tertiary: #9CA3AF;
                --border: #E5E7EB;
                --shadow: rgba(0, 0, 0, 0.08);
            }
        }
"""

PAGES_BASE_CSS = """
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        html { font-size: 15px; }
        body {
            font-family: var(--font-body);
            background-color: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            -webkit-font-smoothing: antialiased;
        }
        .site-header {
            position: sticky; top: 0; z-index: 10;
            height: var(--header-height);
            background-color: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
            display: flex; align-items: center; padding: 0 1.5rem; gap: 1rem;
        }
        .site-header .logo { font-weight: 700; font-size: 1.1rem; color: var(--text-primary); text-decoration: none; white-space: nowrap; }
        .site-header .logo span { color: var(--accent); }
        .site-header .search-bar { flex: 1; max-width: 400px; margin: 0 auto; }
        .site-header .search-bar input {
            width: 100%; padding: 0.4rem 0.75rem; border-radius: 6px;
            border: 1px solid var(--border); background-color: var(--bg-tertiary);
            color: var(--text-primary); font-family: var(--font-body); font-size: 0.9rem;
        }
        .site-header .theme-toggle {
            background: none; border: 1px solid var(--border); border-radius: 6px;
            padding: 0.3rem 0.6rem; cursor: pointer; color: var(--text-secondary); font-size: 0.9rem;
        }
        .page-layout { display: flex; min-height: calc(100vh - var(--header-height)); }
        .site-nav {
            width: var(--nav-width); flex-shrink: 0; padding: 2rem 1.5rem;
            background-color: var(--bg-primary); border-right: 1px solid var(--border);
            overflow-y: auto; position: sticky; top: var(--header-height);
            height: calc(100vh - var(--header-height));
        }
        .site-nav .nav-section { margin-bottom: 1.5rem; }
        .site-nav .nav-section-title { font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-tertiary); margin-bottom: 0.5rem; }
        .site-nav a { display: block; padding: 0.3rem 0; color: var(--text-secondary); text-decoration: none; font-size: 0.9rem; }
        .site-nav a:hover { color: var(--text-primary); }
        .site-nav a.active { color: var(--accent); font-weight: 500; }
        .content { flex: 1; max-width: var(--content-max-width); margin: 0 auto; padding: 2rem 2rem 4rem; width: 100%; }
        h1 { font-size: 1.87rem; font-weight: 600; line-height: 1.3; margin: 2rem 0 1rem; }
        h1:first-child { margin-top: 0; }
        h2 { font-size: 1.47rem; font-weight: 600; line-height: 1.3; margin: 2rem 0 0.75rem; }
        h3 { font-size: 1.2rem; font-weight: 600; line-height: 1.4; margin: 1.5rem 0 0.5rem; }
        p { margin-bottom: 1rem; }
        a { color: var(--accent); text-decoration: none; }
        a:hover { color: var(--accent-hover); }
        code { font-family: var(--font-code); font-size: 0.87rem; background-color: var(--bg-tertiary); padding: 0.15em 0.4em; border-radius: 3px; }
        pre { background-color: var(--bg-secondary); border: 1px solid var(--border); border-radius: 6px; padding: 1rem 1.25rem; overflow-x: auto; margin-bottom: 1.25rem; font-size: 0.87rem; line-height: 1.5; }
        pre code { background: none; padding: 0; font-size: inherit; }
        table { width: 100%; border-collapse: collapse; margin-bottom: 1.25rem; font-size: 0.9rem; }
        th, td { padding: 0.5rem 0.75rem; text-align: left; border-bottom: 1px solid var(--border); }
        th { font-weight: 600; color: var(--text-secondary); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; }
        .card { background-color: var(--bg-secondary); border: 1px solid var(--border); border-radius: 8px; padding: 1.25rem; margin-bottom: 1rem; }
        .card-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }
        .stat-card { background-color: var(--bg-secondary); border: 1px solid var(--border); border-radius: 8px; padding: 1.25rem; text-align: center; }
        .stat-card .stat-value { font-size: 2rem; font-weight: 700; color: var(--accent); }
        .stat-card .stat-label { font-size: 0.8rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.25rem; }
        .badge { display: inline-block; padding: 0.15em 0.5em; border-radius: 4px; font-size: 0.8rem; font-weight: 500; }
        .badge-green { background-color: rgba(34, 197, 94, 0.15); color: var(--green); }
        .badge-yellow { background-color: rgba(234, 179, 8, 0.15); color: var(--yellow); }
        .badge-red { background-color: rgba(239, 68, 68, 0.15); color: var(--red); }
        .site-footer { height: var(--footer-height); display: flex; align-items: center; justify-content: space-between; padding: 0 1.5rem; border-top: 1px solid var(--border); font-size: 0.8rem; color: var(--text-tertiary); }
        .site-footer a { color: var(--text-secondary); }
        .home-hero { text-align: center; padding: 4rem 0 3rem; }
        .home-hero .eyebrow { font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.12em; color: var(--accent); margin-bottom: 0.8rem; }
        .home-hero h1 { font-size: clamp(2.4rem, 7vw, 4.5rem); line-height: 1.03; max-width: 820px; margin: 0 auto 1rem; letter-spacing: -0.04em; }
        .home-hero .tagline { font-size: 1.18rem; color: var(--text-secondary); max-width: 760px; margin: 0 auto 1.3rem; }
        .hero-command { margin: 0 auto 1rem; }
        .hero-command code { display: inline-block; padding: 0.65rem 0.9rem; border: 1px solid var(--border); background: var(--bg-secondary); box-shadow: 0 10px 30px var(--shadow); }
        .hero-stats { display: flex; justify-content: center; flex-wrap: wrap; gap: 0.45rem 1.2rem; margin: 0 auto 1.7rem; color: var(--text-tertiary); font-size: 0.8rem; }
        .hero-stats strong { color: var(--text-primary); font-size: 0.92rem; margin-right: 0.18rem; }
        .proof-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1rem; margin: 1rem 0 2rem; text-align: left; }
        .proof-card { background: linear-gradient(180deg, var(--bg-secondary), var(--bg-primary)); border: 1px solid var(--border); border-radius: 10px; padding: 1.2rem; min-height: 155px; }
        .proof-card strong { display: block; font-size: 1rem; margin: 0.35rem 0 0.5rem; }
        .proof-card p { color: var(--text-secondary); font-size: 0.9rem; margin: 0; }
        .proof-label { color: var(--accent); font-size: 0.73rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; }
        .home-hero .hero-search { max-width: 500px; margin: 0 auto 3rem; }
        .home-hero .hero-search input { width: 100%; padding: 0.75rem 1rem; border-radius: 8px; border: 1px solid var(--border); background-color: var(--bg-secondary); color: var(--text-primary); font-family: var(--font-body); font-size: 1rem; }
        .home-links { display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap; margin-top: 2rem; }
        .home-links a { display: inline-block; padding: 0.6rem 1.5rem; border-radius: 6px; background-color: var(--bg-secondary); border: 1px solid var(--border); font-weight: 500; }
        .home-links a:hover { background-color: var(--bg-tertiary); border-color: var(--accent); }
        @media (max-width: 768px) {
            .site-nav { display: none; }
            .content { padding: 1.5rem 1rem; }
            .home-hero h1 { font-size: 2.35rem; }
            .proof-grid { grid-template-columns: 1fr; }
        }
        @media print {
            .site-header, .site-nav, .site-footer { display: none; }
            .content { max-width: none; padding: 0; }
            body { background: white; color: black; }
        }
"""

PAGES_STYLE = PAGES_DESIGN_TOKENS + PAGES_BASE_CSS

__all__ = ["PAGES_DESIGN_TOKENS", "PAGES_BASE_CSS", "PAGES_STYLE"]
