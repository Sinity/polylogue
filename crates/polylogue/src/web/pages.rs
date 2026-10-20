//! The reader's pages, rendered on the server from the same requests the
//! CLI sends (spec `surfaces/web.md`). Every page works without script;
//! `reader.js` adds keys (search focus, next and previous, expand, copy as
//! CLI command). Archive text is escaped, so markup in a message stays
//! text.

use polylogue_model::{Message, Session};
use polylogue_render::html::{encode_path_segment as encode_id, escape as esc};
use polylogue_render::{Format, Rendered, View, html, results, shell_quote};
use serde_json::Value;
use std::fmt::Write as _;

/// The page frame: one stylesheet, one script, a search box on every page,
/// and the CLI command that shows the same thing.
pub fn layout(title: &str, query: &str, cli: &str, body: &str) -> String {
    format!(
        r#"<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · Polylogue</title>
<link rel="stylesheet" href="/assets/reader.css">
<script type="module" src="/assets/reader.js"></script>
</head><body>
<header class="top"><a class="home" href="/">Polylogue</a>
<form class="search" action="/search" method="get" role="search">
<label for="q" class="visually-hidden">Query</label>
<input id="q" name="q" type="search" value="{query}" placeholder="Search or query (/)" autocomplete="off" spellcheck="false">
</form>
<button type="button" class="copy-cli" data-cli="{cli}" title="Copy as CLI command (y)">copy as CLI</button>
</header>
<main id="main">{body}</main>
</body></html>
"#,
        title = esc(title),
        query = esc(query),
        cli = esc(cli),
    )
}

/// `/`: the overview bare `polylogue` shows, and the latest sessions.
pub fn home(status: &Value, sessions: &Value) -> String {
    let mut body = String::from("<h1>Overview</h1><dl class=\"status\">");
    if let Some(store) = status["store"].as_object() {
        for (k, v) in store {
            let shown = match v {
                Value::Null => "unknown".to_owned(),
                Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            let _ = write!(
                body,
                "<div><dt>{}</dt><dd>{}</dd></div>",
                esc(&k.replace('_', " ")),
                esc(&shown)
            );
        }
    }
    body.push_str("</dl><h2>Latest sessions</h2>");
    body.push_str(&results_table(sessions));
    layout("Overview", "", "polylogue", &body)
}

pub fn search(query: &str, data: Option<&Value>, error: Option<&Value>) -> String {
    let cli = format!("polylogue list {}", shell_quote(query));
    let mut body = format!("<h1>Results for <code>{}</code></h1>", esc(query));
    if let Some(e) = error {
        let _ = write!(
            body,
            "<div class=\"error\" role=\"alert\"><p>{}</p>",
            esc(e["message"].as_str().unwrap_or("the query failed"))
        );
        if let Some(fix) = e["detail"]["corrected"].as_str() {
            let _ = write!(
                body,
                "<p>Try <a href=\"/search?q={}\"><code>{}</code></a></p>",
                encode_id(fix),
                esc(fix)
            );
        }
        body.push_str("</div>");
    }
    if let Some(data) = data {
        body.push_str(&results_table(data));
    }
    layout(&format!("Search: {query}"), query, &cli, &body)
}

/// A query's answer as a table, each session ID linked to its page.
fn results_table(data: &Value) -> String {
    let mut table = Vec::new();
    let _ = results::render(data, Format::Html, &mut table);
    link_sessions(&String::from_utf8_lossy(&table), data)
}

/// Wraps each session ID that appears as a table cell in a link to its
/// reader page.
fn link_sessions(html: &str, data: &Value) -> String {
    let mut out = html.to_owned();
    let mut seen = std::collections::HashSet::new();
    for row in data["rows"].as_array().into_iter().flatten() {
        for key in ["session", "id"] {
            let Some(id) = row[key].as_str() else {
                continue;
            };
            if !id.contains(':') || !seen.insert(id.to_owned()) {
                continue;
            }
            let cell = format!("<td>{}</td>", esc(id));
            let link = format!("<td><a href=\"/s/{}\">{}</a></td>", encode_id(id), esc(id));
            out = out.replace(&cell, &link);
        }
    }
    out
}

/// `/s/SESSION`: the session in the chosen view; tool calls collapsed to
/// one line, outcome first, expandable.
pub fn session(session: &Session, messages: &[Message], view: View, view_name: &str) -> String {
    let id = session.id.to_string();
    let title = session
        .title
        .as_ref()
        .map_or_else(|| id.clone(), |t| t.text.clone());
    let mut body = format!(
        "<article class=\"session\"><h1>{}</h1><p class=\"meta\"><code>{}</code>",
        esc(&title),
        esc(&id)
    );
    if let Some(cwd) = session.cwd.last() {
        let _ = write!(body, " · {}", esc(cwd));
    }
    if let Some(t) = session.started {
        let _ = write!(body, " · <time>{}</time>", esc(&t.to_string()));
    }
    body.push_str("</p><nav class=\"views\" aria-label=\"View\">");
    for v in ["full", "dialogue", "prompts", "actions"] {
        let _ = write!(
            body,
            "<a href=\"/s/{}?view={v}\"{}>{v}</a> ",
            encode_id(&id),
            if v == view_name {
                " aria-current=\"page\""
            } else {
                ""
            }
        );
    }
    body.push_str("</nav>");
    let items: Vec<_> = messages.iter().map(Rendered::Message).collect();
    body.push_str(&html::fragment(&items, view));
    body.push_str("</article>");
    let cli = format!("polylogue read {} --view {view_name}", shell_quote(&id));
    layout(&title, "", &cli, &body)
}

/// A page for an error: a typed kind and a message, never a local path or
/// a stack trace.
pub fn error_page(kind: &str, message: &str) -> String {
    layout(
        kind,
        "",
        "polylogue status",
        &format!(
            "<div class=\"error\" role=\"alert\"><h1>{}</h1><p>{}</p></div>",
            esc(kind),
            esc(message)
        ),
    )
}

/// `/auth`: takes the token from the URL fragment (never sent to the
/// server or kept in history) and exchanges it for the archive's cookie.
pub fn auth() -> String {
    layout(
        "Sign in",
        "",
        "polylogue web",
        "<div class=\"auth\" id=\"auth\"><p>Opening the reader… If this page stays, run <code>polylogue web</code> again.</p></div>",
    )
}
