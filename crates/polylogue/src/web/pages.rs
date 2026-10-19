//! The reader's pages, rendered on the server from the same requests the
//! CLI sends (spec `surfaces/web.md`). Every page works without script;
//! `reader.js` adds keys (search focus, next and previous, expand, copy as
//! CLI command). Archive text is escaped, so markup in a message stays
//! text.

use polylogue_model::{Block, Message, Session};
use polylogue_render::{Format, View, outcome_label, pieces, results};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt::Write as _;

pub fn esc(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            c => out.push(c),
        }
    }
    out
}

/// An ID percent-encoded once as a whole, for a URL path segment.
pub fn encode_id(id: &str) -> String {
    let mut out = String::new();
    for b in id.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(b as char);
        } else {
            let _ = write!(out, "%{b:02X}");
        }
    }
    out
}

fn shell_quote(s: &str) -> String {
    if !s.is_empty()
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"-_.:/@%+=,".contains(&b))
    {
        s.to_owned()
    } else {
        format!("'{}'", s.replace('\'', r"'\''"))
    }
}

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
    body.push_str(&session_list(sessions));
    layout("Overview", "", "polylogue", &body)
}

fn session_list(rows: &Value) -> String {
    let Some(rows) = rows.as_array() else {
        return "<p class=\"empty\">No sessions.</p>".into();
    };
    if rows.is_empty() {
        return "<p class=\"empty\">No sessions.</p>".into();
    }
    let mut out = String::from(
        "<table class=\"sessions results\"><thead><tr><th>Session</th><th>Kind</th><th>Last activity</th><th>Messages</th><th>Directory</th></tr></thead><tbody>",
    );
    for r in rows {
        let id = r["id"].as_str().unwrap_or_default();
        let title = r["title"].as_str().unwrap_or(id);
        let _ = write!(
            out,
            "<tr class=\"hit\"><td><a href=\"/s/{}\">{}</a></td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
            encode_id(id),
            esc(title),
            esc(r["kind"].as_str().unwrap_or("")),
            esc(r["last_activity"].as_str().unwrap_or("unknown")),
            r["messages"]
                .as_u64()
                .map_or_else(|| "unknown".into(), |n| n.to_string()),
            esc(r["cwd"].as_str().unwrap_or("")),
        );
    }
    out.push_str("</tbody></table>");
    out
}

/// `/search?q=…`: the CLI's result table, with every session ID a link.
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
        let mut table = Vec::new();
        let _ = results::render(data, Format::Html, &mut table);
        body.push_str(&link_sessions(&String::from_utf8_lossy(&table), data));
    }
    layout(&format!("Search: {query}"), query, &cli, &body)
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

/// What a tool call's result looked like, for its one-line summary.
struct Outcome {
    label: String,
    text: String,
}

/// `/s/SESSION`: the session in the chosen view; tool calls collapsed to
/// one line, outcome first, expandable.
pub fn session(session: &Session, messages: &[Message], view: View, view_name: &str) -> String {
    let id = session.id.to_string();
    let title = session
        .title
        .as_ref()
        .map_or_else(|| id.clone(), |t| t.text.clone());
    let results = results_by_call(messages);
    let mut paired = std::collections::HashSet::new();
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
    for m in messages {
        let section = message(m, view, &results, &mut paired);
        body.push_str(&section);
    }
    body.push_str("</article>");
    let cli = format!("polylogue read {} --view {view_name}", shell_quote(&id));
    layout(&title, "", &cli, &body)
}

fn results_by_call(messages: &[Message]) -> HashMap<String, Outcome> {
    let mut out = HashMap::new();
    for m in messages {
        for b in &m.blocks {
            if let Block::ToolResult {
                call_id: Some(id),
                outcome,
                output,
                ..
            } = b
            {
                out.insert(
                    id.clone(),
                    Outcome {
                        label: outcome_label(outcome),
                        text: polylogue_render::mask(output.as_deref().unwrap_or("")),
                    },
                );
            }
        }
    }
    out
}

fn message(
    m: &Message,
    view: View,
    results: &HashMap<String, Outcome>,
    paired: &mut std::collections::HashSet<String>,
) -> String {
    let ps = pieces(m, view);
    // Call IDs of this message's blocks, in the order `pieces` keeps them.
    let call_ids: Vec<Option<&String>> = m
        .blocks
        .iter()
        .filter_map(|b| match b {
            Block::ToolCall { call_id, .. } | Block::ToolResult { call_id, .. } => {
                Some(call_id.as_ref())
            }
            _ => None,
        })
        .collect();
    let mut calls = call_ids.into_iter();
    let mut inner = String::new();
    for p in &ps {
        match p.kind {
            "text" => {
                let _ = write!(inner, "<div class=\"text\">{}</div>", paragraphs(&p.text));
            }
            "thinking" => {
                let _ = write!(
                    inner,
                    "<details class=\"thinking\"><summary>thinking</summary><pre>{}</pre></details>",
                    esc(&p.text)
                );
            }
            "tool_call" => {
                let id = calls.next().flatten();
                let result = id.and_then(|id| {
                    paired.insert(id.clone());
                    results.get(id)
                });
                let (label, output) =
                    result.map_or(("no result", ""), |r| (r.label.as_str(), r.text.as_str()));
                let class = if label == "ok" {
                    "ok"
                } else if label == "no result" || label == "unknown" {
                    "unknown"
                } else {
                    "error"
                };
                let _ = write!(
                    inner,
                    "<details class=\"tool\"><summary><span class=\"outcome {class}\">{}</span> <b>{}</b> <code>{}</code></summary><pre class=\"input\">{}</pre>{}</details>",
                    esc(label),
                    esc(p.label.as_deref().unwrap_or("tool")),
                    esc(&first_line(&p.text)),
                    esc(&p.text),
                    if output.is_empty() {
                        String::new()
                    } else {
                        format!("<pre class=\"output\">{}</pre>", esc(output))
                    }
                );
            }
            "tool_result" => {
                let id = calls.next().flatten();
                if id.is_some_and(|id| paired.contains(id)) {
                    continue;
                }
                let _ = write!(
                    inner,
                    "<details class=\"tool result\"><summary><span class=\"outcome\">{}</span> result</summary><pre class=\"output\">{}</pre></details>",
                    esc(p.label.as_deref().unwrap_or("")),
                    esc(&p.text)
                );
            }
            kind => {
                let _ = write!(
                    inner,
                    "<div class=\"{kind}\">[{}{}] {}</div>",
                    esc(kind),
                    p.label
                        .as_deref()
                        .map(|l| format!(" {}", esc(l)))
                        .unwrap_or_default(),
                    esc(&p.text)
                );
            }
        }
    }
    if inner.is_empty() {
        return String::new();
    }
    let role = role_name(m);
    format!(
        "<section class=\"message {role}\" id=\"m-{}\" tabindex=\"-1\"><h2>{role}{}{}</h2>{inner}</section>",
        encode_id(&m.id.to_string()),
        m.at.map(|t| format!(" <time>{}</time>", esc(&t.to_string())))
            .unwrap_or_default(),
        m.model
            .as_deref()
            .map(|x| format!(" <span class=\"model\">{}</span>", esc(x)))
            .unwrap_or_default(),
    )
}

fn role_name(m: &Message) -> &'static str {
    use polylogue_model::{PromptKind, Role};
    match (m.role, m.prompt_kind) {
        (Role::Assistant, _) => "assistant",
        (Role::System, _) => "system",
        (Role::User, Some(PromptKind::Harness)) => "harness",
        (Role::User, Some(PromptKind::Model)) => "model",
        (Role::User, _) => "user",
    }
}

fn first_line(s: &str) -> String {
    let line = s.lines().next().unwrap_or_default();
    if line.chars().count() > 120 {
        format!("{}…", line.chars().take(119).collect::<String>())
    } else {
        line.to_owned()
    }
}

fn paragraphs(text: &str) -> String {
    text.split("\n\n")
        .filter(|p| !p.trim().is_empty())
        .map(|p| format!("<p>{}</p>", esc(p).replace('\n', "<br>")))
        .collect()
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
