//! The reader's pages, rendered on the server from the same requests the
//! CLI sends. Every page works without script;
//! `reader.js` adds keys (search focus, next and previous, expand, copy as
//! CLI command). Archive text is escaped, so markup in a message stays
//! text.

use polylogue_model::{Message, Session};
use polylogue_render::html::{encode_path_segment as encode_id, escape as esc};
use polylogue_render::{Rendered, View, html, results, shell_quote};
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
<a href="/cost">Cost and usage</a>
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
    body.push_str(
        "</dl><p><a href=\"/search?q=documents\">Browse Documents</a></p><h2>Latest sessions</h2>",
    );
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

/// Navigation follows compiler-owned column reference metadata.
fn results_table(data: &Value) -> String {
    if !data["rows"].is_array() || !data["columns"].is_array() {
        return "<p>This result has no table view. Use copy as CLI to read the report.</p>".into();
    }
    let mut table = Vec::new();
    let _ = results::linked_html(data, &mut table, &|column, value| {
        let id = value.as_str()?;
        let route = match column["reference"].as_str()? {
            "session" => "s",
            "document" => "d",
            _ => return None,
        };
        Some(format!("/{route}/{}", encode_id(id)))
    });
    let mut html = String::from_utf8_lossy(&table).into_owned();
    if data["columns"]
        .as_array()
        .into_iter()
        .flatten()
        .any(|c| matches!(c["reference"].as_str(), Some("message" | "note")))
    {
        html.push_str("<p>Message and note references are shown as text; their dedicated readers are unavailable.</p>");
    }
    html
}

pub fn document(
    document: &polylogue_model::Document,
    sources: &Value,
    view: View,
    view_name: &str,
) -> String {
    let id = document.id.to_string();
    let title = document.title.as_deref().unwrap_or(&id);
    let mut body = String::from("<nav class=\"views\" aria-label=\"Document view\">");
    for v in ["full", "dialogue"] {
        let _ = write!(
            body,
            "<a href=\"/d/{}?view={v}\"{}>{v}</a> ",
            encode_id(&id),
            if v == view_name {
                " aria-current=\"page\""
            } else {
                ""
            }
        );
    }
    body.push_str("</nav>");
    let mut content = Vec::new();
    html::fragment(&[Rendered::Document(document)], view, &mut content).expect("UTF-8 HTML");
    body.push_str(&String::from_utf8(content).expect("UTF-8 HTML"));
    let time = document
        .source_time
        .map_or_else(|| "unknown".into(), |t| t.to_string());
    let _ = write!(
        body,
        "<p class=\"meta\">Source record time: {}</p>",
        esc(&time)
    );
    if !document.metadata.is_null() {
        let _ = write!(
            body,
            "<details><summary>Source metadata</summary><pre>{}</pre></details>",
            esc(&serde_json::to_string_pretty(&document.metadata).expect("JSON"))
        );
    }
    let _ = write!(
        body,
        "<details><summary>Source records</summary><pre>{}</pre></details>",
        esc(&serde_json::to_string_pretty(sources).expect("JSON"))
    );
    layout(
        title,
        "",
        &format!(
            "polylogue read {} --view {view_name}",
            shell_quote(&format!("document:{id}"))
        ),
        &body,
    )
}

/// `/s/SESSION`: the session in the chosen view; tool calls collapsed to
/// one line, outcome first, expandable.
pub fn session(
    session: &Session,
    messages: &[Message],
    view: View,
    view_name: &str,
    data: &Value,
    links: &Value,
) -> String {
    let id = session.id.to_string();
    let title = polylogue_render::mask(
        &data["display_title"]
            .as_str()
            .map(str::to_owned)
            .unwrap_or_else(|| {
                session
                    .title
                    .as_ref()
                    .map_or_else(|| id.clone(), |t| t.text.clone())
            }),
    );
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
    for &v in View::NAMES {
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
    let scope = format!(
        "sessions where id:{}",
        serde_json::to_string(&id).expect("string")
    );
    let _ = write!(
        body,
        "<p><a href=\"/cost?q={}\">Session cost and usage</a></p>",
        encode_id(&scope)
    );
    body.push_str(&parents(&data["parents"], links));
    let items: Vec<_> = messages.iter().map(Rendered::Message).collect();
    let mut body = body.into_bytes();
    html::fragment(&items, view, &mut body).expect("writing UTF-8 HTML to a Vec cannot fail");
    body.extend_from_slice(b"</article>");
    let body = String::from_utf8(body).expect("HTML is rendered from UTF-8 strings");
    let cli = format!(
        "polylogue read {} --with-parents --view {view_name}",
        shell_quote(&id)
    );
    layout(&title, "", &cli, &body)
}

/// Usage text is shared with the CLI, including unknown and partial totals.
pub fn cost(scope: &str, data: &Value) -> String {
    let report = polylogue_render::measures::usage(data);
    let count = |key: &str| {
        data[key]
            .as_u64()
            .map_or_else(|| "unknown".to_owned(), |n| n.to_string())
    };
    let body = format!(
        "<h1>Cost and usage</h1><form action=\"/cost\" method=\"get\"><label for=\"scope\">Session query</label> <input id=\"scope\" name=\"q\" value=\"{}\"> <button>Measure</button></form>\
         <p>Physical response records at source record time. API-equivalent figures use catalog list prices; tool-reported figures remain separate. Known sums exclude unreported usage and unpriced responses shown below.</p>\
         <p>{} known response records; {} model items excluded because response boundaries are unknown. Independent usage observations are reported separately below.</p><pre>{}</pre><p><a href=\"/search?q={}\">Browse selected sessions</a></p>",
        esc(scope),
        count("known_response_messages"),
        count("unknown_boundary_items"),
        esc(&report),
        encode_id(scope)
    );
    layout(
        "Cost and usage",
        "",
        &format!("polylogue measure usage {}", shell_quote(scope)),
        &body,
    )
}

fn parents(status: &Value, links: &Value) -> String {
    let mut body = String::from("<section aria-label=\"Parent sessions\"><h2>Parent sessions</h2>");
    match status["complete"].as_bool() {
        Some(true) => body.push_str("<p>Every parent session is here.</p>"),
        Some(false) => body.push_str("<p role=\"status\">Some parent sessions are missing; only the messages stored are shown.</p>"),
        None => body.push_str("<p>Whether every parent session is here is unknown.</p>"),
    }
    if let Some(issues) = status["issues"].as_array().filter(|v| !v.is_empty()) {
        body.push_str("<ul>");
        for issue in issues {
            let _ = write!(body, "<li><code>{}</code></li>", esc(&issue.to_string()));
        }
        body.push_str("</ul>");
    }
    if let Some(error) = links["error"].as_str() {
        let _ = write!(
            body,
            "<p role=\"status\">Links unavailable: {}</p>",
            esc(error)
        );
    } else {
        body.push_str("<div class=\"parents-table\"><table class=\"results\"><caption>Recorded session relationships</caption><thead><tr>");
        let columns = [
            "parent",
            "child",
            "kind",
            "how",
            "evidence",
            "pending",
            "branch_point",
        ];
        for column in columns {
            let _ = write!(
                body,
                "<th scope=\"col\">{}</th>",
                esc(&column.replace('_', " "))
            );
        }
        body.push_str("</tr></thead><tbody>");
        for row in links["rows"].as_array().into_iter().flatten() {
            body.push_str("<tr>");
            for column in columns {
                let value = &row[column];
                let shown = match value {
                    Value::Null => "unknown".into(),
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                if matches!(column, "parent" | "child") && value.as_str().is_some() {
                    let _ = write!(
                        body,
                        "<td><a href=\"/s/{}\">{}</a></td>",
                        encode_id(&shown),
                        esc(&shown)
                    );
                } else {
                    let _ = write!(body, "<td>{}</td>", esc(&shown));
                }
            }
            body.push_str("</tr>");
        }
        body.push_str("</tbody></table></div>");
        if links["next"].as_str().is_some() {
            body.push_str("<p>Only the first 100 links are shown.</p>");
        }
    }
    body.push_str("</section>");
    body
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

#[cfg(test)]
mod tests;
