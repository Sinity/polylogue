//! The reader's pages, rendered on the server from the same requests the
//! CLI sends. Every page works without script;
//! `reader.js` adds keys (search focus, next and previous, expand, copy as
//! CLI command). Archive text is escaped, so markup in a message stays
//! text.

use polylogue_render::html::{encode_path_segment as encode_id, escape as esc};
use polylogue_render::{results, shell_quote};
mod cost;
mod live;
pub(crate) mod reader;
mod search;
pub use cost::cost;
pub use live::home;
pub use search::search;
use serde_json::Value;
use std::fmt::Write as _;

/// The page frame: one stylesheet, one script, a search box on every page,
/// and the CLI command that shows the same thing.
pub fn layout(title: &str, query: &str, cli: &str, body: &str) -> String {
    format!(
        r##"<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · Polylogue</title>
<link rel="stylesheet" href="/assets/reader.css">
<script type="module" src="/assets/reader.js"></script>
</head><body>
<a class="skip" href="#main">Skip to content</a>
<header class="top"><a class="home" href="/">Polylogue</a>
<a href="/cost">Cost and usage</a>
<form class="search" action="/search" method="get" role="search">
<label for="q" class="visually-hidden">Query</label>
<input id="q" name="q" type="search" value="{query}" placeholder="Search or query (/)" autocomplete="off" spellcheck="false">
</form>
<button type="button" class="copy-cli" data-cli="{cli}" title="Copy as CLI command (y)">copy as CLI</button>
</header>
<details id="key-help"><summary>Keyboard help (?)</summary>
<label><input id="shortcuts" type="checkbox" checked> Single-character shortcuts</label>
<p>/ Search; j/k Move focus; o Expand; [ Previous visited page; ] Next page; y Copy CLI; ? Help.</p></details>
<main id="main" tabindex="-1">{body}</main>
</body></html>
"##,
        title = esc(title),
        query = esc(query),
        cli = esc(cli),
    )
}

/// Navigation follows compiler-owned column reference metadata.
fn results_table(data: &Value) -> String {
    if !data["rows"].is_array() || !data["columns"].is_array() {
        return "<p>This result has no table view. Use copy as CLI to read the report.</p>".into();
    }
    let mut table_data = data.clone();
    if let Some(table) = table_data.as_object_mut() {
        table.remove("next");
    }
    let mut table = Vec::new();
    let _ = results::linked_html(&table_data, &mut table, &|column, value| {
        let id = value.as_str()?;
        let route = match column["reference"].as_str()? {
            "session" => "s",
            "document" => "d",
            "message" => "m",
            "note" => "n",
            _ => return None,
        };
        Some(format!("/{route}/{}", encode_id(id)))
    });
    String::from_utf8_lossy(&table).into_owned()
}

fn next_results(query: &str, data: &Value) -> String {
    data["next"].as_str().map(|cursor|format!("<nav class=\"paging\" aria-label=\"Result pages\"><a rel=\"next\" data-next href=\"/search?q={}&amp;cursor={}\">Next</a></nav>",encode_id(query),encode_id(cursor))).unwrap_or_default()
}

fn parents(status: &Value, links: &Value) -> String {
    let mut body = String::from("<section aria-label=\"Parent sessions\"><h2>Parent sessions</h2>");
    match status["complete"].as_bool() {
        Some(true) => body.push_str("<p>Every selected parent session is here.</p>"),
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
            esc(links["message"].as_str().unwrap_or(error))
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
