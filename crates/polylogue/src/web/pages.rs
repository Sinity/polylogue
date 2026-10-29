//! Product pages share the common frame, assets and keyboard behavior.
//! Read pages consume typed ReadDocument markup from the same renderer.

use polylogue_render::html::{encode_path_segment as encode_id, escape as esc};
use polylogue_render::{results, shell_quote};
mod cost;
mod overview;
pub(crate) use polylogue_render::web as reader;
mod search;
pub use cost::cost;
pub use overview::home;
pub use search::search;
use serde_json::Value;
use std::fmt::Write as _;

/// Every product page uses the common frame and browser behavior.
pub(crate) fn frame<'a>(title: &'a str, query: &'a str, cli: &'a str) -> web_shell::Frame<'a> {
    let mut frame = web_shell::Frame::new(web_shell::Product::Polylogue, title, "/search", cli);
    frame.query = query;
    frame.navigation = polylogue_render::web::navigation_entries();
    frame
}
pub fn layout(title: &str, query: &str, cli: &str, body: &str) -> String {
    // Existing product templates produce escaped markup. Read pages below
    // consume typed ReadDocument markup directly without this string adapter.
    let body = maud::PreEscaped(body.to_owned());
    polylogue_render::terminal::escape(
        &frame(title, query, cli)
            .render(&body, &|body: &web_shell::Markup| body.clone())
            .into_string(),
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
        "<div class=\"auth\" id=\"auth\"><p>Opening the reader… If this page stays, run <code>polylogue web</code> again.</p></div><script type=\"module\" src=\"/assets/auth.js\"></script>",
    )
}

#[cfg(test)]
mod tests;
