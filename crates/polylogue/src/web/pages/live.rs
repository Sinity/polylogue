use super::*;

pub fn home(sessions: &Value, recent: bool, cursor: Option<&str>) -> String {
    let title = if recent {
        "Recent sessions"
    } else {
        "Live sessions"
    };
    let query = if recent {
        crate::web::live::RECENT
    } else {
        crate::web::live::LIVE
    };
    let mut body = format!(
        r#"<link rel="stylesheet" href="/assets/sessions.css"><script type="module" src="/assets/sessions.js"></script>
<h1>{title}</h1><nav aria-label="Session lists"><a href="/">Live sessions</a> · <a href="/?recent=true">Recent sessions</a> · <a href="/search?q=notes%20where%20kind%3Asaved_query">Saved queries</a></nav>
<p>State comes from recorded events and the daemon's clock. Unknown means no current evidence.</p>
<p><a href="">Refresh</a> <button id="pause-updates" type="button" hidden>Pause updates</button> <button id="apply-updates" type="button" hidden>Updates available</button></p>
<p id="live-announcement" role="status" aria-live="polite"></p>
<section id="session-list" aria-label="{title}">"#
    );
    let count = sessions["rows"].as_array().map_or(0, Vec::len);
    let _ = write!(body, "<p>{count} sessions on this page.</p>");
    if count == 0 {
        body.push_str(
            "<p>No sessions match this list. Recent sessions includes ended sessions.</p>",
        );
    }
    body.push_str(
        "<table class=\"results\"><caption>Sessions in daemon order</caption><thead><tr>",
    );
    for label in [
        "State",
        "Session",
        "Repository or directory",
        "Harness",
        "Last observed activity",
        "State evidence",
    ] {
        let _ = write!(body, "<th scope=\"col\">{label}</th>");
    }
    body.push_str("</tr></thead><tbody>");
    for row in sessions["rows"].as_array().into_iter().flatten() {
        let id = row["id"].as_str().unwrap_or_default();
        let text = |key: &str| match &row[key] {
            Value::Null => "Unknown".to_owned(),
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        let _ = write!(
            body,
            "<tr class=\"hit\" data-session=\"{}\"><td>{}</td><td><a href=\"/s/{}?latest=true\">{}</a></td><td>{}</td><td>{}</td><td>{}</td><td><details><summary>{}</summary><p>Last event: {}. State changes at: {}.</p></details></td></tr>",
            esc(id),
            esc(&text("live")),
            encode_id(id),
            esc(row["title"].as_str().unwrap_or(id)),
            esc(row["repo"]
                .as_str()
                .or(row["cwd"].as_str())
                .unwrap_or("Unknown")),
            esc(&text("harness")),
            esc(&text(if row["last_event_at"].is_null() {
                "last_active"
            } else {
                "last_event_at"
            })),
            esc(row["waiting_on"].as_str().unwrap_or("Recorded state")),
            esc(&text("last_event")),
            esc(&text("live_until"))
        );
    }
    body.push_str("</tbody></table><nav class=\"paging\" aria-label=\"Session pages\">");
    if cursor.is_some() {
        body.push_str(
            "<button type=\"button\" data-previous hidden>Previous visited page</button> ",
        );
    }
    if let Some(next) = sessions["next"].as_str() {
        let _ = write!(
            body,
            "<a data-next rel=\"next\" href=\"/?recent={recent}&amp;cursor={}\">Next</a>",
            encode_id(next)
        );
    }
    body.push_str("</nav></section><section aria-label=\"Notices\"><h2>Notices</h2><div id=\"notices\">Notices have not been requested.</div><button id=\"load-notices\" type=\"button\" hidden>Load notices</button><p><a href=\"/search?q=deliveries%20where%20sender%3Asignal%20%7C%20sort%20by%20at%20desc\">Browse all notices</a></p></section>");
    layout(
        title,
        "",
        &format!("polylogue list {}", shell_quote(query)),
        &body,
    )
}
