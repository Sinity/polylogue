use super::*;

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
    } else {
        let message = if let Some(error) = status["totals_error"].as_str() {
            format!("Totals could not be read: {error}")
        } else if !status["index_build"].is_null() {
            "The index is being rebuilt.".to_owned()
        } else if status["warming"].as_bool() == Some(true) {
            "The index is catching up.".to_owned()
        } else {
            "Archive totals are unavailable.".to_owned()
        };
        let _ = write!(
            body,
            "<div><dt>Archive totals</dt><dd>{}</dd></div>",
            esc(&message)
        );
    }
    body.push_str(
        "</dl><p><a href=\"/search?q=documents\">Browse Documents</a></p><h2>Latest sessions</h2>",
    );
    body.push_str(&results_table(sessions));
    body.push_str(&next_results("sessions", sessions));
    layout("Overview", "", "polylogue", &body)
}
