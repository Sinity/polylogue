use super::*;

pub fn search(
    query: &str,
    data: Option<&Value>,
    error: Option<&command_surface::Failure>,
    cursor: Option<&str>,
) -> String {
    let mut cli = format!("polylogue list {}", shell_quote(query));
    if let Some(cursor) = cursor {
        cli.push_str(&format!(" --cursor {}", shell_quote(cursor)));
    }
    let mut body = format!("<h1>Results for <code>{}</code></h1>", esc(query));
    if let Some(e) = error {
        let _ = write!(
            body,
            "<div class=\"error\" role=\"alert\"><p>{}</p>",
            esc(&e.message)
        );
        if let Some(fix) = crate::protocol::suggestion_text(e).as_deref() {
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
        body.push_str(&next_results(query, data));
    }
    layout(&format!("Search: {query}"), query, &cli, &body)
}
