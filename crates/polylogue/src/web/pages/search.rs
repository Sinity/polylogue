use super::*;

pub fn search(query: &str, data: Option<&Value>, cursor: Option<&str>) -> String {
    let mut cli = format!("polylogue {}", shell_quote(query));
    if let Some(cursor) = cursor {
        cli.push_str(&format!(" --cursor {}", shell_quote(cursor)));
    }
    let mut body = format!("<h1>Results for <code>{}</code></h1>", esc(query));
    if let Some(data) = data {
        body.push_str(&results_table(data));
        body.push_str(&next_results(query, data));
    }
    layout(&format!("Search: {query}"), query, &cli, &body)
}
