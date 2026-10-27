use super::*;

pub fn search(
    query: &str,
    data: Option<&Value>,
    error: Option<&Value>,
    cursor: Option<&str>,
) -> String {
    let mut cli = format!("polylogue list {}", shell_quote(query));
    if let Some(cursor) = cursor {
        cli.push_str(&format!(" --cursor {}", shell_quote(cursor)));
    }
    let mut body = format!(
        "<link rel=\"stylesheet\" href=\"/assets/search.css\"><script type=\"module\" src=\"/assets/search.js\"></script><h1>Results for <code>{}</code></h1>",
        esc(query)
    );
    match crate::web::search::fields(query) {
        Ok(fields) => {
            let unit = polylogue_query::parse(query)
                .ok()
                .map(|q| match q.source {
                    polylogue_query::ast::Source::Unit { unit, .. } => unit.name(),
                    _ => "sessions",
                })
                .unwrap_or("sessions");
            let _ = write!(
                body,
                "<p>Selected unit: {unit}. Filters apply before ranking.</p><form action=\"/search\" method=\"get\" aria-label=\"Add a facet\"><input type=\"hidden\" name=\"q\" value=\"{}\"><label for=\"facet-field\">Facet</label><select id=\"facet-field\" name=\"field\">",
                esc(query)
            );
            for field in fields {
                let _ = write!(
                    body,
                    "<option value=\"{}\">{}: {}</option>",
                    esc(field.name),
                    esc(field.name),
                    esc(field.help)
                );
            }
            body.push_str("</select> <label for=\"facet-value\">Value</label><input id=\"facet-value\" name=\"value\" list=\"facet-suggestions\" required autocomplete=\"off\"><datalist id=\"facet-suggestions\"></datalist><button>Add filter</button><button formaction=\"/search/facets\" formnovalidate>Count full matching population</button></form><p id=\"completion-status\" role=\"status\">Completion values are suggestions, not counts.</p>");
        }
        Err(message) => {
            let _ = write!(body, "<p>{}</p>", esc(&message));
        }
    }
    if let Some(e) = error {
        let _ = write!(
            body,
            "<div class=\"error\" role=\"alert\"><p>{}: {}</p>",
            esc(e["kind"].as_str().unwrap_or("query")),
            esc(e["message"].as_str().unwrap_or("the query failed"))
        );
        if let Some(fix) = e["detail"]["suggestion"].as_str() {
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
        if let Some(rows) = data["rows"].as_array() {
            let count = rows.len();
            let _ = write!(
                body,
                "<p>{count} rows on this page. Continue to reach every match.</p>"
            );
        }
        body.push_str(&results_table(data));
        body.push_str(
            "<button type=\"button\" data-previous hidden>Previous visited page</button>",
        );
        body.push_str(&next_results(query, data));
        let keep_disabled = if data["rows"].is_array() {
            ""
        } else {
            " disabled"
        };
        let _ = write!(
            body,
            r#"<section aria-label="Save search"><h2>Keep this work</h2><p>A saved query runs current data. A kept result records all rows when applied.</p><form id="save-search" hidden data-query="{}"><label for="search-name">Name</label><input id="search-name" name="name" required><button name="operation" value="save" type="submit">Save query</button><button name="operation" value="keep" type="submit" formnovalidate{keep_disabled}>Keep results</button></form><p id="save-status" role="status" aria-live="polite"></p><noscript><p>Use copy as CLI, then <code>polylogue save NAME QUERY</code> or <code>polylogue keep QUERY</code>.</p></noscript></section>"#,
            esc(query)
        );
    }
    if data.is_some_and(|data| !data["rows"].is_array()) {
        body.push_str("<p>Keep results applies to table rows. This answer is a report.</p>");
    }
    layout(&format!("Search: {query}"), query, &cli, &body)
}

pub fn facet_counts(query: &str, field: &str, data: &Value) -> String {
    let mut body = format!(
        "<h1>{} counts</h1><p>Counts cover the entire matching population of <code>{}</code>. Choices continue across pages.</p><p><a href=\"/search?q={}\">Back to results</a></p>",
        esc(field),
        esc(query),
        encode_id(query)
    );
    body.push_str(&results_table(data));
    if let Some(next) = data["next"].as_str() {
        let _ = write!(
            body,
            "<a data-next rel=\"next\" href=\"/search/facets?q={}&amp;field={}&amp;cursor={}\">Next choices</a>",
            encode_id(query),
            encode_id(field),
            encode_id(next)
        );
    }
    layout("Facet counts", query, "polylogue list", &body)
}
