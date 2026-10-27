use super::*;

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
