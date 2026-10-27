//! Cost views share the daemon's usage report and response population.
use super::*;
use polylogue_render::html::{encode_path_segment as encode, escape as esc};
use serde_json::Value;
use std::fmt::Write as _;

#[derive(Default, Deserialize)]
pub struct Options {
    pub month: Option<String>,
    #[serde(default)]
    pub undated: bool,
    #[serde(default)]
    pub unknown_model: bool,
    #[serde(default)]
    pub q: String,
    pub date: Option<String>,
    pub model: Option<String>,
    pub cursor: Option<String>,
}

pub async fn cost(State(web): State<Arc<Web>>, Query(mut options): Query<Options>) -> HttpResponse {
    options.date = options.date.filter(|s| !s.is_empty());
    options.model = options.model.filter(|s| !s.is_empty());
    let scope = if options.q.trim().is_empty() {
        "sessions"
    } else {
        options.q.trim()
    };
    match ask(
        &web,
        Request::Cost {
            month: options.month.clone(),
            undated: options.undated,
            unknown_model: options.unknown_model,
            query: scope.into(),
            date: options.date.clone(),
            model: options.model.clone(),
            cursor: options.cursor.clone(),
            page_size: None,
        },
    )
    .await
    {
        Response::Ok { data } => page(render(
            &access::scrub(&polylogue_render::mask(scope), &web.daemon.archive.root),
            &options,
            &scrubbed(&web, data),
        )),
        Response::Error { kind, message, .. } => failed(
            if matches!(kind.as_str(), "store" | "internal") {
                StatusCode::INTERNAL_SERVER_ERROR
            } else {
                StatusCode::BAD_REQUEST
            },
            &kind,
            &access::scrub(&message, &web.daemon.archive.root),
        ),
    }
}

pub fn render(scope: &str, options: &Options, data: &Value) -> String {
    let mut body = if data["usage"]["error"].is_string() {
        format!(
            "<h1>Cost of selected sessions</h1><p><a href=\"/search?q={}\">Browse selected sessions</a></p>",
            encode(scope)
        )
    } else {
        summary(scope, &data["usage"]["data"])
    };
    let _ = write!(
        body,
        "<p><a href=\"/cost?q={}\">All source record times</a> <a href=\"/cost?q={}&amp;undated=true\">Usage with unknown source record times</a></p>",
        encode(scope),
        encode(scope)
    );
    if options.undated {
        body.push_str(
            "<p>Only source records with unknown or unrepresentable times are selected.</p>",
        );
    }
    let date = options.date.as_deref().unwrap_or("");
    let _ = write!(
        body,
        "<form action=\"/cost\" method=\"get\"><input type=\"hidden\" name=\"q\" value=\"{}\"><label for=\"period\">Source record logical day (optional)</label> <input type=\"date\" id=\"period\" name=\"date\" value=\"{}\"><button>Apply period</button></form>",
        esc(scope),
        esc(date)
    );
    if !date.is_empty() {
        let _ = write!(
            body,
            "<p>Source record period: {} in {} ({} to {} Unix milliseconds, end excluded). Recorded response times do not prove payment time. <a href=\"/day/{}\">Open day activity</a></p>",
            esc(date),
            esc(data["zone"].as_str().unwrap_or("system zone")),
            data["since"],
            data["until"],
            encode(date)
        );
    }
    if let Some(error) = data["usage"]["error"].as_str() {
        let _ = write!(
            body,
            "<p role=\"status\">Cost unavailable: {}: {}</p>",
            esc(error),
            esc(data["usage"]["message"].as_str().unwrap_or(""))
        );
    }
    body.push_str("<table class=\"results\"><caption>By model. These rows and the total cover the same selected sessions and source record period. Open a row for its known physical response records.</caption><thead><tr><th scope=\"col\">Model</th><th scope=\"col\">Known response records</th><th scope=\"col\">Known API-equivalent USD</th><th scope=\"col\">Unpriced records</th></tr></thead><tbody>");
    for row in data["usage"]["data"]["by_model"]
        .as_array()
        .into_iter()
        .flatten()
    {
        let model = row["model"].as_str();
        let shown = model.unwrap_or("Unknown model");
        let amount = row["api_cost"]["known_usd"]
            .as_f64()
            .map_or_else(|| "Unknown".into(), |n| format!("${n:.6}"));
        let link = model
            .map(|m| {
                format!(
                    "<a href=\"/cost?q={}&amp;date={}&amp;undated={}&amp;model={}\">{}</a>",
                    encode(scope),
                    encode(date),
                    options.undated,
                    encode(m),
                    esc(shown)
                )
            })
            .unwrap_or_else(||format!("<a href=\"/cost?q={}&amp;date={}&amp;undated={}&amp;unknown_model=true\">Unknown model</a>",encode(scope),encode(date),options.undated));
        let _ = write!(
            body,
            "<tr><th scope=\"row\">{link}</th><td>{}</td><td>{amount}</td><td>{}</td></tr>",
            row["messages"], row["api_cost"]["unpriced_messages"]
        );
    }
    body.push_str("</tbody></table>");
    if !options.undated {
        body.push_str(&daily(scope, options, data));
    }
    body.push_str("<h2>Physical response details</h2><p>Unknown response boundaries and independent usage observations are excluded from this list and counted separately in the shared usage report. Totals include every selected model.</p>");
    if options.unknown_model {
        body.push_str("<p>Response details without a recorded model name.</p>");
    }
    if let Some(model) = &options.model {
        let _ = write!(body, "<p>Response details for model: {}</p>", esc(model));
    }
    body.push_str(&super::timeline::activity(&data["responses"]));
    if let Some(next) = data["responses"]["next"].as_str() {
        let _ = write!(
            body,
            "<nav aria-label=\"Response pages\"><a rel=\"next\" data-next href=\"/cost?q={}&amp;date={}&amp;model={}&amp;undated={}&amp;unknown_model={}&amp;cursor={}\">Next response page</a></nav>",
            encode(scope),
            encode(date),
            encode(&polylogue_render::mask(
                options.model.as_deref().unwrap_or("")
            )),
            options.undated,
            options.unknown_model,
            encode(next)
        );
    }
    pages::layout(
        "Cost and usage",
        "",
        &format!(
            "polylogue cost --query {}{}",
            polylogue_render::shell_quote(scope),
            options
                .date
                .as_ref()
                .map(|d| format!(" --date {}", polylogue_render::shell_quote(d)))
                .unwrap_or_default()
        ),
        &body,
    )
}

fn daily(scope: &str, options: &Options, data: &Value) -> String {
    let mut body = format!(
        "<h2>Response costs by day</h2><p>Displayed month: {}. Physical own responses only; independent observations stay in the usage report. Days without rows have no known physical response records in this selection.</p><nav aria-label=\"Cost months\">",
        esc(data["month"].as_str().unwrap_or(""))
    );
    for (key, label) in [
        ("previous_month", "Previous cost month"),
        ("following_month", "Next cost month"),
    ] {
        if let Some(month) = data[key].as_str() {
            let _ = write!(
                body,
                " <a href=\"/cost?q={}&amp;date={}&amp;month={}\">{label}</a>",
                encode(scope),
                encode(options.date.as_deref().unwrap_or("")),
                encode(month)
            );
        }
    }
    body.push_str("</nav><table class=\"results\"><caption>Known API-equivalent physical response subtotals in the displayed month and selected source record period. Unpriced records are excluded from known USD.</caption><thead><tr><th scope=\"col\">Logical day</th><th scope=\"col\">Response records</th><th scope=\"col\">Known USD</th><th scope=\"col\">Unpriced records</th></tr></thead><tbody>");
    for row in data["days"].as_array().into_iter().flatten() {
        let date = row["date"].as_str().unwrap_or("");
        let amount = row["known_usd"]
            .as_f64()
            .map_or_else(|| "Unknown".into(), |n| format!("${n:.6}"));
        let _ = write!(
            body,
            "<tr><th scope=\"row\"><a href=\"/cost?q={}&amp;date={}\">{}</a></th><td>{}</td><td>{amount}</td><td>{}</td></tr>",
            encode(scope),
            encode(date),
            esc(date),
            row["records"],
            row["unpriced_records"]
        );
    }
    body.push_str("</tbody></table>");
    body
}

pub(crate) fn summary(scope: &str, data: &Value) -> String {
    let report = polylogue_render::measures::usage(data);
    let count = |key: &str| {
        data[key]
            .as_u64()
            .map_or_else(|| "unknown".to_owned(), |n| n.to_string())
    };
    format!(
        "<h1>Cost of selected sessions</h1><form action=\"/cost\" method=\"get\"><label for=\"scope\">Session query</label> <input id=\"scope\" name=\"q\" value=\"{}\"> <button>Measure</button></form>\
         <p>Session queries select sessions, not a response period. Without a source record day filter, costs include responses outside any dates used to select sessions. Parent-inclusive usage is separate and is never added to own totals.</p><p>Physical response records at source record time. API-equivalent figures use catalog list prices; tool-reported figures remain separate. Known sums exclude unreported usage and unpriced responses shown below.</p>\
         <p>{} known response records; {} model items excluded because response boundaries are unknown. Independent usage observations are reported separately below.</p><pre>{}</pre><p><a href=\"/search?q={}\">Browse selected sessions</a></p>",
        esc(scope),
        count("known_response_messages"),
        count("unknown_boundary_items"),
        esc(&report),
        encode(scope)
    )
}

#[cfg(test)]
mod tests;
