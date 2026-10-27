//! Timeline and day pages use the shared calendar operation.
use super::*;
use polylogue_render::html::{encode_path_segment as encode, escape as esc};
use serde_json::Value;
use std::fmt::Write as _;

#[derive(Default, Deserialize)]
pub struct Options {
    pub date: Option<String>,
    pub cursor: Option<String>,
    #[serde(default)]
    pub across_years: bool,
    #[serde(default)]
    pub undated: bool,
}

pub async fn timeline(State(web): State<Arc<Web>>, Query(options): Query<Options>) -> HttpResponse {
    show(web, options).await
}
pub async fn day(
    State(web): State<Arc<Web>>,
    UrlPath(date): UrlPath<String>,
    Query(mut options): Query<Options>,
) -> HttpResponse {
    options.date = Some(date);
    show(web, options).await
}
async fn show(web: Arc<Web>, options: Options) -> HttpResponse {
    match ask(
        &web,
        Request::Timeline {
            date: options.date,
            across_years: options.across_years,
            undated: options.undated,
            cursor: options.cursor,
            page_size: None,
        },
    )
    .await
    {
        Response::Ok { data } => page(render(&scrubbed(&web, data))),
        Response::Error { kind, message, .. } => failed(
            if kind == "bad_request" {
                StatusCode::BAD_REQUEST
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            },
            &kind,
            &access::scrub(&message, &web.daemon.archive.root),
        ),
    }
}

pub fn activity(data: &Value) -> String {
    let mut out = String::from(
        "<table class=\"results\"><caption>Recorded activity. Each time source is named.</caption><thead><tr><th scope=\"col\">Logical day</th><th scope=\"col\">Time</th><th scope=\"col\">Time source</th><th scope=\"col\">Record</th></tr></thead>",
    );
    let mut previous_day = None;
    for row in data["rows"].as_array().into_iter().flatten() {
        let id = row["id"].as_str().unwrap_or("");
        let href = match row["kind"].as_str() {
            Some("message") => format!("/m/{}", encode(id)),
            Some("note") => format!("/n/{}", encode(id)),
            _ => row["session"]
                .as_str()
                .map(|id| format!("/s/{}", encode(id)))
                .unwrap_or_else(|| {
                    format!(
                        "/search?q={}",
                        encode(&format!(
                            "events where id = {}",
                            serde_json::to_string(id).unwrap()
                        ))
                    )
                }),
        };
        let time = row["at"]
            .as_i64()
            .and_then(polylogue_model::Timestamp::from_unix_ms)
            .map_or_else(|| "Time unknown".into(), |t| t.to_string());
        let date = row["date"].as_str().unwrap_or("Undated");
        if previous_day != Some(date) {
            if previous_day.is_some() {
                out.push_str("</tbody>");
            }
            out.push_str("<tbody>");
            let _ = write!(
                out,
                "<tr><th scope=\"rowgroup\" colspan=\"4\">{}</th></tr>",
                esc(date)
            );
            previous_day = Some(date);
        }
        let _ = write!(
            out,
            "<tr class=\"hit\" tabindex=\"-1\"><th scope=\"row\">{}</th><td>{}</td><td>{}</td><td><a href=\"{}\">{}: {}</a> <code>{}</code></td></tr>",
            esc(date),
            esc(&time),
            esc(row["time_source"].as_str().unwrap_or("Unknown")),
            esc(&href),
            esc(row["kind"].as_str().unwrap_or("record")),
            esc(row["label"].as_str().unwrap_or("")),
            esc(id)
        );
    }
    if previous_day.is_some() {
        out.push_str("</tbody>");
    }
    out.push_str("</table>");
    if data["rows"].as_array().is_some_and(Vec::is_empty) {
        out.push_str("<p>No recorded activity in this selection.</p>");
    }
    out
}

pub fn render(data: &Value) -> String {
    let date = data["date"].as_str().unwrap_or("");
    let across = data["across_years"].as_bool().unwrap_or(false);
    let undated = data["undated"].as_bool().unwrap_or(false);
    let minutes = data["day_start_minutes"].as_u64().unwrap_or(240);
    let title = if undated {
        "Undated records".into()
    } else if across {
        format!("This day across years: {date}")
    } else {
        format!("Timeline: {date}")
    };
    let mut body = format!(
        "<h1>{}</h1><p>Logical day: {}, starting {:02}:{:02}. Change shared settings with <code>polylogue setting day.zone</code> and <code>polylogue setting day.logical_start</code>.</p><form action=\"/timeline\" method=\"get\"><label for=\"date\">Logical date</label> <input type=\"date\" id=\"date\" name=\"date\" value=\"{}\"><button>Open day</button></form>",
        esc(&title),
        esc(data["zone"].as_str().unwrap_or("Unknown zone")),
        minutes / 60,
        minutes % 60,
        esc(date)
    );
    body.push_str("<nav aria-label=\"Date navigation\">");
    for (field, label) in [
        ("previous", "Previous day"),
        ("following", "Next day"),
        ("previous_month", "Previous month"),
        ("following_month", "Next month"),
    ] {
        if let Some(date) = data[field].as_str() {
            let _ = write!(body, " <a href=\"/day/{}\">{label}</a>", encode(date));
        }
    }
    let _ = write!(
        body,
        " <a href=\"/timeline?date={}&amp;across_years=true\">This day across years</a> <a href=\"/timeline?date={}&amp;undated=true\">Undated records</a></nav>",
        encode(date),
        encode(date)
    );
    if date.ends_with("02-29") {
        body.push_str("<p>February 29 includes only leap years. <a href=\"/timeline?across_years=true&amp;date=");
        body.push_str(&encode(&date.replace("02-29", "02-28")));
        body.push_str("\">Use February 28</a>.</p>");
    }
    body.push_str("<nav aria-label=\"Calendar strip: messages, events and note revisions\"><ol class=\"calendar\">");
    for row in data["calendar"].as_array().into_iter().flatten() {
        let d = row["date"].as_str().unwrap_or("");
        let _ = write!(
            body,
            "<li><a href=\"/day/{}\">{}: {} records</a></li>",
            encode(d),
            esc(d),
            row["records"]
        );
    }
    body.push_str("</ol></nav>");
    body.push_str("<details><summary>Calendar counts</summary><table class=\"results\"><caption>Messages, events and note revisions in the selected month. Days absent from this table have zero records. Each date opens its activity.</caption><thead><tr><th scope=\"col\">Logical day</th><th scope=\"col\">Records</th></tr></thead><tbody>");
    for row in data["calendar"].as_array().into_iter().flatten() {
        let d = row["date"].as_str().unwrap_or("");
        let _ = write!(
            body,
            "<tr><th scope=\"row\"><a href=\"/day/{}\">{}</a></th><td>{}</td></tr>",
            encode(d),
            esc(d),
            row["records"]
        );
    }
    body.push_str("</tbody></table></details>");
    body.push_str(&activity(&data["activity"]));
    if let Some(next) = data["activity"]["next"].as_str() {
        let _ = write!(
            body,
            "<nav aria-label=\"Activity pages\"><a rel=\"next\" data-next href=\"/timeline?date={}&amp;across_years={across}&amp;undated={undated}&amp;cursor={}\">Next activity page</a></nav>",
            encode(date),
            encode(next)
        );
    }
    if !data["usage"].is_null() {
        body.push_str("<section><h2>Usage during this logical day</h2><p>Own physical responses and independent observations at source record time, in the same interval as this activity. Unknown times are excluded from the interval and remain under Undated records. API-equivalent prices are estimates, not charges.</p>");
        if data["usage"]["error"].is_string() {
            let _ = write!(
                body,
                "<p role=\"status\">Cost unavailable: {}: {}</p>",
                esc(data["usage"]["error"].as_str().unwrap_or("")),
                esc(data["usage"]["message"].as_str().unwrap_or(""))
            );
        } else {
            let _ = write!(
                body,
                "<pre>{}</pre><a href=\"/cost?date={}\">Day cost and response details</a>",
                esc(&polylogue_render::measures::usage(&data["usage"]["data"])),
                encode(date)
            );
        }
        body.push_str("</section>");
    }
    pages::layout_with_styles(
        "/assets/timeline.css",
        &title,
        "",
        &format!(
            "polylogue timeline --date {}{}{}",
            polylogue_render::shell_quote(date),
            if across { " --across-years" } else { "" },
            if undated { " --undated" } else { "" }
        ),
        &body,
    )
}

#[cfg(test)]
mod tests;
