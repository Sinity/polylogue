//! Page templates over structured masked content. View modules share the
//! layout and this reader; they never reconstruct or decode read cursors.
use super::*;
use crate::web::reader::ReadQuery;

fn content(data: &Value) -> String {
    let slices: Vec<polylogue_render::model_pages::BlockSlice> =
        serde_json::from_value(data["slices"].clone()).expect("shared structured read page");
    let mut out = Vec::new();
    polylogue_render::html::page_slices(&slices, &mut out).expect("UTF-8 HTML");
    String::from_utf8(out).expect("UTF-8 HTML")
}

fn navigation(path: &str, data: &Value, p: &ReadQuery) -> String {
    let mut body = format!(
        "<nav class=\"paging\" aria-label=\"Content pages\"><a href=\"{}\">First</a> ",
        esc(&ReadQuery {
            message: None,
            cursor: None,
            latest: None,
            outline_cursor: None,
            links_cursor: None,
            ..p.clone()
        }
        .url(path, None))
    );
    if path.starts_with("/s/") {
        let _ = write!(
            body,
            "<a href=\"{}&amp;latest=true\">Latest</a> ",
            esc(&ReadQuery {
                message: None,
                outline_cursor: None,
                links_cursor: None,
                ..p.clone()
            }
            .url(path, None))
        );
    }
    if let Some(previous) = data["previous"].as_str() {
        let _ = write!(
            body,
            "<a rel=\"prev\" data-previous href=\"{}\">Previous</a> ",
            esc(&p.url(path, Some(previous)))
        );
    }
    if let Some(next) = data["next"].as_str() {
        let _ = write!(
            body,
            "<a rel=\"next\" data-next href=\"{}\">Next</a>",
            esc(&p.url(path, Some(next)))
        );
    }
    body.push_str("</nav>");
    body
}

pub(crate) fn session(
    id: &str,
    data: &Value,
    p: &ReadQuery,
    outline: &Value,
    links: &Value,
) -> String {
    let path = format!("/s/{}", encode_id(id));
    let title = data["display_title"]
        .as_str()
        .or_else(|| data["session"]["title"]["text"].as_str())
        .unwrap_or(id);
    let mut body = format!(
        "<article class=\"session\"><h1>{}</h1><p class=\"meta\"><code>{}</code></p>",
        esc(title),
        esc(id)
    );
    let session = &data["session"];
    let cwd = session["cwd"]
        .as_array()
        .and_then(|paths| paths.last())
        .and_then(Value::as_str)
        .unwrap_or("Working directory unknown");
    let _ = write!(
        body,
        "<p class=\"meta\">{} · {} · {} · {}</p><nav class=\"views\" aria-label=\"Read view\">",
        esc(cwd),
        esc(session["repo"].as_str().unwrap_or("Repository unknown")),
        esc(session["started"].as_str().unwrap_or("Time unknown")),
        esc(session["tool_version"]
            .as_str()
            .unwrap_or("Tool version unknown"))
    );
    for view in polylogue_model::read::View::NAMES {
        let options = ReadQuery {
            view: Some((*view).into()),
            ..p.clone()
        };
        let _ = write!(
            body,
            "<a href=\"{}\"{}>{}</a> ",
            esc(&options.url(&path, None)),
            if *view == p.view() {
                " aria-current=\"page\""
            } else {
                ""
            },
            view
        );
    }
    let options = ReadQuery {
        with_parents: Some(!p.parents()),
        message: None,
        latest: None,
        outline_cursor: None,
        links_cursor: None,
        ..p.clone()
    };
    let _ = write!(
        body,
        "<a href=\"{}\">{}</a></nav>",
        esc(&options.url(&path, None)),
        if p.parents() {
            "Exclude parents"
        } else {
            "Include parents"
        }
    );
    let scope = format!("sessions where id:{}", polylogue_query::quoted(id));
    let _ = write!(
        body,
        "<p><a href=\"/cost?q={}\">Session cost and usage</a></p>",
        encode_id(&scope)
    );
    if let (Some(at), Some(total)) = (
        data["position"]["message"].as_u64(),
        data["position"]["total_messages"].as_u64(),
    ) {
        let _ = write!(
            body,
            "<p class=\"meta\">Source message {} of {total}</p>",
            if total == 0 { 0 } else { at + 1 }
        );
    }
    body.push_str(&navigation(&path, data, p));
    if outline.is_null() {
        let options = ReadQuery {
            outline: Some(true),
            ..p.clone()
        };
        let _ = write!(
            body,
            "<p><a href=\"{}\">Open outline</a></p>",
            esc(&options.url(&path, p.cursor.as_deref()))
        );
    } else {
        body.push_str("<details class=\"outline\" open><summary>Outline</summary>");
        if outline["error"].is_string() {
            let _ = write!(
                body,
                "<p role=\"status\">Outline unavailable: {}</p>",
                esc(outline["message"].as_str().unwrap_or("unknown"))
            );
        } else {
            body.push_str("<ol>");
            for item in outline["entries"].as_array().into_iter().flatten() {
                let options = ReadQuery {
                    message: item["message"].as_str().map(str::to_owned),
                    ..p.clone()
                };
                let label = item["kinds"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_str)
                    .collect::<Vec<_>>()
                    .join(", ");
                let _ = write!(
                    body,
                    "<li><a href=\"{}\">{} <code>{}</code></a></li>",
                    esc(&options.url(&path, None)),
                    esc(&label),
                    esc(item["message"].as_str().unwrap_or("unknown"))
                );
            }
            body.push_str("</ol>");
            if let Some(next) = outline["next"].as_str() {
                let options = ReadQuery {
                    outline_cursor: Some(next.into()),
                    ..p.clone()
                };
                let _ = write!(
                    body,
                    "<a href=\"{}\">Next outline page</a>",
                    esc(&options.url(&path, p.cursor.as_deref()))
                );
            }
        }
        body.push_str("</details>");
    }
    body.push_str(&parents(&data["parents"], links));
    if let Some(next) = links["next"].as_str() {
        let options = ReadQuery {
            links_cursor: Some(next.into()),
            ..p.clone()
        };
        let _ = write!(
            body,
            "<a href=\"{}\">Next relationships page</a>",
            esc(&options.url(&path, p.cursor.as_deref()))
        );
    }
    body.push_str(&content(data));
    body.push_str(&navigation(&path, data, p));
    body.push_str("</article>");
    layout(title, "", &p.cli(id, p.parents()), &body)
}

pub(crate) fn document(id: &str, data: &Value, p: &ReadQuery) -> String {
    let document = &data["document"];
    let title = document["title"].as_str().unwrap_or(id);
    let path = format!("/d/{}", encode_id(id));
    let mut body = format!(
        "<h1>{}</h1><p class=\"meta\">{} · authorship {}</p><nav aria-label=\"Document view\">",
        esc(title),
        esc(document["purpose"].as_str().unwrap_or("document")),
        esc(document["authorship"].as_str().unwrap_or("unknown"))
    );
    for view in ["full", "dialogue"] {
        let options = ReadQuery {
            view: Some(view.into()),
            ..p.clone()
        };
        let _ = write!(
            body,
            "<a href=\"{}\">{view}</a> ",
            esc(&options.url(&path, None))
        );
    }
    body.push_str("</nav>");
    body.push_str(&navigation(&path, data, p));
    body.push_str(&content(data));
    body.push_str(&navigation(&path, data, p));
    for (label, value) in [
        ("Source record time", &document["source_time"]),
        ("Source metadata", &document["metadata"]),
        ("Source records", &data["sources"]),
    ] {
        let _ = write!(
            body,
            "<details><summary>{label}</summary><pre>{}</pre></details>",
            esc(&serde_json::to_string_pretty(value).expect("JSON"))
        );
    }
    layout(title, "", &p.cli(&format!("document:{id}"), false), &body)
}
