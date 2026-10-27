//! Search and facet requests use the daemon's query operation.
use super::*;
use polylogue_query::ast::{Query as Parsed, Source, StageKind, Unit};
use polylogue_query::fields::{FIELDS, Field};

#[derive(Default, Deserialize)]
pub(super) struct Search {
    #[serde(default)]
    q: String,
    cursor: Option<String>,
    field: Option<String>,
    value: Option<String>,
}

pub(super) fn query(text: String, cursor: Option<String>) -> Request {
    Request::Query {
        text,
        cursor,
        page_size: Some(50),
        why: true,
        session: None,
        fields: vec![],
    }
}

fn editable(text: &str) -> Result<(Parsed, Unit), String> {
    let parsed = polylogue_query::parse(text).map_err(|e| e.to_string())?;
    if !parsed.attach.is_empty()
        || parsed
            .stages
            .iter()
            .any(|s| !matches!(s.kind, StageKind::SortBy { .. }))
    {
        return Err("Edit this query directly. Facets are unavailable for attached, combined, scoped, or transformed results.".into());
    }
    let unit = match parsed.source {
        Source::Unit { unit, .. } => unit,
        _ => Unit::Sessions,
    };
    Ok((parsed, unit))
}

pub(super) fn fields(text: &str) -> Result<Vec<&'static Field>, String> {
    let (_, unit) = editable(text)?;
    let names = [
        "origin", "repo", "harness", "model", "since", "until", "live", "tag",
    ];
    Ok(FIELDS
        .iter()
        .filter(|f| f.unit == unit && names.contains(&f.name))
        .collect())
}

/// Replace only the parser's source-condition span; quoted pipes and nested
/// conditions remain byte-for-byte intact, including semantic ranking terms.
pub(super) fn filtered(text: &str, field: &str, value: &str) -> Result<String, String> {
    let (parsed, unit) = editable(text)?;
    if !fields(text)?.iter().any(|f| f.name == field) {
        return Err(format!(
            "Facet {field:?} is unavailable for {}",
            unit.name()
        ));
    }
    let literal = format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""));
    let condition = format!("{field}:{literal}");
    let cond = match &parsed.source {
        Source::Unit { cond, .. } => cond.as_ref(),
        Source::Sessions { cond } => Some(cond),
        Source::Empty => None,
    };
    let result = if let Some(cond) = cond {
        format!(
            "{}({}) AND {condition}{}",
            &text[..cond.span.start],
            &text[cond.span.start..cond.span.end],
            &text[cond.span.end..]
        )
    } else {
        let start = text.len() - text.trim_start().len();
        let end = start
            + text[start..]
                .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                .unwrap_or(text.len() - start);
        let source = if text.trim().is_empty() {
            unit.name()
        } else {
            &text[..end]
        };
        format!("{source} where {condition}{}", &text[end..])
    };
    polylogue_query::parse(&result).map_err(|e| e.to_string())?;
    Ok(result)
}

pub(super) fn grouped(text: &str, field: &str) -> Result<String, String> {
    let (parsed, _) = editable(text)?;
    if !fields(text)?
        .iter()
        .any(|f| f.name == field && f.sql.is_some())
    {
        return Err("Counts are unavailable for this facet. Suggestions are not counts.".into());
    }
    let end = parsed
        .stages
        .first()
        .and_then(|s| text[..s.span.start].rfind('|'))
        .unwrap_or(text.len());
    Ok(format!("{} | group by {field} | count", text[..end].trim()))
}

pub(super) async fn search(State(web): State<Arc<Web>>, Query(p): Query<Search>) -> HttpResponse {
    let text = if p.q.trim().is_empty() {
        "sessions".into()
    } else {
        p.q
    };
    if let (Some(field), Some(value)) = (p.field.as_deref(), p.value.as_deref()) {
        return match filtered(&text, field, value) {
            Ok(text) => Redirect::to(&format!(
                "/search?q={}",
                polylogue_render::html::encode_path_segment(&text)
            ))
            .into_response(),
            Err(message) => page(pages::search::search(
                &text,
                None,
                Some(&serde_json::json!({"kind":"facet", "message":message})),
                None,
            )),
        };
    }
    match ask(&web, query(text.clone(), p.cursor.clone())).await {
        Response::Ok { data } => page(pages::search::search(
            &text,
            Some(&scrubbed(&web, data)),
            None,
            p.cursor.as_deref(),
        )),
        Response::Error {
            kind,
            message,
            detail,
        } => page(pages::search::search(
            &text,
            None,
            Some(&scrubbed(
                &web,
                serde_json::json!({"kind":kind,"message":message,"detail":detail}),
            )),
            p.cursor.as_deref(),
        )),
    }
}

pub(super) async fn facets(State(web): State<Arc<Web>>, Query(p): Query<Search>) -> HttpResponse {
    let field = p.field.as_deref().unwrap_or("origin");
    match grouped(&p.q, field) {
        Ok(text) => match ask(&web, query(text, p.cursor)).await {
            Response::Ok { data } => page(pages::search::facet_counts(
                &p.q,
                field,
                &scrubbed(&web, data),
            )),
            error => json_answer(&web, error),
        },
        Err(message) => failed(StatusCode::BAD_REQUEST, "facet", &message),
    }
}

pub(super) async fn script() -> HttpResponse {
    (
        [(header::CONTENT_TYPE, "text/javascript; charset=utf-8")],
        include_str!("assets/search.js"),
    )
        .into_response()
}

pub(super) async fn style() -> HttpResponse {
    (
        [(header::CONTENT_TYPE, "text/css; charset=utf-8")],
        include_str!("assets/search.css"),
    )
        .into_response()
}

#[cfg(test)]
mod tests;
