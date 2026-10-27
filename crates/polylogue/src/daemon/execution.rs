//! Request deadlines and transport cancellation. Only read work is interrupted;
//! a committed write keeps its ordinary result.
use super::{Request, Response};
use polylogue_store::read::limit::{ReadCancellation, ReadLimit};
use serde::{Deserialize, Serialize};
use std::time::Duration;

pub const DEFAULT_READ_TIMEOUT_MS: u64 = 30_000;

pub fn timeout() -> Duration {
    Duration::from_millis(
        std::env::var("POLYLOGUE_READ_TIMEOUT_MS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(DEFAULT_READ_TIMEOUT_MS),
    )
}

/// Absolute Unix milliseconds on the wire; the daemon converts once to a
/// monotonic duration so clock changes cannot extend a running read.
#[derive(Deserialize)]
pub struct Incoming {
    #[serde(flatten)]
    pub request: Request,
    pub deadline_ms: Option<i64>,
}

#[derive(Serialize)]
pub struct Outgoing<'a> {
    #[serde(flatten)]
    pub request: &'a Request,
    pub deadline_ms: i64,
}

pub fn deadline(duration: Duration) -> i64 {
    super::service::now_ms().saturating_add(i64::try_from(duration.as_millis()).unwrap_or(i64::MAX))
}

pub fn remaining(deadline_ms: Option<i64>) -> Duration {
    deadline_ms.map_or_else(timeout, |deadline| {
        Duration::from_millis(deadline.saturating_sub(super::service::now_ms()).max(0) as u64)
    })
}

pub fn limit(deadline_ms: Option<i64>, cancellation: ReadCancellation) -> ReadLimit {
    ReadLimit::new(remaining(deadline_ms), cancellation)
}

pub fn run(
    request: Request,
    limit: &ReadLimit,
    work: impl FnOnce(Request) -> Response,
) -> Response {
    let read_only = request.read_only();
    let (stage, text) = match &request {
        Request::Query { text, .. } => ("query", Some(text.as_str())),
        Request::Explain { expression, .. } => ("explain", expression.as_deref()),
        Request::Context {
            ask: crate::briefing::Ask::Recall { text, .. },
            ..
        } => ("recall", Some(text.as_str())),
        Request::Context { .. } => ("context", None),
        Request::Sql { .. } => ("sql", None),
        Request::Status => ("status", None),
        Request::Complete { .. } => ("completion", None),
        Request::Read { .. } => ("read", None),
        Request::Outline { .. } => ("outline", None),
        Request::Changes { .. } => ("changes", None),
        Request::Attachment { .. } => ("attachment", None),
        Request::Export { .. } => ("export", None),
        _ => ("read", None),
    };
    let info = serde_json::json!({"op": stage, "text": text});
    limit.scope(|| {
        if let Err(error) = limit.check() {
            return refusal(&error, &info, limit);
        }
        let response = work(request);
        let read_refusal = matches!(&response, Response::Error { kind, .. } if matches!(kind.as_str(), "deadline_exceeded" | "cancelled"));
        if (read_only || read_refusal) && let Err(error) = limit.check() {
            return refusal(&error, &info, limit);
        }
        response
    })
}

pub fn refusal(
    error: &polylogue_store::StoreError,
    info: &serde_json::Value,
    limit: &ReadLimit,
) -> Response {
    let kind = match error {
        polylogue_store::StoreError::ReadCancelled => "cancelled",
        _ => "deadline_exceeded",
    };
    let query = info["text"]
        .as_str()
        .or_else(|| info["expression"].as_str());
    let suggestion = query.and_then(narrower);
    Response::Error {
        kind: kind.into(),
        message: error.to_string(),
        detail: Some(serde_json::json!({
            "stage": limit.stage().map(serde_json::Value::from).unwrap_or_else(|| info["op"].clone()),
            "count": null,
            "rows_scanned": null,
            "sqlite_steps": limit.sqlite_steps(),
            "suggestion": suggestion,
            "hint": "restrict the query to a shorter time range or one session",
            "retryable": true,
        })),
    }.masked()
}

fn narrower(query: &str) -> Option<String> {
    use polylogue_query::ast::Source;
    let parsed = polylogue_query::parse(query).ok()?;
    let suggestion = match parsed.source {
        Source::Empty => "since:1d".to_owned(),
        Source::Sessions { cond }
        | Source::Unit {
            cond: Some(cond), ..
        } => {
            let span = cond.span;
            format!(
                "{}({}) since:1d{}",
                &query[..span.start],
                &query[span.start..span.end],
                &query[span.end..]
            )
        }
        Source::Unit { cond: None, .. } => {
            let start = query.len() - query.trim_start().len();
            let end = query[start..]
                .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                .map_or(query.len(), |end| start + end);
            format!("{} where since:1d{}", &query[..end], &query[end..])
        }
    };
    polylogue_query::parse(&suggestion).ok().map(|_| suggestion)
}

/// Timeout follows the operations exit table; cancellation is a refusal.
pub fn exit_code(kind: &str) -> u8 {
    match kind {
        "deadline_exceeded" | "timeout" => 7,
        "cancelled" => 3,
        _ => 1,
    }
}

#[cfg(test)]
mod tests;
