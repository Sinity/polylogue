//! The routes the extension calls through its native messaging host. Each
//! native request is served in process by this router; nothing listens on
//! a port. A refusal answers `{"error": code, "message": text}` with an
//! HTTP status.

use super::BrowserHost;
use super::actions::Update;
use crate::daemon::{Request, Response};
use axum::Router;
use axum::body::Bytes;
use axum::extract::{Path as UrlPath, Query, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Json, Response as HttpResponse};
use axum::routing::{get, post};
use polylogue_store::backfill_jobs::BackfillReport;
use serde::Deserialize;
use serde_json::{Value, json};
use std::sync::Arc;

pub fn router(host: Arc<BrowserHost>) -> Router {
    Router::new()
        .route("/v1/status", get(status))
        .route("/v1/reader", get(reader))
        .route("/v1/captures", post(store_capture))
        .route("/v1/captures/files", post(capture_files))
        .route("/v1/conversations", post(conversations))
        .route("/v1/notes", post(selection_note))
        .route("/v1/chat-actions", get(chat_actions))
        .route("/v1/chat-actions/{id}/report", post(chat_report))
        .route("/v1/chat-actions/{id}/decision", post(chat_decision))
        .route(
            "/v1/chat-actions/{id}/attachments/{attachment}",
            get(chat_attachment),
        )
        .route("/v1/backfill", get(backfill_jobs))
        .route("/v1/backfill/claim", post(backfill_claim))
        .route("/v1/backfill/{id}/report", post(backfill_report))
        // A capture or a note is as large as the conversation it holds.
        .layer(axum::extract::DefaultBodyLimit::disable())
        .with_state(host)
}

fn refuse(status: StatusCode, error: &str, message: &str) -> HttpResponse {
    (status, Json(json!({ "error": error, "message": message }))).into_response()
}

/// Runs blocking work (files, the socket) off the async thread.
async fn blocking<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> Option<T> {
    tokio::task::spawn_blocking(f).await.ok()
}

/// Asks the daemon and answers with its data, or its refusal's status and
/// code; 503 `daemon_required` when it does not answer (stopped, or too busy).
async fn ask(host: Arc<BrowserHost>, request: Request) -> Result<Value, Box<HttpResponse>> {
    match blocking(move || host.ask(&request)).await.flatten() {
        Some(Response::Ok { data, .. }) => Ok(data),
        Some(Response::Error {
            kind,
            message,
            detail,
        }) => {
            let status = detail
                .as_ref()
                .and_then(|d| d["http_status"].as_u64())
                .and_then(|s| StatusCode::from_u16(u16::try_from(s).ok()?).ok())
                .unwrap_or(match kind.as_str() {
                    "bad_request" | "invalid_query" => StatusCode::BAD_REQUEST,
                    "conflict" => StatusCode::CONFLICT,
                    "not_found" => StatusCode::NOT_FOUND,
                    _ => StatusCode::INTERNAL_SERVER_ERROR,
                });
            Err(Box::new(refuse(status, &kind, &message)))
        }
        None => Err(Box::new(refuse(
            StatusCode::SERVICE_UNAVAILABLE,
            "daemon_required",
            "the Polylogue daemon did not answer; it is stopped or busy",
        ))),
    }
}

async fn answer(host: Arc<BrowserHost>, request: Request) -> HttpResponse {
    match ask(host, request).await {
        Ok(data) => Json(data).into_response(),
        Err(refusal) => *refusal,
    }
}

/// Whether the daemon is serving and chat actions are on. A setting read,
/// not `status`, which counts the whole archive.
async fn status(State(host): State<Arc<BrowserHost>>) -> HttpResponse {
    let setting = blocking(move || host.actions_setting()).await.flatten();
    Json(json!({
        "daemon": if setting.is_some() { "running" } else { "no_answer" },
        "chat_actions": setting == Some(true),
        "version": env!("CARGO_PKG_VERSION"),
    }))
    .into_response()
}

/// The address the running reader reports.
async fn reader(State(host): State<Arc<BrowserHost>>) -> HttpResponse {
    let dir = crate::runtime_dir::runtime_dir(&host.archive_key);
    match blocking(move || crate::web::access::reader_url_in(&dir)).await {
        Some(Ok(url)) => Json(json!({ "url": url })).into_response(),
        _ => refuse(
            StatusCode::SERVICE_UNAVAILABLE,
            "reader_unavailable",
            "no reader is running; start it with `polylogue web`",
        ),
    }
}

/// The conversation a capture names.
fn capture_identity(v: &Value) -> Option<(String, String)> {
    let provider = v.pointer("/session/provider")?.as_str()?.trim();
    let session = v.pointer("/session/provider_session_id")?.as_str()?.trim();
    (!provider.is_empty() && !session.is_empty()).then(|| (provider.to_owned(), session.to_owned()))
}

/// Appends one capture to its conversation's capture log. Bytes the log
/// already holds are not appended again. The answer is sent only after the
/// file and its directory are synced: the extension may then forget it.
async fn store_capture(State(host): State<Arc<BrowserHost>>, body: Bytes) -> HttpResponse {
    let value: Value = match serde_json::from_slice(&body) {
        Ok(v @ Value::Object(_)) => v,
        _ => {
            return refuse(
                StatusCode::BAD_REQUEST,
                "capture_not_json",
                "a capture is one JSON object",
            );
        }
    };
    let Some((provider, conversation)) = capture_identity(&value) else {
        return refuse(
            StatusCode::BAD_REQUEST,
            "capture_names_no_conversation",
            "a capture names session.provider and session.provider_session_id",
        );
    };
    let path = host.capture_log(&provider, &conversation);
    let hash = blake3::hash(&body).to_hex().to_string();
    match blocking(move || super::captures::append(&path, &body)).await {
        Some(Ok(new)) => Json(json!({
            "provider": provider,
            "conversation": conversation,
            "hash": hash,
            "new": new,
        }))
        .into_response(),
        Some(Err(e)) => refuse(
            StatusCode::INTERNAL_SERVER_ERROR,
            "write_failed",
            &e.to_string(),
        ),
        None => refuse(
            StatusCode::INTERNAL_SERVER_ERROR,
            "write_failed",
            "the write did not finish",
        ),
    }
}

/// The conversation a capture holds and the files it names, read with the
/// same parser the loop uses, so the extension can fetch their bytes before
/// it stores the capture.
async fn capture_files(body: Bytes) -> HttpResponse {
    match blocking(move || polylogue_sources::browser_capture::files(&body)).await {
        Some(Ok(result)) => Json(result).into_response(),
        Some(Err(error)) => refuse(
            StatusCode::UNPROCESSABLE_ENTITY,
            "capture_unreadable",
            &error,
        ),
        None => refuse(
            StatusCode::INTERNAL_SERVER_ERROR,
            "capture_unreadable",
            "reading the capture did not finish",
        ),
    }
}

#[derive(Deserialize)]
struct ConversationsAsked {
    provider: String,
    ids: Vec<String>,
}

/// What the archive holds of some conversations: `archived` with its
/// message count and last capture time, `spooled` (stored, not yet read by
/// the daemon), `missing`, or `unknown` while the daemon does not answer.
async fn conversations(
    State(host): State<Arc<BrowserHost>>,
    Json(asked): Json<ConversationsAsked>,
) -> HttpResponse {
    let spooled: Vec<bool> = asked
        .ids
        .iter()
        .map(|id| host.capture_log(&asked.provider, id).exists())
        .collect();
    let request = Request::Conversations {
        provider: asked.provider.clone(),
        ids: asked.ids.clone(),
    };
    let held = ask(host, request).await.ok();
    let daemon = held.is_some();
    let rows: Vec<Value> = asked
        .ids
        .iter()
        .enumerate()
        .map(|(i, id)| {
            let row = held
                .as_ref()
                .map_or(&Value::Null, |h| &h["conversations"][i]);
            let messages = row["messages"].as_u64();
            let state = match (daemon, messages, spooled[i]) {
                (true, Some(_), _) => "archived",
                (_, _, true) => "spooled",
                (true, None, false) => "missing",
                (false, _, false) => "unknown",
            };
            json!({
                "id": id,
                "state": state,
                "session": row["session"],
                "messages": messages,
                "last_captured_at": row["last_captured_at"],
            })
        })
        .collect();
    Json(json!({
        "daemon": if daemon { "running" } else { "no_answer" },
        "conversations": rows,
    }))
    .into_response()
}

async fn selection_note(
    State(host): State<Arc<BrowserHost>>,
    Json(input): Json<super::notes::SelectionNote>,
) -> HttpResponse {
    match blocking(move || super::notes::save(&host, input)).await {
        Some(Response::Ok { data, .. }) => Json(json!({ "note": data })).into_response(),
        Some(Response::Error { kind, message, .. }) => {
            let status = match kind.as_str() {
                "bad_request" | "invalid_query" => StatusCode::BAD_REQUEST,
                "conflict" => StatusCode::CONFLICT,
                "daemon_required" => StatusCode::SERVICE_UNAVAILABLE,
                _ => StatusCode::INTERNAL_SERVER_ERROR,
            };
            refuse(status, &kind, &message)
        }
        None => refuse(
            StatusCode::SERVICE_UNAVAILABLE,
            "save_outcome_unknown",
            "the note may or may not be saved; retry it with the same ID",
        ),
    }
}

#[derive(Deserialize)]
struct ClaimQuery {
    claim_by: Option<String>,
}

/// Lists chat actions, or with `claim_by` hands that extension instance
/// one queued action (`action`, null when none) while chat actions are on.
async fn chat_actions(
    State(host): State<Arc<BrowserHost>>,
    Query(q): Query<ClaimQuery>,
) -> HttpResponse {
    let request = match q.claim_by.filter(|by| !by.is_empty()) {
        None => Request::ChatActions,
        Some(by) => Request::ChatClaim { by },
    };
    answer(host, request).await
}

async fn chat_report(
    State(host): State<Arc<BrowserHost>>,
    UrlPath(id): UrlPath<String>,
    Json(update): Json<Update>,
) -> HttpResponse {
    answer(host, Request::ChatReport { id, update }).await
}

#[derive(Deserialize)]
struct Decision {
    by: String,
    approve: bool,
    #[serde(default)]
    detail: Option<String>,
}

/// The operator's decision from the extension's popup.
async fn chat_decision(
    State(host): State<Arc<BrowserHost>>,
    UrlPath(id): UrlPath<String>,
    Json(d): Json<Decision>,
) -> HttpResponse {
    let request = Request::ChatDecide {
        id,
        approve: d.approve,
        by: d.by,
        detail: d.detail,
    };
    answer(host, request).await
}

async fn chat_attachment(
    State(host): State<Arc<BrowserHost>>,
    UrlPath((id, attachment)): UrlPath<(String, String)>,
) -> HttpResponse {
    use base64::Engine as _;
    let v = match ask(host, Request::ChatAttachment { id, attachment }).await {
        Ok(v) => v,
        Err(refusal) => return *refusal,
    };
    let Ok(bytes) =
        base64::engine::general_purpose::STANDARD.decode(v["bytes"].as_str().unwrap_or(""))
    else {
        return refuse(
            StatusCode::INTERNAL_SERVER_ERROR,
            "attachment_unreadable",
            "the daemon's answer held no bytes",
        );
    };
    let media_type = v["media_type"]
        .as_str()
        .unwrap_or("application/octet-stream")
        .to_owned();
    ([(header::CONTENT_TYPE, media_type)], bytes).into_response()
}

async fn backfill_jobs(State(host): State<Arc<BrowserHost>>) -> HttpResponse {
    answer(host, Request::BackfillJobs).await
}

#[derive(Deserialize)]
struct Claimant {
    by: String,
}

/// Hands the extension instance the job it holds or the oldest waiting
/// one (`job`, null when none).
async fn backfill_claim(
    State(host): State<Arc<BrowserHost>>,
    Json(c): Json<Claimant>,
) -> HttpResponse {
    answer(host, Request::BackfillClaim { by: c.by }).await
}

#[derive(Deserialize)]
struct Report {
    by: String,
    #[serde(flatten)]
    report: BackfillReport,
}

async fn backfill_report(
    State(host): State<Arc<BrowserHost>>,
    UrlPath(id): UrlPath<String>,
    Json(r): Json<Report>,
) -> HttpResponse {
    let request = Request::BackfillReport {
        id,
        by: r.by,
        report: r.report,
    };
    answer(host, request).await
}
