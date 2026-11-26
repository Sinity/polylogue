//! The routes the extension calls through its native messaging host. Each
//! request is served in process by this router; nothing listens on
//! a port. A refusal answers `{"error": code, "message": text}` with an
//! HTTP status.

use super::BrowserHost;
use super::actions::Update;
use axum::Router;
use axum::body::Bytes;
use axum::extract::{Path as UrlPath, Query, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Json, Response as HttpResponse};
use axum::routing::{get, post};
use polylogue_model::JsonExt as _;
use polylogue_model::user::ImportJobReport;
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
        .route("/v1/chat-actions/{id}/answer", post(chat_answer))
        .route(
            "/v1/chat-actions/{id}/attachments/{attachment}",
            get(chat_attachment),
        )
        .route("/v1/import_job", get(import_jobs))
        .route("/v1/import_job/claim", post(import_job_claim))
        .route("/v1/import_job/{id}/report", post(import_job_report))
        // A capture or a note is as large as the conversation it holds.
        .layer(axum::extract::DefaultBodyLimit::disable())
        .with_state(host)
}

fn refuse(status: StatusCode, code: polylogue_model::Code, message: &str) -> HttpResponse {
    (
        status,
        Json(json!({ "error": code.as_str(), "message": message })),
    )
        .into_response()
}

/// A refusal with its typed reason beside its code and sentence.
fn refuse_for(status: StatusCode, reason: polylogue_model::Reason) -> HttpResponse {
    (
        status,
        Json(json!({
            "error": reason.code().as_str(),
            "message": reason.to_string(),
            "reason": reason,
        })),
    )
        .into_response()
}

/// A daemon failure with its shared HTTP status; a failure with no HTTP
/// status (the daemon not running) is 503.
fn refuse_with(failure: &command_surface::Failure) -> HttpResponse {
    let status = failure
        .kind
        .http_status()
        .and_then(|status| StatusCode::from_u16(status).ok())
        .unwrap_or(StatusCode::SERVICE_UNAVAILABLE);
    (
        status,
        Json(json!({
            "error": failure.code.as_deref().unwrap_or_default(),
            "message": failure.message,
        })),
    )
        .into_response()
}

/// Runs blocking work (files, the socket) off the async thread.
async fn blocking<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> Option<T> {
    tokio::task::spawn_blocking(f).await.ok()
}

/// Asks the daemon and answers with its data, or its refusal's status and
/// code; 503 `daemon_required` when it does not answer (stopped, or too busy).
async fn ask(
    host: Arc<BrowserHost>,
    operation: &'static str,
    arguments: Value,
) -> Result<Value, Box<HttpResponse>> {
    match blocking(move || host.ask(operation, arguments))
        .await
        .flatten()
    {
        Some(answer) => {
            crate::protocol::into_value(answer).map_err(|failure| Box::new(refuse_with(&failure)))
        }
        None => Err(Box::new(refuse_for(
            StatusCode::SERVICE_UNAVAILABLE,
            polylogue_model::Reason::DaemonDidNotAnswer {},
        ))),
    }
}

async fn answer(host: Arc<BrowserHost>, operation: &'static str, arguments: Value) -> HttpResponse {
    match ask(host, operation, arguments).await {
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
    let dir = host.runtime_dir.clone();
    match blocking(move || crate::web::access::reader_url_in(&dir)).await {
        Some(Ok(url)) => Json(json!({ "url": url })).into_response(),
        _ => refuse(
            StatusCode::SERVICE_UNAVAILABLE,
            polylogue_model::Code::ReaderUnavailable,
            "no reader is running; start it with `polylogue web`",
        ),
    }
}

/// The conversation a capture names.
fn capture_identity(v: &Value) -> Option<(String, String)> {
    let provider = v.pointer("/session/provider")?.as_str()?.trim();
    let conversation = v.pointer("/session/provider_session_id")?.as_str()?.trim();
    (!provider.is_empty() && !conversation.is_empty())
        .then(|| (provider.to_owned(), conversation.to_owned()))
}

/// Whether a capture carries a conversation: the provider's own JSON, or
/// turns read from the page. A capture with neither would become a session
/// with no messages that reads as archived.
fn holds_conversation(v: &Value) -> bool {
    let raw = v
        .object("raw_provider_payload")
        .is_some_and(|o| !o.is_empty());
    let turns = v
        .pointer("/session/turns")
        .and_then(Value::as_array)
        .is_some_and(|t| !t.is_empty());
    raw || turns
}

/// Appends one capture to its conversation's capture log. Bytes the log
/// already holds are not appended again. The answer is sent only after the
/// file and its directory are synced: the extension may then forget it.
async fn store_capture(State(host): State<Arc<BrowserHost>>, body: Bytes) -> HttpResponse {
    let value: Value = match serde_json::from_slice(&body) {
        Ok(v @ Value::Object(_)) => v,
        _ => {
            return refuse_for(
                StatusCode::BAD_REQUEST,
                polylogue_model::Reason::CaptureNotJson {},
            );
        }
    };
    let Some((provider, conversation)) = capture_identity(&value) else {
        return refuse_for(
            StatusCode::BAD_REQUEST,
            polylogue_model::Reason::CaptureNamesNoConversation {},
        );
    };
    if !holds_conversation(&value) {
        return refuse_for(
            StatusCode::BAD_REQUEST,
            polylogue_model::Reason::CaptureHasNoConversation {},
        );
    }
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
            polylogue_model::Code::Io,
            &e.to_string(),
        ),
        None => refuse(
            StatusCode::INTERNAL_SERVER_ERROR,
            polylogue_model::Code::Io,
            "the write did not finish",
        ),
    }
}

/// The conversation a capture holds and the files it names, read with the
/// same parser the loop uses, so the extension can fetch their bytes before
/// it stores the capture. Each file the archive already holds bytes for
/// names them in `stored` (`{blob, size}`), so the extension refers to them
/// instead of downloading them again; while the daemon does not answer,
/// `stored` is null and the extension downloads.
async fn capture_files(State(host): State<Arc<BrowserHost>>, body: Bytes) -> HttpResponse {
    let mut listed = match blocking(move || polylogue_sources::browser_capture::files(&body)).await
    {
        Some(Ok(listed)) => listed,
        Some(Err(error)) => {
            return refuse_for(
                StatusCode::UNPROCESSABLE_ENTITY,
                polylogue_model::Reason::CaptureUnreadable {
                    problem: error.to_string(),
                },
            );
        }
        None => {
            return refuse(
                StatusCode::INTERNAL_SERVER_ERROR,
                polylogue_model::Code::Internal,
                "reading the capture did not finish",
            );
        }
    };
    let provider = listed["conversation"]["provider"]
        .as_str()
        .map(str::to_owned);
    let conversation = listed["conversation"]["id"].as_str().map(str::to_owned);
    let held = match provider.zip(conversation) {
        Some((provider, conversation)) => ask(
            host,
            "internal stored-files",
            json!({ "provider": provider, "conversation": conversation }),
        )
        .await
        .ok(),
        None => None,
    };
    listed["downloading"] = held
        .as_ref()
        .map_or(Value::Bool(false), |data| data["downloading"].clone());
    let held_files = held.as_ref().map_or(&[][..], |data| {
        data["files"].as_array().map_or(&[][..], Vec::as_slice)
    });
    if let Some(files) = listed["files"].as_array_mut() {
        for file in files {
            let found = file["file_id"]
                .as_str()
                .and_then(|id| held_files.iter().find(|s| s["id"] == id));
            file["stored"] = match found {
                Some(s) => json!({ "blob": s["blob"], "size": s["size"] }),
                None => Value::Null,
            };
        }
    }
    Json(listed).into_response()
}

#[derive(Deserialize)]
struct ConversationsAsked {
    provider: String,
    ids: Vec<String>,
}

/// What the archive holds of some conversations: `archived` with its
/// message count and last capture time, `spooled` (saved in the spool, not yet
/// read by the daemon), `missing`, or `unknown` while the daemon does not answer.
async fn conversations(
    State(host): State<Arc<BrowserHost>>,
    Json(asked): Json<ConversationsAsked>,
) -> HttpResponse {
    let spooled: Vec<bool> = asked
        .ids
        .iter()
        .map(|id| host.capture_log(&asked.provider, id).exists())
        .collect();
    let arguments = json!({ "provider": asked.provider, "ids": asked.ids });
    let held = ask(host, "internal conversations", arguments).await.ok();
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
                "downloading": row["downloading"],
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
    match blocking(move || super::notes::save(&host, input))
        .await
        .map(crate::protocol::into_value)
    {
        Some(Ok(data)) => Json(json!({ "note": data })).into_response(),
        Some(Err(failure)) => refuse_with(&failure),
        None => refuse_for(
            StatusCode::SERVICE_UNAVAILABLE,
            polylogue_model::Reason::SaveOutcomeUnknown {},
        ),
    }
}

#[derive(Deserialize)]
struct ClaimQuery {
    by: Option<String>,
}

/// The first page of a query's rows, newest first for the extension's units.
async fn listed(host: Arc<BrowserHost>, selection: &str, key: &str) -> HttpResponse {
    match ask(host, "rows", json!({ "selection": selection })).await {
        Ok(data) => Json(json!({ key: data["reply"]["rows"] })).into_response(),
        Err(refusal) => *refusal,
    }
}

/// The chat actions awaiting the operator's approval (the unit
/// `chat_actions`), or with `by` hands that extension instance one queued
/// action (`action`, null when none) while chat actions are on.
async fn chat_actions(
    State(host): State<Arc<BrowserHost>>,
    Query(q): Query<ClaimQuery>,
) -> HttpResponse {
    match q.by.filter(|by| !by.is_empty()) {
        None => {
            listed(
                host,
                "chat_actions where state:awaiting_approval",
                "actions",
            )
            .await
        }
        Some(by) => answer(host, "internal chat claim", json!({ "by": by })).await,
    }
}

async fn chat_report(
    State(host): State<Arc<BrowserHost>>,
    UrlPath(id): UrlPath<String>,
    Json(update): Json<Update>,
) -> HttpResponse {
    answer(
        host,
        "internal chat report",
        json!({ "id": id, "update": update }),
    )
    .await
}

#[derive(Deserialize)]
struct HeldAnswer {
    value: String,
    #[serde(default)]
    reason: Option<String>,
}

/// The operator's Allow or Deny from the extension's popup, answered as
/// `answer allow|deny` is.
async fn chat_answer(
    State(host): State<Arc<BrowserHost>>,
    UrlPath(id): UrlPath<String>,
    Json(a): Json<HeldAnswer>,
) -> HttpResponse {
    let arguments = json!({ "id": id, "value": a.value, "reason": a.reason });
    answer(host, "internal chat answer", arguments).await
}

async fn chat_attachment(
    State(host): State<Arc<BrowserHost>>,
    UrlPath((id, attachment)): UrlPath<(String, String)>,
) -> HttpResponse {
    use base64::Engine as _;
    let arguments = json!({ "id": id, "attachment": attachment });
    let v = match ask(host, "internal chat attachment", arguments).await {
        Ok(v) => v,
        Err(refusal) => return *refusal,
    };
    let Ok(bytes) =
        base64::engine::general_purpose::STANDARD.decode(v["bytes"].as_str().unwrap_or(""))
    else {
        return refuse(
            StatusCode::INTERNAL_SERVER_ERROR,
            polylogue_model::Code::Internal,
            "the daemon's answer held no bytes",
        );
    };
    let media_type = v["media_type"]
        .as_str()
        .unwrap_or("application/octet-stream")
        .to_owned();
    ([(header::CONTENT_TYPE, media_type)], bytes).into_response()
}

/// The import jobs, newest first (the unit `import_jobs`).
async fn import_jobs(State(host): State<Arc<BrowserHost>>) -> HttpResponse {
    listed(host, "import_jobs", "jobs").await
}

#[derive(Deserialize)]
struct Claimant {
    by: String,
}

/// Hands the extension instance the job it holds or the oldest waiting
/// one (`job`, null when none).
async fn import_job_claim(
    State(host): State<Arc<BrowserHost>>,
    Json(c): Json<Claimant>,
) -> HttpResponse {
    answer(host, "internal import claim", json!({ "by": c.by })).await
}

#[derive(Deserialize)]
struct Report {
    by: String,
    #[serde(flatten)]
    report: ImportJobReport,
}

async fn import_job_report(
    State(host): State<Arc<BrowserHost>>,
    UrlPath(id): UrlPath<String>,
    Json(r): Json<Report>,
) -> HttpResponse {
    let arguments = json!({ "id": id, "by": r.by, "report": r.report });
    answer(host, "internal import report", arguments).await
}
