//! The routes the extension calls through its native messaging host. Each
//! native request is served in process by this router; nothing listens on
//! a port.

use super::actions::{self, ActionRequest, Update};
use super::{API_SCHEMA, Ctx, artifact_ref, new_id, now_iso, sha256_hex, write_durable};
use crate::daemon::{Request, Response};
use axum::Router;
use axum::body::Bytes;
use axum::extract::{Path as UrlPath, Query, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Json, Response as HttpResponse};
use axum::routing::{get, post};
use serde::Deserialize;
use serde_json::{Value, json};
use std::fs;
use std::sync::Arc;

const RECEIVER: &str = "polylogue-browser-capture";

pub fn router(ctx: Arc<Ctx>) -> Router {
    Router::new()
        .route("/v1/status", get(status))
        .route("/v1/reader-location", get(reader_location))
        .route(
            "/v1/browser-captures/capabilities",
            get(capture_capabilities),
        )
        .route("/v1/browser-captures", post(capture))
        .route("/v1/browser-captures/inspect", post(inspect_capture))
        .route("/v1/archive-state", get(archive_state))
        .route("/v1/notes/selection", post(selection_note))
        .route("/v1/capture-health", get(health_list).post(health_report))
        .route("/v1/browser-actions/capabilities", get(action_capabilities))
        .route(
            "/v1/browser-actions",
            get(actions_list).post(refuse_enqueue),
        )
        .route("/v1/browser-actions/{id}", get(action_read))
        .route("/v1/browser-actions/{id}/events", post(action_event))
        .route("/v1/browser-actions/{id}/approval", post(action_approval))
        .route(
            "/v1/browser-actions/{id}/attachments/{attachment}",
            get(attachment_read),
        )
        .fallback(not_built)
        .layer(middleware::from_fn(echo_request_id))
        // A capture or a note is as large as the conversation it holds.
        .layer(axum::extract::DefaultBodyLimit::disable())
        .with_state(ctx)
}

/// Every answer echoes the request's `X-Request-ID`, which the extension
/// keeps as the receipt of a durable write.
async fn echo_request_id(req: axum::extract::Request, next: Next) -> HttpResponse {
    let id = req.headers().get("x-request-id").cloned();
    let mut r = next.run(req).await;
    let id = id.unwrap_or_else(|| {
        HeaderValue::from_str(&new_id()).unwrap_or(HeaderValue::from_static("none"))
    });
    r.headers_mut().insert("x-request-id", id);
    r
}

/// Routes the extension calls that this host does not serve yet, such as
/// mission control.
async fn not_built() -> HttpResponse {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(json!({ "ok": false, "error": "not_built" })),
    )
        .into_response()
}

fn bad(status: StatusCode, error: &str) -> HttpResponse {
    (status, Json(json!({ "ok": false, "error": error }))).into_response()
}

/// Runs blocking work (files, the socket) off the async thread.
async fn blocking<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> Option<T> {
    tokio::task::spawn_blocking(f).await.ok()
}

async fn status(State(ctx): State<Arc<Ctx>>) -> HttpResponse {
    let c = Arc::clone(&ctx);
    let (daemon, enabled) = blocking(move || {
        (
            matches!(c.ask(&Request::Status), Some(Response::Ok { .. })),
            c.actions_enabled(),
        )
    })
    .await
    .unwrap_or((false, false));
    Json(json!({
        "ok": true,
        "receiver": RECEIVER,
        "receiver_id": ctx.receiver_id,
        "api_schema": API_SCHEMA,
        "schema_version": 1,
        "daemon": if daemon { "running" } else { "not_running" },
        "browser_actions": { "enabled": enabled },
    }))
    .into_response()
}

async fn reader_location(State(ctx): State<Arc<Ctx>>) -> HttpResponse {
    let Some(key) = ctx.receiver_id.strip_prefix("polylogue:") else {
        return bad(StatusCode::SERVICE_UNAVAILABLE, "reader_unavailable");
    };
    // Ctx's receiver key identifies the same archive as the daemon's reader.
    let dir = crate::runtime_dir::runtime_dir(key);
    let location = blocking(move || crate::web::access::reader_url_in(&dir)).await;
    match location {
        Some(Ok(url)) => Json(json!({"ok":true,"url":url})).into_response(),
        _ => bad(StatusCode::SERVICE_UNAVAILABLE, "reader_unavailable"),
    }
}

async fn capture_capabilities() -> HttpResponse {
    Json(json!({
        "ok": true,
        "receiver": RECEIVER,
        "schema_version": 1,
        "durable_ack_fields": ["receiver_request_id", "content_hash", "submitted_content_hash", "outcome"],
        "outcomes": ["accepted", "noop"],
        "capture_jobs": false,
        "inspect": true,
        "selection_notes": true,
    }))
    .into_response()
}

async fn selection_note(
    State(ctx): State<Arc<Ctx>>,
    Json(input): Json<super::notes::SelectionNote>,
) -> HttpResponse {
    match blocking(move || super::notes::save(&ctx, input)).await {
        Some(Response::Ok { data }) => Json(json!({"ok":true,"note":data})).into_response(),
        Some(Response::Error { kind, message, .. }) => {
            let status = match kind.as_str() {
                "bad_request" | "invalid_query" => StatusCode::BAD_REQUEST,
                "conflict" => StatusCode::CONFLICT,
                "daemon_required" => StatusCode::SERVICE_UNAVAILABLE,
                _ => StatusCode::INTERNAL_SERVER_ERROR,
            };
            (
                status,
                Json(json!({"ok":false,"error":kind,"message":message})),
            )
                .into_response()
        }
        _ => bad(StatusCode::SERVICE_UNAVAILABLE, "save_outcome_unknown"),
    }
}

async fn inspect_capture(body: Bytes) -> HttpResponse {
    match blocking(move || polylogue_sources::browser_capture::inspect(&body)).await {
        Some(Ok(mut result)) => {
            result["ok"] = json!(true);
            Json(result).into_response()
        }
        Some(Err(error)) => bad(StatusCode::UNPROCESSABLE_ENTITY, &error),
        None => bad(StatusCode::INTERNAL_SERVER_ERROR, "inspection_failed"),
    }
}

/// The conversation a capture names.
fn capture_identity(v: &Value) -> Option<(String, String)> {
    let provider = v.pointer("/session/provider")?.as_str()?.trim();
    let session = v.pointer("/session/provider_session_id")?.as_str()?.trim();
    (!provider.is_empty() && !session.is_empty()).then(|| (provider.to_owned(), session.to_owned()))
}

/// Accepts one capture into the spool. Identical bytes already there are a
/// no-op; anything else replaces the conversation's file. The answer is
/// sent only after the file and its directory are synced.
async fn capture(State(ctx): State<Arc<Ctx>>, headers: HeaderMap, body: Bytes) -> HttpResponse {
    let value: Value = match serde_json::from_slice(&body) {
        Ok(v @ Value::Object(_)) => v,
        _ => return bad(StatusCode::BAD_REQUEST, "invalid_envelope"),
    };
    let Some((provider, session)) = capture_identity(&value) else {
        return bad(StatusCode::BAD_REQUEST, "missing_session_identity");
    };
    let rel = artifact_ref(&provider, &session);
    let path = ctx.spool.join(&rel);
    let hash = sha256_hex(&body);
    let request_id = headers
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .map_or_else(new_id, str::to_owned);
    let bytes = body.clone();
    let written = blocking(move || super::captures::append(&path, &bytes)).await;
    let new = match written {
        Some(Ok(w)) => w,
        _ => return bad(StatusCode::INTERNAL_SERVER_ERROR, "write_failed"),
    };
    let capture_id = value
        .get("capture_id")
        .and_then(Value::as_str)
        .map_or_else(|| format!("{provider}:{session}"), str::to_owned);
    Json(json!({
        "ok": true,
        "receiver": RECEIVER,
        "schema_version": 1,
        "source": "browser-extension",
        "receiver_request_id": request_id,
        "capture_id": capture_id,
        "capture_instance_id": value.get("capture_instance_id").cloned().unwrap_or(Value::Null),
        "provider": provider,
        "provider_session_id": session,
        "artifact_ref": rel,
        "outcome": if new { "accepted" } else { "noop" },
        "submitted_content_hash": hash,
        "content_hash": hash,
        "dedup_content_hash": hash,
        "bytes_written": if new { body.len() } else { 0 },
        "replaced": false,
        "appended": new,
        "deduplicated": !new,
        "state": "spooled_only",
    }))
    .into_response()
}

#[derive(Deserialize)]
struct StateQuery {
    provider: String,
    provider_session_id: String,
}

fn sql_text(s: &str) -> String {
    format!("'{}'", s.replace('\'', "''"))
}

/// What the archive holds for one conversation: spooled, archived with its
/// message count, or missing. When the daemon is not running, what the
/// index holds is unknown, not missing.
async fn archive_state(State(ctx): State<Arc<Ctx>>, Query(q): Query<StateQuery>) -> HttpResponse {
    let rel = artifact_ref(&q.provider, &q.provider_session_id);
    let spooled = ctx.spool.join(&rel).exists();
    let session_id =
        polylogue_sources::browser_capture::session_for(&q.provider, &q.provider_session_id)
            .map(|s| s.to_string());
    let c = Arc::clone(&ctx);
    let sid = session_id.clone();
    let indexed: Option<Option<i64>> = blocking(move || {
        let sid = sid?;
        let statement = format!(
            "SELECT (SELECT COUNT(*) FROM message m WHERE m.session_pk = s.pk) AS n
               FROM session s WHERE s.id = {}",
            sql_text(&sid)
        );
        match c.ask(&Request::Sql { statement })? {
            Response::Ok { data } => Some(data.get(0).and_then(|r| r["n"].as_i64())),
            _ => None,
        }
    })
    .await
    .flatten();
    let state = match indexed {
        Some(Some(n)) if n > 0 => "archived",
        None if spooled => "spooled_unknown_index",
        _ if spooled => "spooled_only",
        None => "unknown",
        _ => "missing",
    };
    Json(json!({
        "ok": true,
        "provider": q.provider,
        "provider_session_id": q.provider_session_id,
        "state": state,
        "lifecycle": state,
        "captured": state == "archived",
        "spooled": spooled,
        "artifact_ref": rel,
        "indexed_session_id": session_id,
        "indexed_message_count": indexed.flatten(),
        "daemon": if indexed.is_some() { "running" } else { "not_running" },
    }))
    .into_response()
}

/// The latest capture-health report per provider, for the popup.
async fn health_report(State(ctx): State<Arc<Ctx>>, Json(mut report): Json<Value>) -> HttpResponse {
    let provider = report["provider"]
        .as_str()
        .map_or_else(|| "all".into(), super::safe_token);
    if let Some(o) = report.as_object_mut() {
        o.insert("received_at".into(), json!(now_iso()));
    }
    let path = ctx.health.join(format!("{provider}.json"));
    let bytes = serde_json::to_vec(&report).unwrap_or_default();
    match blocking(move || write_durable(&path, &bytes)).await {
        Some(Ok(())) => Json(json!({ "ok": true })).into_response(),
        _ => bad(StatusCode::INTERNAL_SERVER_ERROR, "write_failed"),
    }
}

async fn health_list(State(ctx): State<Arc<Ctx>>) -> HttpResponse {
    let dir = ctx.health.clone();
    let reports: Vec<Value> = fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| serde_json::from_slice(&fs::read(e.path()).ok()?).ok())
        .collect();
    Json(json!({ "ok": true, "reports": reports })).into_response()
}

async fn action_capabilities(State(ctx): State<Arc<Ctx>>) -> HttpResponse {
    let c = Arc::clone(&ctx);
    let enabled = blocking(move || c.actions_enabled()).await.unwrap_or(false);
    let providers: Vec<Value> = actions::providers()
        .map(|p| {
            json!({
                "provider": p,
                "operations": actions::OPERATIONS,
                "submit_policies": ["stage_only", "submit_once"],
            })
        })
        .collect();
    Json(json!({
        "ok": true,
        "receiver": RECEIVER,
        "enabled": enabled,
        "providers": providers,
    }))
    .into_response()
}

/// The extension never enqueues: only the operator does, through the CLI
/// or the reader.
async fn refuse_enqueue(_: Json<ActionRequest>) -> HttpResponse {
    bad(StatusCode::FORBIDDEN, "only_the_operator_enqueues")
}

#[derive(Deserialize)]
struct ListQuery {
    claim_by: Option<String>,
}

/// Asks the daemon about chat actions, and answers as the extension expects:
/// the daemon's data with `ok`, or a refusal's status and code.
async fn ask_chat(ctx: Arc<Ctx>, request: Request) -> Result<Value, Box<HttpResponse>> {
    match blocking(move || ctx.ask(&request)).await.flatten() {
        Some(Response::Ok { mut data }) => {
            if let Some(object) = data.as_object_mut() {
                object.remove("created");
                object.insert("ok".into(), json!(true));
                object.insert("receiver".into(), json!(RECEIVER));
            }
            Ok(data)
        }
        Some(Response::Error { kind, detail, .. }) => {
            let status = detail
                .as_ref()
                .and_then(|d| d["http_status"].as_u64())
                .and_then(|s| StatusCode::from_u16(u16::try_from(s).ok()?).ok())
                .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            Err(Box::new(bad(status, &kind)))
        }
        None => Err(Box::new(bad(
            StatusCode::SERVICE_UNAVAILABLE,
            "daemon_required",
        ))),
    }
}

/// Lists actions, or with `claim_by` hands one queued action to that
/// extension instance while chat actions are on.
async fn actions_list(State(ctx): State<Arc<Ctx>>, Query(q): Query<ListQuery>) -> HttpResponse {
    let request = match q.claim_by.filter(|o| !o.is_empty()) {
        None => Request::ChatActions,
        Some(by) => Request::ChatClaim { by },
    };
    match ask_chat(ctx, request).await {
        Ok(mut v) => {
            // A claim answers with the list of what it handed out: one
            // action or none.
            if let Some(action) = v.as_object_mut().and_then(|o| o.remove("action")) {
                v["actions"] = json!(
                    Some(action)
                        .filter(|a| !a.is_null())
                        .into_iter()
                        .collect::<Vec<_>>()
                );
                if v.get("enabled").is_none() {
                    v["enabled"] = json!(true);
                }
            }
            Json(v).into_response()
        }
        Err(r) => *r,
    }
}

async fn action_read(State(ctx): State<Arc<Ctx>>, UrlPath(id): UrlPath<String>) -> HttpResponse {
    match ask_chat(ctx, Request::ChatAction { id }).await {
        Ok(v) => Json(v).into_response(),
        Err(r) => *r,
    }
}

async fn action_event(
    State(ctx): State<Arc<Ctx>>,
    UrlPath(id): UrlPath<String>,
    Json(update): Json<Update>,
) -> HttpResponse {
    match ask_chat(ctx, Request::ChatReport { id, update }).await {
        Ok(v) => Json(v).into_response(),
        Err(r) => *r,
    }
}

#[derive(Deserialize)]
struct Approval {
    extension_instance_id: String,
    decision: String,
    #[serde(default)]
    detail: Option<String>,
}

/// The operator's decision from the extension's popup.
async fn action_approval(
    State(ctx): State<Arc<Ctx>>,
    UrlPath(id): UrlPath<String>,
    Json(d): Json<Approval>,
) -> HttpResponse {
    let approve = match d.decision.as_str() {
        "approve" => true,
        "decline" => false,
        _ => return bad(StatusCode::BAD_REQUEST, "decision_is_approve_or_decline"),
    };
    let request = Request::ChatDecide {
        id,
        approve,
        by: d.extension_instance_id,
        detail: d.detail,
    };
    match ask_chat(ctx, request).await {
        Ok(v) => Json(v).into_response(),
        Err(r) => *r,
    }
}

async fn attachment_read(
    State(ctx): State<Arc<Ctx>>,
    UrlPath((id, attachment)): UrlPath<(String, String)>,
) -> HttpResponse {
    use base64::Engine as _;
    let v = match ask_chat(ctx, Request::ChatAttachment { id, attachment }).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(v["bytes"].as_str().unwrap_or(""))
        .unwrap_or_default();
    let media_type = v["media_type"]
        .as_str()
        .unwrap_or("application/octet-stream")
        .to_owned();
    ([(header::CONTENT_TYPE, media_type)], bytes).into_response()
}
