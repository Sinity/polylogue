//! The reader's JSON API under `/api/v1/` (spec `surfaces/web.md`): the
//! same data as `--json`, one route per CLI verb or command, each mapped to
//! the socket's `Request`. Reading verbs are GET, writing verbs POST.
//! Excision is not offered over HTTP. Other programs use the socket.

use super::{Web, ask, json_answer};
use crate::daemon::Request;
use axum::Router;
use axum::extract::{Path as UrlPath, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response as HttpResponse};
use axum::routing::{get, post};
use polylogue_store::NoteFilter;
use serde::Deserialize;
use std::sync::Arc;

pub fn routes() -> Router<Arc<Web>> {
    Router::new()
        .route("/list", get(list))
        .route("/read", get(read))
        .route("/note", post(note))
        .route("/withdraw", post(withdraw))
        .route("/cmd/{command}", get(command_get).post(command_post))
        .route("/browser-actions", post(browser_action))
        .route("/browser-actions/attachments", post(browser_attachment))
}

/// The operator queues a chat action for the extension. It runs only while
/// the `browser.actions` setting is on.
async fn browser_action(
    State(web): State<Arc<Web>>,
    axum::Json(req): axum::Json<crate::browser::actions::ActionRequest>,
) -> HttpResponse {
    let ctx = Arc::clone(&web.browser);
    let setting = Request::Setting {
        request_id: None,
        key: "browser.actions".into(),
        value: None,
        machine: false,
    };
    let enabled = matches!(
        ask(&web, setting).await,
        crate::daemon::Response::Ok { data } if data["value"] == "on"
    );
    let done =
        tokio::task::spawn_blocking(move || crate::browser::actions::enqueue(&ctx, &req, enabled))
            .await;
    match done {
        Ok(Ok((new, action))) => (
            if new {
                StatusCode::CREATED
            } else {
                StatusCode::OK
            },
            axum::Json(serde_json::json!({ "ok": true, "action": action })),
        )
            .into_response(),
        Ok(Err(r)) => (
            StatusCode::from_u16(r.0).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            axum::Json(serde_json::json!({ "ok": false, "error": r.1 })),
        )
            .into_response(),
        Err(_) => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}

/// Stores an attachment's bytes for a later chat action.
async fn browser_attachment(State(web): State<Arc<Web>>, body: axum::body::Bytes) -> HttpResponse {
    let ctx = Arc::clone(&web.browser);
    match tokio::task::spawn_blocking(move || crate::browser::actions::upload(&ctx, &body)).await {
        Ok(Ok(v)) => axum::Json(v).into_response(),
        Ok(Err(r)) => (
            StatusCode::from_u16(r.0).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            axum::Json(serde_json::json!({ "ok": false, "error": r.1 })),
        )
            .into_response(),
        Err(_) => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}

#[derive(Deserialize)]
struct ListParams {
    #[serde(default)]
    q: String,
    cursor: Option<String>,
    page_size: Option<u64>,
    #[serde(default)]
    why: bool,
    limit: Option<u64>,
}

async fn list(State(web): State<Arc<Web>>, Query(p): Query<ListParams>) -> HttpResponse {
    let request = if p.q.trim().is_empty() {
        Request::List {
            limit: p.limit.or(p.page_size).unwrap_or(50),
            before: None,
        }
    } else {
        Request::Query {
            text: p.q,
            cursor: p.cursor,
            page_size: p.page_size.or(p.limit),
            why: p.why,
            session: None,
            fields: vec![],
        }
    };
    json_answer(&web, ask(&web, request).await)
}

#[derive(Deserialize)]
struct ReadParams {
    q: String,
    view: Option<String>,
    format: Option<String>,
    #[serde(default = "yes")]
    logical: bool,
    budget: Option<usize>,
    cursor: Option<String>,
}

fn yes() -> bool {
    true
}

async fn read(State(web): State<Arc<Web>>, Query(p): Query<ReadParams>) -> HttpResponse {
    let Ok(session) = polylogue_model::SessionId::parse(&p.q) else {
        return (StatusCode::BAD_REQUEST, "not a session ID").into_response();
    };
    let request = Request::Read {
        session,
        view: p.view.unwrap_or_else(|| "full".into()),
        format: p.format.unwrap_or_else(|| "json".into()),
        logical: p.logical,
        budget: p.budget,
        cursor: p.cursor,
    };
    json_answer(&web, ask(&web, request).await)
}

/// A note written from the reader. The request ID makes a resent POST a
/// no-op; the reader writes as the operator.
#[derive(Deserialize)]
struct NoteBody {
    request_id: String,
    note: polylogue_model::NewNote,
}

async fn note(State(web): State<Arc<Web>>, axum::Json(b): axum::Json<NoteBody>) -> HttpResponse {
    let request = Request::AddNote {
        request_id: b.request_id,
        note: b.note,
    };
    json_answer(&web, ask(&web, request).await)
}

#[derive(Deserialize)]
struct WithdrawBody {
    request_id: String,
    note: String,
    reason: Option<String>,
}

async fn withdraw(
    State(web): State<Arc<Web>>,
    axum::Json(b): axum::Json<WithdrawBody>,
) -> HttpResponse {
    let request = Request::Withdraw {
        request_id: b.request_id,
        note: b.note,
        author: polylogue_model::Author::Operator,
        reason: b.reason,
    };
    json_answer(&web, ask(&web, request).await)
}

#[derive(Deserialize)]
struct CommandParams {
    key: Option<String>,
    statement: Option<String>,
    cwd: Option<std::path::PathBuf>,
    limit: Option<u64>,
    session: Option<String>,
}

async fn command_get(
    State(web): State<Arc<Web>>,
    UrlPath(command): UrlPath<String>,
    Query(p): Query<CommandParams>,
) -> HttpResponse {
    let request = match command.as_str() {
        "status" => Request::Status,
        "notes" => Request::Notes {
            filter: NoteFilter::default(),
        },
        "sql" => match p.statement {
            Some(statement) => Request::Sql { statement },
            None => return (StatusCode::BAD_REQUEST, "`statement` is required").into_response(),
        },
        "setting" => match p.key {
            Some(key) => Request::Setting {
                request_id: None,
                key,
                value: None,
                machine: false,
            },
            None => return (StatusCode::BAD_REQUEST, "`key` is required").into_response(),
        },
        "candidates" => match p.cwd {
            Some(cwd) => Request::Candidates {
                cwd,
                limit: p.limit,
            },
            None => return (StatusCode::BAD_REQUEST, "`cwd` is required").into_response(),
        },
        "continue" => match p.session.as_deref().map(polylogue_model::SessionId::parse) {
            Some(Ok(session)) => Request::Continue { session },
            _ => return (StatusCode::BAD_REQUEST, "`session` is required").into_response(),
        },
        "excise" => {
            return (StatusCode::FORBIDDEN, "excision is not available over HTTP").into_response();
        }
        _ => return (StatusCode::NOT_FOUND, "no such command").into_response(),
    };
    json_answer(&web, ask(&web, request).await)
}

#[derive(Deserialize)]
struct SettingBody {
    request_id: Option<String>,
    key: String,
    value: String,
    #[serde(default)]
    machine: bool,
}

async fn command_post(
    State(web): State<Arc<Web>>,
    UrlPath(command): UrlPath<String>,
    body: String,
) -> HttpResponse {
    match command.as_str() {
        "setting" => {
            let Ok(b) = serde_json::from_str::<SettingBody>(&body) else {
                return (StatusCode::BAD_REQUEST, "expected {key, value}").into_response();
            };
            let request = Request::Setting {
                request_id: b.request_id,
                key: b.key,
                value: Some(b.value),
                machine: b.machine,
            };
            json_answer(&web, ask(&web, request).await)
        }
        "excise" => (StatusCode::FORBIDDEN, "excision is not available over HTTP").into_response(),
        _ => (StatusCode::NOT_FOUND, "no such command").into_response(),
    }
}
