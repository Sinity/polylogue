//! The reader's JSON API under `/api/v1/` (spec `surfaces/web.md`): the
//! same data as `--json`, one route per command the table offers over
//! HTTP, at its words joined by `/` (`/api/v1/note/add`). A command that
//! writes is a POST with a JSON object body; one that reads is a GET with
//! its arguments as query parameters (a list argument repeats its name).
//! The browser's chat actions are the only other routes. Other programs
//! use the socket.

use super::{Web, ask, json_answer};
use crate::commands::{Action, COMMANDS, Caller, Command, Surface, Values};
use crate::daemon::Request;
use axum::Router;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response as HttpResponse};
use axum::routing::{get, post};
use std::sync::Arc;

/// Routes that are not table commands: the operator's chat actions, which
/// the browser host carries out rather than the daemon.
pub const BROWSER_ROUTES: &[&str] = &["/browser-actions", "/browser-actions/attachments"];

/// The commands offered over HTTP.
pub fn http_commands() -> impl Iterator<Item = &'static Command> {
    COMMANDS
        .iter()
        .filter(|c| c.http && matches!(c.action, Action::Request(_)))
}

pub fn routes() -> Router<Arc<Web>> {
    let mut router = Router::new()
        .route(BROWSER_ROUTES[0], post(browser_action))
        .route(BROWSER_ROUTES[1], post(browser_attachment));
    for c in http_commands() {
        router = if c.writes {
            router.route(
                &c.http_path(),
                post(
                    move |State(web): State<Arc<Web>>, body: String| async move {
                        let object = if body.trim().is_empty() {
                            Ok(serde_json::Map::new())
                        } else {
                            serde_json::from_str(&body)
                                .map_err(|e| format!("the body is not a JSON object: {e}"))
                        };
                        let values = object.and_then(|o| Values::from_json(c, Surface::Http, o));
                        answer(&web, c, values).await
                    },
                ),
            )
        } else {
            router.route(
                &c.http_path(),
                get(
                    move |State(web): State<Arc<Web>>,
                          Query(pairs): Query<Vec<(String, String)>>| async move {
                        answer(&web, c, Values::from_pairs(c, Surface::Http, pairs)).await
                    },
                ),
            )
        };
    }
    router
}

/// Builds the command's request and answers with the daemon's.
async fn answer(
    web: &Arc<Web>,
    c: &'static Command,
    values: Result<Values, String>,
) -> HttpResponse {
    let Action::Request(build) = c.action else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let caller = Caller {
        surface: Surface::Http,
        cwd: None,
        json: true,
    };
    match values.and_then(|v| build(c, &v, &caller)) {
        Ok(request) => json_answer(web, ask(web, request).await),
        Err(message) => (
            StatusCode::BAD_REQUEST,
            axum::Json(serde_json::json!({ "error": "bad_request", "message": message })),
        )
            .into_response(),
    }
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
