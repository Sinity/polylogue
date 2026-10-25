//! The reader's JSON API under `/api/v1/`: the
//! same data as `--json`, one route per command the table offers over
//! HTTP, at its words joined by `/` (`/api/v1/note/add`). A command that
//! writes is a POST with a JSON object body; one that reads is a GET with
//! its arguments as query parameters (a list argument repeats its name).
//! Binary attachment inputs stay binary until their in-process daemon request.

use super::{Web, ask, json_answer};
use crate::commands::{Action, COMMANDS, Caller, Command, Surface, Values};
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response as HttpResponse};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::Value;
use std::sync::Arc;

/// The commands offered over HTTP.
pub fn http_commands() -> impl Iterator<Item = &'static Command> {
    COMMANDS
        .iter()
        .filter(|c| c.http && !matches!(c.action, Action::Local(_)))
}

pub fn routes() -> Router<Arc<Web>> {
    let mut router = Router::new();
    for c in http_commands() {
        match c.action {
            Action::JsonRequest(build) => {
                router = router.route(
                    &c.http_path(),
                    post(
                        move |State(web): State<Arc<Web>>, Json(value): Json<Value>| async move {
                            json_body(web, build, value).await
                        },
                    )
                    .layer(axum::extract::DefaultBodyLimit::max(
                        crate::browser::actions::BODY_LIMIT,
                    )),
                );
                continue;
            }
            Action::BytesRequest(build) => {
                router = router.route(
                    &c.http_path(),
                    post(
                        move |State(web): State<Arc<Web>>, body: axum::body::Bytes| async move {
                            browser_answer(&web, ask(&web, build(body.to_vec())).await)
                        },
                    )
                    .layer(axum::extract::DefaultBodyLimit::max(
                        crate::browser::actions::BODY_LIMIT,
                    )),
                );
                continue;
            }
            _ => {}
        }
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

async fn json_body(
    web: Arc<Web>,
    build: fn(serde_json::Value) -> Result<crate::daemon::Request, String>,
    value: serde_json::Value,
) -> HttpResponse {
    match build(value) {
        Ok(request) => browser_answer(&web, ask(&web, request).await),
        Err(message) => (
            StatusCode::UNPROCESSABLE_ENTITY,
            json_answer(&web, crate::daemon::service::error("bad_request", message)),
        )
            .into_response(),
    }
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
        Err(message) => json_answer(web, crate::daemon::service::error("bad_request", message)),
    }
}

/// Keeps the existing browser response envelope; action decisions live in the daemon.
fn browser_answer(web: &super::Web, response: crate::daemon::Response) -> HttpResponse {
    match response {
        crate::daemon::Response::Ok { mut data } => {
            let created = data
                .as_object_mut()
                .and_then(|data| data.remove("created"))
                .is_some_and(|created| created == true);
            (
                if created {
                    StatusCode::CREATED
                } else {
                    StatusCode::OK
                },
                axum::Json(super::scrubbed(web, data)),
            )
                .into_response()
        }
        crate::daemon::Response::Error {
            kind,
            detail: Some(detail),
            ..
        } if detail["http_status"].as_u64().is_some() => (
            StatusCode::from_u16(detail["http_status"].as_u64().unwrap_or(500) as u16)
                .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            axum::Json(serde_json::json!({ "ok": false, "error": kind })),
        )
            .into_response(),
        other => json_answer(web, other),
    }
}
