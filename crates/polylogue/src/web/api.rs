//! The reader's JSON API under `/api/v1/`: the
//! same data as `--json`, one route per command the table offers over
//! HTTP, at its words joined by `/` (`/api/v1/note/add`). A command that
//! writes is a POST with a JSON object body; one that reads is a GET with
//! its arguments as query parameters (a list argument repeats its name).

use super::{Web, ask, json_answer};
use crate::commands::{Action, COMMANDS, Caller, Command, Surface, Values};
use axum::Router;
use axum::extract::{Query, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response as HttpResponse};
use axum::routing::{get, post};
use std::sync::Arc;

/// The commands offered over HTTP.
pub fn http_commands() -> impl Iterator<Item = &'static Command> {
    COMMANDS.iter().filter(|c| c.http)
}

pub fn routes() -> Router<Arc<Web>> {
    let mut router = Router::new();
    for c in http_commands() {
        router = if c.writes {
            router.route(
                &c.http_path(),
                post(
                    move |State(web): State<Arc<Web>>, headers: HeaderMap, body: String| async move {
                        if !super::access::content_type(&headers,"application/json") {
                            return (StatusCode::UNSUPPORTED_MEDIA_TYPE, "command writes require application/json").into_response();
                        }
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
    // A request is as large as what it carries.
    router.layer(axum::extract::DefaultBodyLimit::disable())
}

/// Builds the command's request and answers with the daemon's.
async fn answer(
    web: &Arc<Web>,
    c: &'static Command,
    values: Result<Values, String>,
) -> HttpResponse {
    let build = match c.action {
        Action::Request(build) | Action::Stream { build, .. } => build,
        Action::Local(_) => return StatusCode::NOT_FOUND.into_response(),
    };
    let caller = Caller {
        surface: Surface::Http,
        cwd: None,
        json: true,
    };
    if c.register.is_some() {
        let values = match values {
            Ok(values) => values,
            Err(error) => {
                return json_answer(web, crate::daemon::service::error("bad_request", error));
            }
        };
        let request_id = values.request_id();
        let deadline = super::READ_LIMIT
            .try_with(|(deadline, _)| {
                deadline.map_or(command_surface::Deadline::Default, |deadline| {
                    command_surface::Deadline::Milliseconds(
                        crate::daemon::execution::remaining(Some(deadline))
                            .as_millis()
                            .try_into()
                            .unwrap_or(u64::MAX),
                    )
                })
            })
            .unwrap_or_default();
        let request = crate::protocol::operations::request(c, values, request_id, deadline);
        let table = crate::protocol::operations::table(Arc::clone(&web.daemon));
        let limit = super::request_limit();
        let trusted = command_surface::Caller::authenticated("local", limit.cancellation);
        let answer = table
            .call_as(
                request,
                command_surface::Origin::Http,
                trusted,
                Default::default(),
            )
            .await;
        return json_answer(web, answer.into());
    }
    match values.and_then(|v| build(c, &v, &caller)) {
        Ok(request) => match c.action {
            Action::Stream { prepare, .. } => stream(web, request, prepare).await,
            _ => json_answer(web, ask(web, request).await),
        },
        Err(message) => json_answer(web, crate::daemon::service::error("bad_request", message)),
    }
}

/// Stream a prepared finite batch after its reader has been released. The
/// spool belongs to the response, so a slow or disconnected client owns no
/// database transaction, reader slot or detached producer.
async fn stream(
    web: &Arc<Web>,
    request: crate::daemon::Request,
    prepare: crate::commands::StreamHandler,
) -> HttpResponse {
    let web = Arc::clone(web);
    let worker = Arc::clone(&web);
    let limit = super::request_limit();
    let mut cancel = super::CancelRead(Some(limit.cancellation.clone()));
    let result = tokio::task::spawn_blocking(move || {
        limit.scope(|| {
            let result = prepare(&request, &worker.daemon, Some(super::access::scrub))
                .expect("command owns its request");
            limit.check().map_err(|e| {
                crate::daemon::execution::refusal(&e, &serde_json::json!({"op":"changes"}), &limit)
            })?;
            result
        })
    })
    .await;
    cancel.0.take();
    match result {
        Ok(Ok(file)) => {
            let stream = tokio_util::io::ReaderStream::new(tokio::fs::File::from_std(file));
            (
                [(header::CONTENT_TYPE, "application/x-ndjson")],
                axum::body::Body::from_stream(stream),
            )
                .into_response()
        }
        Ok(Err(response)) => json_answer(&web, response),
        Err(error) => json_answer(&web, crate::daemon::service::error("internal", error)),
    }
}

#[cfg(test)]
mod tests;
