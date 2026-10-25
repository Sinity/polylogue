//! The reader's JSON API under `/api/v1/`: the
//! same data as `--json`, one route per command the table offers over
//! HTTP, at its words joined by `/` (`/api/v1/note/add`). A command that
//! writes is a POST with a JSON object body; one that reads is a GET with
//! its arguments as query parameters (a list argument repeats its name).

use super::{Web, ask, json_answer};
use crate::commands::{Action, COMMANDS, Caller, Command, Surface, Values};
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response as HttpResponse};
use axum::routing::{get, post};
use axum::Router;
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
    // A request is as large as what it carries.
    router.layer(axum::extract::DefaultBodyLimit::disable())
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
