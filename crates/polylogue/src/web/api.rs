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
    let Action::Request(build) = c.action else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let caller = Caller {
        surface: Surface::Http,
        cwd: None,
        json: true,
    };
    match values.and_then(|v| build(c, &v, &caller)) {
        Ok(crate::daemon::Request::Changes { since }) => changes(web, since).await,
        Ok(request) => json_answer(web, ask(web, request).await),
        Err(message) => json_answer(web, crate::daemon::service::error("bad_request", message)),
    }
}

/// Stream a prepared finite batch after its reader has been released. The
/// spool belongs to the response, so a slow or disconnected client owns no
/// database transaction, reader slot or detached producer.
async fn changes(web: &Arc<Web>, since: Option<String>) -> HttpResponse {
    let web = Arc::clone(web);
    let worker = Arc::clone(&web);
    let result = tokio::task::spawn_blocking(move || {
        let file = crate::daemon::service::change_feed(&worker.daemon, since.as_deref())?;
        scrub_changes(&worker, file)
    })
    .await;
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

fn scrub_changes(web: &Web, file: std::fs::File) -> Result<std::fs::File, crate::daemon::Response> {
    use std::io::{BufRead, Seek, SeekFrom, Write};
    let failed = |error: std::io::Error| crate::daemon::service::error("io", error);
    let mut output = tempfile::tempfile_in(&web.daemon.archive.root).map_err(failed)?;
    for line in std::io::BufReader::new(file).lines() {
        let line = line.map_err(failed)?;
        let response: crate::daemon::Response =
            serde_json::from_str(&line).map_err(|e| failed(std::io::Error::other(e)))?;
        let crate::daemon::Response::Ok { data } = response else {
            return Err(response);
        };
        let response = crate::daemon::Response::Ok {
            data: super::scrubbed(web, data),
        };
        serde_json::to_writer(&mut output, &response)
            .map_err(|e| failed(std::io::Error::other(e)))?;
        output.write_all(b"\n").map_err(failed)?;
    }
    output.flush().map_err(failed)?;
    output.seek(SeekFrom::Start(0)).map_err(failed)?;
    Ok(output)
}

#[cfg(test)]
mod tests;
