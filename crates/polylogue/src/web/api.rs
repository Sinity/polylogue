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
        if let Action::Stream { produce, .. } = c.action
            && let Some(build) = c.build_typed
        {
            match build(request.arguments.clone(), &caller, &request.request_id) {
                Ok(internal) if internal.streaming() => {
                    return stream(web, internal, produce).await;
                }
                Err(failure) => {
                    return json_answer(
                        web,
                        command_surface::Answer::<serde_json::Value>::error(failure),
                    );
                }
                _ => {}
            }
        }
        let read_scope = if !c.writes {
            match web.daemon.reads.scope_token() {
                Ok(scope) => Some(scope),
                Err(error) => return json_answer(web, crate::daemon::service::store_error(&error)),
            }
        } else {
            None
        };
        let table = &web.operations;
        let limit = super::request_limit();
        let accepted = crate::protocol::caller::accept(
            &web.daemon.callers,
            &Default::default(),
            polylogue_model::authorship::Channel::Web,
            limit.cancellation,
        );
        let scope = accepted.authorship.author.to_string();
        let trusted = command_surface::Caller::authenticated(scope, accepted);
        let answer = table
            .call_as(
                request,
                command_surface::Origin::Http,
                trusted,
                Default::default(),
            )
            .await;
        return json_answer(
            web,
            crate::protocol::scope_answer(answer, read_scope.as_ref()),
        );
    }
    match values.and_then(|v| build(c, &v, &caller)) {
        Ok(request) => match c.action {
            Action::Stream { produce, .. } => stream(web, request, produce).await,
            _ => json_answer(web, ask(web, request).await),
        },
        Err(message) => json_answer(web, crate::daemon::service::error("bad_request", message)),
    }
}

/// Headers are committed on the first produced bytes. Subsequent failures
/// terminate the stream with a canonical error event, without completion.
async fn stream(
    web: &Arc<Web>,
    request: crate::daemon::Request,
    produce: crate::commands::StreamProducer,
) -> HttpResponse {
    let worker = Arc::clone(web);
    let limit = super::request_limit();
    let cancellation = limit.cancellation.clone();
    let accepted = crate::protocol::caller::accept(
        &web.daemon.callers,
        &Default::default(),
        polylogue_model::authorship::Channel::Web,
        cancellation.clone(),
    );
    let (output, input) = tokio::io::duplex(64 * 1024);
    let (ready, first) = tokio::sync::oneshot::channel();
    let runtime = tokio::runtime::Handle::current();
    let worker_request = request.clone();
    tokio::task::spawn_blocking(move || {
        let mut output = ProducerWriter {
            output,
            runtime,
            ready: Some(ready),
        };
        let mut guarded = crate::protocol::stream::GuardedWriter::new(&mut output, limit.clone());
        let result = limit.scope(|| {
            produce(
                &worker_request,
                &worker.daemon,
                &accepted.authorship,
                &mut guarded,
            )
        });
        let result = match guarded.failure() {
            Some(failure) => Some(Err(failure)),
            None => result,
        };
        drop(guarded);
        let Some(result) = result else {
            if let Some(ready) = output.ready.take() {
                let _ = ready.send(Ok(false));
            }
            return;
        };
        match result {
            Err(response) => {
                if let Some(ready) = output.ready.take() {
                    let _ = ready.send(Err(response));
                } else {
                    let _ = crate::protocol::stream::error(&mut output, &response);
                }
            }
            Ok(()) => {
                if let Some(ready) = output.ready.take() {
                    let _ = ready.send(Err(crate::daemon::service::error(
                        "internal",
                        "stream completed without producing events",
                    )));
                }
            }
        }
    });
    let mut cancel = super::CancelRead(Some(cancellation.clone()));
    match first.await {
        Ok(Ok(true)) => {
            cancel.0.take();
            let input = StreamInput {
                input,
                cancellation,
            };
            (
                [
                    (header::CONTENT_TYPE, "text/event-stream"),
                    (header::CACHE_CONTROL, "no-cache"),
                ],
                axum::body::Body::from_stream(tokio_util::io::ReaderStream::new(input)),
            )
                .into_response()
        }
        Ok(Ok(false)) => json_answer(web, ask(web, request).await),
        Ok(Err(response)) => json_answer(web, response),
        Err(error) => json_answer(web, crate::daemon::service::error("internal", error)),
    }
}

struct ProducerWriter {
    output: tokio::io::DuplexStream,
    runtime: tokio::runtime::Handle,
    ready: Option<tokio::sync::oneshot::Sender<Result<bool, crate::daemon::Response>>>,
}
impl std::io::Write for ProducerWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        use tokio::io::AsyncWriteExt;
        if bytes.is_empty() {
            return Ok(0);
        }
        if let Some(ready) = self.ready.take() {
            ready
                .send(Ok(true))
                .map_err(|_| std::io::Error::from(std::io::ErrorKind::BrokenPipe))?;
        }
        self.runtime.block_on(async {
            tokio::time::timeout(
                std::time::Duration::from_millis(10),
                self.output.write(bytes),
            )
            .await
            .map_err(|_| std::io::Error::from(std::io::ErrorKind::WouldBlock))?
        })
    }
    fn flush(&mut self) -> std::io::Result<()> {
        use tokio::io::AsyncWriteExt;
        self.runtime.block_on(self.output.flush())
    }
}

struct StreamInput {
    input: tokio::io::DuplexStream,
    cancellation: polylogue_store::read::limit::ReadCancellation,
}
impl tokio::io::AsyncRead for StreamInput {
    fn poll_read(
        mut self: std::pin::Pin<&mut Self>,
        context: &mut std::task::Context<'_>,
        buffer: &mut tokio::io::ReadBuf<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        std::pin::Pin::new(&mut self.input).poll_read(context, buffer)
    }
}
impl Drop for StreamInput {
    fn drop(&mut self) {
        self.cancellation.cancel();
    }
}

#[cfg(test)]
mod tests;
