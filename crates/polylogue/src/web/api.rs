//! The reader's JSON API under `/api/v1/`: the
//! same data as `--json`, one route per command the table offers over
//! HTTP, at its words joined by `/` (`/api/v1/note/add`). A command that
//! writes is a POST with a JSON object body; one that reads is a GET with
//! its arguments as query parameters (a list argument repeats its name).

use super::{Web, json_answer};
use crate::commands::{COMMANDS, Caller, Command, Run, Surface};
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
                        answer(&web, c, body_request(c, &body), &headers).await
                    },
                ),
            )
        } else {
            router.route(
                &c.http_path(),
                get(
                    move |State(web): State<Arc<Web>>,
                          Query(pairs): Query<Vec<(String, String)>>,
                          headers: HeaderMap| async move {
                        let request = match c.run {
                            // A stream the shared declarations do not hold
                            // takes its pairs as text arguments.
                            Run::Feed { .. } => Ok(crate::client::request(
                                c.name,
                                serde_json::Value::Object(
                                    pairs
                                        .into_iter()
                                        .map(|(key, value)| (key, serde_json::Value::String(value)))
                                        .collect(),
                                ),
                                polylogue_store::user_writes::new_request_id(),
                            )),
                            _ => crate::commands::declarations()
                                .parse_http_query(
                                    c.name,
                                    &pairs,
                                    polylogue_store::user_writes::new_request_id(),
                                )
                                .map_err(|failure| failure.message),
                        };
                        answer(&web, c, request, &headers).await
                    },
                ),
            )
        };
    }
    // A request is as large as what it carries.
    router.layer(axum::extract::DefaultBodyLimit::disable())
}

/// A POST body: the whole request, or the operation's arguments with an
/// optional `request_id` beside them.
fn body_request(c: &Command, body: &str) -> Result<command_surface::RunnableRequest, String> {
    let mut object: serde_json::Map<String, serde_json::Value> = if body.trim().is_empty() {
        serde_json::Map::new()
    } else {
        serde_json::from_str(body).map_err(|e| format!("the body is not a JSON object: {e}"))?
    };
    if object.contains_key("operation") {
        return serde_json::from_value(serde_json::Value::Object(object))
            .map_err(|error| error.to_string());
    }
    let request_id = match object.remove("request_id") {
        Some(serde_json::Value::String(id)) => id,
        Some(_) => return Err("request_id must be text".into()),
        None => polylogue_store::user_writes::new_request_id(),
    };
    Ok(crate::client::request(
        c.name,
        serde_json::Value::Object(object),
        request_id,
    ))
}

/// Answers one request through the row's handler.
async fn answer(
    web: &Arc<Web>,
    c: &'static Command,
    request: Result<command_surface::RunnableRequest, String>,
    headers: &HeaderMap,
) -> HttpResponse {
    let caller = Caller {
        surface: Surface::Http,
        cwd: None,
    };
    let mut request = match request {
        Ok(request) if request.operation == c.name => request,
        Ok(_) => {
            return json_answer(
                web,
                crate::daemon::service::error("bad_request", "route does not match operation"),
            );
        }
        Err(error) => {
            return json_answer(web, crate::daemon::service::error("bad_request", error));
        }
    };
    let parent_deadline = super::parent_deadline();
    let stream_limit = |request: &command_surface::RunnableRequest, default_deadline: bool| {
        let declared = crate::commands::deadlines().get(c.latency);
        let duration = match request.deadline {
            command_surface::Deadline::Default if !default_deadline => None,
            command_surface::Deadline::Default if declared.is_none() => {
                Some(crate::daemon::execution::timeout())
            }
            deadline => deadline.duration(declared),
        };
        let logical = duration.and_then(|duration| std::time::Instant::now().checked_add(duration));
        let cancellation = super::request_limit().cancellation;
        let deadline = match (parent_deadline, logical) {
            (Some(parent), Some(logical)) => Some(parent.min(logical)),
            (parent, None) => parent,
            (None, logical) => logical,
        };
        match deadline {
            Some(deadline) => polylogue_store::read::limit::ReadLimit::at(deadline, cancellation),
            None => polylogue_store::read::limit::ReadLimit::without_deadline(cancellation),
        }
    };
    let registration = match c.run {
        Run::Daemon(registration) | Run::Delivered { registration, .. } => registration,
        Run::Feed { registration, .. } => {
            let limit = stream_limit(&request, true);
            let arguments = request.arguments.clone();
            return stream(web, limit, request, move |d, _, context, output| {
                registration.produce(&arguments, d, context, output)
            })
            .await;
        }
        Run::Local(_) | Run::ConfiguredLocal(_) => return StatusCode::NOT_FOUND.into_response(),
    };
    let policy = match registration.admit(&request.arguments, c, &caller) {
        Ok(policy) => policy,
        Err(failure) => return json_answer(web, failure.answer::<serde_json::Value>()),
    };
    if !policy.read_phase {
        request.deadline = command_surface::Deadline::None;
    }
    if policy.streaming {
        let limit = stream_limit(&request, policy.default_deadline);
        let arguments = request.arguments.clone();
        return stream(
            web,
            limit,
            request,
            move |d, invocation, context, output| {
                registration.produce(&arguments, c, d, invocation, &caller, context, output)
            },
        )
        .await;
    }
    let read_scope = if policy.read_only {
        match web.daemon.reads.scope_token() {
            Ok(scope) => Some(scope),
            Err(error) => return json_answer(web, crate::daemon::service::store_error(&error)),
        }
    } else {
        None
    };
    let table = &web.operations;
    let limit = super::request_limit();
    let mut accepted = crate::protocol::caller::accept(
        &web.daemon.callers,
        &Default::default(),
        polylogue_model::authorship::Channel::Web,
        limit.cancellation,
    );
    accepted.parent = super::parent_lifetime();
    accepted.read_scope = read_scope.clone();
    accepted.invocation = Some(
        web.daemon
            .invocation(&accepted.authorship)
            .with_accepted_request(Some(Arc::new(request.clone())))
            .with_parent(accepted.parent.clone()),
    );
    let scope = accepted.scope();
    let trusted = command_surface::Caller::authenticated(scope, accepted)
        .with_access(command_surface::Access::Write);
    let dispatched = table
        .call_as(
            request.clone(),
            command_surface::Via::Web,
            trusted,
            Default::default(),
        )
        .await;
    let publication = crate::protocol::ReadScopes::single(read_scope.clone());
    let answer = crate::protocol::scope_answer(dispatched, &publication);
    if c.name == "read"
        && headers
            .get(header::ACCEPT)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| {
                value.split(',').any(|kind| {
                    kind.split(';')
                        .next()
                        .is_some_and(|kind| kind.trim() == "text/html")
                })
            })
        && let Some(value) = answer.value()
    {
        let page = serde_json::from_value::<polylogue_model::read::ReadPage>(value.clone());
        match page
            .map_err(|error| {
                command_surface::Failure::new(command_surface::FailureKind::Io, error.to_string())
            })
            .and_then(|page| super::pages::reader::read(&page, table, &request))
        {
            Ok(html) => {
                if let Err(error) = publication.check() {
                    return json_answer(web, crate::daemon::service::store_error(&error));
                }
                return scoped_body_response(
                    web,
                    StatusCode::OK,
                    "text/html; charset=utf-8",
                    html.into_bytes(),
                    &publication,
                );
            }
            Err(error) => {
                return json_answer(
                    web,
                    command_surface::Answer::<serde_json::Value>::error(error),
                );
            }
        }
    }
    let (status, bytes) = super::prepare_json_answer(web, answer);
    scoped_body_response(web, status, "application/json", bytes, &publication)
}

struct ScopedBytes {
    bytes: Vec<u8>,
    offset: usize,
    scopes: crate::protocol::ReadScopes,
}
impl tokio::io::AsyncRead for ScopedBytes {
    fn poll_read(
        self: std::pin::Pin<&mut Self>,
        _: &mut std::task::Context<'_>,
        buffer: &mut tokio::io::ReadBuf<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        let this = self.get_mut();
        let scopes = this.scopes.clone();
        std::task::Poll::Ready(
            scopes
                .with_current(|| {
                    let end = this
                        .bytes
                        .len()
                        .min(this.offset.saturating_add(buffer.remaining()));
                    buffer.put_slice(&this.bytes[this.offset..end]);
                    this.offset = end;
                    Ok(())
                })
                .map_err(std::io::Error::other),
        )
    }
}

/// Encode before this boundary, then fence each bounded body-frame activation.
fn scoped_body_response(
    web: &Web,
    status: StatusCode,
    media_type: &'static str,
    bytes: Vec<u8>,
    scopes: &crate::protocol::ReadScopes,
) -> HttpResponse {
    let length = bytes.len();
    if let Err(error) = scopes.check() {
        return json_answer(web, crate::daemon::service::store_error(&error));
    }
    let body = if scopes.has_scope() {
        axum::body::Body::from_stream(tokio_util::io::ReaderStream::new(ScopedBytes {
            bytes,
            offset: 0,
            scopes: scopes.clone(),
        }))
    } else {
        axum::body::Body::from(bytes)
    };
    let mut response = (status, body).into_response();
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        axum::http::HeaderValue::from_static(media_type),
    );
    response.headers_mut().insert(
        header::CONTENT_LENGTH,
        length.to_string().parse().expect("numeric content length"),
    );
    response
}

/// Headers are committed on the first produced bytes. Subsequent failures
/// terminate the stream with an error event, without completion.
async fn stream(
    web: &Arc<Web>,
    limit: polylogue_store::read::limit::ReadLimit,
    accepted_request: command_surface::RunnableRequest,
    produce: impl FnOnce(
        &crate::daemon::service::Daemon,
        &crate::operation::Invocation,
        &Arc<crate::operation::Context>,
        &mut dyn std::io::Write,
    ) -> Result<(), crate::daemon::Failure>
    + Send
    + 'static,
) -> HttpResponse {
    let scope = match web.daemon.reads.scope_token() {
        Ok(scope) => scope,
        Err(error) => return json_answer(web, crate::daemon::service::store_error(&error)),
    };
    let worker = Arc::clone(web);
    let cancellation = limit.cancellation.clone();
    let accepted = crate::protocol::caller::accept(
        &web.daemon.callers,
        &Default::default(),
        polylogue_model::authorship::Channel::Web,
        cancellation.clone(),
    );
    let invocation = web
        .daemon
        .invocation(&accepted.authorship)
        .with_accepted_request(Some(Arc::new(accepted_request)))
        .with_parent(super::parent_lifetime());
    let (output, input) = tokio::io::duplex(64 * 1024);
    let (ready, first) = tokio::sync::oneshot::channel();
    let runtime = tokio::runtime::Handle::current();
    tokio::task::spawn_blocking(move || {
        let mut output = ProducerWriter {
            output,
            runtime,
            ready: Some(ready),
        };
        let mut guarded =
            crate::daemon::server::GuardedWriter::new(&mut output, limit.clone(), scope);
        let result = limit.scope(|| {
            let context = worker
                .daemon
                .resolve_operation_context(invocation.clone())?;
            produce(&worker.daemon, &invocation, &context, &mut guarded)
        });
        let result = match guarded.failure() {
            Some(failure) => Err(failure),
            None => result,
        };
        drop(guarded);
        match result {
            Err(failure) => {
                if let Some(ready) = output.ready.take() {
                    let _ = ready.send(Err(failure));
                } else {
                    let _ = crate::protocol::stream::error(&mut output, failure);
                }
            }
            Ok(()) => {
                if let Some(ready) = output.ready.take() {
                    let _ = ready.send(Err(crate::daemon::Failure::new(
                        "internal",
                        "stream completed without producing events",
                    )));
                }
            }
        }
    });
    let mut cancel = super::CancelRead(Some(cancellation.clone()));
    match first.await {
        Ok(Ok(())) => {
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
        Ok(Err(failure)) => json_answer(web, failure),
        Err(error) => json_answer(web, crate::daemon::service::error("internal", error)),
    }
}

struct ProducerWriter {
    output: tokio::io::DuplexStream,
    runtime: tokio::runtime::Handle,
    ready: Option<tokio::sync::oneshot::Sender<Result<(), crate::daemon::Failure>>>,
}
impl std::io::Write for ProducerWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        use tokio::io::AsyncWriteExt;
        if bytes.is_empty() {
            return Ok(0);
        }
        if let Some(ready) = self.ready.take() {
            ready
                .send(Ok(()))
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
