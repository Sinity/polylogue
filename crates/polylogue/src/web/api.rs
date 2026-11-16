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
use std::sync::Arc;

/// The commands offered over HTTP.
pub fn http_commands() -> impl Iterator<Item = &'static Command> {
    COMMANDS.iter().filter(|c| c.http)
}

pub fn routes() -> Router<Arc<Web>> {
    let mut router = Router::new();
    for c in http_commands() {
        let mut methods = axum::routing::MethodRouter::new();
        if c.writes {
            methods = methods.post(
                move |State(web): State<Arc<Web>>, headers: HeaderMap, body: String| async move {
                    if !super::access::content_type(&headers, "application/json") {
                        return (
                            StatusCode::UNSUPPORTED_MEDIA_TYPE,
                            "command writes require application/json",
                        )
                            .into_response();
                    }
                    answer_as(&web, c, body_request(c, &body), &headers).await
                },
            );
        }
        if !c.writes || matches!(c.name, "rows" | "set") {
            methods = methods.get(
                move |State(web): State<Arc<Web>>,
                      Query(pairs): Query<Vec<(String, String)>>,
                      headers: HeaderMap| async move {
                    let request = query_request(c, &pairs).and_then(|request| {
                        if c.name == "set"
                            && c.registration()
                                .expect("set handler")
                                .tool_access(&request.arguments, c)
                                .map_err(|failure| failure.error.message)?
                                > crate::commands::ToolAccess::Read
                        {
                            return Err("use POST /api/v1/set".into());
                        }
                        Ok(HttpCall {
                            request,
                            caller: Default::default(),
                        })
                    });
                    answer_as(&web, c, request, &headers).await
                },
            );
        }
        router = router.route(&c.http_path(), methods);
    }
    router.layer(axum::extract::DefaultBodyLimit::disable())
}

struct HttpCall {
    request: command_surface::RunnableRequest,
    caller: command_surface::CallerEvidence,
}

/// A POST takes the shared request or the operation's arguments. The calling
/// session belongs to the transport, outside those arguments.
fn body_request(c: &Command, body: &str) -> Result<HttpCall, String> {
    use serde_json::{Map, Value, json};
    let mut object: Map<String, Value> = if body.trim().is_empty() {
        Map::new()
    } else {
        serde_json::from_str(body)
            .map_err(|error| format!("the body is not a JSON object: {error}"))?
    };
    let mut caller = command_surface::CallerEvidence::default();
    if let Some(session) = object.remove("session") {
        let session = session.as_str().ok_or("session must be text")?;
        caller.session = Some(
            command_surface::AgentSession::parse(session)
                .ok_or("session must name a harness and its session")?,
        );
    }
    if !object.contains_key("arguments") {
        let mut call = Map::new();
        for name in [
            "operation",
            "request_id",
            "deadline",
            "budget_tokens",
            "cursor",
            "fresh",
            "format",
        ] {
            if let Some(value) = object.remove(name) {
                call.insert(name.into(), value);
            }
        }
        call.insert("arguments".into(), Value::Object(object));
        object = call;
    }
    object.entry("operation").or_insert_with(|| json!(c.name));
    let author = command_surface::author_of(&caller, command_surface::Via::Web, None).author;
    let request = crate::commands::declarations()
        .parse_http_call(Value::Object(object), &author)
        .map_err(|failure| failure.message)?;
    Ok(HttpCall { request, caller })
}

/// GET carries a listing or one read step. Each step's values use the same
/// query decoder as command arguments, with its own typed declaration.
fn query_request(
    c: &Command,
    pairs: &[(String, String)],
) -> Result<command_surface::RunnableRequest, String> {
    let declarations = crate::commands::declarations();
    let id = polylogue_store::user_writes::new_request_id();
    if c.name != "rows" {
        return declarations
            .parse_http_query(c.name, pairs, id)
            .map_err(|failure| failure.message);
    }
    let verbs: Vec<_> = pairs
        .iter()
        .filter(|(name, _)| name == "steps")
        .map(|(_, value)| value.as_str())
        .collect();
    if verbs.is_empty() {
        let request = declarations
            .parse_http_query("rows", pairs, id)
            .map_err(|failure| failure.message)?;
        check_get_rows(&request)?;
        return Ok(request);
    }
    let [verb @ ("read" | "brief")] = verbs.as_slice() else {
        return Err("use POST /api/v1/rows".into());
    };
    let fields = declarations
        .descriptions()
        .iter()
        .find(|row| row.spec.name == "rows")
        .expect("rows declaration");
    let mut line_pairs = Vec::new();
    let mut step_pairs = Vec::new();
    for (name, value) in pairs {
        if name == "steps" {
            continue;
        }
        if fields.spec.argument(name).is_some()
            || matches!(
                name.as_str(),
                "request_id" | "deadline" | "budget_tokens" | "fresh" | "operation" | "arguments"
            )
        {
            line_pairs.push((name.clone(), value.clone()));
        } else {
            step_pairs.push((name.clone(), value.clone()));
        }
    }
    let step = step_query_declarations()
        .parse_http_query(verb, &step_pairs, "http-step")
        .map_err(|failure| failure.message)?;
    let cursor = step.cursor;
    let mut step = step.arguments;
    if let Some(cursor) = cursor {
        step.as_object_mut()
            .expect("step arguments")
            .insert("cursor".into(), cursor.into());
    }
    step.as_object_mut()
        .expect("step arguments")
        .insert("verb".into(), (*verb).into());
    let mut request = declarations
        .parse_http_query("rows", &line_pairs, id)
        .map_err(|failure| failure.message)?;
    request
        .arguments
        .as_object_mut()
        .expect("rows arguments")
        .insert("steps".into(), serde_json::json!([step]));
    check_get_rows(&request)?;
    Ok(request)
}

fn check_get_rows(request: &command_surface::RunnableRequest) -> Result<(), String> {
    let rows: crate::protocol::line::Rows =
        serde_json::from_value(request.arguments.clone()).map_err(|error| error.to_string())?;
    if rows.steps.len() > 1
        || rows.steps.iter().any(|step| {
            !matches!(
                step,
                crate::protocol::line::Step::Read(_) | crate::protocol::line::Step::Brief(_)
            )
        })
    {
        return Err("use POST /api/v1/rows".into());
    }
    Ok(())
}

fn step_query_declarations() -> &'static command_surface::OperationDeclarations {
    static STEPS: std::sync::LazyLock<std::sync::Arc<command_surface::OperationDeclarations>> =
        std::sync::LazyLock::new(|| {
            fn declare<A: command_surface::JsonSchema>(
                builder: &mut command_surface::DeclarationsBuilder,
                name: &'static str,
            ) {
                let row = Command {
                    name,
                    ..*crate::commands::command("rows").expect("rows command")
                };
                let mut spec = crate::commands::specification::<A>(&row);
                spec.kind = command_surface::OperationKind::Request { tool: None };
                // These declarations only decode HTTP query values; step fields
                // do not consume the command line's budget or cursor options.
                for argument in &mut spec.arguments {
                    if matches!(argument.name.as_str(), "budget" | "cursor") {
                        argument.form = command_surface::ArgumentForm::Positional;
                    }
                }
                builder
                    .declare::<A, serde_json::Value>(
                        spec,
                        crate::cli::print::json::<serde_json::Value>,
                    )
                    .expect("read step query declaration");
            }
            let mut builder = command_surface::OperationDeclarations::builder(
                "polylogue",
                crate::commands::deadlines(),
            );
            declare::<crate::protocol::steps::ReadStep>(&mut builder, "read");
            declare::<crate::protocol::steps::BriefStep>(&mut builder, "brief");
            builder.finish()
        });
    &STEPS
}

#[cfg(test)]
async fn answer(
    web: &Arc<Web>,
    c: &'static Command,
    request: Result<command_surface::RunnableRequest, String>,
    headers: &HeaderMap,
) -> HttpResponse {
    answer_as(
        web,
        c,
        request.map(|request| HttpCall {
            request,
            caller: Default::default(),
        }),
        headers,
    )
    .await
}

/// Answers one request through the row's handler.
async fn answer_as(
    web: &Arc<Web>,
    c: &'static Command,
    request: Result<HttpCall, String>,
    headers: &HeaderMap,
) -> HttpResponse {
    let caller = Caller {
        surface: Surface::Http,
        cwd: None,
    };
    let HttpCall {
        mut request,
        caller: details,
    } = match request {
        Ok(call) if call.request.operation == c.name => call,
        Ok(_) => {
            return json_answer(
                web,
                crate::daemon::service::error(
                    polylogue_model::Code::BadRequest,
                    "route does not match operation",
                ),
            );
        }
        Err(error) => {
            return json_answer(
                web,
                crate::daemon::service::error(polylogue_model::Code::BadRequest, error),
            );
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
            if let Err(failure) = registration.admit(&request) {
                return json_answer(web, failure.answer::<serde_json::Value>());
            }
            let limit = stream_limit(&request, true);
            let arguments = request.arguments.clone();
            return stream_as(
                web,
                crate::daemon::executor::Class::Ordinary,
                limit,
                request,
                details.clone(),
                Box::new(move |d, _, context, output| {
                    registration.produce(&arguments, d, context, output)
                }),
            )
            .await;
        }
        Run::Local(..) | Run::ConfiguredLocal(..) => return StatusCode::NOT_FOUND.into_response(),
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
        return stream_as(
            web,
            policy.class,
            limit,
            request,
            details.clone(),
            Box::new(move |d, invocation, context, output| {
                registration.produce(&arguments, c, d, invocation, &caller, context, output)
            }),
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
    let mut accepted = crate::operation::Accepted::new(
        &web.daemon.callers,
        &details,
        command_surface::Via::Web,
        limit.cancellation,
    );
    accepted.parent = super::parent_lifetime();
    accepted.read_scope = read_scope.clone();
    accepted.invocation = Some(
        web.daemon
            .invocation(&accepted.authorship)
            .with_accepted_request(Some(Arc::new(request.clone())))
            .with_calling_session(accepted.calling_session.clone())
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
    let scopes = crate::protocol::ReadScopes::single(read_scope.clone());
    let answer = crate::protocol::scope_answer(dispatched, &scopes);
    let line = answer
        .value()
        .filter(|_| c.name == "rows")
        .and_then(|value| {
            serde_json::from_value::<crate::protocol::line::LineAnswer>(value.clone()).ok()
        });
    let failed_step = line.as_ref().and_then(|line| {
        let crate::protocol::line::LineAnswer::Steps(steps) = line else {
            return None;
        };
        steps.steps.iter().find_map(|step| match &step.outcome {
            crate::protocol::line::StepOutcome::Failed { error } => Some(error.clone()),
            _ => None,
        })
    });
    let single_read = matches!(&line, Some(crate::protocol::line::LineAnswer::Steps(steps))
        if steps.steps.len() == 1 && matches!(steps.steps[0].outcome,
            crate::protocol::line::StepOutcome::Done { reply: crate::protocol::line::StepAnswer::Read(_), .. }));
    let single_step = matches!(&line, Some(crate::protocol::line::LineAnswer::Steps(steps)) if steps.steps.len() == 1);
    let step_status = failed_step
        .as_ref()
        .and_then(|error| error.kind.http_status())
        .and_then(|status| StatusCode::from_u16(status).ok());
    let wants_html = headers
        .get(header::ACCEPT)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| {
            value.split(',').any(|kind| {
                kind.split(';')
                    .next()
                    .is_some_and(|kind| kind.trim() == "text/html")
            })
        });
    if wants_html
        && single_step
        && let Some(error) = &failed_step
    {
        return super::failed_with(error);
    }
    if wants_html && c.name == "rows" && !single_read {
        let declarations = crate::commands::declarations();
        let rendered = declarations
            .render_text("rows", &answer, command_surface::TextStyle::Plain)
            .and_then(|text| declarations.cli_command(&request).map(|cli| (text, cli)));
        let (mut text, cli) = match rendered {
            Ok(rendered) => rendered,
            Err(error) => {
                return json_answer(
                    web,
                    command_surface::Answer::<serde_json::Value>::error(error),
                );
            }
        };
        text.push_str(&declarations.render_footer(command_surface::Via::Web, &answer));
        let html = super::pages::layout(
            "Polylogue",
            "",
            &cli,
            &format!("<pre>{}</pre>", polylogue_render::html::escape(&text)),
        );
        let status = step_status
            .or_else(|| {
                answer
                    .http_status()
                    .and_then(|status| StatusCode::from_u16(status).ok())
            })
            .unwrap_or(StatusCode::OK);
        return scoped_body_response(
            web,
            status,
            "text/html; charset=utf-8",
            html.into_bytes(),
            &scopes,
        );
    }
    if matches!(c.name, "read" | "rows")
        && wants_html
        && let Some(value) = answer.value()
    {
        let page = if c.name == "rows" {
            serde_json::from_value::<crate::protocol::line::LineAnswer>(value.clone()).and_then(
                |line| match line {
                    crate::protocol::line::LineAnswer::Steps(mut steps)
                        if steps.steps.len() == 1 =>
                    {
                        match steps.steps.remove(0).outcome {
                            crate::protocol::line::StepOutcome::Done {
                                reply: crate::protocol::line::StepAnswer::Read(page),
                                ..
                            } => Ok(*page),
                            _ => Err(serde::de::Error::custom(
                                "the line did not return a read page",
                            )),
                        }
                    }
                    _ => Err(serde::de::Error::custom("HTML requires one read step")),
                },
            )
        } else {
            serde_json::from_value::<polylogue_model::read::ReadPage>(value.clone())
        };
        match page
            .map_err(|error| {
                command_surface::Failure::new(command_surface::FailureKind::Io, error.to_string())
            })
            .and_then(|page| super::pages::reader::read(&page, table, &request))
        {
            Ok(html) => {
                if let Err(error) = scopes.check() {
                    return json_answer(web, crate::daemon::service::store_error(&error));
                }
                return scoped_body_response(
                    web,
                    StatusCode::OK,
                    "text/html; charset=utf-8",
                    html.into_bytes(),
                    &scopes,
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
    let status = step_status.unwrap_or(status);
    scoped_body_response(web, status, "application/json", bytes, &scopes)
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

#[cfg(test)]
async fn stream(
    web: &Arc<Web>,
    class: crate::daemon::executor::Class,
    limit: polylogue_store::read::limit::ReadLimit,
    request: command_surface::RunnableRequest,
    produce: crate::daemon::streaming::Produce,
) -> HttpResponse {
    stream_as(web, class, limit, request, Default::default(), produce).await
}

/// Headers are committed on the first produced bytes. Subsequent failures
/// terminate the stream with an error event, without completion.
async fn stream_as(
    web: &Arc<Web>,
    class: crate::daemon::executor::Class,
    limit: polylogue_store::read::limit::ReadLimit,
    accepted_request: command_surface::RunnableRequest,
    details: command_surface::CallerEvidence,
    produce: crate::daemon::streaming::Produce,
) -> HttpResponse {
    let cancellation = limit.cancellation.clone();
    let accepted = crate::operation::Accepted::new(
        &web.daemon.callers,
        &details,
        command_surface::Via::Web,
        cancellation.clone(),
    );
    let invocation = web
        .daemon
        .invocation(&accepted.authorship)
        .with_accepted_request(Some(Arc::new(accepted_request)))
        .with_calling_session(accepted.calling_session.clone())
        .with_parent(super::parent_lifetime());
    let mut cancel = super::CancelRead(Some(cancellation.clone()));
    let input =
        match crate::daemon::streaming::start(&web.daemon, class, limit, invocation, produce) {
            Ok(input) => input,
            Err(failure) => return json_answer(web, failure.answer::<serde_json::Value>()),
        };
    let first = match input.next().await {
        Ok(crate::daemon::stream_buffer::Next::Bytes(bytes)) => bytes,
        Ok(crate::daemon::stream_buffer::Next::Failed(failure)) => {
            return json_answer(web, failure.answer::<serde_json::Value>());
        }
        Ok(crate::daemon::stream_buffer::Next::End) => {
            return json_answer(
                web,
                crate::daemon::service::error(
                    polylogue_model::Code::Internal,
                    "stream completed without producing events",
                ),
            );
        }
        Err(error) => {
            return json_answer(
                web,
                crate::daemon::service::error(polylogue_model::Code::Io, error),
            );
        }
    };
    cancel.0.take();
    let body = StreamBody {
        input: Arc::new(input),
        pending: first,
        offset: 0,
        next: None,
        ended: false,
        cancellation,
    };
    (
        [
            (header::CONTENT_TYPE, "text/event-stream"),
            (header::CACHE_CONTROL, "no-cache"),
        ],
        axum::body::Body::from_stream(tokio_util::io::ReaderStream::new(body)),
    )
        .into_response()
}

type NextChunk = std::pin::Pin<
    Box<
        dyn std::future::Future<Output = std::io::Result<crate::daemon::stream_buffer::Next>>
            + Send,
    >,
>;

/// The body of a streamed response, read from the stream buffer; dropping
/// it, as when the browser goes away, cancels the producer.
struct StreamBody {
    input: Arc<crate::daemon::stream_buffer::StreamInput>,
    pending: Vec<u8>,
    offset: usize,
    next: Option<NextChunk>,
    ended: bool,
    cancellation: polylogue_store::cancellation::Cancellation,
}
impl tokio::io::AsyncRead for StreamBody {
    fn poll_read(
        self: std::pin::Pin<&mut Self>,
        context: &mut std::task::Context<'_>,
        buffer: &mut tokio::io::ReadBuf<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        use crate::daemon::stream_buffer::Next;
        let this = self.get_mut();
        loop {
            if this.offset < this.pending.len() {
                let end = this
                    .pending
                    .len()
                    .min(this.offset.saturating_add(buffer.remaining()));
                buffer.put_slice(&this.pending[this.offset..end]);
                this.offset = end;
                return std::task::Poll::Ready(Ok(()));
            }
            if this.ended {
                return std::task::Poll::Ready(Ok(()));
            }
            let next = this.next.get_or_insert_with(|| {
                let input = Arc::clone(&this.input);
                Box::pin(async move { input.next().await })
            });
            let chunk = std::task::ready!(next.as_mut().poll(context));
            this.next = None;
            this.offset = 0;
            match chunk? {
                Next::Bytes(bytes) => this.pending = bytes,
                Next::Failed(failure) => {
                    this.pending.clear();
                    crate::protocol::stream::error(&mut this.pending, failure)?;
                    this.ended = true;
                }
                Next::End => {
                    this.pending.clear();
                    this.ended = true;
                }
            }
        }
    }
}
impl Drop for StreamBody {
    fn drop(&mut self) {
        self.cancellation.cancel();
    }
}

#[cfg(test)]
mod tests;
