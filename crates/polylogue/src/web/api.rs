//! The reader's JSON API under `/api/v1/`: the same data as `--json`, one
//! route per command the table offers over HTTP, at its words joined by `/`
//! (`/api/v1/dev/sql`). A POST carries a JSON object body; a GET carries its
//! arguments as query parameters (a list argument repeats its name) and runs
//! only a request whose policy reads and leaves the archive unchanged.
//!
//! Each request is answered by the daemon's one `answer`, in a blocking
//! task. A reply with no events is one `application/json` body with the
//! answer's HTTP status; a reply whose first event was written is
//! `text/event-stream` with the shared frames, ending with its `answer`.

use super::{Lifetime, Web, json_answer};
use crate::commands::{COMMANDS, Command};
use crate::daemon::output::{Output, Sink, Spill, answer_frame};
use axum::Router;
use axum::extract::{Extension, Query, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response as HttpResponse};
use command_surface::{Answer, Failure, FailureKind};
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::mpsc;

/// The commands offered over HTTP.
pub fn http_commands() -> impl Iterator<Item = &'static Command> {
    COMMANDS.iter().filter(|c| c.http)
}

pub fn routes() -> Router<Arc<Web>> {
    let mut router = Router::new();
    for c in http_commands() {
        let methods = axum::routing::post(
            move |State(web): State<Arc<Web>>,
                  Extension(lifetime): Extension<Lifetime>,
                  headers: HeaderMap,
                  body: String| async move {
                if !super::access::content_type(&headers, "application/json") {
                    return (
                        StatusCode::UNSUPPORTED_MEDIA_TYPE,
                        "a POST body is application/json",
                    )
                        .into_response();
                }
                match body_request(c, &body) {
                    Ok((request, evidence)) => {
                        respond(&web, request, lifetime.caller(&evidence)).await
                    }
                    Err(error) => refused(&web, error, unbuilt_in_body(c, &body)),
                }
            },
        )
        .get(
            move |State(web): State<Arc<Web>>,
                  Extension(lifetime): Extension<Lifetime>,
                  Query(pairs): Query<Vec<(String, String)>>| async move {
                match get_request(c, &pairs) {
                    Ok(request) => {
                        respond(&web, request, lifetime.caller(&Default::default())).await
                    }
                    Err(failure) => json_answer(&web, Answer::error(failure)),
                }
            },
        );
        router = router.route(&c.http_path(), methods);
    }
    router.layer(axum::extract::DefaultBodyLimit::disable())
}

/// A refused request: `not_built` when it names an option on the list of
/// what is designed and not built, else a bad request.
fn refused(web: &Web, error: String, unbuilt: Option<command_surface::Failure>) -> HttpResponse {
    match unbuilt {
        Some(failure) => json_answer(web, command_surface::Answer::<Value>::error(failure)),
        None => json_answer(
            web,
            crate::daemon::service::error(polylogue_model::Code::BadRequest, error),
        ),
    }
}

/// A listed option among a POST body's arguments.
fn unbuilt_in_body(c: &Command, body: &str) -> Option<command_surface::Failure> {
    let body: Value = serde_json::from_str(body).ok()?;
    crate::unbuilt::in_arguments(c.name, body.get("arguments").unwrap_or(&body))
}

/// A listed option among a GET's query names, of the row or of its step.
fn unbuilt_in_query(c: &Command, pairs: &[(String, String)]) -> Option<command_surface::Failure> {
    let verb = pairs
        .iter()
        .find(|(name, _)| name == "steps")
        .map(|(_, verb)| verb.as_str());
    pairs.iter().find_map(|(name, _)| {
        verb.and_then(|verb| crate::unbuilt::option(verb, name))
            .or_else(|| crate::unbuilt::option(c.name, name))
    })
}

/// A POST takes the shared request or the operation's arguments. The calling
/// session belongs to the transport, outside those arguments.
fn body_request(
    c: &Command,
    body: &str,
) -> Result<
    (
        command_surface::RunnableRequest,
        command_surface::CallerEvidence,
    ),
    String,
> {
    use serde_json::{Map, json};
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
    if request.operation != c.name {
        return Err("route does not match operation".into());
    }
    Ok((request, caller))
}

/// A GET runs only what reads and leaves the archive unchanged, as its
/// policy says.
pub(super) fn get_request(
    c: &Command,
    pairs: &[(String, String)],
) -> Result<command_surface::RunnableRequest, Failure> {
    let request = query_request(c, pairs).map_err(|message| {
        unbuilt_in_query(c, pairs).unwrap_or_else(|| Failure::new(FailureKind::BadRequest, message))
    })?;
    if request.operation != c.name {
        return Err(Failure::new(
            FailureKind::BadRequest,
            "route does not match operation",
        ));
    }
    let policy = crate::daemon::handler::policy(
        c,
        &request.arguments,
        polylogue_model::authorship::Channel::Web,
    )
    .map_err(|failure| failure.error)?;
    if !policy.takes_get() {
        return Err(Failure::new(
            FailureKind::BadRequest,
            format!("use POST /api/v1{}", c.http_path()),
        ));
    }
    Ok(request)
}

/// A request from query pairs. A row with declared steps takes one step's
/// arguments as query values under their own names, beside the row's own
/// and the envelope's; `steps` names its verb.
fn query_request(
    c: &Command,
    pairs: &[(String, String)],
) -> Result<command_surface::RunnableRequest, String> {
    let declarations = crate::commands::declarations();
    let id = polylogue_store::user_writes::new_request_id();
    let verbs: Vec<_> = pairs
        .iter()
        .filter(|(name, _)| name == "steps")
        .map(|(_, value)| value.as_str())
        .collect();
    let Some(steps) = declarations
        .steps(c.name, "steps")
        .filter(|_| !verbs.is_empty())
    else {
        return declarations
            .parse_http_query(c.name, pairs, id)
            .map_err(|failure| failure.message);
    };
    let [verb] = verbs.as_slice() else {
        return Err(format!(
            "a GET carries one step; use POST /api/v1{} for several",
            c.http_path()
        ));
    };
    if !steps
        .verbs()
        .iter()
        .any(|declared| declared.verb() == *verb)
        || step_query_declarations(c).spec_named(verb).is_none()
    {
        return Err(format!(
            "a GET carries only the read and brief steps, not {verb}; use POST /api/v1{}",
            c.http_path()
        ));
    }
    let fields = declarations.spec_named(c.name).expect("declared row");
    let mut line_pairs = Vec::new();
    let mut step_pairs = Vec::new();
    for (name, value) in pairs {
        if name == "steps" {
            continue;
        }
        if fields.argument(name).is_some()
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
    let step = step_query_declarations(c)
        .parse_http_query(verb, &step_pairs, "http-step")
        .map_err(|failure| failure.message)?;
    let cursor = step.cursor;
    let mut step = step.arguments;
    let object = step
        .as_object_mut()
        .ok_or("step arguments must be an object")?;
    if let Some(cursor) = cursor {
        object.insert("cursor".into(), cursor.into());
    }
    object.insert("verb".into(), (*verb).into());
    let mut request = declarations
        .parse_http_query(c.name, &line_pairs, id)
        .map_err(|failure| failure.message)?;
    request
        .arguments
        .as_object_mut()
        .ok_or("arguments must be an object")?
        .insert("steps".into(), serde_json::json!([step]));
    Ok(request)
}

/// Declarations that decode one step's query values: the read-only steps a
/// GET may carry.
fn step_query_declarations(row: &Command) -> &'static command_surface::OperationDeclarations {
    static STEPS: std::sync::OnceLock<std::sync::Arc<command_surface::OperationDeclarations>> =
        std::sync::OnceLock::new();
    STEPS.get_or_init(|| {
        fn declare<A: command_surface::JsonSchema>(
            builder: &mut command_surface::DeclarationsBuilder,
            row: &Command,
            name: &'static str,
        ) {
            let step = Command { name, ..*row };
            let mut spec = crate::commands::specification::<A>(&step);
            spec.kind = command_surface::OperationKind::Request { tool: None };
            // These declarations only decode HTTP query values; step fields
            // do not consume the command line's budget or cursor options.
            for argument in &mut spec.arguments {
                if matches!(argument.name.as_str(), "budget" | "cursor") {
                    argument.form = command_surface::ArgumentForm::Positional;
                }
            }
            builder
                .declare::<A, serde_json::Value>(spec, crate::cli::print::json::<serde_json::Value>)
                .expect("read step query declaration");
        }
        let mut builder = command_surface::OperationDeclarations::builder(
            crate::commands::PRODUCT,
            crate::commands::deadlines(),
        );
        declare::<crate::protocol::steps::ReadStep>(&mut builder, row, "read");
        declare::<crate::protocol::steps::BriefStep>(&mut builder, row, "brief");
        builder.finish()
    })
}

/// One piece of a reply on its way from the blocking task to the response.
enum Piece {
    Bytes(Vec<u8>),
    /// A reply with no events, and the failure its reply carries inside it.
    Answer(Box<Answer<Value>>, Option<StatusCode>),
}

/// The response side of a reply as a [`Sink`].
struct Channel(mpsc::Sender<Piece>);

impl Sink for Channel {
    fn try_send(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        match self.0.try_send(Piece::Bytes(bytes.to_vec())) {
            Ok(()) => Ok(bytes.len()),
            Err(mpsc::error::TrySendError::Full(_)) => Ok(0),
            Err(mpsc::error::TrySendError::Closed(_)) => Err(gone()),
        }
    }
    fn send(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        self.0
            .blocking_send(Piece::Bytes(bytes.to_vec()))
            .map_err(|_| gone())
    }
}

fn gone() -> std::io::Error {
    std::io::ErrorKind::BrokenPipe.into()
}

/// One HTTP reply: until its first event it may still be a JSON answer.
struct Reply {
    spill: Option<Spill<Channel>>,
    streaming: bool,
}

impl Output for Reply {
    fn events(&mut self, frames: &[u8]) -> std::io::Result<()> {
        self.streaming = true;
        self.spill.as_mut().ok_or_else(gone)?.write(frames)
    }

    fn answer(
        &mut self,
        answer: Answer<Value>,
        inside: Option<command_surface::FailureKind>,
    ) -> std::io::Result<()> {
        let mut spill = self.spill.take().ok_or_else(gone)?;
        if self.streaming {
            spill.write(&answer_frame(&answer)?)?;
            spill.finish().map(drop)
        } else {
            let Channel(sender) = spill.finish()?;
            let step_status = inside
                .and_then(FailureKind::http_status)
                .and_then(|status| StatusCode::from_u16(status).ok());
            sender
                .blocking_send(Piece::Answer(Box::new(answer), step_status))
                .map_err(|_| gone())
        }
    }
}

/// A complete answer or the stream already in progress.
pub(super) enum HttpReply {
    Answer(Box<Answer<Value>>, Option<StatusCode>),
    Stream(HttpResponse),
}

/// Answers one request in a blocking task. Dropping the response before
/// the reply ends cancels the request.
pub(super) async fn execute(
    web: &Arc<Web>,
    request: command_surface::RunnableRequest,
    caller: crate::operation::Caller,
) -> HttpReply {
    let (sender, mut pieces) = mpsc::channel(16);
    let cancel = polylogue_store::cancellation::CancelOnDrop::new(caller.stop.clone());
    let daemon = Arc::clone(&web.daemon);
    let staging = daemon.archive.staging_dir();
    tokio::task::spawn_blocking(move || {
        let mut reply = Reply {
            spill: Some(Spill::new(Channel(sender), staging)),
            streaming: false,
        };
        crate::daemon::handler::answer(&daemon, &request, &caller, &mut reply);
    });
    match pieces.recv().await {
        Some(Piece::Answer(answer, step_status)) => {
            let mut cancel = cancel;
            cancel.disarm();
            HttpReply::Answer(answer, step_status)
        }
        Some(Piece::Bytes(first)) => HttpReply::Stream(
            (
                [
                    (header::CONTENT_TYPE, "text/event-stream"),
                    (header::CACHE_CONTROL, "no-cache"),
                ],
                axum::body::Body::from_stream(tokio_util::io::ReaderStream::new(EventBody {
                    pieces,
                    pending: first,
                    offset: 0,
                    _cancel: cancel,
                })),
            )
                .into_response(),
        ),
        None => HttpReply::Answer(
            Box::new(
                crate::daemon::Failure::internal("the request ended without an answer").answer(),
            ),
            None,
        ),
    }
}

pub(super) async fn respond(
    web: &Arc<Web>,
    request: command_surface::RunnableRequest,
    caller: crate::operation::Caller,
) -> HttpResponse {
    match execute(web, request, caller).await {
        HttpReply::Answer(answer, step_status) => {
            let (status, bytes) = super::prepare_json_answer(web, *answer);
            (
                step_status.unwrap_or(status),
                [(header::CONTENT_TYPE, "application/json")],
                bytes,
            )
                .into_response()
        }
        HttpReply::Stream(response) => response,
    }
}

/// The body of a streamed response; dropping it, as when the browser goes
/// away, cancels the request.
struct EventBody {
    pieces: mpsc::Receiver<Piece>,
    pending: Vec<u8>,
    offset: usize,
    _cancel: polylogue_store::cancellation::CancelOnDrop,
}

impl tokio::io::AsyncRead for EventBody {
    fn poll_read(
        self: std::pin::Pin<&mut Self>,
        context: &mut std::task::Context<'_>,
        buffer: &mut tokio::io::ReadBuf<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
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
            match std::task::ready!(this.pieces.poll_recv(context)) {
                Some(Piece::Bytes(bytes)) => {
                    this.pending = bytes;
                    this.offset = 0;
                }
                Some(Piece::Answer(..)) | None => return std::task::Poll::Ready(Ok(())),
            }
        }
    }
}

#[cfg(test)]
mod tests;
