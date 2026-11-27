//! One function answers every request on every transport.
//!
//! [`answer`] takes the shared request, the [`Caller`] its transport made
//! and the transport's [`Output`], and runs the row's [`Operation`] on the
//! calling thread: the socket's connection thread, a blocking task of the
//! HTTP listener, a web view or a test. [`ask`] is the same path for
//! in-process callers, with typed arguments and a typed answer. Each row
//! names its operation as one erased [`Row`] built from the operation's
//! type, so a row without an operation does not compile and nothing joins
//! two tables by name.
use super::Failure;
use super::output::{Collected, Output};
use super::service::Daemon;
use crate::commands::{Command, ToolAccess};
use crate::operation::Caller;
use command_surface::{
    Answer, Deadline, DeclarationsBuilder, JsonSchema, Outcome, RunnableRequest, TextStyle,
    TokenBudget,
};
use polylogue_model::authorship::Channel;
use polylogue_reading::operation::{Context, Invocation};
pub(crate) use polylogue_store::Lane;
use polylogue_store::read::limit::ReadLimit;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value;
use std::any::TypeId;
use std::cell::{Cell, RefCell};
use std::io::Write;
use std::marker::PhantomData;
use std::sync::Arc;

/// One daemon operation: its argument and reply types, how a request is
/// treated, and what answers it.
pub(crate) trait Operation: 'static {
    type Arguments: Serialize + DeserializeOwned + JsonSchema;
    type Reply: Serialize + DeserializeOwned + JsonSchema;
    /// The reply as text for a person or a model.
    const PRINT: fn(&Self::Reply, TextStyle) -> String = crate::cli::print::json::<Self::Reply>;
    /// How this request is treated, decided from its typed arguments and the
    /// channel that carried it. A refusal here runs nothing.
    fn policy(arguments: &Self::Arguments, channel: Channel) -> Result<Policy, Failure>;
    fn run(arguments: Self::Arguments, call: &Call<'_>) -> Result<Answer<Self::Reply>, Failure>;
    /// The failure a successful reply carries inside it, such as a line's
    /// failed step; HTTP answers with its status.
    fn failure(_reply: &Self::Reply) -> Option<command_surface::FailureKind> {
        None
    }
    /// The same request answered as one reply, for a caller that cannot
    /// hold a stream; none when it does not stream.
    fn without_stream(_arguments: &Self::Arguments) -> Option<Self::Arguments> {
        None
    }
    /// The reply as text for a model, when it has one other than its JSON.
    const MODEL_TEXT: Option<ModelText<Self::Reply>> = None;
}

/// A reply as text for a model, or why it cannot be.
pub(crate) type ModelText<Reply> = fn(&Reply) -> Result<String, String>;

/// How one request is treated before and while it runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Policy {
    /// The smallest MCP tool that may run it.
    pub access: ToolAccess,
    /// Refused while writes are blocked; over HTTP only by POST.
    pub changes_archive: bool,
    pub bound: Bound,
    /// Which readers it may take.
    pub lane: Lane,
    pub late: Late,
    /// Continues a page from the request's cursor.
    pub pages: bool,
    /// Prepares context within the request's token budget.
    pub budgeted: bool,
}

/// How long a request's reads may run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Bound {
    /// A read: the caller's deadline, else its class's, else 30 s.
    Whole,
    /// Reads, then hands a write to the writer: the same deadline, until the
    /// hand-over; the write then answers with its committed result.
    UntilHandOver,
    /// Only a deadline the caller names: a stream, idle-priority work.
    CallerOnly,
    /// A write: no deadline; it answers with its committed result.
    None,
}

/// What a request whose limit runs out answers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Late {
    /// A deadline refusal naming the stage that ran out.
    Refuse,
    /// Its empty answer, within the optional advisory's cap.
    AnswerEmpty,
}

impl Policy {
    /// A read the read tool runs.
    pub(crate) const READ: Policy = Policy {
        access: ToolAccess::Read,
        changes_archive: false,
        bound: Bound::Whole,
        lane: Lane::Ordinary,
        late: Late::Refuse,
        pages: false,
        budgeted: false,
    };

    /// A write that the tool of `access` runs.
    pub(crate) const fn write(access: ToolAccess) -> Policy {
        Policy {
            access,
            changes_archive: true,
            bound: Bound::None,
            ..Self::READ
        }
    }

    /// Whether an HTTP GET may carry the request: it only reads.
    pub(crate) fn takes_get(&self) -> bool {
        self.access == ToolAccess::Read && !self.changes_archive
    }
}

/// What an operation works with beside its typed arguments.
pub(crate) struct Call<'a> {
    pub daemon: &'a Daemon,
    pub caller: &'a Caller,
    /// The request as accepted, with its clock and authorship.
    pub invocation: &'a Invocation,
    /// The caller's retry identity for writes.
    pub request_id: &'a str,
    /// The page a continuing request resumes from.
    pub cursor: Option<&'a str>,
    pub budget: TokenBudget,
    /// The request's one read limit.
    pub limit: &'a ReadLimit,
    pub lane: Lane,
    /// The archive position a preceding write needs the next reader to see.
    pub after_write: Option<polylogue_store::archive::writer::Position>,
    /// Where the request's events go; its final answer is the return value.
    pub output: &'a dyn Events,
    /// The query the request runs, which a deadline refusal narrows.
    pub query: &'a RefCell<Option<String>>,
}

impl Call<'_> {
    /// The evaluation context (clock, logical day, Sinex context) of this request.
    pub fn context(&self) -> Result<Arc<Context>, Failure> {
        self.daemon
            .resolve_operation_context(self.invocation.clone())
    }

    /// The request's reader, taken in its lane: under its limit while it
    /// reads, and only under its cancellation once a write was handed over
    /// or for a write, so a committed result is read back whatever the time.
    pub fn reader(&self) -> Result<polylogue_store::Reader<'_>, Failure> {
        let reads = &self.daemon.reads;
        let limit = ReadLimit::current()
            .unwrap_or_else(|| ReadLimit::without_deadline(self.limit.cancellation.clone()));
        Ok(match self.after_write {
            Some(position) => reads.checkout_after(
                super::service::writer(self.daemon)?,
                position,
                self.lane,
                &limit,
            )?,
            None => reads.checkout(self.lane, &limit)?,
        })
    }

    /// Ends the read phase before a write is handed to the writer: a read
    /// phase that has run out refuses the hand-over; after it, the write
    /// answers with its committed result however long that takes.
    pub fn hand_over(&self, stage: &'static str) -> Result<(), Failure> {
        hand_over(stage)
    }

    /// Records the query this request runs, for a deadline refusal to narrow.
    pub fn runs_query(&self, text: &str) {
        self.query.replace(Some(text.to_owned()));
    }

    /// The request's events as a writer of server-sent-event frames.
    pub fn events(&self) -> EventWriter<'_> {
        EventWriter(self.output)
    }

    /// The prepared-context sender a request from this caller is recorded
    /// under; none for the command line and HTTP.
    pub fn sender(&self) -> Option<polylogue_model::DeliverySender> {
        self.caller.sender()
    }

    /// A directory the caller named, made absolute against its working
    /// directory; without one, the caller's working directory.
    pub fn directory(
        &self,
        named: Option<&std::path::Path>,
    ) -> Result<std::path::PathBuf, Failure> {
        match (named, &self.caller.cwd) {
            (Some(dir), Some(base)) => Ok(base.join(dir)),
            (Some(dir), None) if dir.is_absolute() => Ok(dir.into()),
            (Some(_), None) => Err(Failure::bad_request("`cwd` must be an absolute path")),
            (None, Some(base)) => Ok(base.clone()),
            (None, None) => Err(Failure::bad_request("`cwd` is required")),
        }
    }

    /// A path the caller named, made absolute against its working directory.
    pub fn path(&self, path: &std::path::Path, what: &str) -> Result<std::path::PathBuf, Failure> {
        resolve_path(self.caller.cwd.as_deref(), path, what)
    }

    /// The same call; its fields are all borrowed.
    pub(crate) fn copy(&self) -> Self {
        Self { ..*self }
    }
}

impl Clone for Call<'_> {
    fn clone(&self) -> Self {
        *self
    }
}
impl Copy for Call<'_> {}

/// Ends the read phase of the request this thread answers before its write
/// is handed over. An already expired read refuses the hand-over.
pub(crate) fn hand_over(stage: &'static str) -> Result<(), Failure> {
    ReadLimit::finish_scope(stage).map_err(Failure::from)
}

/// A path a caller named, made absolute against its working directory; a
/// caller without one (HTTP) names an absolute path.
pub(crate) fn resolve_path(
    cwd: Option<&std::path::Path>,
    path: &std::path::Path,
    what: &str,
) -> Result<std::path::PathBuf, Failure> {
    if path.is_absolute() {
        return Ok(path.into());
    }
    cwd.map(|base| base.join(path)).ok_or_else(|| {
        Failure::bad_request(format!(
            "a relative {what} requires the caller's working directory; supply an absolute path"
        ))
    })
}

/// Where a request writes its events. Every write first checks the
/// request's limit and the read scope it was accepted under, so a
/// cancelled, expired or revoked stream stops at its next event.
pub(crate) trait Events {
    fn write(&self, frames: &[u8]) -> std::io::Result<()>;
}

/// [`Call::events`]: the request's events as `std::io::Write`.
pub(crate) struct EventWriter<'a>(&'a dyn Events);

impl Write for EventWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.write(bytes)?;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// A transport's output as the events of one request.
struct Transport<'o> {
    output: RefCell<&'o mut dyn Output>,
    written: Cell<bool>,
}

impl Events for Transport<'_> {
    fn write(&self, frames: &[u8]) -> std::io::Result<()> {
        self.written.set(true);
        self.output.borrow_mut().events(frames)
    }
}

/// One row's operation, erased: its declaration, its policy and its answer
/// over the wire's JSON, all derived from the operation's type.
pub(crate) trait Row: Sync {
    fn operation(&self) -> TypeId;
    fn declare(&self, declarations: &mut DeclarationsBuilder, row: &'static Command);
    /// The schema of the operation's request type.
    fn arguments(&self) -> schemars::Schema;
    /// Decodes the arguments and states how they are treated.
    fn policy(&self, arguments: &Value, channel: Channel) -> Result<Policy, Failure>;
    /// The request answered as one reply, when it would stream.
    fn without_stream(&self, arguments: &Value) -> Result<Option<Value>, Failure>;
    /// The failure a successful reply carries inside it.
    fn failure_inside(&self, reply: &Value) -> Option<command_surface::FailureKind>;
    /// The reply as text for a model, when it has one other than its JSON.
    fn model_text(&self, reply: &Value) -> Option<Result<String, String>>;
    /// The answer, and the failure its reply carries inside it.
    fn respond(
        &self,
        request: &Asked<'_>,
        events: &dyn Events,
    ) -> (Answer<Value>, Option<command_surface::FailureKind>);
}

struct Handled<O>(PhantomData<fn() -> O>);

/// The row of operation `O`.
pub(crate) const fn daemon<O: Operation>() -> &'static dyn Row {
    &Handled::<O>(PhantomData)
}

impl<O: Operation> Row for Handled<O> {
    fn operation(&self) -> TypeId {
        TypeId::of::<O>()
    }

    fn declare(&self, declarations: &mut DeclarationsBuilder, row: &'static Command) {
        declarations
            .declare::<O::Arguments, O::Reply>(
                crate::commands::specification::<O::Arguments>(row),
                O::PRINT,
            )
            .expect("declared command is unique");
    }

    fn arguments(&self) -> schemars::Schema {
        schemars::schema_for!(O::Arguments)
    }

    fn policy(&self, arguments: &Value, channel: Channel) -> Result<Policy, Failure> {
        O::policy(&decode::<O::Arguments>(arguments)?, channel)
    }

    fn without_stream(&self, arguments: &Value) -> Result<Option<Value>, Failure> {
        O::without_stream(&decode::<O::Arguments>(arguments)?)
            .map(|arguments| serde_json::to_value(arguments).map_err(Failure::internal))
            .transpose()
    }

    fn failure_inside(&self, reply: &Value) -> Option<command_surface::FailureKind> {
        O::failure(&O::Reply::deserialize(reply).ok()?)
    }

    fn model_text(&self, reply: &Value) -> Option<Result<String, String>> {
        let text = O::MODEL_TEXT?;
        Some(
            O::Reply::deserialize(reply)
                .map_err(|error| error.to_string())
                .and_then(|reply| text(&reply)),
        )
    }

    fn respond(
        &self,
        request: &Asked<'_>,
        events: &dyn Events,
    ) -> (Answer<Value>, Option<command_surface::FailureKind>) {
        let arguments = match decode::<O::Arguments>(&request.request.arguments) {
            Ok(arguments) => arguments,
            Err(failure) => return (failure.answer(), None),
        };
        let answer = respond::<O>(request, arguments, events);
        let inside = answer.value().and_then(O::failure);
        let mut problem = None;
        let value = answer.map(|reply| {
            serde_json::to_value(reply)
                .map_err(|error| problem = Some(error))
                .unwrap_or_default()
        });
        match problem {
            Some(error) => (Failure::internal(error).answer(), None),
            None => (value, inside),
        }
    }
}

fn decode<T: DeserializeOwned>(arguments: &Value) -> Result<T, Failure> {
    if !(arguments.is_object() || arguments.is_null()) {
        return Err(Failure::bad_request("arguments must be an object"));
    }
    T::deserialize(arguments).map_err(Failure::bad_request)
}

/// The policy of a request for `row`, decoded from its JSON arguments;
/// clients ask it before sending, and a ready call before it is offered.
pub(crate) fn policy(
    row: &Command,
    arguments: &Value,
    channel: Channel,
) -> Result<Policy, Failure> {
    if let Some(unbuilt) = crate::unbuilt::in_arguments(row.name, arguments) {
        return Err(unbuilt.into());
    }
    match row.run {
        crate::commands::Run::Daemon(operation) => operation.policy(arguments, channel),
        _ => Err(Failure::bad_request(format!(
            "{} runs in the calling process",
            row.name
        ))),
    }
}

/// A request accepted by its transport, before its operation decodes it.
pub(crate) struct Asked<'a> {
    daemon: &'a Daemon,
    row: &'static Command,
    request: &'a RunnableRequest,
    caller: &'a Caller,
}

/// Answers one request on the calling thread: its events and then its
/// final answer go to `output`. This is the only way an operation runs.
pub(crate) fn answer(
    daemon: &Daemon,
    request: &RunnableRequest,
    caller: &Caller,
    output: &mut dyn Output,
) {
    let (answer, inside) = match accept(daemon, request) {
        Ok((row, operation, _running)) => {
            let events = Transport {
                output: RefCell::new(&mut *output),
                written: Cell::new(false),
            };
            let accepted = Asked {
                daemon,
                row,
                request,
                caller,
            };
            operation.respond(&accepted, &events)
        }
        Err(failure) => (failure.answer(), None),
    };
    if let Err(error) = output.answer(answer, inside) {
        log_unsent(request, &error);
    }
}

fn log_unsent(request: &RunnableRequest, error: &std::io::Error) {
    if !matches!(
        error.kind(),
        std::io::ErrorKind::BrokenPipe
            | std::io::ErrorKind::ConnectionReset
            | std::io::ErrorKind::TimedOut
            | std::io::ErrorKind::WouldBlock
    ) {
        polylogue_store::log!(
            warning,
            "the answer to {} ({}) was not sent: {error}",
            request.operation,
            request.request_id
        );
    }
}

/// The row a request names, once its envelope is checked, while the daemon
/// counts it as running.
fn accept<'d>(
    daemon: &'d Daemon,
    request: &RunnableRequest,
) -> Result<
    (
        &'static Command,
        &'static dyn Row,
        super::service::Running<'d>,
    ),
    Failure,
> {
    let running = daemon.requests.enter()?;
    let row = crate::commands::command(&request.operation)
        .ok_or_else(|| Failure::bad_request(format!("unknown operation {}", request.operation)))?;
    let crate::commands::Run::Daemon(operation) = row.run else {
        return Err(Failure::bad_request(format!(
            "{} runs in the calling process",
            row.name
        )));
    };
    check_envelope(row, request)?;
    Ok((row, operation, running))
}

/// What the shared request format promises before an operation decodes its
/// arguments: a request ID, only the envelope options the product takes,
/// only declared arguments and well-formed steps.
fn check_envelope(row: &Command, request: &RunnableRequest) -> Result<(), Failure> {
    if request.request_id.is_empty() {
        return Err(Failure::bad_request("request_id must not be empty"));
    }
    let declarations = crate::commands::declarations();
    if let Some(option) = command_surface::RequestOption::ALL
        .into_iter()
        .find(|option| option.used_by(request) && !declarations.accepts(*option))
    {
        return Err(Failure::bad_request(format!(
            "{} does not take {}",
            crate::commands::PRODUCT,
            option.name()
        )));
    }
    let Some(object) = request.arguments.as_object() else {
        return Err(Failure::bad_request("arguments must be an object"));
    };
    if let Some(unbuilt) = crate::unbuilt::in_arguments(row.name, &request.arguments) {
        return Err(unbuilt.into());
    }
    if let Some(description) = declarations
        .descriptions()
        .iter()
        .find(|description| description.spec.name == row.name)
        && let Some(properties) = description.input_schema["properties"].as_object()
        && let Some(key) = object.keys().find(|key| !properties.contains_key(*key))
    {
        return Err(Failure::bad_request(format!("unknown argument {key}")));
    }
    for (argument, value) in object {
        let (Some(steps), Value::Array(values)) = (declarations.steps(row.name, argument), value)
        else {
            continue;
        };
        for (index, step) in values.iter().enumerate() {
            steps.validate_step(step).map_err(|mut failure| {
                failure.message = format!("{argument}[{index}]: {}", failure.message);
                Failure::from(failure)
            })?;
        }
    }
    Ok(())
}

/// The row whose operation is `O`.
pub(crate) fn row_of<O: Operation>() -> Option<&'static Command> {
    crate::commands::COMMANDS.iter().find(|row| {
        matches!(row.run, crate::commands::Run::Daemon(operation) if operation.operation() == TypeId::of::<O>())
    })
}

/// Answers one in-process request for operation `O` with typed arguments
/// and a typed answer: the web views, the daemon's own pages and tests.
pub(crate) fn ask<O: Operation>(
    daemon: &Daemon,
    arguments: O::Arguments,
    cursor: Option<String>,
    caller: &Caller,
) -> Answer<O::Reply> {
    let Some(row) = row_of::<O>() else {
        return Failure::internal("the operation has no row").answer();
    };
    let mut request = crate::client::request(
        row.name,
        match serde_json::to_value(&arguments) {
            Ok(value) => value,
            Err(error) => return Failure::internal(error).answer(),
        },
        polylogue_store::user_writes::new_request_id(),
    );
    request.cursor = cursor;
    let _running = match daemon.requests.enter() {
        Ok(running) => running,
        Err(failure) => return failure.answer(),
    };
    let mut output = Collected::default();
    let events = Transport {
        output: RefCell::new(&mut output),
        written: Cell::new(false),
    };
    let answer = respond::<O>(
        &Asked {
            daemon,
            row,
            request: &request,
            caller,
        },
        arguments,
        &events,
    );
    if events.written.get() {
        return Failure::internal("an in-process caller was sent events").answer();
    }
    answer
}

/// The steps every request takes once decoded: its policy, its checks, its
/// one read limit, then its operation on this thread under that limit.
fn respond<O: Operation>(
    accepted: &Asked<'_>,
    arguments: O::Arguments,
    events: &dyn Events,
) -> Answer<O::Reply> {
    let Asked {
        daemon,
        row,
        request,
        caller,
    } = *accepted;
    let policy = match O::policy(&arguments, caller.channel()) {
        Ok(policy) => policy,
        Err(failure) => return failure.answer(),
    };
    if let Err(failure) = refused_options(row, &policy, request) {
        return failure.answer();
    }
    if policy.changes_archive && daemon.blocked() {
        return daemon.write_refusal().answer();
    }
    let limit = limit(request.deadline, &policy, row.latency, caller);
    let scope = if policy.changes_archive {
        None
    } else {
        match daemon.reads.scope_token() {
            Ok(scope) => Some(scope),
            Err(error) => return Failure::store(&error).answer(),
        }
    };
    // A read meets the caller's bound in its limit; a write meets it when
    // the writer is about to take it.
    let invocation = caller.invocation(daemon, request);
    let guarded = Checked {
        inner: events,
        limit: limit.clone(),
        scope: scope.clone(),
        stopped: RefCell::new(None),
    };
    let query = RefCell::new(None);
    let call = Call {
        daemon,
        caller,
        invocation: &invocation,
        request_id: &request.request_id,
        cursor: request.cursor.as_deref(),
        budget: request.budget_tokens,
        limit: &limit,
        lane: policy.lane,
        after_write: None,
        output: &guarded,
        query: &query,
    };
    let _foreground = (policy.lane != Lane::Background).then(|| daemon.foreground());
    let _doing = polylogue_store::activity::doing_on("answering a request", row.name);
    let mut answer = within_limit(
        &limit,
        policy.bound != Bound::None,
        policy.late == Late::AnswerEmpty,
        row.name,
        &|| query.borrow().clone(),
        || {
            polylogue_model::caught(|| O::run(arguments, &call))
                .unwrap_or_else(|panic| Err(Failure::from(panic)))
                .unwrap_or_else(Failure::answer)
        },
    );
    if let Some(failure) = guarded.stopped.take() {
        answer = match limit.check() {
            Err(error) => {
                super::execution::stopped_read(&error, row.name, query.borrow().as_deref(), &limit)
            }
            Ok(()) => failure.answer(),
        };
    }
    if let Outcome::Error(failure) = &mut answer.outcome {
        // A cursor that no longer names its place suggests the same request
        // from the start, unless the operation named a better one.
        if failure.code.as_deref() == Some(polylogue_model::Code::StaleCursor.as_str())
            && failure.suggestion().is_none()
        {
            let mut start_over = request.clone();
            start_over.request_id = polylogue_store::user_writes::new_request_id();
            start_over.cursor = None;
            *failure = failure
                .clone()
                .with_suggestion(command_surface::Suggestion::Request(start_over));
        }
        // A read stopped by its deadline offers the same request without one.
        if failure.code.as_deref() == Some(polylogue_model::Code::DeadlineExceeded.as_str())
            && request.deadline != Deadline::None
            && policy.late == Late::Refuse
        {
            let mut lifted = request.clone();
            lifted.deadline = Deadline::None;
            answer = answer.with_next("complete without a deadline", lifted);
        }
    }
    let mut answer = match scope.as_ref().map(polylogue_store::ReadScopeToken::check) {
        Some(Err(error)) => Failure::store(&error).answer(),
        _ => answer,
    };
    let degraded = matches!(answer.outcome, Outcome::Degraded { .. });
    answer.envelope.mark_lower_bounds(degraded);
    answer
}

/// The envelope options a request's policy does not take: a page cursor
/// belongs to an operation that pages, a token budget to one that prepares
/// context.
pub(crate) fn refused_options(
    row: &Command,
    policy: &Policy,
    request: &RunnableRequest,
) -> Result<(), Failure> {
    let refused = [
        (request.cursor.is_some() && !policy.pages, "cursor"),
        (
            !request.budget_tokens.is_default() && !policy.budgeted,
            "budget_tokens",
        ),
    ];
    match refused.into_iter().find(|(used, _)| *used) {
        Some((_, option)) => Err(Failure::bad_request(format!(
            "{} does not take {option}",
            row.name
        ))),
        None => Ok(()),
    }
}

/// The events of one request, checked against its own limit and scope.
struct Checked<'e> {
    inner: &'e dyn Events,
    limit: ReadLimit,
    scope: Option<polylogue_store::ReadScopeToken>,
    stopped: RefCell<Option<Failure>>,
}

impl Events for Checked<'_> {
    fn write(&self, frames: &[u8]) -> std::io::Result<()> {
        let checked = self
            .limit
            .check()
            .and_then(|()| self.scope.as_ref().map_or(Ok(()), |scope| scope.check()));
        if let Err(error) = checked {
            self.stopped.replace(Some(Failure::store(&error)));
            return Err(std::io::Error::other(error));
        }
        self.inner.write(frames)
    }
}

/// The one read limit of a request: the caller's deadline, else its
/// latency class's, else the 30 s default, while it reads; within the
/// transport's bound; for the optional advisory, within its 20 ms cap.
pub(crate) fn limit(
    deadline: Deadline,
    policy: &Policy,
    class: command_surface::LatencyClass,
    caller: &Caller,
) -> ReadLimit {
    #[cfg(test)]
    if let Some(limit) = &caller.read_limit {
        return limit.clone();
    }
    let cancellation = caller.stop.child();
    if policy.bound == Bound::None {
        return ReadLimit::without_deadline(cancellation);
    }
    let declared = crate::commands::deadlines().get(class);
    let duration = match (policy.bound, deadline) {
        (Bound::CallerOnly, Deadline::Default) => None,
        (_, Deadline::Default) => {
            Some(declared.unwrap_or_else(super::execution::default_read_deadline))
        }
        (_, deadline) => deadline.duration(declared),
    };
    let now = std::time::Instant::now();
    let cap = (policy.late == Late::AnswerEmpty).then(|| now + crate::advisory::BUDGET);
    match [
        duration.and_then(|duration| now.checked_add(duration)),
        caller.outer_deadline,
        cap,
    ]
    .into_iter()
    .flatten()
    .min()
    {
        Some(at) => ReadLimit::at(at, cancellation),
        None => ReadLimit::without_deadline(cancellation),
    }
}

/// Runs `work` under the request's read limit. A read phase that has
/// already run out is refused before the work starts and, unless the
/// request answers empty when late, after it ends; a write without a read
/// phase runs outside the limit.
pub(crate) fn within_limit<T>(
    limit: &ReadLimit,
    read_phase: bool,
    fails_open: bool,
    stage: &str,
    query: &dyn Fn() -> Option<String>,
    work: impl FnOnce() -> Answer<T>,
) -> Answer<T> {
    if !read_phase {
        return work();
    }
    if fails_open {
        return limit.scope(work);
    }
    limit.scope(|| {
        if let Err(error) = limit.check() {
            return super::execution::stopped_read(&error, stage, query().as_deref(), limit);
        }
        let response = work();
        // A writer hand-over ends required reads; an answer prepared after
        // it keeps its result.
        if ReadLimit::current().is_some()
            && let Err(error) = limit.check()
        {
            return super::execution::stopped_read(&error, stage, query().as_deref(), limit);
        }
        response
    })
}

/// A finished request's answer, kept for in-process callers.
impl Collected {
    pub(crate) fn take(self) -> (Vec<u8>, Option<Answer<Value>>) {
        (self.events, self.answer)
    }
}

/// Answers one request on the calling thread for an in-process caller that
/// has only JSON: tests and the demo. Events are returned beside the answer.
pub(crate) fn answer_collected(
    daemon: &Daemon,
    request: &RunnableRequest,
    caller: &Caller,
) -> (Vec<u8>, Answer<Value>) {
    let mut output = Collected::default();
    answer(daemon, request, caller, &mut output);
    let (events, answer) = output.take();
    (
        events,
        answer.unwrap_or_else(|| Failure::internal("the request ended without an answer").answer()),
    )
}

#[cfg(test)]
mod tests;
