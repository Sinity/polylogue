//! A typed operation and its handler, declared together in the command table.
//!
//! Each daemon row names one [`Handler`]: its request and answer types and the
//! function that answers it. The row's [`Registration`] declares the types to
//! the shared operation declarations and binds the handler to the runtime
//! table from the same row, so an operation without a handler does not
//! compile and nothing joins two tables by name.
use super::Failure;
use super::query::{CountPageSender, CountReply};
use super::service::Daemon;
use crate::commands::{Caller, Command, Surface, ToolAccess};
use command_surface::{
    Access, Answer, Deadline, DeclarationsBuilder, FailureKind, JsonSchema, Operation,
    OperationFuture, OperationTable, RequestContext, RunnableRequest, TextStyle, TokenBudget, Via,
};
use polylogue_reading::operation::{Context, Invocation};
use polylogue_store::{cancellation::Cancellation, read::limit::ReadLimit};
use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;
use std::io::Write;
use std::marker::PhantomData;
use std::ops::ControlFlow;
use std::sync::Arc;

pub(crate) type CountReplyMap<T> = fn(CountReply) -> Answer<T>;

/// What a handler receives beside its typed arguments.
pub(crate) struct Call<'a> {
    pub daemon: &'a Daemon,
    /// The row being answered (`measure NAME` rows share one handler).
    pub row: &'static Command,
    pub invocation: &'a Invocation,
    pub caller: &'a Caller,
    /// The caller's retry identity for writes.
    pub request_id: &'a str,
    /// The page a continuing request resumes from.
    pub cursor: Option<&'a str>,
    pub budget: TokenBudget,
    #[cfg(test)]
    pub(crate) reader_acquired: &'a std::sync::atomic::AtomicBool,
}

impl Call<'_> {
    /// The evaluation context (clock, logical day, Sinex context) of this request.
    pub fn context(&self) -> Result<Arc<Context>, Failure> {
        let invocation = if self.invocation.working_directory().is_some() {
            self.invocation.clone()
        } else {
            self.invocation
                .clone()
                .with_working_directory(self.caller.cwd.clone())
        };
        self.daemon.resolve_operation_context(invocation)
    }

    /// A reader from the pool; a closed pool means the daemon is stopping.
    pub fn reader(&self) -> Result<polylogue_store::Reader<'_>, Failure> {
        let reader = self.daemon.reads.reader().map_err(|error| match error {
            polylogue_store::StoreError::ReadPoolClosed => Failure::shutting_down(error),
            error => Failure::from(error),
        })?;
        #[cfg(test)]
        self.reader_acquired
            .store(true, std::sync::atomic::Ordering::Release);
        Ok(reader)
    }

    /// The prepared-context sender a request from this surface is recorded
    /// under; none for the command line and HTTP.
    pub fn sender(&self) -> Option<String> {
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

/// One daemon operation: its request and answer types and what answers it.
/// The defaults describe a plain read; a row overrides what differs.
pub(crate) trait Handler: Send + Sync + 'static {
    type Arguments: Serialize + DeserializeOwned + JsonSchema + Send + 'static;
    type Answer: Serialize + DeserializeOwned + JsonSchema + Send + 'static;
    /// Continues a page from the request's `cursor`.
    const PAGES: bool = false;
    /// Prepares context within the request's token budget.
    const BUDGETED: bool = false;
    /// Which executor places its work may take: interactive answers may take
    /// the reserved place; idle-priority work runs in the background place.
    const CLASS: super::executor::Class = super::executor::Class::Ordinary;
    /// Argument-dependent work uses the same class at admission and execution.
    fn class(_arguments: &Self::Arguments) -> super::executor::Class {
        Self::CLASS
    }
    /// Answers with nothing, never with a timeout, when its limit runs out.
    const FAILS_OPEN: bool = false;
    /// Reads under a deadline before it writes.
    const READS_FIRST: bool = false;
    /// Records a delivery when the caller's surface names a sender, so it
    /// then writes.
    const RECORDS_DELIVERY: bool = false;
    /// Presents a prepared query page when its optional count outlasts
    /// the time available to reply.
    fn count_reply(_arguments: &Self::Arguments) -> Option<CountReplyMap<Self::Answer>> {
        None
    }
    /// The answer as text for a person or a model.
    const PRINT: fn(&Self::Answer, TextStyle) -> String = crate::cli::print::json::<Self::Answer>;

    /// Check argument and surface constraints before admitting any work.
    fn validate(_arguments: &Self::Arguments, _caller: &Caller) -> Result<(), Failure> {
        Ok(())
    }
    /// Whether this request leaves the archive unchanged; a request that
    /// changes it is refused while writes are blocked.
    fn read_only(_arguments: &Self::Arguments, row: &Command, caller: &Caller) -> bool {
        !row.writes && !(Self::RECORDS_DELIVERY && caller.sender().is_some())
    }
    /// Whether the request reads under a deadline before any write.
    fn read_phase(arguments: &Self::Arguments, row: &Command, caller: &Caller) -> bool {
        Self::READS_FIRST || Self::read_only(arguments, row, caller)
    }
    /// Whether a read without a caller deadline gets the default one.
    fn default_deadline(_arguments: &Self::Arguments) -> bool {
        true
    }
    /// The access this request needs; the declaration states the row's highest.
    fn access(_arguments: &Self::Arguments, declared: Access) -> Access {
        declared
    }
    /// The smallest tool that accepts these arguments.
    fn tool_access(_arguments: &Self::Arguments, row: &Command) -> ToolAccess {
        row.tool
            .map_or(ToolAccess::Read, crate::commands::Tool::access)
    }
    /// The query a refusal narrows, for a request that runs one.
    fn query_text(_arguments: &Self::Arguments) -> Option<&str> {
        None
    }
    /// The query this request executes with its ranking, whose query
    /// vector is asked for before the request holds a reader. Only requests
    /// that run a ranking name one.
    fn ranking_query(_arguments: &Self::Arguments, _row: &Command) -> Option<String> {
        None
    }
    /// A tighter bound than the request's own, such as an advisory's cap.
    fn read_cap() -> Option<std::time::Duration> {
        None
    }
    fn run(arguments: Self::Arguments, call: &Call<'_>) -> Result<Answer<Self::Answer>, Failure>;
}

/// A daemon operation that may also deliver its answer as a stream of
/// events written to the caller's socket.
pub(crate) trait Streams: Handler {
    fn streaming(arguments: &Self::Arguments) -> bool;
    fn produce(
        arguments: Self::Arguments,
        call: &Call<'_>,
        context: &Arc<Context>,
        output: &mut dyn Write,
    ) -> Result<(), Failure>;
}

/// A stream-only operation with no finite answer.
pub(crate) trait Feed: Send + Sync + 'static {
    type Arguments: Serialize + DeserializeOwned + JsonSchema + Send + 'static;
    fn produce(
        arguments: Self::Arguments,
        daemon: &Daemon,
        context: &Arc<Context>,
        output: &mut dyn Write,
    ) -> Result<(), Failure>;
}

/// How the daemon treats one decoded request before it runs.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Policy {
    pub read_only: bool,
    pub read_phase: bool,
    pub default_deadline: bool,
    pub streaming: bool,
    /// The executor places its work may take.
    pub class: super::executor::Class,
}

impl Policy {
    pub(crate) fn deadline(self, deadline: Deadline) -> Deadline {
        // Writes keep their durable result. An unbounded stream still takes
        // an explicit caller deadline.
        if !self.read_phase || (!self.default_deadline && deadline == Deadline::Default) {
            Deadline::None
        } else {
            deadline
        }
    }
}

/// One request answered on the calling thread, outside the runtime table:
/// the daemon's own pages and tests.
pub(crate) struct Local<'a> {
    pub daemon: &'a Daemon,
    pub invocation: &'a Invocation,
    pub caller: &'a Caller,
    pub limit: &'a ReadLimit,
    pub request_id: &'a str,
    pub cursor: Option<&'a str>,
}

/// One row's declaration, binding, request check, answer and stream
/// production, all derived from its handler type.
#[derive(Clone, Copy)]
pub struct Registration {
    declare: fn(&mut DeclarationsBuilder, &'static Command),
    bind: fn(Arc<Daemon>, &mut OperationTable, &'static Command),
    admit: fn(&Value, &'static Command, &Caller) -> Result<Policy, Failure>,
    answer: fn(&Value, &'static Command, &Local<'_>) -> Answer<Value>,
    tool_access: fn(&Value, &Command) -> Result<ToolAccess, Failure>,
}

/// A row answered by the daemon through `H`.
pub(crate) const fn daemon<H: Handler>() -> Registration {
    Registration {
        declare: declare::<H>,
        bind: bind::<H>,
        admit: admit::<H>,
        answer: answer_local::<H>,
        tool_access: tool_access::<H>,
    }
}

/// A row answered by the daemon through `H`, as a page or as a stream.
pub(crate) const fn streams<H: Streams>() -> Registration {
    Registration {
        declare: declare::<H>,
        bind: bind::<H>,
        admit: admit_stream::<H>,
        answer: answer_local::<H>,
        tool_access: tool_access::<H>,
    }
}

/// A stream-only row, with shared argument declarations and product stream delivery.
#[derive(Clone, Copy)]
pub struct FeedRegistration {
    declare: fn(&mut DeclarationsBuilder, &'static Command),
    validate: fn(&command_surface::RunnableRequest) -> Result<(), Failure>,
    prepare: fn(&command_surface::RunnableRequest) -> Result<super::streaming::Produce, Failure>,
}

pub(crate) const fn feed<F: Feed>() -> FeedRegistration {
    FeedRegistration {
        declare: declare_feed::<F>,
        validate: |request| feed_arguments::<F>(request).map(drop),
        prepare: prepare_feed::<F>,
    }
}

fn declare_feed<F: Feed>(declarations: &mut DeclarationsBuilder, row: &'static Command) {
    let mut spec = crate::commands::specification::<F::Arguments>(row);
    spec.kind = command_surface::OperationKind::Stream;
    if spec.reason.is_empty() {
        spec.reason = row.help.into();
    }
    declarations
        .declare_stream::<F::Arguments, Value>(spec, crate::cli::print::json::<Value>)
        .expect("feed arguments match their declaration");
}
impl FeedRegistration {
    pub(crate) fn declare(&self, declarations: &mut DeclarationsBuilder, row: &'static Command) {
        (self.declare)(declarations, row);
    }

    /// Check a feed request before sending it or offering it as a link.
    pub(crate) fn validate(
        &self,
        request: &command_surface::RunnableRequest,
    ) -> Result<(), Failure> {
        (self.validate)(request)
    }

    /// Check the envelope and decode the feed's arguments before opening a
    /// stream. Feed positions belong to those arguments, outside page cursors.
    pub(crate) fn prepare(
        &self,
        request: &command_surface::RunnableRequest,
    ) -> Result<super::streaming::Produce, Failure> {
        (self.prepare)(request)
    }
}

impl Registration {
    /// Decodes the arguments before deciding which tool can run them.
    pub(crate) fn tool_access(
        &self,
        arguments: &Value,
        row: &Command,
    ) -> Result<ToolAccess, Failure> {
        (self.tool_access)(arguments, row)
    }

    pub(crate) fn declare(&self, declarations: &mut DeclarationsBuilder, row: &'static Command) {
        (self.declare)(declarations, row)
    }
    pub(crate) fn bind(
        &self,
        daemon: Arc<Daemon>,
        table: &mut OperationTable,
        row: &'static Command,
    ) {
        (self.bind)(daemon, table, row)
    }
    /// Decodes the request's arguments and states how they are treated.
    pub(crate) fn admit(
        &self,
        arguments: &Value,
        row: &'static Command,
        caller: &Caller,
    ) -> Result<Policy, Failure> {
        (self.admit)(arguments, row, caller)
    }
    /// Answers one request on the calling thread.
    pub(crate) fn answer(
        &self,
        arguments: &Value,
        row: &'static Command,
        local: &Local<'_>,
    ) -> Answer<Value> {
        (self.answer)(arguments, row, local)
    }
}

fn decode<T: DeserializeOwned>(arguments: &Value) -> Result<T, Failure> {
    T::deserialize(arguments).map_err(Failure::bad_request)
}

fn tool_access<H: Handler>(arguments: &Value, row: &Command) -> Result<ToolAccess, Failure> {
    Ok(H::tool_access(&decode::<H::Arguments>(arguments)?, row))
}

fn declare<H: Handler>(declarations: &mut DeclarationsBuilder, row: &'static Command) {
    declarations
        .declare::<H::Arguments, H::Answer>(
            crate::commands::specification::<H::Arguments>(row),
            H::PRINT,
        )
        .expect("declared command is unique");
}

fn bind<H: Handler>(daemon: Arc<Daemon>, table: &mut OperationTable, row: &'static Command) {
    table
        .bind(
            row.name,
            Row::<H> {
                daemon,
                row,
                marker: PhantomData,
            },
        )
        .expect("a row binds its own declaration");
}

fn answer_local<H: Handler>(
    arguments: &Value,
    row: &'static Command,
    local: &Local<'_>,
) -> Answer<Value> {
    let arguments: H::Arguments = match decode(arguments) {
        Ok(arguments) => arguments,
        Err(failure) => return failure.answer(),
    };
    let answer = answer::<H>(
        arguments,
        local.daemon,
        row,
        local.invocation,
        local.caller,
        local.limit,
        local.request_id,
        local.cursor,
        TokenBudget::Default,
    );
    // The local caller consumes a wire value, just as a transport does.
    let mut problem = None;
    let serialized = answer.map(|value| {
        serde_json::to_value(value)
            .map_err(|error| problem = Some(error))
            .ok()
    });
    match problem {
        Some(error) => Failure::internal(error).answer(),
        None => serialized.map(|value| value.expect("serialized without a problem")),
    }
}

fn feed_arguments<F: Feed>(
    request: &command_surface::RunnableRequest,
) -> Result<F::Arguments, Failure> {
    if request.request_id.is_empty() {
        return Err(Failure::bad_request("request_id must not be empty"));
    }
    if !request.arguments.is_object() {
        return Err(Failure::bad_request("arguments must be an object"));
    }
    if let Some(option) = command_surface::RequestOption::ALL
        .into_iter()
        .find(|option| option.used_by(request))
    {
        return Err(Failure::bad_request(format!(
            "{} does not take {}",
            request.operation,
            option.name()
        )));
    }
    decode(&request.arguments)
}

fn prepare_feed<F: Feed>(
    request: &command_surface::RunnableRequest,
) -> Result<super::streaming::Produce, Failure> {
    let arguments = feed_arguments::<F>(request)?;
    Ok(Box::new(move |daemon, _, context, output| {
        super::service::accepting(daemon)?;
        F::produce(arguments, daemon, context, output)
    }))
}

fn admit<H: Handler>(
    arguments: &Value,
    row: &'static Command,
    caller: &Caller,
) -> Result<Policy, Failure> {
    let arguments: H::Arguments = decode(arguments)?;
    policy::<H>(&arguments, row, caller, false)
}

fn policy<H: Handler>(
    arguments: &H::Arguments,
    row: &'static Command,
    caller: &Caller,
    streaming: bool,
) -> Result<Policy, Failure> {
    H::validate(arguments, caller)?;
    Ok(Policy {
        read_only: H::read_only(arguments, row, caller),
        read_phase: H::read_phase(arguments, row, caller),
        default_deadline: H::default_deadline(arguments),
        streaming,
        class: H::class(arguments),
    })
}

fn admit_stream<H: Streams>(
    arguments: &Value,
    row: &'static Command,
    caller: &Caller,
) -> Result<Policy, Failure> {
    let arguments: H::Arguments = decode(arguments)?;
    stream_policy::<H>(&arguments, row, caller)
}

fn stream_policy<H: Streams>(
    arguments: &H::Arguments,
    row: &'static Command,
    caller: &Caller,
) -> Result<Policy, Failure> {
    policy::<H>(arguments, row, caller, H::streaming(arguments))
}

#[allow(clippy::too_many_arguments)]
fn produce<H: Streams>(
    arguments: H::Arguments,
    row: &'static Command,
    daemon: &Daemon,
    invocation: &Invocation,
    caller: &Caller,
    context: &Arc<Context>,
    output: &mut dyn Write,
) -> Result<(), Failure> {
    #[cfg(test)]
    let acquired = std::sync::atomic::AtomicBool::new(false);
    let call = Call {
        daemon,
        row,
        invocation,
        caller,
        request_id: "",
        cursor: None,
        budget: TokenBudget::Default,
        #[cfg(test)]
        reader_acquired: &acquired,
    };
    super::service::accepting(daemon)?;
    H::produce(arguments, &call, context, output)
}

pub(crate) enum NativeReply {
    Refused(Answer<Value>),
    Answer(NativeAnswer, Answer<Value>),
    Stream {
        request: RunnableRequest,
        accepted: Box<crate::operation::Accepted>,
        policy: Policy,
        produce: super::streaming::Produce,
    },
}

pub(crate) struct NativeAnswer {
    pub web_request: Option<Arc<RunnableRequest>>,
    pub scope: Option<polylogue_store::ReadScopeToken>,
    pub publication_deadline: Option<std::time::Instant>,
}

/// Select native framing before finite execution. Rows use the arguments
/// already decoded by the binding, including for an HTTP GET check.
pub(crate) async fn dispatch_native(
    daemon: &Daemon,
    table: &OperationTable,
    row: &'static Command,
    mut request: RunnableRequest,
    mut accepted: crate::operation::Accepted,
    via: Via,
    http_get: bool,
) -> NativeReply {
    use crate::daemon::service::line::Line;
    use crate::protocol::line::{Rows, Step};
    let surface = match via {
        Via::Web => Surface::Http,
        Via::Mcp => Surface::Mcp,
        Via::Hook => Surface::Hook,
        _ => Surface::Cli,
    };
    let table_via = match via {
        Via::Web | Via::Mcp => via,
        _ => Via::Cli,
    };
    let caller = Caller {
        surface,
        cwd: accepted.working_directory.clone(),
    };
    if row.name != "rows" {
        let registration = row.registration().expect("native daemon command");
        let policy = match registration.admit(&request.arguments, row, &caller) {
            Ok(policy) => policy,
            Err(failure) => return NativeReply::Refused(failure.answer()),
        };
        request.deadline = policy.deadline(request.deadline);
        let (trusted, ready) = match prepare_native_answer(
            daemon,
            row,
            &request,
            request.deadline,
            accepted,
            via,
            policy,
        ) {
            Ok(call) => call,
            Err(error) => return NativeReply::Refused(super::service::store_error(&error)),
        };
        let answer = table
            .call_as(request, table_via, trusted, Default::default())
            .await;
        return NativeReply::Answer(ready, answer);
    }

    let mut ready = None;
    let answer = table
        .call_with_admission(
            request,
            table_via,
            Default::default(),
            |rows: &Rows, raw| {
                let admit = || -> Result<_, Failure> {
                    if http_get
                        && (rows.steps.len() > 1
                            || rows
                                .steps
                                .iter()
                                .any(|step| !matches!(step, Step::Read(_) | Step::Brief(_))))
                    {
                        return Err(Failure::bad_request("use POST /api/v1/rows"));
                    }
                    let policy = stream_policy::<Line>(rows, row, &caller)?;
                    let deadline = policy.deadline(raw.deadline);
                    if policy.streaming {
                        if policy.read_only && via != Via::Web {
                            accepted.read_scope = Some(daemon.reads.scope_token()?);
                        }
                        let mut request = raw.clone();
                        request.deadline = deadline;
                        let rows = rows.clone();
                        return Ok(ControlFlow::Break(NativeReply::Stream {
                            request,
                            accepted: Box::new(accepted),
                            policy,
                            produce: Box::new(move |daemon, invocation, context, output| {
                                produce::<Line>(
                                    rows, row, daemon, invocation, &caller, context, output,
                                )
                            }),
                        }));
                    }
                    let (trusted, finite) =
                        prepare_native_answer(daemon, row, raw, deadline, accepted, via, policy)?;
                    ready = Some(finite);
                    Ok(ControlFlow::Continue((trusted, deadline)))
                };
                admit().unwrap_or_else(|failure| {
                    ControlFlow::Break(NativeReply::Refused(failure.answer()))
                })
            },
        )
        .await;
    match answer {
        Err(error) => NativeReply::Refused(Failure::bad_request(error).answer()),
        Ok(ControlFlow::Break(reply)) => reply,
        Ok(ControlFlow::Continue(answer)) => match ready {
            Some(ready) => NativeReply::Answer(ready, answer),
            None => NativeReply::Refused(answer),
        },
    }
}

fn prepare_native_answer(
    daemon: &Daemon,
    row: &'static Command,
    request: &RunnableRequest,
    deadline: Deadline,
    mut accepted: crate::operation::Accepted,
    via: Via,
    policy: Policy,
) -> polylogue_store::Result<(command_surface::Caller, NativeAnswer)> {
    if policy.read_only {
        accepted.read_scope = Some(daemon.reads.scope_token()?);
    }
    let (web_request, publication_deadline) = if via == Via::Web {
        let mut request = request.clone();
        request.deadline = deadline;
        let request = Arc::new(request);
        accepted.invocation = Some(
            daemon
                .invocation(&accepted.authorship)
                .with_accepted_request(Some(request.clone()))
                .with_calling_session(accepted.calling_session.clone())
                .with_parent(accepted.parent.clone()),
        );
        (Some(request), None)
    } else {
        let parent = accepted
            .parent
            .as_ref()
            .and_then(|parent| parent.read_limit(accepted.cancellation.clone()).deadline());
        let publication_deadline = super::execution::request_limit(
            deadline,
            row.latency,
            true,
            parent,
            accepted.cancellation.clone(),
        )
        .deadline();
        (None, publication_deadline)
    };
    let scope = accepted.read_scope.clone();
    let trusted = command_surface::Caller::authenticated(accepted.scope(), accepted)
        .with_access(Access::Write);
    Ok((
        trusted,
        NativeAnswer {
            web_request,
            scope,
            publication_deadline,
        },
    ))
}

/// The caller as handlers see it: its surface, working directory and output.
pub(crate) fn caller(context: &RequestContext) -> Caller {
    let accepted = context.caller().identity::<crate::operation::Accepted>();
    Caller {
        surface: if accepted.is_some_and(|accepted| {
            accepted.authorship.via == polylogue_model::authorship::Channel::Hook
        }) {
            Surface::Hook
        } else {
            match context.via {
                command_surface::Via::Web => Surface::Http,
                command_surface::Via::Mcp => Surface::Mcp,
                _ => Surface::Cli,
            }
        },
        cwd: accepted.and_then(|caller| caller.working_directory.clone()),
    }
}

/// The handler of one row bound to the runtime table.
struct Row<H> {
    daemon: Arc<Daemon>,
    row: &'static Command,
    marker: PhantomData<H>,
}

/// A dropped shared operation cancels its blocking reader before releasing ownership.
struct CancelRead(Cancellation);
impl Drop for CancelRead {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

impl<H: Handler> Operation for Row<H> {
    type Request = H::Arguments;
    type Response = H::Answer;

    fn required_access(
        &self,
        arguments: &Self::Request,
        declared: Access,
        _caller: &command_surface::Caller,
    ) -> Access {
        H::access(arguments, declared)
    }

    fn execute(
        &self,
        arguments: Self::Request,
        context: RequestContext,
    ) -> OperationFuture<'_, Answer<Self::Response>> {
        Box::pin(async move {
            if context.request_id.is_empty() {
                return Answer::error(command_surface::Failure::new(
                    FailureKind::BadRequest,
                    "request_id must not be empty",
                ));
            }
            if !H::PAGES && context.cursor().is_some() {
                return Answer::error(command_surface::Failure::new(
                    FailureKind::BadRequest,
                    "operation does not continue a page",
                ));
            }
            let budget = context.request().budget_tokens;
            if !H::BUDGETED && !budget.is_default() {
                return Answer::error(command_surface::Failure::new(
                    FailureKind::BadRequest,
                    "operation does not prepare budgeted context",
                ));
            }
            let narrower = H::query_text(&arguments)
                .and_then(crate::daemon::execution::narrower)
                .and_then(|text| {
                    let mut request = context.request().clone();
                    let object = request.arguments.as_object_mut()?;
                    let field = if self.row.name == "rows" {
                        "selection"
                    } else {
                        "query"
                    };
                    object.get(field)?;
                    object.insert(field.into(), Value::String(text));
                    request.request_id = polylogue_store::user_writes::new_request_id();
                    request.deadline = command_surface::Deadline::Default;
                    request.cursor = None;
                    request.budget_tokens = TokenBudget::Default;
                    Some(request)
                });
            if let Err(error) = context.progress_unknown(self.row.name, narrower) {
                return Answer::error(error);
            }
            if let Err(error) = context.check_cancelled() {
                return Answer::error(error);
            }
            let caller = caller(&context);
            let accepted = context.caller().identity::<crate::operation::Accepted>();
            let cancellation = accepted.map_or_else(
                || self.daemon.archive.stop().child(),
                |caller| caller.cancellation.child(),
            );
            let authorship = accepted
                .map(|caller| caller.authorship.clone())
                .unwrap_or_else(|| {
                    command_surface::author_of(
                        &command_surface::CallerEvidence::default(),
                        context.via,
                        None,
                    )
                    .into()
                });
            let invocation = accepted.and_then(|caller| caller.invocation.clone());
            let calling_session = accepted.and_then(|caller| caller.calling_session.clone());
            let parent = accepted.and_then(|caller| caller.parent.clone());
            let request_id = context.request_id.clone();
            let cursor = context.cursor().map(str::to_owned);
            // A read whose class has no deadline and whose caller named none
            // still stops at the default read deadline.
            let default_limit = parent.is_none()
                && context.read_deadline().is_none()
                && matches!(
                    context.request().deadline,
                    command_surface::Deadline::Default
                )
                && H::default_deadline(&arguments)
                && H::read_phase(&arguments, self.row, &caller);
            let guard = CancelRead(cancellation.clone());
            let daemon = Arc::clone(&self.daemon);
            let row = self.row;
            let table_context = context.clone();
            let deadline = match &parent {
                Some(parent) => parent.read_limit(cancellation.clone()).deadline(),
                None if default_limit => {
                    super::execution::limit(None, cancellation.clone()).deadline()
                }
                None => None,
            };
            let table_deadline = context.read_deadline().map(tokio::time::Instant::into_std);
            let count_deadline = [deadline, table_deadline].into_iter().flatten().min();
            let count_reply = H::count_reply(&arguments)
                .zip(count_deadline)
                .filter(|_| H::read_only(&arguments, self.row, &caller));
            let deadline = if count_reply.is_some() {
                count_deadline
            } else {
                deadline
            };
            let (count_sender, pending_count) =
                count_reply.map_or((None, None), |(map, deadline)| {
                    let (sender, receiver) = tokio::sync::oneshot::channel();
                    (Some(sender), Some((receiver, map, deadline)))
                });
            let class = H::class(&arguments);
            let job = move || {
                let _table = TableContext::enter(table_context, count_sender);
                let limit = match deadline {
                    Some(deadline) => ReadLimit::at(deadline, cancellation),
                    None => ReadLimit::without_deadline(cancellation),
                };
                let invocation = invocation
                    .unwrap_or_else(|| daemon.invocation(&authorship))
                    .with_calling_session(calling_session)
                    .with_parent(parent);
                answer::<H>(
                    arguments,
                    &daemon,
                    row,
                    &invocation,
                    &caller,
                    &limit,
                    &request_id,
                    cursor.as_deref(),
                    budget,
                )
            };
            let completed = async {
                self.daemon
                    .executor
                    .run(class, deadline, job)
                    .await
                    .unwrap_or_else(|| Answer::error(super::service::shutting_down().error))
            };
            let deferred = async {
                if let Some((receiver, map, deadline)) = pending_count
                    && let Ok(page) = receiver.await
                {
                    let left = deadline.saturating_duration_since(std::time::Instant::now());
                    let reply_at = deadline - super::service::answer_reserve(left);
                    tokio::time::sleep_until(reply_at.into()).await;
                    if let Err(error) = guard.0.check() {
                        return super::service::store_error(&error);
                    }
                    if let Some(answer) = page.without_count() {
                        return map(answer);
                    }
                }
                std::future::pending().await
            };
            let answer = tokio::select! {
                biased;
                answer = completed => answer,
                answer = deferred => answer,
            };
            // If the page won, the count still owns its executor place
            // until SQLite returns. Cancelling it here does not free a
            // place for another blocked count.
            drop(guard);
            answer
        })
    }
}

/// Runs one handler on the calling thread under the request's read limit.
#[allow(clippy::too_many_arguments)]
fn answer<H: Handler>(
    arguments: H::Arguments,
    daemon: &Daemon,
    row: &'static Command,
    invocation: &Invocation,
    caller: &Caller,
    limit: &ReadLimit,
    request_id: &str,
    cursor: Option<&str>,
    budget: TokenBudget,
) -> Answer<H::Answer> {
    if let Err(failure) = super::service::accepting(daemon) {
        return failure.answer();
    }
    if let Err(failure) = H::validate(&arguments, caller) {
        return failure.answer();
    }
    let capped = H::read_cap().map(|cap| {
        let cap = std::time::Instant::now() + cap;
        ReadLimit::at(
            limit.deadline().map_or(cap, |parent| parent.min(cap)),
            limit.cancellation.child(),
        )
    });
    let limit = capped.as_ref().unwrap_or(limit);
    let reader_acquired = Arc::new(std::sync::atomic::AtomicBool::new(false));
    #[cfg(test)]
    if let Some(sender) = daemon.read_limits.lock().unwrap().as_ref() {
        let _ = sender.send(crate::test_daemon::ReadObservation {
            limit: limit.clone(),
            owner: std::thread::current().id(),
            acquired: Arc::clone(&reader_acquired),
        });
    }
    if let Err(error) = invocation.check_parent_lifetime() {
        return super::service::store_error(&error);
    }
    let read_only = H::read_only(&arguments, row, caller);
    let read_phase = H::read_phase(&arguments, row, caller);
    let text = H::query_text(&arguments).map(str::to_owned);
    let class = H::class(&arguments);
    let work = || {
        if daemon.blocked() && !read_only {
            return daemon.write_refusal().answer();
        }
        let _foreground =
            (class != super::executor::Class::Background).then(|| daemon.foreground());
        let call = Call {
            daemon,
            row,
            invocation,
            caller,
            request_id,
            cursor,
            budget,
            #[cfg(test)]
            reader_acquired: &reader_acquired,
        };
        if let Some(text) = H::ranking_query(&arguments, row) {
            super::query::prefetch_ranking(&call, &text);
        }
        H::run(arguments, &call).unwrap_or_else(Failure::answer)
    };
    #[cfg(not(test))]
    let _ = &reader_acquired;
    let bounded = || {
        within_limit(
            limit,
            read_phase,
            H::FAILS_OPEN,
            row.name,
            text.as_deref(),
            work,
        )
    };
    if class == super::executor::Class::Interactive {
        polylogue_store::reserved_reads(bounded)
    } else {
        bounded()
    }
}

/// Runs `work` under the request's read limit. A read phase that has
/// already run out is refused before the work starts and, unless the row
/// fails open, after it ends; a write without a read phase runs outside.
pub(crate) fn within_limit<T>(
    limit: &ReadLimit,
    read_phase: bool,
    fails_open: bool,
    stage: &str,
    text: Option<&str>,
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
            return super::execution::refusal(&error, stage, text, limit);
        }
        let response = work();
        // A checked writer handoff or completed primary answer ends required
        // reads. An optional count can then time out without discarding rows.
        if ReadLimit::current().is_some()
            && let Err(error) = limit.check()
        {
            return super::execution::refusal(&error, stage, text, limit);
        }
        response
    })
}

thread_local! {
    /// The operation table's context of the request this thread is answering.
    static TABLE_CONTEXT: std::cell::RefCell<Option<TableRequest>> =
        const { std::cell::RefCell::new(None) };
}

struct TableRequest {
    context: RequestContext,
    count_sender: Option<CountPageSender>,
}

/// Holds the table context for the request a worker thread answers, and
/// forgets it when the answer is done.
struct TableContext;
impl TableContext {
    fn enter(context: RequestContext, count_sender: Option<CountPageSender>) -> Self {
        TABLE_CONTEXT.with(|current| {
            current.replace(Some(TableRequest {
                context,
                count_sender,
            }))
        });
        Self
    }
}
impl Drop for TableContext {
    fn drop(&mut self) {
        TABLE_CONTEXT.with(|current| current.replace(None));
    }
}

pub(super) fn count_page_sender() -> Option<CountPageSender> {
    TABLE_CONTEXT.with(|current| current.borrow_mut().as_mut()?.count_sender.take())
}

/// Ends required reads before an acknowledged mutation or after the primary
/// answer is prepared. An already expired read refuses this boundary; optional
/// aggregates keep their original limit without replacing completed rows.
pub(crate) fn finish_read_phase(stage: &'static str) -> Result<(), Failure> {
    ReadLimit::finish_scope(stage).map_err(Failure::from)?;
    TABLE_CONTEXT.with(|current| match current.borrow().as_ref() {
        Some(request) => request.context.finish_read_phase().map_err(Failure::from),
        None => Ok(()),
    })
}

/// Answers one request on the calling thread as the socket would, for the
/// daemon's own pages and for tests.
pub(crate) fn local(operation: &str, arguments: Value, local: &Local<'_>) -> Answer<Value> {
    let Some(row) = crate::commands::command(operation) else {
        return Failure::bad_request(format!("unknown operation {operation}")).answer();
    };
    let Some(registration) = row.registration() else {
        return Failure::bad_request(format!("{operation} runs in the calling process")).answer();
    };
    registration.answer(&arguments, row, local)
}

#[cfg(test)]
mod tests;
