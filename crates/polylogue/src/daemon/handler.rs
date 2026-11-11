//! A typed operation and its handler, declared together in the command table.
//!
//! Each daemon row names one [`Handler`]: its request and answer types and the
//! function that answers it. The row's [`Registration`] declares the types to
//! the shared operation declarations and binds the handler to the runtime
//! table from the same row, so an operation without a handler does not
//! compile and nothing joins two tables by name.
use super::Failure;
use super::service::Daemon;
use crate::commands::{Caller, Command, Surface};
use crate::operation::{Context, Invocation};
use command_surface::{
    Access, Answer, DeclarationsBuilder, FailureKind, JsonSchema, Operation, OperationFuture,
    OperationTable, RequestContext, TextStyle, TokenBudget,
};
use polylogue_store::read::limit::{ReadCancellation, ReadLimit};
use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;
use std::io::Write;
use std::marker::PhantomData;
use std::sync::Arc;

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
    /// The evaluation context (clock, logical day, owner) of this request.
    pub fn context(&self) -> Result<Arc<Context>, Failure> {
        self.daemon
            .resolve_operation_context(self.invocation.clone())
    }

    /// A reader from the pool; a closed pool means the daemon is stopping.
    pub fn reader(&self) -> Result<polylogue_store::Reader<'_>, Failure> {
        let reader = self.daemon.reads.reader().map_err(|error| match error {
            polylogue_store::StoreError::ReadPoolClosed => Failure::new("shutting_down", error),
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
            (Some(_), None) => Err(Failure::new(
                "bad_request",
                "`cwd` must be an absolute path",
            )),
            (None, Some(base)) => Ok(base.clone()),
            (None, None) => Err(Failure::new("bad_request", "`cwd` is required")),
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
        Failure::new(
            "bad_request",
            format!(
                "a relative {what} requires the caller's working directory; supply an absolute path"
            ),
        )
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
    /// Answers with nothing, never with a timeout, when its limit runs out.
    const FAILS_OPEN: bool = false;
    /// Reads under a deadline before it writes.
    const READS_FIRST: bool = false;
    /// Records a delivery when the caller's surface names a sender, so it
    /// then writes.
    const RECORDS_DELIVERY: bool = false;
    /// The answer as text for a person or a model.
    const PRINT: fn(&Value, bool) -> String = crate::cli::print::json;

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
    fn run(arguments: Self::Arguments, call: &Call<'_>) -> Result<Answer<Value>, Failure>;
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

/// A stream the command line follows over the socket; it has no finite answer.
pub(crate) trait Feed: Send + Sync + 'static {
    type Arguments: DeserializeOwned + Send + 'static;
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

type Produce = fn(
    &Value,
    &'static Command,
    &Daemon,
    &Invocation,
    &Caller,
    &Arc<Context>,
    &mut dyn Write,
) -> Result<(), Failure>;

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
    produce: Option<Produce>,
}

/// A row answered by the daemon through `H`.
pub(crate) const fn daemon<H: Handler>() -> Registration {
    Registration {
        declare: declare::<H>,
        bind: bind::<H>,
        admit: admit::<H>,
        answer: answer_local::<H>,
        produce: None,
    }
}

/// A row answered by the daemon through `H`, as a page or as a stream.
pub(crate) const fn streams<H: Streams>() -> Registration {
    Registration {
        declare: declare::<H>,
        bind: bind::<H>,
        admit: admit_stream::<H>,
        answer: answer_local::<H>,
        produce: Some(produce::<H>),
    }
}

/// A stream-only row; the shared declarations do not hold it.
#[derive(Clone, Copy)]
pub struct FeedRegistration {
    produce: ProduceFeed,
}

type ProduceFeed = fn(&Value, &Daemon, &Arc<Context>, &mut dyn Write) -> Result<(), Failure>;

pub(crate) const fn feed<F: Feed>() -> FeedRegistration {
    FeedRegistration {
        produce: produce_feed::<F>,
    }
}

impl Registration {
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
    /// Writes the stream a streaming request asks for.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn produce(
        &self,
        arguments: &Value,
        row: &'static Command,
        daemon: &Daemon,
        invocation: &Invocation,
        caller: &Caller,
        context: &Arc<Context>,
        output: &mut dyn Write,
    ) -> Result<(), Failure> {
        match self.produce {
            Some(produce) => produce(arguments, row, daemon, invocation, caller, context, output),
            None => Err(Failure::new(
                "bad_request",
                format!("{} answers once; it has no stream", row.name),
            )),
        }
    }
}

impl FeedRegistration {
    pub(crate) fn produce(
        &self,
        arguments: &Value,
        daemon: &Daemon,
        context: &Arc<Context>,
        output: &mut dyn Write,
    ) -> Result<(), Failure> {
        (self.produce)(arguments, daemon, context, output)
    }
}

fn decode<T: DeserializeOwned>(arguments: &Value) -> Result<T, Failure> {
    T::deserialize(arguments).map_err(|error| Failure::new("bad_request", error))
}

fn declare<H: Handler>(declarations: &mut DeclarationsBuilder, row: &'static Command) {
    declarations
        .declare::<H::Arguments, H::Answer>(crate::commands::specification(row), render::<H>)
        .expect("declared command is unique");
}

fn render<H: Handler>(answer: &H::Answer, style: TextStyle) -> String {
    let value = serde_json::to_value(answer).expect("typed answer serializes");
    (H::PRINT)(&value, style == TextStyle::Ansi)
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
    answer::<H>(
        arguments,
        local.daemon,
        row,
        local.invocation,
        local.caller,
        local.limit,
        local.request_id,
        local.cursor,
        TokenBudget::Default,
    )
}

fn admit<H: Handler>(
    arguments: &Value,
    row: &'static Command,
    caller: &Caller,
) -> Result<Policy, Failure> {
    let arguments: H::Arguments = decode(arguments)?;
    Ok(Policy {
        read_only: H::read_only(&arguments, row, caller),
        read_phase: H::read_phase(&arguments, row, caller),
        default_deadline: H::default_deadline(&arguments),
        streaming: false,
        class: H::CLASS,
    })
}

fn admit_stream<H: Streams>(
    arguments: &Value,
    row: &'static Command,
    caller: &Caller,
) -> Result<Policy, Failure> {
    let arguments: H::Arguments = decode(arguments)?;
    Ok(Policy {
        read_only: H::read_only(&arguments, row, caller),
        read_phase: H::read_phase(&arguments, row, caller),
        default_deadline: H::default_deadline(&arguments),
        streaming: H::streaming(&arguments),
        class: H::CLASS,
    })
}

#[allow(clippy::too_many_arguments)]
fn produce<H: Streams>(
    arguments: &Value,
    row: &'static Command,
    daemon: &Daemon,
    invocation: &Invocation,
    caller: &Caller,
    context: &Arc<Context>,
    output: &mut dyn Write,
) -> Result<(), Failure> {
    let arguments: H::Arguments = decode(arguments)?;
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

fn produce_feed<F: Feed>(
    arguments: &Value,
    daemon: &Daemon,
    context: &Arc<Context>,
    output: &mut dyn Write,
) -> Result<(), Failure> {
    let arguments: F::Arguments = decode(arguments)?;
    super::service::accepting(daemon)?;
    F::produce(arguments, daemon, context, output)
}

/// The caller as handlers see it: its surface, working directory and output.
pub(crate) fn caller(context: &RequestContext) -> Caller {
    let accepted = context
        .caller()
        .identity::<crate::protocol::caller::Accepted>();
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
struct CancelRead(ReadCancellation);
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
                    object.get("query")?;
                    object.insert("query".into(), Value::String(text));
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
            let accepted = context
                .caller()
                .identity::<crate::protocol::caller::Accepted>();
            let cancellation = accepted
                .map(|caller| caller.cancellation.child())
                .unwrap_or_default();
            let authorship = accepted
                .map(|caller| caller.authorship.clone())
                .unwrap_or_else(|| {
                    let via = match context.via {
                        command_surface::Via::Web => polylogue_model::authorship::Channel::Web,
                        command_surface::Via::Mcp => polylogue_model::authorship::Channel::Mcp,
                        _ => polylogue_model::authorship::Channel::Cli,
                    };
                    polylogue_model::authorship::compute_authorship(via, &Default::default(), None)
                });
            let invocation = accepted.and_then(|caller| caller.invocation.clone());
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
            let job = move || {
                let _table = TableContext::enter(table_context);
                let limit = match (&parent, deadline) {
                    (Some(parent), _) => parent.read_limit(cancellation),
                    (None, Some(deadline)) => ReadLimit::at(deadline, cancellation),
                    (None, None) => ReadLimit::without_deadline(cancellation),
                };
                let invocation = invocation
                    .unwrap_or_else(|| daemon.invocation(&authorship))
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
            let Some(answer) = self.daemon.executor.run(H::CLASS, deadline, job).await else {
                return Answer::error(super::service::shutting_down().error);
            };
            drop(guard);
            typed(answer, self.row.name)
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
) -> Answer<Value> {
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
    let work = || {
        if let Err(failure) = super::service::accepting(daemon) {
            return failure.answer();
        }
        if daemon.blocked() && !read_only {
            return daemon.write_refusal().answer();
        }
        let _foreground =
            (H::CLASS != super::executor::Class::Background).then(|| daemon.foreground());
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
    if H::CLASS == super::executor::Class::Interactive {
        polylogue_store::reserved_reads(bounded)
    } else {
        bounded()
    }
}

/// Runs `work` under the request's read limit. A read phase that has
/// already run out is refused before the work starts and, unless the row
/// fails open, after it ends; a write without a read phase runs outside.
pub(crate) fn within_limit(
    limit: &ReadLimit,
    read_phase: bool,
    fails_open: bool,
    stage: &str,
    text: Option<&str>,
    work: impl FnOnce() -> Answer<Value>,
) -> Answer<Value> {
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
        // A checked writer handoff ends the read phase. Its accepted result
        // must not become a timeout after the write commits.
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
    static TABLE_CONTEXT: std::cell::RefCell<Option<RequestContext>> =
        const { std::cell::RefCell::new(None) };
}

/// Holds the table context for the request a worker thread answers, and
/// forgets it when the answer is done.
struct TableContext;
impl TableContext {
    fn enter(context: RequestContext) -> Self {
        TABLE_CONTEXT.with(|current| current.replace(Some(context)));
        Self
    }
}
impl Drop for TableContext {
    fn drop(&mut self) {
        TABLE_CONTEXT.with(|current| current.replace(None));
    }
}

/// Ends the read phase before a mutation is handed to its writer: the
/// daemon's read limit stops applying, and the operation table no longer
/// answers a timeout for this request. A read phase that has already run
/// out refuses the handoff instead, so a write is never handed over and
/// then reported as timed out.
pub(crate) fn finish_read_phase(stage: &'static str) -> Result<(), Failure> {
    ReadLimit::finish_scope(stage).map_err(Failure::from)?;
    TABLE_CONTEXT.with(|current| match current.borrow().as_ref() {
        Some(context) => context.finish_read_phase().map_err(Failure::from),
        None => Ok(()),
    })
}

/// The daemon's answer decoded into the operation's declared answer type;
/// the outcome, notes and next actions pass through unchanged.
fn typed<T: DeserializeOwned>(answer: Answer<Value>, name: &str) -> Answer<T> {
    let mut problem = None;
    let decoded = answer.map(|value| {
        serde_json::from_value::<T>(value)
            .map_err(|error| problem = Some(error))
            .ok()
    });
    match problem {
        Some(error) => Answer::error(crate::protocol::protocol_mismatch(name, error)),
        None => decoded.map(|value| value.expect("decoded without a problem")),
    }
}

/// Answers one request on the calling thread as the socket would, for the
/// daemon's own pages and for tests.
pub(crate) fn local(operation: &str, arguments: Value, local: &Local<'_>) -> Answer<Value> {
    let Some(row) = crate::commands::command(operation) else {
        return Failure::new("bad_request", format!("unknown operation {operation}")).answer();
    };
    let Some(registration) = row.registration() else {
        return Failure::new(
            "bad_request",
            format!("{operation} runs in the calling process"),
        )
        .answer();
    };
    registration.answer(&arguments, row, local)
}
