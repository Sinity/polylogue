//! Application operations and their shared archive state. Socket, HTTP and
//! MCP adapters use the same typed requests; transport framing stays outside.

mod execution;
mod status;

use super::{Failure, Request};
use crate::briefing::{self, Ask};
use crate::operation::{Context, Invocation};
use command_surface::Answer;
use polylogue_engine::{Config, Engine, Lifetime, LoopMsg, LoopStatus, RunSummary};
use polylogue_model::user::{Delivery, UserWrite};
use polylogue_store::user_writes;
use polylogue_store::{Archive, ReadPool, StoreError, WriterSender};
use serde_json::Value;
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Sender, channel, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

/// What request handlers share.
pub struct Daemon {
    pub(crate) config: Arc<crate::config::Config>,
    pub archive: Archive,
    pub reads: ReadPool,
    pub(crate) completion_path: std::path::PathBuf,
    pub(crate) callers: crate::protocol::caller::hooks::Registry,
    operations: std::sync::OnceLock<std::sync::Weak<command_surface::OperationTable>>,

    pub(crate) warm: super::warm::Warm,
    /// Sessions by working directory, kept current by the scope worker.
    pub(crate) scope: super::scope::SharedScopeIndex,
    /// Briefs answered whose delivery records are still queued.
    pub(crate) queued_deliveries: super::queued_deliveries::QueuedDeliveries,
    /// User writes (notes, deliveries, settings) through the one writer.
    pub(crate) writes: Option<WriterSender>,
    /// Live-session acts: the mailbox, resumed runs and hosted sessions.
    pub(crate) acts: crate::live_acts::Acts,
    mode: ModeState,
    loop_tx: Mutex<Sender<LoopMsg>>,
    pub(crate) status: Arc<Mutex<LoopStatus>>,
    started: Instant,
    /// The signal rules in force and the ones refused, for `status`.
    pub(crate) signals: Mutex<serde_json::Value>,
    #[cfg(test)]
    pub(crate) signal_trace: Mutex<Option<Sender<crate::signals::observations::Phase>>>,
    stopping: AtomicBool,
    pub(super) idle_admission: super::schema::Admission,
    #[cfg(test)]
    pub(crate) read_limits: Mutex<Option<Sender<crate::test_daemon::ReadObservation>>>,
}

impl Daemon {
    pub(crate) fn invocation(
        &self,
        authorship: &polylogue_model::authorship::Authorship,
    ) -> crate::operation::Invocation {
        crate::operation::Invocation::capture(&self.config.requests, authorship)
    }
    pub(crate) fn operation_context(
        &self,
        authorship: &polylogue_model::authorship::Authorship,
    ) -> Result<Arc<crate::operation::Context>, Failure> {
        self.resolve_operation_context(self.invocation(authorship))
    }
    pub(crate) fn resolve_operation_context(
        &self,
        invocation: crate::operation::Invocation,
    ) -> Result<Arc<crate::operation::Context>, Failure> {
        let context = match &self.mode {
            ModeState::Standalone => crate::operation::Context::standalone(
                invocation,
                self.config
                    .standalone_day()
                    .map_err(|problem| Failure::new("not_yet_available", problem.to_string()))?
                    .clone(),
            ),
            #[cfg(feature = "sinex")]
            ModeState::Primary { connection } => {
                let policy = connection
                    .context(|name| {
                        jiff::tz::TimeZone::get(name).map_err(|error| error.to_string())
                    })
                    .map_err(crate::protocol::context_failure)?;
                crate::operation::Context::primary(invocation, policy)?
            }
            #[cfg(feature = "sinex")]
            ModeState::Blocked { error: reason, .. } => {
                return Err(Failure::new("sinex_unavailable", reason.clone()));
            }
        };
        Ok(Arc::new(context))
    }

    pub(crate) fn ready_call(
        &self,
        request: &command_surface::RunnableRequest,
    ) -> Result<polylogue_model::read::ReadyCall, command_surface::Failure> {
        use command_surface::{Failure, FailureKind};
        let table = self
            .operations
            .get()
            .and_then(std::sync::Weak::upgrade)
            .ok_or_else(|| {
                Failure::new(
                    FailureKind::NotRunning,
                    "the operation table is unavailable",
                )
            })?;
        let description = table
            .descriptions()
            .iter()
            .find(|description| description.spec.name == request.operation)
            .ok_or_else(|| Failure::new(FailureKind::BadRequest, "unknown operation"))?;
        let command =
            crate::commands::command(&request.operation).expect("registered command exists");
        if command.build_typed.is_some() {
            crate::protocol::operations::admit(
                command,
                request,
                &crate::commands::Caller {
                    surface: crate::commands::Surface::Http,
                    cwd: None,
                    json: true,
                },
            )?;
        }
        let mcp = command.tool.map_or(serde_json::Value::Null, |tool| {
            let call = crate::protocol::operations::ToolCall {
                operation: Some(request.operation.clone()),
                arguments: request.arguments.clone(),
                request_id: request.request_id.clone(),
                deadline: request.deadline,
                cursor: request.cursor.clone(),
                budget_tokens: request.budget_tokens,
            };
            serde_json::json!({"name": tool.name(), "arguments": call})
        });
        Ok(polylogue_model::read::ReadyCall {
            cli: table.cli_command(request)?,
            mcp,
            http: polylogue_model::read::HttpCall {
                method: if description.spec.access != command_surface::Access::Read {
                    "POST"
                } else {
                    "GET"
                }
                .into(),
                path: description.spec.http_route.clone().ok_or_else(|| {
                    Failure::new(FailureKind::BadRequest, "operation has no HTTP route")
                })?,
                arguments: serde_json::to_value(request).expect("request serializes"),
            },
        })
    }

    /// Currency comes from the reader checkpoint and cached owner observations.
    pub(crate) fn currency(
        &self,
        reader: &polylogue_store::Reader<'_>,
    ) -> polylogue_store::Result<polylogue_model::currency::IndexCurrency> {
        use polylogue_model::currency::IndexCurrency;
        let _ = reader;
        match &self.mode {
            ModeState::Standalone => Ok(IndexCurrency::Standalone),
            #[cfg(feature = "sinex")]
            ModeState::Primary { connection } => connection
                .currency(reader.confirmed_checkpoint()?, reader.confirmed_scope()?)
                .map_err(StoreError::from),
            #[cfg(feature = "sinex")]
            ModeState::Blocked { error, .. } => Ok(IndexCurrency::primary(
                reader.confirmed_checkpoint()?,
                None,
                None,
                Some(false),
                Some(error.clone()),
            )
            .with_scope(reader.confirmed_scope()?)),
        }
    }
}

/// A query answer's outcome comes from the query itself and the index currency.
fn query_answer(
    d: &Daemon,
    reader: &polylogue_store::Reader<'_>,
    answer: super::query::QueryAnswer,
) -> Answer<Value> {
    match d.currency(reader) {
        Ok(currency) => query_outcome(answer, &currency),
        Err(error) => store_error(&error),
    }
}

/// A query's answer over an index of the given currency: degraded, with a
/// note, when the index is not current or a part is incomplete; empty when
/// it matched nothing; ok otherwise.
fn query_outcome(
    answer: super::query::QueryAnswer,
    currency: &polylogue_model::currency::IndexCurrency,
) -> Answer<Value> {
    let super::query::QueryAnswer {
        mut value,
        outcome,
        notes,
    } = answer;
    value.read_at_mut().currency = Some(currency.clone());
    let value = serde_json::to_value(value).expect("a query result serializes");
    let mut notes = notes;
    notes.extend(crate::protocol::currency_note(currency).map(str::to_owned));
    let degraded = currency.degraded() || outcome == super::query::Outcome::Incomplete;
    if degraded {
        command_surface::Answer {
            outcome: command_surface::Outcome::Degraded {
                value,
                missing: notes,
            },
            envelope: command_surface::Envelope {
                next: vec![],
                ..Default::default()
            },
        }
    } else if outcome == super::query::Outcome::Empty {
        command_surface::Answer {
            outcome: command_surface::Outcome::Empty(value),
            envelope: command_surface::Envelope {
                notes,
                next: vec![],
                ..Default::default()
            },
        }
    } else {
        command_surface::Answer {
            outcome: command_surface::Outcome::Ok(value),
            envelope: command_surface::Envelope {
                notes,
                next: vec![],
                ..Default::default()
            },
        }
    }
}

/// An import whose restored user state Sinex has not confirmed yet is
/// degraded, with the awaiting-confirmation note.
fn import_answer(report: polylogue_engine::ImportReport) -> Answer<Value> {
    let pending = report.user_state.pending > 0;
    match serde_json::to_value(report) {
        Ok(value) => crate::protocol::written(value, pending),
        Err(problem) => error("internal", problem),
    }
}

pub(crate) fn error(code: &str, message: impl ToString) -> Answer<Value> {
    Failure::new(code, message).answer()
}

/// A store failure. A capture waiting for Sinex is not a failure: it is the
/// degraded answer naming the durable capture.
pub(crate) fn store_error(e: &StoreError) -> Answer<Value> {
    match e {
        StoreError::PrimaryPending { identity } => {
            crate::protocol::written(serde_json::json!({ "identity": identity }), true)
        }
        e => Failure::store(e).answer(),
    }
}

/// How much of a session-start brief's remaining time its sections leave
/// for the answer to reach the caller: a fifth, at most a second.
fn answer_reserve(left: std::time::Duration) -> std::time::Duration {
    (left / 5).min(std::time::Duration::from_secs(1))
}

/// Prepares context and, when a sender is named, queues its delivery record
/// and answers without waiting for the writer, so a busy writer never delays
/// a brief. The delivery ID is the request ID, chosen before the text is
/// prepared and written into it; the writer records the delivery, and any
/// later outcome event for it, in queue order. A session-start brief stops
/// its sections early enough to answer before the request's deadline.
fn context(
    d: &Daemon,
    request_id: String,
    ask: &Ask,
    sender: Option<String>,
    budget_tokens: command_surface::TokenBudget,
    operation: &Arc<Context>,
) -> Answer<Value> {
    use polylogue_store::read::limit::ReadLimit;
    let request_limit = ReadLimit::current();
    let brief_limit = match (ask, &request_limit) {
        (Ask::SessionStart { .. }, Some(limit)) => limit.deadline().map(|deadline| {
            let left = deadline.saturating_duration_since(Instant::now());
            ReadLimit::at(
                deadline
                    .checked_sub(answer_reserve(left))
                    .unwrap_or(deadline),
                limit.cancellation.child(),
            )
        }),
        _ => None,
    };
    let queued = ask
        .recipient()
        .and_then(|recipient| d.queued_deliveries.latest(&recipient.to_string()));
    let prepare = || -> Result<_, Box<Answer<Value>>> {
        let reader = d.reads.reader().map_err(|e| Box::new(store_error(&e)))?;
        let settings = reader
            .settings_snapshot(&operation.machine)
            .map_err(|e| Box::new(store_error(&e)))?;
        let provider =
            crate::embeddings::Provider::from_settings(&settings, &d.config.embedding_credentials)
                .ok()
                .flatten();
        let near = |q: &str| match &provider {
            Some(p) => d.warm.similarity(&reader, p, q),
            None => Err("embeddings are off".into()),
        };
        let near: Option<&super::query::NearFn<'_>> = provider.is_some().then_some(&near);
        let ranking_identity = crate::embeddings::ranking_key(&reader, provider.as_ref());
        // Assertions written after this are the next brief's delta.
        let written_up_to = reader
            .change_position()
            .map_err(|e| Box::new(store_error(&e)))?
            .archive;
        let prepared = briefing::prepare(
            &reader,
            &d.scope,
            queued,
            ask,
            &request_id,
            operation,
            ranking_identity.as_deref(),
            near,
            budget_tokens,
        )
        .map_err(|e| match &request_limit {
            Some(limit) if limit.check().is_err() => {
                let text = match ask {
                    Ask::Recall { text, .. } => Some(text.as_str()),
                    _ => None,
                };
                Box::new(super::execution::refusal(&e, "context", text, limit))
            }
            _ => Box::new(store_error(&e)),
        })?;
        Ok((prepared, reader.scope_token(), written_up_to))
    };
    let prepared = match &brief_limit {
        Some(limit) => limit.scope(prepare),
        None => prepare(),
    };
    let (prepared, selected_scope, written_up_to) = match prepared {
        Ok(prepared) => prepared,
        Err(answer) => return *answer,
    };
    // The request's own deadline and cancellation still apply.
    if let Some(limit) = &request_limit
        && let Err(error) = limit.check()
    {
        let text = match ask {
            Ask::Recall { text, .. } => Some(text.as_str()),
            _ => None,
        };
        return super::execution::refusal(&error, "context", text, limit);
    }
    let mut delivery = None;
    if let Some(sender) = sender
        && !prepared.empty
    {
        let write = UserWrite::Deliver(Delivery {
            recipient_kind: if ask.recipient().is_some() {
                "session".into()
            } else {
                "operator".into()
            },
            recipient_id: ask.recipient().map(ToString::to_string),
            sender,
            event: prepared.event.clone(),
            cwd: ask.cwd().map(|c| c.to_string_lossy().into_owned()),
            text: prepared.text.clone(),
            refs: prepared.refs.clone(),
            left_out: prepared.left_out.clone(),
        });
        let writes = match writer(d) {
            Ok(writes) => writes,
            Err(failure) => return failure.answer(),
        };
        if let Err(error) = ReadLimit::finish_scope("record delivery") {
            return store_error(&error);
        }
        if let Err(error) = operation.check_parent_admission() {
            return store_error(&error);
        }
        // Queued without the caller's lifetime: the text has been handed
        // over, so its record is written even when the caller has gone.
        let request = operation.request(request_id.clone(), write);
        let since = briefing::LastBrief {
            at: request.at,
            seq: written_up_to,
        };
        let recipient = ask.recipient().map(|r| (r.to_string(), since));
        if let Err(e) = d
            .queued_deliveries
            .submit(writes, request, Some(selected_scope), recipient)
        {
            return store_error(&e);
        }
        delivery = Some(request_id);
    }
    crate::protocol::written(
        serde_json::json!({ "delivery": delivery, "prepared": prepared }),
        false,
    )
}

/// Queues what happened to a delivery and answers at once: the writer
/// applies it after the delivery it names, and a repeated request ID is
/// applied once. The hook waits only for the hand-over and may have exited
/// before the daemon reads its request; the outcome it reports has happened,
/// so it is recorded whether or not the caller is still connected.
fn delivery_event(
    d: &Daemon,
    invocation: &Invocation,
    request_id: String,
    write: UserWrite,
) -> Answer<Value> {
    if let UserWrite::DeliveryEvent {
        delivery,
        what: polylogue_model::DeliveryOutcome::NotSent,
        detail,
    } = &write
        && detail.as_deref() == Some("timeout")
        && !d.queued_deliveries.owns_preparation(delivery)
    {
        return ok(serde_json::json!({ "queued": false, "preparation": "unconfirmed" }));
    }
    let writes = match writer(d) {
        Ok(writes) => writes,
        Err(failure) => return failure.answer(),
    };
    match d.queued_deliveries.submit(
        writes,
        invocation.request(request_id.clone(), write),
        None,
        None,
    ) {
        Ok(()) => crate::protocol::written(
            serde_json::json!({ "id": request_id, "queued": true }),
            false,
        ),
        Err(error) => store_error(&error),
    }
}

fn writer(d: &Daemon) -> Result<&WriterSender, Failure> {
    d.writes.as_ref().ok_or_else(|| d.mode.write_refusal())
}

pub(crate) fn authored_preparation() -> user_writes::PreparationContext {
    user_writes::PreparationContext::new(
        polylogue_sources::origin::ORIGINS
            .iter()
            .map(|origin| origin.name.to_owned())
            .collect(),
    )
}

/// Submits one authored write; a write prepared from a read carries the
/// selected generation through the writer queue to custody commit. A write
/// accepted locally but not yet confirmed by Sinex is a degraded answer.
fn submit<T: serde::Serialize>(
    d: &Daemon,
    invocation: &Invocation,
    request_id: String,
    w: UserWrite,
    selected_scope: Option<polylogue_store::ReadScopeToken>,
    present: impl FnOnce(polylogue_model::user::UserOutcome) -> T,
) -> Answer<Value> {
    let accepted = || -> Result<_, Failure> {
        if let Some(limit) = polylogue_store::read::limit::ReadLimit::current() {
            limit.check()?;
        }
        let writes = writer(d)?;
        invocation.check_parent_admission()?;
        Ok(writes)
    };
    let writes = match accepted() {
        Ok(writes) => writes,
        Err(failure) => return failure.answer(),
    };
    let request = invocation.request(request_id, w);
    let preparation = authored_preparation();
    let result = match invocation.parent_lifetime() {
        Some(parent) => {
            writes.user_preparing_with_parent(request, preparation, selected_scope, parent)
        }
        None => writes.user_preparing(request, preparation, selected_scope),
    };
    match result {
        Ok(outcome) => {
            let pending = outcome.pending;
            match serde_json::to_value(present(outcome)) {
                Ok(value) => crate::protocol::written(value, pending),
                Err(cause) => error("internal", cause),
            }
        }
        Err(error) => store_error(&error),
    }
}

fn write(d: &Daemon, invocation: &Invocation, request_id: String, w: UserWrite) -> Answer<Value> {
    submit(d, invocation, request_id, w, None, |outcome| outcome)
}

fn write_from_read(
    d: &Daemon,
    invocation: &Invocation,
    request_id: String,
    w: UserWrite,
    selected_scope: polylogue_store::ReadScopeToken,
) -> Answer<Value> {
    submit(
        d,
        invocation,
        request_id,
        w,
        Some(selected_scope),
        |outcome| outcome,
    )
}

pub(crate) fn browser_refusal(refused: crate::browser::actions::Refused) -> Answer<Value> {
    error(refused.0, refused.0)
}

pub(crate) fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
}

/// A list read: empty when it matched nothing.
fn listed<T: serde::Serialize>(items: Vec<T>) -> Answer<Value> {
    let empty = items.is_empty();
    match serde_json::to_value(items) {
        Ok(value) if empty => command_surface::Answer {
            outcome: command_surface::Outcome::Empty(value),
            envelope: command_surface::Envelope {
                notes: vec![],
                next: vec![],
                ..Default::default()
            },
        },
        Ok(value) => Answer::ok(value),
        Err(cause) => error("internal", cause),
    }
}

pub(crate) fn ok(data: impl serde::Serialize) -> Answer<Value> {
    match serde_json::to_value(data) {
        Ok(data) => Answer::ok(data),
        Err(cause) => error("internal", cause),
    }
}

fn repeated_request(
    reader: &polylogue_store::Reader<'_>,
    id: &str,
    intended: &UserWrite,
    invocation: &Invocation,
) -> Result<Option<UserWrite>, Failure> {
    match reader.user_request(id)? {
        Some(previous)
            if previous.machine == invocation.machine
                && previous.authorship == invocation.authorship
                && intended.intent() == previous.write.intent() =>
        {
            Ok(Some(previous.write))
        }
        Some(_) => Err(StoreError::Conflict(id.into()).into()),
        None => Ok(None),
    }
}

/// Refuses new requests once the daemon is stopping.
pub(super) fn accepting(d: &Daemon) -> Result<(), Failure> {
    if d.stopping.load(Ordering::Acquire) {
        return Err(Failure::new(
            "shutting_down",
            "the daemon has stopped accepting requests",
        ));
    }
    Ok(())
}

/// Produces this finite feed, or leaves another request to its operation.
pub(crate) fn change_feed(
    request: &Request,
    d: &Daemon,
    operation: &Arc<crate::operation::Context>,
    output: &mut dyn std::io::Write,
) -> Option<Result<(), Failure>> {
    let Request::Changes { since } = request else {
        return None;
    };
    Some(accepting(d).and_then(|()| {
        let mut file = feeds::read(d, since.as_deref(), operation)?;
        std::io::copy(&mut file, output).map_err(|e| Failure::new("io", e))?;
        Ok(())
    }))
}

/// Answers one request. Reads use the read pool; imports go to the loop
/// thread, which owns the writer.
pub fn handle(request: Request, d: &Daemon) -> Answer<Value> {
    let limit = super::execution::limit_for(&request, None, Default::default());
    handle_with_limit(request, d, &limit)
}

pub fn handle_with_limit(
    request: Request,
    d: &Daemon,
    limit: &polylogue_store::read::limit::ReadLimit,
) -> Answer<Value> {
    let authorship = polylogue_model::authorship::compute_authorship(
        polylogue_model::authorship::Channel::Daemon,
        &Default::default(),
        None,
    );
    handle_with_caller(request, d, limit, &authorship)
}

pub fn handle_with_caller(
    request: Request,
    d: &Daemon,
    limit: &polylogue_store::read::limit::ReadLimit,
    authorship: &polylogue_model::authorship::Authorship,
) -> Answer<Value> {
    handle_with_invocation(request, d, limit, &d.invocation(authorship))
}

pub(crate) fn handle_with_invocation(
    request: Request,
    d: &Daemon,
    limit: &polylogue_store::read::limit::ReadLimit,
    invocation: &crate::operation::Invocation,
) -> Answer<Value> {
    #[cfg(test)]
    let reader_acquired = Arc::new(AtomicBool::new(false));
    #[cfg(test)]
    if let Some(sender) = d.read_limits.lock().unwrap().as_ref() {
        let _ = sender.send(crate::test_daemon::ReadObservation {
            limit: limit.clone(),
            owner: std::thread::current().id(),
            acquired: Arc::clone(&reader_acquired),
        });
    }
    if let Err(error) = invocation.check_parent_admission() {
        return store_error(&error);
    }
    super::execution::run(request, limit, |request| {
        handle_inner(
            request,
            d,
            invocation,
            #[cfg(test)]
            &reader_acquired,
        )
    })
}

fn handle_inner(
    request: Request,
    d: &Daemon,
    invocation: &crate::operation::Invocation,
    #[cfg(test)] reader_acquired: &AtomicBool,
) -> Answer<Value> {
    if let Err(failure) = accepting(d) {
        return failure.answer();
    }
    if d.mode.blocked() && !request.read_only() {
        return d.mode.write_refusal().answer();
    }
    if let Request::Complete { text, position } = &request {
        if *position > text.len() || !text.is_char_boundary(*position) {
            return error(
                "bad_request",
                "completion position must be a UTF-8 byte boundary within the query",
            );
        }
        let token = match d.archive.scope_token() {
            Ok(token) => token,
            Err(error) => return store_error(&error),
        };
        let candidates = crate::completion_catalog::complete_file(
            &d.completion_path,
            &crate::runtime_dir::archive_key(&d.archive.root),
            text,
            *position,
        );
        if let Err(error) = token.check() {
            return store_error(&error);
        }
        return listed(candidates);
    }
    if let Request::SchemaInfer { selection } = request {
        return super::schema::infer(d, selection);
    }
    if let Request::Verify { manifest } = &request {
        return super::verify::run(d, manifest.as_deref());
    }
    if matches!(&request, Request::Attachment { .. }) {
        return super::attachment::read(request, d);
    }
    if let Request::DeliveryEvent {
        request_id,
        delivery,
        what,
        detail,
    } = request
    {
        return delivery_event(
            d,
            invocation,
            request_id,
            UserWrite::DeliveryEvent {
                delivery,
                what,
                detail,
            },
        );
    }
    let _foreground = d.idle_admission.foreground();
    let reader = match d.reads.reader() {
        Ok(r) => r,
        Err(StoreError::ReadPoolClosed) => {
            return error("shutting_down", StoreError::ReadPoolClosed);
        }
        Err(e) => return store_error(&e),
    };
    #[cfg(test)]
    reader_acquired.store(true, Ordering::Release);
    let operation = if matches!(
        &request,
        Request::Query { .. }
            | Request::Timeline { .. }
            | Request::Cost { .. }
            | Request::Watch { .. }
            | Request::Embed { .. }
            | Request::Read { .. }
            | Request::Context { .. }
            | Request::Resumable { .. }
    ) || matches!(&request, Request::Explain { subject, .. } if subject == "query")
    {
        Some(match d.resolve_operation_context(invocation.clone()) {
            Ok(operation) => operation,
            Err(failure) => return failure.answer(),
        })
    } else {
        None
    };
    match request {
        Request::Attachment { .. } => unreachable!("attachment reads own their reader"),
        Request::ChatSend { .. }
        | Request::ChatActions
        | Request::ChatAction { .. }
        | Request::ChatClaim { .. }
        | Request::ChatReport { .. }
        | Request::ChatDecide { .. }
        | Request::ChatAttachment { .. }
        | Request::BackfillStart { .. }
        | Request::BackfillJobs
        | Request::BackfillCancel { .. }
        | Request::BackfillClaim { .. }
        | Request::BackfillReport { .. }
        | Request::Conversations { .. }
        | Request::StoredFiles { .. } => browser::handle(request, d, reader, invocation),
        Request::Changes { .. } => error(
            "bad_request",
            "the change feed uses the finite socket stream",
        ),
        Request::Follow { .. } => error("bad_request", "follow uses the socket stream"),
        Request::Ping { .. }
        | Request::Claim { .. }
        | Request::Answer { .. }
        | Request::Permission { .. }
        | Request::Interrupt { .. }
        | Request::Start { .. }
        | Request::HandOff { .. } => {
            drop(reader);
            acts::handle(request, d, invocation)
        }
        Request::Explain {
            subject,
            expression,
            unit,
            page_size,
            cursor,
            session,
        } => {
            let result = if subject == "query" {
                if unit.is_some() || page_size.is_some() || cursor.is_some() {
                    return error(
                        "bad_request",
                        "unit, page_size and cursor apply to declaration pages",
                    );
                }
                let Some(text) = expression.as_deref().filter(|s| !s.trim().is_empty()) else {
                    return error("bad_request", "expression is required for query");
                };
                super::query::with_ranking(
                    &reader,
                    &d.warm,
                    &d.config,
                    |embeddings, ranking_identity, _near| {
                        super::query::explain(
                            &reader,
                            super::query::Ask {
                                text,
                                cursor: None,
                                page_size: None,
                                why: false,
                                session,
                                context: operation
                                    .as_ref()
                                    .expect("evaluated operation context")
                                    .clone(),
                                embeddings,
                                ranking_identity,
                                near: None,
                                fields: &[],
                            },
                        )
                    },
                )
            } else {
                crate::explain::answer(
                    &subject,
                    expression.as_deref(),
                    unit.as_deref(),
                    page_size,
                    cursor.as_deref(),
                )
            };
            match result {
                Ok(data) => ok(data),
                Err(f) => f.answer(),
            }
        }
        Request::Timeline {
            date,
            across_years,
            undated,
            cursor,
            page_size,
        } => {
            match super::calendar::timeline(
                &reader,
                super::calendar::Timeline {
                    date: date.as_deref(),
                    across_years,
                    undated,
                    cursor: cursor.as_deref(),
                    page_size,
                },
                operation.as_ref().expect("calendar operation context"),
            ) {
                Ok(data) => ok(data),
                Err(f) => f.answer(),
            }
        }
        Request::Cost {
            month,
            undated,
            unknown_model,
            query,
            date,
            model,
            cursor,
            page_size,
        } => super::query::with_ranking(
            &reader,
            &d.warm,
            &d.config,
            |embeddings, ranking_identity, near| match super::calendar::cost(
                &reader,
                super::calendar::Cost {
                    embeddings,
                    ranking_identity,
                    near: Some(near),
                    month: month.as_deref(),
                    query: &query,
                    date: date.as_deref(),
                    model: model.as_deref(),
                    undated,
                    unknown_model,
                    cursor: cursor.as_deref(),
                    page_size,
                },
                operation.as_ref().expect("calendar operation context"),
            ) {
                Ok(data) => ok(data),
                Err(f) => f.answer(),
            },
        ),
        Request::Status => status::answer(d, &reader),
        Request::Query {
            text,
            cursor,
            page_size,
            why,
            session,
            fields,
        } => super::query::with_ranking(
            &reader,
            &d.warm,
            &d.config,
            |embeddings, ranking_identity, near| {
                let ask = super::query::Ask {
                    text: &text,
                    cursor: cursor.as_deref(),
                    page_size,
                    why,
                    session,
                    context: operation
                        .as_ref()
                        .expect("evaluated operation context")
                        .clone(),
                    embeddings,
                    ranking_identity,
                    near: Some(near),
                    fields: &fields,
                };
                match super::query::run(&reader, ask) {
                    Ok(answer) => query_answer(d, &reader, answer),
                    Err(f) => f.answer(),
                }
            },
        ),
        Request::Complete { .. } => {
            unreachable!("completion is handled before opening an archive reader")
        }

        Request::Sql { statement } => match reader.query(&statement, &[]) {
            // The connection is read-only: a statement that writes fails.
            Ok(rows) => listed(rows),
            Err(e) => error("sql", e),
        },
        request @ Request::Read { .. } => read::page(
            &reader,
            d,
            &request,
            operation.as_ref().expect("read operation context"),
        ),
        request @ (Request::Import { .. } | Request::ImportArchive { .. }) => {
            drop(reader);
            let (paths, from_archive, dry_run) = match request {
                Request::Import { paths, .. } => (paths, None, false),
                Request::ImportArchive { path, dry_run, .. } => (Vec::new(), Some(path), dry_run),
                _ => unreachable!("an import request"),
            };
            let (reply, wait) = sync_channel(1);
            let tx = match d.loop_tx.lock() {
                Ok(tx) => tx,
                Err(_) => return error("internal", "the loop has stopped"),
            };
            if let Err(error) = invocation.check_parent_admission() {
                return store_error(&error);
            }
            let sent = tx
                .send(LoopMsg::Import {
                    paths,
                    from_archive,
                    dry_run,
                    authorship: invocation.authorship.clone(),
                    parent: invocation.parent_lifetime(),
                    reply,
                })
                .is_ok();
            drop(tx);
            if !sent {
                return error("internal", "the loop has stopped");
            }
            match wait.recv() {
                Ok(Ok(report)) => import_answer(report),
                Ok(Err(polylogue_engine::ImportFailure::Parent(error))) => store_error(&error),
                Ok(Err(polylogue_engine::ImportFailure::Import(error))) => {
                    self::error("import", error)
                }
                Err(_) => error("internal", "the loop has stopped"),
            }
        }
        Request::SchemaInfer { .. } | Request::Verify { .. } => {
            unreachable!("idle-priority jobs are admitted before acquiring a reader")
        }
        Request::SchemaUnread { origin } => super::schema::unread(&reader, origin.as_deref()),
        Request::Embed {
            what,
            query,
            limit,
            yes,
        } => {
            let settings = match reader.settings_snapshot(d.config.requests.machine()) {
                Ok(settings) => settings,
                Err(error) => return store_error(&error),
            };
            crate::embeddings::handle(
                &reader,
                &what,
                query.as_deref(),
                limit,
                yes,
                crate::embeddings::Context {
                    settings: &settings,
                    credentials: &d.config.embedding_credentials,
                    operation: operation.as_ref().expect("embedding operation context"),
                },
            )
        }

        Request::Keep {
            request_id,
            text,
            name,
        } => {
            let selected_scope = reader.scope_token();
            let intended = UserWrite::KeepResult {
                name: name.clone(),
                query: text.clone(),
                rows_json: String::new(),
                read_at: String::new(),
            };
            match repeated_request(&reader, &request_id, &intended, invocation) {
                Ok(Some(previous)) => {
                    drop(reader);
                    return write_from_read(d, invocation, request_id, previous, selected_scope);
                }
                Err(failure) => return failure.answer(),
                Ok(None) => {}
            }
            let operation = match d.resolve_operation_context(invocation.clone()) {
                Ok(operation) => operation,
                Err(failure) => return failure.answer(),
            };
            let kept = super::query::with_ranking(
                &reader,
                &d.warm,
                &d.config,
                |embeddings, ranking_identity, near| {
                    let ask = super::query::Ask {
                        text: &text,
                        cursor: None,
                        page_size: None,
                        why: false,
                        session: None,
                        context: operation.clone(),
                        embeddings,
                        ranking_identity,
                        near: Some(near),
                        fields: &[],
                    };
                    super::query::keep(&reader, ask)
                },
            );
            let kept = match kept {
                Ok(k) => k,
                Err(f) => {
                    return f.answer();
                }
            };
            drop(reader);
            let read_at = serde_json::json!({ "now": kept["now"] }).to_string();
            write_from_read(
                d,
                invocation,
                request_id,
                UserWrite::KeepResult {
                    name,
                    query: text,
                    rows_json: kept.to_string(),
                    read_at,
                },
                selected_scope,
            )
        }
        Request::Save {
            request_id,
            name,
            text,
            denominator,
        } => {
            let intended = UserWrite::SaveQuery {
                name,
                text,
                denominator,
            };
            let resolved = match repeated_request(&reader, &request_id, &intended, invocation) {
                Ok(Some(previous)) => previous,
                Ok(None) => intended,
                Err(failure) => return failure.answer(),
            };
            drop(reader);
            write(d, invocation, request_id, resolved)
        }
        Request::Watch {
            request_id,
            query,
            name,
            hold,
            notify,
        } => {
            let selected_scope = reader.scope_token();
            if let Some(hold) = &hold
                && crate::signals::duration(hold).is_none()
            {
                return error(
                    "invalid",
                    format!("--for {hold:?} is not a duration such as 20m"),
                );
            }
            // A watch whose query cannot run would only be refused later.
            let ask = super::query::Ask {
                text: &query,
                cursor: None,
                page_size: Some(1),
                why: false,
                session: None,
                context: operation
                    .as_ref()
                    .expect("evaluated operation context")
                    .clone(),
                embeddings: false,
                ranking_identity: None,
                near: None,
                fields: &[],
            };
            if let Err(f) = super::query::run(&reader, ask) {
                return f.answer();
            }
            drop(reader);
            let mut rule = serde_json::json!({
                "name": name.clone().unwrap_or_else(|| query.clone()),
                "query": query,
                "fires": "on_new",
                "deliver": "notice",
                "interrupt": notify,
            });
            if let Some(hold) = hold {
                rule["for"] = serde_json::json!(hold);
            }
            let assertion = polylogue_model::assertion::NewAssertion {
                kind: polylogue_model::assertion::AssertionKind::Signal,
                key: rule["name"].as_str().map(str::to_owned),
                value: Some(rule),
                links: vec![],
                valid_until: None,
            };
            write_from_read(
                d,
                invocation,
                request_id,
                UserWrite::Assert {
                    assertions: vec![assertion],
                },
                selected_scope,
            )
        }
        Request::BrowserSelectionNote {
            request_id,
            body,
            quote,
            provider,
            conversation_id,
            message_id,
            page_url,
            selected_at,
        } => {
            let input = crate::browser::notes::SelectionNote {
                request_id,
                body,
                quote,
                source: crate::browser::notes::Source {
                    provider,
                    conversation_id,
                    message_id,
                    page_url,
                    selected_at,
                },
            };
            let (session, message_id) = match input.references() {
                Ok(references) => references,
                Err(failure) => return failure.answer(),
            };
            let message = polylogue_model::Reference::Message(message_id.clone());
            let intended = UserWrite::Assert {
                assertions: vec![input.assertion(session.clone(), message.clone())],
            };
            let selected_scope = reader.scope_token();
            match repeated_request(&reader, &input.request_id, &intended, invocation) {
                Ok(Some(previous)) => {
                    drop(reader);
                    return write_from_read(
                        d,
                        invocation,
                        input.request_id,
                        previous,
                        selected_scope,
                    );
                }
                Ok(None) => {}
                Err(failure) => return failure.answer(),
            }
            // A message reference names exactly one message whatever its
            // reachability; `id:` takes the native ID, not the reference text.
            let query = format!(
                "messages where message:{}",
                polylogue_query::quoted(&message_id.to_string())
            );
            let targets = match query_targets_in(d, &reader, &query, invocation, Some(&session)) {
                Ok(targets) => targets,
                Err(failure) => return failure.answer(),
            };
            if targets.as_slice() != [message] {
                return error(
                    "conflict",
                    "the supplied message is missing, ambiguous or belongs to another conversation; capture the conversation and retry",
                );
            }
            drop(reader);
            write_from_read(d, invocation, input.request_id, intended, selected_scope)
        }
        Request::Assert {
            request_id,
            assertions,
        } => {
            drop(reader);
            write(d, invocation, request_id, UserWrite::Assert { assertions })
        }
        Request::AssertQuery {
            request_id,
            query,
            assertion,
        } => {
            let selected_scope = reader.scope_token();
            let intended = UserWrite::AssertQuery {
                query: query.clone(),
                assertion: assertion.clone(),
                resolved: Vec::new(),
            };
            let resolved = match repeated_request(&reader, &request_id, &intended, invocation) {
                Ok(Some(previous)) => previous,
                Ok(None) => {
                    if assertion
                        .links
                        .iter()
                        .any(|link| link.role == polylogue_model::assertion::LinkRole::About)
                    {
                        return error(
                            "bad_request",
                            "a query assertion cannot supply its own about links",
                        );
                    }
                    let targets = match query_targets(d, &reader, &query, invocation) {
                        Ok(targets) => targets,
                        Err(failure) => return failure.answer(),
                    };
                    UserWrite::AssertQuery {
                        query,
                        assertion,
                        resolved: targets,
                    }
                }
                Err(failure) => return failure.answer(),
            };
            let targets = match &resolved {
                UserWrite::AssertQuery { resolved, .. } => resolved.len(),
                _ => return error("conflict", "stored request has another action"),
            };
            drop(reader);
            submit(
                d,
                invocation,
                request_id,
                resolved,
                Some(selected_scope),
                |outcome| polylogue_model::user::AssertQueryOutcome::new(outcome, targets),
            )
        }
        Request::Revise {
            request_id,
            bases,
            patch,
        } => {
            drop(reader);
            write(
                d,
                invocation,
                request_id,
                UserWrite::ReviseAssertions { bases, patch },
            )
        }
        Request::Judge {
            request_id,
            note,
            accepted,
            reason,
        } => {
            let reference = match polylogue_model::Reference::parse(&note) {
                Ok(polylogue_model::Reference::Assertion(reference)) => reference,
                Ok(_) => {
                    return error(
                        "bad_request",
                        "a judgment needs a natural assertion reference",
                    );
                }
                Err(error) => return self::error("bad_request", error),
            };
            let selected_scope = reader.scope_token();
            let found = match reader.assertion(&reference) {
                Ok(Some(found)) => found,
                Ok(None) => return error("not_found", format!("no assertion {note}")),
                Err(e) => return store_error(&e),
            };
            if !found.fact.author.is_agent() {
                return error(
                    "bad_request",
                    "only assertions agents wrote are accepted or rejected; revise or withdraw the operator's own",
                );
            }
            drop(reader);
            let judgment = polylogue_model::assertion::NewAssertion {
                kind: polylogue_model::assertion::AssertionKind::Judgment,
                key: Some(if accepted { "accepted" } else { "rejected" }.into()),
                value: reason.map(serde_json::Value::String),
                links: vec![polylogue_model::assertion::NewAssertionLink {
                    role: polylogue_model::assertion::LinkRole::About,
                    reference: polylogue_model::Reference::Assertion(
                        polylogue_model::RevisionReference {
                            root: found.fact.root,
                            revision: Some(found.fact.id),
                        },
                    ),
                }],
                valid_until: None,
            };
            write_from_read(
                d,
                invocation,
                request_id,
                UserWrite::Assert {
                    assertions: vec![judgment],
                },
                selected_scope,
            )
        }
        Request::Withdraw {
            request_id,
            targets,
            reason,
        } => {
            drop(reader);
            write(
                d,
                invocation,
                request_id,
                UserWrite::WithdrawAssertions { targets, reason },
            )
        }
        Request::Context {
            request_id,
            ask,
            sender,
            budget_tokens,
        } => {
            drop(reader);
            context(
                d,
                request_id,
                &ask,
                sender,
                budget_tokens,
                operation.as_ref().expect("briefing operation context"),
            )
        }
        Request::DeliveryEvent { .. } => {
            unreachable!("delivery events are handed over without a reader")
        }
        Request::Resumable { cwd, limit } => {
            match crate::resume::resumable(
                &reader,
                &d.scope,
                &cwd,
                limit.unwrap_or(10),
                operation.as_ref().expect("resumable operation context"),
            ) {
                Ok(sessions) => listed(sessions),
                Err(e) => store_error(&e),
            }
        }
        Request::Continue { session } => match crate::resume::plan(&reader, &session) {
            Ok(p) => ok(p),
            Err(e) => store_error(&e),
        },
        Request::Setting {
            request_id: _,
            key,
            value: None,
            machine,
        } => {
            let selected = machine.as_deref().unwrap_or(&invocation.machine);
            if let Some(key) = key {
                if let Err(error) = polylogue_model::settings::target_machine(
                    &key,
                    machine.as_deref(),
                    &invocation.machine,
                ) {
                    return self::error("bad_value", error);
                }
                match reader.setting_current(&key, selected) {
                    Ok(current) => ok(current),
                    Err(error) => store_error(&error),
                }
            } else {
                for spec in polylogue_model::settings::CATALOG
                    .iter()
                    .filter(|spec| spec.scope == polylogue_model::settings::Scope::Host)
                {
                    if let Err(error) = polylogue_model::settings::target_machine(
                        spec.key,
                        machine.as_deref(),
                        &invocation.machine,
                    ) {
                        return self::error("bad_value", error);
                    }
                }
                match reader.settings_catalog(selected) {
                    Ok(catalog) => ok(catalog),
                    Err(error) => store_error(&error),
                }
            }
        }
        Request::Setting {
            request_id,
            key,
            value: Some(value),
            machine,
        } => {
            let Some(key) = key else {
                return error("bad_request", "a value requires a setting key");
            };
            drop(reader);
            write(
                d,
                invocation,
                request_id.unwrap_or_else(user_writes::new_request_id),
                UserWrite::SetSetting {
                    key,
                    value,
                    machine,
                },
            )
        }
    }
}

/// Resolve bulk targets through the same pinned selection and acknowledged
/// execution lifecycle used by reads. The returned references retain native grain.
fn query_targets(
    d: &Daemon,
    reader: &polylogue_store::Reader<'_>,
    query: &str,
    invocation: &crate::operation::Invocation,
) -> Result<Vec<polylogue_model::Reference>, Failure> {
    query_targets_in(d, reader, query, invocation, None)
}

/// A browser selection additionally requires one physical row in its supplied
/// conversation. Membership uses the same prepared population authority.
fn query_targets_in(
    d: &Daemon,
    reader: &polylogue_store::Reader<'_>,
    query: &str,
    invocation: &crate::operation::Invocation,
    container: Option<&polylogue_model::Reference>,
) -> Result<Vec<polylogue_model::Reference>, Failure> {
    use polylogue_model::query_objects::{ExecutionPresentation, QueryRun, RunBinding};
    super::query::with_ranking(
        reader,
        &d.warm,
        &d.config,
        |embeddings, ranking_identity, near| {
            let operation = d.resolve_operation_context(invocation.clone())?;
            let ask = super::query::Ask {
                text: query,
                cursor: None,
                page_size: None,
                why: false,
                session: None,
                context: Arc::clone(&operation),
                embeddings,
                ranking_identity: ranking_identity.clone(),
                near: Some(near),
                fields: &[],
            };
            let options =
                super::query::options(reader, &ask, uuid::Uuid::new_v4().as_u128() as u64)?;
            let selection = reader.prepare_query_selection(query, &options)?;
            if selection.program().shape != polylogue_query::Shape::Rows {
                return Err(Failure::new(
                    "bad_request",
                    "assertions require identifiable rows, not a grouped or aggregate population",
                ));
            }
            let object = selection.query_object()?;
            let query_options = polylogue_store::query::execute::options_snapshot(
                &options,
                false,
                ranking_identity,
            )?;
            let accepted = operation
                .accepted_request()
                .map(serde_json::to_value)
                .transpose()
                .map_err(|problem| Failure::new("internal", problem))?;
            let mut binding = selection.execution_binding(
                query_options,
                ExecutionPresentation::Query,
                accepted,
            )?;
            binding
                .replay
                .context
                .versions
                .extend(operation.cursor_versions());
            let run = QueryRun {
                historical_source: None,
                id: crate::operation::new_run_id(),
                question: Some(selection.question_id()),
                at: operation.ran_at,
                read_at: reader.change_position()?,
                machine: operation.machine.clone(),
                authorship: operation.authorship.clone(),
                binding: RunBinding::Original {
                    execution: Box::new(binding),
                },
            };
            let hash = run
                .binding_hash()
                .map_err(|problem| Failure::new("internal", problem))?;
            let writes = writer(d)?;
            operation.check_parent_admission()?;
            match operation.parent_lifetime() {
                Some(parent) => writes.query_object_with_parent(object, parent),
                None => writes.query_object(object),
            }?;
            self::execution::record(
                writes,
                operation.parent_lifetime(),
                run,
                hash,
                || d.config.requests.now(),
                || {
                    if !selection.program().incomplete.is_empty() {
                        return Err(Failure::new(
                            "not_yet_available",
                            "the query population is incomplete; no assertion was written",
                        ));
                    }
                    let mut row_count = 0_u64;
                    let mut targets = Vec::new();
                    let mut seen = HashSet::new();
                    reader.visit_selection::<Failure>(&selection, None, Some(near), |row| {
                        row_count = row_count.checked_add(1).ok_or_else(|| {
                            Failure::new("internal", "selected row count exceeds u64")
                        })?;
                        let reference = row
                            .target
                            .as_ref()
                            .map(polylogue_model::read::ReadTarget::reference)
                            .ok_or_else(|| {
                                Failure::new(
                                    "not_yet_available",
                                    "the selected row has no authoritative source reference",
                                )
                            })?;
                        if seen.insert(reference.clone()) {
                            targets.push(reference);
                        }
                        Ok(std::ops::ControlFlow::Continue(()))
                    })?;
                    if let Some(polylogue_model::Reference::Session(expected)) = container {
                        let Some(polylogue_model::Reference::Message(message)) = targets.first()
                        else {
                            return Err(Failure::new(
                                "conflict",
                                format!(
                                    "the supplied source is not a recorded message; the query selected {row_count} rows, first {:?}",
                                    targets.first()
                                ),
                            ));
                        };
                        let mut memberships = 0_u8;
                        let mut matches = false;
                        reader.visit_message_sessions(message, |session| {
                            memberships += 1;
                            matches = &session == expected;
                            Ok(if memberships == 2 {
                                std::ops::ControlFlow::Break(())
                            } else {
                                std::ops::ControlFlow::Continue(())
                            })
                        })?;
                        if memberships != 1 || !matches {
                            return Err(Failure::new(
                                "conflict",
                                "the message has missing, ambiguous or mismatching recorded conversation membership",
                            ));
                        }
                    }
                    if container.is_some() && row_count != 1 {
                        return Err(Failure::new(
                            "conflict",
                            "the supplied message is missing or ambiguous; capture the conversation and retry",
                        ));
                    }
                    if targets.is_empty() {
                        return Err(Failure::new(
                            "empty",
                            "the query returns no rows; no assertion was written",
                        ));
                    }
                    let count = u64::try_from(targets.len())
                        .map_err(|_| Failure::new("internal", "selected row count exceeds u64"))?;
                    Ok(self::execution::Completed {
                        value: targets,
                        count: Some(count),
                        exact: selection.program().incomplete.is_empty(),
                        next: None,
                    })
                },
            )
        },
    )
}

/// Owns the application workers and writer lock. Transports must stop
/// before this runtime; external daemon references cannot keep workers alive.
pub struct Runtime {
    pub daemon: Arc<Daemon>,
    pub operations: Arc<command_surface::OperationTable>,
    signals_stop: Option<Sender<()>>,
    signals: Option<JoinHandle<()>>,
    counts: Option<JoinHandle<()>>,
    live_evidence_stop: Option<Sender<()>>,
    live_evidence: Option<JoinHandle<()>>,
    completion: Option<crate::completion_catalog::publisher::Publisher>,
    scope: Option<super::scope::ScopeWorker>,
    completion_scope: Option<polylogue_store::ScopeInvalidation>,
    engine: Option<JoinHandle<std::io::Result<()>>>,
    lock: Option<polylogue_store::ArchiveLock>,
    worker_events: tokio::sync::watch::Sender<Option<&'static str>>,
    /// Set once by a loop run until idle, when it ends.
    idle: Arc<tokio::sync::watch::Sender<Option<RunSummary>>>,
}

impl Runtime {
    #[cfg(test)]
    pub(crate) fn signal_stop_for_test(&self) -> Sender<()> {
        self.signals_stop
            .as_ref()
            .expect("running signal worker")
            .clone()
    }

    #[cfg(test)]
    pub(crate) fn pause_signals(&mut self) -> std::io::Result<()> {
        if let Some(stop) = self.signals_stop.take() {
            let _ = stop.send(());
        }
        if let Some(worker) = self.signals.take() {
            worker
                .join()
                .map_err(|_| std::io::Error::other("signal worker panicked"))?;
        }
        Ok(())
    }

    /// Refuses new requests and wakes readers waiting for a reader slot,
    /// before transports and workers are joined. Readers and writes already
    /// running finish as usual.
    pub fn stop_accepting(&self) {
        self.daemon.stopping.store(true, Ordering::Release);
        self.daemon.warm.clear();
        self.daemon.reads.close();
    }

    pub(crate) fn worker_events(&self) -> tokio::sync::watch::Receiver<Option<&'static str>> {
        self.worker_events.subscribe()
    }

    /// Becomes `Some` when a loop run until idle has ended, with what it did.
    pub(crate) fn idle(&self) -> tokio::sync::watch::Receiver<Option<RunSummary>> {
        self.idle.subscribe()
    }

    /// A finished continuous worker is a failure; slow work remains live.
    pub fn worker_failure(&self) -> Option<&'static str> {
        if self.daemon.stopping.load(Ordering::Acquire) {
            return None;
        }
        if self.engine.as_ref().is_some_and(JoinHandle::is_finished) && self.idle.borrow().is_none()
        {
            return Some("engine worker terminated");
        }
        if self.signals.as_ref().is_some_and(JoinHandle::is_finished) {
            return Some("signal worker terminated");
        }
        if self.counts.as_ref().is_some_and(JoinHandle::is_finished) {
            return Some("count worker terminated");
        }
        if let Some(failure) = self
            .scope
            .as_ref()
            .and_then(|worker| worker.worker_failure())
        {
            return Some(failure);
        }
        if self
            .live_evidence
            .as_ref()
            .is_some_and(JoinHandle::is_finished)
        {
            return Some("live evidence worker terminated");
        }
        self.completion
            .as_ref()
            .and_then(|worker| worker.worker_failure())
    }

    pub fn shutdown(&mut self) -> std::io::Result<()> {
        self.stop_accepting();
        let mut failure = None;
        if let Some(completion) = self.completion.as_mut()
            && let Err(error) = completion.shutdown()
        {
            failure = Some(error);
        }
        self.completion.take();
        if let Some(scope) = self.scope.as_mut()
            && let Err(error) = scope.shutdown()
        {
            failure.get_or_insert(error);
        }
        self.scope.take();
        if let Some(stop) = self.signals_stop.take() {
            let _ = stop.send(());
        }
        if self.counts.is_some() {
            self.daemon.warm.stop_counting();
            self.daemon.warm.clear();
        }
        if let Some(counts) = self.counts.take()
            && counts.join().is_err()
        {
            failure.get_or_insert_with(|| std::io::Error::other("count worker panicked"));
        }
        if let Some(stop) = self.live_evidence_stop.take() {
            let _ = stop.send(());
        }
        if let Some(worker) = self.live_evidence.take()
            && worker.join().is_err()
        {
            failure.get_or_insert_with(|| std::io::Error::other("live evidence worker panicked"));
        }
        if let Some(signals) = self.signals.take()
            && signals.join().is_err()
        {
            failure.get_or_insert_with(|| std::io::Error::other("signal worker panicked"));
        }
        if let Ok(tx) = self.daemon.loop_tx.lock() {
            let _ = tx.send(LoopMsg::Stop);
        }
        if let Some(engine) = self.engine.take() {
            match engine.join() {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    failure.get_or_insert(error);
                }
                Err(_) => {
                    failure.get_or_insert_with(|| std::io::Error::other("loop thread panicked"));
                }
            }
        }
        self.daemon.reads.close();
        self.completion_scope.take();
        self.lock.take();
        failure.map_or(Ok(()), Err)
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        if let Err(error) = self.shutdown() {
            eprintln!("polylogue: shutdown: {error}");
        }
    }
}

/// Standalone, primary on Sinex, or blocked: a Sinex configuration that
/// cannot be read, or a Sinex that cannot be reached at start; never a
/// fallback to another mode. Only a build with the
/// `sinex` feature reads a Sinex configuration, so only it can be blocked.
#[derive(Clone)]
pub enum ArchiveMode {
    Standalone,
    #[cfg(feature = "sinex")]
    Primary {
        config: polylogue_sinex::Config,
    },
    #[cfg(feature = "sinex")]
    Blocked {
        /// The mode the configuration asked for, when it could be read.
        configured: Option<&'static str>,
        path: std::path::PathBuf,
        error: String,
    },
}

#[derive(serde::Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
enum ModeState {
    Standalone,
    #[cfg(feature = "sinex")]
    Primary {
        #[serde(skip)]
        connection: Arc<polylogue_sinex::Sinex>,
    },
    #[cfg(feature = "sinex")]
    Blocked {
        configured: Option<&'static str>,
        path: std::path::PathBuf,
        error: String,
    },
}

impl ModeState {
    /// Writes are paused until the Sinex configuration is fixed or Sinex
    /// can be reached.
    fn blocked(&self) -> bool {
        #[cfg(feature = "sinex")]
        {
            matches!(self, Self::Blocked { .. })
        }
        #[cfg(not(feature = "sinex"))]
        {
            false
        }
    }

    /// The answer to a write the daemon cannot take.
    fn write_refusal(&self) -> Failure {
        #[cfg(feature = "sinex")]
        if let Self::Blocked {
            path,
            error: reason,
            ..
        } = self
        {
            let mut failure = Failure::new(
                "sinex_unavailable",
                format!("archive writes are paused: {reason}"),
            );
            failure.error.hint = Some(format!("fix the Sinex settings in {}", path.display()));
            return failure;
        }
        Failure::from(StoreError::WriterGone)
    }
}

/// Opens the archive and starts owned application workers. Startup errors
/// join any workers already started before releasing the archive lock.
/// A loop run until idle ends the runtime's work when it ends ([`Runtime::idle`]).
pub fn start(
    deployment: Arc<crate::config::Config>,
    config: Config,
    mode: ArchiveMode,
    lifetime: Lifetime,
) -> std::io::Result<Runtime> {
    let root = deployment.archive.as_path();
    let completion_path = deployment.paths.completion_catalog(root);
    let to_io = |e: &dyn std::fmt::Display| std::io::Error::other(e.to_string());
    let pointer = polylogue_store::completion::CatalogPointer::new(completion_path.clone());
    let invalidate = move || pointer.invalidate();
    let settings = config.clone();
    let (archive, lock, mode, engine, completion_scope) = match mode {
        ArchiveMode::Standalone => {
            let (archive, lock, scope) = Archive::open_or_create_with_invalidator(root, invalidate)
                .map_err(std::io::Error::other)?;
            let engine = Engine::new(config, archive.clone()).map_err(|e| to_io(&e))?;
            (archive, lock, ModeState::Standalone, Some(engine), scope)
        }
        #[cfg(feature = "sinex")]
        ArchiveMode::Blocked {
            configured,
            path,
            error,
        } => {
            let (archive, lock, scope) =
                Archive::open_primary_cache_with_invalidator(root, invalidate)
                    .map_err(std::io::Error::other)?;
            let state = ModeState::Blocked {
                configured,
                path,
                error,
            };
            (archive, lock, state, None, scope)
        }
        #[cfg(feature = "sinex")]
        ArchiveMode::Primary { config: sinex, .. } => {
            let (archive, lock, scope) =
                Archive::open_primary_with_invalidator(root, &sinex.store, invalidate)
                    .map_err(std::io::Error::other)?;
            let connection = Arc::new(
                polylogue_sinex::Sinex::open(&sinex, &deployment.primary_auth, &settings.host)
                    .map_err(std::io::Error::other)?,
            );
            let engine = Engine::on_sinex(config, archive.clone(), connection.clone())
                .map_err(|error| to_io(&error))?;
            (
                archive,
                lock,
                ModeState::Primary { connection },
                Some(engine),
                scope,
            )
        }
    };
    let mut engine = engine;
    let writes = engine
        .as_ref()
        .map(Engine::writer)
        .transpose()
        .map_err(|e| to_io(&e))?;
    let per_reader = settings
        .reader_memory_per_connection()
        .map_err(|e| to_io(&e))?;
    let reads = ReadPool::with_budget(
        &archive,
        settings.workers,
        per_reader * settings.workers as u64,
    )
    .map_err(|e| to_io(&e))?;
    let (tx, rx) = channel();
    // The loop's first pass starts as soon as it runs.
    let status = Arc::new(Mutex::new(LoopStatus {
        busy: engine.is_some(),
        ..LoopStatus::default()
    }));
    let mail = crate::mailbox::dir(&archive.root);
    let hosts = crate::hosting::dir(&archive.root);
    let daemon = Arc::new(Daemon {
        config: deployment,
        operations: std::sync::OnceLock::new(),
        archive,
        reads,
        completion_path,
        callers: crate::protocol::caller::hooks::Registry::new(
            settings.spool_dir(polylogue_sources::hook_spool::ORIGIN),
        ),
        warm: super::warm::Warm::default(),
        scope: super::scope::SharedScopeIndex::default(),
        queued_deliveries: super::queued_deliveries::QueuedDeliveries::default(),
        acts: crate::live_acts::Acts::new(
            mail,
            hosts,
            settings.spool_dir(polylogue_sources::hook_spool::ORIGIN),
            polylogue_sources::origin::by_name("codex")
                .map(|origin| {
                    settings
                        .root(origin.root)
                        .join("app-server-control/app-server-control.sock")
                })
                .unwrap_or_default(),
            "codex".into(),
        ),
        writes,
        mode,
        loop_tx: Mutex::new(tx.clone()),
        status: Arc::clone(&status),
        started: Instant::now(),
        signals: Mutex::new(serde_json::Value::Null),
        #[cfg(test)]
        signal_trace: Mutex::new(None),
        stopping: AtomicBool::new(false),
        idle_admission: super::schema::Admission::default(),
        #[cfg(test)]
        read_limits: Mutex::new(None),
    });
    daemon.acts.bind(Arc::downgrade(&daemon));
    // Hosted sessions outlive a daemon restart: their hosts, or Codex's
    // app-server, kept them running; connect to each again.
    daemon.acts.reattach();
    // Pings and notices left pending raise their running sessions' mailbox
    // flags; those to sessions no longer running are settled.
    acts::settle_ended(&daemon, daemon.config.requests.now().unix_ms(), true);
    let operations = Arc::new(crate::protocol::operations::table(Arc::clone(&daemon)));
    assert!(
        daemon.operations.set(Arc::downgrade(&operations)).is_ok(),
        "runtime installs its operation table once"
    );
    let mut runtime = Runtime {
        daemon,
        operations,
        signals_stop: None,
        signals: None,
        counts: None,
        live_evidence_stop: None,
        live_evidence: None,
        completion: None,
        scope: None,
        completion_scope: Some(completion_scope),
        engine: None,
        lock: Some(lock),
        worker_events: tokio::sync::watch::channel(None).0,
        idle: Arc::new(tokio::sync::watch::channel(None).0),
    };
    // Subscribe before the initial snapshot or starting the ingest loop.
    let completion_commits = runtime
        .daemon
        .writes
        .as_ref()
        .map(WriterSender::completion_commits);
    runtime.scope = Some(super::scope::ScopeWorker::start(
        Arc::clone(&runtime.daemon),
        runtime
            .daemon
            .writes
            .as_ref()
            .map(WriterSender::completion_commits),
        runtime.worker_events.clone(),
    )?);
    if !runtime.daemon.mode.blocked() {
        runtime.completion = Some(crate::completion_catalog::publisher::Publisher::start(
            Arc::clone(&runtime.daemon),
            completion_commits,
            runtime.worker_events.clone(),
        )?);
    }
    let Some(mut engine) = engine.take() else {
        // Without a loop (writes paused) there is no work to wait for.
        if lifetime == Lifetime::UntilIdle {
            runtime.idle.send_replace(Some(RunSummary::default()));
        }
        return Ok(runtime);
    };
    let (ready, wait) = sync_channel(1);
    let worker_events = runtime.worker_events.clone();
    let idle = Arc::clone(&runtime.idle);
    runtime.engine = Some(
        std::thread::Builder::new()
            .name("polylogue-loop".into())
            .spawn(move || {
                let expected = Arc::new(AtomicBool::new(false));
                let _exit = super::lifecycle::WorkerExit::new(
                    worker_events,
                    "engine worker terminated",
                    Some(Arc::clone(&expected)),
                );
                let started = ready.clone();
                let result = engine.run(tx, &rx, &status, lifetime, || {
                    let _ = started.send(Ok(()));
                });
                match result {
                    Ok(summary) => {
                        // Stopping a service loop is the runtime's own shutdown;
                        // an idle loop's end is the run's end.
                        if lifetime == Lifetime::UntilIdle {
                            expected.store(true, Ordering::Release);
                            idle.send_replace(Some(summary));
                        }
                        Ok(())
                    }
                    Err(error) => {
                        let _ = ready.try_send(Err(error.to_string()));
                        Err(std::io::Error::other(error.to_string()))
                    }
                }
            })?,
    );
    wait.recv()
        .map_err(|_| std::io::Error::other("loop thread exited during startup"))?
        .map_err(std::io::Error::other)?;
    let (stop, stopped) = channel();
    runtime.signals_stop = Some(stop);
    let daemon = Arc::clone(&runtime.daemon);
    let worker_events = runtime.worker_events.clone();
    runtime.signals = Some(
        std::thread::Builder::new()
            .name("polylogue-signals".into())
            .spawn(move || {
                let _exit = super::lifecycle::WorkerExit::new(
                    worker_events,
                    "signal worker terminated",
                    None,
                );
                crate::signals::run(daemon, stopped)
            })?,
    );
    let count_commits = runtime
        .daemon
        .writes
        .as_ref()
        .map(WriterSender::completion_commits);
    let daemon = Arc::clone(&runtime.daemon);
    let worker_events = runtime.worker_events.clone();
    runtime.counts = Some(
        std::thread::Builder::new()
            .name("polylogue-counts".into())
            .spawn(move || {
                let _exit = super::lifecycle::WorkerExit::new(
                    worker_events,
                    "count worker terminated",
                    None,
                );
                daemon.warm.count_worker(&daemon.reads, count_commits)
            })?,
    );
    let (stop, stopped) = channel();
    runtime.live_evidence_stop = Some(stop);
    let daemon = Arc::clone(&runtime.daemon);
    let worker_events = runtime.worker_events.clone();
    let paths = crate::live_evidence::Paths {
        spool: settings.spool_dir(polylogue_sources::hook_spool::ORIGIN),
        claude_sessions: polylogue_sources::origin::by_name("claude-code")
            .map(|origin| settings.root(origin.root).join("sessions"))
            .unwrap_or_default(),
        codex_socket: runtime.daemon.acts.codex().1,
        codex_program: runtime.daemon.acts.codex().0,
    };
    let clock = settings.clock.clone();
    runtime.live_evidence = Some(
        std::thread::Builder::new()
            .name("polylogue-live".into())
            .spawn(move || {
                let _exit = super::lifecycle::WorkerExit::new(
                    worker_events,
                    "live evidence worker terminated",
                    None,
                );
                crate::live_evidence::run(daemon, paths, clock, stopped)
            })?,
    );
    // Load the vectors `similar:` ranks by before the first query needs them.
    let daemon = Arc::clone(&runtime.daemon);
    std::thread::Builder::new()
        .name("polylogue-warm".into())
        .spawn(move || {
            let Ok(reader) = daemon.reads.reader() else {
                return;
            };
            if let Ok(settings) = reader.settings_snapshot(daemon.config.requests.machine())
                && let Ok(Some(provider)) = crate::embeddings::Provider::from_settings(
                    &settings,
                    &daemon.config.embedding_credentials,
                )
            {
                let _ = daemon.warm.vectors(&reader, &provider);
            }
        })?;
    Ok(runtime)
}

mod browser;
pub(crate) mod read;

#[cfg(test)]
mod tests;

pub(crate) mod acts;
mod feeds;
pub(crate) mod follow;
