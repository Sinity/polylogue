//! Application operations and their shared archive state. Socket, HTTP and
//! MCP adapters use the same typed requests; transport framing stays outside.

mod execution;
mod status;

use super::Failure;
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
    /// Runs every request's work within bounded slots.
    pub(crate) executor: super::executor::Executor,
    pub(crate) completion_path: std::path::PathBuf,
    pub(crate) callers: crate::protocol::caller::hooks::Registry,
    operations: std::sync::OnceLock<std::sync::Weak<command_surface::OperationTable>>,

    pub(crate) warm: super::warm::Warm,
    /// Sessions by working directory, kept current by the scope worker.
    pub(crate) scope: super::scope::SharedScopeIndex,
    /// Briefs answered whose delivery records are still queued.
    pub(crate) queued_deliveries: super::queued_deliveries::QueuedDeliveries,
    pub(crate) advisories: crate::advisory::Cooldowns,
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
        if let Some(registration) = command.registration() {
            registration
                .admit(
                    &request.arguments,
                    command,
                    &crate::commands::Caller {
                        surface: crate::commands::Surface::Http,
                        cwd: None,
                    },
                )
                .map_err(|failure| failure.error)?;
        }
        let mcp = command.tool.map_or(serde_json::Value::Null, |tool| {
            let call = crate::mcp::ToolCall {
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

impl Daemon {
    /// Writes are paused until the Sinex configuration is fixed or Sinex
    /// can be reached.
    pub(crate) fn blocked(&self) -> bool {
        self.mode.blocked()
    }
    pub(crate) fn write_refusal(&self) -> Failure {
        self.mode.write_refusal()
    }
    /// Holds idle-priority jobs back while an ordinary request runs.
    pub(super) fn foreground(&self) -> super::schema::Foreground<'_> {
        self.idle_admission.foreground()
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
/// The answer of an advisory that has nothing to offer: also its answer
/// when preparation cannot finish within its cap, on every surface.
fn advisory_silence() -> Answer<Value> {
    crate::protocol::written(
        serde_json::json!({"delivery":null,"prepared":crate::advisory::empty()}),
        false,
    )
}

fn context(
    d: &Daemon,
    request_id: String,
    ask: &Ask,
    sender: Option<String>,
    budget_tokens: command_surface::TokenBudget,
    operation: &Arc<Context>,
) -> Answer<Value> {
    use polylogue_store::read::limit::ReadLimit;
    let is_advisory = matches!(ask, Ask::Advisory { .. });
    #[cfg(feature = "sinex")]
    if is_advisory
        && let ModeState::Primary { connection } = &d.mode
        && !connection.advisory_history_ready()
    {
        return crate::protocol::written(
            serde_json::json!({"delivery":null,"prepared":crate::advisory::empty()}),
            false,
        );
    }
    let mut cooldown = if let Ask::Advisory { session, .. } = ask {
        match d
            .advisories
            .available(session, operation.clock.now.as_millisecond())
        {
            Some(cooldown) => Some(cooldown),
            None => {
                return crate::protocol::written(
                    serde_json::json!({"delivery":null,"prepared":crate::advisory::empty()}),
                    false,
                );
            }
        }
    } else {
        None
    };
    let pending_advice = if let Ask::Advisory { session, .. } = ask {
        match d.queued_deliveries.pending_advice(&session.to_string()) {
            Some(pending) => pending,
            None => {
                return crate::protocol::written(
                    serde_json::json!({"delivery":null,"prepared":crate::advisory::empty()}),
                    false,
                );
            }
        }
    } else {
        Vec::new()
    };
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
    let queued = (!is_advisory)
        .then(|| {
            ask.recipient()
                .and_then(|recipient| d.queued_deliveries.latest(&recipient.to_string()))
        })
        .flatten();
    let prepare = || -> Result<_, Box<Answer<Value>>> {
        let reader = d.reads.reader().map_err(|e| Box::new(store_error(&e)))?;
        let mut applied_advice = Vec::new();
        for (id, _) in &pending_advice {
            if reader
                .delivery_refs(id)
                .map_err(|error| Box::new(store_error(&error)))?
                .is_some()
            {
                applied_advice.push(id.clone());
            }
        }
        d.queued_deliveries.advice_applied(&applied_advice);
        let pending_refs: Vec<_> = pending_advice
            .iter()
            .map(|(_, refs)| refs.clone())
            .collect();
        let provider = if is_advisory {
            None
        } else {
            let settings = reader
                .settings_snapshot(&operation.machine)
                .map_err(|error| Box::new(store_error(&error)))?;
            crate::embeddings::Provider::from_settings(&settings, &d.config.embedding_credentials)
                .ok()
                .flatten()
        };
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
            &pending_refs,
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
    // An advisory never blocks and fails open: preparation that runs out of
    // its 20 ms cap (intersected with the caller's bound), or finds its
    // delivery history incomplete, answers with nothing on every surface.
    let silent = advisory_silence;
    let (prepared, selected_scope, written_up_to) = match prepared {
        Ok(prepared) => prepared,
        Err(_) if is_advisory => return silent(),
        Err(answer) => return *answer,
    };
    if is_advisory
        && request_limit
            .as_ref()
            .is_some_and(|limit| limit.check().is_err())
    {
        return silent();
    }
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
        if let Err(error) = operation.check_parent_lifetime() {
            return store_error(&error);
        }
        if let Err(failure) = super::handler::finish_read_phase("record delivery") {
            return failure.answer();
        }
        // Queued without the caller's lifetime: the text has been handed
        // over, so its record is written even when the caller has gone.
        let request = operation.request(request_id.clone(), write);
        let since = briefing::LastBrief {
            at: request.at,
            seq: written_up_to,
        };
        if is_advisory {
            // A positive preparation reserves this session while its record/ACK
            // is unconfirmed. The stored outcome is still prepared until emitted.
            let session = ask.recipient().expect("advisory recipient");
            let recent = cooldown.as_mut().expect("advisory cooldown held");
            recent.insert(session.to_string(), operation.clock.now.as_millisecond());
            match d
                .queued_deliveries
                .try_submit(writes, request, Some(selected_scope))
            {
                Ok(true) => {}
                Ok(false) => {
                    recent.remove(&session.to_string());
                    return crate::protocol::written(
                        serde_json::json!({"delivery":null,"prepared":crate::advisory::empty()}),
                        false,
                    );
                }
                Err(error) => return store_error(&error),
            }
        } else {
            let recipient = ask.recipient().map(|r| (r.to_string(), since));
            if let Err(error) =
                d.queued_deliveries
                    .submit(writes, request, Some(selected_scope), recipient)
            {
                return store_error(&error);
            }
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
        let writes = writer(d)?;
        invocation.check_parent_lifetime()?;
        super::handler::finish_read_phase("write")?;
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
        return Err(shutting_down());
    }
    Ok(())
}

/// The refusal of a request that arrives, or was queued, while the daemon
/// stops.
pub(crate) fn shutting_down() -> Failure {
    Failure::new("shutting_down", "the daemon has stopped accepting requests")
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
            operation.check_parent_lifetime()?;
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
    live_evidence_stop: Option<Sender<()>>,
    live_evidence: Option<JoinHandle<()>>,
    /// Records hosted-connection outcomes and runs resumes.
    acts: Option<JoinHandle<()>>,
    completion: Option<crate::completion_catalog::publisher::Publisher>,
    scope: Option<super::scope::ScopeWorker>,
    completion_scope: Option<polylogue_store::ScopeInvalidation>,
    engine: Option<JoinHandle<std::io::Result<()>>>,
    /// Ends the loop's pass at its next batch when the daemon stops.
    pub(crate) engine_stop: Option<polylogue_engine::StopRequest>,
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

    /// Refuses new requests, wakes readers waiting for a reader slot and
    /// asks the loop to end its pass at the next batch, before transports
    /// and workers are joined. Readers and writes already running finish as
    /// usual.
    pub fn stop_accepting(&self) {
        self.daemon.stopping.store(true, Ordering::Release);
        if let Some(stop) = &self.engine_stop {
            stop.request();
        }
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
        if self.acts.as_ref().is_some_and(JoinHandle::is_finished) {
            return Some("acts worker terminated");
        }
        self.completion
            .as_ref()
            .and_then(|worker| worker.worker_failure())
    }

    pub fn shutdown(&mut self) -> std::io::Result<()> {
        self.stop_accepting();
        // Transports have stopped; queued requests are refused and running
        // ones end (the closed reader pool ends any still waiting for a read).
        // Work that ignores its cancellation is left behind after the bound,
        // so a stop never hangs on it.
        let running = self.daemon.executor.stop(super::executor::STOP_WAIT);
        if running > 0 {
            eprintln!(
                "polylogue: {running} request(s) did not stop within {} s; stopping without them",
                super::executor::STOP_WAIT.as_secs()
            );
        }
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
        if let Some(stop) = self.live_evidence_stop.take() {
            let _ = stop.send(());
        }
        if let Some(worker) = self.live_evidence.take()
            && worker.join().is_err()
        {
            failure.get_or_insert_with(|| std::io::Error::other("live evidence worker panicked"));
        }
        if let Some(worker) = self.acts.take() {
            self.daemon.acts.stop_reports();
            if worker.join().is_err() {
                failure.get_or_insert_with(|| std::io::Error::other("acts worker panicked"));
            }
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

impl ArchiveMode {
    pub(crate) fn needs_engine(&self) -> bool {
        match self {
            Self::Standalone => true,
            #[cfg(feature = "sinex")]
            Self::Primary { .. } => true,
            #[cfg(feature = "sinex")]
            Self::Blocked { .. } => false,
        }
    }
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
    stopped_attempt: Option<polylogue_model::batch::StoppedAttempt>,
) -> std::io::Result<Runtime> {
    let root = deployment.archive.as_path();
    if let Some(stopped) = super::storage_stop::this_boot(root) {
        return Err(std::io::Error::other(stopped));
    }
    let completion_path = deployment.paths.completion_catalog(root);
    let to_io = |e: &dyn std::fmt::Display| std::io::Error::other(e.to_string());
    let pointer = polylogue_store::completion::CatalogPointer::new(completion_path.clone());
    let invalidate = move || pointer.invalidate();
    let settings = config.clone();
    let attempt = || {
        stopped_attempt.ok_or_else(|| {
            std::io::Error::other("the executable identity is required to start the Engine")
        })
    };
    let (archive, lock, mode, engine, completion_scope) = match mode {
        ArchiveMode::Standalone => {
            let attempt = attempt()?;
            let (archive, lock, scope) = Archive::open_or_create_with_invalidator(root, invalidate)
                .map_err(std::io::Error::other)?;
            let engine = Engine::new(config, archive.clone(), attempt).map_err(|e| to_io(&e))?;
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
            let attempt = attempt()?;
            let (archive, lock, scope) =
                Archive::open_primary_with_invalidator(root, &sinex.store, invalidate)
                    .map_err(std::io::Error::other)?;
            let connection = Arc::new(
                polylogue_sinex::Sinex::open(&sinex, &deployment.primary_auth, &settings.host)
                    .map_err(std::io::Error::other)?,
            );
            let engine = Engine::on_sinex(config, archive.clone(), connection.clone(), attempt)
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
    // One reader per ordinary executor place and one for the background
    // slot; the pool keeps one more for interactive answers.
    let readers = settings.workers + 1;
    let reads = ReadPool::with_budget(&archive, readers, per_reader * readers as u64)
        .map_err(|e| to_io(&e))?;
    let (tx, rx) = channel();
    let (act_reports, act_reports_received) = channel();
    // The loop's first pass starts as soon as it runs.
    let status = Arc::new(Mutex::new(LoopStatus {
        busy: engine.is_some(),
        ..LoopStatus::default()
    }));
    let mail = crate::mailbox::dir(&archive.root);
    let hosts = crate::hosting::dir(&archive.root);
    let warm = super::warm::Warm::new(&archive);
    let daemon = Arc::new(Daemon {
        config: deployment,
        operations: std::sync::OnceLock::new(),
        archive,
        reads,
        executor: super::executor::Executor::new(settings.workers),
        completion_path,
        callers: crate::protocol::caller::hooks::Registry::new(
            settings.spool_dir(polylogue_sources::hook_spool::ORIGIN),
        ),
        warm,
        scope: super::scope::SharedScopeIndex::default(),
        queued_deliveries: super::queued_deliveries::QueuedDeliveries::default(),
        advisories: crate::advisory::Cooldowns::default(),
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
            act_reports,
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
    let acts_worker = {
        let daemon = Arc::clone(&daemon);
        std::thread::Builder::new()
            .name("polylogue-acts".into())
            .spawn(move || acts::run_reports(daemon, act_reports_received))?
    };
    // Hosted sessions outlive a daemon restart: their hosts, or Codex's
    // app-server, kept them running; connect to each again.
    daemon.acts.reattach();
    // Pings and notices left pending raise their running sessions' mailbox
    // flags; those to sessions no longer running are settled.
    acts::settle_ended(&daemon, daemon.config.requests.now().unix_ms(), true);
    let operations = Arc::new(crate::commands::table(Arc::clone(&daemon)));
    assert!(
        daemon.operations.set(Arc::downgrade(&operations)).is_ok(),
        "runtime installs its operation table once"
    );
    let mut runtime = Runtime {
        daemon,
        operations,
        signals_stop: None,
        signals: None,
        live_evidence_stop: None,
        live_evidence: None,
        acts: Some(acts_worker),
        completion: None,
        scope: None,
        completion_scope: Some(completion_scope),
        engine: None,
        engine_stop: None,
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
    // A failed register sync stops the writer; the daemon then stops too,
    // with its storage exit code.
    let storage_events = runtime.worker_events.clone();
    std::thread::Builder::new()
        .name("polylogue-storage".into())
        .spawn(move || {
            polylogue_store::storage_failure::wait();
            storage_events.send_replace(Some("the register failed a disk sync"));
        })?;
    runtime.engine_stop = Some(engine.stop_request());
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

pub(crate) mod browser;
pub(crate) mod handlers;
pub(crate) mod read;

#[cfg(test)]
mod tests;

pub(crate) mod acts;
pub(crate) mod feeds;
pub(crate) mod follow;
