//! Typed answers and the authored writes they perform.
use super::Daemon;
use crate::brief::{self, Ask};
use crate::daemon::Failure;
use command_surface::{Answer, Outcome};
use polylogue_model::user::{Delivery, UserWrite};
use polylogue_model::{Cause, Code};
use polylogue_reading::operation::{Context, Invocation};
use polylogue_store::{StoreError, WriterSender, archive::writer::Committed, user_writes};
use serde_json::Value;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

/// A query's answer over a snapshot of the given freshness: degraded, with a
/// note, when the index is not current or a part is incomplete; empty when
/// it matched nothing; ok otherwise.
pub(crate) fn query_outcome(
    answer: crate::daemon::query::QueryAnswer,
    freshness: &polylogue_model::freshness::ReadFreshness,
) -> Answer<crate::protocol::query::Answer> {
    let crate::daemon::query::QueryAnswer {
        mut value,
        outcome,
        notes,
        ..
    } = answer;
    value.read_position_mut().freshness = Some(freshness.clone());
    let mut notes = notes;
    notes.extend(freshness.notes());
    let degraded = !freshness.up_to_date() || outcome == crate::daemon::query::Outcome::Incomplete;
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
    } else if outcome == crate::daemon::query::Outcome::Empty {
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

pub(crate) fn import_answer(
    report: polylogue_engine::ImportReport,
    dry_run: bool,
) -> Answer<polylogue_engine::ImportReport> {
    let mut missing = Vec::new();
    if !dry_run && (!report.refused.is_empty() || !report.pass.errors.is_empty()) {
        // Staging failures have no entry among the classified members.
        let refused_members: HashSet<_> = report
            .members
            .iter()
            .filter(|member| member.class == "refused")
            .map(|member| &member.path)
            .collect();
        let mut staging_failed = false;
        missing.extend(report.refused.iter().map(|r| {
            let label = if refused_members.contains(&r.path) {
                "refused"
            } else {
                staging_failed = true;
                "failed"
            };
            format!("{label} {}: {}", r.path.display(), r.reason)
        }));
        missing.extend(report.pass.errors.iter().map(ToString::to_string));
        if report.imported.is_empty() {
            let problem = missing.join("\n");
            if let Some(failure) = report.pass.errors.iter().find(|failure| {
                failure.code == Code::NotFound
                    || !matches!(failure.cause, Cause::Content | Cause::Request)
            }) {
                return error(failure.code, problem);
            }
            if staging_failed {
                return error(Code::Io, problem);
            }
            return Failure::refusal(polylogue_model::Reason::ImportRefused { problem }).answer();
        }
        polylogue_store::log!(
            warning,
            "import incomplete: {}",
            missing.join(" | ").replace('\n', " | ")
        );
    }
    if report.user_state.pending > 0 {
        missing.push(crate::protocol::AWAITING_CONFIRMATION.into());
    }
    if let polylogue_engine::Readable::AfterBuild { build } = &report.readable {
        missing.push(format!("stored, not yet readable: {build}"));
    }
    let already_stored = report.imported.is_empty() && !report.already_stored.is_empty();
    let mut answer = if missing.is_empty() {
        Answer::ok(report)
    } else {
        Answer::degraded(report, missing)
    };
    if already_stored && matches!(answer.outcome, Outcome::Ok(_)) {
        answer = answer.with_note("Everything named was already stored.");
    }
    answer
}

pub(crate) fn error<T>(code: polylogue_model::Code, message: impl ToString) -> Answer<T> {
    Failure::new(code, message).answer()
}

/// A failed store operation. Saved preparations have their operation's
/// own declared result and are handled where that preparation is made.
pub(crate) fn store_error<T>(e: &StoreError) -> Answer<T> {
    Failure::store(e).answer()
}

/// How much of a session-start brief's remaining time its sections leave
/// for the answer to reach the caller: a fifth, at most a second.
pub(crate) fn answer_reserve(left: std::time::Duration) -> std::time::Duration {
    (left / 5).min(std::time::Duration::from_secs(1))
}

/// The answer of an advisory that has nothing to offer: also its answer
/// when preparation cannot finish within its cap, on every surface.
pub(crate) fn advisory_silence() -> Answer<crate::brief::BriefReply> {
    crate::protocol::written(
        crate::brief::BriefReply {
            delivery: None,
            prepared: crate::advisory::empty(),
        },
        false,
    )
}

/// Prepares context and, when a sender is named, queues its delivery record
/// and answers without waiting for the writer, so a busy writer never delays
/// a brief. The delivery ID is the request ID, chosen before the text is
/// prepared and written into it; the writer records the delivery, and any
/// later outcome event for it, in queue order. A session-start brief stops
/// its sections early enough to answer before the request's deadline.
pub(crate) fn context(
    d: &Daemon,
    request_id: String,
    ask: &Ask,
    sender: Option<polylogue_model::DeliverySender>,
    budget_tokens: command_surface::TokenBudget,
    operation: &Arc<Context>,
) -> Answer<crate::brief::BriefReply> {
    use polylogue_store::read::limit::ReadLimit;
    let is_advisory = matches!(ask, Ask::Advisory { .. });
    let mut cooldown = if let Ask::Advisory { session, .. } = ask {
        match d
            .advisories
            .available(session, operation.clock.now.as_millisecond())
        {
            Some(cooldown) => Some(cooldown),
            None => {
                return crate::protocol::written(
                    crate::brief::BriefReply {
                        delivery: None,
                        prepared: crate::advisory::empty(),
                    },
                    false,
                );
            }
        }
    } else {
        None
    };
    // Read pending records before opening the snapshot, so every accepted
    // delivery is represented here or in the committed rows we read.
    let pending_deliveries = d
        .writes
        .as_ref()
        .map_or_else(Vec::new, |writes| writes.archive().pending_deliveries());
    let pending_refs: Vec<_> = pending_deliveries
        .iter()
        .filter_map(|pending| match (&pending.write, ask.recipient()) {
            (
                UserWrite::Deliver(delivery) | UserWrite::DeliveredAdvice(delivery),
                Some(recipient),
            ) if delivery.event == polylogue_model::DeliveryEvent::Advice
                && delivery.recipient_id.as_deref() == Some(recipient.to_string().as_str()) =>
            {
                Some(delivery.refs.clone())
            }
            _ => None,
        })
        .collect();
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
    let prepare = || -> Result<_, Box<Answer<crate::brief::BriefReply>>> {
        let reader = d.reads.reader().map_err(|e| Box::new(store_error(&e)))?;
        // A settings error stays the reason similarity cannot rank.
        let provider = if is_advisory {
            Ok(None)
        } else {
            let settings = reader
                .settings_snapshot(&operation.machine)
                .map_err(|error| Box::new(store_error(&error)))?;
            crate::embeddings::Provider::from_settings(&settings, &d.config.embedding_credentials)
        };
        let near =
            |q: &str| super::super::query::ranking_similarity(&reader, &d.warm, &provider, q);
        let near: Option<&crate::daemon::query::NearFn<'_>> =
            (!matches!(provider, Ok(None))).then_some(&near);
        let ranking_identity = crate::embeddings::ranking_key(
            &reader,
            provider.as_ref().ok().and_then(Option::as_ref),
        );
        let prepared_position = reader
            .change_position()
            .map_err(|e| Box::new(store_error(&e)))?
            .archive;
        let prepared = brief::prepare(
            crate::brief::BriefContext {
                reader: &reader,
                pending_deliveries: &pending_deliveries,
                delivery_id: &request_id,
                context: operation,
                ranking_identity: ranking_identity.as_deref(),
                near,
            },
            &d.scope,
            &pending_refs,
            ask,
            budget_tokens,
        )
        .map_err(|e| match &request_limit {
            Some(limit) if limit.check().is_err() => {
                let text = match ask {
                    Ask::Words { text, .. } => Some(text.as_str()),
                    _ => None,
                };
                Box::new(crate::daemon::execution::stopped_read(
                    &e, "context", text, limit,
                ))
            }
            _ => Box::new(store_error(&e)),
        })?;
        Ok((prepared, prepared_position))
    };
    let prepared = match &brief_limit {
        Some(limit) => limit.scope(prepare),
        None => prepare(),
    };
    // An advisory never blocks and fails open: preparation that runs out of
    // its 20 ms cap (intersected with the caller's bound), or whose
    // selection changed under it, answers with nothing on every surface. A
    // store failure is reported, never read as an empty advisory.
    let silent = advisory_silence;
    let (prepared, prepared_position) = match prepared {
        Ok(prepared) => prepared,
        Err(answer)
            if is_advisory
                && (request_limit
                    .as_ref()
                    .is_some_and(|limit| limit.check().is_err())
                    || answer.failure().is_some_and(|failure| {
                        matches!(
                            failure.code.as_deref(),
                            Some("deadline_exceeded" | "cancelled" | "scope_changed")
                        )
                    })) =>
        {
            return silent();
        }
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
            Ask::Words { text, .. } => Some(text.as_str()),
            _ => None,
        };
        return crate::daemon::execution::stopped_read(&error, "context", text, limit);
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
            event: prepared.event,
            cwd: ask.cwd().map(|c| c.to_string_lossy().into_owned()),
            text: prepared.text.clone(),
            refs: prepared.refs.clone(),
            left_out: prepared.left_out.clone(),
            prepared_position: Some(prepared_position),
        });
        let writes = match writer(d) {
            Ok(writes) => writes,
            Err(failure) => return failure.answer(),
        };
        if let Err(error) = operation.check_parent_lifetime() {
            return store_error(&error);
        }
        if let Err(failure) = crate::daemon::handler::hand_over("record delivery") {
            return failure.answer();
        }
        // Queued without the caller's lifetime: the text has been handed
        // over, so its record is written even when the caller has gone.
        let request = operation.request(request_id.clone(), write);
        if is_advisory {
            let session = ask.recipient().expect("advisory recipient");
            cooldown
                .as_mut()
                .expect("advisory cooldown held")
                .insert(session.to_string(), operation.clock.now.as_millisecond());
        }
        match writes.archive().accept_delivery(request) {
            Ok(written) => d.acts.watch_delivery(&request_id, written),
            Err(error) => {
                if let Some(recent) = &mut cooldown {
                    recent.remove(&ask.recipient().expect("advisory recipient").to_string());
                }
                return store_error(&error);
            }
        }
        delivery = Some(request_id);
    }
    crate::protocol::written(crate::brief::BriefReply { delivery, prepared }, false)
}

/// Queues what happened to a delivery and answers at once: the writer
/// applies it after the delivery it names, and a repeated request ID is
/// applied once. The hook waits only for the hand-over and may have exited
/// before the daemon reads its request; the outcome it reports has happened,
/// so it is recorded whether or not the caller is still connected.
pub(crate) fn delivery_event(
    d: &Daemon,
    invocation: &Invocation,
    request_id: String,
    write: UserWrite,
) -> Answer<Value> {
    let writes = match writer(d) {
        Ok(writes) => writes,
        Err(failure) => return failure.answer(),
    };
    if let UserWrite::DeliveryEvent {
        delivery,
        outcome: polylogue_model::DeliveryOutcome::NotSent,
        detail,
    } = &write
        && detail.as_deref() == Some("timeout")
    {
        let pending = writes.archive().pending_deliveries().iter().any(|request| {
            request.id == *delivery && matches!(request.write, UserWrite::Deliver(_))
        });
        if !pending {
            match writes.archive().committed_for(delivery) {
                Ok(Some(request)) if matches!(request.request.write, UserWrite::Deliver(_)) => {}
                Ok(_) => {
                    return ok(
                        serde_json::json!({ "queued": false, "preparation": "unconfirmed" }),
                    );
                }
                Err(error) => return store_error(&error),
            }
        }
    }
    match writes
        .archive()
        .accept_delivery(invocation.request(request_id.clone(), write))
    {
        Ok(written) => {
            d.acts.watch_delivery(&request_id, written);
            crate::protocol::written(
                serde_json::json!({ "id": request_id, "queued": true }),
                false,
            )
        }
        Err(error) => store_error(&error),
    }
}

pub(crate) fn writer(d: &Daemon) -> Result<&WriterSender, Failure> {
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
/// selected generation through the writer queue to its commit. A write
/// accepted locally but not yet confirmed by Sinex is a degraded answer.
pub(crate) fn submit<T: serde::Serialize>(
    d: &Daemon,
    invocation: &Invocation,
    request_id: String,
    w: UserWrite,
    selected_scope: Option<polylogue_store::ReadScopeToken>,
    present: impl FnOnce(polylogue_model::user::UserOutcome) -> T,
) -> Answer<T> {
    submit_result(d, invocation, request_id, w, selected_scope)
        .map(|committed| present(committed.outcome.outcome))
}

pub(crate) fn submit_result(
    d: &Daemon,
    invocation: &Invocation,
    request_id: String,
    w: UserWrite,
    selected_scope: Option<polylogue_store::ReadScopeToken>,
) -> Answer<Committed<user_writes::WriteResult>> {
    let accepted = || -> Result<_, Failure> {
        let writes = writer(d)?;
        invocation.check_parent_lifetime()?;
        crate::daemon::handler::hand_over("write")?;
        Ok(writes)
    };
    let writes = match accepted() {
        Ok(writes) => writes,
        Err(failure) => return failure.answer(),
    };
    let request = invocation.request(request_id, w);
    let preparation = authored_preparation();
    match writes.archive().submit(
        polylogue_store::writer::UserSubmission {
            request,
            scope: selected_scope,
            preparation: Some(preparation),
            parent: invocation.parent_lifetime(),
        },
        None,
    ) {
        Ok(committed) => {
            let pending = committed.outcome.outcome.pending;
            crate::protocol::written(committed, pending)
        }
        Err(error) => store_error(&error),
    }
}

pub(crate) fn write(
    d: &Daemon,
    invocation: &Invocation,
    request_id: String,
    w: UserWrite,
) -> Answer<polylogue_model::user::UserOutcome> {
    submit(d, invocation, request_id, w, None, |outcome| outcome)
}

pub(crate) fn write_from_read(
    d: &Daemon,
    invocation: &Invocation,
    request_id: String,
    w: UserWrite,
    selected_scope: polylogue_store::ReadScopeToken,
) -> Answer<polylogue_model::user::UserOutcome> {
    submit(
        d,
        invocation,
        request_id,
        w,
        Some(selected_scope),
        |outcome| outcome,
    )
}

pub(crate) fn browser_refusal<T>(refused: crate::browser::actions::Refused) -> Answer<T> {
    error(refused.0, refused.0.as_str())
}

pub(crate) fn ok(data: impl serde::Serialize) -> Answer<Value> {
    match serde_json::to_value(data) {
        Ok(data) => Answer::ok(data),
        Err(cause) => error(polylogue_model::Code::Internal, cause),
    }
}

/// The immutable accepted intent, whether confirmed locally or pending in
/// primary mode. The same ID with another intent remains a conflict.
pub(crate) fn accepted_write(
    intended: &UserWrite,
    call: &crate::daemon::handler::Call<'_>,
) -> Result<Option<UserWrite>, Failure> {
    let request = call
        .invocation
        .request(call.request_id.into(), intended.clone());
    let reader = call.reader()?;
    repeated_request(&reader, &request.id, intended, call.invocation)
}

pub(crate) fn repeated_request(
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
pub(crate) fn accepting(d: &Daemon) -> Result<(), Failure> {
    if d.is_stopping() {
        return Err(shutting_down());
    }
    Ok(())
}

/// The refusal of a request that arrives, or was queued, while the daemon
/// stops.
pub(crate) fn shutting_down() -> Failure {
    Failure::shutting_down("the daemon has stopped accepting requests")
}

/// Resolve bulk targets through the same pinned selection and acknowledged
/// execution lifecycle used by reads. The returned references keep the rows' own grain.
pub(crate) fn query_targets(
    d: &Daemon,
    reader: &polylogue_store::Reader<'_>,
    query: &str,
    invocation: &polylogue_reading::operation::Invocation,
) -> Result<Vec<polylogue_model::Reference>, Failure> {
    query_targets_in(d, reader, query, invocation, None)
}

/// A browser selection additionally requires one stored row in its supplied
/// conversation. The selection must resolve to exactly one message, and that
/// message's recorded session membership must be exactly the supplied session.
pub(crate) fn query_targets_in(
    d: &Daemon,
    reader: &polylogue_store::Reader<'_>,
    query: &str,
    invocation: &polylogue_reading::operation::Invocation,
    container: Option<&polylogue_model::Reference>,
) -> Result<Vec<polylogue_model::Reference>, Failure> {
    use polylogue_model::query_objects::{ExecutionPresentation, QueryRun, RunBinding};
    crate::daemon::query::with_ranking(
        reader,
        &d.warm,
        &d.config,
        |embeddings, ranking_identity, near| {
            let operation = d.resolve_operation_context(invocation.clone())?;
            let ask = crate::daemon::query::Ask {
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
            let options = crate::daemon::query::execution_options(
                reader,
                &ask,
                uuid::Uuid::new_v4().as_u128() as u64,
            )?;
            let selection = reader.prepare_query_selection(query, &options)?;
            if selection.program().shape != polylogue_query::Shape::Rows {
                return Err(Failure::bad_request(
                    "annotations require identifiable rows, not a grouped or aggregate population",
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
                .map_err(Failure::internal)?;
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
                id: polylogue_reading::operation::new_run_id(),
                query_id: Some(selection.question_id()),
                at: operation.ran_at,
                read_position: reader.change_position()?,
                machine: operation.machine.clone(),
                authorship: operation.authorship.clone(),
                binding: RunBinding::Original {
                    execution: Box::new(binding),
                },
            };
            let hash = run.binding_hash().map_err(Failure::internal)?;
            let writes = writer(d)?;
            operation.check_parent_lifetime()?;
            writes
                .archive()
                .query_object(object, operation.parent_lifetime())?;
            polylogue_reading::execution::record(
                writes,
                operation.parent_lifetime(),
                run,
                hash,
                || d.config.requests.now(),
                || (|| {
                    if !selection.program().incomplete.is_empty() {
                        return Err(Failure::not_yet_available(
                            "the query population is incomplete; no annotation was written",
                        ));
                    }
                    let mut row_count = 0_u64;
                    let mut targets = Vec::new();
                    let mut seen = HashSet::new();
                    reader.visit_selection::<Failure>(&selection, polylogue_store::read::VisitFrom::First, Some(near), |row| {
                        row_count = row_count
                            .checked_add(1)
                            .ok_or_else(|| Failure::internal("selected row count exceeds u64"))?;
                        let reference = row
                            .target
                            .as_ref()
                            .map(polylogue_model::read::ReadTarget::reference)
                            .ok_or_else(|| {
                                Failure::not_yet_available(
                                    "the selected row has no source reference",
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
                            return Err(Failure::conflict(match targets.first() {
                                None => "the supplied message is missing, ambiguous or belongs to another conversation; capture the conversation and retry".to_string(),
                                Some(first) => format!(
                                    "the supplied source is not a recorded message; the query selected {row_count} rows, first {first}"
                                ),
                            }));
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
                            return Err(Failure::conflict(
                                "the message has missing, ambiguous or mismatching recorded conversation membership",
                            ));
                        }
                    }
                    if container.is_some() && row_count != 1 {
                        return Err(Failure::conflict(
                            "the supplied message is missing or ambiguous; capture the conversation and retry",
                        ));
                    }
                    let count = u64::try_from(targets.len())
                        .map_err(|_| Failure::internal("selected row count exceeds u64"))?;
                    Ok(polylogue_reading::execution::Completed {
                        value: targets,
                        count: Some(count),
                        exact: selection.program().incomplete.is_empty(),
                        next: None,
                    })
                })().map_err(|failure: Failure| polylogue_reading::Error::Operation(failure.error)),
            ).map_err(Failure::from)
        },
    )
}
