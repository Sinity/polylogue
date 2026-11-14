//! What the browser extension and the operator ask about web chats: chat
//! actions (the operator queues, the extension claims and reports), backfill
//! jobs (the operator starts, the extension claims and reports), and what
//! the archive holds of captured conversations. Every change goes through
//! the one writer ([`polylogue_store::chat_actions`],
//! [`polylogue_store::backfill_jobs`]).

use super::{Daemon, browser_refusal, ok, store_error, writer};
use crate::browser::{actions, backfill};
use crate::cli::print;
use crate::daemon::Failure;
use crate::daemon::handler::{Call, Handler};
use crate::operation::Invocation;
use crate::protocol::browser as protocol;
use base64::Engine as _;
use command_surface::Answer;
use polylogue_model::user::{UserOutcome, UserWrite};
use polylogue_store::{Reader, StoreError, user_writes};
use serde_json::Value;
use serde_json::json;

fn refused(e: &StoreError) -> Failure {
    match e {
        StoreError::ChatAction(code) | StoreError::Backfill(code) => {
            let refused = actions::refusal(code);
            Failure::new(refused.0, refused.0)
        }
        other => Failure::store(other),
    }
}

fn enabled(invocation: &Invocation, reader: &Reader<'_>) -> Result<bool, Failure> {
    match reader.setting_value("browser.actions", &invocation.machine) {
        Ok(Some(polylogue_model::settings::Value::Boolean(value))) => Ok(value),
        Ok(_) => Err(Failure::new(
            "invalid",
            "browser.actions has no typed boolean value",
        )),
        Err(error) => Err(Failure::store(&error)),
    }
}

/// Writes `write`; `None` when it found nothing to claim. Chat actions and
/// backfill jobs carry their own retry keys, so each write gets a new
/// request ID.
fn apply(
    d: &Daemon,
    invocation: &Invocation,
    write: UserWrite,
) -> Result<Option<UserOutcome>, Failure> {
    let writes = writer(d)?;
    invocation.check_parent_lifetime()?;
    let request = invocation.request(user_writes::new_request_id(), write);
    let result = match invocation.parent_lifetime() {
        Some(parent) => writes.user_with_parent(request, parent),
        None => writes.user(request),
    };
    match result {
        Ok(outcome) => Ok(Some(outcome)),
        Err(StoreError::NotFound(_)) => Ok(None),
        Err(e) => Err(refused(&e)),
    }
}

/// Writes `write`, then answers with the chat action it names.
fn chat_then_view(
    d: &Daemon,
    invocation: &Invocation,
    write: UserWrite,
) -> Answer<protocol::ChatAnswer> {
    let outcome = match apply(d, invocation, write) {
        Ok(Some(outcome)) => outcome,
        Ok(None) => {
            return Answer::ok(protocol::ChatAnswer::Current(protocol::ChatCurrent {
                enabled: None,
                action: None,
            }));
        }
        Err(failure) => return failure.answer(),
    };
    if outcome.pending {
        return crate::protocol::written(protocol::ChatAnswer::Receipt(outcome), true);
    }
    let reader = match d.reads.reader() {
        Ok(reader) => reader,
        Err(e) => return store_error(&e),
    };
    match reader.chat_action(&outcome.id) {
        Ok(Some(action)) => Answer::ok(protocol::ChatAnswer::Current(protocol::ChatCurrent {
            enabled: None,
            action: Some(actions::view(&action, invocation.at.unix_ms())),
        })),
        Ok(None) => browser_refusal(actions::refusal("unknown_chat_action")),
        Err(e) => store_error(&e),
    }
}

fn backfill_then_view(
    d: &Daemon,
    invocation: &Invocation,
    write: UserWrite,
) -> Answer<protocol::BackfillAnswer> {
    let outcome = match apply(d, invocation, write) {
        Ok(Some(outcome)) => outcome,
        Ok(None) => {
            return Answer::ok(protocol::BackfillAnswer::Current(
                protocol::BackfillCurrent { job: None },
            ));
        }
        Err(failure) => return failure.answer(),
    };
    if outcome.pending {
        return crate::protocol::written(protocol::BackfillAnswer::Receipt(outcome), true);
    }
    let reader = match d.reads.reader() {
        Ok(reader) => reader,
        Err(e) => return store_error(&e),
    };
    match reader.backfill_job(&outcome.id) {
        Ok(Some(job)) => Answer::ok(protocol::BackfillAnswer::Current(
            protocol::BackfillCurrent {
                job: Some(backfill::view(&job, invocation.at.unix_ms())),
            },
        )),
        Ok(None) => browser_refusal(actions::refusal("unknown_backfill_job")),
        Err(e) => store_error(&e),
    }
}

enum PreparedFiles {
    Complete(Vec<polylogue_model::user::ChatAttachment>),
    Pending(protocol::PendingCapture),
}

/// Stores each file's bytes as a blob and makes its attachment row.
fn store_files(d: &Daemon, files: &[actions::ChatFile]) -> Result<PreparedFiles, Failure> {
    let writes = writer(d)?;
    let mut attachments = Vec::with_capacity(files.len());
    for file in files {
        let result = d
            .archive
            .prepare_decoded(&file.bytes)
            .and_then(|(new_blob, prepared)| writes.store_blob(new_blob, prepared));
        match result {
            Ok(blob) => attachments.push(actions::attachment(file, blob)),
            Err(StoreError::PrimaryPending { identity }) => {
                return Ok(PreparedFiles::Pending(protocol::PendingCapture {
                    identity,
                }));
            }
            Err(error) => return Err(Failure::from(error)),
        }
    }
    Ok(PreparedFiles::Complete(attachments))
}

/// The browser handlers: each answers with the JSON view the extension reads.
macro_rules! browser {
    ($(#[$doc:meta])* $name:ident, $arguments:ty, $answer:ty, $print:expr, |$args:ident, $call:ident| $body:expr) => {
        $(#[$doc])*
        pub(crate) struct $name;
        impl Handler for $name {
            type Arguments = $arguments;
            type Answer = $answer;
            const PRINT: fn(&Value, bool) -> String = $print;
            fn run($args: Self::Arguments, $call: &Call<'_>) -> Result<Answer<Self::Answer>, Failure> {
                Ok($body)
            }
        }
    };
}

fn reader<'a>(call: &'a Call<'_>) -> Result<Reader<'a>, Failure> {
    call.reader()
}

browser!(
    ChatSend,
    protocol::ChatSend,
    protocol::ChatAnswer,
    print::json,
    |a, call| {
        let (d, invocation) = (call.daemon, call.invocation);
        let reader = reader(call)?;
        match enabled(invocation, &reader) {
            Ok(true) => {}
            Ok(false) => return Ok(browser_refusal(actions::Refused("browser_actions_off"))),
            Err(failure) => return Err(failure),
        }
        drop(reader);
        let action = actions::ChatRequest {
            provider: a.provider,
            conversation: a.conversation,
            project: a.project,
            text: a.text,
            model: a.model,
            effort: a.effort,
            submit: a.submit,
        };
        // Checked before any file is stored.
        if let Err(r) = actions::new_action(call.request_id, &action, vec![]) {
            return Ok(browser_refusal(r));
        }
        let stored = match store_files(d, &a.files)? {
            PreparedFiles::Complete(files) => files,
            PreparedFiles::Pending(receipt) => {
                return Ok(crate::protocol::written(
                    protocol::ChatAnswer::Capture(receipt),
                    true,
                ));
            }
        };
        match actions::new_action(call.request_id, &action, stored) {
            Ok(new) => chat_then_view(d, invocation, UserWrite::ChatEnqueue(new)),
            Err(r) => browser_refusal(r),
        }
    }
);

browser!(
    ChatList,
    protocol::ChatList,
    Value,
    print::chat_actions,
    |_a, call| {
        let reader = reader(call)?;
        let now = call.invocation.at.unix_ms();
        match reader.chat_actions() {
            Ok(list) => ok(json!({
                "actions": list.iter().map(|a| actions::view(a, now)).collect::<Vec<_>>(),
            })),
            Err(e) => store_error(&e),
        }
    }
);

browser!(
    /// The extension instance `by` takes the oldest queued chat action.
    ChatClaim,
    protocol::ChatClaim,
    protocol::ChatAnswer,
    print::json,
    |a, call| {
        let reader = reader(call)?;
        let on = enabled(call.invocation, &reader)?;
        drop(reader);
        if !on {
            return Ok(Answer::ok(protocol::ChatAnswer::Current(protocol::ChatCurrent {enabled: Some(false), action: None})));
        }
        chat_then_view(
            call.daemon,
            call.invocation,
            UserWrite::ChatClaim { by: a.by },
        )
    }
);

browser!(
    ChatReport,
    protocol::ChatReport,
    protocol::ChatAnswer,
    print::json,
    |a, call| {
        let update = a.update;
        chat_then_view(
            call.daemon,
            call.invocation,
            UserWrite::ChatReport {
                action: a.id,
                by: update.by,
                outcome: update.outcome,
                phase: update.phase,
                detail: update.detail,
                receipt: update.receipt,
            },
        )
    }
);

fn decide(
    call: &Call<'_>,
    id: String,
    approve: bool,
    by: String,
    detail: Option<String>,
) -> Answer<protocol::ChatAnswer> {
    chat_then_view(
        call.daemon,
        call.invocation,
        UserWrite::ChatDecide {
            action: id,
            approve,
            by,
            detail,
        },
    )
}

browser!(
    ChatApprove,
    protocol::ChatDecide,
    protocol::ChatAnswer,
    print::json,
    |a, call| decide(call, a.id, true, "operator".into(), None)
);
browser!(
    ChatDecline,
    protocol::ChatDecide,
    protocol::ChatAnswer,
    print::json,
    |a, call| decide(call, a.id, false, "operator".into(), None)
);
browser!(
    /// A decision the extension's page reports for the operator.
    ChatDecision,
    protocol::ChatDecision,
    protocol::ChatAnswer,
    print::json,
    |a, call| decide(call, a.id, a.approve, a.by, a.detail)
);

browser!(
    ChatAttachment,
    protocol::ChatAttachment,
    Value,
    print::json,
    |a, call| {
        let reader = reader(call)?;
        match reader.chat_attachment(&a.id, &a.attachment) {
            Ok((attachment, bytes)) => ok(json!({
                "media_type": attachment.media_type,
                "bytes": base64::engine::general_purpose::STANDARD.encode(bytes),
            })),
            Err(e) => refused(&e).answer(),
        }
    }
);

browser!(
    BackfillStart,
    protocol::BackfillStart,
    protocol::BackfillAnswer,
    print::backfill_jobs,
    |a, call| {
        match backfill::new_job(&a.provider, a.since.as_deref()) {
            Ok(new) => {
                backfill_then_view(call.daemon, call.invocation, UserWrite::BackfillStart(new))
            }
            Err(r) => browser_refusal(r),
        }
    }
);

browser!(
    BackfillList,
    protocol::BackfillList,
    Value,
    print::backfill_jobs,
    |_a, call| {
        let reader = reader(call)?;
        let now = call.invocation.at.unix_ms();
        match reader.backfill_jobs() {
            Ok(jobs) => ok(json!({
                "jobs": jobs.iter().map(|job| backfill::view(job, now)).collect::<Vec<_>>(),
            })),
            Err(e) => store_error(&e),
        }
    }
);

browser!(
    BackfillCancel,
    protocol::BackfillCancel,
    protocol::BackfillAnswer,
    print::backfill_jobs,
    |a, call| {
        backfill_then_view(
            call.daemon,
            call.invocation,
            UserWrite::BackfillCancel { job: a.id },
        )
    }
);

browser!(
    BackfillClaim,
    protocol::BackfillClaim,
    protocol::BackfillAnswer,
    print::json,
    |a, call| {
        backfill_then_view(
            call.daemon,
            call.invocation,
            UserWrite::BackfillClaim { by: a.by },
        )
    }
);

browser!(
    BackfillReport,
    protocol::BackfillReport,
    protocol::BackfillAnswer,
    print::json,
    |a, call| {
        backfill_then_view(
            call.daemon,
            call.invocation,
            UserWrite::BackfillReport {
                job: a.id,
                by: a.by,
                report: a.report,
            },
        )
    }
);

browser!(
    Conversations,
    protocol::Conversations,
    Value,
    print::json,
    |a, call| {
        let reader = reader(call)?;
        let sessions: Vec<_> = a
            .ids
            .iter()
            .map(|id| polylogue_sources::browser_capture::session_for(&a.provider, id))
            .collect();
        let known: Vec<_> = sessions.iter().flatten().cloned().collect();
        let mut found = match reader.captured_sessions(&known) {
            Ok(found) => found.into_iter(),
            Err(e) => return Ok(store_error(&e)),
        };
        let conversations: Vec<_> = a
            .ids
            .iter()
            .zip(&sessions)
            .map(|(id, session)| {
                let held = session.as_ref().and_then(|_| found.next().flatten());
                json!({
                    "id": id,
                    "session": session.as_ref().map(ToString::to_string),
                    "messages": held.as_ref().map(|h| h.messages),
                    "downloading": held.as_ref().is_some_and(|h| h.downloading),
                    "last_captured_at": held.and_then(|h| h.last_captured_at),
                })
            })
            .collect();
        ok(json!({ "provider": a.provider, "conversations": conversations }))
    }
);

browser!(
    StoredFiles,
    protocol::StoredFiles,
    Value,
    print::json,
    |a, call| {
        let Some(session) =
            polylogue_sources::browser_capture::session_for(&a.provider, &a.conversation)
        else {
            return Ok(ok(json!({ "files": [] })));
        };
        let reader = reader(call)?;
        match reader.stored_files(&session) {
            Ok(files) => match reader.downloading_files(&session) {
                Ok(downloading) => ok(json!({ "files": files, "downloading": downloading })),
                Err(e) => store_error(&e),
            },
            Err(e) => store_error(&e),
        }
    }
);
