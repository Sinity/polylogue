//! What the browser extension and the operator ask about web chats: chat
//! actions (the operator queues, the extension claims and reports), backfill
//! jobs (the operator starts, the extension claims and reports), and what
//! the archive holds of captured conversations. Every change goes through
//! the one writer ([`polylogue_store::chat_actions`],
//! [`polylogue_store::backfill_jobs`]).

use super::{Daemon, browser_refusal, error, ok, store_error, writer};
use crate::browser::{actions, backfill};
use crate::daemon::{Failure, Request};
use crate::operation::Invocation;
use base64::Engine as _;
use command_surface::Answer;
use polylogue_model::user::UserWrite;
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
fn apply(d: &Daemon, invocation: &Invocation, write: UserWrite) -> Result<Option<String>, Failure> {
    let writes = writer(d)?;
    invocation.check_parent_admission()?;
    let request = invocation.request(user_writes::new_request_id(), write);
    let result = match invocation.parent_lifetime() {
        Some(parent) => writes.user_with_parent(request, parent),
        None => writes.user(request),
    };
    match result {
        Ok(outcome) => Ok(Some(outcome.id)),
        Err(StoreError::NotFound(_)) => Ok(None),
        Err(e) => Err(refused(&e)),
    }
}

/// Writes `write`, then answers with the chat action it names.
fn chat_then_view(d: &Daemon, invocation: &Invocation, write: UserWrite) -> Answer<Value> {
    let id = match apply(d, invocation, write) {
        Ok(Some(id)) => id,
        Ok(None) => return ok(json!({ "action": null })),
        Err(failure) => return failure.answer(),
    };
    let reader = match d.reads.reader() {
        Ok(reader) => reader,
        Err(e) => return store_error(&e),
    };
    match reader.chat_action(&id) {
        Ok(Some(a)) => ok(json!({ "action": actions::view(&a, invocation.at.unix_ms()) })),
        Ok(None) => browser_refusal(actions::refusal("unknown_chat_action")),
        Err(e) => store_error(&e),
    }
}

/// Writes `write`, then answers with the backfill job it names.
fn backfill_then_view(d: &Daemon, invocation: &Invocation, write: UserWrite) -> Answer<Value> {
    let id = match apply(d, invocation, write) {
        Ok(Some(id)) => id,
        Ok(None) => return ok(json!({ "job": null })),
        Err(failure) => return failure.answer(),
    };
    let reader = match d.reads.reader() {
        Ok(reader) => reader,
        Err(e) => return store_error(&e),
    };
    match reader.backfill_job(&id) {
        Ok(Some(job)) => ok(json!({ "job": backfill::view(&job, invocation.at.unix_ms()) })),
        Ok(None) => browser_refusal(actions::refusal("unknown_backfill_job")),
        Err(e) => store_error(&e),
    }
}

/// Stores each file's bytes as a blob and makes its attachment row.
fn store_files(
    d: &Daemon,
    files: &[actions::ChatFile],
) -> Result<Vec<polylogue_model::user::ChatAttachment>, Failure> {
    let writes = writer(d)?;
    files
        .iter()
        .map(|file| {
            d.archive
                .prepare_decoded(&file.bytes)
                .and_then(|(new_blob, prepared)| writes.store_blob(new_blob, prepared))
                .map(|blob| actions::attachment(file, blob))
                .map_err(Failure::from)
        })
        .collect()
}

pub(super) fn handle(
    request: Request,
    d: &Daemon,
    reader: Reader<'_>,
    invocation: &Invocation,
) -> Answer<Value> {
    let now = invocation.at.unix_ms();
    match request {
        Request::ChatSend {
            request_id,
            action,
            files,
        } => {
            match enabled(invocation, &reader) {
                Ok(true) => {}
                Ok(false) => {
                    return browser_refusal(actions::Refused("browser_actions_off"));
                }
                Err(failure) => return failure.answer(),
            }
            drop(reader);
            // Checked before any file is stored.
            if let Err(r) = actions::new_action(&request_id, &action, vec![]) {
                return browser_refusal(r);
            }
            let stored = match store_files(d, &files) {
                Ok(stored) => stored,
                Err(failure) => return failure.answer(),
            };
            match actions::new_action(&request_id, &action, stored) {
                Ok(new) => chat_then_view(d, invocation, UserWrite::ChatEnqueue(new)),
                Err(r) => browser_refusal(r),
            }
        }
        Request::ChatActions => match reader.chat_actions() {
            Ok(list) => ok(json!({
                "actions": list.iter().map(|a| actions::view(a, now)).collect::<Vec<_>>(),
            })),
            Err(e) => store_error(&e),
        },
        Request::ChatAction { id } => match reader.chat_action(&id) {
            Ok(Some(a)) => ok(json!({ "action": actions::view(&a, now) })),
            Ok(None) => browser_refusal(actions::refusal("unknown_chat_action")),
            Err(e) => store_error(&e),
        },
        Request::ChatClaim { by } => {
            let on = match enabled(invocation, &reader) {
                Ok(on) => on,
                Err(failure) => return failure.answer(),
            };
            drop(reader);
            if !on {
                return ok(json!({ "enabled": false, "action": null }));
            }
            chat_then_view(d, invocation, UserWrite::ChatClaim { by })
        }
        Request::ChatReport { id, update } => {
            drop(reader);
            chat_then_view(
                d,
                invocation,
                UserWrite::ChatReport {
                    action: id,
                    by: update.by,
                    outcome: update.outcome,
                    phase: update.phase,
                    detail: update.detail,
                    receipt: update.receipt,
                },
            )
        }
        Request::ChatDecide {
            id,
            approve,
            by,
            detail,
        } => {
            drop(reader);
            chat_then_view(
                d,
                invocation,
                UserWrite::ChatDecide {
                    action: id,
                    approve,
                    by,
                    detail,
                },
            )
        }
        Request::ChatAttachment { id, attachment } => {
            match reader.chat_attachment(&id, &attachment) {
                Ok((a, bytes)) => ok(json!({
                    "media_type": a.media_type,
                    "bytes": base64::engine::general_purpose::STANDARD.encode(bytes),
                })),
                Err(e) => refused(&e).answer(),
            }
        }
        Request::BackfillStart { provider, since } => {
            drop(reader);
            match backfill::new_job(&provider, since.as_deref()) {
                Ok(new) => backfill_then_view(d, invocation, UserWrite::BackfillStart(new)),
                Err(r) => browser_refusal(r),
            }
        }
        Request::BackfillJobs => match reader.backfill_jobs() {
            Ok(jobs) => ok(json!({
                "jobs": jobs.iter().map(|job| backfill::view(job, now)).collect::<Vec<_>>(),
            })),
            Err(e) => store_error(&e),
        },
        Request::BackfillCancel { id } => {
            drop(reader);
            backfill_then_view(d, invocation, UserWrite::BackfillCancel { job: id })
        }
        Request::BackfillClaim { by } => {
            drop(reader);
            backfill_then_view(d, invocation, UserWrite::BackfillClaim { by })
        }
        Request::BackfillReport { id, by, report } => {
            drop(reader);
            backfill_then_view(
                d,
                invocation,
                UserWrite::BackfillReport {
                    job: id,
                    by,
                    report,
                },
            )
        }
        Request::Conversations { provider, ids } => {
            let sessions: Vec<_> = ids
                .iter()
                .map(|id| polylogue_sources::browser_capture::session_for(&provider, id))
                .collect();
            let known: Vec<_> = sessions.iter().flatten().cloned().collect();
            let mut found = match reader.captured_sessions(&known) {
                Ok(found) => found.into_iter(),
                Err(e) => return store_error(&e),
            };
            let conversations: Vec<_> = ids
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
            ok(json!({ "provider": provider, "conversations": conversations }))
        }
        Request::StoredFiles {
            provider,
            conversation,
        } => {
            let Some(session) =
                polylogue_sources::browser_capture::session_for(&provider, &conversation)
            else {
                return ok(json!({ "files": [] }));
            };
            match reader.stored_files(&session) {
                Ok(files) => match reader.downloading_files(&session) {
                    Ok(downloading) => ok(json!({ "files": files, "downloading": downloading })),
                    Err(e) => store_error(&e),
                },
                Err(e) => store_error(&e),
            }
        }
        other => error("bad_request", format!("{other:?} is not a browser request")),
    }
}
