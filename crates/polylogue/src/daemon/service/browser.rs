//! What the browser extension and the operator ask about web chats: chat
//! actions (the operator queues, the extension claims and reports), import_job
//! jobs (the operator starts, the extension claims and reports), and what
//! the archive holds of captured conversations. Every change goes through
//! the one writer ([`polylogue_store::chat_actions`],
//! [`polylogue_store::import_jobs`]).

use super::{Daemon, browser_refusal, store_error, writer};
use crate::browser::{actions, import_job};
use crate::cli::print;
use crate::daemon::Failure;
use crate::daemon::handler::{Call, Handler};
use crate::protocol::browser as protocol;
use base64::Engine as _;
use command_surface::{Answer, TextStyle};
use polylogue_model::user::{UserOutcome, UserWrite};
use polylogue_reading::operation::Invocation;
use polylogue_store::{Reader, StoreError, user_writes};

fn refused(e: &StoreError) -> Failure {
    match e {
        StoreError::ChatAction(code) | StoreError::ImportJob(code) => {
            let Some(code) = polylogue_model::Code::from_name(code) else {
                return crate::protocol::protocol_mismatch(
                    "browser action",
                    format!("unknown failure code {code:?}"),
                )
                .into();
            };
            Failure::new(code, code.as_str())
        }
        other => Failure::store(other),
    }
}

fn enabled(invocation: &Invocation, reader: &Reader<'_>) -> Result<bool, Failure> {
    match reader.setting_value("browser.actions", &invocation.machine) {
        Ok(Some(polylogue_model::settings::Value::Boolean(value))) => Ok(value),
        Ok(_) => Err(Failure::invalid(
            "browser.actions has no typed boolean value",
        )),
        Err(error) => Err(Failure::store(&error)),
    }
}

/// Writes `write`; `None` when it found nothing to claim. Chat actions and
/// import jobs carry their own retry keys, so each write gets a new
/// request ID.
fn apply(
    d: &Daemon,
    invocation: &Invocation,
    write: UserWrite,
) -> Result<Option<UserOutcome>, Failure> {
    let writes = writer(d)?;
    invocation.check_parent_lifetime()?;
    let request = invocation.request(user_writes::new_request_id(), write);
    let result = writes.user(polylogue_store::writer::UserSubmission {
        request,
        scope: None,
        preparation: None,
        parent: invocation.parent_lifetime(),
    });
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
        return crate::protocol::written(protocol::ChatAnswer::Outcome(outcome), true);
    }
    let reader = match d.reads.reader() {
        Ok(reader) => reader,
        Err(e) => return store_error(&e),
    };
    match reader.chat_action(&outcome.id) {
        Ok(Some(action)) => match actions::view(&action, invocation.at.unix_ms()) {
            Ok(action) => Answer::ok(protocol::ChatAnswer::Current(protocol::ChatCurrent {
                enabled: None,
                action: Some(Box::new(action)),
            })),
            Err(error) => unreadable_request(&error).answer(),
        },
        Ok(None) => browser_refusal(actions::Refused(polylogue_model::Code::UnknownChatAction)),
        Err(e) => store_error(&e),
    }
}

fn import_job_then_view(
    d: &Daemon,
    invocation: &Invocation,
    write: UserWrite,
) -> Answer<protocol::ImportJobAnswer> {
    let outcome = match apply(d, invocation, write) {
        Ok(Some(outcome)) => outcome,
        Ok(None) => {
            return Answer::ok(protocol::ImportJobAnswer::Current(
                protocol::ImportJobCurrent { job: None },
            ));
        }
        Err(failure) => return failure.answer(),
    };
    if outcome.pending {
        return crate::protocol::written(protocol::ImportJobAnswer::Outcome(outcome), true);
    }
    let reader = match d.reads.reader() {
        Ok(reader) => reader,
        Err(e) => return store_error(&e),
    };
    match reader.import_job(&outcome.id) {
        Ok(Some(job)) => Answer::ok(protocol::ImportJobAnswer::Current(
            protocol::ImportJobCurrent {
                job: Some(Box::new(import_job::view(&job, invocation.at.unix_ms()))),
            },
        )),
        Ok(None) => browser_refusal(actions::Refused(polylogue_model::Code::UnknownImportJob)),
        Err(e) => store_error(&e),
    }
}

/// A stored chat action whose request is not a [`actions::ChatRequest`].
fn unreadable_request(error: &serde_json::Error) -> Failure {
    Failure::new(
        polylogue_model::Code::Internal,
        format!("a stored chat action request is unreadable: {error}"),
    )
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
            .prepare_blob(&file.bytes)
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
            const PRINT: fn(&Self::Answer, TextStyle) -> String = $print;
            fn run($args: Self::Arguments, $call: &Call<'_>) -> Result<Answer<Self::Answer>, Failure> {
                Ok($body)
            }
        }
    };
}

browser!(
    ChatSend,
    protocol::ChatSend,
    protocol::ChatAnswer,
    print::json,
    |a, call| {
        let (d, invocation) = (call.daemon, call.invocation);
        let reader = call.reader()?;
        match enabled(invocation, &reader) {
            Ok(true) => {}
            Ok(false) => {
                return Ok(browser_refusal(actions::Refused(
                    polylogue_model::Code::BrowserActionsOff,
                )));
            }
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
        for file in &a.files {
            if let Err(r) = actions::check_file_name(&file.name) {
                return Ok(browser_refusal(r));
            }
        }
        if let Err(r) = actions::new_action(call.request_id, &action, vec![]) {
            return Ok(browser_refusal(r));
        }
        let stored = match store_files(d, &a.files)? {
            PreparedFiles::Complete(files) => files,
            PreparedFiles::Pending(pending) => {
                return Ok(crate::protocol::written(
                    protocol::ChatAnswer::Capture(pending),
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
    /// The extension instance `by` takes the oldest queued chat action.
    ChatClaim,
    protocol::ChatClaim,
    protocol::ChatAnswer,
    print::json,
    |a, call| {
        let reader = call.reader()?;
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
    protocol::ChatAttachmentBytes,
    print::json,
    |a, call| {
        let reader = call.reader()?;
        match reader.chat_attachment(&a.id, &a.attachment) {
            Ok((attachment, bytes)) => Answer::ok(protocol::ChatAttachmentBytes {
                media_type: attachment.media_type,
                bytes: base64::engine::general_purpose::STANDARD.encode(bytes),
            }),
            Err(e) => refused(&e).answer(),
        }
    }
);

browser!(
    ImportJobStart,
    protocol::ImportJobStart,
    protocol::ImportJobAnswer,
    print::import_job_answer,
    |a, call| {
        match import_job::new_job(&a.provider, a.since.as_deref()) {
            Ok(new) => {
                import_job_then_view(call.daemon, call.invocation, UserWrite::ImportJobStart(new))
            }
            Err(r) => browser_refusal(r),
        }
    }
);

browser!(
    ImportJobCancel,
    protocol::ImportJobCancel,
    protocol::ImportJobAnswer,
    print::import_job_answer,
    |a, call| {
        import_job_then_view(
            call.daemon,
            call.invocation,
            UserWrite::ImportJobCancel { job: a.id },
        )
    }
);

browser!(
    ImportJobClaim,
    protocol::ImportJobClaim,
    protocol::ImportJobAnswer,
    print::json,
    |a, call| {
        import_job_then_view(
            call.daemon,
            call.invocation,
            UserWrite::ImportJobClaim { by: a.by },
        )
    }
);

browser!(
    ImportJobReport,
    protocol::ImportJobReport,
    protocol::ImportJobAnswer,
    print::json,
    |a, call| {
        import_job_then_view(
            call.daemon,
            call.invocation,
            UserWrite::ImportJobReport {
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
    protocol::ConversationsAnswer,
    print::json,
    |a, call| {
        let reader = call.reader()?;
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
                protocol::ConversationState {
                    id: id.clone(),
                    session: session.as_ref().map(ToString::to_string),
                    messages: held.as_ref().map(|h| h.messages),
                    downloading: held.as_ref().map(|h| h.downloading),
                    last_captured_at: held.and_then(|h| h.last_captured_at),
                }
            })
            .collect();
        Answer::ok(protocol::ConversationsAnswer {
            provider: a.provider,
            conversations,
        })
    }
);

browser!(
    StoredFiles,
    protocol::StoredFiles,
    protocol::StoredFilesAnswer,
    print::json,
    |a, call| {
        let Some(session) =
            polylogue_sources::browser_capture::session_for(&a.provider, &a.conversation)
        else {
            return Ok(Answer::ok(protocol::StoredFilesAnswer {
                files: Vec::new(),
                downloading: None,
            }));
        };
        let reader = call.reader()?;
        match reader.stored_files(&session) {
            Ok(files) => match reader.downloading_files(&session) {
                Ok(downloading) => Answer::ok(protocol::StoredFilesAnswer {
                    files: files
                        .into_iter()
                        .map(|file| protocol::StoredFile {
                            id: file.id,
                            blob: file.blob,
                            size: file.size,
                        })
                        .collect(),
                    downloading: Some(downloading),
                }),
                Err(e) => store_error(&e),
            },
            Err(e) => store_error(&e),
        }
    }
);
