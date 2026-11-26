//! What the browser extension and the operator ask about web chats: chat
//! actions (`send`, `start --in` and `answer` make them, the extension
//! claims and reports), import jobs (`import PROVIDER` starts and stops
//! them, the extension claims and reports), and what the archive holds of
//! captured conversations. Every change goes through the one writer
//! ([`polylogue_store::chat_actions`], [`polylogue_store::import_jobs`]).

use super::{Daemon, browser_refusal, store_error, writer};
use crate::browser::{actions, import_job};
use crate::cli::print;
use crate::commands::ToolAccess;
use crate::daemon::Failure;
use crate::daemon::handler::{Call, Operation, Policy};
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

pub(super) fn enabled(invocation: &Invocation, reader: &Reader<'_>) -> Result<bool, Failure> {
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
    let result = writes
        .archive()
        .user(polylogue_store::writer::UserSubmission {
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
pub(super) fn chat_then_view(
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
/// `import PROVIDER`: starts the provider's import job, or answers the one
/// that has not ended; `cancel` stops that one; `wait` returns once the job
/// finishes, fails, is cancelled or paused, or the time passes.
pub(super) fn import_provider(
    call: &Call<'_>,
    arguments: &crate::protocol::import::Arguments,
    provider: &str,
) -> Result<Answer<protocol::ImportJobAnswer>, Failure> {
    let (d, invocation) = (call.daemon, call.invocation);
    if arguments.origin.is_some() || arguments.withdraw_missing || arguments.dry_run {
        return Err(Failure::bad_request(
            "--origin, --withdraw-missing and --dry-run read files; a provider import takes --since, --cancel and --wait",
        ));
    }
    if !arguments.cancel {
        let answer = match import_job::new_job(provider, arguments.since.as_deref()) {
            Ok(new) => import_job_then_view(d, invocation, UserWrite::ImportJobStart(new)),
            Err(refusal) => return Ok(browser_refusal(refusal)),
        };
        let started = match &answer.outcome {
            command_surface::Outcome::Ok(protocol::ImportJobAnswer::Current(current)) => {
                current.job.as_ref().map(|job| job.id.clone())
            }
            _ => None,
        };
        return match (&arguments.wait, started) {
            (Some(wait), Some(job)) => wait_for_job(call, arguments, &job, wait),
            _ => Ok(answer),
        };
    }
    let now = invocation.at.unix_ms();
    let running = call
        .reader()?
        .import_jobs()?
        .into_iter()
        .rev()
        .find(|job| job.provider == provider && !job.status(now).0.ended());
    let Some(job) = running else {
        return Err(Failure::not_found(format!(
            "no import of {provider} is running; start one with `polylogue import {provider}`"
        )));
    };
    Ok(import_job_then_view(
        d,
        invocation,
        UserWrite::ImportJobCancel { job: job.id },
    ))
}

/// Waits for an import job to stop running. A wait that runs out of time
/// answers `empty` with the job as it stands and the same request as its
/// next call.
fn wait_for_job(
    call: &Call<'_>,
    arguments: &crate::protocol::import::Arguments,
    id: &str,
    wait: &str,
) -> Result<Answer<protocol::ImportJobAnswer>, Failure> {
    use polylogue_store::read::limit::ReadLimit;
    use std::time::Instant;
    let mut duration = super::line::listing_wait(wait, call.caller.channel())?;
    let limit = ReadLimit::current();
    if let Some(left) = limit.as_ref().and_then(ReadLimit::remaining) {
        duration = duration.min(left.saturating_sub(super::follow::REPLY_MARGIN));
    }
    let until = Instant::now() + duration;
    let commits = call
        .daemon
        .writes
        .as_ref()
        .map(|writer| writer.completion_commits());
    let _wake = match (&commits, &limit) {
        (Some(commits), Some(limit)) => {
            let wake = commits.wake_handle();
            Some(limit.cancellation.on_cancel(move || {
                let _ = wake.try_send(());
            }))
        }
        _ => None,
    };
    loop {
        super::accepting(call.daemon)?;
        if let Some(limit) = &limit {
            limit.check()?;
        }
        let now = call.daemon.clock.now_ms();
        let job = call.reader()?.import_job(id)?.ok_or_else(|| {
            Failure::new(
                polylogue_model::Code::UnknownImportJob,
                format!("no import job {id}"),
            )
        })?;
        let status = job.status(now).0;
        let current = protocol::ImportJobAnswer::Current(protocol::ImportJobCurrent {
            job: Some(Box::new(import_job::view(&job, now))),
        });
        if status.ended() || status == polylogue_model::user::ImportJobStatus::Paused {
            return Ok(Answer::ok(current));
        }
        let left = until.saturating_duration_since(Instant::now());
        if left.is_zero() {
            let again = match call.context()?.accepted_request() {
                Some(request) => request.clone(),
                None => command_surface::RunnableRequest {
                    operation: "import".into(),
                    arguments: serde_json::to_value(arguments).map_err(Failure::internal)?,
                    request_id: call.request_id.into(),
                    deadline: Default::default(),
                    budget_tokens: call.budget,
                    cursor: None,
                    fresh: false,
                    format: Default::default(),
                },
            };
            return Ok(Answer::empty(current).with_next("wait again", again));
        }
        ReadLimit::set_stage("import wait");
        match &commits {
            Some(commits) => {
                let _ = commits.recv_timeout(left);
            }
            None => std::thread::sleep(left),
        }
    }
}

pub(super) fn unreadable_request(error: &serde_json::Error) -> Failure {
    Failure::new(
        polylogue_model::Code::Internal,
        format!("a stored chat action request is unreadable: {error}"),
    )
}

/// Stores each file's bytes as a blob and makes its attachment row.
pub(super) fn store_files(
    d: &Daemon,
    files: &[actions::ChatFile],
) -> Result<Vec<polylogue_model::user::ChatAttachment>, Failure> {
    let writes = writer(d)?;
    let mut attachments = Vec::with_capacity(files.len());
    for file in files {
        let result = d
            .archive
            .prepare_blob(&file.bytes)
            .and_then(|(new_blob, prepared)| writes.store_blob(new_blob, prepared));
        attachments.push(actions::attachment(file, result.map_err(Failure::from)?));
    }
    Ok(attachments)
}

/// The browser handlers: each answers with the JSON view the extension reads.
macro_rules! browser {
    ($(#[$doc:meta])* $name:ident, $arguments:ty, $answer:ty, $print:expr, $policy:expr, |$args:ident, $call:ident| $body:expr) => {
        $(#[$doc])*
        pub(crate) struct $name;
        impl Operation for $name {
            type Arguments = $arguments;
            type Reply = $answer;
            const PRINT: fn(&Self::Reply, TextStyle) -> String = $print;
            fn policy(_: &Self::Arguments, _: polylogue_model::authorship::Channel) -> Result<Policy, Failure> {
                Ok($policy)
            }
            fn run($args: Self::Arguments, $call: &Call<'_>) -> Result<Answer<Self::Reply>, Failure> {
                Ok($body)
            }
        }
    };
}

browser!(
    /// The extension instance `by` takes the oldest queued chat action.
    ChatClaim,
    protocol::ChatClaim,
    protocol::ChatAnswer,
    print::json,
    Policy::write(ToolAccess::Write),
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
    Policy::write(ToolAccess::Write),
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

browser!(
    /// The popup's Allow or Deny: the operator's answer to a held turn,
    /// through the same path as `answer allow|deny`.
    HeldTurnAnswer,
    protocol::HeldTurnAnswer,
    protocol::ChatAnswer,
    print::json,
    Policy::write(ToolAccess::Write),
    |a, call| super::acts::web_chat::answer_action(call, &a.id, &a.value, a.reason)?
);

browser!(
    ChatAttachment,
    protocol::ChatAttachment,
    protocol::ChatAttachmentBytes,
    print::json,
    Policy::READ,
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
    ImportJobClaim,
    protocol::ImportJobClaim,
    protocol::ImportJobAnswer,
    print::json,
    Policy::write(ToolAccess::Write),
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
    Policy::write(ToolAccess::Write),
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
    Policy::READ,
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
    Policy::READ,
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
