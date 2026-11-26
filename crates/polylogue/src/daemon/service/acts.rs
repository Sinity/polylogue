//! Live-session acts: messages (one session, a role's running instance, or
//! every session a query selects when sent; a web chat's turns are in
//! [`web_chat`]), signal notices to sessions,
//! the mailbox claim a hook makes, answers to permission requests,
//! interrupts, and new hosted sessions (`start`, role instances,
//! `resume --in`). Each act is recorded as a delivery with its computed
//! author, and reaches its session only through the channel
//! [`polylogue_model::live::channel::channel`] chooses; a new session
//! starts on the protocol [`polylogue_model::live::channel::host_protocol`]
//! names.
use super::{Daemon, Failure, authored_preparation, error, writer};
use crate::commands::ToolAccess;
use crate::daemon::handler::{Call, Operation, Policy};
use crate::live_acts::{ResumeRun, probe};
use crate::protocol::acts::{
    ActOutcome, Answered, Claim, Claimed, ClaimedDelivery, Continued, Decided, Delivered,
    Interrupted, Permission, Sent, Start, StartReply, Started,
};
use command_surface::{Answer, TextStyle};
use polylogue_model::live::channel::{Act, Channel, channel};
use polylogue_model::user::{Delivery, UserOutcome, UserWrite};
use polylogue_model::{DeliveryEvent, DeliveryOutcome, DeliverySender, Reason, SessionId};
use polylogue_reading::operation::Invocation;
use serde_json::{Value, json};

#[cfg(test)]
mod tests;
pub(crate) mod web_chat;

/// `internal mail-claim`: a hook takes the messages and notices pending for
/// its session.
pub(crate) struct MailClaim;
impl Operation for MailClaim {
    type Arguments = Claim;
    type Reply = Claimed;
    const PRINT: fn(&Self::Reply, TextStyle) -> String = crate::cli::print::claimed;
    fn policy(_: &Claim, _: polylogue_model::authorship::Channel) -> Result<Policy, Failure> {
        Ok(Policy::write(ToolAccess::Write))
    }
    fn run(arguments: Claim, call: &Call<'_>) -> Result<Answer<Claimed>, Failure> {
        claim(call.daemon, &arguments.session)
    }
}

/// `internal permission`: the synchronous permission hook asks for a
/// decision given ahead.
pub(crate) struct StandingPermission;
impl Operation for StandingPermission {
    type Arguments = Permission;
    type Reply = Decided;
    const PRINT: fn(&Self::Reply, TextStyle) -> String = crate::cli::print::decided;
    fn policy(_: &Permission, _: polylogue_model::authorship::Channel) -> Result<Policy, Failure> {
        Ok(Policy::write(ToolAccess::Write))
    }
    fn run(a: Permission, call: &Call<'_>) -> Result<Answer<Decided>, Failure> {
        permission(call.daemon, &a.session, &a.tool, &a.input)
    }
}

/// `start`: a session Polylogue hosts, or a web chat's new conversation.
pub(crate) struct StartSession;
impl Operation for StartSession {
    type Arguments = Start;
    type Reply = StartReply;
    const PRINT: fn(&Self::Reply, TextStyle) -> String = crate::cli::print::start;
    fn policy(_: &Start, _: polylogue_model::authorship::Channel) -> Result<Policy, Failure> {
        Ok(Policy::write(ToolAccess::Act))
    }
    fn run(a: Start, call: &Call<'_>) -> Result<Answer<StartReply>, Failure> {
        if let Some(provider) = a
            .harness
            .as_deref()
            .and_then(polylogue_sources::origin::web_chat_name)
        {
            return Ok(web_chat::start(call, provider, a)?.map(StartReply::WebChat));
        }
        let pending = [
            (a.submit, "--submit"),
            (a.model.is_some(), "--model"),
            (a.effort.is_some(), "--effort"),
            (a.project.is_some(), "--project"),
            (!a.attach.is_empty(), "--attach"),
        ];
        if let Some((_, option)) = pending.into_iter().find(|(used, _)| *used) {
            return Ok(error(
                polylogue_model::Code::NotBuilt,
                format!("start {option} on a harness session is not built yet"),
            ));
        }
        if a.harness.is_none() && a.role.is_none() {
            return Err(Reason::StartNeedsHarnessOrRole {}.into());
        }
        let cwd = call.directory(a.cwd.as_deref())?;
        Ok(start(
            call.daemon,
            call.invocation,
            call.request_id,
            a.harness.as_deref(),
            a.role.as_deref(),
            &cwd,
            &a.text,
        )?
        .map(StartReply::Hosted))
    }
}

fn now(d: &Daemon) -> i64 {
    d.config.requests.now().unix_ms()
}

/// Writes one authored record and returns what the writer reports.
pub(super) fn author(
    d: &Daemon,
    invocation: &Invocation,
    request_id: String,
    write: UserWrite,
) -> Result<UserOutcome, Failure> {
    let writes = writer(d)?;
    invocation.check_parent_lifetime()?;
    let request = invocation.request(request_id, write);
    if matches!(
        &request.write,
        UserWrite::Deliver(_) | UserWrite::DeliveredAdvice(_) | UserWrite::DeliveryEvent { .. }
    ) {
        Ok(writes
            .archive()
            .accept_delivery(request, None)?
            .recv()
            .map_err(|_| polylogue_store::StoreError::WriterGone)??
            .outcome
            .outcome)
    } else {
        Ok(writes
            .archive()
            .user(polylogue_store::writer::UserSubmission {
                request,
                scope: None,
                preparation: Some(authored_preparation()),
                parent: None,
            })?)
    }
}

fn outcome_event(
    d: &Daemon,
    invocation: &Invocation,
    delivery: &str,
    what: DeliveryOutcome,
    detail: String,
) -> Result<(), Failure> {
    let request = invocation.request(
        polylogue_store::user_writes::new_request_id(),
        UserWrite::DeliveryEvent {
            delivery: delivery.to_owned(),
            what,
            detail: Some(detail),
        },
    );
    let written = writer(d)?.archive().accept_delivery(request, None)?;
    d.acts.watch_delivery(delivery, written);
    Ok(())
}

/// Who an address names when it is sent.
enum Recipients {
    /// Sessions: one reference, or every session a quoted query selects
    /// then (a broadcast fixed when sent: sessions that start matching later
    /// receive nothing).
    Sessions(Vec<SessionId>),
    /// A role with no running instance: a message starts one; a notice does
    /// not.
    NewInstance(Role),
}

/// A role as its current revision defines it.
#[derive(Debug, Clone)]
pub(crate) struct Role {
    pub name: String,
    /// The exact revision, which the sessions it starts record.
    pub annotation: String,
    pub instructions: String,
    pub harness: String,
}

/// The current revision of the role `name`.
pub(crate) fn role(d: &Daemon, invocation: &Invocation, name: &str) -> Result<Role, Failure> {
    use polylogue_model::annotation::values;
    let operation = d.resolve_operation_context(invocation.clone())?;
    let reader = d.reads.reader()?;
    let conditions = format!(
        "kind:role key:{}",
        crate::daemon::query::annotations::quoted(name)
    );
    let found = crate::daemon::query::annotations::list(&reader, &conditions, &operation)?;
    let Some(annotation) = found.into_iter().next() else {
        return Err(Failure::not_found(format!("no role {name}")));
    };
    let Ok(values::Value::Role(role)) = annotation.fact.validate() else {
        return Err(Failure::bad_value(format!(
            "the role {name} has no instructions and harness"
        )));
    };
    Ok(Role {
        name: name.to_owned(),
        annotation: annotation.fact.id.clone(),
        instructions: role.fields.instructions.clone(),
        harness: role.fields.harness.as_str().to_owned(),
    })
}

/// The reference every session a role starts records.
fn role_reference(name: &str) -> String {
    format!("role:{name}")
}

/// The newest running instance of a role: the newest session Polylogue
/// started for it that it still hosts or whose live state is running.
pub(crate) fn role_instance(
    d: &Daemon,
    name: &str,
    now: i64,
) -> Result<Option<SessionId>, Failure> {
    let reader = d.reads.reader()?;
    for (session, _, _) in reader.sessions_started_with(&role_reference(name))? {
        let Ok(id) = SessionId::parse(&session) else {
            continue;
        };
        // A session just started has no live events yet; its open
        // connection says it runs.
        let running = d.acts.is_hosted(&id)
            || reader
                .live_state(&session, now)?
                .is_some_and(|s| s.view.state.running());
        if running {
            return Ok(Some(id));
        }
    }
    Ok(None)
}

/// Where a role's new instance runs when an address starts one: where its
/// latest instance ran.
pub(crate) fn role_cwd(d: &Daemon, name: &str) -> Result<Option<std::path::PathBuf>, Failure> {
    Ok(d.reads
        .reader()?
        .sessions_started_with(&role_reference(name))?
        .into_iter()
        .find_map(|(_, cwd, _)| cwd)
        .map(std::path::PathBuf::from))
}

/// Who an address names now: a session reference, `@[role:NAME]` (its
/// newest running instance), or `@[QUERY]`.
fn recipients(
    d: &Daemon,
    invocation: &Invocation,
    target: &str,
    now: i64,
) -> Result<Recipients, Failure> {
    let address = target
        .trim()
        .strip_prefix("@[")
        .and_then(|rest| rest.strip_suffix(']'))
        .unwrap_or(target.trim());
    if let Some(name) = address.strip_prefix("role:") {
        let role = role(d, invocation, name)?;
        return Ok(match role_instance(d, name, now)? {
            Some(session) => Recipients::Sessions(vec![session]),
            None => Recipients::NewInstance(role),
        });
    }
    if let Ok(session) = SessionId::parse(address) {
        return Ok(Recipients::Sessions(vec![session]));
    }
    let operation = d.resolve_operation_context(invocation.clone())?;
    let reader = d.reads.reader()?;
    Ok(Recipients::Sessions(
        super::follow::selected(&reader, &operation, address, now)?
            .iter()
            .map(|id| SessionId::parse(id).map_err(Failure::internal))
            .collect::<Result<_, _>>()?,
    ))
}

/// A session's live state, live facts and directory for an act: from the
/// archive, or, for a session Polylogue just started and still hosts, no
/// live events yet and state `unknown`.
pub(crate) fn session_state(
    d: &Daemon,
    session: &SessionId,
    now: i64,
) -> Result<
    Option<(
        polylogue_model::live::LiveState,
        polylogue_model::live::LiveFacts,
        Option<String>,
    )>,
    Failure,
> {
    match d.reads.reader()?.live_state(&session.to_string(), now)? {
        Some(s) => Ok(Some((s.view.state, s.facts, s.cwd))),
        None if d.acts.is_hosted(session) => Ok(Some((
            polylogue_model::live::LiveState::Unknown,
            polylogue_model::live::LiveFacts::default(),
            None,
        ))),
        None => Ok(None),
    }
}

/// `resume` of a running session is refused `session_running`, offering the
/// lines that reach the running session instead.
pub(crate) fn refuse_running_resume(d: &Daemon, session: &SessionId) -> Result<(), Failure> {
    let Some((state, facts, _)) = session_state(d, session, now(d))? else {
        return Ok(());
    };
    match channel(Act::Resume, state, &probe(&d.acts, session, &facts)) {
        Err(refusal) if refusal.reason == "session_running" => Err(Failure::refused(format!(
            "session_running: {}; read it with `polylogue {session} read --follow` or send to it with `polylogue {session} send 'TEXT'`",
            refusal.detail
        ))),
        _ => Ok(()),
    }
}

/// Sends a message or notice over a hosted connection: `delivered` when
/// written, recorded later when it waits for the running prompt, and
/// `not_sent` when the connection fails. The delivery stays handed out, so
/// no hook takes it as well.
fn protocol_send(
    d: &Daemon,
    invocation: &Invocation,
    session: &SessionId,
    delivery: &str,
    pack: &str,
) -> Result<Delivered, Failure> {
    let row = |outcome, reason: Option<String>, note: Option<&str>| Delivered {
        session: session.clone(),
        delivery: Some(delivery.to_owned()),
        channel: Some(Channel::Protocol),
        outcome,
        reason,
        detail: None,
        note: note.map(str::to_owned),
        started: None,
    };
    Ok(match d.acts.host_send(session, pack, Some(delivery)) {
        Ok(true) => {
            outcome_event(
                d,
                invocation,
                delivery,
                DeliveryOutcome::Delivered,
                "sent over the hosted connection".into(),
            )?;
            row(ActOutcome::Delivered, None, None)
        }
        Ok(false) => row(
            ActOutcome::Prepared,
            None,
            Some("waits for the running prompt; recorded when it is sent"),
        ),
        Err(reason) => {
            outcome_event(
                d,
                invocation,
                delivery,
                DeliveryOutcome::NotSent,
                format!("failed({reason})"),
            )?;
            row(ActOutcome::NotSent, Some(reason), None)
        }
    })
}

/// Records what happened to a message that waited for a hosted session's
/// running prompt.
fn record_sent(d: &Daemon, delivery: &str, sent: Result<(), String>) {
    let invocation = d.invocation(&polylogue_reading::operation::Context::worker_authorship(
        "hosted",
    ));
    let (what, detail) = match sent {
        Ok(()) => (
            DeliveryOutcome::Delivered,
            "sent over the hosted connection after the running prompt".to_owned(),
        ),
        Err(reason) => (DeliveryOutcome::NotSent, format!("failed({reason})")),
    };
    if let Err(error) = outcome_event(d, &invocation, delivery, what, detail) {
        d.acts.delivery_write_failed(delivery, &error.message);
    }
}

/// The message pack: the text, what it is about, how to answer, and the
/// delivery reference.
fn message_text(
    delivery: &str,
    message: &str,
    author: &str,
    text: &str,
    about: Option<&str>,
) -> String {
    let mut pack =
        format!("Message from {author} (delivery:{delivery}, annotation:{message}):\n{text}\n");
    if let Some(about) = about {
        pack.push_str(&format!("About: {about}\n"));
    }
    pack.push_str(&format!(
        "Answer with Polylogue's MCP `rows` operation (tool `polylogue_write`): `selection` and `replies_to` both name `annotation:{message}@{message}`; use a `note` step with `as: finding` and your answer as `text`. Reach the sender with a `send` step (tool `polylogue_act`)."
    ));
    pack
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn send(
    d: &Daemon,
    invocation: &Invocation,
    request_id: &str,
    target: &str,
    text: &str,
    live: bool,
    about: Option<String>,
) -> Result<Answer<Sent>, Failure> {
    if text.trim().is_empty() {
        return Err(Reason::MessageNeedsText {}.into());
    }
    let now = now(d);
    let sessions = match recipients(d, invocation, target, now)? {
        Recipients::Sessions(sessions) => sessions,
        Recipients::NewInstance(role) => {
            return message_new_instance(d, invocation, request_id, &role, text, about.as_deref());
        }
    };
    if sessions.is_empty() {
        return Ok(error(
            polylogue_model::Code::NotFound,
            "the address names no session",
        ));
    }
    let subject = about
        .as_deref()
        .map(polylogue_model::Reference::parse)
        .transpose()
        .map_err(Failure::bad_value)?;
    let annotation = polylogue_model::annotation::NewAnnotation {
        kind: polylogue_model::annotation::AnnotationKind::Message,
        key: None,
        value: Some(
            json!({"text": text, "subject": subject.as_ref().map(ToString::to_string), "reply_to": null}),
        ),
        valid_until: None,
        links: sessions
            .iter()
            .map(|s| polylogue_model::annotation::NewAnnotationLink {
                role: polylogue_model::annotation::LinkRole::About,
                reference: polylogue_model::Reference::Session(s.clone()),
            })
            .collect(),
    };
    let message = author(
        d,
        invocation,
        format!("{request_id}-message"),
        UserWrite::Assert {
            annotations: vec![annotation],
        },
    )?
    .id;
    let author_name = invocation.authorship.author.to_string();
    let mut sent = Vec::new();
    for (n, session) in sessions.iter().enumerate() {
        let delivery = format!("{request_id}-{n}");
        let (view_state, facts, cwd) = match session_state(d, session, now)? {
            Some(state) => state,
            None => {
                sent.push(Delivered {
                    session: session.clone(),
                    delivery: None,
                    channel: None,
                    outcome: ActOutcome::NotSent,
                    reason: Some("session_missing".into()),
                    detail: None,
                    note: None,
                    started: None,
                });
                continue;
            }
        };
        let pack = message_text(&delivery, &message, &author_name, text, about.as_deref());
        // Held from before its row exists, so nothing closes or claims it
        // until its channel is decided.
        {
            let _claims = d.acts.claim_lock();
            d.acts.take(vec![(delivery.clone(), ())]);
        }
        author(
            d,
            invocation,
            delivery.clone(),
            UserWrite::Deliver(Delivery {
                recipient_kind: "session".into(),
                recipient_id: Some(session.to_string()),
                sender: DeliverySender::Send,
                event: DeliveryEvent::Message,
                cwd,
                text: pack.clone(),
                refs: vec![format!("annotation:{message}")],
                left_out: Vec::new(),
            }),
        )?;
        let chosen = channel(
            Act::Send { live },
            view_state,
            &probe(&d.acts, session, &facts),
        );
        let row = |channel, outcome, note: Option<&str>| Delivered {
            session: session.clone(),
            delivery: Some(delivery.clone()),
            channel,
            outcome,
            reason: None,
            detail: None,
            note: note.map(str::to_owned),
            started: None,
        };
        let row = match chosen {
            Ok(Channel::Hook) => {
                d.acts
                    .hand_to_hook(&delivery, session)
                    .map_err(Failure::io)?;
                row(
                    Some(Channel::Hook),
                    ActOutcome::Prepared,
                    Some("reaches the session at its next tool result or prompt"),
                )
            }
            Ok(Channel::Resume) => {
                let argv = crate::live_acts::headless_resume(session, &pack);
                let cwd = d.reads.reader()?.resume_directory(session)?;
                let (reason, detail) = match (argv, cwd) {
                    (None, _) => ("unsupported", None),
                    (_, None) => ("directory_unknown", None),
                    (Some(_), Some(dir)) if !std::path::Path::new(&dir).is_dir() => {
                        ("directory_missing", Some(dir))
                    }
                    (Some(argv), Some(cwd)) => {
                        let run = ResumeRun {
                            session: session.clone(),
                            delivery: delivery.clone(),
                            argv,
                            cwd: cwd.into(),
                        };
                        match d.acts.resume(run) {
                            Ok(None) => {
                                sent.push(row(Some(Channel::Resume), ActOutcome::Queued, None));
                                continue;
                            }
                            Ok(Some(started)) => match started.recv() {
                                Ok(Ok(())) => {
                                    sent.push(row(
                                        Some(Channel::Resume),
                                        ActOutcome::Started,
                                        None,
                                    ));
                                    continue;
                                }
                                Ok(Err(problem)) => ("failed", Some(problem)),
                                Err(error) => (
                                    "failed",
                                    Some(format!("the resume worker stopped: {error}")),
                                ),
                            },
                            Err(problem) => ("failed", Some(problem)),
                        }
                    }
                };
                let recorded = detail
                    .as_ref()
                    .map_or_else(|| reason.to_owned(), |detail| format!("{reason}({detail})"));
                outcome_event(d, invocation, &delivery, DeliveryOutcome::NotSent, recorded)?;
                Delivered {
                    reason: Some(reason.into()),
                    detail,
                    ..row(Some(Channel::Resume), ActOutcome::NotSent, None)
                }
            }
            Ok(Channel::Protocol) => protocol_send(d, invocation, session, &delivery, &pack)?,
            Err(refusal) => {
                outcome_event(
                    d,
                    invocation,
                    &delivery,
                    DeliveryOutcome::NotSent,
                    refusal.reason.to_owned(),
                )?;
                Delivered {
                    reason: Some(refusal.reason.to_owned()),
                    detail: Some(refusal.detail.to_string()),
                    ..row(None, ActOutcome::NotSent, None)
                }
            }
        };
        sent.push(row);
    }
    Ok(Answer::ok(Sent {
        message: Some(message),
        deliveries: sent,
        undo: "none: a sent message cannot be taken back".into(),
    }))
}

/// A hook's mailbox claim: every message and notice pending for the session,
/// oldest first, each handed out once. Claims run one at a time, so
/// concurrent hooks of one session never take the same delivery. The hook
/// records each `delivered` once it has written it to the harness; a claim
/// whose hook never confirms stays without an outcome and is not handed
/// out again.
fn claim(d: &Daemon, session: &SessionId) -> Result<Answer<Claimed>, Failure> {
    let _claims = d.acts.claim_lock();
    // The flag is lowered before the pending list is read: a delivery handed
    // to the hook after that read raises it again, whereas lowering after
    // the read would clear a flag raised in between and strand that delivery.
    crate::mailbox::lower(d.acts.mail(), session).map_err(Failure::io)?;
    let pending = (|| -> Result<_, Failure> {
        Ok(d.reads
            .reader()?
            .pending_hook_deliveries(&session.to_string())?)
    })();
    let pending = match pending {
        Ok(pending) => pending,
        Err(failure) => {
            // Nothing was claimed, so the flag goes back up for the next hook.
            crate::mailbox::raise(d.acts.mail(), session).map_err(Failure::io)?;
            return Err(failure);
        }
    };
    let deliveries = d
        .acts
        .take(pending)
        .into_iter()
        .map(|(id, text)| ClaimedDelivery { id, text })
        .collect();
    Ok(Answer::ok(Claimed { deliveries }))
}

/// Messages and notices still pending for sessions that stopped running
/// reach no hook: each is recorded `not_sent:session_<state>` (`ended`,
/// `stale`, or `missing` when the session has no live state) and the flag
/// lowered. Run by the live worker after each change, and at start for
/// every session with pending items (a resumed run queued when the daemon
/// stopped is among them).
/// `everyone` reads every session with pending items (at start); otherwise
/// only sessions whose flag is up are looked at. Items already handed out
/// are left to their hook or run.
pub(crate) fn finish_ended(d: &Daemon, now: i64, everyone: bool) {
    let invocation = d.invocation(&polylogue_reading::operation::Context::worker_authorship(
        "mailbox",
    ));
    let reader = match d.reads.reader() {
        Ok(reader) => reader,
        Err(error) => {
            d.acts.mailbox_failed("open the index", error);
            return;
        }
    };
    // A hosted session that ended needs no connection or record.
    for session in d.acts.hosted_sessions() {
        let ended = match reader.live_state(&session.to_string(), now) {
            Ok(state) => {
                state.is_some_and(|s| s.view.state == polylogue_model::live::LiveState::Ended)
            }
            Err(error) => {
                d.acts
                    .mailbox_failed(&format!("read live state for {session}"), error);
                continue;
            }
        };
        if ended {
            d.acts.release_host(&session);
        }
    }
    let sessions: Vec<String> = if everyone {
        match reader.sessions_with_pending_hook_deliveries() {
            Ok(sessions) => sessions,
            Err(error) => {
                d.acts.mailbox_failed("read pending sessions", error);
                return;
            }
        }
    } else {
        crate::mailbox::flagged(d.acts.mail())
            .unwrap_or_else(|error| {
                d.acts.mailbox_failed("read flags", error);
                Vec::new()
            })
            .iter()
            .map(ToString::to_string)
            .collect()
    };
    let mut stopped = Vec::new();
    for session in sessions {
        let id = match SessionId::parse(&session) {
            Ok(id) => id,
            Err(error) => {
                d.acts
                    .mailbox_failed(&format!("read session {session}"), error);
                continue;
            }
        };
        let state = match reader.live_state(&session, now) {
            Ok(state) => state.map(|s| s.view.state),
            Err(error) => {
                d.acts
                    .mailbox_failed(&format!("read live state for {session}"), error);
                continue;
            }
        };
        let pending = match reader.pending_hook_deliveries(&session) {
            Ok(pending) => pending,
            Err(error) => {
                d.acts
                    .mailbox_failed(&format!("read deliveries for {session}"), error);
                continue;
            }
        };
        let pending: Vec<(String, ())> = pending
            .into_iter()
            .filter(|(item, _)| !d.acts.handed_out(item))
            .map(|(item, _)| (item, ()))
            .collect();
        match state {
            Some(state) if state.running() => {
                if !pending.is_empty()
                    && let Err(error) = crate::mailbox::raise(d.acts.mail(), &id)
                {
                    d.acts
                        .mailbox_failed(&format!("raise the flag for {id}"), error);
                }
            }
            _ => stopped.push((id, state, pending)),
        }
    }
    drop(reader);
    for (session, state, pending) in stopped {
        let taken = {
            let _claims = d.acts.claim_lock();
            d.acts.take(pending)
        };
        let reason = format!("session_{}", state.map_or("missing", |s| s.as_str()));
        for (id, ()) in taken {
            if let Err(error) = outcome_event(
                d,
                &invocation,
                &id,
                DeliveryOutcome::NotSent,
                reason.clone(),
            ) {
                d.acts.delivery_write_failed(&id, &error.message);
            }
        }
        if let Err(error) = crate::mailbox::lower(d.acts.mail(), &session) {
            d.acts
                .mailbox_failed(&format!("lower the flag for {session}"), error);
        }
    }
}

/// The pending request an answer is about, as `request:<session>#<at>`.
fn request_reference(session: &SessionId, at: i64) -> String {
    format!("request:{session}#{at}")
}

/// An answer to a session's pending permission request, or, with
/// `standing`, to the same tool and input whenever it asks next.
#[allow(clippy::too_many_arguments)]
pub(crate) fn answer(
    d: &Daemon,
    invocation: &Invocation,
    request_id: &str,
    session: &SessionId,
    approve: bool,
    text: Option<String>,
    standing: bool,
) -> Result<Answer<Answered>, Failure> {
    let now = now(d);
    let Some(state) = d.reads.reader()?.live_state(&session.to_string(), now)? else {
        return Ok(error(
            polylogue_model::Code::NotFound,
            format!("no session {session}"),
        ));
    };
    let chosen = match channel(
        Act::Answer,
        state.view.state,
        &probe(&d.acts, session, &state.facts),
    ) {
        Ok(chosen) => chosen,
        Err(refusal) => {
            return Ok(error(
                polylogue_model::Code::Refused,
                format!("{}: {}", refusal.reason, refusal.detail),
            ));
        }
    };
    // A hosted session's requests come to Polylogue's connection; an
    // observed one's from its permission hook, into its live facts.
    let (reference, tool, input) = if chosen == Channel::Protocol {
        let Some(request) = d.acts.host_request(session) else {
            return Ok(error(
                polylogue_model::Code::Refused,
                "the hosted session has no pending request to answer",
            ));
        };
        (
            format!("request:{session}#{request}"),
            Value::Null,
            Value::Null,
        )
    } else {
        let Some(pending) = state.facts.pending_request.clone() else {
            return Ok(error(
                polylogue_model::Code::Refused,
                "the session has no pending permission request to answer",
            ));
        };
        (
            request_reference(session, pending.at),
            Value::String(pending.tool),
            pending.input,
        )
    };
    if !standing && chosen == Channel::Hook {
        return Ok(error(
            polylogue_model::Code::Refused,
            "the harness already shows its own prompt for this request: answer it in the terminal, \
             or answer ahead with --standing for the same tool and input next time",
        ));
    }
    let _claims = d.acts.claim_lock();
    let reader = d.reads.reader()?;
    let earlier = reader
        .deliveries_to(&session.to_string())?
        .into_iter()
        .filter(|row| row.event == DeliveryEvent::Control)
        .filter_map(|row| reader.delivery(&row.id).ok().flatten())
        .find(|(_, body)| {
            serde_json::from_str::<Value>(body)
                .is_ok_and(|b| b["request"] == reference && b["standing"] == standing)
        });
    drop(reader);
    let body = json!({
        "request": reference,
        "decision": if approve { "approve" } else { "deny" },
        "text": text,
        "standing": standing,
        "tool": tool,
        "input": input,
    });
    author(
        d,
        invocation,
        request_id.to_owned(),
        UserWrite::Deliver(Delivery {
            recipient_kind: "session".into(),
            recipient_id: Some(session.to_string()),
            sender: DeliverySender::Act,
            event: DeliveryEvent::Control,
            cwd: state.cwd.clone(),
            text: body.to_string(),
            refs: vec![reference.clone()],
            left_out: Vec::new(),
        }),
    )?;
    if let Some((first, _)) = earlier {
        outcome_event(
            d,
            invocation,
            request_id,
            DeliveryOutcome::NotSent,
            format!("already_answered by delivery:{}", first.id),
        )?;
        return Ok(error(
            polylogue_model::Code::Conflict,
            format!(
                "already_answered: {reference} was answered by delivery:{}",
                first.id
            ),
        ));
    }
    match chosen {
        Channel::Protocol => {
            let result = d.acts.host_answer(session, approve, text.as_deref());
            let (outcome, what, detail, cancellation, note) = match result {
                Ok(None) => (
                    ActOutcome::Delivered,
                    DeliveryOutcome::Delivered,
                    "answered over the hosted connection".to_owned(),
                    None,
                    None,
                ),
                Ok(Some(cancellation)) => (
                    ActOutcome::Cancelled,
                    DeliveryOutcome::NotSent,
                    format!(
                        "permission request cancelled: no matching option (offered: {:?})",
                        cancellation.options
                    ),
                    Some(cancellation),
                    None,
                ),
                Err(reason) => (
                    ActOutcome::NotSent,
                    DeliveryOutcome::NotSent,
                    format!("failed({reason})"),
                    None,
                    Some(reason),
                ),
            };
            outcome_event(d, invocation, request_id, what, detail)?;
            Ok(Answer::ok(Answered {
                delivery: request_id.to_owned(),
                request: reference,
                channel: Channel::Protocol,
                outcome,
                cancellation,
                note,
                undo: "none: an answer cannot be taken back".into(),
            }))
        }
        _ => Ok(Answer::ok(Answered {
            delivery: request_id.to_owned(),
            request: reference,
            channel: Channel::Hook,
            outcome: ActOutcome::Prepared,
            cancellation: None,
            note: Some("applies when the session next asks for the same tool and input".into()),
            undo: "none: an answer cannot be taken back".into(),
        })),
    }
}

/// The synchronous permission hook asks for a decision: a standing answer
/// for the same tool and input. The hook records delivery after writing it.
fn permission(
    d: &Daemon,
    session: &SessionId,
    tool: &str,
    input: &Value,
) -> Result<Answer<Decided>, Failure> {
    let none = || {
        Answer::ok(Decided {
            decision: None,
            text: None,
            delivery: None,
        })
    };
    let reader = d.reads.reader()?;
    let standing = reader
        .deliveries_to(&session.to_string())?
        .into_iter()
        .filter(|row| row.event == DeliveryEvent::Control)
        .filter_map(|row| reader.delivery(&row.id).ok().flatten())
        .filter_map(|(row, body)| Some((row, serde_json::from_str::<Value>(&body).ok()?)))
        // An answer given ahead applies once: to the next request for the
        // same tool and input.
        .find(|(row, body)| {
            body["standing"] == true
                && body["tool"] == tool
                && &body["input"] == input
                && row.events.is_empty()
        });
    drop(reader);
    let Some((row, body)) = standing else {
        return Ok(none());
    };
    // Concurrent permission hooks: only the first applies it.
    let taken = {
        let _claims = d.acts.claim_lock();
        d.acts.take(vec![(row.id.clone(), ())])
    };
    if taken.is_empty() {
        return Ok(none());
    }
    Ok(Answer::ok(Decided {
        decision: body["decision"].as_str().map(str::to_owned),
        text: body["text"].as_str().map(str::to_owned),
        delivery: Some(row.id),
    }))
}

/// Stops a session's turn, recorded as a `control` delivery with its
/// outcome.
pub(crate) fn interrupt(
    d: &Daemon,
    invocation: &Invocation,
    request_id: &str,
    session: &SessionId,
) -> Result<Answer<Interrupted>, Failure> {
    let now = now(d);
    let Some(state) = d.reads.reader()?.live_state(&session.to_string(), now)? else {
        return Ok(error(
            polylogue_model::Code::NotFound,
            format!("no session {session}"),
        ));
    };
    author(
        d,
        invocation,
        request_id.to_owned(),
        UserWrite::Deliver(Delivery {
            recipient_kind: "session".into(),
            recipient_id: Some(session.to_string()),
            sender: DeliverySender::Act,
            event: DeliveryEvent::Control,
            cwd: state.cwd.clone(),
            text: json!({"interrupt": true}).to_string(),
            refs: Vec::new(),
            left_out: Vec::new(),
        }),
    )?;
    let (what, detail, answer) = match channel(
        Act::Interrupt,
        state.view.state,
        &probe(&d.acts, session, &state.facts),
    ) {
        Ok(_) => match d.acts.host_interrupt(session) {
            Ok(()) => (
                DeliveryOutcome::Delivered,
                "interrupted over the hosted connection".to_owned(),
                Answer::ok(Interrupted {
                    session: session.clone(),
                    delivery: request_id.to_owned(),
                    channel: Channel::Protocol,
                    outcome: ActOutcome::Delivered,
                    undo: "none: an interrupted turn is not resumed".into(),
                }),
            ),
            Err(reason) => (
                DeliveryOutcome::NotSent,
                format!("failed({reason})"),
                error(
                    polylogue_model::Code::Refused,
                    format!("not_sent: failed({reason})"),
                ),
            ),
        },
        Err(refusal) => (
            DeliveryOutcome::NotSent,
            refusal.reason.to_owned(),
            error(
                polylogue_model::Code::Refused,
                format!("{}: {}", refusal.reason, refusal.detail),
            ),
        ),
    };
    outcome_event(d, invocation, request_id, what, detail)?;
    Ok(answer)
}

/// The acts worker: records the outcomes hosted connections report and
/// starts each session's resumed runs on a thread of that session's own,
/// so one session's run never delays another's.
pub(crate) fn run_reports(
    daemon: std::sync::Arc<Daemon>,
    reports: std::sync::mpsc::Receiver<crate::live_acts::Report>,
) {
    use crate::live_acts::Report;
    let mut queue_lines = crate::live_events::PendingLines::new(
        "registry",
        "acts",
        std::sync::Arc::clone(&daemon.live_spool),
    );
    while let Ok(report) = reports.recv() {
        match report {
            Report::Queued { session, queued } => {
                let line = crate::live_events::line(
                    "registry",
                    "HostedQueued",
                    &session,
                    daemon.clock.now(),
                    json!({"queued": queued}),
                );
                let boot = crate::live_events::boot_id().unwrap_or_else(|| "unknown".into());
                let _ = queue_lines.append(&daemon.acts.spool, &boot, &[line]);
            }
            Report::Sent { delivery, sent } => record_sent(&daemon, &delivery, sent),
            Report::Resume(run, started) => resume_session(&daemon, run, started),
            Report::DeliveryWrite { id, written } => match written.recv() {
                Ok(Ok(_)) => {}
                Ok(Err(error)) => daemon.acts.delivery_write_failed(&id, error),
                Err(error) => daemon.acts.delivery_write_failed(&id, error),
            },
            Report::Stop => return,
        }
    }
}

/// Runs a session's resumed run, then each run queued behind it.
fn resume_session(
    daemon: &std::sync::Arc<Daemon>,
    run: ResumeRun,
    started: std::sync::mpsc::Sender<Result<(), String>>,
) {
    let session = run.session.clone();
    let daemon_for_runs = std::sync::Arc::clone(daemon);
    let spawned = polylogue_model::spawn_named("polylogue-resume", move || {
        let daemon = daemon_for_runs;
        let mut next = Some(run);
        let mut started = Some(started);
        while let Some(run) = next.take() {
            resume_once(&daemon, &run, started.take());
            next = daemon.acts.resume_finished(&run.session);
            while let Some(candidate) = next.take() {
                // A queued message waits for its session to stop running:
                // if it runs again (resumed by hand), the message goes to
                // its next hook instead of a second resume.
                let running = daemon
                    .reads
                    .reader()
                    .ok()
                    .and_then(|r| {
                        r.live_state(&candidate.session.to_string(), daemon.clock.now_ms())
                            .ok()
                            .flatten()
                    })
                    .is_some_and(|s| s.view.state.running());
                if !running {
                    next = Some(candidate);
                    break;
                }
                if let Err(error) = daemon
                    .acts
                    .hand_to_hook(&candidate.delivery, &candidate.session)
                {
                    polylogue_store::log!(warning, "session mailbox update failed: {error}");
                }
                next = daemon.acts.resume_finished(&candidate.session);
            }
        }
    });
    if let Err(error) = spawned {
        let invocation = daemon.invocation(
            &polylogue_reading::operation::Context::worker_authorship("resume"),
        );
        while let Some(queued) = daemon.acts.resume_finished(&session) {
            if let Err(write_error) = outcome_event(
                daemon,
                &invocation,
                &queued.delivery,
                DeliveryOutcome::NotSent,
                format!("failed({error})"),
            ) {
                daemon
                    .acts
                    .delivery_write_failed(&queued.delivery, &write_error.message);
            }
        }
    }
}

/// A headless resume takes the first message when the harness starts.
fn resume_once(
    d: &Daemon,
    run: &ResumeRun,
    started: Option<std::sync::mpsc::Sender<Result<(), String>>>,
) {
    let invocation = d.invocation(&polylogue_reading::operation::Context::worker_authorship(
        "resume",
    ));
    let record = |what, detail: String| {
        if let Err(error) = outcome_event(d, &invocation, &run.delivery, what, detail) {
            d.acts.delivery_write_failed(&run.delivery, &error.message);
        }
    };
    let child = std::process::Command::new(&run.argv[0])
        .args(&run.argv[1..])
        .current_dir(&run.cwd)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn();
    let mut child = match child {
        Ok(child) => child,
        Err(error) => {
            let problem = format!("{} in {}: {error}", run.argv[0], run.cwd.display());
            if let Some(started) = started {
                let _ = started.send(Err(problem));
            } else {
                record(DeliveryOutcome::NotSent, format!("failed({problem})"));
            }
            return;
        }
    };
    record(
        DeliveryOutcome::Delivered,
        format!("resumed headlessly ({} pid {})", run.argv[0], child.id()),
    );
    if let Some(started) = started {
        let _ = started.send(Ok(()));
    }
    let stderr = crate::hosting::stderr_tail(child.stderr.take().expect("piped stderr"));
    match child.wait() {
        Ok(status) if status.success() => {}
        Ok(status) => record(
            DeliveryOutcome::Failed,
            format!("failed({})", crate::hosting::exit_reason(status, stderr)),
        ),
        Err(e) => record(DeliveryOutcome::Failed, format!("failed({e})")),
    }
}

/// A new session Polylogue starts and hosts, and what its first turn
/// carries.
struct NewSession<'a> {
    request_id: &'a str,
    harness: &'a str,
    cwd: &'a std::path::Path,
    /// The first turn's text, without its delivery line.
    text: String,
    /// What the first turn refers to: the role revision, the hand-off's
    /// source, the message.
    refs: Vec<String>,
}

/// Starts a hosted session on the harness's control protocol, its first
/// turn opening with the delivery reference, and records that turn as a
/// `first_turn` delivery to the new session. The channel is
/// [`polylogue_model::live::channel::host_protocol`]'s.
fn start_session(
    d: &Daemon,
    invocation: &Invocation,
    new: NewSession<'_>,
) -> Result<Result<SessionId, Failure>, Failure> {
    use polylogue_model::live::channel::{HostProtocol, host_protocol};
    let protocol = match host_protocol(new.harness) {
        Ok(protocol) => protocol,
        Err(refusal) => {
            return Ok(Err(Failure::refused(format!(
                "{}: {}",
                refusal.reason, refusal.detail
            ))));
        }
    };
    if !new.cwd.is_dir() {
        return Ok(Err(Failure::refusal(Reason::NotADirectory {
            path: new.cwd.display().to_string(),
        })));
    }
    let first = format!("delivery:{}\n{}", new.request_id, new.text);
    let session = match protocol {
        HostProtocol::CodexAppServer => crate::live_acts::HostedCodex::start(
            &d.acts.codex().0,
            &d.acts.codex().1,
            &new.cwd.to_string_lossy(),
            &first,
        )
        .and_then(|hosted| d.acts.host_codex(hosted)),
        _ => crate::hosting::HostedStdio::start(
            new.harness,
            protocol,
            &d.acts.hosts,
            new.cwd,
            &d.acts.spool,
            &first,
        )
        .map(|(hosted, session)| {
            d.acts
                .hold(&session, crate::live_acts::Hosted::Stdio(hosted));
            session
        }),
    }
    .map_err(|e| Failure::io(format!("{}: {e}", protocol.as_str())))?;
    author(
        d,
        invocation,
        new.request_id.to_owned(),
        UserWrite::Deliver(Delivery {
            recipient_kind: "session".into(),
            recipient_id: Some(session.to_string()),
            sender: DeliverySender::Act,
            event: DeliveryEvent::FirstTurn,
            cwd: Some(new.cwd.to_string_lossy().into_owned()),
            text: first,
            refs: new.refs,
            left_out: Vec::new(),
        }),
    )?;
    outcome_event(
        d,
        invocation,
        new.request_id,
        DeliveryOutcome::Delivered,
        format!(
            "the first turn of a session hosted over {}",
            protocol.as_str()
        ),
    )?;
    Ok(Ok(session))
}

/// The first turn of a role's instance: the role's instructions, then the
/// work.
fn role_text(role: &Role, text: &str) -> String {
    format!(
        "You run as the role `{}` (annotation:{}):\n{}\n\n{}",
        role.name, role.annotation, role.instructions, text
    )
}

/// Starts a session Polylogue hosts, on the harness's control protocol:
/// its approvals, questions and turns can then be answered, steered and
/// interrupted from anywhere. With a role, the role's harness runs with
/// the role's instructions, and the session records the role revision.
fn start(
    d: &Daemon,
    invocation: &Invocation,
    request_id: &str,
    harness: Option<&str>,
    role_name: Option<&str>,
    cwd: &std::path::Path,
    text: &str,
) -> Result<Answer<Started>, Failure> {
    let role = role_name
        .map(|name| role(d, invocation, name))
        .transpose()?;
    let harness = match (harness, &role) {
        (Some(harness), Some(role)) if harness != role.harness => {
            return Err(Reason::RoleRunsAnotherHarness {
                role: role.name.clone(),
                runs: role.harness.clone(),
                asked: harness.to_owned(),
            }
            .into());
        }
        (Some(harness), _) => harness.to_owned(),
        (None, Some(role)) => role.harness.clone(),
        (None, None) => return Err(Reason::StartNeedsHarnessOrRole {}.into()),
    };
    let (text, refs) = match &role {
        Some(role) => (
            role_text(role, text),
            vec![
                role_reference(&role.name),
                format!("annotation:{}", role.annotation),
            ],
        ),
        None => (text.to_owned(), Vec::new()),
    };
    let session = match start_session(
        d,
        invocation,
        NewSession {
            request_id,
            harness: &harness,
            cwd,
            text,
            refs,
        },
    )? {
        Ok(session) => session,
        Err(refused) => return Ok(refused.answer()),
    };
    Ok(Answer::ok(Started {
        session,
        delivery: request_id.to_owned(),
        channel: Channel::Protocol,
        role: role.map(|r| r.name),
        undo: "none: interrupt it, or let it finish".into(),
    }))
}

/// The first turn's message text for a new role instance, which has no message
/// annotation to answer: the reply goes as a note replying to the
/// delivery, or as a message to the sender.
fn new_instance_message_text(
    delivery: &str,
    author: &str,
    text: &str,
    about: Option<&str>,
) -> String {
    let mut pack = format!("Message from {author}:\n{text}\n");
    if let Some(about) = about {
        pack.push_str(&format!("About: {about}\n"));
    }
    pack.push_str(&format!(
        "Answer with a note replying to `delivery:{delivery}` (`replies_to`, tool `polylogue_write`), or reach the sender with a `send` step (tool `polylogue_act`)."
    ));
    pack
}

/// A message to a role with no running instance starts one, in the directory
/// its latest instance ran in, with the message as its first turn.
fn message_new_instance(
    d: &Daemon,
    invocation: &Invocation,
    request_id: &str,
    role: &Role,
    text: &str,
    about: Option<&str>,
) -> Result<Answer<Sent>, Failure> {
    let Some(cwd) = role_cwd(d, &role.name)? else {
        return Ok(error(
            polylogue_model::Code::Refused,
            format!(
                "no_live_instance: the role {} has no running session and none ran before to take a directory from; start one with `polylogue start --role {} --cwd DIR TEXT`",
                role.name, role.name
            ),
        ));
    };
    let author_name = invocation.authorship.author.to_string();
    let pack = new_instance_message_text(request_id, &author_name, text, about);
    let session = match start_session(
        d,
        invocation,
        NewSession {
            request_id,
            harness: &role.harness,
            cwd: &cwd,
            text: role_text(role, &pack),
            refs: vec![
                role_reference(&role.name),
                format!("annotation:{}", role.annotation),
            ],
        },
    )? {
        Ok(session) => session,
        Err(refused) => return Ok(refused.answer()),
    };
    Ok(Answer::ok(Sent {
        message: None,
        deliveries: vec![Delivered {
            session,
            delivery: Some(request_id.to_owned()),
            channel: Some(Channel::Protocol),
            outcome: ActOutcome::Delivered,
            reason: None,
            detail: None,
            note: None,
            started: Some(role.name.clone()),
        }],
        undo: "none: a sent message cannot be taken back".into(),
    }))
}

/// The new session `resume --in` starts: its harness, its directory, the
/// message it continues from, and whether the source is interrupted.
pub(crate) struct Continuation<'a> {
    pub request_id: &'a str,
    pub source: &'a SessionId,
    pub harness: &'a str,
    /// Where the new session runs, when not where the source ran.
    pub cwd: Option<&'a std::path::Path>,
    pub at: Option<&'a str>,
    pub stop_source: bool,
}

/// What a continuation's new session is briefed with: the source's
/// session brief and the line naming the message to continue from.
pub(crate) struct Briefing {
    /// Where the new session runs; none when the source recorded no
    /// directory that exists here.
    pub cwd: Option<std::path::PathBuf>,
    pub at: Option<polylogue_model::Reference>,
    pub text: String,
    pub refs: Vec<String>,
}

/// The briefing of a continuation, read and prepared; nothing starts.
pub(crate) fn briefing(
    d: &Daemon,
    invocation: &Invocation,
    c: &Continuation<'_>,
) -> Result<Briefing, Failure> {
    let reader = d.reads.reader()?;
    let Some(source) = reader.session(c.source)? else {
        return Err(Failure::not_found(format!("no session {}", c.source)));
    };
    let cwd = match c.cwd {
        Some(cwd) => Some(cwd.to_owned()),
        None => source
            .cwd
            .last()
            .cloned()
            .map(std::path::PathBuf::from)
            .filter(|c| c.is_dir()),
    };
    let at =
        c.at.map(polylogue_model::Reference::parse)
            .transpose()
            .map_err(|problem| {
                Failure::refusal(Reason::BadReference {
                    text: c.at.unwrap_or_default().to_owned(),
                    problem: problem.to_string(),
                })
            })?;
    let operation = d.resolve_operation_context(invocation.clone())?;
    let prepared = crate::brief::prepare(
        crate::brief::BriefContext {
            reader: &reader,
            pending_deliveries: &[],
            delivery_id: c.request_id,
            context: &operation,
            ranking_identity: None,
            near: None,
        },
        &d.scope,
        &[],
        &crate::brief::Ask::Session {
            session: c.source.clone(),
            for_session: None,
        },
        Default::default(),
    )?;
    drop(reader);
    let mut text = format!("Hand-off from {}", c.source);
    if let Some(at) = &at {
        text.push_str(&format!(", continuing from {at}"));
    }
    text.push_str(":\n");
    text.push_str(&prepared.text);
    let mut refs = vec![format!("session:{}", c.source)];
    refs.extend(at.as_ref().map(polylogue_model::Reference::to_string));
    refs.extend(prepared.refs.iter().cloned());
    Ok(Briefing {
        cwd,
        at,
        text,
        refs,
    })
}

/// Continues a session's work in a new hosted session in any harness: the
/// new session starts in the source's directory with the source's session
/// brief and a line naming the message to continue from (`at`), its first
/// turn carrying the brief's delivery reference, and records the source as
/// what it was briefed from. It is a new session, never a resume or a fork
/// of the source. `stop_source` then interrupts the source.
pub(crate) fn resume_in(
    d: &Daemon,
    invocation: &Invocation,
    c: &Continuation<'_>,
) -> Result<Answer<Continued>, Failure> {
    let briefing = briefing(d, invocation, c)?;
    let Some(cwd) = briefing.cwd else {
        return Ok(error(
            polylogue_model::Code::Refused,
            format!(
                "{} recorded no working directory that exists here; give one with --cwd",
                c.source
            ),
        ));
    };
    let previous = writer(d)?.archive().committed_for(c.request_id)?;
    let session = if let Some(previous) = previous {
        let request = previous.request;
        let UserWrite::Deliver(first_turn) = request.write else {
            return Err(Reason::ContinuationRequestUsed {}.into());
        };
        let source = format!("session:{}", c.source);
        if request.authorship != invocation.authorship
            || first_turn.event != DeliveryEvent::FirstTurn
            || first_turn.sender != DeliverySender::Act
            || first_turn.refs.first() != Some(&source)
            || first_turn.cwd.as_deref() != cwd.to_str()
        {
            return Err(Reason::ContinuationRequestUsed {}.into());
        }
        let session = first_turn
            .recipient_id
            .as_deref()
            .ok_or_else(|| Failure::internal("the first turn has no recipient"))?;
        let session = SessionId::parse(session).map_err(Failure::internal)?;
        if session.origin().as_str() != c.harness {
            return Err(Reason::ContinuationRequestUsed {}.into());
        }
        session
    } else {
        match start_session(
            d,
            invocation,
            NewSession {
                request_id: c.request_id,
                harness: c.harness,
                cwd: &cwd,
                text: briefing.text,
                refs: briefing.refs,
            },
        )? {
            Ok(session) => session,
            Err(refused) => return Ok(refused.answer()),
        }
    };
    let stop_source = if c.stop_source {
        Some(
            interrupt(d, invocation, &format!("{}-stop", c.request_id), c.source)
                .unwrap_or_else(Failure::answer),
        )
    } else {
        None
    };
    Ok(Answer::ok(Continued {
        session,
        delivery: c.request_id.to_owned(),
        channel: Channel::Protocol,
        briefed_from: c.source.clone(),
        at: briefing.at.map(|at| at.to_string()),
        stop_source,
        undo: "none: the new session runs on its own; interrupt it to stop it".into(),
    }))
}
