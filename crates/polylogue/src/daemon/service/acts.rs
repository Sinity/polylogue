//! Live-session acts: messages (one session, a role's running instance, or
//! every session a query selects when sent; a web chat's turns are in
//! [`web_chat`]), signal notices to sessions,
//! the mailbox claim a hook makes, answers to permission requests,
//! interrupts, and new hosted sessions (`start`, role instances,
//! `hand-off`). Each act is recorded as a delivery with its computed
//! author, and reaches its session only through the channel
//! [`polylogue_model::live::channel::channel`] chooses; a new session
//! starts on the protocol [`polylogue_model::live::channel::host_protocol`]
//! names.
use super::{Daemon, Failure, authored_preparation, error, ok, writer};
use crate::commands::ToolAccess;
use crate::daemon::handler::{Call, Operation, Policy};
use crate::live_acts::{ResumeRun, probe};
use crate::protocol::act as protocol;
use command_surface::Answer;
use polylogue_model::live::channel::{Act, Channel, channel};
use polylogue_model::user::{Delivery, UserOutcome, UserWrite};
use polylogue_model::{DeliveryEvent, DeliveryOutcome, DeliverySender, SessionId};
use polylogue_reading::operation::Invocation;
use serde_json::{Value, json};

#[cfg(test)]
mod tests;
pub(crate) mod web_chat;

/// The acts' handlers: each records its act and reaches its session.
macro_rules! act {
    ($(#[$doc:meta])* $name:ident, $arguments:ty, $access:expr, |$args:ident, $call:ident| $body:expr) => {
        $(#[$doc])*
        pub(crate) struct $name;
        impl Operation for $name {
            type Arguments = $arguments;
            type Reply = Value;
            fn policy(_: &Self::Arguments, _: polylogue_model::authorship::Channel) -> Result<Policy, Failure> {
                Ok(Policy::write($access))
            }
            fn run($args: Self::Arguments, $call: &Call<'_>) -> Result<Answer<Value>, Failure> {
                $body
            }
        }
    };
}

act!(
    SendMessage,
    protocol::SendMessage,
    ToolAccess::Act,
    |a, call| send_message(
        call.daemon,
        call.invocation,
        call.request_id,
        &a.target,
        &a.text,
        a.live,
        a.about
    )
);
act!(
    /// A hook takes the messages and notices pending for its session.
    Claim,
    protocol::Claim, ToolAccess::Write,
    |a, call| claim(call.daemon, &a.session)
);
act!(
    AnswerRequest,
    protocol::Answer,
    ToolAccess::Act,
    |a, call| answer(
        call.daemon,
        call.invocation,
        call.request_id,
        &a.session,
        a.decision == protocol::Decision::Approve,
        a.text,
        a.standing
    )
);
act!(
    Permission,
    protocol::Permission,
    ToolAccess::Write,
    |a, call| permission(call.daemon, call.invocation, &a.session, &a.tool, &a.input)
);
act!(
    Interrupt,
    protocol::Interrupt,
    ToolAccess::Act,
    |a, call| interrupt(call.daemon, call.invocation, call.request_id, &a.session)
);
act!(Start, protocol::Start, ToolAccess::Act, |a, call| {
    if let Some(provider) = a
        .harness
        .as_deref()
        .and_then(polylogue_sources::origin::web_chat_name)
    {
        return web_chat::start(call, provider, a);
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
        Err(Failure::bad_request("start takes --in or --role"))
    } else {
        let cwd = call.directory(a.cwd.as_deref())?;
        start(
            call.daemon,
            call.invocation,
            call.request_id,
            a.harness.as_deref(),
            a.role.as_deref(),
            &cwd,
            &a.text,
        )
    }
});
act!(
    HandOffSession,
    protocol::HandOff,
    ToolAccess::Act,
    |a, call| hand_off(
        call.daemon,
        call.invocation,
        &HandOff {
            request_id: call.request_id,
            source: &a.session,
            to: &a.to,
            cwd: a.cwd.as_deref(),
            at: a.at.as_deref(),
            stop_source: a.stop_source,
            print: a.print,
        },
    )
);

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
    Ok(writes
        .archive()
        .user(polylogue_store::writer::UserSubmission {
            request,
            scope: None,
            preparation: Some(authored_preparation()),
            parent: None,
        })?)
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
    d.queued_deliveries
        .submit(writer(d)?, request, None, None)
        .map_err(Failure::from)
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
) -> Result<Value, Failure> {
    Ok(match d.acts.host_send(session, pack, Some(delivery)) {
        Ok(true) => {
            outcome_event(
                d,
                invocation,
                delivery,
                DeliveryOutcome::Delivered,
                "sent over the hosted connection".into(),
            )?;
            json!({"session": session, "delivery": delivery, "channel": "protocol", "outcome": "delivered"})
        }
        Ok(false) => {
            json!({"session": session, "delivery": delivery, "channel": "protocol", "outcome": "prepared",
                   "note": "waits for the running prompt; recorded when it is sent"})
        }
        Err(reason) => {
            outcome_event(
                d,
                invocation,
                delivery,
                DeliveryOutcome::NotSent,
                format!("failed({reason})"),
            )?;
            json!({"session": session, "delivery": delivery, "channel": "protocol", "outcome": "not_sent", "reason": reason})
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
    let _ = outcome_event(d, &invocation, delivery, what, detail);
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
fn send_message(
    d: &Daemon,
    invocation: &Invocation,
    request_id: &str,
    target: &str,
    text: &str,
    live: bool,
    about: Option<String>,
) -> Result<Answer<Value>, Failure> {
    if text.trim().is_empty() {
        return Ok(error(
            polylogue_model::Code::BadRequest,
            "a message needs text",
        ));
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
                sent.push(
                    json!({"session": session, "outcome": "not_sent", "reason": "session_missing"}),
                );
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
        let row = match chosen {
            Ok(Channel::Hook) => {
                d.acts
                    .hand_to_hook(&delivery, session)
                    .map_err(Failure::io)?;
                json!({"session": session, "delivery": delivery, "channel": "hook",
                       "outcome": "prepared", "note": "reaches the session at its next tool result or prompt"})
            }
            Ok(Channel::Resume) => {
                // Held for the resumed run: no hook takes it.
                let started = d.acts.resume(ResumeRun {
                    session: session.clone(),
                    delivery: delivery.clone(),
                    text: pack,
                });
                json!({"session": session, "delivery": delivery, "channel": "resume",
                       "outcome": if started { "started" } else { "queued" }})
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
                json!({"session": session, "delivery": delivery, "outcome": "not_sent",
                       "reason": refusal.reason, "detail": refusal.detail})
            }
        };
        sent.push(row);
    }
    Ok(ok(
        json!({"message": message, "deliveries": sent, "undo": "none: a sent message cannot be taken back"}),
    ))
}

/// A hook's mailbox claim: every message and notice pending for the session,
/// oldest first, each handed out once. Claims run one at a time, so
/// concurrent hooks of one session never take the same delivery. The hook
/// records each `delivered` once it has written it to the harness; a claim
/// whose hook never confirms stays without an outcome and is not handed
/// out again.
fn claim(d: &Daemon, session: &SessionId) -> Result<Answer<Value>, Failure> {
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
    let claimed: Vec<Value> = d
        .acts
        .take(pending)
        .into_iter()
        .map(|(id, text)| json!({"id": id, "text": text}))
        .collect();
    Ok(ok(json!({ "deliveries": claimed })))
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
    let Ok(reader) = d.reads.reader() else {
        return;
    };
    // A hosted session that ended needs no connection or record.
    for session in d.acts.hosted_sessions() {
        let ended = reader
            .live_state(&session.to_string(), now)
            .ok()
            .flatten()
            .is_some_and(|s| s.view.state == polylogue_model::live::LiveState::Ended);
        if ended {
            d.acts.release_host(&session);
        }
    }
    let sessions: Vec<String> = if everyone {
        reader
            .sessions_with_pending_hook_deliveries()
            .unwrap_or_default()
    } else {
        crate::mailbox::flagged(d.acts.mail())
            .unwrap_or_else(|error| {
                polylogue_store::log!(warning, "session mailbox read failed: {error}");
                Vec::new()
            })
            .iter()
            .map(ToString::to_string)
            .collect()
    };
    let mut stopped = Vec::new();
    for session in sessions {
        let Ok(id) = SessionId::parse(&session) else {
            continue;
        };
        let state = reader
            .live_state(&session, now)
            .ok()
            .flatten()
            .map(|s| s.view.state);
        let pending: Vec<(String, ())> = reader
            .pending_hook_deliveries(&session)
            .unwrap_or_default()
            .into_iter()
            .filter(|(item, _)| !d.acts.handed_out(item))
            .map(|(item, _)| (item, ()))
            .collect();
        match state {
            Some(state) if state.running() => {
                if !pending.is_empty()
                    && let Err(error) = crate::mailbox::raise(d.acts.mail(), &id)
                {
                    polylogue_store::log!(warning, "session mailbox update failed: {error}");
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
            let _ = outcome_event(
                d,
                &invocation,
                &id,
                DeliveryOutcome::NotSent,
                reason.clone(),
            );
        }
        if let Err(error) = crate::mailbox::lower(d.acts.mail(), &session) {
            polylogue_store::log!(warning, "session mailbox update failed: {error}");
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
fn answer(
    d: &Daemon,
    invocation: &Invocation,
    request_id: &str,
    session: &SessionId,
    approve: bool,
    text: Option<String>,
    standing: bool,
) -> Result<Answer<Value>, Failure> {
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
            let (what, detail) = match &result {
                Ok(()) => (
                    DeliveryOutcome::Delivered,
                    "answered over the hosted connection".to_owned(),
                ),
                Err(reason) => (DeliveryOutcome::NotSent, format!("failed({reason})")),
            };
            outcome_event(d, invocation, request_id, what, detail)?;
            Ok(ok(
                json!({"delivery": request_id, "request": reference, "channel": "protocol",
                         "outcome": what.as_str(), "undo": "none: an answer cannot be taken back"}),
            ))
        }
        _ => Ok(ok(
            json!({"delivery": request_id, "request": reference, "channel": "hook",
                          "outcome": "prepared", "note": "applies when the session next asks for the same tool and input",
                          "undo": "none: an answer cannot be taken back"}),
        )),
    }
}

/// The synchronous permission hook asks for a decision: a standing answer
/// for the same tool and input, recorded delivered when it applies. With
/// none it returns nothing at once, and the harness shows its own prompt.
fn permission(
    d: &Daemon,
    invocation: &Invocation,
    session: &SessionId,
    tool: &str,
    input: &Value,
) -> Result<Answer<Value>, Failure> {
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
        return Ok(ok(json!({ "decision": null })));
    };
    // Concurrent permission hooks: only the first applies it.
    let taken = {
        let _claims = d.acts.claim_lock();
        d.acts.take(vec![(row.id.clone(), ())])
    };
    if taken.is_empty() {
        return Ok(ok(json!({ "decision": null })));
    }
    outcome_event(
        d,
        invocation,
        &row.id,
        DeliveryOutcome::Delivered,
        format!("applied to the {tool} permission request"),
    )?;
    Ok(ok(json!({
        "decision": body["decision"],
        "text": body["text"],
        "delivery": row.id,
    })))
}

/// Stops a session's turn, recorded as a `control` delivery with its
/// outcome.
fn interrupt(
    d: &Daemon,
    invocation: &Invocation,
    request_id: &str,
    session: &SessionId,
) -> Result<Answer<Value>, Failure> {
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
                ok(
                    json!({"session": session, "delivery": request_id, "channel": "protocol",
                              "outcome": "delivered", "undo": "none: an interrupted turn is not resumed"}),
                ),
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
            Report::Resume(run) => resume_session(&daemon, run),
            Report::Stop => return,
        }
    }
}

/// Runs a session's resumed run, then each run queued behind it.
fn resume_session(daemon: &std::sync::Arc<Daemon>, run: ResumeRun) {
    let session = run.session.clone();
    let delivery = run.delivery.clone();
    let daemon_for_runs = std::sync::Arc::clone(daemon);
    let spawned = polylogue_model::spawn_named("polylogue-resume", move || {
        let daemon = daemon_for_runs;
        let mut next = Some(run);
        while let Some(run) = next.take() {
            resume_once(&daemon, &run);
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
    if spawned.is_err() {
        // Nothing runs: the messages go back to the mailbox, where they are
        // handled like any other.
        daemon.acts.release(&delivery);
        while let Some(queued) = daemon.acts.resume_finished(&session) {
            daemon.acts.release(&queued.delivery);
        }
    }
}

/// One headless resume: `delivered` when the harness starts with the message
/// as its prompt, `failed` with the first line of its error output when it
/// exits unsuccessfully. It runs in the session's resume directory, and is
/// `not_sent` (`directory_unknown`, `directory_missing(DIR)`) without one:
/// the daemon's own directory is never borrowed.
fn resume_once(d: &Daemon, run: &ResumeRun) {
    let invocation = d.invocation(&polylogue_reading::operation::Context::worker_authorship(
        "resume",
    ));
    let record = |what, detail: String| {
        let _ = outcome_event(d, &invocation, &run.delivery, what, detail);
    };
    let Some(argv) = crate::live_acts::headless_resume(&run.session, &run.text) else {
        record(DeliveryOutcome::NotSent, "unsupported".into());
        return;
    };
    let directory = match d
        .reads
        .reader()
        .map_err(|e| e.to_string())
        .and_then(|reader| {
            reader
                .resume_directory(&run.session)
                .map_err(|e| e.to_string())
        }) {
        Ok(Some(dir)) if std::path::Path::new(&dir).is_dir() => dir,
        Ok(Some(dir)) => {
            record(
                DeliveryOutcome::NotSent,
                format!("directory_missing({dir})"),
            );
            return;
        }
        Ok(None) => {
            record(DeliveryOutcome::NotSent, "directory_unknown".into());
            return;
        }
        Err(e) => {
            record(DeliveryOutcome::NotSent, format!("failed({e})"));
            return;
        }
    };
    let mut command = std::process::Command::new(&argv[0]);
    command
        .args(&argv[1..])
        .current_dir(directory)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped());
    let child = match command.spawn() {
        Ok(child) => child,
        Err(e) => {
            record(DeliveryOutcome::NotSent, format!("failed({e})"));
            return;
        }
    };
    record(
        DeliveryOutcome::Delivered,
        format!("resumed headlessly ({} pid {})", argv[0], child.id()),
    );
    match child.wait_with_output() {
        Ok(output) if output.status.success() => {}
        Ok(output) => {
            let first = String::from_utf8_lossy(&output.stderr)
                .lines()
                .next()
                .unwrap_or("")
                .to_owned();
            record(DeliveryOutcome::Failed, format!("failed({first})"));
        }
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
) -> Result<Result<SessionId, Answer<Value>>, Failure> {
    use polylogue_model::live::channel::{HostProtocol, host_protocol};
    let protocol = match host_protocol(new.harness) {
        Ok(protocol) => protocol,
        Err(refusal) => {
            return Ok(Err(error(
                polylogue_model::Code::Refused,
                format!("{}: {}", refusal.reason, refusal.detail),
            )));
        }
    };
    if !new.cwd.is_dir() {
        return Ok(Err(error(
            polylogue_model::Code::BadValue,
            format!("{} is not a directory", new.cwd.display()),
        )));
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
    .map_err(|e| Failure::refused(format!("{}: {e}", protocol.as_str())))?;
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
) -> Result<Answer<Value>, Failure> {
    let role = role_name
        .map(|name| role(d, invocation, name))
        .transpose()?;
    let harness = match (harness, &role) {
        (Some(harness), Some(role)) if harness != role.harness => {
            return Ok(error(
                polylogue_model::Code::BadValue,
                format!(
                    "the role {} runs {}, not {harness}",
                    role.name, role.harness
                ),
            ));
        }
        (Some(harness), _) => harness.to_owned(),
        (None, Some(role)) => role.harness.clone(),
        (None, None) => {
            return Ok(error(
                polylogue_model::Code::BadRequest,
                "start takes --in or --role",
            ));
        }
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
        Err(refused) => return Ok(refused),
    };
    Ok(ok(json!({
        "session": session,
        "delivery": request_id,
        "channel": "protocol",
        "hosting": "hosted",
        "role": role.map(|r| r.name),
        "undo": "none: interrupt it, or let it finish",
    })))
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
) -> Result<Answer<Value>, Failure> {
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
        Err(refused) => return Ok(refused),
    };
    Ok(ok(json!({
        "deliveries": [{"session": session, "delivery": request_id, "channel": "protocol",
                        "outcome": "delivered", "started": role.name}],
        "undo": "none: a sent message cannot be taken back",
    })))
}

/// A hand-off's arguments.
struct HandOff<'a> {
    request_id: &'a str,
    source: &'a SessionId,
    to: &'a str,
    /// Where the new session runs, when not where the source ran.
    cwd: Option<&'a std::path::Path>,
    at: Option<&'a str>,
    stop_source: bool,
    print: bool,
}

/// Hands a session's work to a new hosted session in any harness: the new
/// session starts in the source's directory with the source's takeover
/// pack and a line naming the message to continue from (`at`), its first
/// turn carrying the pack's
/// delivery reference, and records the source as what it was briefed
/// from. It is a new session, never a resume or a fork of the source.
/// `stop_source` then interrupts the source; `print` shows the pack and
/// starts nothing.
fn hand_off(
    d: &Daemon,
    invocation: &Invocation,
    h: &HandOff<'_>,
) -> Result<Answer<Value>, Failure> {
    let reader = d.reads.reader()?;
    let Some(source) = reader.session(h.source)? else {
        return Ok(error(
            polylogue_model::Code::NotFound,
            format!("no session {}", h.source),
        ));
    };
    let cwd = match h.cwd {
        Some(cwd) => Some(cwd.to_owned()),
        None => source
            .cwd
            .last()
            .cloned()
            .map(std::path::PathBuf::from)
            .filter(|c| c.is_dir()),
    };
    let at =
        h.at.map(polylogue_model::Reference::parse)
            .transpose()
            .map_err(Failure::bad_value)?;
    let operation = d.resolve_operation_context(invocation.clone())?;
    let prepared = crate::brief::prepare(
        crate::brief::BriefContext {
            reader: &reader,
            delivery_id: h.request_id,
            context: &operation,
            ranking_identity: None,
            near: None,
        },
        &d.scope,
        None,
        &[],
        &crate::brief::Ask::Takeover {
            session: h.source.clone(),
            for_session: None,
        },
        Default::default(),
    )?;
    drop(reader);
    let mut text = format!("Hand-off from {}", h.source);
    if let Some(at) = &at {
        text.push_str(&format!(", continuing from {at}"));
    }
    text.push_str(":\n");
    text.push_str(&prepared.text);
    let mut refs = vec![format!("session:{}", h.source)];
    refs.extend(at.as_ref().map(polylogue_model::Reference::to_string));
    refs.extend(prepared.refs.iter().cloned());
    if h.print {
        let channel = polylogue_model::live::channel::host_protocol(h.to);
        return Ok(ok(json!({
            "to": h.to,
            "cwd": cwd,
            "channel": channel.as_ref().ok().map(|p| p.as_str()),
            "refused": channel
                .as_ref()
                .err()
                .map(|r| format!("{}: {}", r.reason, r.detail)),
            "pack": text,
            "refs": refs,
        })));
    }
    let Some(cwd) = cwd else {
        return Ok(error(
            polylogue_model::Code::Refused,
            format!(
                "{} recorded no working directory that exists here; give one with --cwd",
                h.source
            ),
        ));
    };
    let session = match start_session(
        d,
        invocation,
        NewSession {
            request_id: h.request_id,
            harness: h.to,
            cwd: &cwd,
            text,
            refs,
        },
    )? {
        Ok(session) => session,
        Err(refused) => return Ok(refused),
    };
    let stopped = if h.stop_source {
        let answer = interrupt(d, invocation, &format!("{}-stop", h.request_id), h.source)?;
        serde_json::to_value(&answer).unwrap_or(Value::Null)
    } else {
        Value::Null
    };
    Ok(ok(json!({
        "session": session,
        "delivery": h.request_id,
        "channel": "protocol",
        "hosting": "hosted",
        "briefed_from": h.source,
        "at": at.map(|a| a.to_string()),
        "stop_source": stopped,
        "undo": "none: the new session runs on its own; interrupt it to stop it",
    })))
}
