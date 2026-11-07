//! Live-session acts: pings (one session, a role's running instance, or
//! every session a query selects when sent), signal notices to sessions,
//! the mailbox claim a hook makes, answers to permission requests,
//! interrupts, and new hosted sessions (`start`, role instances,
//! `hand-off`). Each act is recorded as a delivery with its computed
//! author, and reaches its session only through the channel
//! [`polylogue_model::live::channel::channel`] chooses; a new session
//! starts on the protocol [`polylogue_model::live::channel::host_channel`]
//! names.
use super::{Daemon, Failure, Invocation, authored_preparation, error, ok, writer};
use crate::daemon::Request;
use crate::live_acts::{ResumeRun, probe};
use command_surface::Answer;
use polylogue_model::live::channel::{Act, Channel, channel};
use polylogue_model::user::{Delivery, UserOutcome, UserWrite};
use polylogue_model::{DeliveryEvent, DeliveryOutcome, DeliverySender, SessionId};
use serde_json::{Value, json};

#[cfg(test)]
mod tests;

pub(super) fn handle(request: Request, d: &Daemon, invocation: &Invocation) -> Answer<Value> {
    let result = match request {
        Request::Ping {
            request_id,
            target,
            text,
            live,
            about,
        } => ping(d, invocation, &request_id, &target, &text, live, about),
        Request::Claim { session, event: _ } => claim(d, &session),
        Request::Answer {
            request_id,
            session,
            approve,
            text,
            standing,
        } => answer(
            d,
            invocation,
            &request_id,
            &session,
            approve,
            text,
            standing,
        ),
        Request::Permission {
            session,
            tool,
            input,
        } => permission(d, invocation, &session, &tool, &input),
        Request::Interrupt {
            request_id,
            session,
        } => interrupt(d, invocation, &request_id, &session),
        Request::Start {
            request_id,
            harness,
            role,
            cwd,
            text,
        } => start(
            d,
            invocation,
            &request_id,
            harness.as_deref(),
            role.as_deref(),
            &cwd,
            &text,
        ),
        Request::HandOff {
            request_id,
            session,
            to,
            cwd,
            at,
            stop_source,
            print,
        } => hand_off(
            d,
            invocation,
            &HandOff {
                request_id: &request_id,
                source: &session,
                to: &to,
                cwd: cwd.as_deref(),
                at: at.as_deref(),
                stop_source,
                print,
            },
        ),
        _ => unreachable!("only live-session acts reach here"),
    };
    result.unwrap_or_else(|failure| failure.answer())
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
    invocation.check_parent_admission()?;
    let request = invocation.request(request_id, write);
    Ok(writes.user_preparing(request, authored_preparation(), None)?)
}

fn outcome_event(
    d: &Daemon,
    invocation: &Invocation,
    delivery: &str,
    what: DeliveryOutcome,
    detail: String,
) -> Result<(), Failure> {
    author(
        d,
        invocation,
        polylogue_store::user_writes::new_request_id(),
        UserWrite::DeliveryEvent {
            delivery: delivery.to_owned(),
            what,
            detail: Some(detail),
        },
    )
    .map(drop)
}

/// Who an address names when it is sent.
enum Recipients {
    /// Sessions: one reference, or every session a quoted query selects
    /// then (a broadcast, frozen: sessions that start matching later
    /// receive nothing).
    Sessions(Vec<SessionId>),
    /// A role with no running instance: a ping starts one; a notice does
    /// not.
    NewInstance(Role),
}

/// A role as its current revision defines it.
#[derive(Debug, Clone)]
pub(crate) struct Role {
    pub name: String,
    /// The exact revision, which the sessions it starts record.
    pub assertion: String,
    pub instructions: String,
    pub harness: String,
}

/// The current revision of the role `name`.
fn role(d: &Daemon, invocation: &Invocation, name: &str) -> Result<Role, Failure> {
    use polylogue_model::assertion::values;
    let operation = d.resolve_operation_context(invocation.clone())?;
    let reader = d.reads.reader()?;
    let conditions = format!(
        "kind:role key:{}",
        crate::daemon::query::assertions::quoted(name)
    );
    let found = crate::daemon::query::assertions::list(&reader, &conditions, &operation)?;
    let Some(assertion) = found.into_iter().next() else {
        return Err(Failure::new("not_found", format!("no role {name}")));
    };
    let Ok(values::Value::Role(role)) = assertion.fact.validate() else {
        return Err(Failure::new(
            "bad_value",
            format!("the role {name} has no instructions and harness"),
        ));
    };
    Ok(Role {
        name: name.to_owned(),
        assertion: assertion.fact.id.clone(),
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
fn role_instance(d: &Daemon, name: &str, now: i64) -> Result<Option<SessionId>, Failure> {
    let reader = d.reads.reader()?;
    for (session, _, _) in reader.sessions_started_with(&role_reference(name))? {
        let Ok(id) = SessionId::parse(&session) else {
            continue;
        };
        // A session just started has no live evidence yet; its open
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
fn role_cwd(d: &Daemon, name: &str) -> Result<Option<std::path::PathBuf>, Failure> {
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
            .map(|id| SessionId::parse(id).map_err(|e| Failure::new("internal", e)))
            .collect::<Result<_, _>>()?,
    ))
}

/// A session's live state, live facts and directory for an act: from the
/// archive, or, for a session Polylogue just started and still hosts, no
/// evidence yet and state `unknown`.
fn session_state(
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

/// Sends a ping or notice over a hosted connection: `delivered` when
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
pub(crate) fn record_sent(d: &Daemon, delivery: &str, sent: Result<(), String>) {
    let invocation = d.invocation(&crate::operation::Context::worker_authorship("hosted"));
    let (what, detail) = match sent {
        Ok(()) => (
            DeliveryOutcome::Delivered,
            "sent over the hosted connection after the running prompt".to_owned(),
        ),
        Err(reason) => (DeliveryOutcome::NotSent, format!("failed({reason})")),
    };
    let _ = outcome_event(d, &invocation, delivery, what, detail);
}

/// The ping pack: the text, what it is about, how to answer, and the
/// delivery reference.
fn ping_text(delivery: &str, ping: &str, author: &str, text: &str, about: Option<&str>) -> String {
    let mut pack = format!("Ping from {author} (delivery:{delivery}, assertion:{ping}):\n{text}\n");
    if let Some(about) = about {
        pack.push_str(&format!("About: {about}\n"));
    }
    pack.push_str(&format!(
        "Answer in your reply with `::finding(answers=assertion:{ping}): …`, or address the sender with `@[…] …`."
    ));
    pack
}

#[allow(clippy::too_many_arguments)]
fn ping(
    d: &Daemon,
    invocation: &Invocation,
    request_id: &str,
    target: &str,
    text: &str,
    live: bool,
    about: Option<String>,
) -> Result<Answer<Value>, Failure> {
    if text.trim().is_empty() {
        return Ok(error("bad_request", "a ping needs text"));
    }
    let now = now(d);
    let sessions = match recipients(d, invocation, target, now)? {
        Recipients::Sessions(sessions) => sessions,
        Recipients::NewInstance(role) => {
            return ping_new_instance(d, invocation, request_id, &role, text, about.as_deref());
        }
    };
    if sessions.is_empty() {
        return Ok(error("not_found", "the address names no session"));
    }
    let subject = about
        .as_deref()
        .map(polylogue_model::Reference::parse)
        .transpose()
        .map_err(|e| Failure::new("bad_value", e))?;
    let assertion = polylogue_model::assertion::NewAssertion {
        kind: polylogue_model::assertion::AssertionKind::Ping,
        key: None,
        value: Some(
            json!({"text": text, "subject": subject.as_ref().map(ToString::to_string), "reply_to": null}),
        ),
        valid_until: None,
        links: sessions
            .iter()
            .map(|s| polylogue_model::assertion::NewAssertionLink {
                role: polylogue_model::assertion::LinkRole::About,
                reference: polylogue_model::Reference::Session(s.clone()),
            })
            .collect(),
    };
    let ping = author(
        d,
        invocation,
        format!("{request_id}-ping"),
        UserWrite::Assert {
            assertions: vec![assertion],
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
        let pack = ping_text(&delivery, &ping, &author_name, text, about.as_deref());
        // Held from before its row exists, so nothing settles or claims it
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
                sender: DeliverySender::Ping.as_str().into(),
                event: DeliveryEvent::Ping.as_str().into(),
                cwd: cwd.clone(),
                text: pack.clone(),
                refs: vec![format!("assertion:{ping}")],
                left_out: Vec::new(),
            }),
        )?;
        let chosen = channel(
            Act::Ping { live },
            view_state,
            &probe(&d.acts, session, &facts),
        );
        let row = match chosen {
            Ok(Channel::Hook) => {
                crate::mailbox::raise(d.acts.mail(), session).map_err(|e| Failure::new("io", e))?;
                d.acts.release(&delivery);
                json!({"session": session, "delivery": delivery, "channel": "hook",
                       "outcome": "prepared", "note": "reaches the session at its next tool result or prompt"})
            }
            Ok(Channel::Resume) => {
                // Held for the resumed run: no hook takes it.
                let started = d.acts.resume(ResumeRun {
                    session: session.clone(),
                    delivery: delivery.clone(),
                    text: pack,
                    cwd,
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
        json!({"ping": ping, "deliveries": sent, "undo": "none: a sent ping cannot be taken back"}),
    ))
}

/// A hook's mailbox claim: every ping and notice pending for the session,
/// oldest first, each handed out once. Claims run one at a time, so
/// concurrent hooks of one session never take the same delivery. The hook
/// records each `delivered` once it has written it to the harness; a claim
/// whose hook never confirms stays without an outcome and is not handed
/// out again.
fn claim(d: &Daemon, session: &SessionId) -> Result<Answer<Value>, Failure> {
    let _claims = d.acts.claim_lock();
    let pending = d
        .reads
        .reader()?
        .pending_hook_deliveries(&session.to_string())?;
    let claimed: Vec<Value> = d
        .acts
        .take(pending)
        .into_iter()
        .map(|(id, text)| json!({"id": id, "text": text}))
        .collect();
    crate::mailbox::lower(d.acts.mail(), session).map_err(|e| Failure::new("io", e))?;
    Ok(ok(json!({ "deliveries": claimed })))
}

/// Pings and notices still pending for sessions that stopped running
/// reach no hook: each is recorded `not_sent:session_ended` and the flag
/// lowered. Run by the live worker after each change, and at start for
/// every session with pending items (a resumed run queued when the daemon
/// stopped is among them).
/// `everyone` reads every session with pending items (at start); otherwise
/// only sessions whose flag is up are looked at. Items already handed out
/// are left to their hook or run.
pub(crate) fn settle_ended(d: &Daemon, now: i64, everyone: bool) {
    let invocation = d.invocation(&crate::operation::Context::worker_authorship("mailbox"));
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
                if !pending.is_empty() {
                    let _ = crate::mailbox::raise(d.acts.mail(), &id);
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
        let _ = crate::mailbox::lower(d.acts.mail(), &session);
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
        return Ok(error("not_found", format!("no session {session}")));
    };
    let chosen = match channel(
        Act::Answer,
        state.view.state,
        &probe(&d.acts, session, &state.facts),
    ) {
        Ok(chosen) => chosen,
        Err(refusal) => {
            return Ok(error(
                "refused",
                format!("{}: {}", refusal.reason, refusal.detail),
            ));
        }
    };
    // A hosted session's requests come to Polylogue's connection; an
    // observed one's from its permission hook, into its live facts.
    let (reference, tool, input) = if chosen == Channel::Protocol {
        let Some(request) = d.acts.host_request(session) else {
            return Ok(error(
                "refused",
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
                "refused",
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
            "refused",
            "the harness already shows its own prompt for this request: answer it in the terminal, \
             or answer ahead with --standing for the same tool and input next time",
        ));
    }
    let _claims = d.acts.claim_lock();
    let reader = d.reads.reader()?;
    let earlier = reader
        .deliveries_to(&session.to_string())?
        .into_iter()
        .filter(|row| row.event == DeliveryEvent::Control.as_str())
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
            sender: DeliverySender::Act.as_str().into(),
            event: DeliveryEvent::Control.as_str().into(),
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
            "refused",
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
        .filter(|row| row.event == DeliveryEvent::Control.as_str())
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
        return Ok(error("not_found", format!("no session {session}")));
    };
    author(
        d,
        invocation,
        request_id.to_owned(),
        UserWrite::Deliver(Delivery {
            recipient_kind: "session".into(),
            recipient_id: Some(session.to_string()),
            sender: DeliverySender::Act.as_str().into(),
            event: DeliveryEvent::Control.as_str().into(),
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
                error("refused", format!("not_sent: failed({reason})")),
            ),
        },
        Err(refusal) => (
            DeliveryOutcome::NotSent,
            refusal.reason.to_owned(),
            error("refused", format!("{}: {}", refusal.reason, refusal.detail)),
        ),
    };
    outcome_event(d, invocation, request_id, what, detail)?;
    Ok(answer)
}

/// Runs a resumed run on its own thread, then any queued behind it whose
/// delivery no hook has claimed meanwhile.
pub(crate) fn run_resume(acts: &crate::live_acts::Acts, run: ResumeRun) {
    let Some(daemon) = acts.daemon() else {
        let _ = acts.resume_finished(&run.session);
        return;
    };
    let session = run.session.clone();
    let delivery = run.delivery.clone();
    let spawned = std::thread::Builder::new()
        .name("polylogue-resume".into())
        .spawn(move || {
            let mut next = Some(run);
            while let Some(run) = next.take() {
                resume_once(&daemon, &run);
                next = daemon.acts.resume_finished(&run.session);
                while let Some(candidate) = next.take() {
                    // A queued ping waits for its session to stop running:
                    // if it runs again (resumed by hand), the ping goes to
                    // its next hook instead of a second resume.
                    let running = daemon
                        .reads
                        .reader()
                        .ok()
                        .and_then(|r| {
                            r.live_state(
                                &candidate.session.to_string(),
                                crate::daemon::service::now_ms(),
                            )
                            .ok()
                            .flatten()
                        })
                        .is_some_and(|s| s.view.state.running());
                    if !running {
                        next = Some(candidate);
                        break;
                    }
                    let _ = crate::mailbox::raise(daemon.acts.mail(), &candidate.session);
                    daemon.acts.release(&candidate.delivery);
                    next = daemon.acts.resume_finished(&candidate.session);
                }
            }
        });
    if spawned.is_err() {
        // Nothing runs: the pings go back to the mailbox, where they are
        // settled like any other.
        acts.release(&delivery);
        while let Some(queued) = acts.resume_finished(&session) {
            acts.release(&queued.delivery);
        }
    }
}

/// One headless resume: `delivered` when the harness starts with the ping
/// as its prompt, `failed` with the first line of its error output when it
/// exits unsuccessfully.
fn resume_once(d: &Daemon, run: &ResumeRun) {
    let invocation = d.invocation(&crate::operation::Context::worker_authorship("resume"));
    let record = |what, detail: String| {
        let _ = outcome_event(d, &invocation, &run.delivery, what, detail);
    };
    let Some(argv) = crate::live_acts::headless_resume(&run.session, &run.text) else {
        record(DeliveryOutcome::NotSent, "unsupported".into());
        return;
    };
    let mut command = std::process::Command::new(&argv[0]);
    command
        .args(&argv[1..])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped());
    if let Some(cwd) = run
        .cwd
        .as_deref()
        .filter(|c| std::path::Path::new(c).is_dir())
    {
        command.current_dir(cwd);
    }
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
    /// source, the ping.
    refs: Vec<String>,
}

/// Starts a hosted session on the harness's control protocol, its first
/// turn opening with the delivery reference, and records that turn as a
/// `pack` delivery to the new session. The channel is
/// [`polylogue_model::live::channel::host_channel`]'s.
fn start_session(
    d: &Daemon,
    invocation: &Invocation,
    new: NewSession<'_>,
) -> Result<Result<SessionId, Answer<Value>>, Failure> {
    use polylogue_model::live::channel::{HostProtocol, host_channel};
    let protocol = match host_channel(new.harness) {
        Ok(protocol) => protocol,
        Err(refusal) => {
            return Ok(Err(error(
                "refused",
                format!("{}: {}", refusal.reason, refusal.detail),
            )));
        }
    };
    if !new.cwd.is_dir() {
        return Ok(Err(error(
            "bad_value",
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
    .map_err(|e| Failure::new("refused", format!("{}: {e}", protocol.as_str())))?;
    author(
        d,
        invocation,
        new.request_id.to_owned(),
        UserWrite::Deliver(Delivery {
            recipient_kind: "session".into(),
            recipient_id: Some(session.to_string()),
            sender: DeliverySender::Act.as_str().into(),
            event: DeliveryEvent::Pack.as_str().into(),
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
        "You run as the role `{}` (assertion:{}):\n{}\n\n{}",
        role.name, role.assertion, role.instructions, text
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
                "bad_value",
                format!(
                    "the role {} runs {}, not {harness}",
                    role.name, role.harness
                ),
            ));
        }
        (Some(harness), _) => harness.to_owned(),
        (None, Some(role)) => role.harness.clone(),
        (None, None) => return Ok(error("bad_request", "start takes --harness or --role")),
    };
    let (text, refs) = match &role {
        Some(role) => (
            role_text(role, text),
            vec![
                role_reference(&role.name),
                format!("assertion:{}", role.assertion),
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

/// A ping to a role with no running instance starts one, in the directory
/// its latest instance ran in, with the ping as its first turn.
fn ping_new_instance(
    d: &Daemon,
    invocation: &Invocation,
    request_id: &str,
    role: &Role,
    text: &str,
    about: Option<&str>,
) -> Result<Answer<Value>, Failure> {
    let Some(cwd) = role_cwd(d, &role.name)? else {
        return Ok(error(
            "refused",
            format!(
                "no_live_instance: the role {} has no running session and none ran before to take a directory from; start one with `polylogue start --role {} --cwd DIR TEXT`",
                role.name, role.name
            ),
        ));
    };
    let author_name = invocation.authorship.author.to_string();
    let mut pack = format!("Ping from {author_name}:\n{text}\n");
    if let Some(about) = about {
        pack.push_str(&format!("About: {about}\n"));
    }
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
                format!("assertion:{}", role.assertion),
            ],
        },
    )? {
        Ok(session) => session,
        Err(refused) => return Ok(refused),
    };
    Ok(ok(json!({
        "deliveries": [{"session": session, "delivery": request_id, "channel": "protocol",
                        "outcome": "delivered", "started": role.name}],
        "undo": "none: a sent ping cannot be taken back",
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
        return Ok(error("not_found", format!("no session {}", h.source)));
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
            .map_err(|e| Failure::new("bad_value", e))?;
    let operation = d.resolve_operation_context(invocation.clone())?;
    let prepared = crate::briefing::prepare(
        &reader,
        &d.scope,
        None,
        &crate::briefing::Ask::Takeover {
            session: h.source.clone(),
            for_session: None,
        },
        h.request_id,
        &operation,
        None,
        None,
        None,
    )?;
    drop(reader);
    let mut text = format!("Hand-off from {}", h.source);
    if let Some(at) = &at {
        text.push_str(&format!(", continuing from {at}"));
    }
    text.push_str(":\n");
    text.push_str(&prepared.text);
    let mut refs = vec![format!("session:{}", h.source)];
    refs.extend(at.as_ref().map(ToString::to_string));
    refs.extend(prepared.refs.iter().cloned());
    if h.print {
        return Ok(ok(json!({
            "to": h.to,
            "cwd": cwd,
            "channel": polylogue_model::live::channel::host_channel(h.to)
                .map(|p| p.as_str())
                .map_err(|r| r.reason),
            "pack": text,
            "refs": refs,
        })));
    }
    let Some(cwd) = cwd else {
        return Ok(error(
            "refused",
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

/// A signal rule's notice to the sessions `to` addresses (a session, a
/// role's running instance, or every session a quoted query selects now):
/// one delivery each, sender `signal`, event `notice`, reaching a running
/// session through its channel. A notice never starts or resumes a
/// session: to one that is not running, and to a role with no running
/// instance, it records `not_sent`.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the signal evaluator calls it once authored rules fire; they are held until the signals adoption lands"
    )
)]
pub(crate) fn notice(d: &Daemon, rule: &str, to: &str, text: &str) -> Result<Vec<Value>, Failure> {
    let invocation = d.invocation(&crate::operation::Context::worker_authorship(
        "polylogue-signals",
    ));
    let now = now(d);
    let sessions = match recipients(d, &invocation, to, now)? {
        Recipients::Sessions(sessions) => sessions,
        Recipients::NewInstance(role) => {
            return Ok(vec![json!({"role": role.name, "outcome": "not_sent",
                "reason": "session_not_live", "detail": "a notice never starts a session"})]);
        }
    };
    let base = polylogue_store::user_writes::new_request_id();
    let mut sent = Vec::new();
    for (n, session) in sessions.iter().enumerate() {
        let delivery = format!("{base}-{n}");
        let Some((state, facts, cwd)) = session_state(d, session, now)? else {
            sent.push(
                json!({"session": session, "outcome": "not_sent", "reason": "session_missing"}),
            );
            continue;
        };
        let pack = format!("Notice from the rule {rule} (delivery:{delivery}):\n{text}");
        {
            let _claims = d.acts.claim_lock();
            d.acts.take(vec![(delivery.clone(), ())]);
        }
        author(
            d,
            &invocation,
            delivery.clone(),
            UserWrite::Deliver(Delivery {
                recipient_kind: "session".into(),
                recipient_id: Some(session.to_string()),
                sender: DeliverySender::Signal.as_str().into(),
                event: DeliveryEvent::Notice.as_str().into(),
                cwd,
                text: pack.clone(),
                refs: vec![format!("signal:{rule}")],
                left_out: Vec::new(),
            }),
        )?;
        let row = match channel(Act::Notice, state, &probe(&d.acts, session, &facts)) {
            Ok(Channel::Protocol) => protocol_send(d, &invocation, session, &delivery, &pack)?,
            Ok(_) => {
                crate::mailbox::raise(d.acts.mail(), session).map_err(|e| Failure::new("io", e))?;
                d.acts.release(&delivery);
                json!({"session": session, "delivery": delivery, "channel": "hook", "outcome": "prepared"})
            }
            Err(refusal) => {
                outcome_event(
                    d,
                    &invocation,
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
    Ok(sent)
}
