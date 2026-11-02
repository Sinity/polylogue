//! Live-session acts: pings (one session, or every session a query selects
//! when sent), the mailbox claim a hook makes, answers to permission
//! requests, and interrupts. Each act is recorded as a delivery with its
//! computed author, and reaches its session only through the channel
//! [`polylogue_model::live::channel::channel`] chooses.
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
        } => answer(d, invocation, &request_id, &session, approve, text, standing),
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
            cwd,
            text,
        } => start(d, invocation, &request_id, &harness, &cwd, &text),
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

/// The sessions an address names when the ping is sent: a session
/// reference, or the sessions a quoted query selects then (a broadcast,
/// frozen: sessions that start matching later receive nothing).
fn recipients(d: &Daemon, invocation: &Invocation, target: &str, now: i64) -> Result<Vec<SessionId>, Failure> {
    let address = target
        .trim()
        .strip_prefix("@[")
        .and_then(|rest| rest.strip_suffix(']'))
        .unwrap_or(target.trim());
    if let Ok(session) = SessionId::parse(address) {
        return Ok(vec![session]);
    }
    if address.starts_with("role:") {
        return Err(Failure::new(
            "refused",
            "a role address needs the session's role record, which no session carries yet",
        ));
    }
    let operation = d.resolve_operation_context(invocation.clone())?;
    let reader = d.reads.reader()?;
    super::follow::selected(&reader, &operation, address, now)?
        .iter()
        .map(|id| SessionId::parse(id).map_err(|e| Failure::new("internal", e)))
        .collect()
}

/// The ping pack: the text, what it is about, how to answer, and the
/// delivery reference.
fn ping_text(
    delivery: &str,
    ping: &str,
    author: &str,
    text: &str,
    about: Option<&str>,
) -> String {
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
    let sessions = recipients(d, invocation, target, now)?;
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
        value: Some(json!({"text": text, "subject": subject.as_ref().map(ToString::to_string), "reply_to": null})),
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
        let state = d.reads.reader()?.live_state(&session.to_string(), now)?;
        let (view_state, facts, cwd) = match state {
            Some(s) => (s.view.state, s.facts, s.cwd),
            None => {
                sent.push(json!({"session": session, "outcome": "not_sent", "reason": "session_missing"}));
                continue;
            }
        };
        let pack = ping_text(&delivery, &ping, &author_name, text, about.as_deref());
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
        let chosen = channel(Act::Ping { live }, view_state, &probe(&d.acts, session, &facts));
        let row = match chosen {
            Ok(Channel::Hook) => {
                crate::mailbox::raise(d.acts.mail(), session).map_err(|e| Failure::new("io", e))?;
                json!({"session": session, "delivery": delivery, "channel": "hook",
                       "outcome": "prepared", "note": "reaches the session at its next tool result or prompt"})
            }
            Ok(Channel::Resume) => {
                // Handed to the resumed run: no hook takes it.
                {
                    let _claims = d.acts.claim_lock();
                    d.acts.take(vec![(delivery.clone(), ())]);
                }
                let started = d.acts.resume(ResumeRun {
                    session: session.clone(),
                    delivery: delivery.clone(),
                    text: pack,
                    cwd,
                });
                json!({"session": session, "delivery": delivery, "channel": "resume",
                       "outcome": if started { "started" } else { "queued" }})
            }
            Ok(Channel::Protocol) => match d.acts.host_send(session, &pack) {
                Ok(()) => {
                    outcome_event(d, invocation, &delivery, DeliveryOutcome::Delivered, "sent over the hosted connection".into())?;
                    json!({"session": session, "delivery": delivery, "channel": "protocol", "outcome": "delivered"})
                }
                Err(reason) => {
                    outcome_event(d, invocation, &delivery, DeliveryOutcome::NotSent, format!("failed({reason})"))?;
                    json!({"session": session, "delivery": delivery, "channel": "protocol", "outcome": "not_sent", "reason": reason})
                }
            },
            Err(refusal) => {
                outcome_event(d, invocation, &delivery, DeliveryOutcome::NotSent, refusal.reason.to_owned())?;
                json!({"session": session, "delivery": delivery, "outcome": "not_sent",
                       "reason": refusal.reason, "detail": refusal.detail})
            }
        };
        sent.push(row);
    }
    Ok(ok(json!({"ping": ping, "deliveries": sent, "undo": "none: a sent ping cannot be taken back"})))
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
pub(crate) fn settle_ended(d: &Daemon, now: i64) {
    let invocation = d.invocation(&crate::operation::Context::worker_authorship("mailbox"));
    let Ok(reader) = d.reads.reader() else {
        return;
    };
    let sessions = reader.sessions_with_pending_hook_deliveries().unwrap_or_default();
    let mut ended = Vec::new();
    for session in sessions {
        let Ok(id) = SessionId::parse(&session) else {
            continue;
        };
        let running = reader
            .live_state(&session, now)
            .ok()
            .flatten()
            .is_some_and(|s| s.view.state.running());
        if running {
            let _ = crate::mailbox::raise(d.acts.mail(), &id);
            continue;
        }
        let pending = reader.pending_hook_deliveries(&session).unwrap_or_default();
        ended.push((id, pending));
    }
    drop(reader);
    for (session, pending) in ended {
        let _claims = d.acts.claim_lock();
        for (id, _) in d.acts.take(pending) {
            let _ = outcome_event(
                d,
                &invocation,
                &id,
                DeliveryOutcome::NotSent,
                "session_ended".into(),
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
    let chosen = match channel(Act::Answer, state.view.state, &probe(&d.acts, session, &state.facts)) {
        Ok(chosen) => chosen,
        Err(refusal) => return Ok(error("refused", format!("{}: {}", refusal.reason, refusal.detail))),
    };
    // A hosted session's requests come to Polylogue's connection; an
    // observed one's from its permission hook, into its live facts.
    let (reference, tool, input) = if chosen == Channel::Protocol {
        let Some(request) = d.acts.host_request(session) else {
            return Ok(error("refused", "the hosted session has no pending request to answer"));
        };
        (format!("request:{session}#{request}"), Value::Null, Value::Null)
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
            serde_json::from_str::<Value>(body).is_ok_and(|b| {
                b["request"] == reference && b["standing"] == standing
            })
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
            format!("already_answered: {reference} was answered by delivery:{}", first.id),
        ));
    }
    match chosen {
        Channel::Protocol => {
            let result = d.acts.host_answer(session, approve, text.as_deref());
            let (what, detail) = match &result {
                Ok(()) => (DeliveryOutcome::Delivered, "answered over the hosted connection".to_owned()),
                Err(reason) => (DeliveryOutcome::NotSent, format!("failed({reason})")),
            };
            outcome_event(d, invocation, request_id, what, detail)?;
            Ok(ok(json!({"delivery": request_id, "request": reference, "channel": "protocol",
                         "outcome": what.as_str(), "undo": "none: an answer cannot be taken back"})))
        }
        _ => Ok(ok(json!({"delivery": request_id, "request": reference, "channel": "hook",
                          "outcome": "prepared", "note": "applies when the session next asks for the same tool and input",
                          "undo": "none: an answer cannot be taken back"}))),
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
    let (what, detail, answer) =
        match channel(Act::Interrupt, state.view.state, &probe(&d.acts, session, &state.facts)) {
            Ok(_) => match d.acts.host_interrupt(session) {
                Ok(()) => (
                    DeliveryOutcome::Delivered,
                    "interrupted over the hosted connection".to_owned(),
                    ok(json!({"session": session, "delivery": request_id, "channel": "protocol",
                              "outcome": "delivered", "undo": "none: an interrupted turn is not resumed"})),
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
    let spawned = std::thread::Builder::new()
        .name("polylogue-resume".into())
        .spawn(move || {
            let mut next = Some(run);
            while let Some(run) = next.take() {
                resume_once(&daemon, &run);
                next = daemon.acts.resume_finished(&run.session);
                while let Some(candidate) = next.take() {
                    // A queued ping still pending is the run's; one a hook
                    // or an ending took meanwhile is not run.
                    let claims = daemon.acts.claim_lock();
                    let still_pending = daemon
                        .reads
                        .reader()
                        .and_then(|r| r.pending_hook_deliveries(&candidate.session.to_string()))
                        .is_ok_and(|p| p.iter().any(|(id, _)| *id == candidate.delivery));
                    if still_pending {
                        next = Some(candidate);
                        break;
                    }
                    drop(claims);
                    next = daemon.acts.resume_finished(&candidate.session);
                }
            }
        });
    if spawned.is_err() {
        // Nothing runs; the deliveries stay prepared for the session's hook.
        while acts.resume_finished(&session).is_some() {}
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
    if let Some(cwd) = run.cwd.as_deref().filter(|c| std::path::Path::new(c).is_dir()) {
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

/// Starts a session Polylogue hosts: a Codex thread on the shared
/// app-server, whose approvals and questions come to Polylogue's
/// connection and whose turns Polylogue can steer and interrupt. The first
/// turn's text is recorded as a delivery to the new session.
fn start(
    d: &Daemon,
    invocation: &Invocation,
    request_id: &str,
    harness: &str,
    cwd: &std::path::Path,
    text: &str,
) -> Result<Answer<Value>, Failure> {
    if harness != "codex" {
        return Ok(error(
            "refused",
            format!("unsupported: Polylogue hosts sessions of codex only; {harness} has no control protocol client here"),
        ));
    }
    if !cwd.is_dir() {
        return Ok(error("bad_value", format!("{} is not a directory", cwd.display())));
    }
    let hosted = crate::live_acts::HostedCodex::start(
        &d.acts.codex_program,
        &d.acts.codex_socket,
        &cwd.to_string_lossy(),
        text,
    )
    .map_err(|e| Failure::new("refused", format!("codex app-server: {e}")))?;
    let session = d
        .acts
        .host(hosted)
        .ok_or_else(|| Failure::new("internal", "the app-server named an unusable thread"))?;
    author(
        d,
        invocation,
        request_id.to_owned(),
        UserWrite::Deliver(Delivery {
            recipient_kind: "session".into(),
            recipient_id: Some(session.to_string()),
            sender: DeliverySender::Act.as_str().into(),
            event: DeliveryEvent::Control.as_str().into(),
            cwd: Some(cwd.to_string_lossy().into_owned()),
            text: text.to_owned(),
            refs: Vec::new(),
            left_out: Vec::new(),
        }),
    )?;
    outcome_event(
        d,
        invocation,
        request_id,
        DeliveryOutcome::Delivered,
        "the first turn of a hosted session".into(),
    )?;
    Ok(ok(json!({"session": session, "delivery": request_id, "channel": "protocol", "hosting": "hosted"})))
}
