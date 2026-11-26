//! The acts on live sessions: `send`, `answer` and `interrupt`, through the
//! acts' handlers; a web-chat session's turns go through
//! [`acts::web_chat`]. Each recipient's act is its own request.
use super::{
    RowRule, Selection, StepCall, StepDone, Verb, delegated, not_built, not_built_use, sessions,
};
use crate::daemon::Failure;
use crate::daemon::handler::{Call, Handler};
use crate::daemon::service::acts;
use crate::protocol::act;
use crate::protocol::line::{SendReport, Skipped, StepAnswer};
use crate::protocol::steps::{AnswerStep, InterruptStep, SendStep};
use command_surface::Envelope;
use polylogue_model::live::channel::{Act, Channel, channel};
use polylogue_model::{Reference, SessionId};
use serde_json::Value;

#[cfg(test)]
mod tests;

pub(crate) struct Send;
impl Verb for Send {
    type Step = SendStep;
    const NAME: &'static str = "send";
    const ROWS: RowRule = RowRule::Things;
    fn run(
        step: SendStep,
        selection: &Selection,
        call: &StepCall<'_>,
    ) -> Result<StepDone, Failure> {
        let targets = recipients(&step, selection)?;
        let several = targets.len() > 1;
        let mut report = SendReport {
            reached: Vec::new(),
            skipped: Vec::new(),
        };
        let mut envelope = Envelope::default();
        let mut outcome = command_surface::Outcome::Empty(());
        for (n, target) in targets.into_iter().enumerate() {
            let request_id = format!("{}-{}", call.call.request_id, n + 1);
            let child = Call {
                request_id: &request_id,
                ..call.call.copy()
            };
            let one = if several { &child } else { &call.call };
            let web_chat = SessionId::parse(&target)
                .ok()
                .filter(|session| acts::web_chat::conversation(session).is_some());
            if let Some(session) = web_chat {
                let done = delegated(
                    acts::web_chat::send(one, &session, &step.text, &step.attach)?,
                    |reached| {
                        StepAnswer::Sent(SendReport {
                            reached: vec![reached],
                            skipped: Vec::new(),
                        })
                    },
                )?;
                super::merge_outcome(&mut outcome, done.outcome);
                envelope.notes.extend(done.envelope.notes);
                envelope.next.extend(done.envelope.next);
                if let StepAnswer::Sent(sent) = done.reply {
                    report.reached.extend(sent.reached);
                }
                continue;
            }
            if !step.attach.is_empty() {
                return Err(not_built("send", "attach to a harness session"));
            }
            if several {
                // The same notion of running as `would`: a session just
                // started and still hosted has no live events yet.
                let session = SessionId::parse(&target).map_err(Failure::bad_value)?;
                let running = acts::session_state(
                    call.call.daemon,
                    &session,
                    call.call.daemon.config.requests.now().unix_ms(),
                )?
                .is_some_and(|(state, _, _)| state.running());
                if !running {
                    report.skipped.push(Skipped {
                        session: target.clone(),
                        line: format!(
                            "polylogue {} send {}",
                            target,
                            polylogue_render::shell_quote(&step.text)
                        ),
                    });
                    continue;
                }
            }
            let done = delegated(
                acts::SendMessage::run(
                    act::SendMessage {
                        target,
                        text: step.text.clone(),
                        live: several,
                        about: step.about.clone(),
                    },
                    one,
                )?,
                |reached| {
                    StepAnswer::Sent(SendReport {
                        reached: vec![reached],
                        skipped: Vec::new(),
                    })
                },
            )?;
            super::merge_outcome(&mut outcome, done.outcome);
            envelope.notes.extend(done.envelope.notes);
            envelope.next.extend(done.envelope.next);
            if let StepAnswer::Sent(sent) = done.reply {
                report.reached.extend(sent.reached);
            }
        }
        Ok(StepDone {
            reply: StepAnswer::Sent(report),
            undo: Vec::new(),
            envelope,
            outcome,
        })
    }
    fn would(
        step: &SendStep,
        selection: &Selection,
        call: &StepCall<'_>,
    ) -> Result<String, Failure> {
        let targets = recipients(step, selection)?;
        let several = targets.len() > 1;
        targets
            .iter()
            .map(|target| {
                let d = call.call.daemon;
                let now = d.config.requests.now().unix_ms();
                let session = if let Some(name) = target.strip_prefix("role:") {
                    acts::role(d, call.call.invocation, name)?;
                    match acts::role_instance(d, name, now)? {
                        Some(session) => session,
                        None => {
                            let delivery = if acts::role_cwd(d, name)?.is_some() {
                                "hosted".to_owned()
                            } else {
                                format!(
                                    "refused: no_live_instance: the role {name} has no running session and none ran before"
                                )
                            };
                            return Ok(format!(
                                "would send {:?} to {target}: {delivery}",
                                step.text
                            ));
                        }
                    }
                } else {
                    SessionId::parse(target).map_err(Failure::bad_value)?
                };
                if acts::web_chat::conversation(&session).is_some() {
                    return Ok(format!(
                        "would send {:?} to {target}: the browser extension, held for the operator's answer",
                        step.text
                    ));
                }
                let state = acts::session_state(d, &session, now)?;
                let delivery = match state {
                    Some((state, _, _)) if several && !state.running() => {
                        format!("skipped: {}", state.as_str())
                    }
                    Some((state, facts, _)) => match channel(
                        Act::Send { live: several },
                        state,
                        &crate::live_acts::probe(&d.acts, &session, &facts),
                    ) {
                        Ok(Channel::Protocol) => "hosted".to_owned(),
                        Ok(Channel::Hook) => "mailbox".to_owned(),
                        Ok(Channel::Resume) => "resumed headlessly".to_owned(),
                        Err(reason) => format!("refused: {}: {}", reason.reason, reason.detail),
                    },
                    None if several => "skipped: ended".to_owned(),
                    None => "refused: session missing".to_owned(),
                };
                Ok(format!(
                    "would send {:?} to {target}: {delivery}",
                    step.text
                ))
            })
            .collect::<Result<Vec<_>, Failure>>()
            .map(|lines| lines.join("\n"))
    }
}

/// Sessions or one role, preserving its send-time resolution.
fn recipients(step: &SendStep, selection: &Selection) -> Result<Vec<String>, Failure> {
    for (given, option) in [
        (step.page.is_some(), "page"),
        (step.section.is_some(), "section"),
    ] {
        if given {
            return Err(not_built("send", option));
        }
    }
    let references = selection.references()?;
    references
        .iter()
        .map(|reference| match reference {
            Reference::Session(session) => Ok(session.to_string()),
            Reference::Role(name) if references.len() == 1 => Ok(format!("role:{name}")),
            other => Err(Failure::bad_request(format!(
                "send acts on sessions; {other} is not one"
            ))),
        })
        .collect()
}

pub(crate) struct Answer;
impl Verb for Answer {
    type Step = AnswerStep;
    const NAME: &'static str = "answer";
    const ROWS: RowRule = RowRule::Things;
    fn run(
        step: AnswerStep,
        selection: &Selection,
        call: &StepCall<'_>,
    ) -> Result<StepDone, Failure> {
        let decision = decision(&step)?;
        each_session(Self::NAME, selection, call, |session, call| {
            if acts::web_chat::conversation(&session).is_some() {
                return acts::web_chat::answer(call, &session, &step.value, step.reason.clone());
            }
            acts::AnswerRequest::run(
                act::Answer {
                    session,
                    decision,
                    text: step.reason.clone(),
                    standing: step.standing,
                },
                call,
            )
        })
    }
    fn would(
        step: &AnswerStep,
        selection: &Selection,
        call: &StepCall<'_>,
    ) -> Result<String, Failure> {
        decision(step)?;
        sessions("answer", selection)?
            .iter()
            .map(|session| {
                if acts::web_chat::conversation(session).is_some() {
                    return Ok(format!(
                        "would answer {} for {session}: its held web chat turn",
                        step.value
                    ));
                }
                let state = acts::session_state(
                    call.call.daemon,
                    session,
                    call.call.daemon.config.requests.now().unix_ms(),
                )?;
                let result = match state {
                    Some((state, facts, _)) => {
                        let waiting = call.call.daemon.acts.host_request(session).is_some()
                            || facts.pending_request.is_some();
                        match channel(
                            Act::Answer,
                            state,
                            &crate::live_acts::probe(&call.call.daemon.acts, session, &facts),
                        ) {
                            Ok(Channel::Hook) if !step.standing => "refused: the harness already shows its own prompt for this request; answer it in the terminal or use --standing".to_owned(),
                            Ok(_) if waiting => "waiting on a permission".to_owned(),
                            Ok(_) => "not waiting on a permission".to_owned(),
                            Err(reason) => {
                                format!("refused: {}: {}", reason.reason, reason.detail)
                            }
                        }
                    }
                    None => "refused: session missing".to_owned(),
                };
                Ok(format!("would answer {} for {session}: {result}", step.value))
            })
            .collect::<Result<Vec<_>, Failure>>()
            .map(|lines| lines.join("\n"))
    }
}

fn decision(step: &AnswerStep) -> Result<act::Decision, Failure> {
    match step.value.as_str() {
        "allow" => Ok(act::Decision::Approve),
        "deny" => Ok(act::Decision::Deny),
        _ => Err(not_built_use("answer", "with text")),
    }
}

pub(crate) struct Interrupt;
impl Verb for Interrupt {
    type Step = InterruptStep;
    const NAME: &'static str = "interrupt";
    const ROWS: RowRule = RowRule::Things;
    fn run(
        _: InterruptStep,
        selection: &Selection,
        call: &StepCall<'_>,
    ) -> Result<StepDone, Failure> {
        each_session(Self::NAME, selection, call, |session, call| {
            acts::Interrupt::run(act::Interrupt { session }, call)
        })
    }
    fn would(
        _: &InterruptStep,
        selection: &Selection,
        call: &StepCall<'_>,
    ) -> Result<String, Failure> {
        sessions("interrupt", selection)?
            .iter()
            .map(|session| {
                let state = acts::session_state(
                    call.call.daemon,
                    session,
                    call.call.daemon.config.requests.now().unix_ms(),
                )?;
                let result = match state {
                    Some((state, facts, _)) => match channel(
                        Act::Interrupt,
                        state,
                        &crate::live_acts::probe(&call.call.daemon.acts, session, &facts),
                    ) {
                        Ok(_) => "hosted".to_owned(),
                        Err(reason) => format!("refused: {}: {}", reason.reason, reason.detail),
                    },
                    None => "refused: session missing".to_owned(),
                };
                Ok(format!("would interrupt {session}: {result}"))
            })
            .collect::<Result<Vec<_>, Failure>>()
            .map(|lines| lines.join("\n"))
    }
}

/// One act per session, request `<step>-<n>`; the first failure stops the step.
fn each_session(
    verb: &str,
    selection: &Selection,
    call: &StepCall<'_>,
    one: impl Fn(SessionId, &Call<'_>) -> Result<command_surface::Answer<Value>, Failure>,
) -> Result<StepDone, Failure> {
    let mut acted = Vec::new();
    let mut envelope = Envelope::default();
    let mut outcome = command_surface::Outcome::Empty(());
    for (n, session) in sessions(verb, selection)?.into_iter().enumerate() {
        let request_id = format!("{}-{}", call.call.request_id, n + 1);
        let done = delegated(
            one(
                session,
                &Call {
                    request_id: &request_id,
                    ..call.call.copy()
                },
            )?,
            |value| StepAnswer::Acted(vec![value]),
        )?;
        super::merge_outcome(&mut outcome, done.outcome);
        envelope.notes.extend(done.envelope.notes);
        envelope.next.extend(done.envelope.next);
        if let StepAnswer::Acted(values) = done.reply {
            acted.extend(values);
        }
    }
    Ok(StepDone {
        reply: StepAnswer::Acted(acted),
        undo: Vec::new(),
        envelope,
        outcome,
    })
}
