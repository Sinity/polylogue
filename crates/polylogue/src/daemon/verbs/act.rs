//! The acts on live sessions: `send`, `answer` and `interrupt`. A web-chat
//! session's turns go through [`acts::web_chat`]. Each recipient's act is
//! its own request, `<step>-<n>` when the step has several.
use super::{RowRule, Selection, StepCall, StepDone, Verb, sessions};
use crate::commands::ToolAccess;
use crate::daemon::Failure;
use crate::daemon::handler::Call;
use crate::daemon::service::acts;
use crate::protocol::acts::{Acted, Reached, SendReport, SkipReason, Skipped};
use crate::protocol::line::StepAnswer;
use crate::protocol::steps::{AnswerStep, InterruptStep, SendStep};
use command_surface::{Answer as Answered, Envelope, Outcome};
use polylogue_model::live::channel::{Act, Channel, channel};
use polylogue_model::{Reason, Reference, SessionId};

#[cfg(test)]
mod tests;

pub(crate) struct Send;
impl Verb for Send {
    type Step = SendStep;
    const NAME: &'static str = "send";
    const ROWS: RowRule = RowRule::Things;
    const ACCESS: ToolAccess = ToolAccess::Act;
    fn run(
        step: SendStep,
        selection: &Selection,
        call: &StepCall<'_>,
    ) -> Result<StepDone, Failure> {
        let targets = recipients(selection)?;
        send(&step, &targets, &call.call).map(|answer| answer.map(StepAnswer::Sent).into())
    }
    fn would(
        step: &SendStep,
        selection: &Selection,
        call: &StepCall<'_>,
    ) -> Result<String, Failure> {
        let targets = recipients(selection)?;
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
                    session_id(target)?
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

/// A session reference as an act takes it.
fn session_id(text: &str) -> Result<SessionId, Failure> {
    SessionId::parse(text).map_err(|problem| {
        Failure::refusal(Reason::BadReference {
            text: text.into(),
            problem: problem.to_string(),
        })
    })
}

/// `send TEXT` to each target: one session or role alone is sent to
/// whatever its state; of several, the running ones are, and the others are
/// skipped with the line that sends to each alone, since resuming spends.
fn send(
    step: &SendStep,
    targets: &[String],
    call: &Call<'_>,
) -> Result<Answered<SendReport>, Failure> {
    let several = targets.len() > 1;
    let mut report = SendReport {
        reached: Vec::new(),
        skipped: Vec::new(),
    };
    let mut envelope = Envelope::default();
    let mut outcome = Outcome::Empty(());
    for (n, target) in targets.iter().enumerate() {
        let request_id = format!("{}-{}", call.request_id, n + 1);
        let child = Call {
            request_id: &request_id,
            ..call.copy()
        };
        let one = if several { &child } else { call };
        let web_chat = SessionId::parse(target)
            .ok()
            .filter(|session| acts::web_chat::conversation(session).is_some());
        let reached = if let Some(session) = web_chat {
            acts::web_chat::send(one, &session, &step.text, &step.attach)?.map(Reached::WebChat)
        } else {
            if !step.attach.is_empty() {
                return Err(
                    crate::unbuilt::refusal("send", "--attach to a harness session").into(),
                );
            }
            if several {
                // The same notion of running as `would`: a session just
                // started and still hosted has no live events yet.
                let session = session_id(target)?;
                let state = acts::session_state(
                    call.daemon,
                    &session,
                    call.daemon.config.requests.now().unix_ms(),
                )?;
                let reason = match state {
                    Some((state, _, _)) if state.running() => None,
                    Some((state, _, _)) => Some(SkipReason::SessionNotRunning { state }),
                    None => Some(SkipReason::SessionMissing),
                };
                if let Some(reason) = reason {
                    report.skipped.push(Skipped {
                        session: target.clone(),
                        reason,
                        line: format!(
                            "polylogue {} send {}",
                            target,
                            polylogue_render::shell_quote(&step.text)
                        ),
                    });
                    continue;
                }
            }
            acts::send(
                one.daemon,
                one.invocation,
                one.request_id,
                target,
                &step.text,
                several,
                step.about.clone(),
            )?
            .map(Reached::Session)
        };
        report
            .reached
            .push(super::gather(reached, &mut outcome, &mut envelope)?);
    }
    use crate::protocol::acts::ActOutcome;
    let unconfirmed = report
        .reached
        .iter()
        .filter_map(|reached| match reached {
            Reached::Session(sent) => Some(&sent.deliveries),
            Reached::WebChat(_) => None,
        })
        .flatten()
        .filter(|delivery| matches!(delivery.outcome, ActOutcome::NotSent | ActOutcome::Unknown));
    let mut missing: Vec<_> = unconfirmed
        .map(|delivery| format!("message to {}", delivery.session))
        .collect();
    missing.extend(
        report
            .skipped
            .iter()
            .map(|skipped| format!("message to {}", skipped.session)),
    );
    if !missing.is_empty() {
        super::merge_outcome(&mut outcome, Outcome::Degraded { value: (), missing });
    }
    Ok(Answered {
        outcome: outcome.map(|()| report),
        envelope,
    })
}

/// Sessions or one role, preserving its send-time resolution.
fn recipients(selection: &Selection) -> Result<Vec<String>, Failure> {
    let references = selection.references()?;
    references
        .iter()
        .map(|reference| match reference {
            Reference::Session(session) => Ok(session.to_string()),
            Reference::Role(name) if references.len() == 1 => Ok(format!("role:{name}")),
            other => Err(Failure::refusal(Reason::ActsOnSessions {
                verb: "send".into(),
                reference: other.to_string(),
            })),
        })
        .collect()
}

pub(crate) struct Answer;
impl Verb for Answer {
    type Step = AnswerStep;
    const NAME: &'static str = "answer";
    const ROWS: RowRule = RowRule::Things;
    const ACCESS: ToolAccess = ToolAccess::Act;
    fn run(
        step: AnswerStep,
        selection: &Selection,
        call: &StepCall<'_>,
    ) -> Result<StepDone, Failure> {
        let approve = approves(&step)?;
        each_session(Self::NAME, selection, call, |session, call| {
            if acts::web_chat::conversation(&session).is_some() {
                return Ok(acts::web_chat::answer(
                    call,
                    &session,
                    &step.value,
                    step.reason.clone(),
                )?
                .map(Acted::WebChat));
            }
            Ok(acts::answer(
                call.daemon,
                call.invocation,
                call.request_id,
                &session,
                approve,
                step.reason.clone(),
                step.standing,
            )?
            .map(Acted::Answered))
        })
    }
    fn would(
        step: &AnswerStep,
        selection: &Selection,
        call: &StepCall<'_>,
    ) -> Result<String, Failure> {
        approves(step)?;
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

/// `allow` approves and `deny` refuses; an answer in words is not built.
fn approves(step: &AnswerStep) -> Result<bool, Failure> {
    match step.value.as_str() {
        "allow" => Ok(true),
        "deny" => Ok(false),
        _ => Err(crate::unbuilt::refusal("answer", "with text").into()),
    }
}

pub(crate) struct Interrupt;
impl Verb for Interrupt {
    type Step = InterruptStep;
    const NAME: &'static str = "interrupt";
    const ROWS: RowRule = RowRule::Things;
    const ACCESS: ToolAccess = ToolAccess::Act;
    fn run(
        _: InterruptStep,
        selection: &Selection,
        call: &StepCall<'_>,
    ) -> Result<StepDone, Failure> {
        each_session(Self::NAME, selection, call, |session, call| {
            Ok(
                acts::interrupt(call.daemon, call.invocation, call.request_id, &session)?
                    .map(Acted::Interrupted),
            )
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
    one: impl Fn(SessionId, &Call<'_>) -> Result<Answered<Acted>, Failure>,
) -> Result<StepDone, Failure> {
    let mut acted = Vec::new();
    let mut envelope = Envelope::default();
    let mut outcome = Outcome::Empty(());
    for (n, session) in sessions(verb, selection)?.into_iter().enumerate() {
        let request_id = format!("{}-{}", call.call.request_id, n + 1);
        let answer = one(
            session,
            &Call {
                request_id: &request_id,
                ..call.call.copy()
            },
        )?;
        acted.push(super::gather(answer, &mut outcome, &mut envelope)?);
    }
    Ok(Answered {
        outcome: outcome.map(|()| StepAnswer::Acted(acted)),
        envelope,
    }
    .into())
}
