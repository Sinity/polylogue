//! The acts on live sessions: `send`, `answer` and `interrupt`, through the
//! handlers of the acts they replace. Each recipient's act is its own
//! request.
use super::{
    RowRule, Selection, StepCall, StepDone, Verb, delegated, not_built, not_built_use, sessions,
};
use crate::daemon::Failure;
use crate::daemon::handler::{Call, Handler};
use crate::daemon::service::acts;
use crate::protocol::act;
use crate::protocol::line::{SendReport, StepAnswer};
use crate::protocol::steps::{AnswerStep, InterruptStep, SendStep};
use command_surface::Envelope;
use polylogue_model::SessionId;
use serde_json::Value;

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
        let session = recipient(&step, selection)?;
        delegated(
            acts::Ping::run(
                act::Ping {
                    target: session.to_string(),
                    text: step.text,
                    live: false,
                    about: step.about,
                },
                &call.call,
            )?,
            |reached| {
                StepAnswer::Sent(SendReport {
                    reached: vec![reached],
                    skipped: Vec::new(),
                })
            },
        )
    }
    fn would(step: &SendStep, selection: &Selection, _: &StepCall<'_>) -> Result<String, Failure> {
        let session = recipient(step, selection)?;
        Ok(format!("would send {:?} to {session}", step.text))
    }
}

/// The one session a send reaches.
fn recipient(step: &SendStep, selection: &Selection) -> Result<SessionId, Failure> {
    for (given, option) in [
        (!step.attach.is_empty(), "attach"),
        (step.page.is_some(), "page"),
        (step.section.is_some(), "section"),
    ] {
        if given {
            return Err(not_built("send", option));
        }
    }
    match sessions("send", selection)?.as_slice() {
        [session] => Ok(session.clone()),
        _ => Err(not_built_use("send", "to several sessions")),
    }
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
        each_session(selection, call, |session, call| {
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
        _: &StepCall<'_>,
    ) -> Result<String, Failure> {
        decision(step)?;
        sessions("answer", selection)?;
        Ok(format!(
            "would answer {} for {}",
            step.value,
            selection.described()
        ))
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
        each_session(selection, call, |session, call| {
            acts::Interrupt::run(act::Interrupt { session }, call)
        })
    }
    fn would(
        _: &InterruptStep,
        selection: &Selection,
        _: &StepCall<'_>,
    ) -> Result<String, Failure> {
        sessions("interrupt", selection)?;
        Ok(format!("would interrupt {}", selection.described()))
    }
}

/// One act per session, request `<step>-<n>`; the first failure stops the step.
fn each_session(
    selection: &Selection,
    call: &StepCall<'_>,
    one: impl Fn(SessionId, &Call<'_>) -> Result<command_surface::Answer<Value>, Failure>,
) -> Result<StepDone, Failure> {
    let mut acted = Vec::new();
    let mut envelope = Envelope::default();
    for (n, session) in sessions("this act", selection)?.into_iter().enumerate() {
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
    })
}
