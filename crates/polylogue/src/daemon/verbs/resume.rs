//! `resume`: one session's resume command through its own tool, the open
//! sessions here when there is no selection, or with `--in` its work
//! continued in a new session of another harness.
use super::{Given, RowRule, Selection, StepCall, StepDone, Verb, sessions};
use crate::commands::ToolAccess;
use crate::daemon::Failure;
use crate::daemon::handler::Call;
use crate::daemon::service::acts;
use crate::protocol::acts::Acted;
use crate::protocol::line::StepAnswer;
use crate::protocol::resume;
use crate::protocol::steps::ResumeStep;
use command_surface::Answer;
use polylogue_model::{Reason, SessionId};

#[cfg(test)]
mod tests;

/// Sessions worth resuming that `resume` with no selection lists.
const RESUMABLE: u64 = 10;

pub(crate) struct Resume;
impl Verb for Resume {
    type Step = ResumeStep;
    const NAME: &'static str = "resume";
    const ROWS: RowRule = RowRule::One;
    const ACCESS: ToolAccess = ToolAccess::Act;
    fn run(
        step: ResumeStep,
        selection: &Selection,
        call: &StepCall<'_>,
    ) -> Result<StepDone, Failure> {
        let session = session(&step, selection, call)?;
        match (&step.harness, session) {
            (None, session) => Ok(resume(session, &call.call)?
                .map(|found| StepAnswer::Resume(Box::new(found)))
                .into()),
            (Some(_), None) => Err(Reason::ResumeInNeedsSession {}.into()),
            (Some(harness), Some(session)) => Ok(acts::resume_in(
                call.call.daemon,
                call.call.invocation,
                &continuation(&step, harness, &session, &call.call),
            )?
            .map(|continued| StepAnswer::Acted(vec![Acted::Continued(Box::new(continued))]))
            .into()),
        }
    }
    fn would(
        step: &ResumeStep,
        selection: &Selection,
        call: &StepCall<'_>,
    ) -> Result<String, Failure> {
        let session = session(step, selection, call)?;
        match (&step.harness, session) {
            (None, session) => {
                let answer = resume(session, &call.call)?;
                let found = match answer.outcome {
                    command_surface::Outcome::Error(error) => return Err(error.into()),
                    command_surface::Outcome::Ok(found)
                    | command_surface::Outcome::Empty(found)
                    | command_surface::Outcome::Degraded { value: found, .. } => found,
                };
                match found {
                    resume::Answer::Plan(plan) => {
                        let command = plan.harness_command.ok_or_else(|| {
                            Failure::refusal(Reason::NoHarnessResume {
                                session: plan.session.to_string(),
                            })
                        })?;
                        if command.cwd.is_none() {
                            return Err(Reason::ResumeDirectoryUnknown {
                                line: crate::cli::print::command_line(&plan.context_assisted),
                            }
                            .into());
                        }
                        Ok(format!(
                            "would resume {} with {}",
                            plan.session,
                            crate::cli::print::command_line(&command)
                        ))
                    }
                    found @ resume::Answer::Sessions(_) => Ok(crate::cli::print::resume(
                        &found,
                        command_surface::TextStyle::Plain,
                    )),
                }
            }
            (Some(_), None) => Err(Reason::ResumeInNeedsSession {}.into()),
            (Some(harness), Some(session)) => {
                let briefing = acts::briefing(
                    call.call.daemon,
                    call.call.invocation,
                    &continuation(step, harness, &session, &call.call),
                )?;
                let tokens = polylogue_model::read::estimate_tokens(&briefing.text);
                Ok(format!(
                    "would continue {session} in {harness} with a brief of {tokens} tokens"
                ))
            }
        }
    }
}

/// With a session, its resume commands; without, the sessions in the
/// caller's directory worth resuming.
fn resume(session: Option<SessionId>, call: &Call<'_>) -> Result<Answer<resume::Answer>, Failure> {
    let reader = call.reader()?;
    if let Some(session) = session {
        return Ok(match crate::resume::plan(&reader, &session) {
            Ok(plan) => Answer::ok(resume::Answer::Plan(plan)),
            Err(e) => crate::daemon::service::store_error(&e),
        });
    }
    let cwd = call.directory(None)?;
    let context = call.context()?;
    let sessions = crate::resume::resumable(&reader, &cwd, RESUMABLE, &context)?;
    Ok(if sessions.is_empty() {
        Answer::empty(resume::Answer::Sessions(sessions))
    } else {
        Answer::ok(resume::Answer::Sessions(sessions))
    })
}

/// The new session `resume --in HARNESS` starts for `session`.
fn continuation<'a>(
    step: &'a ResumeStep,
    harness: &'a str,
    session: &'a SessionId,
    call: &'a Call<'_>,
) -> acts::Continuation<'a> {
    acts::Continuation {
        request_id: call.request_id,
        source: session,
        harness,
        cwd: step.cwd.as_deref(),
        at: step.at.as_deref(),
        stop_source: step.stop_source,
    }
}

/// The one session resumed; none when the line names no rows.
fn session(
    step: &ResumeStep,
    selection: &Selection,
    call: &StepCall<'_>,
) -> Result<Option<SessionId>, Failure> {
    if step.harness.is_none() && (step.at.is_some() || step.cwd.is_some() || step.stop_source) {
        return Err(Reason::ResumeOptionsNeedIn {}.into());
    }
    if selection.given == Given::None {
        return Ok(None);
    }
    match sessions("resume", selection)?.as_slice() {
        [] => Err(Failure::not_found(
            "resume found no session in the selection",
        )),
        [session] => {
            if step.harness.is_none() {
                acts::refuse_running_resume(call.call.daemon, session)?;
            }
            Ok(Some(session.clone()))
        }
        several => Err(crate::protocol::failure(
            polylogue_model::Code::Ambiguous,
            format!("resume takes one session; {} match", several.len()),
        )
        .with_candidates(several.iter().map(ToString::to_string).collect())
        .into()),
    }
}
