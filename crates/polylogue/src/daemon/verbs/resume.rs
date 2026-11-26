//! `resume`: one session's resume command through its own tool, the open
//! sessions here when there is no selection, or with `--in` a hand-off of
//! its work to a new session of another harness.
use super::{Given, RowRule, Selection, StepCall, StepDone, Verb, delegated, not_built, sessions};
use crate::daemon::Failure;
use crate::daemon::handler::Operation;
use crate::daemon::service::{acts, handlers};
use crate::protocol::line::StepAnswer;
use crate::protocol::steps::ResumeStep;
use crate::protocol::{act, resume};
use polylogue_model::SessionId;

#[cfg(test)]
mod tests;

pub(crate) struct Resume;
impl Verb for Resume {
    type Step = ResumeStep;
    const NAME: &'static str = "resume";
    const ROWS: RowRule = RowRule::One;
    fn run(
        step: ResumeStep,
        selection: &Selection,
        call: &StepCall<'_>,
    ) -> Result<StepDone, Failure> {
        let session = session(&step, selection, call)?;
        match (step.harness, session) {
            (None, session) => delegated(
                handlers::Continue::run(
                    resume::Arguments {
                        session,
                        cwd: None,
                        limit: None,
                    },
                    &call.call,
                )?,
                |found| StepAnswer::Resume(Box::new(found)),
            ),
            (Some(_), None) => Err(Failure::bad_request(
                "resume --in needs the session whose work it continues",
            )),
            (Some(to), Some(session)) => delegated(
                acts::HandOffSession::run(
                    act::HandOff {
                        session,
                        to,
                        at: step.at,
                        cwd: step.cwd,
                        stop_source: step.stop_source,
                        print: false,
                    },
                    &call.call,
                )?,
                |value| StepAnswer::Acted(vec![value]),
            ),
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
                let done = delegated(
                    handlers::Continue::run(
                        resume::Arguments {
                            session,
                            cwd: None,
                            limit: None,
                        },
                        &call.call,
                    )?,
                    |found| StepAnswer::Resume(Box::new(found)),
                )?;
                match done.reply {
                    StepAnswer::Resume(found) => match *found {
                        resume::Answer::Plan(plan) => {
                            let command = plan.harness_command.ok_or_else(|| {
                                Failure::bad_request(format!(
                                    "{} has no harness resume command; use resume --in HARNESS",
                                    plan.session
                                ))
                            })?;
                            if command.cwd.is_none() {
                                return Err(Failure::bad_request(format!(
                                    "{}; continue with context instead: {}",
                                    crate::cli::print::DIRECTORY_UNKNOWN,
                                    crate::cli::print::command_line(&plan.context_assisted)
                                )));
                            }
                            Ok(format!(
                                "would resume {} with {}",
                                plan.session,
                                crate::cli::print::command_line(&command)
                            ))
                        }
                        resume::Answer::Sessions(found) => Ok(crate::cli::print::resume(
                            &resume::Answer::Sessions(found),
                            command_surface::TextStyle::Plain,
                        )),
                    },
                    _ => unreachable!(),
                }
            }
            (Some(_), None) => Err(Failure::bad_request(
                "resume --in needs the session whose work it continues",
            )),
            (Some(to), Some(session)) => {
                let done = delegated(
                    acts::HandOffSession::run(
                        act::HandOff {
                            session: session.clone(),
                            to: to.clone(),
                            at: step.at.clone(),
                            cwd: step.cwd.clone(),
                            stop_source: step.stop_source,
                            print: true,
                        },
                        &call.call,
                    )?,
                    |value| StepAnswer::Acted(vec![value]),
                )?;
                let StepAnswer::Acted(values) = done.reply else {
                    unreachable!()
                };
                let text = values[0]["pack"]
                    .as_str()
                    .ok_or_else(|| Failure::internal("the dry-run hand-off carries no brief"))?;
                let tokens = polylogue_model::read::estimate_tokens(text);
                Ok(format!(
                    "would continue {session} in {to} with a brief of {tokens} tokens"
                ))
            }
        }
    }
}

/// The one session resumed; none when the line names no rows.
fn session(
    step: &ResumeStep,
    selection: &Selection,
    call: &StepCall<'_>,
) -> Result<Option<SessionId>, Failure> {
    if step.fork {
        return Err(not_built("resume", "fork"));
    }
    if step.anyway {
        return Err(not_built("resume", "anyway"));
    }
    if step.harness.is_none() && (step.at.is_some() || step.cwd.is_some() || step.stop_source) {
        return Err(Failure::bad_request(
            "resume --at, --cwd and --stop-source apply only with --in HARNESS",
        ));
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
