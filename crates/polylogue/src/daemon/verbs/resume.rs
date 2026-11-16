//! `resume`: one session's resume command through its own tool, the open
//! sessions here when there is no selection, or with `--in` a hand-off of
//! its work to a new session of another harness.
use super::{Given, RowRule, Selection, StepCall, StepDone, Verb, delegated, not_built, sessions};
use crate::daemon::Failure;
use crate::daemon::handler::Handler;
use crate::daemon::service::{acts, handlers};
use crate::protocol::line::StepAnswer;
use crate::protocol::steps::ResumeStep;
use crate::protocol::{act, resume};
use polylogue_model::SessionId;

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
        let session = session(&step, selection)?;
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
        _: &StepCall<'_>,
    ) -> Result<String, Failure> {
        Ok(match (&step.harness, session(step, selection)?) {
            (_, None) => "would list the sessions here that ended open".into(),
            (None, Some(session)) => format!("would resume {session}"),
            (Some(to), Some(session)) => format!("would continue {session} in {to}"),
        })
    }
}

/// The one session resumed; none when the line names no rows.
fn session(step: &ResumeStep, selection: &Selection) -> Result<Option<SessionId>, Failure> {
    if step.fork {
        return Err(not_built("resume", "fork"));
    }
    if step.anyway {
        return Err(not_built("resume", "anyway"));
    }
    if selection.given == Given::None {
        return Ok(None);
    }
    match sessions("resume", selection)?.as_slice() {
        [session] => Ok(Some(session.clone())),
        several => Err(crate::protocol::failure(
            polylogue_model::Code::Ambiguous,
            format!("resume takes one session; {} match", several.len()),
        )
        .with_candidates(several.iter().map(ToString::to_string).collect())
        .into()),
    }
}
