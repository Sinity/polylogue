//! `save [NAME]`: name the question and keep this line's rows; `--watch`
//! also watches the question.
use super::{RowRule, Selection, StepCall, StepDone, Verb, not_built, undo_line, written};
use crate::daemon::Failure;
use crate::daemon::handler::{Call, Handler};
use crate::daemon::service::handlers;
use crate::protocol::line::{SaveReport, StepAnswer};
use crate::protocol::saved;
use crate::protocol::steps::SaveStep;
use command_surface::{Answer, Envelope, Outcome};
use polylogue_model::user::UserOutcome;

pub(crate) struct Save;
impl Verb for Save {
    type Step = SaveStep;
    const NAME: &'static str = "save";
    const ROWS: RowRule = RowRule::Any;
    fn run(
        step: SaveStep,
        selection: &Selection,
        call: &StepCall<'_>,
    ) -> Result<StepDone, Failure> {
        check(&step)?;
        let rows = selection
            .query_text()
            .ok_or_else(|| Failure::bad_request("save needs rows"))?;
        let question = selection.question.clone().unwrap_or_else(|| rows.clone());
        let mut envelope = Envelope::default();
        let mut undo = Vec::new();
        let id = |part: &str| format!("{}-{part}", call.call.request_id);
        let saved = step
            .name
            .as_ref()
            .map(|name| {
                let request_id = id("save");
                let outcome = take(
                    handlers::Save::run(
                        saved::Save {
                            name: name.clone(),
                            query: question.clone(),
                            denominator: step.denominator.clone(),
                        },
                        &Call {
                            request_id: &request_id,
                            ..call.call.copy()
                        },
                    )?,
                    &mut envelope,
                )?;
                undo.extend(undo_line(&request_id, &written(&outcome.id, 1)));
                Ok::<_, Failure>(outcome)
            })
            .transpose()?;
        let request_id = id("keep");
        let kept = take(
            handlers::Keep::run(
                saved::Keep {
                    name: step.name.clone(),
                    query: rows,
                },
                &Call {
                    request_id: &request_id,
                    ..call.call.copy()
                },
            )?,
            &mut envelope,
        )?;
        let watch = match (&step.name, step.watch) {
            (Some(name), true) => {
                let request_id = id("watch");
                let outcome = take(
                    handlers::Watch::run(
                        saved::Watch {
                            name: Some(name.clone()),
                            hold: step.hold.clone(),
                            notify: step.notify,
                            query: question,
                        },
                        &Call {
                            request_id: &request_id,
                            ..call.call.copy()
                        },
                    )?,
                    &mut envelope,
                )?;
                undo.extend(undo_line(&request_id, &written(&outcome.id, 1)));
                Some(outcome)
            }
            _ => None,
        };
        Ok(StepDone {
            reply: StepAnswer::Saved(SaveReport { saved, kept, watch }),
            undo,
            envelope,
        })
    }
    fn would(step: &SaveStep, selection: &Selection, _: &StepCall<'_>) -> Result<String, Failure> {
        check(step)?;
        Ok(match &step.name {
            Some(name) if step.watch => {
                format!(
                    "would save and watch {name} and keep {}",
                    selection.described()
                )
            }
            Some(name) => format!("would save {name} and keep {}", selection.described()),
            None => format!("would keep {}", selection.described()),
        })
    }
}

fn check(step: &SaveStep) -> Result<(), Failure> {
    if step
        .name
        .as_deref()
        .is_some_and(|name| name.starts_with("page:"))
    {
        return Err(super::not_built_use("save", "page:NAME"));
    }
    for (given, option) in [
        (step.section.is_some(), "section"),
        (step.after.is_some(), "after"),
        (step.fixed, "fixed"),
        (step.live, "live"),
        (step.kept, "kept"),
        (step.template.is_some(), "template"),
        (step.kind.is_some(), "as"),
    ] {
        if given {
            return Err(not_built("save", option));
        }
    }
    if step.watch && step.name.is_none() {
        return Err(Failure::bad_request(
            "a watch needs a name: save NAME --watch",
        ));
    }
    Ok(())
}

/// One write's outcome, its notes added to the step's.
fn take(answer: Answer<UserOutcome>, envelope: &mut Envelope) -> Result<UserOutcome, Failure> {
    let Answer {
        outcome,
        envelope: written,
    } = answer;
    envelope.notes.extend(written.notes);
    envelope.next.extend(written.next);
    match outcome {
        Outcome::Ok(value) | Outcome::Empty(value) => Ok(value),
        Outcome::Degraded { value, missing } => {
            envelope.notes.extend(
                missing
                    .into_iter()
                    .map(|missing| format!("missing: {missing}")),
            );
            Ok(value)
        }
        Outcome::Error(error) => Err(Failure::from(error)),
    }
}
