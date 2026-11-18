//! `save [NAME]`: name the question and keep this line's rows; `--watch`
//! also watches the question.
use super::{RowRule, Selection, StepCall, StepDone, Verb, not_built};
use crate::daemon::Failure;
use crate::daemon::service::handlers;
use crate::protocol::line::StepAnswer;
use crate::protocol::steps::SaveStep;

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
        let mut question = selection.question.clone().unwrap_or_else(|| rows.clone());
        if let (Some(name), Some(retained), super::Given::Query(selected)) =
            (&step.name, selection.retained.as_deref(), &selection.given)
            && (selected == name || selected.strip_prefix("query:") == Some(name.as_str()))
            && let Some(definition) = retained
                .binding
                .replay
                .definitions
                .iter()
                .find(|definition| definition.name == *name)
        {
            let plan = polylogue_query::plan::replay_binding(&retained.binding.replay)?;
            question = query_language::format::format_query(&definition.ast, plan.catalog())
                .map_err(Failure::bad_value)?;
        }
        let mut undo = Vec::new();
        let mut done = super::delegated(
            handlers::save_selection(step, question, selection, &call.call)?,
            |(report, committed)| {
                undo = committed;
                StepAnswer::Saved(report)
            },
        )?;
        for reference in undo {
            done.undo
                .extend(super::undo_line(&reference.to_string(), &[reference])?);
        }
        Ok(done)
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
    if let Some(name) = &step.name {
        let captured = if crate::commands::is_cli_command(name) {
            Some("command")
        } else if crate::protocol::steps::declarations()
            .verbs()
            .iter()
            .any(|verb| verb.verb() == name)
        {
            Some("verb")
        } else {
            None
        };
        if let Some(kind) = captured {
            return Err(Failure::bad_value(format!(
                "{name:?} names a {kind}; choose another saved query name"
            )));
        }
    }

    if step
        .name
        .as_deref()
        .is_some_and(|name| name.starts_with("page:"))
    {
        return Err(super::not_built_use("save", "page:NAME"));
    }
    if let Some(name) = &step.name
        && !polylogue_model::assertion::policy::sayable_query_name(name)
    {
        return Err(Failure::bad_value(format!(
            "{name:?} cannot follow `query:`"
        )));
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
    if step.watch
        && let Some(hold) = &step.hold
        && crate::signals::duration(hold).is_none()
    {
        return Err(Failure::invalid(format!(
            "--for {hold:?} is not a duration such as 20m"
        )));
    }
    if step.watch && step.name.is_none() {
        return Err(Failure::bad_request(
            "a watch needs a name: save NAME --watch",
        ));
    }
    Ok(())
}
