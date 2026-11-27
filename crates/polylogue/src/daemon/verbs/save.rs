//! `save [NAME]`: name the question and keep this line's rows; `--watch`
//! also watches the question.
use super::{RowRule, Selection, StepCall, StepDone, Verb};
use crate::commands::ToolAccess;
use crate::daemon::Failure;
use crate::daemon::service::handlers;
use crate::protocol::line::StepAnswer;
use crate::protocol::steps::SaveStep;

pub(crate) struct Save;
impl Verb for Save {
    type Step = SaveStep;
    const NAME: &'static str = "save";
    const ROWS: RowRule = RowRule::Any;
    const ACCESS: ToolAccess = ToolAccess::Write;
    fn run(
        step: SaveStep,
        selection: &Selection,
        call: &StepCall<'_>,
    ) -> Result<StepDone, Failure> {
        check(&step, selection)?;
        let rows = selection
            .query_text()
            .ok_or_else(|| Failure::bad_request("save needs rows"))?;
        let mut question = selection.query.clone().unwrap_or_else(|| rows.clone());
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
            let reader = call.call.reader()?;
            let reference = polylogue_model::RevisionReference {
                root: definition.root.clone(),
                revision: Some(definition.revision.clone()),
            };
            let saved = reader
                .annotation(&reference)?
                .ok_or_else(|| Failure::not_found(format!("query:{name} accepted revision")))?;
            let value: polylogue_model::annotation::values::SavedQuery = serde_json::from_value(
                saved
                    .fact
                    .value
                    .ok_or_else(|| Failure::internal("saved query has no value"))?,
            )
            .map_err(|error| Failure::internal(error.to_string()))?;
            question = value.text;
        }
        let mut undo = Vec::new();
        let mut position = None;
        let mut done = StepDone::from(
            handlers::save_selection(step, question, selection, &call.call)?.map(|committed| {
                position = Some(committed.position);
                let result = committed.outcome;
                undo = result.written;
                let mut report = result
                    .save
                    .expect("Save transaction returns its resolved outcomes");
                report.pending = result.outcome.pending;
                report.repeat |= result.outcome.repeat;
                StepAnswer::Saved(report)
            }),
        )
        .checked()?;
        done.position = position;
        done.undo
            .extend(super::undo_line(call.call.request_id, &undo)?);
        Ok(done)
    }

    fn would(step: &SaveStep, selection: &Selection, _: &StepCall<'_>) -> Result<String, Failure> {
        check(step, selection)?;
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

fn check(step: &SaveStep, selection: &Selection) -> Result<(), Failure> {
    if let Some(name) = &step.name {
        let catalog = polylogue_query::catalog::catalog(&[]);
        let captured = if crate::commands::is_cli_command(name) {
            Some("command")
        } else if crate::daemon::verbs::declarations()
            .verbs()
            .iter()
            .any(|verb| verb.verb() == name)
        {
            Some("verb")
        } else if catalog.units.iter().any(|unit| unit.name == *name) {
            Some("unit")
        } else if polylogue_query::catalog::REFERENCE_WORDS.contains(&name.as_str()) {
            Some("reserved word")
        } else if ["query", "result"].contains(&name.as_str())
            || catalog.references.iter().any(|reference| {
                matches!(reference,
                query_language::catalog::ReferenceForm::Prefix { prefix, .. }
                    if prefix.strip_suffix(':') == Some(name.as_str()))
            })
        {
            Some("reference prefix")
        } else if !name.is_empty() && name.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            Some("hexadecimal identifier")
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
        return Err(crate::unbuilt::refusal("save", "page:NAME").into());
    }
    if let Some(name) = &step.name
        && !polylogue_model::annotation::policy::valid_saved_name(name)
    {
        return Err(Failure::bad_value(format!(
            "{name:?} is not a saved query name: use lowercase letters, digits and hyphens, starting with a letter"
        )));
    }
    if !step.watch {
        for (given, option) in [(step.hold.is_some(), "hold"), (step.notify, "notify")] {
            if given {
                return Err(Failure::bad_request(format!(
                    "save --{option} needs --watch"
                )));
            }
        }
    }
    if step.watch
        && let Some(hold) = &step.hold
        && crate::signals::duration(hold).is_none()
    {
        return Err(Failure::invalid(format!(
            "--hold {hold:?} is not a duration such as 20m"
        )));
    }
    if step.watch && step.name.is_none() {
        let query = selection
            .query
            .clone()
            .or_else(|| selection.query_text())
            .unwrap_or_else(|| "QUERY".into());
        let line =
            crate::cli::line::shell_line(&[query, "save".into(), "NAME".into(), "--watch".into()]);
        return Err(Failure::bad_request(format!(
            "a watch needs a name: {line}"
        )));
    }
    if step.denominator.is_some() && step.name.is_none() {
        return Err(Failure::bad_request("a figure needs a saved query name"));
    }
    Ok(())
}
