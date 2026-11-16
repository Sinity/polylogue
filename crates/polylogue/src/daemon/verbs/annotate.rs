//! The verbs that write annotations: `note`, `tag`, `edit`, `withdraw` and
//! `judge`. Notes and tags over the rows are one query assertion, which the
//! writer checks and expands into one annotation per row.
use super::{
    RowRule, Selection, StepCall, StepDone, Verb, delegated, not_built, not_built_use, undo_line,
    written,
};
use crate::daemon::Failure;
use crate::daemon::handler::Handler;
use crate::daemon::service::handlers;
use crate::protocol::assertion;
use crate::protocol::line::StepAnswer;
use crate::protocol::steps::{EditStep, JudgeStep, NoteStep, TagStep, WithdrawStep};
use polylogue_model::assertion::{
    AssertionKind, AssertionPatch, Change, LinkRole, NewAssertion, NewAssertionLink,
};
use polylogue_model::{Reference, RevisionReference, Timestamp};
use serde_json::Value;

pub(crate) struct Note;
impl Verb for Note {
    type Step = NoteStep;
    const NAME: &'static str = "note";
    const ROWS: RowRule = RowRule::Things;
    fn run(
        step: NoteStep,
        selection: &Selection,
        call: &StepCall<'_>,
    ) -> Result<StepDone, Failure> {
        let assertion = note(&step)?;
        each(assertion, selection, call)
    }
    fn would(step: &NoteStep, selection: &Selection, _: &StepCall<'_>) -> Result<String, Failure> {
        let assertion = note(step)?;
        Ok(format!(
            "would write a {} about each of {}",
            assertion.kind.as_str(),
            selection.described()
        ))
    }
}

/// The annotation a note step writes about each row.
fn note(step: &NoteStep) -> Result<NewAssertion, Failure> {
    let NoteStep {
        text,
        kind,
        key,
        parts,
        evidence,
        replies_to,
        supports,
        challenges,
        qualifies,
        corrects,
        derived_from,
        blocks,
        scope,
        valid_until,
        check,
        expect,
        page,
        section,
        level,
        template,
        param,
        layout,
        about,
        role,
        brief,
        rename,
    } = step;
    for (given, option) in [
        (parts.is_some(), "parts"),
        (check.is_some(), "check"),
        (expect.is_some(), "expect"),
        (page.is_some(), "page"),
        (section.is_some(), "section"),
        (level.is_some(), "level"),
        (template.is_some(), "template"),
        (!param.is_empty(), "param"),
        (layout.is_some(), "layout"),
        (about.is_some(), "about"),
        (role.is_some(), "role"),
        (brief.is_some(), "brief"),
        (*rename, "rename"),
    ] {
        if given {
            return Err(not_built("note", option));
        }
    }
    let kind = match kind.as_deref() {
        None => AssertionKind::Note,
        Some(name) => AssertionKind::from_name(name)
            .filter(|kind| !matches!(kind, AssertionKind::Tag))
            .ok_or_else(|| not_built_use("note", &format!("--as {name}")))?,
    };
    let mut links = Vec::new();
    for (references, role) in [
        (evidence, LinkRole::Evidence),
        (replies_to, LinkRole::Answers),
        (supports, LinkRole::Supports),
        (challenges, LinkRole::Challenges),
        (qualifies, LinkRole::Qualifies),
        (corrects, LinkRole::Corrects),
        (derived_from, LinkRole::DerivedFrom),
        (blocks, LinkRole::Blocks),
        (scope, LinkRole::Scope),
    ] {
        for text in references {
            links.push(NewAssertionLink {
                role,
                reference: Reference::parse(text).map_err(Failure::bad_value)?,
            });
        }
    }
    let valid_until = valid_until
        .as_deref()
        .map(|text| {
            serde_json::from_value::<Timestamp>(Value::String(text.to_owned()))
                .map_err(|error| Failure::bad_value(format!("note --valid-until: {error}")))
        })
        .transpose()?;
    Ok(NewAssertion {
        kind,
        key: key.clone(),
        value: Some(Value::String(text.clone())),
        valid_until,
        links,
    })
}

pub(crate) struct Tag;
impl Verb for Tag {
    type Step = TagStep;
    const NAME: &'static str = "tag";
    const ROWS: RowRule = RowRule::Things;
    fn run(step: TagStep, selection: &Selection, call: &StepCall<'_>) -> Result<StepDone, Failure> {
        each(tag(&step)?, selection, call)
    }
    fn would(step: &TagStep, selection: &Selection, _: &StepCall<'_>) -> Result<String, Failure> {
        tag(step)?;
        Ok(format!("would tag {} {}", selection.described(), step.name))
    }
}

/// A tag's key and value split at the first `:`.
fn tag(step: &TagStep) -> Result<NewAssertion, Failure> {
    let (key, value) = match step.name.split_once(':') {
        Some((key, value)) => (key, Some(Value::String(value.to_owned()))),
        None => (step.name.as_str(), None),
    };
    if key.is_empty() {
        return Err(Failure::bad_request(
            "tag needs a name: tag NAME or NAME:VALUE",
        ));
    }
    Ok(NewAssertion {
        kind: AssertionKind::Tag,
        key: Some(key.to_owned()),
        value,
        valid_until: None,
        links: Vec::new(),
    })
}

/// One annotation about each row of the selection, in one request.
fn each(
    assertion: NewAssertion,
    selection: &Selection,
    call: &StepCall<'_>,
) -> Result<StepDone, Failure> {
    let query = selection
        .query_text()
        .ok_or_else(|| Failure::bad_request("a write needs rows"))?;
    let mut done = delegated(
        handlers::AssertQuery::run(assertion::AssertQuery { query, assertion }, &call.call)?,
        StepAnswer::WroteEach,
    )?;
    if let StepAnswer::WroteEach(outcome) = &done.reply {
        done.undo.extend(undo_line(
            call.call.request_id,
            &written(&outcome.id, outcome.targets),
        ));
    }
    Ok(done)
}

pub(crate) struct Edit;
impl Verb for Edit {
    type Step = EditStep;
    const NAME: &'static str = "edit";
    const ROWS: RowRule = RowRule::Things;
    fn run(
        step: EditStep,
        selection: &Selection,
        call: &StepCall<'_>,
    ) -> Result<StepDone, Failure> {
        let text = text(&step)?;
        let annotations = annotations("edit", selection)?;
        let bases = {
            let reader = call.call.reader()?;
            annotations
                .into_iter()
                .map(|reference| match reference.revision {
                    Some(_) => Ok(reference),
                    None => match reader.assertion(&reference)? {
                        Some(found) => Ok(RevisionReference::exact(found.fact.root, found.fact.id)),
                        None => Err(Failure::not_found(format!("no annotation {reference}"))),
                    },
                })
                .collect::<Result<Vec<_>, Failure>>()?
        };
        let count = bases.len();
        let patch = AssertionPatch {
            key: step.key.map_or(Change::Keep, Change::Set),
            value: Change::Set(Value::String(text)),
            ..AssertionPatch::default()
        };
        wrote(
            handlers::Revise::run(assertion::Revise { bases, patch }, &call.call)?,
            call,
            count,
        )
    }
    fn would(step: &EditStep, selection: &Selection, _: &StepCall<'_>) -> Result<String, Failure> {
        text(step)?;
        annotations("edit", selection)?;
        Ok(format!(
            "would write a new revision of {}",
            selection.described()
        ))
    }
}

fn text(step: &EditStep) -> Result<String, Failure> {
    step.text.clone().ok_or_else(|| {
        Failure::bad_request("edit needs its new text; with no text a terminal opens your editor")
    })
}

/// The rows as annotations, refusing any other row.
fn annotations(verb: &str, selection: &Selection) -> Result<Vec<RevisionReference>, Failure> {
    selection
        .references()?
        .iter()
        .map(|reference| match reference {
            Reference::Assertion(revision) => Ok(revision.clone()),
            other => Err(Failure::bad_request(format!(
                "{verb} acts on annotations; {other} is not one"
            ))),
        })
        .collect()
}

/// A write's outcome as a finished step with its undo line.
fn wrote(
    answer: command_surface::Answer<polylogue_model::user::UserOutcome>,
    call: &StepCall<'_>,
    count: usize,
) -> Result<StepDone, Failure> {
    let mut done = delegated(answer, StepAnswer::Wrote)?;
    if let StepAnswer::Wrote(outcome) = &done.reply {
        done.undo.extend(undo_line(
            call.call.request_id,
            &written(&outcome.id, count),
        ));
    }
    Ok(done)
}

pub(crate) struct Withdraw;
impl Verb for Withdraw {
    type Step = WithdrawStep;
    const NAME: &'static str = "withdraw";
    const ROWS: RowRule = RowRule::Things;
    fn run(
        step: WithdrawStep,
        selection: &Selection,
        call: &StepCall<'_>,
    ) -> Result<StepDone, Failure> {
        let targets = withdrawn(selection)?;
        wrote(
            handlers::Withdraw::run(
                assertion::Withdraw {
                    targets,
                    reason: step.reason,
                },
                &call.call,
            )?,
            call,
            1,
        )
    }
    fn would(_: &WithdrawStep, selection: &Selection, _: &StepCall<'_>) -> Result<String, Failure> {
        withdrawn(selection)?;
        Ok(format!("would withdraw {}", selection.described()))
    }
}

/// The annotations and alternates a withdrawal names.
fn withdrawn(selection: &Selection) -> Result<Vec<Reference>, Failure> {
    selection
        .references()?
        .iter()
        .map(|reference| match reference {
            Reference::Assertion(_) | Reference::Variant { .. } => Ok(reference.clone()),
            other => Err(Failure::bad_request(format!(
                "withdraw takes annotations back; {other} is not one"
            ))),
        })
        .collect()
}

pub(crate) struct Judge;
impl Verb for Judge {
    type Step = JudgeStep;
    const NAME: &'static str = "judge";
    const ROWS: RowRule = RowRule::Things;
    fn run(
        step: JudgeStep,
        selection: &Selection,
        call: &StepCall<'_>,
    ) -> Result<StepDone, Failure> {
        let (note, accepted) = verdict(&step, selection)?;
        wrote(
            handlers::Judge::run(
                assertion::Judge {
                    note: note.to_string(),
                    accepted,
                    reason: step.reason,
                },
                &call.call,
            )?,
            call,
            1,
        )
    }
    fn would(step: &JudgeStep, selection: &Selection, _: &StepCall<'_>) -> Result<String, Failure> {
        let (note, accepted) = verdict(step, selection)?;
        Ok(format!(
            "would {} {note}",
            if accepted { "accept" } else { "reject" }
        ))
    }
}

/// The one annotation judged and whether it is accepted.
fn verdict<'s>(
    step: &JudgeStep,
    selection: &'s Selection,
) -> Result<(&'s Reference, bool), Failure> {
    if step.on.is_some() {
        return Err(not_built("judge", "on"));
    }
    if step.seed.is_some() {
        return Err(not_built("judge", "seed"));
    }
    let accepted = match step.verdict.as_str() {
        "accept" => true,
        "reject" => false,
        other => return Err(not_built_use("judge", other)),
    };
    match selection.references()? {
        [note @ Reference::Assertion(_)] => Ok((note, accepted)),
        [other] => Err(Failure::bad_request(format!(
            "judge accepts or rejects annotations; {other} is not one"
        ))),
        _ => Err(not_built_use("judge", "over several rows")),
    }
}
