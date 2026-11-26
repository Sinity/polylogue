//! The verbs that write annotations: `note`, `tag`, `edit`, `withdraw` and
//! `judge`. Notes and tags over the rows are one query annotation, which the
//! writer checks and expands into one annotation per row.
use super::{RowRule, Selection, StepCall, StepDone, Verb, not_built, not_built_use, undo_line};
use crate::commands::ToolAccess;
use crate::daemon::Failure;
use crate::protocol::line::StepAnswer;
use crate::protocol::steps::{EditStep, JudgeStep, NoteStep, TagStep, WithdrawStep};
use polylogue_model::annotation::{
    AnnotationKind, AnnotationPatch, Change, LinkRole, NewAnnotation, NewAnnotationLink,
};
use polylogue_model::user::UserWrite;
use polylogue_model::{Reference, RevisionReference, Timestamp};
use serde_json::Value;

pub(crate) struct Note;
impl Verb for Note {
    type Step = NoteStep;
    const NAME: &'static str = "note";
    const ROWS: RowRule = RowRule::Things;
    const ACCESS: ToolAccess = ToolAccess::Write;
    fn run(
        step: NoteStep,
        selection: &Selection,
        call: &StepCall<'_>,
    ) -> Result<StepDone, Failure> {
        let annotation = note(&step)?;
        each(annotation, selection, call)
    }
    fn would(step: &NoteStep, selection: &Selection, _: &StepCall<'_>) -> Result<String, Failure> {
        let annotation = note(step)?;
        Ok(format!(
            "would write a {} about each of {}",
            annotation.kind.as_str(),
            selection.described()
        ))
    }
}

/// The annotation a note step writes about each row.
fn note(step: &NoteStep) -> Result<NewAnnotation, Failure> {
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
        None => AnnotationKind::Note,
        Some(name) => AnnotationKind::from_name(name)
            .filter(|kind| !matches!(kind, AnnotationKind::Tag))
            .ok_or_else(|| not_built_use("note", &format!("--as {name}")))?,
    };
    let mut links = Vec::new();
    for (references, role) in [
        (evidence, LinkRole::Evidence),
        (replies_to, LinkRole::RepliesTo),
        (supports, LinkRole::Supports),
        (challenges, LinkRole::Challenges),
        (qualifies, LinkRole::Qualifies),
        (corrects, LinkRole::Corrects),
        (derived_from, LinkRole::DerivedFrom),
        (blocks, LinkRole::Blocks),
        (scope, LinkRole::Scope),
    ] {
        for text in references {
            links.push(NewAnnotationLink {
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
    Ok(NewAnnotation {
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
    const ACCESS: ToolAccess = ToolAccess::Write;
    fn run(step: TagStep, selection: &Selection, call: &StepCall<'_>) -> Result<StepDone, Failure> {
        each(tag(&step)?, selection, call)
    }
    fn would(step: &TagStep, selection: &Selection, _: &StepCall<'_>) -> Result<String, Failure> {
        tag(step)?;
        Ok(format!("would tag {} {}", selection.described(), step.name))
    }
}

/// A tag's key and value split at the first `:`.
fn tag(step: &TagStep) -> Result<NewAnnotation, Failure> {
    let (key, value) = match step.name.split_once(':') {
        Some((key, value)) => (key, Some(serde_json::json!({"value": value}))),
        None => (step.name.as_str(), None),
    };
    if key.trim().is_empty() {
        return Err(Failure::bad_request(
            "tag needs a name: tag NAME or NAME:VALUE",
        ));
    }
    Ok(NewAnnotation {
        kind: AnnotationKind::Tag,
        key: Some(key.to_owned()),
        value,
        valid_until: None,
        links: Vec::new(),
    })
}

/// One annotation about each row of the selection, in one request.
fn each(
    annotation: NewAnnotation,
    selection: &Selection,
    call: &StepCall<'_>,
) -> Result<StepDone, Failure> {
    let query = selection
        .query_text()
        .ok_or_else(|| Failure::bad_request("a write needs rows"))?;
    let resolved = selection.references()?.to_vec();
    let targets = resolved.len();
    let reader = call.call.reader()?;
    let all_tagged = if annotation.kind == AnnotationKind::Tag {
        let value = annotation
            .value
            .as_ref()
            .and_then(|value| value.get("value"));
        reader
            .tagged(
                &resolved,
                annotation.key.as_deref().expect("tag key checked"),
                value,
                &call.call.invocation.authorship.author,
            )?
            .len()
            == targets
    } else {
        false
    };
    let scope = reader.scope_token();
    drop(reader);
    let write = UserWrite::AssertQuery {
        query,
        annotation,
        resolved,
    };
    // The original accepted or pending write owns a retry, even when all
    // of its targets are tagged now.
    if all_tagged && crate::daemon::service::accepted_write(&write, &call.call)?.is_none() {
        let UserWrite::AssertQuery { resolved, .. } = write else {
            return Err(Failure::internal(
                "an annotation step constructed another write",
            ));
        };
        let reply = StepAnswer::WroteEach(polylogue_model::user::AssertQueryOutcome::all_tagged(
            resolved,
        ));
        return Ok(if targets == 0 {
            command_surface::Answer::empty(reply)
        } else {
            command_surface::Answer::ok(reply)
        }
        .into());
    }
    let mut committed = Vec::new();
    let answer = crate::daemon::service::submit_result(
        call.call.daemon,
        call.call.invocation,
        call.call.request_id.into(),
        write,
        Some(scope),
        |receipt| {
            committed = receipt.written;
            polylogue_model::user::AssertQueryOutcome::new(
                receipt.outcome,
                targets,
                committed.len(),
                receipt.already,
            )
        },
    );
    let answer = if targets == 0 {
        match answer.outcome {
            command_surface::Outcome::Ok(value) => command_surface::Answer {
                outcome: command_surface::Outcome::Empty(value),
                envelope: answer.envelope,
            },
            _ => answer,
        }
    } else {
        answer
    };
    let mut done = StepDone::from(answer.map(StepAnswer::WroteEach)).checked()?;
    done.undo
        .extend(undo_line(call.call.request_id, &committed)?);
    Ok(done)
}

pub(crate) struct Edit;
impl Verb for Edit {
    type Step = EditStep;
    const NAME: &'static str = "edit";
    const ROWS: RowRule = RowRule::Things;
    const ACCESS: ToolAccess = ToolAccess::Write;
    fn run(
        step: EditStep,
        selection: &Selection,
        call: &StepCall<'_>,
    ) -> Result<StepDone, Failure> {
        let text = text(&step, selection)?;
        let annotations = annotations("edit", selection)?;
        let bases = {
            let reader = call.call.reader()?;
            annotations
                .into_iter()
                .map(|reference| match reference.revision {
                    Some(_) => Ok(reference),
                    None => {
                        let found = stored_annotation(&reader, &reference)?;
                        Ok(RevisionReference::exact(found.fact.root, found.fact.id))
                    }
                })
                .collect::<Result<Vec<_>, Failure>>()?
        };
        let patch = AnnotationPatch {
            key: step.key.map_or(Change::Keep, Change::Set),
            value: Change::Set(Value::String(text)),
            ..AnnotationPatch::default()
        };
        if bases.is_empty() {
            return Err(Failure::bad_request("bases must not be empty"));
        }
        wrote(UserWrite::ReviseAnnotations { bases, patch }, call, None)
    }
    fn would(step: &EditStep, selection: &Selection, _: &StepCall<'_>) -> Result<String, Failure> {
        text(step, selection)?;
        annotations("edit", selection)?;
        Ok(format!(
            "would write a new revision of {}",
            selection.described()
        ))
    }
}

/// The stored annotation a reference names, or the not-found refusal.
fn stored_annotation(
    reader: &polylogue_store::Reader<'_>,
    reference: &RevisionReference,
) -> Result<polylogue_model::annotation::Annotation, Failure> {
    reader
        .annotation(reference)?
        .ok_or_else(|| Failure::not_found(format!("no annotation {reference}")))
}

/// The new text. Without it the daemon has no editor to open, so the
/// refusal gives the lines that do the same by hand (docs/annotations.md).
fn text(step: &EditStep, selection: &Selection) -> Result<String, Failure> {
    step.text.clone().ok_or_else(|| {
        let target = selection
            .query_text()
            .map(|text| polylogue_render::shell_quote(&text))
            .unwrap_or_else(|| "SELECTION".into());
        Failure::bad_request(format!(
            "edit needs its new text; with no text a terminal opens your editor. Without one, write the text out with `polylogue {target} read --to FILE.md`, change the file, and check it in with `polylogue import FILE.md`"
        ))
    })
}

/// The rows as annotations, refusing any other row.
fn annotations(verb: &str, selection: &Selection) -> Result<Vec<RevisionReference>, Failure> {
    selection
        .references()?
        .iter()
        .map(|reference| match reference {
            Reference::Annotation(revision) => Ok(revision.clone()),
            other => Err(Failure::bad_request(format!(
                "{verb} acts on annotations; {other} is not one"
            ))),
        })
        .collect()
}

/// A write's outcome as a finished step with its undo line.
fn wrote(
    write: UserWrite,
    call: &StepCall<'_>,
    scope: Option<polylogue_store::ReadScopeToken>,
) -> Result<StepDone, Failure> {
    let mut committed = Vec::new();
    let answer = crate::daemon::service::submit_result(
        call.call.daemon,
        call.call.invocation,
        call.call.request_id.into(),
        write,
        scope,
        |receipt| {
            committed = receipt.written;
            receipt.outcome
        },
    );
    let mut done = StepDone::from(answer.map(StepAnswer::Wrote)).checked()?;
    done.undo
        .extend(undo_line(call.call.request_id, &committed)?);
    Ok(done)
}

pub(crate) struct Withdraw;
impl Verb for Withdraw {
    type Step = WithdrawStep;
    const NAME: &'static str = "withdraw";
    const ROWS: RowRule = RowRule::Things;
    const ACCESS: ToolAccess = ToolAccess::Write;
    fn run(
        step: WithdrawStep,
        selection: &Selection,
        call: &StepCall<'_>,
    ) -> Result<StepDone, Failure> {
        let targets = withdrawn(&step, selection)?;
        if targets.is_empty() {
            return Err(Failure::bad_request("targets must not be empty"));
        }
        wrote(
            UserWrite::WithdrawAnnotations {
                targets,
                reason: step.reason,
            },
            call,
            None,
        )
    }
    fn would(
        step: &WithdrawStep,
        selection: &Selection,
        _: &StepCall<'_>,
    ) -> Result<String, Failure> {
        withdrawn(step, selection)?;
        Ok(format!("would withdraw {}", selection.described()))
    }
}

/// The annotations and alternates a withdrawal names.
fn withdrawn(step: &WithdrawStep, selection: &Selection) -> Result<Vec<Reference>, Failure> {
    let references = selection.references()?;
    if let Some(other) = references.iter().find(|reference| {
        !matches!(
            reference,
            Reference::Annotation(_) | Reference::Alternate { .. }
        )
    }) {
        let address = other.to_string();
        let query = format!(
            "annotations where about:{}",
            polylogue_query::quoted(&address)
        );
        let request = command_surface::RunnableRequest::new(
            "rows",
            serde_json::json!({"selection":query, "all":true,
                "steps":[crate::protocol::line::Step::Withdraw(step.clone())]}),
            polylogue_reading::operation::new_run_id().to_string(),
        );
        let line = crate::commands::declarations()
            .cli_command(&request)
            .map_err(Failure::from)?;
        return Err(Failure::bad_request(format!(
            "withdraw needs annotations or alternates; {} is one of {} selected rows. To withdraw annotations about that row, use {}",
            polylogue_render::terminal::escape(&address), references.len(),
            polylogue_render::terminal::escape(&line),
        )).error.with_suggestion(command_surface::Suggestion::Request(request)).into());
    }
    Ok(references.to_vec())
}

pub(crate) struct Judge;
impl Verb for Judge {
    type Step = JudgeStep;
    const NAME: &'static str = "judge";
    const ROWS: RowRule = RowRule::Things;
    const ACCESS: ToolAccess = ToolAccess::Write;
    fn run(
        step: JudgeStep,
        selection: &Selection,
        call: &StepCall<'_>,
    ) -> Result<StepDone, Failure> {
        let (note, accepted) = verdict(&step, selection)?;
        let Reference::Annotation(reference) = note else {
            unreachable!("verdict checks the selection")
        };
        let reader = call.call.reader()?;
        let scope = reader.scope_token();
        let found = stored_annotation(&reader, reference)?;
        if !found.fact.author.is_agent() {
            return Err(Failure::bad_request(
                "only annotations agents wrote are accepted or rejected; revise or withdraw the operator's own",
            ));
        }
        drop(reader);
        let judgment = NewAnnotation {
            kind: AnnotationKind::Judgment,
            key: Some(if accepted { "accepted" } else { "rejected" }.into()),
            value: step.reason.map(Value::String),
            links: vec![NewAnnotationLink {
                role: LinkRole::About,
                reference: Reference::Annotation(RevisionReference::exact(
                    found.fact.root,
                    found.fact.id,
                )),
            }],
            valid_until: None,
        };
        wrote(
            UserWrite::Assert {
                annotations: vec![judgment],
            },
            call,
            Some(scope),
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
        [note @ Reference::Annotation(_)] => Ok((note, accepted)),
        [other] => Err(Failure::bad_request(format!(
            "judge accepts or rejects annotations; {other} is not one"
        ))),
        _ => Err(not_built_use("judge", "over several rows")),
    }
}
