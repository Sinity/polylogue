//! Direct Session input influence, never inferred from matching Message rows.
//! Catalog traversal and every receipt read share the caller's pinned Session.
use crate::{owner, source_rows, source_turn::SourceContext, state::claude};
use polylogue_model::settlement::RecordFailureId;
use polylogue_store::{
    Error, InputFormat,
    contributions::{SessionCatalog, SessionInputDependency, SessionView},
};
use std::time::Instant;
#[derive(Clone, Debug, Default)]
pub struct Qualification {
    pub record_errors: u64,
    pub interpreted_nonempty_inputs: u64,
    pub pending_inputs: u64,
    pub unresolved_inputs: u64,
    pub unavailable_inputs: u64,
}
fn add(value: &mut u64, n: u64) -> Result<(), Error> {
    *value = value.checked_add(n).ok_or(Error::Range)?;
    Ok(())
}
fn owner_error(error: owner::Error) -> Error {
    match error {
        owner::Error::Store(error) => error,
        error => Error::Invariant(error.to_string()),
    }
}
fn deadline(until: Instant) -> Result<(), Error> {
    if Instant::now() >= until {
        Err(Error::Cancelled)
    } else {
        Ok(())
    }
}
/// Currently qualifies direct Claude main-file inputs only. Other dependency
/// shapes are reported unresolved, never counted as irrelevant by row overlap.
/// Each input is visited once in immutable catalog order. Same input replays,
/// source/candidate fanout and historical citation paths cannot multiply errors.
/// Whole admitted effect cells remain the existing allocation/work unit.
pub fn qualify(
    session: &SessionView,
    until: Instant,
    allowance: usize,
) -> Result<Qualification, Error> {
    qualify_with_cancel(session, until, allowance, &mut || false)
}
pub fn qualify_with_cancel(
    session: &SessionView,
    until: Instant,
    allowance: usize,
    cancelled: &mut dyn FnMut() -> bool,
) -> Result<Qualification, Error> {
    let header = session.header()?;
    let plan = polylogue_model::query::SessionQuery {
        origin: Some(header.origin.as_str().into()),
        id: Some(header.id.as_str().into()),
        ..Default::default()
    };
    qualify_catalog(&session.catalog(), &plan, until, allowance, cancelled)
}
/// Qualify every possible retained contributor, including those that published
/// no Session or matching Message. EOF alone never becomes a query proof.
pub fn qualify_catalog(
    session: &SessionCatalog,
    plan: &polylogue_model::query::SessionQuery,
    until: Instant,
    allowance: usize,
    cancelled: &mut dyn FnMut() -> bool,
) -> Result<Qualification, Error> {
    let mut result = Qualification::default();
    let mut cursor = None;
    loop {
        deadline(until)?;
        if cancelled() {
            return Err(Error::Cancelled);
        }
        let Some((next, dependency)) = session.next_dependency(cursor.as_ref(), allowance)? else {
            break;
        };
        cursor = Some(next);
        // Format owns origin; a Codex rollout cannot contribute Claude rows.
        if dependency.format() == InputFormat::CodexRollout {
            if plan.origin.as_deref().is_some_and(|o| o != "codex") {
                continue;
            }
            add(&mut result.unresolved_inputs, 1)?;
            continue;
        }
        if plan.origin.as_deref().is_some_and(|o| o != "claude-code") {
            continue;
        }
        let Some(input) = dependency.registered() else {
            // Context remains available before disposable input_ref repair.
            match dependency.context() {
                [kind, stem] if kind == "claude-main" => {
                    if plan.id.as_deref().is_none_or(|target| {
                        polylogue_sources::claude_code::FileCtx::main(stem)
                            .direct_main_session()
                            .as_ref()
                            .is_some_and(|id| id.as_str() == target)
                    }) {
                        add(&mut result.pending_inputs, 1)?;
                    }
                }
                _ => add(&mut result.unresolved_inputs, 1)?,
            }
            continue;
        };
        let SourceContext::Claude(context) = owner::parser_context(&input).map_err(owner_error)?
        else {
            return Err(Error::Invariant("Claude format/context mismatch".into()));
        };
        match context.direct_main_session() {
            Some(id)
                if plan
                    .id
                    .as_deref()
                    .is_some_and(|target| id.as_str() != target) =>
            {
                continue;
            }
            Some(_) => {}
            None => {
                add(&mut result.unresolved_inputs, 1)?;
                continue;
            }
        }
        let Some(head) = dependency.head() else {
            add(&mut result.pending_inputs, 1)?;
            continue;
        };
        if head.token.basis != owner::parser_basis(&input)
            || head.parser_version != owner::parser_version(input.format)
        {
            add(&mut result.pending_inputs, 1)?;
            continue;
        }
        if head.counts.used > 0 {
            add(&mut result.interpreted_nonempty_inputs, 1)?;
        }
        let mut unavailable = dependency.failure().is_some();
        let receipt = session.dependency_cell(
            &dependency,
            polylogue_store::parser_state::SOURCE_ROWS_NAMESPACE,
            b"",
            16384,
        )?;
        match receipt {
            Some(bytes) => {
                let receipt = source_rows::decode(&bytes, head, &input).map_err(owner_error)?;
                if matches!(
                    receipt.rows.state,
                    polylogue_model::settlement::SourceRowState::Unavailable { .. }
                ) {
                    unavailable = true;
                }
            }
            None if head.status == "done" => {
                return Err(Error::Invariant(
                    "done Session dependency lost its settlement receipt".into(),
                ));
            }
            None => add(&mut result.pending_inputs, 1)?,
        }
        if unavailable {
            add(&mut result.unavailable_inputs, 1)?;
        }
        count_record_failures(
            session,
            &dependency,
            until,
            allowance,
            &mut result,
            cancelled,
        )?;
    }
    Ok(result)
}
fn count_record_failures(
    session: &SessionCatalog,
    dependency: &SessionInputDependency,
    until: Instant,
    allowance: usize,
    result: &mut Qualification,
    cancelled: &mut dyn FnMut() -> bool,
) -> Result<(), Error> {
    let head = dependency.head().ok_or(Error::Range)?;
    let mut previous: Option<RecordFailureId> = None;
    let mut count = 0u64;
    for step in 1..=head.token.step {
        deadline(until)?;
        if cancelled() {
            return Err(Error::Cancelled);
        }
        let key = serde_json::to_vec(&step).map_err(|e| Error::Invariant(e.to_string()))?;
        let bytes = session
            .dependency_cell(dependency, claude::EFFECTS, &key, allowance)?
            .ok_or_else(|| {
                Error::Invariant("missing Session dependency disposition receipt".into())
            })?;
        let effects: claude::Effects = serde_json::from_slice(&bytes)
            .map_err(|e| Error::Invariant(format!("Session dependency dispositions: {e}")))?;
        for record_start in effects.undecodable {
            deadline(until)?;
            if cancelled() {
                return Err(Error::Cancelled);
            }
            let id = RecordFailureId {
                input: dependency.id().into(),
                chain: head.token.chain,
                revision: 1,
                record_start,
            };
            if record_start >= head.parsed_upto || previous.as_ref().is_some_and(|old| old >= &id) {
                return Err(Error::Invariant(
                    "duplicated/out-of-range Session record failure identity".into(),
                ));
            }
            previous = Some(id);
            add(&mut count, 1)?;
        }
    }
    if count != head.counts.undecodable {
        return Err(Error::Invariant(
            "Session dependency failure identities disagree with accounting".into(),
        ));
    }
    add(&mut result.record_errors, count)
}
