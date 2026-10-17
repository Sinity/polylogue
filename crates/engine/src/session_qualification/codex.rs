//! Finite rollout membership from provider state and exact selected source proofs.
use super::*;
use crate::state::codex::RESUME_NAMESPACE;
use polylogue_sources::codex::keyed::{Cell, Key, Resume};

fn provider_cell(
    catalog: &SessionCatalog,
    dependency: &SessionInputDependency,
    key: Key,
    allowance: usize,
) -> Result<Cell, Error> {
    let bytes = catalog
        .dependency_cell(
            dependency,
            &format!("codex.{}", key.family()),
            &key.ordered_bytes(),
            allowance,
        )?
        .ok_or_else(|| Error::Invariant("missing Codex Session membership cell".into()))?;
    let cell: Cell = serde_json::from_slice(&bytes)
        .map_err(|e| Error::Invariant(format!("Codex membership cell: {e}")))?;
    if !cell.valid_for(&key) {
        return Err(Error::Invariant(
            "Codex membership cell/key mismatch".into(),
        ));
    }
    Ok(cell)
}
pub(super) fn qualify(
    catalog: &SessionCatalog,
    dependency: &SessionInputDependency,
    plan: &polylogue_model::query::SessionQuery,
    exact: Option<&str>,
    until: Instant,
    allowance: usize,
    result: &mut Qualification,
    cancelled: &mut dyn FnMut() -> bool,
) -> Result<(), Error> {
    let Some(input) = dependency.registered() else {
        add(&mut result.pending_inputs, 1)?;
        return Ok(());
    };
    let Some(head) = dependency.head() else {
        add(&mut result.pending_inputs, 1)?;
        return Ok(());
    };
    let Some(bytes) = catalog.dependency_cell(dependency, RESUME_NAMESPACE, b"", allowance)? else {
        add(&mut result.pending_inputs, 1)?;
        return Ok(());
    };
    let resume: Resume = serde_json::from_slice(&bytes)
        .map_err(|e| Error::Invariant(format!("Codex Session resume: {e}")))?;
    let Some(source_session) = resume.session() else {
        add(&mut result.unresolved_inputs, 1)?;
        return Ok(());
    };
    if plan
        .id
        .as_deref()
        .is_some_and(|id| source_session.id.as_str() != id)
    {
        return Ok(());
    }
    result.basis.push(source_rows::basis(&input));
    let mut entry = json!({"id":dependency.id(),"format":dependency.format().name(),"parser_version":head.parser_version});
    if let Some(path) = catalog.dependency_first_path(dependency, allowance)? {
        entry["path"] = json!(path);
    }
    result.inputs.push(entry);
    if head.acquired_upto != input.input.length
        || head.token.basis != owner::parser_basis(&input)
        || head.parser_version != owner::parser_version(input.format)
        || resume.parsed_upto != head.parsed_upto
    {
        add(&mut result.pending_inputs, 1)?;
        result.known_rows_readable = false;
        return Ok(());
    }
    if head.counts.used > 0 {
        add(&mut result.interpreted_nonempty_inputs, 1)?;
    }
    if resume.normalization_incomplete() || dependency.failure().is_some() {
        result.known_rows_readable = false;
        result
            .unsupported
            .insert("Codex source normalization or owner failure remains unresolved".into());
    }
    add(&mut result.record_errors, head.counts.undecodable)?;
    if head.counts.unrecognized > 0 {
        result.unsupported.insert(
            "Unrecognized Codex records leave population and neighboring products unknown".into(),
        );
    }
    let receipt = catalog.dependency_cell(
        dependency,
        polylogue_store::parser_state::SOURCE_ROWS_NAMESPACE,
        b"",
        16384,
    )?;
    match receipt {
        Some(bytes) => {
            let receipt = source_rows::decode(&bytes, head, &input).map_err(owner_error)?;
            if receipt.unsupported_effects {
                result.unsupported.insert(
                    "Codex non-Message effects remain unsupported for complete Session reads"
                        .into(),
                );
            }
            match receipt.rows.state {
                polylogue_model::settlement::SourceRowState::Settled => {}
                polylogue_model::settlement::SourceRowState::Pending { .. } => {
                    add(&mut result.pending_inputs, 1)?;
                }
                polylogue_model::settlement::SourceRowState::Unavailable {
                    usable_result, ..
                } => {
                    add(&mut result.unavailable_inputs, 1)?;
                    result.known_rows_readable &= usable_result;
                }
            }
        }
        None => {
            add(&mut result.pending_inputs, 1)?;
            result.known_rows_readable = false;
        }
    }
    if exact.is_none() {
        return Ok(());
    }
    match &result.metadata {
        Some(old) if old != source_session => {
            result
                .unsupported
                .insert("Competing Codex Session metadata remains unresolved".into());
            result.known_rows_readable = false;
        }
        None => result.metadata = Some(source_session.clone()),
        _ => {}
    }
    // Rollout order is finite and explicit. Active-path inference is a separate
    // product; transcript readers requesting all branches can use these rows.
    result.own_topology_unknown = true;
    for ordinal in 0..resume.message_count() {
        deadline(until)?;
        if cancelled() {
            return Err(Error::Cancelled);
        }
        let Cell::Header { message, .. } =
            provider_cell(catalog, dependency, Key::Message(ordinal), allowance)?
        else {
            return Err(Error::Range);
        };
        let Cell::EvidenceHead(evidence) =
            provider_cell(catalog, dependency, Key::EvidenceHead(ordinal), allowance)?
        else {
            return Err(Error::Range);
        };
        if message.id.session() != &source_session.id {
            return Err(Error::Invariant(
                "Codex retained Message/Session mismatch".into(),
            ));
        }
        let selected = catalog.dependency_selected_projection(
            dependency,
            &owner::canonical_key(ordinal).map_err(owner_error)?,
            allowance,
        )?;
        let proved = match selected {
            Some(source) => crate::canonical_message::current_codex_projection(
                &source,
                &head.token,
                &message.id,
                &evidence,
                input.binding,
            )?,
            None => false,
        };
        if !proved {
            result.known_rows_readable = false;
            result
                .unsupported
                .insert("Known Codex Message lacks its current selected source projection".into());
        }
        let id = message.id.as_str();
        if !result.expected.contains_key(id) {
            admit_membership(result, id, false, allowance)?;
        }
        let sequence = u64::try_from(ordinal).map_err(|_| Error::Range)?;
        if result
            .expected
            .insert(id.to_owned(), sequence)
            .is_some_and(|old| old != sequence)
        {
            result.known_rows_readable = false;
            result
                .unsupported
                .insert("Independent Codex inputs disagree about Message order".into());
        }
    }
    Ok(())
}
