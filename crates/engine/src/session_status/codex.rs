//! Codex rollout membership from provider state and the selected sources.
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
struct FlatReads<'a> {
    catalog: &'a SessionCatalog,
    dependency: &'a SessionInputDependency,
    until: Instant,
    cancelled: &'a mut dyn FnMut() -> bool,
    allowance: usize,
    used: usize,
}
impl FlatReads<'_> {
    fn cell(&mut self, key: &[u8], maximum: usize) -> Result<Option<Vec<u8>>, Error> {
        deadline(self.until)?;
        if (self.cancelled)() {
            return Err(Error::Cancelled);
        }
        let namespace = crate::state::codex::EFFECTS_NAMESPACE;
        self.charge(namespace.len().saturating_add(key.len()))?;
        let bytes = self.catalog.dependency_cell(
            self.dependency,
            namespace,
            key,
            (self.allowance - self.used).min(maximum),
        )?;
        if let Some(bytes) = &bytes {
            self.charge(bytes.len())?;
        }
        Ok(bytes)
    }
    fn charge(&mut self, bytes: usize) -> Result<(), Error> {
        self.used = self.used.saturating_add(bytes);
        if self.used > self.allowance {
            return Err(Error::ParserAdmission {
                resource: "Codex active event proof",
                required: self.used as u64,
            });
        }
        Ok(())
    }
}
fn flat_effects(
    catalog: &SessionCatalog,
    dependency: &SessionInputDependency,
    session: &polylogue_model::SessionId,
    until: Instant,
    allowance: usize,
    cancelled: &mut dyn FnMut() -> bool,
) -> Result<bool, Error> {
    use crate::state::codex::{FlatSessionEvents, FlatSessionStep};
    let head = dependency.head().ok_or(Error::Range)?;
    let mut reads = FlatReads {
        catalog,
        dependency,
        until,
        cancelled,
        allowance,
        used: 0,
    };
    let mut events = FlatSessionEvents::new(session);
    for step in 1..=u64::try_from(head.token.step).map_err(|_| Error::Range)? {
        let bytes = reads
            .cell(&FlatSessionStep::header_key(step), 256)?
            .ok_or_else(|| Error::Invariant("missing Codex effects header".into()))?;
        let Some(proof) = FlatSessionStep::decode(&bytes, step)? else {
            return Ok(false);
        };
        // Steps without events are fast. Steps with events read every event
        // plus the probe past the last one.
        if proof.events() == 0 {
            continue;
        }
        for ordinal in 0..=proof.events() {
            let bytes = reads.cell(&proof.event_key(ordinal)?, allowance)?;
            if let Some(event) = proof.event(ordinal, bytes.as_deref())? {
                if !events.accepts(&event) {
                    return Ok(false);
                }
            }
        }
    }
    Ok(true)
}
pub(super) fn check(
    catalog: &SessionCatalog,
    dependency: &SessionInputDependency,
    plan: &polylogue_model::query::SessionQuery,
    exact: Option<&str>,
    until: Instant,
    allowance: usize,
    result: &mut SessionStatus,
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
    result.add_contributor(source_rows::computed_from(&input), true, allowance)?;
    let mut entry = json!({"id":dependency.id(),"format":dependency.format().name(),"parser_version":head.parser_version});
    if let Some(path) = catalog.dependency_first_path(dependency, allowance)? {
        entry["path"] = json!(path);
    }
    result.include_input(entry, allowance)?;
    if head.acquired_upto != input.input.length
        || head.token.basis != owner::parser_identity(&input)
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
    // A complete flat log with no topology-changing events is entirely on the
    // path. Unknown controls, lineage and conflicts are not, even when the
    // remaining Messages look linear.
    let mut flat = head.counts.unrecognized == 0
        && head.counts.undecodable == 0
        && !resume.normalization_incomplete()
        && dependency.failure().is_none()
        && flat_effects(
            catalog,
            dependency,
            &source_session.id,
            until,
            allowance,
            cancelled,
        )?;
    let mut active = Vec::new();
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
                "Codex stored Message/Session mismatch".into(),
            ));
        }
        let selected = selected_current_projection(
            catalog,
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
        flat &= proved;
        if !proved {
            result.known_rows_readable = false;
            result
                .unsupported
                .insert("Known Codex Message lacks its current selected source projection".into());
        }
        let id = message.id.as_str();
        active.push(id.to_owned());
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
    if flat {
        for id in active {
            if !result.active_messages.contains(&id) {
                admit_membership(result, &id, true, allowance)?;
                result.active_messages.insert(id);
            }
        }
    } else {
        result.own_topology_unknown = true;
    }
    Ok(())
}

#[cfg(test)]
#[path = "codex/tests.rs"]
mod tests;
