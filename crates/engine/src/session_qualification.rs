//! Direct Session input influence, never inferred from matching Message rows.
//! Catalog traversal and every receipt read share the caller's pinned Session.
use crate::{owner, source_rows, source_turn::SourceContext, state::claude};
use polylogue_model::settlement::RecordFailureId;
use polylogue_model::{Session, settlement::SourceRowBasis};
use polylogue_sources::claude_code::keyed::{ResponseCell, SessionCell};
use polylogue_store::{
    Error, InputFormat,
    contributions::{SessionCatalog, SessionInputDependency, SessionView},
};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    time::Instant,
};
#[derive(Clone, Debug, Default)]
pub struct Qualification {
    pub record_errors: u64,
    /// Counted interpreted-text repairs, separate from exclusive record outcomes.
    pub text_repairs: u64,
    pub interpreted_nonempty_inputs: u64,
    pub pending_inputs: u64,
    pub unresolved_inputs: u64,
    pub unavailable_inputs: u64,
    pub unsupported: BTreeSet<String>,
    pub metadata: Option<Session>,
    pub expected: BTreeMap<String, u64>,
    pub basis: Vec<SourceRowBasis>,
    pub inputs: Vec<Value>,
    pub up_to_date: bool,
    pub active_messages: BTreeSet<String>,
    document: bool,
    exact: bool,
    metadata_conflicts: BTreeSet<String>,
    own_topology_unknown: bool,
    own_counting_unknown: bool,
    membership_bytes: usize,
    known_rows_readable: bool,
}
impl Qualification {
    pub fn header(
        &self,
        header: &polylogue_store::contributions::SessionHeader,
    ) -> polylogue_store::contributions::SessionHeader {
        let mut h = header.clone();
        if let Some(s) = &self.metadata {
            h.title = s.title.clone();
            h.cwd = s.cwd.clone();
            h.branch = s.branch.clone();
            h.harness = s.harness.clone();
            h.harness_version = s.harness_version.clone();
            h.started_at = s.started_at.and_then(|t| t.at()).map(|t| t.0);
            h.last_activity_at = s.last_activity_at.and_then(|t| t.at()).map(|t| t.0);
        }
        h
    }
    /// Known catalog membership/order and every interpreted row have current
    /// source-row audit authority. This is independent of active-path knowledge
    /// and never upgrades unavailable population/record qualification.
    pub fn known_rows_readable(&self) -> bool {
        self.exact
            && self.known_rows_readable
            && self.pending_inputs == 0
            && self.unresolved_inputs == 0
    }
    pub fn active_path_known(&self) -> bool {
        !self.own_topology_unknown
    }
    pub fn complete(&self) -> bool {
        self.record_errors == 0
            && self.pending_inputs == 0
            && self.unresolved_inputs == 0
            && self.unavailable_inputs == 0
            && self.unsupported.is_empty()
    }
    pub fn notes(&self) -> Vec<String> {
        let mut notes = self.unsupported.iter().cloned().collect::<Vec<_>>();
        for (count, detail) in [
            (
                self.record_errors,
                "undecodable complete records in relevant retained inputs",
            ),
            (
                self.text_repairs,
                "unpaired UTF-16 code units repaired in interpreted source text",
            ),
            (
                self.pending_inputs,
                "relevant retained inputs have pending interpretation or row effects",
            ),
            (
                self.unavailable_inputs,
                "relevant retained inputs have unavailable interpretation or effects",
            ),
            (
                self.unresolved_inputs,
                "retained input dependency shapes remain unresolved",
            ),
        ] {
            if count != 0 {
                notes.push(format!("{count} {detail}"));
            }
        }
        notes
    }
    pub fn result(&self, catalog: &SessionCatalog) -> Value {
        let mut unknown = json!({});
        for (name, n) in [
            ("record_errors", self.record_errors),
            ("pending_inputs", self.pending_inputs),
            ("unavailable_inputs", self.unavailable_inputs),
            ("unresolved_inputs", self.unresolved_inputs),
        ] {
            if n > 0 {
                unknown[name] = json!(n);
            }
        }
        json!({"outcome":if self.complete() {"ok"} else {"degraded"},
            "notes":self.notes(), "unknown":unknown,
            "diagnostics":{"text_repairs":self.text_repairs},
            "settlement":{"scope":if self.exact{"exact_session"}else{"session_selection"},"operation":if self.document{"normalized_read"}else{"selection"},"state":if self.complete(){"settled"}else if self.pending_inputs>0{"pending"}else if self.unavailable_inputs>0||self.record_errors>0{"unavailable"}else{"unsupported"},
                "target_basis":{"archive_counter":catalog.source_archive_counter(),"recipe":"native-session-v2","configuration":if self.basis.iter().any(|b|b.original_hash.is_none()){"retained-native-context-v1"}else{"fixed-immutable-native-context-v1"},"inputs":self.basis},
                "serving_view":{"index_generation":catalog.position().generation,"index_counter":catalog.position().counter,"archive_counter":catalog.source_archive_counter()}}})
    }
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
    qualify_session(session, until, allowance, cancelled, false)
}
/// Exact member reads preserve unavailable Session context while independently
/// checking the selected Message's retained body witnesses at handoff.
pub fn qualify_member_with_cancel(
    session: &SessionView,
    until: Instant,
    allowance: usize,
    cancelled: &mut dyn FnMut() -> bool,
) -> Result<Qualification, Error> {
    qualify_session(session, until, allowance, cancelled, true)
}
fn qualify_session(
    session: &SessionView,
    until: Instant,
    allowance: usize,
    cancelled: &mut dyn FnMut() -> bool,
    member: bool,
) -> Result<Qualification, Error> {
    let header = session.header()?;
    let plan = polylogue_model::query::SessionQuery {
        origin: Some(header.origin.as_str().into()),
        id: Some(header.id.as_str().into()),
        ..Default::default()
    };
    qualify_operation(
        &session.catalog(),
        &plan,
        until,
        allowance,
        cancelled,
        true,
        member,
    )
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
    qualify_operation(session, plan, until, allowance, cancelled, false, false)
}
fn qualify_operation(
    session: &SessionCatalog,
    plan: &polylogue_model::query::SessionQuery,
    until: Instant,
    allowance: usize,
    cancelled: &mut dyn FnMut() -> bool,
    document: bool,
    member: bool,
) -> Result<Qualification, Error> {
    let mut result = Qualification {
        up_to_date: true,
        document,
        known_rows_readable: true,
        ..Default::default()
    };
    let exact = plan.id.as_deref().filter(|_| {
        plan.text.is_empty() && plan.origin.as_deref().is_none_or(|o| o == "claude-code")
    });
    result.exact = exact.is_some();
    let mut cursor = None;
    let mut dependency_bytes = 0usize;
    loop {
        deadline(until)?;
        if cancelled() {
            return Err(Error::Cancelled);
        }
        let Some((next, dependency)) = session.next_dependency(cursor.as_ref(), allowance)? else {
            break;
        };
        cursor = Some(next);
        dependency_bytes = dependency_bytes.saturating_add(
            dependency.id().len()
                + dependency.context().iter().map(String::len).sum::<usize>()
                + 256,
        );
        if dependency_bytes > allowance {
            return Err(Error::ParserAdmission {
                resource: "Session dependency metadata",
                required: dependency_bytes as u64,
            });
        }
        // Global freshness is independent of this operation's dependency scope.
        // A retained unavailable interpretation is caught up, but not complete.
        let caught_up =
            if let (Some(input), Some(head)) = (dependency.registered(), dependency.head()) {
                if head.acquired_upto == input.input.length
                    && head.token.basis == owner::parser_basis(&input)
                    && head.parser_version == owner::parser_version(input.format)
                {
                    match session.dependency_cell(
                        &dependency,
                        polylogue_store::parser_state::SOURCE_ROWS_NAMESPACE,
                        b"",
                        16384,
                    )? {
                        Some(bytes) => {
                            source_rows::decode(&bytes, head, &input).map_err(owner_error)?;
                            true
                        }
                        None => false,
                    }
                } else {
                    false
                }
            } else {
                false
            };
        result.up_to_date &= caught_up;
        // Format owns origin; a Codex rollout cannot contribute Claude rows.
        if matches!(
            dependency.format(),
            InputFormat::CodexRollout
                | InputFormat::CodexHistorySqlite
                | InputFormat::CodexPromptHistory
        ) {
            if document {
                add(&mut result.unresolved_inputs, 1)?;
                continue;
            }
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
                    if document
                        || plan.id.as_deref().is_none_or(|target| {
                            polylogue_sources::claude_code::FileCtx::main(stem)
                                .direct_main_session()
                                .as_ref()
                                .is_some_and(|id| id.as_str() == target)
                        })
                    {
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
                if document {
                    if !caught_up {
                        result.basis.push(source_rows::basis(&input));
                        add(&mut result.pending_inputs, 1)?;
                    } else {
                        let head = dependency.head().ok_or(Error::Range)?;
                        let mut possible =
                            head.counts.unrecognized > 0 || head.counts.undecodable > 0;
                        for step in 1..=head.token.step {
                            deadline(until)?;
                            if cancelled() {
                                return Err(Error::Cancelled);
                            }
                            let effects: claude::Effects =
                                cell(session, &dependency, claude::EFFECTS, &step, allowance)?
                                    .ok_or_else(|| {
                                        Error::Invariant("missing foreign Session effects".into())
                                    })?;
                            if effects.links.iter().any(|link| {
                                Some(link.child.as_str()) == exact
                                    || Some(link.parent.as_str()) == exact
                            }) {
                                possible = true;
                            }
                            for pending in effects.pending_messages {
                                for item in 0..pending.items {
                                    deadline(until)?;
                                    if cancelled() {
                                        return Err(Error::Cancelled);
                                    }
                                    let contribution:polylogue_sources::claude_code::keyed::Contribution=cell(session,&dependency,claude::CONTRIBUTION,&(pending.message.session().as_str(),pending.message.as_str(),item),allowance)?.ok_or_else(||Error::Invariant("missing foreign Session contribution".into()))?;
                                    if contribution.blocks.iter().any(|block|matches!(block,polylogue_model::BlockContent::Text{text} if exact.is_some_and(|id|text.contains(id) || id.strip_prefix("claude-code:").is_some_and(|native|text.contains(native))))) {possible=true;}
                                }
                            }
                        }
                        if possible {
                            result.basis.push(source_rows::basis(&input));
                            result.unsupported.insert("A foreign main input may contribute Session links or unknown records to this document".into());
                            count_record_failures(
                                session,
                                &dependency,
                                until,
                                allowance,
                                &mut result,
                                cancelled,
                            )?;
                        }
                    }
                }
                continue;
            }
            Some(_) => {}
            None => {
                add(&mut result.unresolved_inputs, 1)?;
                continue;
            }
        }
        if let Err(error) = session.verify_dependency_bodies(&dependency) {
            let unavailable_body = matches!(&error, Error::Blob(_) | Error::UnknownBlob)
                || matches!(&error, Error::Invariant(detail) if detail == "required Claude retained body metadata unavailable");
            if !member || !unavailable_body {
                return Err(error);
            }
            add(&mut result.unavailable_inputs, 1)?;
            result.unsupported.insert(format!(
                "Required body unavailable in Session context: {error}"
            ));
        }
        result.basis.push(source_rows::basis(&input));
        result.own_topology_unknown |= !caught_up;
        let Some(head) = dependency.head() else {
            add(&mut result.pending_inputs, 1)?;
            continue;
        };
        if head.acquired_upto != input.input.length
            || head.token.basis != owner::parser_basis(&input)
            || head.parser_version != owner::parser_version(input.format)
        {
            add(&mut result.pending_inputs, 1)?;
            continue;
        }
        result.own_topology_unknown |= head.counts.unrecognized > 0 || head.counts.undecodable > 0;
        if head.counts.used > 0 {
            add(&mut result.interpreted_nonempty_inputs, 1)?;
        }
        let mut unavailable = dependency.failure().is_some();
        if unavailable {
            result.known_rows_readable = false;
        }
        let receipt = session.dependency_cell(
            &dependency,
            polylogue_store::parser_state::SOURCE_ROWS_NAMESPACE,
            b"",
            16384,
        )?;
        match receipt {
            Some(bytes) => {
                let receipt = source_rows::decode(&bytes, head, &input).map_err(owner_error)?;
                // The existing row audit has checked every known canonical
                // Message against this exact retained receipt. A bad record
                // alone leaves topology/population unknown, not those rows.
                // Unsupported effects, unrecognized records and owner failures
                // cannot acquire that narrower permission by matching counts.
                let readable = !receipt.unsupported_effects && head.counts.unrecognized == 0;
                match receipt.rows.state {
                    polylogue_model::settlement::SourceRowState::Unavailable {
                        usable_result,
                        record_errors,
                        ..
                    } => {
                        unavailable = true;
                        result.known_rows_readable &=
                            readable && usable_result && record_errors > 0;
                    }
                    polylogue_model::settlement::SourceRowState::Pending { .. } => {
                        add(&mut result.pending_inputs, 1)?
                    }
                    polylogue_model::settlement::SourceRowState::Settled => {
                        result.known_rows_readable &= readable;
                    }
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
        if let Some(target) = exact {
            qualify_shape(
                session,
                &dependency,
                target,
                until,
                allowance,
                &mut result,
                cancelled,
            )?;
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
    if let Some(target) = exact {
        let id = polylogue_model::SessionId::from_stored(target)
            .map_err(|e| Error::Invariant(e.to_string()))?;
        if session.has_other_session_products(&id)? {
            result.unsupported.insert("Authored annotations or non-Message Session products require an unsupported reader".into());
        }
        if let Some(view) = session.session(&id, allowance)? {
            let mut seen = BTreeSet::new();
            let mut cursor = None;
            let mut started: Option<i64> = None;
            let mut last: Option<i64> = None;
            let mut models = BTreeMap::<String, u64>::new();
            loop {
                deadline(until)?;
                if cancelled() {
                    return Err(Error::Cancelled);
                }
                match view.next(cursor.as_ref(), allowance)? {
                    polylogue_store::contributions::SessionStep::Ready {
                        cursor: next,
                        sequence,
                        message,
                    } => {
                        if result.expected.get(message.id().as_str()) != Some(&sequence) {
                            result.own_topology_unknown = true;
                            result.known_rows_readable = false;
                            result.unsupported.insert("Selected Message membership/order is not proved by the retained source contributions".into());
                        }
                        if message.header()?.written_by == polylogue_model::WrittenBy::Model
                            && !result.own_counting_unknown
                        {
                            use polylogue_model::merge::{MessageField, Slot};
                            match message.field(&Slot::Message(MessageField::Model), allowance)? {
                                Some(polylogue_store::contributions::FieldRead::Value(value)) => {
                                    let range = value.range(0, allowance)?;
                                    if !range.eof {
                                        return Err(Error::ParserAdmission {
                                            resource: "Session model name",
                                            required: range.next_offset.saturating_add(1),
                                        });
                                    }
                                    let name: String = serde_json::from_slice(&range.bytes)
                                        .map_err(|e| {
                                            Error::Invariant(format!("Session model name: {e}"))
                                        })?;
                                    add(models.entry(name).or_default(), 1)?;
                                }
                                _ => {
                                    result.unsupported.insert("Model Message counts require a resolved recorded model name".into());
                                }
                            }
                        }
                        if let Some(at) = message.provider_time()? {
                            started = Some(started.map_or(at, |v| v.min(at)));
                            last = Some(last.map_or(at, |v| v.max(at)));
                        }
                        seen.insert(message.id().as_str().to_owned());
                        cursor = Some(next);
                    }
                    polylogue_store::contributions::SessionStep::Pending {
                        cursor: next, ..
                    } => {
                        add(&mut result.pending_inputs, 1)?;
                        cursor = Some(next);
                    }
                    polylogue_store::contributions::SessionStep::UnresolvedOrder { .. } => {
                        result.own_topology_unknown = true;
                        result.known_rows_readable = false;
                        result
                            .unsupported
                            .insert("Selected Message order is unresolved".into());
                        break;
                    }
                    polylogue_store::contributions::SessionStep::End { .. } => break,
                }
            }
            if seen.len() != result.expected.len() {
                result.own_topology_unknown = true;
                result.known_rows_readable = false;
                result.unsupported.insert(
                    "Retained source Message membership differs from the visible selected catalog"
                        .into(),
                );
            }
            if let Some(metadata) = &mut result.metadata {
                let time = |at| polylogue_model::ProviderTime::Intrinsic {
                    at: polylogue_model::Timestamp(at),
                    precision: polylogue_model::Precision::Instant,
                    until: None,
                };
                metadata.started_at = (!result.own_counting_unknown)
                    .then_some(started)
                    .flatten()
                    .map(time);
                metadata.last_activity_at = (!result.own_counting_unknown)
                    .then_some(last)
                    .flatten()
                    .map(time);
                metadata.models =
                    (!result.own_counting_unknown && !models.is_empty()).then_some(models);
                if metadata
                    .title
                    .as_deref()
                    .is_none_or(|title| title.is_empty() || title == "New chat")
                {
                    result.unsupported.insert(
                        "Display title requires the archive-wide preamble/title reduction".into(),
                    );
                }
            } else {
                result
                    .unsupported
                    .insert("Session metadata has no interpreted source owner".into());
            }
        } else if result.metadata.is_some() || !result.expected.is_empty() {
            result.known_rows_readable = false;
            result
                .unsupported
                .insert("Retained Session contributions have no published catalog owner".into());
        }
    }
    if exact.is_none() && result.interpreted_nonempty_inputs > 0 {
        result.unsupported.insert("Complete nonempty settlement is supported only for an exact Claude Code Session without prose search".into());
    }
    if result.own_topology_unknown {
        result.active_messages.clear();
    }
    Ok(result)
}
fn cell<T: serde::de::DeserializeOwned>(
    catalog: &SessionCatalog,
    dependency: &SessionInputDependency,
    namespace: &str,
    key: &impl serde::Serialize,
    allowance: usize,
) -> Result<Option<T>, Error> {
    let key = serde_json::to_vec(key).map_err(|e| Error::Invariant(e.to_string()))?;
    catalog
        .dependency_cell(dependency, namespace, &key, allowance)?
        .map(|bytes| {
            serde_json::from_slice(&bytes)
                .map_err(|e| Error::Invariant(format!("Session dependency cell: {e}")))
        })
        .transpose()
}
/// Prove the supported shape from retained interpretation, never from the rows
/// a search happened to find. Events, lineage/copy and ambiguous metadata keep
/// their existing degraded qualification until their connected products ship.
fn qualify_shape(
    catalog: &SessionCatalog,
    dependency: &SessionInputDependency,
    target: &str,
    until: Instant,
    allowance: usize,
    result: &mut Qualification,
    cancelled: &mut dyn FnMut() -> bool,
) -> Result<(), Error> {
    let context = dependency.context();
    let [_, native] = context else { return Ok(()) };
    if let Some(source) =
        cell::<SessionCell>(catalog, dependency, claude::SESSION, native, allowance)?
    {
        if source.session.id.as_str() != target {
            return Err(Error::Invariant(
                "Session source identity disagrees with context".into(),
            ));
        }
        let mut input = json!({"id":dependency.id(),"format":dependency.format().name(),"parser_version":dependency.head().ok_or(Error::Range)?.parser_version});
        if let Some(path) = catalog.dependency_first_path(dependency, allowance)? {
            input["path"] = json!(path);
        }
        result.inputs.push(input);
        if let Some(old) = &mut result.metadata {
            for (name, old_value, new_value) in [
                ("title", &mut old.title, &source.session.title),
                ("cwd", &mut old.cwd, &source.session.cwd),
                ("branch", &mut old.branch, &source.session.branch),
                (
                    "harness_version",
                    &mut old.harness_version,
                    &source.session.harness_version,
                ),
            ] {
                if result.metadata_conflicts.contains(name) {
                    continue;
                }
                match (old_value.as_ref(), new_value) {
                    (None, Some(value)) => *old_value = Some(value.clone()),
                    (Some(a), Some(b)) if a != b => {
                        result.metadata_conflicts.insert(name.into());
                        *old_value = None;
                        result.unsupported.insert(
                            "Competing Session metadata requires provider-time/route reduction"
                                .into(),
                        );
                    }
                    _ => {}
                }
            }
        } else {
            result.metadata = Some(source.session);
        }
    }
    let head = dependency.head().ok_or(Error::Range)?;
    let mut pending = BTreeMap::new();
    let mut pending_bytes = 0usize;
    for step in 1..=head.token.step {
        deadline(until)?;
        if cancelled() {
            return Err(Error::Cancelled);
        }
        let effects: claude::Effects =
            cell(catalog, dependency, claude::EFFECTS, &step, allowance)?
                .ok_or_else(|| Error::Invariant("missing Session effect receipt".into()))?;
        if !effects.links.is_empty() || !effects.events.is_empty() {
            result.unsupported.insert(
                "Linked/subagent and event products remain unsupported for complete Session reads"
                    .into(),
            );
        }
        for p in effects.pending_messages {
            if p.message.session().as_str() == target {
                if !pending.contains_key(p.message.as_str()) {
                    pending_bytes = pending_bytes.saturating_add(p.message.as_str().len() + 128);
                }
                if pending_bytes > allowance {
                    return Err(Error::ParserAdmission {
                        resource: "Session retained Message membership",
                        required: pending_bytes as u64,
                    });
                }
                pending.insert(p.message.as_str().to_owned(), p);
            }
        }
    }
    let mut tree = BTreeMap::new();
    for (message, p) in pending {
        deadline(until)?;
        if cancelled() {
            return Err(Error::Cancelled);
        }
        let response: ResponseCell = cell(
            catalog,
            dependency,
            claude::RESPONSE,
            &(
                &p.session_native,
                p.response.role,
                &p.response.response,
                &p.response.request,
            ),
            allowance,
        )?
        .ok_or_else(|| Error::Invariant("missing retained Session response".into()))?;
        if response.message.id.as_str() != message {
            return Err(Error::Invariant(
                "retained Session response identity mismatch".into(),
            ));
        }
        if response.message.written_by == polylogue_model::WrittenBy::Unknown {
            result
                .unsupported
                .insert("An unclassified Message author remains unresolved".into());
        }
        for item in 0..response.items {
            deadline(until)?;
            if cancelled() {
                return Err(Error::Cancelled);
            }
            let contribution: polylogue_sources::claude_code::keyed::Contribution = cell(
                catalog,
                dependency,
                claude::CONTRIBUTION,
                &(target, &message, item),
                allowance,
            )?
            .ok_or_else(|| Error::Invariant("missing retained Session contribution".into()))?;
            for block in &contribution.blocks {
                match block {
                    polylogue_model::BlockContent::Text { text }
                        if text.split_whitespace().any(|word| {
                            word.split_once(':').is_some_and(|(origin, tail)| {
                                !tail.is_empty() && polylogue_model::Origin::new(origin).is_ok()
                            })
                        }) =>
                    {
                        result.unsupported.insert("Prose reference candidates require the unimplemented reference-link reduction".into());
                    }
                    polylogue_model::BlockContent::ToolCall(call)
                        if call
                            .tool
                            .as_deref()
                            .is_some_and(|tool| matches!(tool, "Agent" | "Task")) =>
                    {
                        result
                            .unsupported
                            .insert("Agent/Task calls require linked subagent products".into());
                    }
                    _ => {}
                }
            }
        }
        if response.message.copy_origin.is_some() || response.message.inherited_from.is_some() {
            result.own_counting_unknown = true;
            result.unsupported.insert(
                "Inherited/copy counting and original operation scope remain unsupported".into(),
            );
        }
        tree.insert(
            p.ordinal,
            (
                message.clone(),
                response
                    .message
                    .parent
                    .as_ref()
                    .map(|id| id.as_str().to_owned()),
            ),
        );
        if !result.expected.contains_key(&message) {
            admit_membership(result, &message, false, allowance)?;
        }
        if result
            .expected
            .insert(message, p.ordinal)
            .is_some_and(|old| old != p.ordinal)
        {
            result.own_topology_unknown = true;
            result.known_rows_readable = false;
            result
                .unsupported
                .insert("Independent inputs disagree about Session Message order".into());
        }
    }
    let mut previous = None;
    let mut linear = true;
    let mut active = Vec::new();
    for (_, (message, parent)) in tree {
        if parent != previous {
            linear = false;
            result.unsupported.insert(
                "Branched or incomplete parent topology requires active-path reduction".into(),
            );
        }
        active.push(message.clone());
        previous = Some(message);
    }
    if linear {
        for message in active {
            if !result.active_messages.contains(&message) {
                admit_membership(result, &message, true, allowance)?;
                result.active_messages.insert(message);
            }
        }
    } else {
        result.own_topology_unknown = true;
    }
    Ok(())
}
fn admit_membership(
    result: &mut Qualification,
    id: &str,
    active: bool,
    allowance: usize,
) -> Result<(), Error> {
    let required = result
        .membership_bytes
        .saturating_add(id.len())
        .saturating_add(if active { 32 } else { 128 });
    if required > allowance {
        return Err(Error::ParserAdmission {
            resource: "Session cumulative Message membership",
            required: required as u64,
        });
    }
    result.membership_bytes = required;
    Ok(())
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
    let mut previous_repair = None;
    let mut repairs = 0u64;
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
        for repair in effects.text_repairs {
            deadline(until)?;
            if cancelled() {
                return Err(Error::Cancelled);
            }
            let position = (repair.record, repair.byte_offset);
            if repair.count == 0
                || repair.record > repair.byte_offset
                || repair.byte_offset >= head.parsed_upto
                || previous_repair.is_some_and(|previous| previous >= position)
            {
                return Err(Error::Invariant(
                    "invalid Session interpreted-text repair receipt".into(),
                ));
            }
            previous_repair = Some(position);
            add(&mut repairs, repair.count)?;
        }
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
    add(&mut result.record_errors, count)?;
    add(&mut result.text_repairs, repairs)
}
