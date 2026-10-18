//! Direct Session input influence, never inferred from matching Message rows.
//! Catalog traversal and every receipt read share the caller's pinned Session.
mod claude_active;
mod codex;
mod contributors;
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
pub struct SessionStatus {
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
    pub contributors: Vec<SourceRowBasis>,
    contributor_digest: contributors::ContributorDigest,
    membership_sources: u64,
    input_bytes: usize,
    population_pending_inputs: u64,
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
    observed_member_verified: bool,
    prefix_effects_supported: bool,
}
impl SessionStatus {
    pub fn header(
        &self,
        header: &polylogue_store::contributions::SessionHeader,
    ) -> polylogue_store::contributions::SessionHeader {
        let mut h = header.clone();
        if let Some(s) = &self.metadata {
            h.title = s.title.clone();
            h.cwd = s.cwd.clone();
            h.repo = s.repo.clone();
            h.checkout = s.checkout.clone();
            h.branch = s.branch.clone();
            h.commit = s.commit.clone();
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
    /// One observed source can independently certify an exact complete member
    /// while another row or suffix remains pending. Session context is unchanged.
    pub fn member_rows_readable(&self) -> bool {
        self.exact
            && self.unresolved_inputs == 0
            && ((self.known_rows_readable && self.pending_inputs == 0)
                || (self.observed_member_verified
                    && self.membership_sources == 1
                    && self.prefix_effects_supported))
    }
    pub fn active_path_known(&self) -> bool {
        !self.own_topology_unknown
    }
    pub fn complete(&self) -> bool {
        self.record_errors == 0
            && self.pending_inputs == 0
            && self.unresolved_inputs == 0
            && self.unavailable_inputs == 0
            && self.population_pending_inputs == 0
            && self.unsupported.is_empty()
    }
    pub fn notes(&self) -> Vec<String> {
        self.reason_notes(true)
    }
    /// Causes of incomplete coverage, excluding known interpreted-text repairs.
    pub fn incomplete_reasons(&self) -> Vec<String> {
        self.reason_notes(false)
    }
    fn reason_notes(&self, include_diagnostics: bool) -> Vec<String> {
        let mut notes = self.unsupported.iter().cloned().collect::<Vec<_>>();
        for (count, detail) in [
            (
                self.population_pending_inputs,
                "foreign retained inputs have pending possible Session relationships; requested membership is qualified separately",
            ),
            (
                self.record_errors,
                "undecodable complete records in relevant retained inputs",
            ),
            (
                if include_diagnostics {
                    self.text_repairs
                } else {
                    0
                },
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
            ("population_pending_inputs", self.population_pending_inputs),
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
            "settlement":{"scope":if self.exact{"exact_session"}else{"session_selection"},"operation":if self.document{"normalized_read"}else{"selection"},"state":if self.complete(){"settled"}else if self.pending_inputs>0||self.population_pending_inputs>0{"pending"}else if self.unavailable_inputs>0||self.record_errors>0{"unavailable"}else{"unsupported"},
                "settled_through":{"archive_counter":catalog.source_archive_counter(),"recipe":"native-session-v3","configuration":if self.contributor_digest.observed{"retained-native-context-v1"}else{"fixed-immutable-native-context-v1"},"inputs":self.contributors,"inputs_total":self.contributor_digest.count,"inputs_enumerated":self.contributors.len() as u64==self.contributor_digest.count,"inputs_digest":self.contributor_digest.digest()},
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
/// Qualifies finite provider-owned Claude and Codex source membership. Other
/// dependency shapes remain unresolved, never excluded by selected row overlap.
/// Each input is visited once in immutable catalog order. Same input replays,
/// source/candidate fanout and historical citation paths cannot multiply errors.
/// Whole admitted effect cells remain the existing allocation/work unit.
pub fn check(
    session: &SessionView,
    until: Instant,
    allowance: usize,
) -> Result<SessionStatus, Error> {
    check_with_cancel(session, until, allowance, &mut || false)
}
pub fn check_with_cancel(
    session: &SessionView,
    until: Instant,
    allowance: usize,
    cancelled: &mut dyn FnMut() -> bool,
) -> Result<SessionStatus, Error> {
    check_session(session, until, allowance, cancelled, None)
}
/// Exact member reads preserve unavailable Session context while independently
/// checking the selected Message's retained body witnesses at handoff.
pub fn check_member_with_cancel(
    session: &SessionView,
    message: &polylogue_store::contributions::MessageView,
    until: Instant,
    allowance: usize,
    cancelled: &mut dyn FnMut() -> bool,
) -> Result<SessionStatus, Error> {
    check_session(session, until, allowance, cancelled, Some(message))
}
fn check_session(
    session: &SessionView,
    until: Instant,
    allowance: usize,
    cancelled: &mut dyn FnMut() -> bool,
    member: Option<&polylogue_store::contributions::MessageView>,
) -> Result<SessionStatus, Error> {
    let header = session.header()?;
    let plan = polylogue_model::query::SessionQuery {
        origin: Some(header.origin.as_str().into()),
        id: Some(header.id.as_str().into()),
        ..Default::default()
    };
    check_operation(
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
pub fn check_catalog(
    session: &SessionCatalog,
    plan: &polylogue_model::query::SessionQuery,
    until: Instant,
    allowance: usize,
    cancelled: &mut dyn FnMut() -> bool,
) -> Result<SessionStatus, Error> {
    check_operation(session, plan, until, allowance, cancelled, false, None)
}
fn check_operation(
    session: &SessionCatalog,
    plan: &polylogue_model::query::SessionQuery,
    until: Instant,
    allowance: usize,
    cancelled: &mut dyn FnMut() -> bool,
    document: bool,
    member: Option<&polylogue_store::contributions::MessageView>,
) -> Result<SessionStatus, Error> {
    let mut result = SessionStatus {
        up_to_date: true,
        document,
        known_rows_readable: true,
        prefix_effects_supported: true,
        ..Default::default()
    };
    let exact = plan.id.as_deref().filter(|_| {
        plan.text.is_empty()
            && plan
                .origin
                .as_deref()
                .is_none_or(|o| matches!(o, "claude-code" | "codex"))
    });
    result.exact = exact.is_some();
    let target_native = exact
        .map(|target| {
            let id = polylogue_model::SessionId::from_stored(target)
                .map_err(|e| Error::Invariant(e.to_string()))?;
            session
                .session(&id, allowance)?
                .map(|view| view.header().map(|h| h.native_id.clone()))
                .transpose()
        })
        .transpose()?
        .flatten();
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
        // next_dependency admits one descriptor. Streamed work is not live
        // retained metadata; actual witnesses and Message membership are bounded.
        // Global freshness is independent of this operation's dependency scope.
        // A retained unavailable interpretation is caught up, but not complete.
        let caught_up =
            if let (Some(input), Some(head)) = (dependency.registered(), dependency.head()) {
                if head.acquired_upto == input.input.length
                    && head.token.basis == owner::parser_identity(&input)
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
        // Provider-owned source membership is checked before selected row overlap.
        if dependency.format() == InputFormat::CodexRollout {
            if plan.origin.as_deref().is_none_or(|o| o == "codex") {
                codex::check(
                    session,
                    &dependency,
                    plan,
                    exact,
                    until,
                    allowance,
                    &mut result,
                    cancelled,
                )?;
            }
            continue;
        }
        if matches!(
            dependency.format(),
            InputFormat::CodexHistorySqlite | InputFormat::CodexPromptHistory
        ) {
            if plan.origin.as_deref().is_none_or(|o| o == "codex") {
                add(&mut result.unresolved_inputs, 1)?;
            }
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
        match claude_membership(session, &dependency, &input, &context, allowance)? {
            Some(id)
                if plan
                    .id
                    .as_deref()
                    .is_some_and(|target| id.as_str() != target)
                    && !target_native
                        .as_deref()
                        .map(|native| {
                            cell::<SessionCell>(
                                session,
                                &dependency,
                                claude::SESSION,
                                &native,
                                allowance,
                            )
                            .map(|source| {
                                source.is_some_and(|s| Some(s.session.id.as_str()) == exact)
                            })
                        })
                        .transpose()?
                        .unwrap_or(false) =>
            {
                if document {
                    if !caught_up {
                        result.add_contributor(
                            source_rows::computed_from(&input),
                            false,
                            allowance,
                        )?;
                        add(&mut result.population_pending_inputs, 1)?;
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
                                    let key = (
                                        pending.message.session().as_str(),
                                        pending.message.as_str(),
                                        item,
                                    );
                                    let contribution: polylogue_sources::claude_code::keyed::Contribution =
                                        cell(session, &dependency, claude::CONTRIBUTION, &key, allowance)?
                                            .ok_or_else(|| Error::Invariant("missing foreign Session contribution".into()))?;
                                    // A native suffix can be an ordinary word or
                                    // letter. Only the full public ID supplies
                                    // possible reference evidence in this scan.
                                    if contribution.blocks.iter().any(|block| {
                                        matches!(block, polylogue_model::BlockContent::Text { text }
                                            if exact.is_some_and(|id| text.contains(id)))
                                    }) {
                                        possible = true;
                                    }
                                }
                            }
                        }
                        if possible {
                            result.add_contributor(
                                source_rows::computed_from(&input),
                                false,
                                allowance,
                            )?;
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
            if member.is_none() || !unavailable_body {
                return Err(error);
            }
            add(&mut result.unavailable_inputs, 1)?;
            result.unsupported.insert(format!(
                "Required body unavailable in Session context: {error}"
            ));
        }
        result.add_contributor(source_rows::computed_from(&input), true, allowance)?;
        result.own_topology_unknown |= !caught_up;
        let Some(head) = dependency.head() else {
            add(&mut result.pending_inputs, 1)?;
            continue;
        };
        if head.acquired_upto != input.input.length
            || head.token.basis != owner::parser_identity(&input)
            || head.parser_version != owner::parser_version(input.format)
        {
            add(&mut result.pending_inputs, 1)?;
            continue;
        }
        result.own_topology_unknown |= head.counts.unrecognized > 0 || head.counts.undecodable > 0;
        if head.counts.used > 0 {
            add(&mut result.interpreted_nonempty_inputs, 1)?;
        }
        if let Some(message) = member {
            result.observed_member_verified |=
                observed_member_verified(session, &dependency, &input, message, allowance)?;
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
                // Each known Message below must independently carry the exact
                // current selected source projection. Unsupported neighboring
                // products keep coverage degraded; they cannot veto that proof.
                let readable = head.counts.unrecognized == 0;
                match receipt.rows.state {
                    polylogue_model::settlement::SourceRowState::Unavailable {
                        usable_result,
                        ..
                    } => {
                        unavailable = true;
                        result.known_rows_readable &= readable && usable_result;
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
            None => {
                add(&mut result.pending_inputs, 1)?;
            }
        }
        if unavailable {
            add(&mut result.unavailable_inputs, 1)?;
        }
        if let Some(target) = exact {
            check_shape(
                session,
                &dependency,
                target,
                target_native.as_deref().or_else(|| match &context {
                    // A declared main file can own metadata before any Message
                    // publishes its Session header. The dispatch above already
                    // proves this is the requested main or an owned side Session.
                    polylogue_sources::claude_code::FileCtx::Main { stem } => Some(stem.as_str()),
                    _ => None,
                }),
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
                    .then(|| started.map(time).or(metadata.started_at))
                    .flatten();
                metadata.last_activity_at = (!result.own_counting_unknown)
                    .then(|| last.map(time).or(metadata.last_activity_at))
                    .flatten();
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
fn observed_member_verified(
    catalog: &SessionCatalog,
    dependency: &SessionInputDependency,
    input: &polylogue_store::StoredSource,
    message: &polylogue_store::contributions::MessageView,
    allowance: usize,
) -> Result<bool, Error> {
    let Some(head) = dependency.head() else {
        return Ok(false);
    };
    if !input.observed()
        || input.format != InputFormat::ClaudeCodeJsonl
        || !(head.parsed_upto == input.input.length
            || (head.status == "waiting_tail"
                && head.finalization_pending
                && head.parsed_upto < input.input.length))
        || head.counts.unrecognized > 0
        || head.counts.undecodable > 0
    {
        return Ok(false);
    }
    let Some(pending) = cell::<polylogue_sources::claude_code::keyed::PendingMessage>(
        catalog,
        dependency,
        claude::PENDING,
        &(message.id().session().as_str(), message.id().as_str()),
        allowance,
    )?
    else {
        return Ok(false);
    };
    let logical = owner::claude_key(pending.message.as_str());
    let Some(source) = catalog.dependency_selected_projection(dependency, &logical, allowance)?
    else {
        return Ok(false);
    };
    if source.version_hash() != message.version_hash()? {
        return Ok(false);
    }
    crate::canonical_message::current_claude_projection(
        &source,
        &head.token,
        &pending,
        input.binding,
        &input.context,
    )
    .map_err(|error| Error::Invariant(error.to_string()))
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
/// Main context names its own Session. Separate-agent context only names a
/// role/lineage: its current provider header and Session cell must prove owner.
fn claude_membership(
    catalog: &SessionCatalog,
    dependency: &SessionInputDependency,
    input: &polylogue_store::StoredSource,
    context: &polylogue_sources::claude_code::FileCtx,
    allowance: usize,
) -> Result<Option<polylogue_model::SessionId>, Error> {
    use polylogue_sources::claude_code::{FileCtx, keyed::Header};
    let FileCtx::Subagent { agent, .. } = context else {
        return Ok(context.direct_main_session());
    };
    let Some(head) = dependency.head() else {
        return Ok(None);
    };
    if head.parser_version != owner::parser_version(input.format)
        || head.token.basis != owner::parser_identity(input)
    {
        return Ok(None);
    }
    let Some(header) = cell::<Header>(catalog, dependency, claude::HEADER, &(), allowance)? else {
        return Ok(None);
    };
    let Some(owner) = header.owner else {
        return Ok(None);
    };
    let native = format!("{owner}/{agent}");
    let Some(source) =
        cell::<SessionCell>(catalog, dependency, claude::SESSION, &native, allowance)?
    else {
        return Err(Error::Invariant(
            "Claude agent owner lost its Session cell".into(),
        ));
    };
    let id = polylogue_model::SessionId::new(
        &polylogue_model::Origin::new("claude-code").expect("constant origin"),
        &native,
    )
    .map_err(|e| Error::Invariant(e.to_string()))?;
    if source.session.id != id || source.session.native_id != native {
        return Err(Error::Invariant(
            "Claude agent Session membership disagrees with provider state".into(),
        ));
    }
    Ok(Some(id))
}
// A different pending row may lack a current selected projection. That makes
// whole-catalog coverage unavailable, while an exact independently certified
// member can still be read. Its own member proof above keeps errors intact.
fn selected_current_projection(
    catalog: &SessionCatalog,
    dependency: &SessionInputDependency,
    logical: &[u8; 32],
    allowance: usize,
) -> Result<Option<polylogue_store::contributions::SourceProjection>, Error> {
    match catalog.dependency_selected_projection(dependency, logical, allowance) {
        Err(Error::StaleMessage) => Ok(None),
        result => result,
    }
}
/// Prove the supported shape from retained interpretation, never from the rows
/// a search happened to find. Events, lineage/copy and ambiguous metadata keep
/// their existing degraded qualification until their connected products ship.
fn check_shape(
    catalog: &SessionCatalog,
    dependency: &SessionInputDependency,
    target: &str,
    native: Option<&str>,
    until: Instant,
    allowance: usize,
    result: &mut SessionStatus,
    cancelled: &mut dyn FnMut() -> bool,
) -> Result<(), Error> {
    let Some(native) = native else { return Ok(()) };
    let mut active_leaf = None;
    if let Some(source) =
        cell::<SessionCell>(catalog, dependency, claude::SESSION, &native, allowance)?
    {
        active_leaf = source.active_leaf.clone();
        if source.session.id.as_str() != target {
            return Err(Error::Invariant(
                "Session source identity disagrees with context".into(),
            ));
        }
        let mut input = json!({"id":dependency.id(),"format":dependency.format().name(),"parser_version":dependency.head().ok_or(Error::Range)?.parser_version});
        if let Some(path) = catalog.dependency_first_path(dependency, allowance)? {
            input["path"] = json!(path);
        }
        result.include_input(input, allowance)?;
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
            result.prefix_effects_supported = false;
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
    let mut peers = Vec::new();
    for (message, p) in &pending {
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
        if response.message.id.as_str() != message.as_str() {
            return Err(Error::Invariant(
                "retained Session response identity mismatch".into(),
            ));
        }
        if response.message.written_by == polylogue_model::WrittenBy::Unknown {
            result
                .unsupported
                .insert("An unclassified Message author remains unresolved".into());
        }
        let mut peer_calls = BTreeSet::new();
        let mut pure_results = true;
        let mut peer_valid = true;
        let mut result_blocks = 0usize;
        let mut peer_parent = None;
        let mut peer_caller = None;
        let mut peer_record_start = None;
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
            if item == 0 {
                let raw = polylogue_model::RawJson::parse(&contribution.raw)
                    .map_err(|e| Error::Invariant(e.to_string()))?;
                peer_caller = raw.get("sourceToolAssistantUUID").cloned();
                peer_record_start = contribution
                    .origins
                    .fields
                    .get(&vec!["message".into(), "content".into()])
                    .map(|d| d.record.start);
                peer_parent = raw
                    .get("parentUuid")
                    .and_then(polylogue_model::RawJson::native)
                    .cloned();
            }
            for block in &contribution.blocks {
                match block {
                    polylogue_model::BlockContent::ToolResult(output) => {
                        result_blocks += 1;
                        if let Some(id) = &output.call_id {
                            if id.as_str().is_empty() || !peer_calls.insert(id.clone()) {
                                peer_valid = false;
                            }
                        } else {
                            peer_valid = false;
                        }
                    }
                    _ => pure_results = false,
                }
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
        if response.message.role == polylogue_model::Role::User && pure_results && result_blocks > 0
        {
            if let Some(record_start) = peer_record_start {
                let calls = peer_calls;
                {
                    pending_bytes = pending_bytes.saturating_add(
                        serde_json::to_vec(&(&message, &peer_parent, &peer_caller, &calls))
                            .map_err(|e| Error::Invariant(e.to_string()))?
                            .len(),
                    );
                    if pending_bytes > allowance {
                        return Err(Error::ParserAdmission {
                            resource: "Claude result peer membership",
                            required: pending_bytes as u64,
                        });
                    }
                    peers.push(claude_active::ResultPeer {
                        message: message.clone(),
                        parent: peer_parent,
                        caller: peer_caller,
                        record_start,
                        valid: peer_valid,
                        calls,
                    });
                }
            }
        }
        if response.message.copy_origin.is_some() || response.message.inherited_from.is_some() {
            result.own_counting_unknown = true;
            result.unsupported.insert(
                "Inherited/copy counting and original operation scope remain unsupported".into(),
            );
        }
        let input = dependency.registered().ok_or(Error::Range)?;
        let logical = owner::claude_key(&message);
        let selected = selected_current_projection(catalog, dependency, &logical, allowance)?;
        let proved = match selected {
            Some(source) => crate::canonical_message::current_claude_projection(
                &source,
                &head.token,
                &p,
                input.binding,
                dependency.context(),
            )
            .map_err(|e| Error::Invariant(e.to_string()))?,
            None => false,
        };
        if !proved {
            result.known_rows_readable = false;
            result
                .unsupported
                .insert("Known Claude Message lacks its current selected source projection".into());
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
        if !result.expected.contains_key(message) {
            admit_membership(result, &message, false, allowance)?;
        }
        if result
            .expected
            .insert(message.clone(), p.ordinal)
            .is_some_and(|old| old != p.ordinal)
        {
            result.own_topology_unknown = true;
            result.known_rows_readable = false;
            result
                .unsupported
                .insert("Independent inputs disagree about Session Message order".into());
        }
    }
    let active = if let Some(leaf) = active_leaf {
        // An invalid later marker must not fall back to an older or linear path.
        claude_active::prove(
            catalog, dependency, target, native, &leaf, &pending, &peers, until, allowance,
            cancelled,
        )?
    } else {
        let mut previous = None;
        let mut active = BTreeSet::new();
        let mut linear = true;
        for (_, (message, parent)) in tree {
            if parent != previous {
                linear = false;
            }
            active.insert(message.clone());
            previous = Some(message);
        }
        linear.then_some(active)
    };
    if let Some(active) = active {
        if !result.active_messages.is_empty() && result.active_messages != active {
            result.own_topology_unknown = true;
            result
                .unsupported
                .insert("Independent inputs disagree about the active Message path".into());
        } else {
            for message in active {
                if !result.active_messages.contains(&message) {
                    admit_membership(result, &message, true, allowance)?;
                    result.active_messages.insert(message);
                }
            }
        }
    } else {
        result.own_topology_unknown = true;
        result.unsupported.insert(
            "Branched or incomplete parent topology requires a proved native active leaf".into(),
        );
    }
    Ok(())
}
fn admit_membership(
    result: &mut SessionStatus,
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
    result: &mut SessionStatus,
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
