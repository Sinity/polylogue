//! Replay evidence shares the fact transaction; delivery and application are distinct.
use super::*;
use crate::primary::{
    NativeFactContext, ProjectionAvailability, ProjectionCoverage, RawInputCoverage,
    ReplayBasisStatus, ReplayContinuation, ReplayControl, ReplayCursor, ReplayState,
};

#[derive(PartialEq, Eq, serde::Serialize, serde::Deserialize)]
enum CompletionBasis {
    Snapshot(ProjectionCoverage),
    Native(crate::primary::DeliveredCompletion),
}
#[derive(serde::Serialize, serde::Deserialize)]
struct CompletionEvidence {
    basis: CompletionBasis,
    publication: Option<crate::primary::RawInputProof>,
}

pub(crate) fn replay_state(conn: &Connection) -> Result<ReplayState> {
    let value: Option<String> = conn
        .query_row(
            "SELECT value FROM meta WHERE key='confirmed_replay'",
            [],
            |row| row.get(0),
        )
        .optional()?;
    let mut state = match value {
        Some(value) => serde_json::from_str(&value)?,
        None => ReplayState {
            applied_position: checkpoint(conn)?,
            scope: super::super::scope::current(conn)?,
            ..Default::default()
        },
    };
    // A durable cancellation may precede a failed SDK revocation/clear. Its
    // old serialized certificate cannot authorize publication after restart.
    let blocked = conn.query_row("SELECT EXISTS(SELECT 1 FROM meta WHERE key IN ('confirmed_scope_pending','confirmed_replay_blocked'))", [], |row| row.get::<_,bool>(0))?;
    if blocked {
        state.publication = None;
        if state.availability == ProjectionAvailability::Certified {
            state.availability = ProjectionAvailability::Unverified;
        }
    }
    Ok(state)
}

pub(crate) fn save(conn: &Connection, state: &ReplayState) -> Result<()> {
    conn.execute("INSERT INTO meta(key,value) VALUES ('confirmed_replay',?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value", [serde_json::to_string(state)?])?;
    Ok(())
}

pub(crate) fn native_context(conn: &Connection, event: &str) -> Result<Option<NativeFactContext>> {
    let bytes: Option<Vec<u8>> = conn
        .query_row(
            "SELECT native_context FROM primary_fact WHERE event=?1",
            [event],
            |row| row.get(0),
        )
        .optional()?
        .flatten();
    bytes.as_deref().map(unpacked).transpose()
}

fn invalid(message: &str) -> StoreError {
    StoreError::Invalid(message.into())
}
fn nonempty(value: &str) -> bool {
    !value.is_empty()
}
fn hash(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}
pub(super) fn validate_context(context: &NativeFactContext, epoch: Option<i64>) -> Result<()> {
    if !nonempty(&context.kind.name)
        || context.kind.version == 0
        || !hash(&context.kind_definition_hash)
        || !hash(&context.projection.payload_hash)
        || context.projection.policy_epoch < 0
        || epoch.is_some_and(|epoch| epoch != context.projection.policy_epoch)
        || context
            .origin
            .source
            .as_ref()
            .is_some_and(|source| source.name.is_empty() || source.version == 0)
        || [
            &context.origin.source_fingerprint,
            &context.origin.producer_family,
            &context.origin.producer_identity,
        ]
        .into_iter()
        .flatten()
        .any(|value| value.is_empty())
    {
        return Err(invalid(
            "confirmed native context has inconsistent owner metadata",
        ));
    }
    Ok(())
}
fn validate_raw(coverage: &RawInputCoverage, position: i64, epoch: i64, scope: &str) -> Result<()> {
    if let RawInputCoverage::Complete { proof } = coverage
        && (proof.stamp.is_empty()
            || proof.position != position
            || proof.policy_epoch != epoch
            || proof.scope != scope)
    {
        return Err(invalid(
            "raw-input proof differs from its owner replay basis",
        ));
    }
    Ok(())
}

fn validate_pending(conn: &Connection, position: i64, epoch: i64) -> Result<()> {
    let mut query = conn.prepare(
        "SELECT native_context FROM primary_pending WHERE position<=?1 AND change='live'",
    )?;
    let mut rows = query.query([position])?;
    while let Some(row) = rows.next()? {
        let bytes: Option<Vec<u8>> = row.get(0)?;
        let bytes =
            bytes.ok_or_else(|| invalid("completion is missing original owner fact context"))?;
        validate_context(&unpacked(&bytes)?, Some(epoch))?;
    }
    Ok(())
}

fn basis(state: &mut ReplayState, status: &ReplayBasisStatus) {
    state.cursor = None;
    state.publication = None;
    state.availability = match status {
        ReplayBasisStatus::Unverified => ProjectionAvailability::Unverified,
        ReplayBasisStatus::Invalidated {
            policy_epoch,
            scope,
        } => ProjectionAvailability::Invalidated {
            policy_epoch: *policy_epoch,
            scope: scope.clone(),
        },
    };
}
fn unavailable(
    conn: &Connection,
    state: &mut ReplayState,
    status: &ReplayBasisStatus,
) -> Result<()> {
    basis(state, status);
    // Retain staged facts on transient refusal. A completion received before
    // that refusal cannot subsequently restore currentness on its own.
    conn.execute("DELETE FROM primary_complete", [])?;
    if matches!(status, ReplayBasisStatus::Invalidated { .. }) {
        super::super::derived::revoke(conn)?;
        conn.execute("DELETE FROM primary_pending", [])?;
    }
    Ok(())
}

fn clear_offered_publication(conn: &Connection) -> Result<()> {
    let mut query =
        conn.prepare("SELECT position,evidence FROM primary_complete WHERE evidence IS NOT NULL")?;
    let mut rows = query.query([])?;
    while let Some(row) = rows.next()? {
        let position: i64 = row.get(0)?;
        let mut evidence: CompletionEvidence = unpacked(&row.get::<_, Vec<u8>>(1)?)?;
        if evidence.publication.take().is_some() {
            conn.execute(
                "UPDATE primary_complete SET evidence=?2 WHERE position=?1",
                params![position, packed(&evidence)?],
            )?;
        }
    }
    Ok(())
}

fn barrier(conn: &Connection, position: i64, basis: CompletionBasis) -> Result<()> {
    if position < 0 {
        return Err(invalid("a completed owner position cannot be negative"));
    }
    let old: Option<Option<Vec<u8>>> = conn
        .query_row(
            "SELECT evidence FROM primary_complete WHERE position=?1",
            [position],
            |row| row.get(0),
        )
        .optional()?;
    let publication = if let Some(Some(bytes)) = old {
        let old: CompletionEvidence = unpacked(&bytes)?;
        if old.basis != basis {
            return Err(invalid("a completed owner position changed its evidence"));
        }
        old.publication
    } else {
        None
    };
    let bytes = packed(&CompletionEvidence { basis, publication })?;
    conn.execute("INSERT INTO primary_complete(position,evidence) VALUES (?1,?2) ON CONFLICT(position) DO UPDATE SET evidence=excluded.evidence", params![position,bytes])?;
    Ok(())
}

pub(super) fn stage_control(
    conn: &Connection,
    state: &mut ReplayState,
    control: &ReplayControl,
) -> Result<()> {
    match control {
        ReplayControl::Reset { .. } => {
            return Err(invalid(
                "owner replay reset must be a separate control page",
            ));
        }
        ReplayControl::BasisStatus { status } => unavailable(conn, state, status)?,
        ReplayControl::ErasedThrough { position } => {
            if *position <= 0 {
                return Err(invalid(
                    "an owner erasure needs its actual positive position",
                ));
            }
            state.erasure_fence = Some(
                state
                    .erasure_fence
                    .map_or(*position, |old| old.max(*position)),
            );
            state.publication = None;
            state.availability = ProjectionAvailability::Unverified;
        }
        ReplayControl::Delivered { delivery } => {
            if delivery.stream_instance.is_empty() || delivery.stream_sequence == 0 {
                return Err(invalid("a staged delivery needs its native stream receipt"));
            }
            if state.staged_delivery.as_ref().is_some_and(|old| {
                old.stream_instance == delivery.stream_instance
                    && old.stream_sequence > delivery.stream_sequence
            }) {
                return Err(invalid("a staged native delivery moved backwards"));
            }
            state.staged_delivery = Some(delivery.clone());
        }
        ReplayControl::Coverage { coverage } => match coverage {
            RawInputCoverage::Complete { proof } => {
                let epoch = state
                    .policy_epoch
                    .ok_or_else(|| invalid("publication proof has no owner epoch"))?;
                let scope = state
                    .scope
                    .as_deref()
                    .ok_or_else(|| invalid("publication proof has no owner scope"))?;
                validate_raw(coverage, proof.position, epoch, scope)?;
                if matches!(
                    state.availability,
                    ProjectionAvailability::Invalidated { .. }
                ) {
                    return Err(invalid(
                        "publication proof cannot revive an invalidated full basis",
                    ));
                }
                let staged: Option<Option<Vec<u8>>> = conn
                    .query_row(
                        "SELECT evidence FROM primary_complete WHERE position=?1",
                        [proof.position],
                        |row| row.get(0),
                    )
                    .optional()?;
                if let Some(Some(bytes)) = staged {
                    let mut evidence: CompletionEvidence = unpacked(&bytes)?;
                    if evidence
                        .publication
                        .as_ref()
                        .is_some_and(|old| !old.same_basis(proof))
                    {
                        return Err(invalid("a staged completion changed its publication proof"));
                    }
                    evidence.publication.get_or_insert_with(|| proof.clone());
                    conn.execute(
                        "UPDATE primary_complete SET evidence=?2 WHERE position=?1",
                        params![proof.position, packed(&evidence)?],
                    )?;
                } else if state.certified_position() == Some(proof.position) {
                    if state
                        .publication
                        .as_ref()
                        .is_some_and(|old| !old.same_basis(proof))
                    {
                        return Err(invalid(
                            "publication proof differs from its retained applied basis",
                        ));
                    }
                    state.publication.get_or_insert_with(|| proof.clone());
                } else {
                    return Err(invalid(
                        "publication proof has no matching staged or applied full completion",
                    ));
                }
            }
            RawInputCoverage::Incomplete { basis: status, .. } => unavailable(conn, state, status)?,
            RawInputCoverage::NotRequested => {
                state.publication = None;
                clear_offered_publication(conn)?;
            }
        },
        ReplayControl::Page { page } => {
            if matches!(
                state.availability,
                ProjectionAvailability::Invalidated { .. }
            ) {
                return Err(invalid(
                    "an invalidated full basis requires its authoritative reset before replay",
                ));
            }
            if page.position < 0
                || page.policy_epoch < 0
                || page.scope.is_empty()
                || state.scope.as_deref() != Some(page.scope.as_str())
                || state
                    .policy_epoch
                    .is_some_and(|epoch| epoch != page.policy_epoch)
            {
                return Err(invalid(
                    "replay page requires its matching owner reset identity",
                ));
            }
            state.policy_epoch = Some(page.policy_epoch);
            state.publication = None;
            match &page.continuation {
                ReplayContinuation::Next { cursor } => {
                    if cursor.is_empty() {
                        return Err(invalid("a replay continuation cursor cannot be empty"));
                    }
                    if state.cursor.as_ref().is_some_and(|old| {
                        old.position != page.position
                            || old.policy_epoch != page.policy_epoch
                            || old.scope != page.scope
                    }) {
                        return Err(invalid(
                            "replay continuation changed its immutable owner cut",
                        ));
                    }
                    state.cursor = Some(ReplayCursor {
                        position: page.position,
                        policy_epoch: page.policy_epoch,
                        scope: page.scope.clone(),
                        cursor: cursor.clone(),
                    });
                }
                ReplayContinuation::Complete {
                    projection,
                    raw_inputs,
                } => {
                    if projection.position != page.position
                        || projection.scope != page.scope
                        || projection.stamp.is_empty()
                        || projection.catalog.is_empty()
                    {
                        return Err(invalid(
                            "full projection certificate differs from its replay page",
                        ));
                    }
                    validate_raw(raw_inputs, page.position, page.policy_epoch, &page.scope)?;
                    validate_pending(conn, page.position, page.policy_epoch)?;
                    barrier(
                        conn,
                        page.position,
                        CompletionBasis::Snapshot(ProjectionCoverage {
                            policy_epoch: page.policy_epoch,
                            projection: projection.clone(),
                            raw_inputs: raw_inputs.clone(),
                        }),
                    )?;
                    state.cursor = None;
                }
            }
        }
        ReplayControl::Completion {
            position,
            delivered,
        } => {
            if matches!(
                state.availability,
                ProjectionAvailability::Invalidated { .. }
            ) {
                return Err(invalid(
                    "an invalidated full basis cannot admit an old native completion",
                ));
            }
            if state.applied_position == Some(*position)
                && state.completion.as_ref() == Some(delivered)
                && state.certified_position() == Some(*position)
            {
                return Ok(());
            }
            if state
                .applied_position
                .is_some_and(|applied| *position < applied)
            {
                return Err(invalid("native completion moved behind the applied prefix"));
            }
            if delivered.stream_instance.is_empty() || delivered.stream_sequence == 0 {
                return Err(invalid(
                    "native completion needs the actual delivered marker",
                ));
            }
            let coverage = state
                .coverage
                .as_ref()
                .ok_or_else(|| invalid("native completion has no full projection basis"))?;
            if *position < coverage.projection.position
                || state.policy_epoch != Some(coverage.policy_epoch)
                || state.scope.as_deref() != Some(coverage.projection.scope.as_str())
            {
                return Err(invalid(
                    "native completion differs from its certified full projection basis",
                ));
            }
            validate_pending(conn, *position, coverage.policy_epoch)?;
            barrier(conn, *position, CompletionBasis::Native(delivered.clone()))?;
        }
    }
    Ok(())
}

pub(super) fn complete(conn: &Connection, position: i64) -> Result<()> {
    let evidence: Option<Vec<u8>> = conn.query_row(
        "SELECT evidence FROM primary_complete WHERE position=?1",
        [position],
        |row| row.get(0),
    )?;
    let mut state = replay_state(conn)?;
    state.applied_position = Some(position);
    state.publication = None;
    if let Some(evidence) = evidence {
        let evidence: CompletionEvidence = unpacked(&evidence)?;
        match evidence.basis {
            CompletionBasis::Snapshot(coverage) => {
                state.policy_epoch = Some(coverage.policy_epoch);
                state.scope = Some(coverage.projection.scope.clone());
                state.coverage = Some(coverage);
                state.completion = None;
            }
            CompletionBasis::Native(delivered) => state.completion = Some(delivered),
        }
        if state.erasure_fence.is_some_and(|fence| position >= fence) {
            state.erasure_fence = None;
        }
        if state.erasure_fence.is_none() {
            state.availability = ProjectionAvailability::Certified;
            state.publication = evidence.publication;
            conn.execute("DELETE FROM meta WHERE key='confirmed_replay_blocked'", [])?;
        }
    }
    save(conn, &state)
}
