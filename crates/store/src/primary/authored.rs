//! Prepare authored changes once in a private overlay; confirmed records apply
//! their frozen effects without selecting targets, families or claims again.
mod conserve;
pub mod historical;
pub(crate) use conserve::{conserve_local, merge_original, same_historical};
pub(crate) use conserve::{restored_contains, same_prepared};
mod records;
#[cfg(test)]
mod tests;

use crate::source::Source;
use crate::{NewBlob, Result, StoreError};
use polylogue_model::authored::{
    AuthoredEffect, ImportAttribution, ImportHistory, PendingAuthored, PreparedRequest,
};
use polylogue_model::user::{UserOutcome, UserRequest};
use rusqlite::{Connection, OptionalExtension, Transaction, TransactionBehavior, params};
use std::collections::HashMap;

/// Semantic idempotence includes the accepting machine and authored intent;
/// a retry timestamp does not change the admitted request.
pub fn request_hash(request: &UserRequest) -> Result<[u8; 32]> {
    Ok(*blake3::hash(&serde_json::to_vec(&(
        &request.machine,
        &request.authorship,
        request.write.intent(),
    ))?)
    .as_bytes())
}
pub(crate) fn pack(value: &impl serde::Serialize) -> Result<Vec<u8>> {
    Ok(crate::encoding::pack_bytes(&serde_json::to_vec(value)?)?)
}

enum FrozenRequest {
    Prepared(Box<PreparedRequest>),
    Historical,
}

fn frozen(conn: &Connection, id: &str) -> Result<Option<FrozenRequest>> {
    let source = Source::reading(conn)?;
    let stored=conn.query_row(&source.sql("SELECT content_hash,request,outcome,effect,historical IS NOT NULL,original FROM @source.client_request WHERE id=?1"),[id],|row|Ok((row.get::<_,Vec<u8>>(0)?,row.get::<_,Option<Vec<u8>>>(1)?,row.get::<_,Option<Vec<u8>>>(2)?,row.get::<_,Option<Vec<u8>>>(3)?,row.get::<_,bool>(4)?,row.get::<_,Option<Vec<u8>>>(5)?))).optional()?;
    let Some((hash, body, outcome, effect, historical, original)) = stored else {
        return Ok(None);
    };
    if historical {
        return Ok(Some(FrozenRequest::Historical));
    }
    let missing = || StoreError::Invalid(format!("request {id} has no complete frozen outcome"));
    let prepared = PreparedRequest {
        request: crate::encoding::unpack_json(&body.ok_or_else(missing)?)?,
        outcome: crate::encoding::unpack_json(&outcome.ok_or_else(missing)?)?,
        effect: crate::encoding::unpack_json(&effect.ok_or_else(missing)?)?,
        original: original
            .map(|bytes| crate::encoding::unpack_json(&bytes))
            .transpose()?,
    };
    if prepared.request.id != id || hash.as_slice() != request_hash(&prepared.request)? {
        return Err(StoreError::Conflict(id.into()));
    }
    if prepared.outcome.pending {
        return Err(StoreError::Invalid(
            "pending is not a canonical outcome".into(),
        ));
    }
    if let Some(original) = &prepared.original {
        original
            .validate(&prepared.request, &prepared.outcome)
            .map_err(|error| StoreError::Invalid(error.to_string()))?;
        if original.content_hash.0 != request_hash(&prepared.request)? {
            return Err(StoreError::Conflict(id.into()));
        }
    }
    Ok(Some(FrozenRequest::Prepared(Box::new(prepared))))
}

/// Immutable evidence lookup. This does not certify a native acceptance or
/// authorize executing an action retained from a historical projection.
pub fn frozen_request(conn: &Connection, id: &str) -> Result<Option<PreparedRequest>> {
    Ok(match frozen(conn, id)? {
        Some(FrozenRequest::Prepared(prepared)) => Some(*prepared),
        _ => None,
    })
}

/// The existing request row owns the original native outcome and immutable
/// effect. Historical projection equivalence is never a fresh native retry.
pub fn known_request(conn: &Connection, request: &UserRequest) -> Result<Option<PreparedRequest>> {
    match frozen(conn, &request.id)? {
        None => Ok(None),
        Some(FrozenRequest::Historical) => Err(StoreError::Conflict(request.id.clone())),
        Some(FrozenRequest::Prepared(prepared)) => {
            if prepared.original.as_ref().is_some_and(|original| {
                matches!(
                    original.source,
                    polylogue_model::historical::SourceSchema::NativeProjection { .. }
                        | polylogue_model::historical::SourceSchema::LocalProjection { .. }
                        | polylogue_model::historical::SourceSchema::RetainedPublication { .. }
                )
            }) || request_hash(&prepared.request)? != request_hash(request)?
            {
                return Err(StoreError::Conflict(request.id.clone()));
            }
            Ok(Some(*prepared))
        }
    }
}

/// Durable pending requests retain their complete first admitted body. A
/// transport retry never rebuilds that effect against newer definitions.
pub(crate) fn pending_outcome(
    sinex: &dyn crate::primary::Sinex,
    request: &UserRequest,
) -> Result<Option<UserOutcome>> {
    let mut outcome = None;
    sinex.visit_pending_requests(&mut |pending| {
        let Some(prepared) = pending.prepared() else {
            return Ok(());
        };
        if prepared.request.id == request.id {
            if prepared.original.as_ref().is_some_and(|original| {
                matches!(
                    original.source,
                    polylogue_model::historical::SourceSchema::NativeProjection { .. }
                        | polylogue_model::historical::SourceSchema::LocalProjection { .. }
                        | polylogue_model::historical::SourceSchema::RetainedPublication { .. }
                )
            }) {
                return Err(StoreError::Conflict(request.id.clone()));
            }
            if request_hash(&prepared.request)? != request_hash(request)? {
                return Err(StoreError::Conflict(request.id.clone()));
            }
            let mut original = prepared.outcome.clone();
            original.pending = true;
            outcome = Some(original);
        }
        Ok(())
    })?;
    Ok(outcome)
}

/// `visit_pending` includes publisher-acknowledged requests until the confirmed
/// consumer has applied them. Blob metadata is verified by the spool owner.
pub fn prepare(
    index: &Connection,
    request: &UserRequest,
    preparation: Option<&crate::user_writes::PreparationContext>,
    pending: Option<&dyn crate::primary::Sinex>,
    visit_pending: impl FnOnce(&mut dyn FnMut(&PendingAuthored) -> Result<()>) -> Result<()>,
    visit_blobs: impl FnOnce(&mut dyn FnMut(&NewBlob) -> Result<()>) -> Result<()>,
) -> Result<PreparedRequest> {
    let _snapshot = if index.is_autocommit() {
        Some(Transaction::new_unchecked(
            index,
            TransactionBehavior::Deferred,
        )?)
    } else {
        None
    };
    if let Some(known) = known_request(index, request)? {
        return Ok(known);
    }
    let mut pending_blobs = HashMap::new();
    visit_blobs(&mut |blob| {
        if pending_blobs
            .insert(blob.hash, blob.size)
            .is_some_and(|size| size != blob.size)
        {
            return Err(StoreError::Invalid(
                "a pending blob has conflicting lengths".into(),
            ));
        }
        Ok(())
    })?;
    with_overlay(index, visit_pending, |tx| {
        if let Some(known) = known_request(tx, request)? {
            return Ok(known);
        }
        let source = Source::of(index)?;
        let available = |attachment: &polylogue_model::user::ChatAttachment| -> Result<bool> {
            if let Some(size) = pending_blobs.get(&attachment.blob) {
                return Ok(*size == attachment.size);
            }
            Ok(index
                .prepare_cached(
                    &source.sql("SELECT 1 FROM @source.blob WHERE hash=?1 AND size=?2"),
                )?
                .exists(params![
                    &attachment.blob.0[..],
                    i64::try_from(attachment.size).map_err(|_| StoreError::Invalid(
                        "attachment size is out of range".into()
                    ))?
                ])?)
        };
        crate::user_writes::apply_using(
            tx,
            index,
            request,
            1,
            &available,
            crate::user_writes::Preparation::native(preparation, pending),
        )?;
        known_request(tx, request)?
            .ok_or_else(|| StoreError::Invalid("prepared request was not recorded".into()))
    })
}

/// One original snapshot plus this host's immutable pending effects. The
/// callback cannot retain a SQLite borrow across a subsequent custody ACK.
pub(crate) fn with_overlay<T>(
    index: &Connection,
    visit_pending: impl FnOnce(&mut dyn FnMut(&PendingAuthored) -> Result<()>) -> Result<()>,
    visit: impl FnOnce(&Transaction<'_>) -> Result<T>,
) -> Result<T> {
    let _snapshot = if index.is_autocommit() {
        Some(Transaction::new_unchecked(
            index,
            TransactionBehavior::Deferred,
        )?)
    } else {
        None
    };
    // Private scratch can spill beyond the SQLite page cache; it owns no
    // durable fact and needs no journal or sync boundary.
    let scratch_file = tempfile::NamedTempFile::new()?;
    let mut scratch = Connection::open(scratch_file.path())?;
    scratch.execute_batch(
        "PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; PRAGMA cache_size=-4096;",
    )?;
    scratch.pragma_update(None, "foreign_keys", true)?;
    scratch.execute_batch(&crate::source::schema()?.ddl)?;
    let tx = scratch.transaction()?;
    records::copy(index, &tx)?;
    visit_pending(&mut |pending| {
        match pending {
            PendingAuthored::Live { prepared } => apply(&tx, prepared, 0)?,
            PendingAuthored::Restore {
                prepared,
                claims,
                current,
            } => {
                restore(&tx, prepared, claims, current, 0)?;
            }
            PendingAuthored::RestoreHistory {
                evidence,
                claims,
                current,
            } => {
                historical::restore_record(&tx, evidence, claims, current, 0)?;
            }
            PendingAuthored::RestoreAssertion {
                fact,
                claims,
                current,
            } => {
                crate::assertions::restore(&tx, fact, claims, current, 0)?;
            }
        }
        Ok(())
    })?;
    visit(&tx)
}

/// Checks internal frozen history consistency without applying or executing it.
pub fn validate(prepared: &PreparedRequest) -> Result<()> {
    if let Some(original) = &prepared.original {
        original
            .validate(&prepared.request, &prepared.outcome)
            .map_err(|error| StoreError::Invalid(error.to_string()))?;
        if original.content_hash.0 != request_hash(&prepared.request)? {
            return Err(StoreError::Conflict(prepared.request.id.clone()));
        }
    }
    crate::signals::validate(prepared)?;
    for fact in &prepared.effect.assertions {
        fact.validate()
            .map_err(|error| StoreError::Invalid(error.to_string()))?;
    }
    fn events<T: PartialEq>(
        all: &[T],
        added: &[T],
        request: &str,
        identity: impl Fn(&T) -> &str,
    ) -> Result<()> {
        let mut seen = std::collections::HashSet::new();
        for event in all {
            let id = identity(event);
            if id.is_empty() || !seen.insert(id) {
                return Err(StoreError::Invalid(
                    "authored history has a duplicate or empty request identity".into(),
                ));
            }
        }
        for event in added {
            if identity(event) != request || !all.contains(event) {
                return Err(StoreError::Invalid(
                    "authored event additions disagree with their frozen history".into(),
                ));
            }
        }
        Ok(())
    }
    for change in &prepared.effect.deliveries {
        events(
            &change.delivery.events,
            &change.added_events,
            &prepared.request.id,
            |e| &e.request,
        )?;
    }
    for change in &prepared.effect.chats {
        events(
            &change.action.events,
            &change.added_events,
            &prepared.request.id,
            |e| &e.request,
        )?;
    }
    for change in &prepared.effect.backfills {
        events(
            &change.job.events,
            &change.added_events,
            &prepared.request.id,
            |e| &e.request,
        )?;
    }
    Ok(())
}

/// Apply immutable facts only. No request selection or domain precondition is
/// rerun after Sinex confirms the prepared body.
pub fn apply(tx: &Transaction<'_>, prepared: &PreparedRequest, counter: i64) -> Result<()> {
    let prepared = conserve::declared_prepared(prepared);
    if prepared.original.as_ref().is_some_and(|original| {
        matches!(
            original.source,
            polylogue_model::historical::SourceSchema::NativeProjection { .. }
                | polylogue_model::historical::SourceSchema::LocalProjection { .. }
                | polylogue_model::historical::SourceSchema::RetainedPublication { .. }
        )
    }) {
        return Err(StoreError::Conflict(prepared.request.id.clone()));
    }
    if let Some(known) = known_request(tx, &prepared.request)? {
        return if known == prepared {
            Ok(())
        } else {
            Err(StoreError::Conflict(prepared.request.id.clone()))
        };
    }
    if prepared.outcome.pending {
        return Err(StoreError::Invalid(
            "pending is not a canonical outcome".into(),
        ));
    }
    validate(&prepared)?;
    records::apply(tx, &prepared.effect, counter, false)?;
    insert_request(tx, &prepared, counter, &ImportHistory::default())
}

/// Historical imported states retain their author and status, while execution
/// eligibility is computed locally and cannot be restored from a wire claim.
pub fn restore(
    tx: &Transaction<'_>,
    prepared: &PreparedRequest,
    claims: &[ImportAttribution],
    current: &ImportAttribution,
    counter: i64,
) -> Result<UserOutcome> {
    if prepared.original.as_ref().is_some_and(|original| {
        matches!(
            original.source,
            polylogue_model::historical::SourceSchema::RetainedPublication { .. }
        )
    }) {
        return Err(StoreError::Invalid(
            "queued publication is not canonical accepted evidence".into(),
        ));
    }
    let prepared = conserve::declared_prepared(prepared);
    let known = match frozen(tx, &prepared.request.id)? {
        Some(FrozenRequest::Historical) => {
            return Err(StoreError::Conflict(prepared.request.id.clone()));
        }
        Some(FrozenRequest::Prepared(prepared)) => Some(*prepared),
        None => None,
    };
    if let Some(mut known) = known {
        if !conserve::same_prepared(&known, &prepared) {
            return Err(StoreError::Conflict(prepared.request.id.clone()));
        }
        conserve::merge_declared_prepared(tx, &mut known, &prepared)?;
        let source = Source::of(tx)?;
        let prior: String = tx.query_row(
            &source.sql("SELECT imports FROM @source.client_request WHERE id=?1"),
            [&prepared.request.id],
            |row| row.get(0),
        )?;
        let mut imports: ImportHistory = serde_json::from_str(&prior)?;
        imports.merge(claims, current);
        let merged = serde_json::to_string(&imports)?;
        if merged != prior {
            tx.execute(
                &source.sql("UPDATE @source.client_request SET imports=?2 WHERE id=?1"),
                params![prepared.request.id, merged],
            )?;
        }
        let mut outcome = known.outcome;
        outcome.repeat = true;
        return Ok(outcome);
    }
    if prepared.outcome.pending {
        return Err(StoreError::Invalid(
            "pending is not a restorable outcome".into(),
        ));
    }
    validate(&prepared)?;
    records::apply(tx, &prepared.effect, counter, true)?;
    let mut imports = ImportHistory::default();
    imports.merge(claims, current);
    insert_request(tx, &prepared, counter, &imports)?;
    Ok(prepared.outcome.clone())
}

fn insert_request(
    conn: &Connection,
    prepared: &PreparedRequest,
    counter: i64,
    imports: &ImportHistory,
) -> Result<()> {
    let source = Source::of(conn)?;
    conn.execute(&source.sql("INSERT INTO @source.client_request(id,content_hash,created_at,request,outcome,effect,imports,commit_seq,original) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9)"),params![prepared.request.id,&request_hash(&prepared.request)?[..],prepared.request.at.unix_ms(),pack(&prepared.request)?,pack(&prepared.outcome)?,pack(&prepared.effect)?,serde_json::to_string(imports)?,counter,prepared.original.as_ref().map(pack).transpose()?])?;
    Ok(())
}

pub(crate) fn store_delivery(
    conn: &Connection,
    value: &polylogue_model::authored::DeliveryRecord,
    counter: i64,
    historical: bool,
) -> Result<()> {
    records::insert_delivery(conn, value, counter, historical)
}

pub(crate) fn capture(
    conn: &Connection,
    request: &UserRequest,
    counter: i64,
) -> Result<AuthoredEffect> {
    records::capture(conn, request, counter)
}

/// Authored lifecycle rebuilds use only the confirmed live authored facts.
pub fn clear(conn: &Connection) -> Result<()> {
    let source = Source::of(conn)?;
    for table in [
        "assertion_link",
        "assertion",
        "variant_part",
        "variant",
        "signal_event",
        "signal_state",
        "delivery_event",
        "chat_action_attachment",
        "chat_action_event",
        "backfill_job_event",
        "kept_result",
        "delivery",
        "chat_action",
        "backfill_job",
        "client_request",
    ] {
        conn.execute(&source.sql(&format!("DELETE FROM @source.{table}")), [])?;
    }
    Ok(())
}

/// Stream confirmed authored records with their retained import provenance.
/// The caller supplies current import attribution only at `restore` ingress.
pub fn visit(
    conn: &Connection,
    accept: &mut dyn FnMut(&PreparedRequest, &[ImportAttribution]) -> Result<()>,
) -> Result<()> {
    let source = Source::reading(conn)?;
    let mut statement = conn.prepare(&source.sql(
        "SELECT request,imports FROM @source.client_request WHERE historical IS NULL ORDER BY commit_seq,created_at,id",
    ))?;
    let mut rows = statement.query([])?;
    while let Some(row) = rows.next()? {
        let request: UserRequest = crate::encoding::unpack_json(&row.get::<_, Vec<u8>>(0)?)?;
        let prepared = frozen_request(conn, &request.id)?
            .ok_or_else(|| StoreError::NotFound(request.id.clone()))?;
        let history: ImportHistory = serde_json::from_str(&row.get::<_, String>(1)?)?;
        // Exported provenance is historical at the next importing host.
        let imports: Vec<_> = history
            .observed
            .into_iter()
            .chain(history.declared)
            .collect();
        accept(&prepared, &imports)?;
    }
    Ok(())
}
impl crate::Reader<'_> {
    pub fn visit_authored(
        &self,
        accept: &mut dyn FnMut(&PreparedRequest, &[ImportAttribution]) -> Result<()>,
    ) -> Result<()> {
        visit(self.conn()?, accept)
    }
}

/// Uses the logical source views of a Reader in either storage mode.
pub(crate) fn signal_outbox(
    conn: &Connection,
    id: &str,
) -> Result<polylogue_model::signal::StoredOutbox> {
    let record = records::delivery_from(conn, Source::Cache, id)?;
    let metadata = record
        .signal
        .ok_or_else(|| StoreError::Invalid("delivery has no signal metadata".into()))?;
    Ok(polylogue_model::signal::StoredOutbox {
        id: record.id,
        at: record.at,
        outbox: polylogue_model::signal::Outbox {
            record: record.prepared,
            metadata,
        },
        events: crate::signals::events(conn, Source::Cache, id)?,
    })
}
