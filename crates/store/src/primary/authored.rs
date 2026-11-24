//! Authored changes are fixed once. Writes that resolve earlier records use a private
//! overlay; literal deliveries come directly from their committed body. Confirmed
//! records apply recorded effects without selecting targets, families or claims again.
mod records;
pub(crate) use records::insert_kept;
#[cfg(test)]
mod tests;

use crate::source::Source;
use crate::{NewBlob, Result, StoreError};
use crate::{OptionExt as _, ResultExt as _};
use polylogue_model::authored::{AuthoredEffect, PendingAuthored, PreparedRequest};
use polylogue_model::user::{UserOutcome, UserRequest, UserWrite};
use rusqlite::{Connection, OptionalExtension, Transaction, TransactionBehavior, params};
use std::collections::HashMap;

/// A retry is the same request when its accepting machine and authored intent
/// match; a retry timestamp does not change the committed request. Stored requests
/// keep this hash, so its bytes carry no tag.
pub fn request_hash(request: &UserRequest) -> Result<[u8; 32]> {
    let canonical = canonical_json::canonical_json(&(
        &request.machine,
        &request.authorship,
        request.write.intent(),
    ))
    .map_err(crate::encoding::identity_encoding)?;
    Ok(*blake3::hash(&canonical).as_bytes())
}
pub(crate) fn pack(value: &impl serde::Serialize) -> Result<Vec<u8>> {
    Ok(crate::encoding::pack_bytes(&serde_json::to_vec(value)?)?)
}

/// The stored request with its outcome and effect. A request whose outcome
/// and effect were not yet written is incomplete.
pub fn stored_request(conn: &Connection, id: &str) -> Result<Option<PreparedRequest>> {
    let source = Source::reading(conn)?;
    let stored = conn
        .query_row(
            &source.sql("SELECT content_hash,request,outcome,effect FROM @source.client_request WHERE id=?1"),
            [id],
            |row| <(Vec<u8>, Vec<u8>, Option<Vec<u8>>, Option<Vec<u8>>)>::try_from(row),
        )
        .optional()?;
    let Some((hash, body, outcome, effect)) = stored else {
        return Ok(None);
    };
    let missing = || StoreError::Invalid(format!("request {id} has no complete recorded outcome"));
    let prepared = PreparedRequest {
        request: crate::encoding::unpack_json(&body)?,
        outcome: crate::encoding::unpack_json(&outcome.ok_or_else(missing)?)?,
        effect: crate::encoding::unpack_json(&effect.ok_or_else(missing)?)?,
    };
    if prepared.request.id != id || hash.as_slice() != request_hash(&prepared.request)? {
        return Err(StoreError::Conflict(id.into()));
    }
    if prepared.outcome.pending {
        return Err(StoreError::Invalid(
            "pending is not a confirmed outcome".into(),
        ));
    }
    Ok(Some(prepared))
}

/// The existing request row owns the original outcome and immutable effect.
pub fn known_request(conn: &Connection, request: &UserRequest) -> Result<Option<PreparedRequest>> {
    match stored_request(conn, &request.id)? {
        None => Ok(None),
        Some(prepared) => {
            if request_hash(&prepared.request)? != request_hash(request)? {
                return Err(StoreError::Conflict(request.id.clone()));
            }
            Ok(Some(prepared))
        }
    }
}

/// The first accepted body, including requests awaiting confirmation. The
/// request journal owns resolution as well as the immutable authored effect.
pub fn pending_request(
    sinex: &dyn crate::primary::Sinex,
    request: &UserRequest,
) -> Result<Option<UserRequest>> {
    Ok(pending_prepared(sinex, request)?.map(|prepared| prepared.request))
}

pub(crate) fn pending_prepared(
    sinex: &dyn crate::primary::Sinex,
    request: &UserRequest,
) -> Result<Option<PreparedRequest>> {
    let mut original = None;
    sinex.visit_pending_requests(&mut |pending| {
        if let Some(prepared) = pending.prepared()
            && prepared.request.id == request.id
        {
            if request_hash(&prepared.request)? != request_hash(request)?
                || original.as_ref().is_some_and(|known| known != prepared)
            {
                return Err(StoreError::Conflict(request.id.clone()));
            }
            original = Some(prepared.clone());
        }
        Ok(())
    })?;
    Ok(original)
}

/// A pending receipt keeps its first accepted body and effect.
pub(crate) fn pending_outcome(
    sinex: &dyn crate::primary::Sinex,
    request: &UserRequest,
) -> Result<Option<UserOutcome>> {
    Ok(
        pending_prepared(sinex, request)?.map(|prepared| UserOutcome {
            pending: true,
            repeat: true,
            ..prepared.outcome
        }),
    )
}

/// `visit_pending` includes requests Sinex acknowledged until the confirmed
/// consumer has applied them. The spool checks blob metadata.
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
    if let Some(prepared) = crate::user_writes::prepare_delivery(request) {
        let mut known = None;
        let mut delivery_exists = false;
        visit_pending(&mut |pending| {
            let Some(original) = pending.prepared() else {
                return Ok(());
            };
            if original.request.id != request.id {
                delivery_exists |= original
                    .effect
                    .deliveries
                    .iter()
                    .any(|change| change.delivery.id == request.id);
                return Ok(());
            }
            match pending {
                PendingAuthored::Live { .. } => validate_live(original)?,
                PendingAuthored::Restore { .. } => validate(original)?,
                PendingAuthored::RestoreAnnotation { .. } => unreachable!(),
            }
            if request_hash(&original.request)? != request_hash(request)?
                || known.as_ref().is_some_and(|known| known != original)
            {
                return Err(StoreError::Conflict(request.id.clone()));
            }
            known = Some(original.clone());
            Ok(())
        })?;
        if let Some(known) = known {
            return Ok(known);
        }
        if delivery_exists {
            return Err(StoreError::Conflict(request.id.clone()));
        }
        require_new_delivery(index, &request.id)?;
        validate_live(&prepared)?;
        return Ok(prepared);
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
                    crate::to_i64(attachment.size, "attachment size is out of range")?
                ])?)
        };
        let receipt = crate::user_writes::apply_using(
            tx,
            index,
            request,
            1,
            &available,
            crate::user_writes::Preparation::plain(preparation, pending),
        )?;
        match known_request(tx, request)? {
            Some(prepared) => Ok(prepared),
            None if matches!(request.write, UserWrite::MarkSeen { .. })
                && receipt.outcome.repeat =>
            {
                Ok(PreparedRequest {
                    request: request.clone(),
                    outcome: receipt.outcome,
                    effect: AuthoredEffect::default(),
                })
            }
            None => Err(StoreError::invalid("prepared request was not recorded")),
        }
    })
}

/// One original snapshot plus this host's pending effects. The callback cannot
/// hold a SQLite borrow across a later spool acknowledgment.
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
    // Private scratch can spill beyond the SQLite page cache; it holds nothing
    // that must survive a crash, so it needs no journal or sync.
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
            PendingAuthored::Restore { prepared } => {
                restore(&tx, prepared, 0)?;
            }
            PendingAuthored::RestoreAnnotation { fact } => {
                crate::annotations::apply(&tx, std::slice::from_ref(fact), 0)?;
            }
        }
        Ok(())
    })?;
    visit(&tx)
}

/// Checks a recorded request's internal consistency without applying it.
pub fn validate(prepared: &PreparedRequest) -> Result<()> {
    if let UserWrite::Save(save) = &prepared.request.write {
        let parts = save
            .outcomes
            .as_ref()
            .or_invalid("recorded Save has no resolved outcomes")?;
        let [kept] = prepared.effect.kept_results.as_slice() else {
            return Err(StoreError::invalid(
                "recorded Save must carry one kept effect",
            ));
        };
        crate::kept::validate_body(&kept.id, &save.rows_json)?;
        let rows: serde_json::Value = serde_json::from_str(&save.rows_json)?;
        if parts.request != prepared.request.id
            || parts.repeat
            || parts.pending
            || parts.kept.result != format!("result:{}", kept.id)
            || parts.kept.at != kept.ran_at
            || Some(parts.kept.rows) != rows["rows"].as_array().map(|rows| rows.len() as u64)
            || (save.watch.is_some() && parts.watch.is_none())
            || (parts.watch.is_some() && save.name.is_none())
            || prepared.outcome.id != prepared.request.id
            || kept.name != save.name
            || kept.query != save.query
            || kept.ran_at != prepared.request.at
            || kept.read_at != save.read_at
            || kept.rows_json != save.rows_json
        {
            return Err(StoreError::invalid(
                "recorded Save outcomes differ from its kept effect",
            ));
        }
        for (id, name, created, kind) in parts
            .saved
            .iter()
            .map(|part| {
                (
                    &part.id,
                    &part.name,
                    part.new_revision,
                    polylogue_model::annotation::AnnotationKind::SavedQuery,
                )
            })
            .chain(parts.watch.iter().map(|part| {
                (
                    &part.id,
                    &part.name,
                    part.new_rule,
                    polylogue_model::annotation::AnnotationKind::Signal,
                )
            }))
        {
            if id.is_empty()
                || Some(name) != save.name.as_ref()
                || (created
                    && !prepared.effect.annotations.iter().any(|fact| {
                        fact.id == *id && fact.kind == kind && fact.key.as_ref() == Some(name)
                    }))
            {
                return Err(StoreError::invalid(
                    "recorded Save has inconsistent annotation outcomes",
                ));
            }
        }
        if parts.saved.as_ref().is_some_and(|part| part.revision == 0) {
            return Err(StoreError::invalid("saved revision numbers start at one"));
        }
    }
    // Restored records can include later events written by other requests.
    // The request still owns its delivery body and its own delivery receipt.
    if let Some(expected) = crate::user_writes::prepare_delivery(&prepared.request) {
        let mut effect = prepared.effect.clone();
        for change in &mut effect.deliveries {
            change
                .delivery
                .events
                .retain(|event| event.request == prepared.request.id);
        }
        if effect != expected.effect || prepared.outcome != expected.outcome {
            return Err(StoreError::Invalid(
                "delivery effect differs from its committed request".into(),
            ));
        }
    }
    crate::signals::validate(prepared)?;
    for fact in &prepared.effect.annotations {
        fact.validate().map_err(crate::StoreError::invalid)?;
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
                    "authored event additions disagree with their recorded history".into(),
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
    for change in &prepared.effect.import_jobs {
        events(
            &change.job.events,
            &change.added_events,
            &prepared.request.id,
            |e| &e.request,
        )?;
    }
    Ok(())
}

fn validate_live(prepared: &PreparedRequest) -> Result<()> {
    validate(prepared)?;
    if let UserWrite::Save(save) = &prepared.request.write {
        let parts = save.outcomes.as_ref().expect("Save validated");
        if parts.saved.is_some() != save.name.is_some() {
            return Err(StoreError::invalid(
                "a new named Save accepts its question as well as its rows",
            ));
        }
        if parts.kept.result != format!("result:{}", prepared.request.id) {
            return Err(StoreError::invalid(
                "a new Save keeps rows under its request identity",
            ));
        }
        let created = parts
            .saved
            .iter()
            .filter(|part| part.new_revision)
            .map(|part| &part.id)
            .chain(
                parts
                    .watch
                    .iter()
                    .filter(|part| part.new_rule)
                    .map(|part| &part.id),
            )
            .collect::<Vec<_>>();
        for (index, id) in created.iter().enumerate() {
            if **id != authored::write_id(&prepared.request.id, index + 1, created.len()) {
                return Err(StoreError::invalid(
                    "a new Save numbers its annotations within the request",
                ));
            }
        }
        if prepared.effect.annotations.len() != created.len() {
            return Err(StoreError::invalid(
                "a new Save carries exactly its accepted annotation parts",
            ));
        }
    }
    if let Some(expected) = crate::user_writes::prepare_delivery(&prepared.request)
        && prepared.effect != expected.effect
    {
        return Err(StoreError::Invalid(
            "a new delivery carries events from other requests".into(),
        ));
    }
    Ok(())
}

// A new literal delivery used to meet the delivery primary key in the
// preparation transaction. Keep that check before the request enters the spool.
pub(crate) fn require_new_delivery(conn: &Connection, id: &str) -> Result<()> {
    let source = Source::reading(conn)?;
    if conn
        .prepare_cached(&source.sql("SELECT 1 FROM @source.delivery WHERE id=?1"))?
        .exists([id])?
    {
        return Err(StoreError::Conflict(id.into()));
    }
    Ok(())
}

/// Apply recorded effects only. No request selection or domain precondition is
/// rerun after Sinex confirms the prepared body.
pub fn apply(tx: &Transaction<'_>, prepared: &PreparedRequest, counter: i64) -> Result<()> {
    if let Some(known) = known_request(tx, &prepared.request)? {
        return if known == *prepared {
            Ok(())
        } else {
            Err(StoreError::Conflict(prepared.request.id.clone()))
        };
    }
    if prepared.outcome.pending {
        return Err(StoreError::Invalid(
            "pending is not a confirmed outcome".into(),
        ));
    }
    validate_live(prepared)?;
    records::apply(tx, &prepared.effect, counter, false)?;
    insert_request(tx, prepared, counter)
}

/// Restores a request a bundle carried. Its records keep their author and
/// state, and the actions among them never run on this host.
pub fn restore(
    tx: &Transaction<'_>,
    prepared: &PreparedRequest,
    counter: i64,
) -> Result<UserOutcome> {
    if let Some(known) = stored_request(tx, &prepared.request.id)? {
        if known != *prepared {
            return Err(StoreError::Conflict(prepared.request.id.clone()));
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
    validate(prepared)?;
    records::apply(tx, &prepared.effect, counter, true)?;
    insert_request(tx, prepared, counter)?;
    Ok(prepared.outcome.clone())
}

fn insert_request(conn: &Connection, prepared: &PreparedRequest, counter: i64) -> Result<()> {
    let source = Source::of(conn)?;
    conn.execute(&source.sql("INSERT INTO @source.client_request(id,content_hash,created_at,request,outcome,effect,commit_seq) VALUES(?1,?2,?3,?4,?5,?6,?7)"),params![prepared.request.id,&request_hash(&prepared.request)?[..],prepared.request.at.unix_ms(),pack(&prepared.request)?,pack(&prepared.outcome)?,pack(&prepared.effect)?,counter])?;
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

/// Authored lifecycle rebuilds use only the confirmed live authored records.
pub fn clear(conn: &Connection) -> Result<()> {
    let source = Source::of(conn)?;
    for table in [
        "revision_state",
        "annotation_slot",
        "annotation_link",
        "annotation",
        "alternate_part",
        "alternate",
        "signal_event",
        "signal_state",
        "delivery_event",
        "chat_action_attachment",
        "chat_action_event",
        "import_job_event",
        "kept_result",
        "delivery",
        "chat_action",
        "import_job",
        "client_request",
    ] {
        conn.execute(&source.sql(&format!("DELETE FROM @source.{table}")), [])?;
    }
    Ok(())
}

/// Streams stored requests in write order.
pub fn visit(
    conn: &Connection,
    accept: &mut dyn FnMut(&PreparedRequest) -> Result<()>,
) -> Result<()> {
    let source = Source::reading(conn)?;
    let mut statement = conn.prepare(
        &source.sql("SELECT id FROM @source.client_request ORDER BY commit_seq,created_at,id"),
    )?;
    let ids = statement
        .query_map([], |row| row.get::<_, String>(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    for id in ids {
        let prepared =
            stored_request(conn, &id)?.ok_or_else(|| StoreError::NotFound(id.clone()))?;
        accept(&prepared)?;
    }
    Ok(())
}
impl crate::Reader<'_> {
    pub fn visit_authored(
        &self,
        accept: &mut dyn FnMut(&PreparedRequest) -> Result<()>,
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
        .or_invalid("delivery has no signal metadata")?;
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
