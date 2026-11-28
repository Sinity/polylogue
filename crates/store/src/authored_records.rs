//! Authored records: the stored requests with their outcomes and effects.
//! Authored changes are fixed once. A saved query's preflight resolves earlier
//! records in a private copy ([`with_overlay`]); literal deliveries come
//! directly from their committed body; restored records apply recorded
//! effects without selecting targets, families or claims again.
mod records;
pub(crate) use records::{delivery_events_from, insert_kept};

use crate::source::Source;
use crate::{OptionExt as _, ResultExt as _};
use crate::{Result, StoreError};
use polylogue_model::authored::{AuthoredEffect, PreparedRequest};
use polylogue_model::user::{UserOutcome, UserRequest, UserWrite};
use rusqlite::{Connection, OptionalExtension, Transaction, TransactionBehavior, params};

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
    let missing = || StoreError::internal(format!("request {id} has no complete recorded outcome"));
    let prepared = PreparedRequest {
        request: crate::encoding::unpack_json(&body)?,
        outcome: crate::encoding::unpack_json(&outcome.ok_or_else(missing)?)?,
        effect: crate::encoding::unpack_json(&effect.ok_or_else(missing)?)?,
    };
    if prepared.request.id != id || hash.as_slice() != request_hash(&prepared.request)? {
        return Err(StoreError::Conflict(id.into()));
    }
    if prepared.outcome.pending {
        return Err(StoreError::internal("pending is not a confirmed outcome"));
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

/// A private copy of the original snapshot's authored records, in which a
/// preflight resolves its request without writing the archive.
pub(crate) fn with_overlay<T>(
    index: &Connection,
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
    visit(&tx)
}

/// Checks a recorded request's internal consistency without applying it.
pub fn validate(prepared: &PreparedRequest) -> Result<()> {
    if let UserWrite::Save(save) = &prepared.request.write {
        let parts = save
            .outcomes
            .as_ref()
            .or_internal("recorded Save has no resolved outcomes")?;
        let [kept] = prepared.effect.saved_rows.as_slice() else {
            return Err(StoreError::internal(
                "recorded Save must carry one kept effect",
            ));
        };
        crate::kept::validate_body(&kept.id, &save.rows_json)?;
        let rows: serde_json::Value = serde_json::from_str(&save.rows_json)?;
        if parts.request != prepared.request.id
            || parts.repeat
            || parts.pending
            || parts.saved_rows.result != format!("result:{}", kept.id)
            || parts.saved_rows.at != kept.ran_at
            || Some(parts.saved_rows.rows) != rows["rows"].as_array().map(|rows| rows.len() as u64)
            || (save.watch.is_some() && parts.watch.is_none())
            || (parts.watch.is_some() && save.name.is_none())
            || prepared.outcome.id != prepared.request.id
            || kept.name != save.name
            || kept.query != save.query
            || kept.ran_at != prepared.request.at
            || kept.read_position != save.read_position
            || kept.rows_json != save.rows_json
        {
            return Err(StoreError::internal(
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
                return Err(StoreError::internal(
                    "recorded Save has inconsistent annotation outcomes",
                ));
            }
        }
        if parts.saved.as_ref().is_some_and(|part| part.revision == 0) {
            return Err(StoreError::internal("saved revision numbers start at one"));
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
            return Err(StoreError::internal(
                "delivery effect differs from its committed request",
            ));
        }
    }
    crate::signals::validate(prepared)?;
    for fact in &prepared.effect.annotations {
        fact.validate().map_err(crate::StoreError::internal)?;
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
                return Err(StoreError::internal(
                    "authored history has a duplicate or empty request identity",
                ));
            }
        }
        for event in added {
            if identity(event) != request || !all.contains(event) {
                return Err(StoreError::internal(
                    "authored event additions disagree with their recorded history",
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
            return Err(StoreError::internal(
                "a new named Save accepts its question as well as its rows",
            ));
        }
        if parts.saved_rows.result != format!("result:{}", prepared.request.id) {
            return Err(StoreError::internal(
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
                return Err(StoreError::internal(
                    "a new Save numbers its annotations within the request",
                ));
            }
        }
        if prepared.effect.annotations.len() != created.len() {
            return Err(StoreError::internal(
                "a new Save carries exactly its accepted annotation parts",
            ));
        }
    }
    if let Some(expected) = crate::user_writes::prepare_delivery(&prepared.request)
        && prepared.effect != expected.effect
    {
        return Err(StoreError::internal(
            "a new delivery carries events from other requests",
        ));
    }
    Ok(())
}

/// A literal delivery's ID is new: it is checked before the request applies.
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
/// run again for a prepared body.
pub fn apply(tx: &Transaction<'_>, prepared: &PreparedRequest, counter: i64) -> Result<()> {
    if let Some(known) = known_request(tx, &prepared.request)? {
        return if known == *prepared {
            Ok(())
        } else {
            Err(StoreError::Conflict(prepared.request.id.clone()))
        };
    }
    if prepared.outcome.pending {
        return Err(StoreError::internal("pending is not a confirmed outcome"));
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
        return Err(StoreError::internal("pending is not a restorable outcome"));
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

/// Uses the logical source views of a Reader.
pub(crate) fn pending_delivery(
    conn: &Connection,
    id: &str,
) -> Result<polylogue_model::signal::StoredPendingDelivery> {
    let record = records::delivery_from(conn, Source::Cache, id)?;
    let metadata = record
        .signal
        .or_internal("delivery has no signal metadata")?;
    Ok(polylogue_model::signal::StoredPendingDelivery {
        id: record.id,
        at: record.at,
        pending_delivery: polylogue_model::signal::PendingDelivery {
            record: record.prepared,
            metadata,
        },
        events: crate::signals::events(conn, Source::Cache, id)?,
    })
}
