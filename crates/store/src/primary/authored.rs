//! Prepare authored changes once in a private overlay; confirmed records apply
//! their frozen effects without selecting targets, families or claims again.
mod records;
#[cfg(test)]
mod tests;

use crate::source::Source;
use crate::{NewBlob, Result, StoreError};
use polylogue_model::authored::{AuthoredEffect, ImportAttribution, PreparedRequest};
use polylogue_model::user::{UserOutcome, UserRequest};
use rusqlite::{Connection, OptionalExtension, Transaction, TransactionBehavior, params};
use std::collections::HashMap;

/// Semantic idempotence includes the accepting machine and authored intent;
/// a retry timestamp does not change the admitted request.
pub fn request_hash(request: &UserRequest) -> Result<[u8; 32]> {
    Ok(*blake3::hash(&serde_json::to_vec(&(
        &request.machine,
        request.write.intent(),
    ))?)
    .as_bytes())
}
pub(crate) fn pack(value: &impl serde::Serialize) -> Result<Vec<u8>> {
    Ok(crate::encoding::pack_bytes(&serde_json::to_vec(value)?)?)
}

/// The existing request row owns the original outcome and immutable effect.
/// A transport retry returns this value, including its original resolved body.
fn read_source(conn: &Connection) -> Result<Source> {
    if conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM temp.sqlite_schema WHERE name='source_client_request')",
        [],
        |row| row.get::<_, bool>(0),
    )? {
        Ok(Source::Cache)
    } else {
        Source::of(conn)
    }
}

pub fn known_request(conn: &Connection, request: &UserRequest) -> Result<Option<PreparedRequest>> {
    let source = read_source(conn)?;
    let stored=conn.query_row(&source.sql("SELECT content_hash,request,outcome,effect FROM @source.client_request WHERE id=?1"),[&request.id],|row|Ok((row.get::<_,Vec<u8>>(0)?,row.get::<_,Option<Vec<u8>>>(1)?,row.get::<_,Option<Vec<u8>>>(2)?,row.get::<_,Option<Vec<u8>>>(3)?))).optional()?;
    let Some((hash, body, outcome, effect)) = stored else {
        return Ok(None);
    };
    if hash.as_slice() != request_hash(request)? {
        return Err(StoreError::Conflict(request.id.clone()));
    }
    let missing = || {
        StoreError::Invalid(format!(
            "request {} has no complete frozen outcome",
            request.id
        ))
    };
    let stored_request: UserRequest = crate::encoding::unpack_json(&body.ok_or_else(missing)?)?;
    let prepared = PreparedRequest {
        request: stored_request,
        outcome: crate::encoding::unpack_json(&outcome.ok_or_else(missing)?)?,
        effect: crate::encoding::unpack_json(&effect.ok_or_else(missing)?)?,
    };
    if prepared.outcome.pending {
        return Err(StoreError::Invalid(
            "pending is not a canonical outcome".into(),
        ));
    }
    if prepared.request.id != request.id
        || request_hash(&prepared.request)? != request_hash(request)?
    {
        return Err(StoreError::Conflict(request.id.clone()));
    }
    Ok(Some(prepared))
}

/// `visit_pending` includes publisher-acknowledged requests until the confirmed
/// consumer has applied them. Blob metadata is verified by the spool owner.
pub fn prepare(
    index: &Connection,
    request: &UserRequest,
    visit_pending: impl FnOnce(&mut dyn FnMut(&PreparedRequest) -> Result<()>) -> Result<()>,
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
    visit_pending(&mut |pending| apply(&tx, pending, 0))?;
    if let Some(known) = known_request(&tx, request)? {
        return Ok(known);
    }
    let source = Source::of(index)?;
    let available = |attachment: &polylogue_model::user::ChatAttachment| -> Result<bool> {
        if let Some(size) = pending_blobs.get(&attachment.blob) {
            return Ok(*size == attachment.size);
        }
        Ok(index
            .prepare_cached(&source.sql("SELECT 1 FROM @source.blob WHERE hash=?1 AND size=?2"))?
            .exists(params![
                &attachment.blob.0[..],
                i64::try_from(attachment.size)
                    .map_err(|_| StoreError::Invalid("attachment size is out of range".into()))?
            ])?)
    };
    crate::user_writes::apply_using(&tx, request, 1, &available)?;
    known_request(&tx, request)?
        .ok_or_else(|| StoreError::Invalid("prepared request was not recorded".into()))
}

/// Apply immutable facts only. No request selection or domain precondition is
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
            "pending is not a canonical outcome".into(),
        ));
    }
    records::apply(tx, &prepared.effect, counter, false)?;
    insert_request(tx, prepared, counter, &[])
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
    if let Some(known) = known_request(tx, &prepared.request)? {
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
    records::apply(tx, &prepared.effect, counter, true)?;
    let mut imports = claims.to_vec();
    imports.push(current.clone());
    insert_request(tx, prepared, counter, &imports)?;
    Ok(prepared.outcome.clone())
}

fn insert_request(
    conn: &Connection,
    prepared: &PreparedRequest,
    counter: i64,
    imports: &[ImportAttribution],
) -> Result<()> {
    let source = Source::of(conn)?;
    conn.execute(&source.sql("INSERT INTO @source.client_request(id,content_hash,created_at,request,outcome,effect,imports,commit_seq) VALUES(?1,?2,?3,?4,?5,?6,?7,?8)"),params![prepared.request.id,&request_hash(&prepared.request)?[..],prepared.request.at.unix_ms(),pack(&prepared.request)?,pack(&prepared.outcome)?,pack(&prepared.effect)?,serde_json::to_string(imports)?,counter])?;
    Ok(())
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
        "variant_part",
        "variant",
        "note_target",
        "note_relation",
        "delivery_event",
        "chat_action_attachment",
        "chat_action_event",
        "backfill_job_event",
        "note",
        "setting",
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
    let source = read_source(conn)?;
    let mut statement = conn.prepare(&source.sql(
        "SELECT request,imports FROM @source.client_request ORDER BY commit_seq,created_at,id",
    ))?;
    let mut rows = statement.query([])?;
    while let Some(row) = rows.next()? {
        let request: UserRequest = crate::encoding::unpack_json(&row.get::<_, Vec<u8>>(0)?)?;
        let prepared = known_request(conn, &request)?
            .ok_or_else(|| StoreError::NotFound(request.id.clone()))?;
        let imports: Vec<ImportAttribution> = serde_json::from_str(&row.get::<_, String>(1)?)?;
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
