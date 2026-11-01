//! Frozen canonical bodies kept in the SQL outbox until a confirmed product discharges them.
use super::*;
use crate::primary::{ProjectionAvailability, RawInputProof, ReplayState};

pub(crate) const SCHEMA: &str = "
CREATE TABLE primary_derived (
 identity TEXT PRIMARY KEY, body BLOB NOT NULL, proof BLOB, cancelled_at INTEGER,
 revoked INTEGER NOT NULL DEFAULT 0 CHECK(revoked IN (0,1)), custody TEXT
) STRICT;
CREATE TABLE primary_derived_parent (
 identity TEXT NOT NULL REFERENCES primary_derived(identity) ON DELETE CASCADE,
 event TEXT NOT NULL, PRIMARY KEY(identity,event)
) STRICT, WITHOUT ROWID;
";

struct StoredIntent {
    body: Vec<u8>,
    proof: Option<Vec<u8>>,
    cancelled_at: Option<i64>,
    revoked: bool,
}

/// The caller validates domain witnesses in this same transaction. Missing
/// proof retains custody only. Restaging an identical body with a proof requires
/// the caller to validate it again against that proof's current native basis.
pub fn stage_products(
    tx: &Transaction<'_>,
    identity: &str,
    writes: &[ProductWrite],
    proof: Option<&RawInputProof>,
) -> Result<()> {
    if identity.is_empty() || writes.is_empty() {
        return Err(StoreError::Invalid(
            "a canonical intent needs an identity and products".into(),
        ));
    }
    for write in writes {
        if write.location.is_some() || !matches!(write.product, Product::AttachmentBinding { .. }) {
            return Err(StoreError::Invalid(
                "canonical custody requires an unlocated supported derived product".into(),
            ));
        }
    }
    if let Some(proof) = proof
        && (proof.position < 0
            || proof.policy_epoch < 0
            || proof.scope.is_empty()
            || proof.stamp.is_empty())
    {
        return Err(StoreError::Invalid(
            "canonical custody needs an actual owner publication proof".into(),
        ));
    }
    let body = pack(&writes)?;
    let offered_proof = proof.map(pack).transpose()?;
    let old: Option<StoredIntent> = tx
        .query_row(
            "SELECT body,proof,cancelled_at,revoked FROM primary_derived WHERE identity=?1",
            [identity],
            |row| {
                Ok(StoredIntent {
                    body: row.get(0)?,
                    proof: row.get(1)?,
                    cancelled_at: row.get(2)?,
                    revoked: row.get(3)?,
                })
            },
        )
        .optional()?;
    if let Some(old) = old {
        if old.revoked {
            return Err(StoreError::Conflict(identity.into()));
        }
        let same_proof = match (old.proof.as_deref(), proof) {
            (Some(bytes), Some(offered)) => {
                let old: RawInputProof = crate::encoding::unpack_json(bytes)?;
                old.position == offered.position
                    && old.policy_epoch == offered.policy_epoch
                    && old.scope == offered.scope
            }
            _ => true,
        };
        if old.body != body || old.cancelled_at.is_some() || !same_proof {
            return Err(StoreError::Conflict(identity.into()));
        }
        if old.proof.is_none() && offered_proof.is_some() {
            tx.execute(
                "UPDATE primary_derived SET proof=?2 WHERE identity=?1",
                params![identity, offered_proof],
            )?;
        }
        return Ok(());
    }
    tx.execute(
        "INSERT INTO primary_derived(identity,body,proof) VALUES (?1,?2,?3)",
        params![identity, body, offered_proof],
    )?;
    for parent in writes.iter().flat_map(|write| &write.parents) {
        tx.execute(
            "INSERT OR IGNORE INTO primary_derived_parent(identity,event) VALUES (?1,?2)",
            params![identity, parent],
        )?;
    }
    Ok(())
}

/// A native confirmed product, rather than a transport receipt, discharges an
/// outbox batch. Every output must be present in the same applied owner view.
pub(crate) fn confirmed(tx: &Transaction<'_>, position: i64) -> Result<()> {
    let mut intents = tx.prepare(
        "SELECT identity,body FROM primary_derived WHERE cancelled_at IS NULL AND revoked=0",
    )?;
    let mut rows = intents.query([])?;
    while let Some(row) = rows.next()? {
        let identity: String = row.get(0)?;
        let writes: Vec<ProductWrite> = crate::encoding::unpack_json(&row.get::<_, Vec<u8>>(1)?)?;
        let mut complete = true;
        for write in &writes {
            let mut facts = tx.prepare("SELECT product FROM primary_next WHERE state='live' AND kind='attachment_binding' AND native_context IS NOT NULL AND product IS NOT NULL")?;
            let mut facts = facts.query([])?;
            let mut found = false;
            while let Some(fact) = facts.next()? {
                if product(&fact.get::<_, Vec<u8>>(0)?)? == write.product {
                    found = true;
                    break;
                }
            }
            complete &= found;
        }
        if complete {
            tx.execute("DELETE FROM primary_derived WHERE identity=?1", [identity])?;
        }
    }
    tx.execute("UPDATE primary_derived SET cancelled_at=?1 WHERE cancelled_at IS NULL AND revoked=0 AND proof IS NOT NULL AND EXISTS(SELECT 1 FROM primary_derived_parent p LEFT JOIN primary_next f ON f.event=p.event WHERE p.identity=primary_derived.identity AND coalesce(f.state,'missing')<>'live')",[position])?;
    Ok(())
}

pub(crate) fn publish(index: &Connection, sinex: &dyn Sinex, state: &ReplayState) -> Result<()> {
    let applied = state.applied_position.unwrap_or(0);
    let mut query = index.prepare(
        "SELECT identity,body,proof,cancelled_at FROM primary_derived WHERE revoked=0 ORDER BY identity",
    )?;
    let mut rows = query.query([])?;
    let mut discarded = Vec::new();
    while let Some(row) = rows.next()? {
        let identity: String = row.get(0)?;
        if let Some(position) = row.get::<_, Option<i64>>(3)? {
            sinex.discard_confirmed(std::slice::from_ref(&identity), position, applied)?;
            discarded.push(identity);
            continue;
        }
        let Some(proof) = row.get::<_, Option<Vec<u8>>>(2)? else {
            continue;
        };
        let proof: RawInputProof = crate::encoding::unpack_json(&proof)?;
        if state.availability != ProjectionAvailability::Certified
            || state.erasure_fence.is_some()
            || !state.publication.as_ref().is_some_and(|current| {
                current.position == proof.position
                    && current.policy_epoch == proof.policy_epoch
                    && current.scope == proof.scope
            })
            || state.applied_position != Some(proof.position)
            || state.policy_epoch != Some(proof.policy_epoch)
            || state.scope.as_ref() != Some(&proof.scope)
        {
            continue;
        }
        let writes: Vec<ProductWrite> = crate::encoding::unpack_json(&row.get::<_, Vec<u8>>(1)?)?;
        sinex.prepare_canonical_products(&identity, &writes, &proof)?;
    }
    drop(rows);
    drop(query);
    for identity in discarded {
        index.execute("DELETE FROM primary_derived WHERE identity=?1", [identity])?;
    }
    Ok(())
}

/// Revocation preserves frozen intent bytes and historical receipts. Neither
/// confirmation, cancellation nor publication may consume these old intents.
pub(crate) fn revoke(conn: &Connection) -> Result<()> {
    conn.execute("UPDATE primary_derived SET revoked=1", [])?;
    Ok(())
}

/// The producer spool owns the immutable body and revocation. Recovery never
/// converts a historical receipt into current publication eligibility.
pub(crate) fn restore_custody(
    tx: &Transaction<'_>,
    custody: &super::super::CanonicalCustody,
) -> Result<()> {
    if custody.identity.is_empty() || custody.products.is_empty() {
        return Err(StoreError::Invalid(
            "canonical recovery needs its original identified body".into(),
        ));
    }
    for write in &custody.products {
        if write.location.is_some() || !matches!(write.product, Product::AttachmentBinding { .. }) {
            return Err(StoreError::Invalid(
                "canonical recovery contains an unsupported located product".into(),
            ));
        }
        if let Product::AttachmentBinding { record } = &write.product {
            record.validate()?;
        }
    }
    let body = pack(&custody.products)?;
    let old: Option<(Vec<u8>, bool)> = tx
        .query_row(
            "SELECT body,revoked FROM primary_derived WHERE identity=?1",
            [&custody.identity],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()?;
    if let Some((old, revoked)) = old {
        if old != body || (revoked && !custody.revoked) {
            return Err(StoreError::Conflict(custody.identity.clone()));
        }
        if custody.revoked {
            tx.execute(
                "UPDATE primary_derived SET revoked=1 WHERE identity=?1",
                [&custody.identity],
            )?;
        }
    } else {
        // Historical proof stays in durable producer custody. The cache has no
        // current proof until an actual revalidation stages a fresh intent.
        tx.execute(
            "INSERT INTO primary_derived(identity,body,revoked) VALUES (?1,?2,?3)",
            params![custody.identity, body, custody.revoked],
        )?;
        for parent in custody.products.iter().flat_map(|write| &write.parents) {
            tx.execute(
                "INSERT INTO primary_derived_parent(identity,event) VALUES (?1,?2)",
                params![custody.identity, parent],
            )?;
        }
    }
    Ok(())
}

/// Every unacknowledged immutable body reaches the producer's durable local
/// custody before broker acknowledgment; this performs no publication.
pub(crate) fn hold(conn: &Connection, sinex: &dyn Sinex) -> Result<()> {
    let mut query = conn.prepare(
        "SELECT identity,body,proof FROM primary_derived WHERE revoked=0 AND custody IS NULL ORDER BY identity",
    )?;
    let mut rows = query.query([])?;
    while let Some(row) = rows.next()? {
        let identity: String = row.get(0)?;
        let products: Vec<ProductWrite> = crate::encoding::unpack_json(&row.get::<_, Vec<u8>>(1)?)?;
        let proof: Option<RawInputProof> = row
            .get::<_, Option<Vec<u8>>>(2)?
            .as_deref()
            .map(crate::encoding::unpack_json)
            .transpose()?;
        let receipt = sinex.stage_canonical_products(&identity, &products, proof.as_ref())?;
        if receipt.identity.is_empty() {
            return Err(StoreError::Invalid(
                "durable canonical custody has no body reference".into(),
            ));
        }
        conn.execute(
            "UPDATE primary_derived SET custody=?2 WHERE identity=?1",
            params![identity, receipt.identity],
        )?;
    }
    Ok(())
}
