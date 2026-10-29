//! Frozen canonical bodies share the existing SQL outbox with work replacements.
use super::*;
use crate::primary::{ProjectionAvailability, RawInputProof, ReplayState};

pub(crate) const SCHEMA: &str = "
CREATE TABLE primary_derived (
 identity TEXT PRIMARY KEY, body BLOB NOT NULL, proof BLOB, cancelled_at INTEGER
) STRICT;
CREATE TABLE primary_derived_parent (
 identity TEXT NOT NULL REFERENCES primary_derived(identity) ON DELETE CASCADE,
 event TEXT NOT NULL, PRIMARY KEY(identity,event)
) STRICT, WITHOUT ROWID;
CREATE TABLE primary_work_dirty (session TEXT PRIMARY KEY) STRICT, WITHOUT ROWID;
";

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
        if write.location.is_some()
            || !matches!(
                write.product,
                Product::WorkSnapshot { .. }
                    | Product::WorkComponentStatus { .. }
                    | Product::AttachmentBinding { .. }
            )
        {
            return Err(StoreError::Invalid(
                "canonical custody requires an unlocated supported derived product".into(),
            ));
        }
    }
    if let Some(proof) = proof {
        if proof.position < 0
            || proof.policy_epoch < 0
            || proof.scope.is_empty()
            || proof.stamp.is_empty()
        {
            return Err(StoreError::Invalid(
                "canonical custody needs an actual owner publication proof".into(),
            ));
        }
    }
    let body = pack(&writes)?;
    let offered_proof = proof.map(pack).transpose()?;
    let old: Option<(Vec<u8>, Option<Vec<u8>>, Option<i64>)> = tx
        .query_row(
            "SELECT body,proof,cancelled_at FROM primary_derived WHERE identity=?1",
            [identity],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .optional()?;
    if let Some((old_body, old_proof, cancelled)) = old {
        if old_body != body
            || cancelled.is_some()
            || (old_proof.is_some() && offered_proof.is_some() && old_proof != offered_proof)
        {
            return Err(StoreError::Conflict(identity.into()));
        }
        if old_proof.is_none() && offered_proof.is_some() {
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
pub(super) fn confirmed(tx: &Transaction<'_>, position: i64) -> Result<()> {
    let mut intents =
        tx.prepare("SELECT identity,body FROM primary_derived WHERE cancelled_at IS NULL")?;
    let mut rows = intents.query([])?;
    while let Some(row) = rows.next()? {
        let identity: String = row.get(0)?;
        let writes: Vec<ProductWrite> = crate::encoding::unpack_json(&row.get::<_, Vec<u8>>(1)?)?;
        let mut complete = true;
        for write in &writes {
            let mut facts = tx.prepare("SELECT product FROM primary_next WHERE state='live' AND kind IN ('derived','attachment_binding') AND native_context IS NOT NULL AND product IS NOT NULL")?;
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
    tx.execute("UPDATE primary_derived SET cancelled_at=?1 WHERE cancelled_at IS NULL AND proof IS NOT NULL AND EXISTS(SELECT 1 FROM primary_derived_parent p LEFT JOIN primary_next f ON f.event=p.event WHERE p.identity=primary_derived.identity AND coalesce(f.state,'missing')<>'live')",[position])?;
    Ok(())
}

pub(crate) fn publish(index: &Connection, sinex: &dyn Sinex, state: &ReplayState) -> Result<()> {
    let applied = state.applied_position.unwrap_or(0);
    let mut query = index.prepare(
        "SELECT identity,body,proof,cancelled_at FROM primary_derived ORDER BY identity",
    )?;
    let mut rows = query.query([])?;
    let mut discarded = Vec::new();
    while let Some(row) = rows.next()? {
        let identity: String = row.get(0)?;
        if let Some(position) = row.get::<_, Option<i64>>(3)? {
            sinex.discard_confirmed(&[identity.clone()], position, applied)?;
            discarded.push(identity);
            continue;
        }
        let Some(proof) = row.get::<_, Option<Vec<u8>>>(2)? else {
            continue;
        };
        let proof: RawInputProof = crate::encoding::unpack_json(&proof)?;
        if state.availability != ProjectionAvailability::Certified
            || state.erasure_fence.is_some()
            || state.publication.as_ref() != Some(&proof)
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
