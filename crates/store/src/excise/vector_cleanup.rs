//! Exact vector deletions whose canonical excision has committed.
use crate::{Archive, Result, StoreError};
use rusqlite::Connection;

pub(crate) fn pending(conn: &Connection) -> Result<bool> {
    Ok(conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM pending_vector_excision)",
        [],
        |r| r.get(0),
    )?)
}

pub(crate) fn clear(conn: &Connection, id: &str) -> Result<()> {
    conn.execute(
        "DELETE FROM pending_vector_excision WHERE excision_id=?1",
        [id],
    )?;
    Ok(())
}

/// Caller holds the archive lock. Commit exact vector deletions before clearing
/// their intent; an interrupted attempt repeats those same deletions.
pub(crate) fn recover(archive: &Archive, canonical: &mut Connection) -> Result<()> {
    let mut vectors =
        crate::archive::open_write(&archive.embeddings_db(), crate::archive::cache::SETUP)?;
    vectors.pragma_update(None, "synchronous", "full")?;
    let _publication = crate::publication::write(canonical)?;
    let vtx = vectors.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
    let atx = canonical.transaction()?;
    super::vector_policy::reconcile(&vtx, &atx)?;
    {
        let mut select =
            atx.prepare("SELECT text_hash FROM pending_vector_excision WHERE excision_id NOT IN (SELECT excision_id FROM vector_deletion_policy) ORDER BY text_hash")?;
        let mut rows = select.query([])?;
        while let Some(row) = rows.next()? {
            let hash: Vec<u8> = row.get(0)?;
            vtx.execute("DELETE FROM embedded WHERE text_hash=?1", [&hash])?;
            vtx.execute("DELETE FROM embed_failure WHERE text_hash=?1", [&hash])?;
        }
    }
    vtx.commit()?;
    crate::fault::check("excision.after_vector_cleanup", "")?;
    atx.execute("DELETE FROM pending_vector_excision", [])?;
    atx.commit()?;
    Ok(())
}

/// Vector mutation already holds the embeddings writer lock. Excision takes
/// that lock first too, so it cannot commit between this check and the write.
pub(crate) fn check_writer(conn: &Connection, expected: &super::recovery::Revision) -> Result<()> {
    let (version, id, pending) = conn.query_row(
        "SELECT
         COALESCE((SELECT value FROM canonical.meta WHERE key='excision_version'),'0'),
         COALESCE((SELECT value FROM canonical.meta WHERE key='excision_revision'),''),
         EXISTS(SELECT 1 FROM canonical.pending_vector_excision)",
        [],
        |r| Ok((r.get(0)?, r.get(1)?, r.get::<_, bool>(2)?)),
    )?;
    if super::recovery::Revision::decode(version, id)? != *expected || pending {
        return Err(StoreError::ExcisionPending);
    }
    super::vector_policy::check(conn)
}
