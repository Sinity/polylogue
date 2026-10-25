//! Change cursors name an index file, not just its reusable write counter.
use crate::{Result, archive::cache};
use rusqlite::{Connection, OptionalExtension};

pub(crate) fn create(index: &Connection) -> Result<()> {
    let boot = std::fs::read_to_string("/proc/sys/kernel/random/boot_id")?;
    index.execute(
        "INSERT INTO meta(key,value) VALUES ('generation',?1),('boot_id',?2)",
        [uuid::Uuid::now_v7().to_string(), boot.trim().to_owned()],
    )?;
    Ok(())
}

pub(super) fn open(archive: &super::Archive) -> Result<()> {
    let boot = std::fs::read_to_string("/proc/sys/kernel/random/boot_id")?;
    let index = super::open_write(&archive.index_db(), cache::SETUP)?;
    begin(&index, boot.trim())
}

fn begin(index: &Connection, boot: &str) -> Result<()> {
    if boot.is_empty() {
        return Err(crate::StoreError::Invalid(
            "the current boot has no identity".into(),
        ));
    }
    let previous: i64 = index.pragma_query_value(None, "synchronous", |r| r.get(0))?;
    index.pragma_update(None, "synchronous", previous.max(2))?;
    let result = (|| {
        let tx = crate::write_lock::transaction(index, rusqlite::TransactionBehavior::Deferred)?;
        let value = |key| -> Result<Option<String>> {
            Ok(tx
                .query_row("SELECT value FROM meta WHERE key=?1", [key], |r| r.get(0))
                .optional()?)
        };
        let generation = value("generation")?;
        // Changing boot always invalidates cursors. A stale clean-stop marker
        // therefore cannot authorize reuse after a lost NORMAL transaction.
        if generation.as_deref().is_none_or(str::is_empty)
            || value("boot_id")?.as_deref() != Some(boot)
        {
            let generation = uuid::Uuid::now_v7().to_string();
            tx.execute("INSERT INTO meta(key,value) VALUES ('generation',?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value", [generation])?;
            tx.execute("DELETE FROM removed_session", [])?;
        }
        tx.execute("INSERT INTO meta(key,value) VALUES ('boot_id',?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value", [boot])?;
        tx.commit()?;
        Ok(())
    })();
    index.pragma_update(None, "synchronous", previous)?;
    result
}

#[cfg(test)]
mod tests;
