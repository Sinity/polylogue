//! The index generation: a fresh ID for each index build, so a change feed
//! position from an earlier build is recognized as one.
use crate::Result;
use rusqlite::Connection;

pub(crate) fn create(index: &Connection) -> Result<()> {
    let boot = std::fs::read_to_string("/proc/sys/kernel/random/boot_id")?;
    index.execute(
        "INSERT INTO meta(key,value) VALUES ('generation',?1),('boot_id',?2),('cursor_key',?3)",
        [
            uuid::Uuid::now_v7().to_string(),
            boot.trim().to_owned(),
            cursor_key()?,
        ],
    )?;
    Ok(())
}

/// A new secret for sealing read cursors: 32 random bytes, hex.
fn cursor_key() -> Result<String> {
    use std::io::Read;
    let mut bytes = [0u8; 32];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut bytes)?;
    Ok(polylogue_model::hex(&bytes))
}

pub(crate) fn rotate(index: &Connection) -> Result<()> {
    for (key, value) in [
        ("generation", uuid::Uuid::now_v7().to_string()),
        ("cursor_key", cursor_key()?),
    ] {
        crate::meta_set(index, key, value)?;
    }
    Ok(())
}

pub(super) fn open(archive: &super::Archive, index: &Connection) -> Result<()> {
    let boot = std::fs::read_to_string("/proc/sys/kernel/random/boot_id")?;
    begin(index, boot.trim(), Some(&archive.read_scope))
}

fn begin(
    index: &Connection,
    boot: &str,
    scope: Option<&crate::read::scope::ReadScopeState>,
) -> Result<()> {
    if boot.is_empty() {
        return Err(crate::StoreError::internal(
            "the current boot has no identity",
        ));
    }
    with_full_sync(index, || {
        let tx = index.unchecked_transaction()?;
        let generation = crate::meta_get(&tx, "generation")?;
        let mut changed = false;
        // Changing boot always invalidates cursors, so a lost NORMAL
        // transaction cannot leave positions that look reusable.
        if generation.as_deref().is_none_or(str::is_empty)
            || crate::meta_get(&tx, "boot_id")?.as_deref() != Some(boot)
        {
            if let Some(scope) = scope {
                let mut reset = scope.begin_reset()?;
                reset.invalidate()?;
            }
            changed = true;
            rotate(&tx)?;
            tx.execute("DELETE FROM removed_session", [])?;
        }
        crate::meta_set(&tx, "boot_id", boot)?;
        tx.commit()?;
        if changed && let Some(scope) = scope {
            scope.finish_reset();
        }
        Ok(())
    })
}

fn with_full_sync(index: &Connection, work: impl FnOnce() -> Result<()>) -> Result<()> {
    let previous: i64 = index.pragma_query_value(None, "synchronous", |r| r.get(0))?;
    index.pragma_update(None, "synchronous", previous.max(2))?;
    let result = work();
    index.pragma_update(None, "synchronous", previous)?;
    result
}

#[cfg(test)]
mod tests;
