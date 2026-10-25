//! Excising bytes read from a ZIP removes the ZIP: its compressed bytes
//! hold the same text, and they cannot be rewritten in place.
use super::*;
use polylogue_model::BlobHash;
use std::collections::BTreeSet;

/// The ZIPs the `selected` inputs were read from, and the ZIPs those were
/// read from in turn.
pub(crate) fn containing(
    conn: &Connection,
    selected: &BTreeSet<String>,
) -> Result<BTreeSet<String>> {
    let mut originals = BTreeSet::new();
    let mut pending: Vec<String> = selected.iter().cloned().collect();
    let mut seen: BTreeSet<String> = selected.clone();
    let mut stmt = conn.prepare_cached(
        "SELECT DISTINCT c.container_input FROM observation_container c
         JOIN observation o ON o.id = c.observation_id WHERE o.input = ?1",
    )?;
    while let Some(input) = pending.pop() {
        for zip in stmt.query_map([&input], |r| r.get::<_, String>(0))? {
            let zip = zip?;
            originals.insert(zip.clone());
            if seen.insert(zip.clone()) {
                pending.push(zip);
            }
        }
    }
    Ok(originals)
}

/// The removed ZIPs no longer back their members' observations.
pub(crate) fn detach(conn: &Connection, originals: &BTreeSet<String>) -> Result<()> {
    let mut stmt =
        conn.prepare_cached("DELETE FROM observation_container WHERE container_input=?1")?;
    for original in originals {
        stmt.execute([original])?;
    }
    Ok(())
}

/// Refuses when a ZIP's bytes are also another kept input's, or shown in
/// the index, or the base of a delta: removing the ZIP would take them too.
pub(crate) fn require_removable(
    archive: &Connection,
    index: &Connection,
    originals: &BTreeSet<String>,
    whole: &BTreeSet<String>,
) -> Result<()> {
    for original in originals {
        let hash = BlobHash::parse(original)
            .map_err(|_| StoreError::Invalid("invalid original container identity".into()))?;
        let unsupported = || {
            StoreError::ExcisionFormat {
            input: original.clone(), format: "zip".into(),
            detail: "the ZIP's bytes are also used elsewhere; removing them is not supported".into(),
        }
        };
        let mut owners = archive.prepare_cached("SELECT input FROM extent WHERE blob=?1")?;
        for input in owners.query_map([&hash.0[..]], |r| r.get::<_, String>(0))? {
            if !whole.contains(&input?) {
                return Err(unsupported());
            }
        }
        let presentation: bool=index.query_row("SELECT EXISTS(SELECT 1 FROM document_block WHERE blob_hash=?1) OR EXISTS(SELECT 1 FROM block WHERE blob_hash=?1)",[&hash.0[..]],|r|r.get(0))?;
        let delta: bool = archive.query_row(
            "SELECT EXISTS(SELECT 1 FROM blob WHERE base=?1)",
            [&hash.0[..]],
            |r| r.get(0),
        )?;
        if presentation || delta {
            return Err(unsupported());
        }
    }
    Ok(())
}
