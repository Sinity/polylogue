//! Retire containing representations without manufacturing replacement originals.
use super::*;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// Iterative closure over canonical member ownership, independent of Sessions.
/// All source facts must still have their admitted exact byte basis.
pub(crate) fn containing(
    conn: &Connection,
    selected: &BTreeSet<String>,
) -> Result<BTreeSet<String>> {
    verify_all(conn)?;
    let mut visited = BTreeSet::new();
    let mut pending: VecDeque<_> = selected.iter().cloned().collect();
    let mut edges = BTreeMap::<String, BTreeSet<String>>::new();
    let mut originals = BTreeSet::new();
    while let Some(input) = pending.pop_front() {
        if !visited.insert(input.clone()) {
            continue;
        }
        let mut stmt = conn.prepare_cached("SELECT DISTINCT c.container_input FROM observation_container c JOIN observation o ON o.id=c.observation_id WHERE o.input=?1 ORDER BY c.container_input")?;
        for parent in stmt.query_map([&input], |r| r.get::<_, String>(0))? {
            let parent = parent?;
            edges
                .entry(input.clone())
                .or_default()
                .insert(parent.clone());
            originals.insert(parent.clone());
            pending.push_back(parent);
        }
    }
    // A forged cycle cannot describe nested original ZIP byte streams.
    let mut indegree: BTreeMap<_, usize> = visited.iter().map(|s| (s.clone(), 0)).collect();
    for parents in edges.values() {
        for parent in parents {
            *indegree.get_mut(parent).expect("visited parent") += 1;
        }
    }
    let mut ready: VecDeque<_> = indegree
        .iter()
        .filter(|(_, n)| **n == 0)
        .map(|(s, _)| s.clone())
        .collect();
    let mut count = 0;
    while let Some(input) = ready.pop_front() {
        count += 1;
        if let Some(parents) = edges.get(&input) {
            for parent in parents {
                let degree = indegree.get_mut(parent).expect("visited parent");
                *degree -= 1;
                if *degree == 0 {
                    ready.push_back(parent.clone());
                }
            }
        }
    }
    if count != visited.len() {
        return Err(StoreError::Invalid(
            "cyclic container relationships cannot be excised".into(),
        ));
    }
    Ok(originals)
}

/// The original is unavailable after retirement. Member observations and their
/// acquired names survive, but cannot continue to claim an available container.
pub(crate) fn detach(conn: &Connection, originals: &BTreeSet<String>) -> Result<()> {
    let mut stmt =
        conn.prepare_cached("DELETE FROM observation_container WHERE container_input=?1")?;
    for original in originals {
        stmt.execute([original])?;
    }
    Ok(())
}

/// Input removal preserves unrelated byte owners. If one holds an entire
/// forbidden container representation, retiring only its import is insufficient.
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
            detail: "original ZIP bytes have another retained owner; coordinated removal of that representation is unsupported".into(),
        }
        };
        let mut owners=archive.prepare_cached("SELECT input FROM extent WHERE blob=?1 UNION SELECT input FROM extracted_blob WHERE blob=?1")?;
        for input in owners.query_map([&hash.0[..]], |r| r.get::<_, String>(0))? {
            if !whole.contains(&input?) {
                return Err(unsupported());
            }
        }
        let presentation: bool=index.query_row("SELECT EXISTS(SELECT 1 FROM version_block WHERE blob_hash=?1) OR EXISTS(SELECT 1 FROM block WHERE blob_hash=?1)",[&hash.0[..]],|r|r.get(0))?;
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
