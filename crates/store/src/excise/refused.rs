//! What excisions removed, as `excised` rows, and the writer's lookups in
//! them: excised sessions, messages and documents are never written again,
//! and excised inputs, bytes and locations are refused on later reads.

use super::*;
use std::collections::HashSet;

pub(super) fn record(archive: &Connection, excision: &str, kind: &str, value: &str) -> Result<()> {
    archive
        .prepare_cached(
            "INSERT OR IGNORE INTO excised(excision_id, kind, value) VALUES (?1, ?2, ?3)",
        )?
        .execute(params![excision, kind, value])?;
    Ok(())
}

/// Whether the archive holds any excision, so the writer can skip its
/// lookups when none does.
pub(crate) fn any_excision(archive: &Connection) -> Result<bool> {
    Ok(archive.query_row("SELECT EXISTS(SELECT 1 FROM excised)", [], |r| r.get(0))?)
}

/// The excised identities among those a writer group names.
#[derive(Debug, Default)]
pub(crate) struct RefusedIds {
    pub sessions: HashSet<String>,
    pub messages: HashSet<String>,
    pub documents: HashSet<String>,
}

impl RefusedIds {
    pub fn is_empty(&self) -> bool {
        self.sessions.is_empty() && self.messages.is_empty() && self.documents.is_empty()
    }
}

fn excised(archive: &Connection, kind: &str, value: &str) -> Result<bool> {
    Ok(archive
        .prepare_cached("SELECT EXISTS(SELECT 1 FROM excised WHERE kind = ?1 AND value = ?2)")?
        .query_row(params![kind, value], |r| r.get(0))?)
}

pub(crate) fn matching<'a>(
    archive: &Connection,
    sessions: impl IntoIterator<Item = &'a String>,
    messages: impl IntoIterator<Item = &'a String>,
    documents: impl IntoIterator<Item = &'a String>,
) -> Result<RefusedIds> {
    let mut out = RefusedIds::default();
    for (kind, ids, found) in [
        (
            "session",
            sessions.into_iter().collect::<Vec<_>>(),
            &mut out.sessions,
        ),
        ("message", messages.into_iter().collect(), &mut out.messages),
        (
            "document",
            documents.into_iter().collect(),
            &mut out.documents,
        ),
    ] {
        for id in ids {
            if excised(archive, kind, id)? {
                found.insert(id.clone());
            }
        }
    }
    Ok(out)
}

/// Whether bytes read at `observation` are refused: the input, these bytes
/// or the location was excised.
pub(crate) fn raw_import(
    archive: &Connection,
    input: &str,
    hash: &BlobHash,
    observation: &crate::Observation,
) -> Result<bool> {
    let hash = hash.to_string();
    Ok(excised(archive, "input", input)?
        || excised(archive, "blob", &hash)?
        || excised(
            archive,
            "location",
            &serde_json::to_string(&(&observation.machine, &observation.path))?,
        )?)
}

impl crate::Reader<'_> {
    /// Paths (`(machine, path)`) the loop must never collect again.
    pub fn refused_paths(&self) -> Result<Vec<(String, String)>> {
        let mut stmt = self
            .conn()?
            .prepare("SELECT DISTINCT value FROM a.excised WHERE kind = 'location'")?;
        let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
        let mut out = Vec::new();
        for row in rows {
            out.push(serde_json::from_str(&row?)?);
        }
        Ok(out)
    }
}
