//! The stored currency index. `revision_state` holds, for every assertion
//! revision, whether it is current and whether it is withdrawn, by the shared
//! descendant rule. Writes recompute exactly the roots they can affect in the
//! writing transaction, so the index always equals the rule over the links;
//! reads only look it up.
use super::{StoredLinks, currency_error, exact, invalid};
use crate::{Reader, Result, source::Source};
use authored::Currency;
use polylogue_model::{Reference, RevisionReference, Timestamp};
use rusqlite::{Connection, OptionalExtension, params};
use std::collections::BTreeSet;

/// Whether an assertion family is present and which of its revisions are
/// current. No current revision means every revision is withdrawn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FamilyState {
    pub root: Reference,
    pub heads: Vec<Reference>,
}
impl FamilyState {
    pub fn withdrawn(&self) -> bool {
        self.heads.is_empty()
    }
    pub fn conflict(&self) -> bool {
        self.heads.len() > 1
    }
}

/// One revision's state in its family.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RevisionState {
    pub reference: Reference,
    pub current: bool,
    pub withdrawn: bool,
    pub conflict: bool,
}

/// Recomputes `revision_state` for `roots` from the stored links. The caller
/// owns the writing transaction.
pub(crate) fn recompute(conn: &Connection, roots: BTreeSet<String>) -> Result<()> {
    let source = Source::of(conn)?;
    let mut links = StoredLinks::new(conn)?;
    let mut states = Vec::with_capacity(roots.len());
    {
        let mut currency = Currency::new(&mut links);
        for root in &roots {
            states.push(currency.root_state(root).map_err(currency_error)?);
        }
    }
    // Every root's old rows go first: a revision that moved between two of
    // the roots must not meet its old row.
    for state in &states {
        conn.execute(
            &source.sql("DELETE FROM @source.revision_state WHERE root=?1"),
            [&state.root],
        )?;
    }
    for state in states {
        for revision in &state.revisions {
            conn.execute(
                &source.sql("INSERT INTO @source.revision_state(id,root,current,withdrawn) VALUES(?1,?2,?3,?4)"),
                params![
                    revision.revision.rev,
                    state.root,
                    revision.current,
                    revision.withdrawn
                ],
            )?;
        }
    }
    Ok(())
}

pub(crate) fn stored_heads(conn: &Connection, root: &str) -> Result<Vec<Reference>> {
    let source = Source::reading(conn)?;
    let mut statement = conn
        .prepare_cached(&source.sql(
            "SELECT id FROM @source.revision_state WHERE root=?1 AND current=1 ORDER BY id",
        ))?;
    Ok(statement
        .query_map([root], |row| Ok(exact(root, &row.get::<_, String>(0)?)))?
        .collect::<rusqlite::Result<_>>()?)
}

fn family(conn: &Connection, root: &str) -> Result<Option<FamilyState>> {
    let source = Source::reading(conn)?;
    let present = conn
        .prepare_cached(&source.sql("SELECT 1 FROM @source.revision_state WHERE root=?1 LIMIT 1"))?
        .exists([root])?;
    if !present {
        return Ok(None);
    }
    Ok(Some(FamilyState {
        root: Reference::Assertion(RevisionReference {
            root: root.into(),
            revision: None,
        }),
        heads: stored_heads(conn, root)?,
    }))
}

/// Fills `temp.assertion_history` with which revisions were current among the
/// records written before `cutoff` (Unix milliseconds), for as-of reads. The
/// rows belong to the pinned read transaction.
pub(crate) fn prepare_history(conn: &Connection, cutoff: i64) -> Result<()> {
    if conn.is_autocommit() {
        return Err(invalid(
            "as-of assertion currency requires a pinned read transaction",
        ));
    }
    conn.execute_batch(
        "CREATE TEMP TABLE IF NOT EXISTS assertion_history (
            cutoff INTEGER NOT NULL,
            id TEXT NOT NULL,
            root TEXT NOT NULL,
            current INTEGER NOT NULL,
            PRIMARY KEY(cutoff,id)
        ) STRICT, WITHOUT ROWID;",
    )?;
    let until = Timestamp::from_unix_ms(cutoff)
        .ok_or_else(|| invalid("as-of assertion cutoff is out of range"))?;
    let source = Source::reading(conn)?;
    let roots = conn
        .prepare_cached(&source.sql("SELECT DISTINCT root FROM @source.assertion ORDER BY root"))?
        .query_map([], |row| row.get::<_, String>(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    let mut links = StoredLinks::before(conn, until)?;
    let mut currency = Currency::new(&mut links);
    let mut rows = Vec::new();
    for root in roots {
        for revision in currency
            .root_state(&root)
            .map_err(currency_error)?
            .revisions
        {
            rows.push((revision.revision, revision.current));
        }
    }
    drop(currency);
    for (revision, current) in rows {
        conn.execute(
            "INSERT OR REPLACE INTO temp.assertion_history(cutoff,id,root,current) VALUES(?1,?2,?3,?4)",
            params![cutoff, revision.rev, revision.root, current],
        )?;
    }
    Ok(())
}

impl Reader<'_> {
    pub fn assertion_family_currency(
        &self,
        root: &RevisionReference,
    ) -> Result<Option<FamilyState>> {
        if root.revision.is_some() {
            return Err(invalid("family currency requires an assertion root"));
        }
        let state = family(self.conn()?, &root.root)?;
        self.check()?;
        Ok(state)
    }
    pub fn assertion_revision_currency(
        &self,
        revision: &RevisionReference,
    ) -> Result<Option<RevisionState>> {
        let Some(id) = &revision.revision else {
            return Err(invalid("revision currency requires an exact assertion"));
        };
        let conn = self.conn()?;
        let source = Source::reading(conn)?;
        let state = conn
            .query_row(
                &source.sql("SELECT r.current,r.withdrawn,(SELECT count(*) FROM @source.revision_state h WHERE h.root=r.root AND h.current=1)>1 FROM @source.revision_state r WHERE r.id=?1 AND r.root=?2"),
                params![id, revision.root],
                |row| {
                    Ok(RevisionState {
                        reference: Reference::Assertion(revision.clone()),
                        current: row.get(0)?,
                        withdrawn: row.get(1)?,
                        conflict: row.get(2)?,
                    })
                },
            )
            .optional()?;
        self.check()?;
        Ok(state)
    }
}

#[cfg(test)]
mod tests;
