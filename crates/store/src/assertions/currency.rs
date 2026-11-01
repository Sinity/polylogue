//! A pinned reader's disposable SQL projection of the pure graph result.
//! The family row is published last; interrupted builds have no visible rows.
use super::{Graph, exact, graph_error, invalid};
use crate::{Reader, Result};
use polylogue_model::{
    Reference, RevisionReference,
    assertion::currency::{Currency, FamilyState, RevisionLinks, RevisionState},
};
use rusqlite::{Connection, OptionalExtension, params};

// These TEMP objects are created inside Reader's transaction. Its rollback
// removes them before the pooled connection can serve another snapshot.
const SCHEMA: &str = r#"
CREATE TEMP VIEW IF NOT EXISTS assertion_about AS
    SELECT assertion,ref FROM source_assertion_link WHERE role='about';
CREATE TEMP TABLE IF NOT EXISTS assertion_history (
    cutoff INTEGER NOT NULL,
    id TEXT NOT NULL,
    root TEXT NOT NULL,
    current INTEGER NOT NULL,
    PRIMARY KEY(cutoff,id)
) STRICT, WITHOUT ROWID;
CREATE TEMP TABLE IF NOT EXISTS assertion_family_state (
    root TEXT PRIMARY KEY,
    present INTEGER NOT NULL,
    withdrawn INTEGER NOT NULL,
    head_count INTEGER NOT NULL
) STRICT, WITHOUT ROWID;
CREATE TEMP TABLE IF NOT EXISTS assertion_revision_state (
    id TEXT PRIMARY KEY,
    root TEXT NOT NULL,
    current INTEGER NOT NULL,
    withdrawn INTEGER NOT NULL
) STRICT, WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS temp.assertion_revision_state_root
    ON assertion_revision_state(root,current,id);
CREATE TEMP VIEW IF NOT EXISTS assertion_currency AS
    SELECT r.id,r.root,r.current,r.withdrawn,(f.head_count>1) AS conflict
    FROM assertion_revision_state r JOIN assertion_family_state f ON f.root=r.root
    WHERE f.present=1;
"#;

/// Builds the currency of every assertion family in the pinned snapshot.
/// `Reader::conn` is its one caller, at the reader's first SQL access.
pub(crate) fn prepare_all(conn: &Connection) -> Result<()> {
    if conn.is_autocommit() {
        return Err(invalid(
            "assertion currency requires a pinned read transaction",
        ));
    }
    conn.execute_batch(SCHEMA)?;
    let source = crate::source::Source::reading(conn)?;
    let roots = conn
        .prepare_cached(&source.sql("SELECT DISTINCT root FROM @source.assertion ORDER BY root"))?
        .query_map([], |row| row.get::<_, String>(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    for root in roots {
        prepare_root(conn, &root)?;
    }
    Ok(())
}

/// Requires the currency that [`prepare_all`] built for this read.
pub(crate) fn prepare_history(conn: &Connection, cutoff: i64) -> Result<()> {
    let until = polylogue_model::Timestamp::from_unix_ms(cutoff)
        .ok_or_else(|| invalid("historical assertion cutoff is out of range"))?;
    let source = crate::source::Source::reading(conn)?;
    let roots = conn
        .prepare_cached(&source.sql("SELECT DISTINCT root FROM @source.assertion ORDER BY root"))?
        .query_map([], |row| row.get::<_, String>(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    for root in roots {
        let reference = Reference::Assertion(RevisionReference {
            root: root.clone(),
            revision: None,
        });
        let mut graph = Graph::before(conn, until)?;
        let revisions = graph.revisions(&reference)?;
        let mut currency = Currency::new(&mut graph);
        for revision in revisions {
            let current = currency.current(&revision).map_err(graph_error)?;
            let Reference::Assertion(revision) = revision else {
                return Err(invalid("assertion history contains another reference kind"));
            };
            let id = revision
                .revision
                .ok_or_else(|| invalid("assertion history revision is not exact"))?;
            conn.execute("INSERT OR REPLACE INTO temp.assertion_history(cutoff,id,root,current) VALUES(?1,?2,?3,?4)",params![cutoff,id,root,current])?;
        }
    }
    Ok(())
}

/// Populates `temp.assertion_currency` for one candidate root. The view exposes
/// id/root/current/withdrawn/conflict; every fork head has current=1. Candidate
/// selection belongs to the caller's existing indexed kind/about selector.
/// Only connections owned by a pinned Reader may call this function.
pub(crate) fn prepare_root(conn: &Connection, root: &str) -> Result<()> {
    if conn.is_autocommit() {
        return Err(invalid(
            "assertion currency requires a pinned read transaction",
        ));
    }
    conn.execute_batch(SCHEMA)?;
    if conn
        .query_row(
            "SELECT 1 FROM temp.assertion_family_state WHERE root=?1",
            [root],
            |_| Ok(()),
        )
        .optional()?
        .is_some()
    {
        return Ok(());
    }
    let reference = Reference::Assertion(RevisionReference {
        root: root.into(),
        revision: None,
    });
    let mut graph = Graph::new(conn)?;
    let revisions = graph.revisions(&reference)?;
    let mut currency = Currency::new(&mut graph);
    let withdrawn = currency.withdrawn(&reference).map_err(graph_error)?;
    let mut states = Vec::with_capacity(revisions.len());
    let mut head_count = 0;
    for revision in revisions {
        let Reference::Assertion(RevisionReference {
            revision: Some(id), ..
        }) = &revision
        else {
            return Err(invalid(
                "assertion family contains a non-assertion revision",
            ));
        };
        let current = currency.current(&revision).map_err(graph_error)?;
        let withdrawn = currency.withdrawn(&revision).map_err(graph_error)?;
        head_count += i64::from(current);
        states.push((id.clone(), current, withdrawn));
    }
    // A prior interrupted attempt may have inserted rows without its final
    // family marker. They are excluded by the view until this build completes.
    conn.execute(
        "DELETE FROM temp.assertion_revision_state WHERE root=?1",
        [root],
    )?;
    for (id, current, withdrawn) in &states {
        conn.execute("INSERT INTO temp.assertion_revision_state(id,root,current,withdrawn) VALUES(?1,?2,?3,?4)", params![id,root,current,withdrawn])?;
    }
    conn.execute("INSERT INTO temp.assertion_family_state(root,present,withdrawn,head_count) VALUES(?1,?2,?3,?4)", params![root,!states.is_empty(),withdrawn,head_count])?;
    Ok(())
}

fn family(conn: &Connection, root: &str) -> Result<Option<FamilyState>> {
    prepare_root(conn, root)?;
    let status = conn.query_row(
        "SELECT present,withdrawn FROM temp.assertion_family_state WHERE root=?1",
        [root],
        |row| Ok((row.get::<_, bool>(0)?, row.get::<_, bool>(1)?)),
    )?;
    if !status.0 {
        return Ok(None);
    }
    let mut statement = conn.prepare_cached(
        "SELECT id FROM temp.assertion_currency WHERE root=?1 AND current=1 ORDER BY id",
    )?;
    let heads = statement
        .query_map([root], |row| Ok(exact(root, &row.get::<_, String>(0)?)))?
        .collect::<rusqlite::Result<_>>()?;
    Ok(Some(FamilyState {
        root: Reference::Assertion(RevisionReference {
            root: root.into(),
            revision: None,
        }),
        withdrawn: status.1,
        heads,
    }))
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
        prepare_root(conn, &revision.root)?;
        let state = conn.query_row("SELECT current,withdrawn,conflict FROM temp.assertion_currency WHERE id=?1 AND root=?2",params![id,revision.root], |row| Ok(RevisionState{reference:Reference::Assertion(revision.clone()),current:row.get(0)?,withdrawn:row.get(1)?,conflict:row.get(2)?})).optional()?;
        self.check()?;
        Ok(state)
    }
}
