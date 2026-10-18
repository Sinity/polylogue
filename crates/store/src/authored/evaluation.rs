//! Works out which revisions of a note family are current and which are
//! withdrawn. The revises/withdraws graph is copied into a temporary on-disk
//! SQLite database one indexed lookup at a time, so memory stays small however
//! large the graph is. Note values are never copied.
use super::*;
use std::time::{Duration, Instant};

const STEPS: u64 = 1_000_000;
const SCRATCH_PAGES: u64 = 4096; // 16 MiB on disk; the page cache is 64 KiB.
const SCHEMA: &str = "
    CREATE TABLE node(
        id TEXT PRIMARY KEY, root TEXT, seq INTEGER, source_rowid INTEGER,
        expanded INTEGER NOT NULL DEFAULT 0,
        control_left INTEGER NOT NULL DEFAULT 0, control_done INTEGER NOT NULL DEFAULT 0,
        withdrawal_left INTEGER NOT NULL DEFAULT 0, withdrawn INTEGER NOT NULL DEFAULT 0,
        withdrawal_done INTEGER NOT NULL DEFAULT 0
    );
    CREATE INDEX node_expand ON node(expanded);
    CREATE INDEX node_control ON node(control_done, control_left);
    CREATE INDEX node_withdrawal ON node(withdrawal_done, withdrawal_left);
    CREATE INDEX node_root ON node(root, id);
    CREATE INDEX node_source ON node(source_rowid);
    CREATE TABLE family(root TEXT PRIMARY KEY, expanded INTEGER NOT NULL DEFAULT 0);
    CREATE INDEX family_expand ON family(expanded);
    CREATE TABLE edge(child TEXT, kind TEXT, parent TEXT, PRIMARY KEY(child, kind, parent));
    CREATE INDEX edge_parent ON edge(parent, kind, child);
";
/// The next revision of family `root` after `rowid`: its rowid, its ID (None
/// if longer than 128 bytes) and its `local_seq`.
pub(super) fn next_in_family(
    conn: &Connection,
    path: &Path,
    root: &str,
    rowid: i64,
) -> Result<Option<(i64, Option<String>, u64)>> {
    db(path, || {
        conn.query_row(
            "SELECT rowid, CASE WHEN length(CAST(id AS BLOB)) <= 128 THEN id END, local_seq \
             FROM note INDEXED BY note_root WHERE root = ?1 AND rowid > ?2 \
             ORDER BY rowid LIMIT 1",
            params![root, rowid],
            |r| Ok((r.get(0)?, r.get(1)?, crate::parser_state::unsigned(r, 2)?)),
        )
        .optional()
    })
}
fn scratch_error(e: rusqlite::Error) -> Error {
    unavailable(format!("note evaluation scratch database failed: {e}"))
}
pub(super) struct Evaluation<'a> {
    source: &'a Connection,
    path: &'a Path,
    scratch: Connection,
    _file: tempfile::NamedTempFile,
    cutoff: u64,
    until: Instant,
    cancel: &'a mut dyn FnMut() -> bool,
    steps: u64,
    #[cfg(test)]
    cache_peak: i32,
}
impl<'a> Evaluation<'a> {
    pub(super) fn new(
        source: &'a Connection,
        path: &'a Path,
        root: &str,
        cutoff: u64,
        until: Instant,
        cancel: &'a mut dyn FnMut() -> bool,
    ) -> Result<Self> {
        let (scratch, file) = workspace()?;
        scratch.execute_batch(SCHEMA).map_err(scratch_error)?;
        let mut this = Self {
            source,
            path,
            scratch,
            _file: file,
            cutoff,
            until,
            cancel,
            steps: 0,
            #[cfg(test)]
            cache_peak: 0,
        };
        this.sql("INSERT INTO family(root) VALUES(?1)", [root])?;
        this.build()?;
        this.evaluate()?;
        Ok(this)
    }
    pub(super) fn checkpoint(&mut self) -> Result<()> {
        self.check()
    }
    fn check(&mut self) -> Result<()> {
        if Instant::now() >= self.until || (self.cancel)() {
            return Err(Error::Cancelled);
        }
        // Bundled SQLite shares one page cache across connections, so
        // cache_size alone does not cap this one. Drop this database's cache
        // before every step; archive connections keep theirs.
        #[cfg(test)]
        {
            self.cache_peak = self.cache_peak.max(
                crate::native::cache_bytes(&self.scratch)
                    .map_err(|e| failure(ErrorKind::NotYetAvailable, e.to_string()))?,
            );
        }
        reset_cache(&self.scratch)?;
        self.steps += 1;
        if self.steps > STEPS {
            return Err(unavailable(
                "note evaluation took too many steps; read a pinned revision with view:value",
            ));
        }
        Ok(())
    }
    fn sql<P: rusqlite::Params>(&mut self, sql: &str, params: P) -> Result<()> {
        self.check()?;
        self.scratch
            .execute(sql, params)
            .map(|_| ())
            .map_err(scratch_error)
    }
    fn item(&mut self, table: &str, column: &str, filter: &str) -> Result<Option<String>> {
        self.check()?;
        self.scratch
            .query_row(
                &format!("SELECT {column} FROM {table} WHERE {filter} LIMIT 1"),
                [],
                |r| r.get(0),
            )
            .optional()
            .map_err(scratch_error)
    }
    /// Adds every revision of each unexpanded family and, for each new node,
    /// its family, the notes it revises or withdraws and the notes that
    /// withdraw it, until nothing new is found.
    fn build(&mut self) -> Result<()> {
        loop {
            if let Some(root) = self.item("family", "root", "expanded = 0")? {
                let mut after = 0i64;
                loop {
                    self.check()?;
                    let next = next_in_family(self.source, self.path, &root, after)?;
                    let Some((row, id, seq)) = next else { break };
                    after = row;
                    if seq > self.cutoff {
                        continue;
                    }
                    let id = id.ok_or_else(|| unavailable("note ID is too long"))?;
                    self.sql("INSERT OR IGNORE INTO node(id) VALUES(?1)", [id])?;
                }
                self.sql("UPDATE family SET expanded = 1 WHERE root = ?1", [root])?;
                continue;
            }
            let Some(id) = self.item("node", "id", "expanded = 0")? else {
                break;
            };
            self.check()?;
            let metadata: Option<(Option<String>, u64, bool, i64)> = db(self.path, || {
                self.source
                    .query_row(
                        "SELECT CASE WHEN length(CAST(root AS BLOB)) <= 128 THEN root END, \
                         local_seq, kind IN ('note', 'correction', 'withdrawal') \
                         AND key IS NULL AND check_json IS NULL AND valid_until IS NULL \
                         AND source IS NULL, rowid \
                         FROM note WHERE id = ?1 AND local_seq <= ?2",
                        params![id, self.cutoff as i64],
                        |r| {
                            Ok((
                                r.get(0)?,
                                crate::parser_state::unsigned(r, 1)?,
                                r.get(2)?,
                                r.get(3)?,
                            ))
                        },
                    )
                    .optional()
            })?;
            let (root, seq, supported, rowid) = metadata.ok_or_else(missing_parent)?;
            let root = root
                .filter(|r| authored::uuid_v7(r))
                .ok_or_else(|| unavailable("note root is not a UUIDv7"))?;
            if !authored::uuid_v7(&id) || !supported {
                return Err(unavailable("note has an unsupported kind or field"));
            }
            self.sql(
                "UPDATE node SET root = ?2, seq = ?3, source_rowid = ?4, expanded = 1 WHERE id = ?1",
                params![id, root, seq as i64, rowid],
            )?;
            self.sql("INSERT OR IGNORE INTO family(root) VALUES(?1)", [&root])?;
            self.sql("INSERT OR IGNORE INTO node(id) VALUES(?1)", [&root])?;
            // At most eight relations; their size is checked in SQL first.
            self.check()?;
            let mut s = db(self.path, || {
                self.source.prepare(
                    "SELECT \
                     CASE WHEN length(CAST(kind AS BLOB)) + length(CAST(other AS BLOB)) <= 17408 \
                     THEN kind END, \
                     CASE WHEN length(CAST(kind AS BLOB)) + length(CAST(other AS BLOB)) <= 17408 \
                     THEN other END \
                     FROM note_relation WHERE note_id = ?1 ORDER BY kind, other LIMIT 9",
                )
            })?;
            let rows = db(self.path, || {
                s.query_map([&id], |r| {
                    Ok((
                        r.get::<_, Option<String>>(0)?,
                        r.get::<_, Option<String>>(1)?,
                    ))
                })
            })?;
            let mut relations = Vec::new();
            for row in rows {
                self.check()?;
                if relations.len() == 8 {
                    return Err(unavailable("note has too many relations"));
                }
                let (kind, other) = db(self.path, || row)?;
                relations.push(
                    kind.zip(other)
                        .ok_or_else(|| unavailable("note relation is too long"))?,
                );
            }
            drop(s);
            for (kind, other) in relations {
                if !matches!(kind.as_str(), "revises" | "withdraws") {
                    continue;
                }
                let (parent_root, pin) = checked_reference(&other)?;
                let parent = pin.unwrap_or(parent_root);
                self.check()?;
                let exists: bool = db(self.path, || {
                    self.source.query_row(
                        "SELECT EXISTS(SELECT 1 FROM note \
                         WHERE id = ?1 AND root = ?2 AND local_seq <= ?3)",
                        params![parent, parent_root, self.cutoff as i64],
                        |r| r.get(0),
                    )
                })?;
                if !exists {
                    return Err(missing_parent());
                }
                self.sql("INSERT OR IGNORE INTO node(id) VALUES(?1)", [parent])?;
                self.sql(
                    "INSERT INTO edge(child, kind, parent) VALUES(?1, ?2, ?3)",
                    params![id, kind, parent],
                )?;
                self.sql(
                    "UPDATE node SET control_left = control_left + 1 WHERE id = ?1",
                    [&id],
                )?;
                if kind == "withdraws" {
                    self.sql(
                        "UPDATE node SET withdrawal_left = withdrawal_left + 1 WHERE id = ?1",
                        [parent],
                    )?;
                }
            }
            let reference = format!("note:{root}@{id}");
            let mut after = 0i64;
            loop {
                self.check()?;
                let next: Option<(i64, Option<String>, u64)> = db(self.path, || {
                    self.source
                        .query_row(
                            "SELECT r.rowid, \
                             CASE WHEN length(CAST(r.note_id AS BLOB)) <= 128 THEN r.note_id END, \
                             n.local_seq FROM note_relation r INDEXED BY note_relation_other \
                             JOIN note n ON n.id = r.note_id \
                             WHERE r.other = ?1 AND r.kind = 'withdraws' AND r.rowid > ?2 \
                             ORDER BY r.rowid LIMIT 1",
                            params![reference, after],
                            |r| Ok((r.get(0)?, r.get(1)?, crate::parser_state::unsigned(r, 2)?)),
                        )
                        .optional()
                })?;
                let Some((row, other, seq)) = next else { break };
                after = row;
                if seq > self.cutoff {
                    continue;
                }
                let other = other.ok_or_else(|| unavailable("note ID is too long"))?;
                self.sql("INSERT OR IGNORE INTO node(id) VALUES(?1)", [other])?;
            }
        }
        Ok(())
    }
    fn evaluate(&mut self) -> Result<()> {
        // Kahn's algorithm over the revises/withdraws graph rejects cycles.
        // Timestamps, local_seq and arrival order never pick a winner.
        while let Some(parent) = self.item("node", "id", "control_done = 0 AND control_left = 0")? {
            self.sql("UPDATE node SET control_done = 1 WHERE id = ?1", [&parent])?;
            let mut after = (String::new(), String::new());
            loop {
                self.check()?;
                let edge: Option<(String, String)> = self
                    .scratch
                    .query_row(
                        "SELECT kind, child FROM edge WHERE parent = ?1 \
                         AND (kind, child) > (?2, ?3) ORDER BY kind, child LIMIT 1",
                        params![parent, after.0, after.1],
                        |r| Ok((r.get(0)?, r.get(1)?)),
                    )
                    .optional()
                    .map_err(scratch_error)?;
                let Some((kind, child)) = edge else { break };
                self.sql(
                    "UPDATE node SET control_left = control_left - 1 WHERE id = ?1",
                    [&child],
                )?;
                after = (kind, child);
            }
        }
        if self.item("node", "id", "control_done = 0")?.is_some() {
            return Err(unavailable("note revisions or withdrawals form a cycle"));
        }
        // A withdrawal that is itself withdrawn no longer withdraws its parent.
        while let Some(child) =
            self.item("node", "id", "withdrawal_done = 0 AND withdrawal_left = 0")?
        {
            let withdrawn = self.withdrawn(&child)?;
            self.sql(
                "UPDATE node SET withdrawal_done = 1 WHERE id = ?1",
                [&child],
            )?;
            let mut after = String::new();
            loop {
                self.check()?;
                let parent: Option<String> = self
                    .scratch
                    .query_row(
                        "SELECT parent FROM edge WHERE child = ?1 AND kind = 'withdraws' \
                         AND parent > ?2 ORDER BY parent LIMIT 1",
                        params![child, after],
                        |r| r.get(0),
                    )
                    .optional()
                    .map_err(scratch_error)?;
                let Some(parent) = parent else { break };
                self.sql(
                    "UPDATE node SET withdrawal_left = withdrawal_left - 1, \
                     withdrawn = max(withdrawn, ?2) WHERE id = ?1",
                    params![parent, !withdrawn],
                )?;
                after = parent;
            }
        }
        if self.item("node", "id", "withdrawal_done = 0")?.is_some() {
            return Err(unavailable("withdrawals could not be fully evaluated"));
        }
        Ok(())
    }
    pub(super) fn withdrawn(&mut self, id: &str) -> Result<bool> {
        self.check()?;
        self.scratch
            .query_row("SELECT withdrawn FROM node WHERE id = ?1", [id], |r| {
                r.get(0)
            })
            .optional()
            .map_err(scratch_error)?
            .ok_or_else(|| failure(ErrorKind::NotFound, "note is not part of this evaluation"))
    }
    /// True unless the note or its family is withdrawn, or a revision of it
    /// in the same family is not withdrawn.
    pub(super) fn is_current(&mut self, id: &str, root: &str) -> Result<bool> {
        if self.withdrawn(root)? || self.withdrawn(id)? {
            return Ok(false);
        }
        let mut after = String::new();
        loop {
            self.check()?;
            let next: Option<(String, String, bool)> = self
                .scratch
                .query_row(
                    "SELECT n.id, n.root, n.withdrawn FROM edge e JOIN node n ON n.id = e.child \
                     WHERE e.parent = ?1 AND e.kind = 'revises' AND e.child > ?2 \
                     ORDER BY e.child LIMIT 1",
                    params![id, after],
                    |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
                )
                .optional()
                .map_err(scratch_error)?;
            let Some((other, other_root, withdrawn)) = next else {
                return Ok(true);
            };
            if other_root == root && !withdrawn {
                return Ok(false);
            }
            after = other;
        }
    }
    pub(super) fn heads(&mut self, root: &str) -> Result<Vec<String>> {
        let mut heads = Vec::new();
        let mut after = String::new();
        loop {
            self.check()?;
            let next: Option<String> = self
                .scratch
                .query_row(
                    "SELECT id FROM node WHERE root = ?1 AND id > ?2 ORDER BY id LIMIT 1",
                    params![root, after],
                    |r| r.get(0),
                )
                .optional()
                .map_err(scratch_error)?;
            let Some(id) = next else { return Ok(heads) };
            if self.is_current(&id, root)? {
                if heads.len() == 128 {
                    return Err(unavailable(
                        "more than 128 current revisions; read a pinned revision with view:value",
                    ));
                }
                heads.push(id.clone());
            }
            after = id;
        }
    }
    /// Computes the current revisions of every family in the graph, so that a
    /// write fails if any of them could not be read.
    pub(super) fn check_all_families(&mut self) -> Result<()> {
        let mut after = String::new();
        loop {
            self.check()?;
            let root: Option<String> = self
                .scratch
                .query_row(
                    "SELECT root FROM family WHERE root > ?1 ORDER BY root LIMIT 1",
                    [&after],
                    |r| r.get(0),
                )
                .optional()
                .map_err(scratch_error)?;
            let Some(root) = root else { return Ok(()) };
            self.heads(&root)?;
            after = root;
        }
    }
    /// Notes from other families (withdrawals) in the graph, in source rowid
    /// order after `after`; returns up to `limit + 1` so the caller can page.
    pub(super) fn trail(
        &mut self,
        root: &str,
        after: i64,
        limit: usize,
    ) -> Result<Vec<(i64, String)>> {
        let mut out = vec![];
        let mut pos = after;
        loop {
            self.check()?;
            let next: Option<(i64, String, String)> = self
                .scratch
                .query_row(
                    "SELECT source_rowid, id, root FROM node INDEXED BY node_source \
                     WHERE source_rowid > ?1 ORDER BY source_rowid LIMIT 1",
                    [pos],
                    |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
                )
                .optional()
                .map_err(scratch_error)?;
            let Some((row, id, other_root)) = next else {
                return Ok(out);
            };
            pos = row;
            if other_root != root {
                out.push((row, id));
                if out.len() > limit {
                    return Ok(out);
                }
            }
        }
    }
}
fn missing_parent() -> Error {
    unavailable("a revised or withdrawn note is missing at this archive counter")
}
pub(super) fn default_until() -> Instant {
    Instant::now() + Duration::from_secs(30)
}

/// A new temporary SQLite database in `$TMPDIR`, removed when the file drops.
/// It is never placed in the archive or a provider directory, and SQLite's
/// process-wide temp directory is left alone.
pub(super) fn workspace() -> Result<(Connection, tempfile::NamedTempFile)> {
    let file = tempfile::Builder::new()
        .prefix("polylogue-authored-")
        .tempfile_in(std::env::temp_dir())?;
    let scratch = Connection::open(file.path()).map_err(scratch_error)?;
    scratch
        .execute_batch(&format!(
            "PRAGMA page_size = 4096; PRAGMA journal_mode = OFF; PRAGMA synchronous = OFF; \
             PRAGMA cache_size = -64; PRAGMA max_page_count = {SCRATCH_PAGES};"
        ))
        .map_err(scratch_error)?;
    Ok((scratch, file))
}

#[cfg(test)]
mod tests;
