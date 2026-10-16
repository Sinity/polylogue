//! Request-local, disk-backed control evaluation. Payload bodies never enter
//! this database. Each source lookup is an indexed point/keyset step; no SQL
//! recursion, population aggregate or unbounded Rust frontier is used.
use super::*;
use std::time::{Duration, Instant};

const STEPS: u64 = 1_000_000;
const SCRATCH_PAGES: u64 = 4096; // 16 MiB on disk, with a 64 KiB SQLite page cache.
pub(super) struct Evaluation<'a> {
    source: &'a Connection,
    path: &'a Path,
    scratch: Connection,
    _file: tempfile::NamedTempFile,
    cutoff: u64,
    until: Instant,
    cancel: &'a mut dyn FnMut() -> bool,
    steps: u64,
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
        scratch.execute_batch("CREATE TABLE node(id TEXT PRIMARY KEY, root TEXT, seq INTEGER, source_rowid INTEGER, expanded INTEGER NOT NULL DEFAULT 0, control_left INTEGER NOT NULL DEFAULT 0, control_done INTEGER NOT NULL DEFAULT 0, withdrawal_left INTEGER NOT NULL DEFAULT 0, withdrawn INTEGER NOT NULL DEFAULT 0, withdrawal_done INTEGER NOT NULL DEFAULT 0);
            CREATE INDEX node_expand ON node(expanded);
            CREATE INDEX node_control ON node(control_done,control_left);
            CREATE INDEX node_withdrawal ON node(withdrawal_done,withdrawal_left);
            CREATE INDEX node_root ON node(root,id); CREATE INDEX node_source ON node(source_rowid);
            CREATE TABLE family(root TEXT PRIMARY KEY, expanded INTEGER NOT NULL DEFAULT 0);
            CREATE INDEX family_expand ON family(expanded);
            CREATE TABLE edge(child TEXT, kind TEXT, parent TEXT, PRIMARY KEY(child,kind,parent));
            CREATE INDEX edge_parent ON edge(parent,kind,child);")
            .map_err(|e|failure(ErrorKind::NotYetAvailable,e.to_string()))?;
        let mut this = Self {
            source,
            path,
            scratch,
            _file: file,
            cutoff,
            until,
            cancel,
            steps: 0,
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
        self.steps += 1;
        if self.steps > STEPS {
            return Err(failure(
                ErrorKind::NotYetAvailable,
                "authored evaluation step capacity exceeded; exact payload remains available without evaluated status",
            ));
        }
        Ok(())
    }
    fn sql<P: rusqlite::Params>(&mut self, sql: &str, params: P) -> Result<()> {
        self.check()?;
        self.scratch.execute(sql, params).map(|_| ()).map_err(|e| {
            failure(
                ErrorKind::NotYetAvailable,
                format!("authored evaluation scratch capacity/unavailability: {e}"),
            )
        })
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
            .map_err(|e| failure(ErrorKind::NotYetAvailable, e.to_string()))
    }
    fn build(&mut self) -> Result<()> {
        loop {
            if let Some(root) = self.item("family", "root", "expanded=0")? {
                let mut after = 0i64;
                loop {
                    self.check()?;
                    // note_root includes rowid as its unique tie key. This is a
                    // keyset seek, not ORDER BY id's per-family sort.
                    let next: Option<(i64, Option<String>, u64)> = db(self.path, || {
                        self.source.query_row(
                        "SELECT rowid,CASE WHEN length(CAST(id AS BLOB))<=128 THEN id END,local_seq FROM note INDEXED BY note_root WHERE root=?1 AND rowid>?2 ORDER BY rowid LIMIT 1",
                        params![root,after],|r|Ok((r.get(0)?,r.get(1)?,crate::parser_state::unsigned(r,2)?))).optional()
                    })?;
                    let Some((row, id, seq)) = next else { break };
                    after = row;
                    if seq > self.cutoff {
                        continue;
                    }
                    let id = id.ok_or_else(|| {
                        failure(
                            ErrorKind::NotYetAvailable,
                            "authored identity admission exceeded",
                        )
                    })?;
                    self.sql("INSERT OR IGNORE INTO node(id) VALUES(?1)", [id])?;
                }
                self.sql("UPDATE family SET expanded=1 WHERE root=?1", [root])?;
                continue;
            }
            let Some(id) = self.item("node", "id", "expanded=0")? else {
                break;
            };
            self.check()?;
            let metadata: Option<(Option<String>, u64, bool, i64)> = db(self.path, || {
                self.source.query_row(
                "SELECT CASE WHEN length(CAST(root AS BLOB))<=128 THEN root END,local_seq,kind IN ('note','correction','withdrawal') AND key IS NULL AND check_json IS NULL AND valid_until IS NULL AND source IS NULL,rowid FROM note WHERE id=?1 AND local_seq<=?2",
                params![id,self.cutoff as i64],|r|Ok((r.get(0)?,crate::parser_state::unsigned(r,1)?,r.get(2)?,r.get(3)?))).optional()
            })?;
            let (root, seq, supported, rowid) = metadata.ok_or_else(|| {
                failure(
                    ErrorKind::NotYetAvailable,
                    "authored control parent is unavailable at read basis",
                )
            })?;
            let root = root.filter(|r| authored::uuid_v7(r)).ok_or_else(|| {
                failure(
                    ErrorKind::NotYetAvailable,
                    "authored identity admission exceeded",
                )
            })?;
            if !authored::uuid_v7(&id) || !supported {
                return Err(failure(
                    ErrorKind::NotYetAvailable,
                    "unsupported authored control metadata",
                ));
            }
            self.sql(
                "UPDATE node SET root=?2,seq=?3,source_rowid=?4,expanded=1 WHERE id=?1",
                params![id, root, seq as i64, rowid],
            )?;
            self.sql("INSERT OR IGNORE INTO family(root) VALUES(?1)", [&root])?;
            self.sql("INSERT OR IGNORE INTO node(id) VALUES(?1)", [&root])?;
            // A revision has at most eight retained relations. Admission is
            // checked before allocating strings, independently of its body.
            self.check()?;
            let mut s = db(self.path, || {
                self.source.prepare("SELECT CASE WHEN length(CAST(kind AS BLOB))+length(CAST(other AS BLOB))<=17408 THEN kind END,CASE WHEN length(CAST(kind AS BLOB))+length(CAST(other AS BLOB))<=17408 THEN other END FROM note_relation WHERE note_id=?1 ORDER BY kind,other LIMIT 9")
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
                    return Err(failure(
                        ErrorKind::NotYetAvailable,
                        "authored relation admission exceeded",
                    ));
                }
                let (kind, other) = db(self.path, || row)?;
                relations.push((kind.zip(other)).ok_or_else(|| {
                    failure(
                        ErrorKind::NotYetAvailable,
                        "authored relation admission exceeded",
                    )
                })?);
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
                    self.source.query_row("SELECT EXISTS(SELECT 1 FROM note WHERE id=?1 AND root=?2 AND local_seq<=?3)",params![parent,parent_root,self.cutoff as i64],|r|r.get(0))
                })?;
                if !exists {
                    return Err(failure(
                        ErrorKind::NotYetAvailable,
                        "authored control parent is unavailable at read basis",
                    ));
                }
                self.sql("INSERT OR IGNORE INTO node(id) VALUES(?1)", [parent])?;
                self.sql(
                    "INSERT INTO edge(child,kind,parent) VALUES(?1,?2,?3)",
                    params![id, kind, parent],
                )?;
                self.sql(
                    "UPDATE node SET control_left=control_left+1 WHERE id=?1",
                    [&id],
                )?;
                if kind == "withdraws" {
                    self.sql(
                        "UPDATE node SET withdrawal_left=withdrawal_left+1 WHERE id=?1",
                        [parent],
                    )?;
                }
            }
            let reference = format!("note:{root}@{id}");
            let mut after = 0i64;
            loop {
                self.check()?;
                let next: Option<(i64, Option<String>, u64)> = db(self.path, || {
                    self.source.query_row(
                    "SELECT r.rowid,CASE WHEN length(CAST(r.note_id AS BLOB))<=128 THEN r.note_id END,n.local_seq FROM note_relation r INDEXED BY note_relation_other JOIN note n ON n.id=r.note_id WHERE r.other=?1 AND r.kind='withdraws' AND r.rowid>?2 ORDER BY r.rowid LIMIT 1",
                    params![reference,after],|r|Ok((r.get(0)?,r.get(1)?,crate::parser_state::unsigned(r,2)?))).optional()
                })?;
                let Some((row, other, seq)) = next else { break };
                after = row;
                if seq > self.cutoff {
                    continue;
                }
                let other = other.ok_or_else(|| {
                    failure(
                        ErrorKind::NotYetAvailable,
                        "authored identity admission exceeded",
                    )
                })?;
                self.sql("INSERT OR IGNORE INTO node(id) VALUES(?1)", [other])?;
            }
        }
        Ok(())
    }
    fn evaluate(&mut self) -> Result<()> {
        // Kahn elimination validates the combined revises/withdraws DAG. It
        // does not use timestamps, local_seq or arrival order as a winner.
        while let Some(parent) = self.item("node", "id", "control_done=0 AND control_left=0")? {
            self.sql("UPDATE node SET control_done=1 WHERE id=?1", [&parent])?;
            let mut after = (String::new(), String::new());
            loop {
                self.check()?;
                let edge:Option<(String,String)>=self.scratch.query_row("SELECT kind,child FROM edge WHERE parent=?1 AND (kind,child)>(?2,?3) ORDER BY kind,child LIMIT 1",params![parent,after.0,after.1],|r|Ok((r.get(0)?,r.get(1)?))).optional().map_err(|e|failure(ErrorKind::NotYetAvailable,e.to_string()))?;
                let Some((kind, child)) = edge else { break };
                self.sql(
                    "UPDATE node SET control_left=control_left-1 WHERE id=?1",
                    [&child],
                )?;
                after = (kind, child);
            }
        }
        if self.item("node", "id", "control_done=0")?.is_some() {
            return Err(failure(
                ErrorKind::NotYetAvailable,
                "cyclic authored revision/withdrawal controls",
            ));
        }
        while let Some(child) =
            self.item("node", "id", "withdrawal_done=0 AND withdrawal_left=0")?
        {
            let withdrawn = self.withdrawn(&child)?;
            self.sql("UPDATE node SET withdrawal_done=1 WHERE id=?1", [&child])?;
            let mut after = String::new();
            loop {
                self.check()?;
                let parent:Option<String>=self.scratch.query_row("SELECT parent FROM edge WHERE child=?1 AND kind='withdraws' AND parent>?2 ORDER BY parent LIMIT 1",params![child,after],|r|r.get(0)).optional().map_err(|e|failure(ErrorKind::NotYetAvailable,e.to_string()))?;
                let Some(parent) = parent else { break };
                self.sql("UPDATE node SET withdrawal_left=withdrawal_left-1,withdrawn=max(withdrawn,?2) WHERE id=?1",params![parent,!withdrawn])?;
                after = parent;
            }
        }
        if self.item("node", "id", "withdrawal_done=0")?.is_some() {
            return Err(failure(
                ErrorKind::NotYetAvailable,
                "incomplete authored withdrawal evaluation",
            ));
        }
        Ok(())
    }
    pub(super) fn withdrawn(&mut self, id: &str) -> Result<bool> {
        self.check()?;
        self.scratch
            .query_row("SELECT withdrawn FROM node WHERE id=?1", [id], |r| r.get(0))
            .optional()
            .map_err(|e| failure(ErrorKind::NotYetAvailable, e.to_string()))?
            .ok_or_else(|| failure(ErrorKind::NotFound, "note is absent from evaluated basis"))
    }
    pub(super) fn is_current(&mut self, id: &str, root: &str) -> Result<bool> {
        if self.withdrawn(root)? || self.withdrawn(id)? {
            return Ok(false);
        }
        let mut after = String::new();
        loop {
            self.check()?;
            let next:Option<(String,String,bool)>=self.scratch.query_row("SELECT n.id,n.root,n.withdrawn FROM edge e JOIN node n ON n.id=e.child WHERE e.parent=?1 AND e.kind='revises' AND e.child>?2 ORDER BY e.child LIMIT 1",params![id,after],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?))).optional().map_err(|e|failure(ErrorKind::NotYetAvailable,e.to_string()))?;
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
                    "SELECT id FROM node WHERE root=?1 AND id>?2 ORDER BY id LIMIT 1",
                    params![root, after],
                    |r| r.get(0),
                )
                .optional()
                .map_err(|e| failure(ErrorKind::NotYetAvailable, e.to_string()))?;
            let Some(id) = next else { return Ok(heads) };
            if self.is_current(&id, root)? {
                if heads.len() == 128 {
                    return Err(failure(
                        ErrorKind::NotYetAvailable,
                        "current authored conflict response capacity exceeded; use a pinned revision with view:value",
                    ));
                }
                heads.push(id.clone());
            }
            after = id;
        }
    }
    pub(super) fn qualify_all_families(&mut self) -> Result<()> {
        let mut after = String::new();
        loop {
            self.check()?;
            let root: Option<String> = self
                .scratch
                .query_row(
                    "SELECT root FROM family WHERE root>?1 ORDER BY root LIMIT 1",
                    [&after],
                    |r| r.get(0),
                )
                .optional()
                .map_err(|e| failure(ErrorKind::NotYetAvailable, e.to_string()))?;
            let Some(root) = root else { return Ok(()) };
            self.heads(&root)?;
            after = root;
        }
    }
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
            let next:Option<(i64,String,String)>=self.scratch.query_row("SELECT source_rowid,id,root FROM node INDEXED BY node_source WHERE source_rowid>?1 ORDER BY source_rowid LIMIT 1",[pos],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?))).optional().map_err(|e|failure(ErrorKind::NotYetAvailable,e.to_string()))?;
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
pub(super) fn default_until() -> Instant {
    Instant::now() + Duration::from_secs(30)
}

pub(super) fn workspace() -> Result<(Connection, tempfile::NamedTempFile)> {
    // TMPDIR is the existing process-owned scratch contract (managed jobs
    // place it on /realm). Never use archive authority or provider paths as
    // scratch, and never change SQLite's process-global temp directory.
    let file = tempfile::Builder::new()
        .prefix("polylogue-authored-")
        .tempfile_in(std::env::temp_dir())?;
    let scratch = Connection::open(file.path()).map_err(|e| {
        failure(
            ErrorKind::NotYetAvailable,
            format!("authored evaluation scratch unavailable: {e}"),
        )
    })?;
    scratch.execute_batch(&format!("PRAGMA page_size=4096; PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; PRAGMA cache_size=-64; PRAGMA max_page_count={SCRATCH_PAGES};"))
        .map_err(|e|failure(ErrorKind::NotYetAvailable,format!("authored evaluation scratch unavailable: {e}")))?;
    Ok((scratch, file))
}

#[cfg(test)]
mod tests {
    use super::*;
    use polylogue_model::authored::GraphNode;
    use std::collections::BTreeMap;
    fn id(n: u32) -> String {
        format!("01900000-0000-7000-8000-{n:012}")
    }
    #[test]
    fn indexed_disk_evaluation_matches_the_published_pure_dag_oracle() {
        for seed in 0..32u32 {
            let dir = tempfile::tempdir().unwrap();
            let writer = Writer::open(dir.path()).unwrap();
            let path = dir.path().join("archive.db");
            let mut graph: BTreeMap<String, GraphNode> = BTreeMap::new();
            let mut random = seed + 1;
            writer.archive.execute_batch("BEGIN IMMEDIATE").unwrap();
            let mut family = vec![];
            for n in 1..=12 {
                random = random.wrapping_mul(1664525).wrapping_add(1013904223);
                let withdrawal = n > 1 && random % 3 == 0;
                let root = if withdrawal { id(n) } else { id(1) };
                let parent = if withdrawal {
                    Some(("withdraws", 1 + random % (n - 1)))
                } else if n > 1 {
                    Some(("revises", family[random as usize % family.len()]))
                } else {
                    None
                };
                writer.archive.execute("INSERT INTO note(id,root,kind,value,author,machine,created_at,local_seq) VALUES(?1,?2,?3,'payload is irrelevant to topology','agent','machine',?4,?5)",params![id(n),root,if withdrawal {"withdrawal"}else{"note"},1000-n,n]).unwrap();
                let mut relations = vec![];
                if let Some((kind, parent)) = parent {
                    let other_root = &graph[&id(parent)].root;
                    writer
                        .archive
                        .execute(
                            "INSERT INTO note_relation(note_id,kind,other) VALUES(?1,?2,?3)",
                            params![id(n), kind, format!("note:{other_root}@{}", id(parent))],
                        )
                        .unwrap();
                    relations.push(Relation {
                        kind: kind.into(),
                        other: id(parent),
                    });
                }
                graph.insert(
                    id(n),
                    GraphNode {
                        root: root.clone(),
                        relations,
                    },
                );
                if !withdrawal {
                    family.push(n);
                }
            }
            writer.archive.execute_batch("COMMIT").unwrap();
            for root in graph
                .values()
                .map(|n| n.root.clone())
                .collect::<std::collections::BTreeSet<_>>()
            {
                let mut cancel = || false;
                let mut actual = Evaluation::new(
                    &writer.archive,
                    &path,
                    &root,
                    12,
                    default_until(),
                    &mut cancel,
                )
                .unwrap();
                assert_eq!(
                    actual.heads(&root).unwrap(),
                    authored::current(&root, &graph).unwrap(),
                    "seed {seed} root {root}"
                );
                for (node, n) in &graph {
                    // Unrelated families need not enter this request's proof.
                    if actual
                        .scratch
                        .query_row(
                            "SELECT EXISTS(SELECT 1 FROM node WHERE id=?1)",
                            [node],
                            |r| r.get::<_, bool>(0),
                        )
                        .unwrap()
                    {
                        assert_eq!(
                            actual.withdrawn(node).unwrap(),
                            authored::withdrawn(node, &graph).unwrap(),
                            "seed {seed} node {node}"
                        );
                        assert_eq!(
                            actual.is_current(node, &n.root).unwrap(),
                            authored::current(&n.root, &graph).unwrap().contains(node)
                        );
                    }
                }
            }
        }
    }
    #[test]
    fn growing_topology_spills_to_owned_disk_and_cancelled_work_cleans_up() {
        let dir = tempfile::tempdir().unwrap();
        let writer = Writer::open(dir.path()).unwrap();
        let path = dir.path().join("archive.db");
        let root = id(1);
        writer.archive.execute_batch("BEGIN IMMEDIATE").unwrap();
        for n in 1..=1000 {
            writer.archive.execute("INSERT INTO note(id,root,kind,value,author,machine,created_at,local_seq) VALUES(?1,?2,'note','body','agent','machine',0,?3)",params![id(n),root,n]).unwrap();
            if n > 1 {
                writer
                    .archive
                    .execute(
                        "INSERT INTO note_relation(note_id,kind,other) VALUES(?1,'revises',?2)",
                        params![id(n), format!("note:{root}@{}", id(n - 1))],
                    )
                    .unwrap();
            }
        }
        writer.archive.execute_batch("COMMIT").unwrap();
        let mut cancel = || false;
        let mut eval = Evaluation::new(
            &writer.archive,
            &path,
            &root,
            1000,
            default_until(),
            &mut cancel,
        )
        .unwrap();
        assert_eq!(eval.heads(&root).unwrap(), [id(1000)]);
        assert_eq!(
            eval.scratch
                .query_row("PRAGMA cache_size", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            -64
        );
        let scratch = eval._file.path().to_owned();
        let bytes = std::fs::metadata(&scratch).unwrap().len();
        assert!(bytes > 65536 && bytes <= SCRATCH_PAGES * 4096, "{bytes}");
        let cache = crate::native::cache_bytes(&eval.scratch).unwrap();
        assert!(
            cache > 0 && cache <= 128 * 1024 && (cache as u64) < bytes,
            "cache={cache} file={bytes}"
        );
        eprintln!(
            "authored scratch: file_bytes={bytes}, cache_bytes={cache}, configured_cache_kib=64, disk_limit_bytes={}",
            SCRATCH_PAGES * 4096
        );
        assert_eq!(scratch.parent(), Some(std::env::temp_dir().as_path()));
        eval.until = Instant::now();
        assert!(matches!(eval.checkpoint(), Err(Error::Cancelled)));
        drop(eval);
        assert!(!scratch.exists());
        // All source queries used for growing populations are index seeks and
        // require no hidden ORDER BY/DISTINCT temporary population operation.
        for sql in [
            "SELECT rowid,id FROM note INDEXED BY note_root WHERE root='r' AND rowid>0 ORDER BY rowid LIMIT 1",
            "SELECT r.rowid,r.note_id,n.local_seq FROM note_relation r INDEXED BY note_relation_other JOIN note n ON n.id=r.note_id WHERE r.other='r' AND r.kind='withdraws' AND r.rowid>0 ORDER BY r.rowid LIMIT 1",
            "SELECT t.rowid,n.root,n.local_seq FROM note_target t INDEXED BY note_target_ref JOIN note n ON n.id=t.note_id WHERE t.ref='t' AND t.rowid>0 ORDER BY t.rowid LIMIT 1",
        ] {
            let plan = writer
                .archive
                .prepare(&format!("EXPLAIN QUERY PLAN {sql}"))
                .unwrap()
                .query_map([], |r| r.get::<_, String>(3))
                .unwrap()
                .collect::<rusqlite::Result<Vec<_>>>()
                .unwrap();
            assert!(
                plan.iter()
                    .all(|p| !p.contains("SCAN") && !p.contains("TEMP B-TREE")),
                "{sql}: {plan:?}"
            );
        }
    }
}
