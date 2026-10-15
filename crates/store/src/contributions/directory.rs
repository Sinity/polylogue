//! Immutable Message-local radix16 directories. A mutation visits at most16
//! fixed-size nodes. Canonical publication/leases are owned by the caller's
//! transaction; this module never selects a Message or marks parsing complete.
use crate::{Error, Result, db};
use rusqlite::{Connection, OptionalExtension, Transaction, params};
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    Groups,
    Blocks,
    Fields,
    Operations,
}
impl Kind {
    fn name(self) -> &'static str {
        match self {
            Self::Groups => "groups",
            Self::Blocks => "blocks",
            Self::Fields => "fields",
            Self::Operations => "operations",
        }
    }
    fn parse(s: &str) -> Result<Self> {
        match s {
            "groups" => Ok(Self::Groups),
            "blocks" => Ok(Self::Blocks),
            "fields" => Ok(Self::Fields),
            "operations" => Ok(Self::Operations),
            _ => Err(bad("unknown directory kind")),
        }
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Counts {
    pub ordinary: i64,
    pub reasoning: i64,
    pub source_blocks: i64,
    pub source_text_blocks: i64,
    pub source_fields: i64,
}
impl Counts {
    fn add(self, rhs: Self) -> Result<Self> {
        let plus = |a: i64, b: i64| {
            a.checked_add(b)
                .ok_or_else(|| bad("directory count overflow"))
        };
        Ok(Self {
            ordinary: plus(self.ordinary, rhs.ordinary)?,
            reasoning: plus(self.reasoning, rhs.reasoning)?,
            source_blocks: plus(self.source_blocks, rhs.source_blocks)?,
            source_text_blocks: plus(self.source_text_blocks, rhs.source_text_blocks)?,
            source_fields: plus(self.source_fields, rhs.source_fields)?,
        })
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Owner {
    pub message: i64,
    pub session: i64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Link {
    Node { pk: i64, kind: Kind, height: u8 },
    Block { pk: i64, call: Option<i64> },
    Field(i64),
    Operation(i64),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Edge {
    pub link: Link,
    pub counts: Counts,
}
#[derive(Clone, Copy)]
struct Node {
    height: u8,
    edges: [Option<Edge>; 16],
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Work {
    pub nodes_read: u32,
    pub edges_read: u32,
    pub nodes_written: u32,
    pub edges_written: u32,
    pub references_changed: u32,
}
fn bad(s: &str) -> Error {
    Error::Invariant(s.into())
}
fn counts(row: &rusqlite::Row<'_>, i: usize) -> rusqlite::Result<Counts> {
    Ok(Counts {
        ordinary: row.get(i)?,
        reasoning: row.get(i + 1)?,
        source_blocks: row.get(i + 2)?,
        source_text_blocks: row.get(i + 3)?,
        source_fields: row.get(i + 4)?,
    })
}
fn load(
    dbh: &Connection,
    path: &Path,
    owner: Owner,
    kind: Kind,
    pk: i64,
    work: &mut Work,
) -> Result<Node> {
    let (height,summary):(u8,Counts)=db(path,||dbh.query_row("SELECT height,ordinary_count,reasoning_count,source_blocks,source_text_blocks,source_fields FROM message_body_node WHERE pk=?1 AND message_pk=?2 AND session_pk=?3 AND kind=?4 AND state='ready'",params![pk,owner.message,owner.session,kind.name()],|r|Ok((r.get(0)?,counts(r,1)?))).optional())?.ok_or_else(||bad("missing or unready directory node"))?;
    work.nodes_read += 1;
    let mut node = Node {
        height,
        edges: [None; 16],
    };
    let mut stmt = db(path, || {
        dbh.prepare("SELECT slot,child_node_pk,child_kind,child_height,block_pk,field_fact_pk,tool_call_pk,association_pk,ordinary_count,reasoning_count,source_blocks,source_text_blocks,source_fields FROM message_body_edge WHERE parent_pk=?1 ORDER BY slot")
    })?;
    let mut rows = db(path, || stmt.query([pk]))?;
    let mut total = Counts::default();
    while let Some(row) = db(path, || rows.next())? {
        let slot = usize::from(db(path, || row.get::<_, u8>(0))?);
        if slot >= 16 || node.edges[slot].is_some() {
            return Err(bad("directory edge slot out of range or duplicated"));
        }
        let child: Option<i64> = db(path, || row.get(1))?;
        let block: Option<i64> = db(path, || row.get(4))?;
        let field: Option<i64> = db(path, || row.get(5))?;
        let call: Option<i64> = db(path, || row.get(6))?;
        let association: Option<i64> = db(path, || row.get(7))?;
        let link = if let Some(child) = child {
            let k: String = db(path, || row.get(2))?;
            let h: u8 = db(path, || row.get(3))?;
            Link::Node {
                pk: child,
                kind: Kind::parse(&k)?,
                height: h,
            }
        } else if let Some(pk) = block {
            Link::Block { pk, call }
        } else if let Some(pk) = field {
            Link::Field(pk)
        } else if let Some(pk) = association {
            Link::Operation(pk)
        } else {
            return Err(bad("directory leaf has no payload"));
        };
        let edge = Edge {
            link,
            counts: db(path, || counts(row, 8))?,
        };
        total = total.add(edge.counts)?;
        node.edges[slot] = Some(edge);
        work.edges_read += 1;
    }
    if total != summary {
        return Err(bad("directory count cache disagrees with immutable edges"));
    }
    Ok(node)
}
fn insert(
    tx: &Transaction<'_>,
    path: &Path,
    owner: Owner,
    kind: Kind,
    height: u8,
    edges: [Option<Edge>; 16],
    work: &mut Work,
) -> Result<(i64, Counts)> {
    let mut total = Counts::default();
    for edge in edges.iter().flatten() {
        total = total.add(edge.counts)?;
    }
    db(path, || {
        tx.execute("INSERT INTO message_body_node(message_pk,session_pk,kind,height,state,incoming,ordinary_count,reasoning_count,source_blocks,source_text_blocks,source_fields) VALUES(?1,?2,?3,?4,'building',0,?5,?6,?7,?8,?9)",params![owner.message,owner.session,kind.name(),height,total.ordinary,total.reasoning,total.source_blocks,total.source_text_blocks,total.source_fields])
    })?;
    let pk = tx.last_insert_rowid();
    work.nodes_written += 1;
    for (slot, edge) in edges.into_iter().enumerate() {
        let Some(edge) = edge else { continue };
        let (child, k, h, block, field, call, association) = match edge.link {
            Link::Node { pk, kind, height } => (
                Some(pk),
                Some(kind.name()),
                Some(height),
                None,
                None,
                None,
                None,
            ),
            Link::Block { pk, call } => (None, None, None, Some(pk), None, call, None),
            Link::Field(pk) => (None, None, None, None, Some(pk), None, None),
            Link::Operation(pk) => (None, None, None, None, None, None, Some(pk)),
        };
        if let Some(child) = child {
            if db(path, || {
                tx.execute("UPDATE message_body_node SET incoming=incoming+1 WHERE pk=?1 AND message_pk=?2 AND session_pk=?3 AND state='ready'",params![child,owner.message,owner.session])
            })? != 1
            {
                return Err(bad("cannot acquire missing/retiring directory child"));
            }
            work.references_changed += 1;
        }
        db(path, || {
            tx.execute("INSERT INTO message_body_edge(parent_pk,message_pk,session_pk,parent_kind,parent_height,slot,child_node_pk,child_kind,child_height,block_pk,field_fact_pk,tool_call_pk,association_pk,ordinary_count,reasoning_count,source_blocks,source_text_blocks,source_fields) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18)",params![pk,owner.message,owner.session,kind.name(),height,slot as i64,child,k,h,block,field,call,association,edge.counts.ordinary,edge.counts.reasoning,edge.counts.source_blocks,edge.counts.source_text_blocks,edge.counts.source_fields])
        })?;
        work.edges_written += 1;
    }
    db(path, || {
        tx.execute(
            "UPDATE message_body_node SET state='ready' WHERE pk=?1",
            [pk],
        )
    })?;
    Ok((pk, total))
}
/// Caller must durably acquire the returned root before committing. No token
/// escapes a failed transaction. Cloned unchanged edges share their children.
pub(super) fn replace(
    tx: &Transaction<'_>,
    path: &Path,
    owner: Owner,
    kind: Kind,
    root: Option<i64>,
    key: u64,
    leaf: Option<Edge>,
    work: &mut Work,
) -> Result<(i64, Counts)> {
    let mut nodes: [Option<Node>; 16] = [None; 16];
    let mut current = root;
    for (depth, entry) in nodes.iter_mut().enumerate() {
        let height = 15 - depth as u8;
        if let Some(pk) = current {
            let node = load(tx, path, owner, kind, pk, work)?;
            if node.height != height {
                return Err(bad("directory root/path height mismatch"));
            }
            let slot = ((key >> (height * 4)) & 15) as usize;
            current = if height == 0 {
                None
            } else {
                match node.edges[slot].map(|e| e.link) {
                    Some(Link::Node {
                        pk,
                        kind: child_kind,
                        height: child_height,
                    }) if child_kind == kind && child_height == height - 1 => Some(pk),
                    None => None,
                    _ => return Err(bad("directory internal edge shape")),
                }
            };
            *entry = Some(node);
        }
    }
    let mut next = leaf;
    let mut result = None;
    for depth in (0..16).rev() {
        let height = 15 - depth as u8;
        let mut edges = nodes[depth].map(|n| n.edges).unwrap_or([None; 16]);
        edges[((key >> (height * 4)) & 15) as usize] = next;
        let (pk, summary) = insert(tx, path, owner, kind, height, edges, work)?;
        next = Some(Edge {
            link: Link::Node { pk, kind, height },
            counts: summary,
        });
        result = Some((pk, summary));
    }
    result.ok_or_else(|| bad("empty fixed directory path"))
}
/// A lease is an owning row/root reference, not a worker-local memory handle.
pub(super) fn acquire(tx: &Transaction<'_>, path: &Path, owner: Owner, pk: i64) -> Result<()> {
    if db(path, || {
        tx.execute("UPDATE message_body_node SET incoming=incoming+1 WHERE pk=?1 AND message_pk=?2 AND session_pk=?3 AND state='ready'",params![pk,owner.message,owner.session])
    })? != 1
    {
        return Err(bad("directory root cannot be acquired"));
    }
    Ok(())
}
pub(super) fn release(tx: &Transaction<'_>, path: &Path, owner: Owner, pk: i64) -> Result<()> {
    if db(path, || {
        tx.execute("UPDATE message_body_node SET incoming=incoming-1 WHERE pk=?1 AND message_pk=?2 AND session_pk=?3 AND state='ready' AND incoming>0",params![pk,owner.message,owner.session])
    })? != 1
    {
        return Err(bad("directory root lease missing"));
    }
    Ok(())
}
/// Claims and drains ONE zero-reference node; <=16 outgoing references. The
/// transaction also removes the drained edges and node, so crash cannot double
/// decrement. Recursive descendants remain separate admitted work.
pub(super) fn collect(tx: &Transaction<'_>, path: &Path, pk: i64) -> Result<bool> {
    if db(path, || {
        tx.execute("UPDATE message_body_node SET state='retiring' WHERE pk=?1 AND incoming=0 AND state IN('building','ready')",[pk])
    })? == 0
    {
        return Ok(false);
    }
    let mut children = [None; 16];
    let mut stmt = db(path, || {
        tx.prepare(
            "SELECT slot,child_node_pk FROM message_body_edge WHERE parent_pk=?1 ORDER BY slot",
        )
    })?;
    let mut rows = db(path, || stmt.query([pk]))?;
    while let Some(row) = db(path, || rows.next())? {
        let slot = usize::from(db(path, || row.get::<_, u8>(0))?);
        if slot >= 16 {
            return Err(bad("retirement edge slot"));
        }
        children[slot] = db(path, || row.get::<_, Option<i64>>(1))?;
    }
    drop(rows);
    drop(stmt);
    db(path, || {
        tx.execute("DELETE FROM message_body_edge WHERE parent_pk=?1", [pk])
    })?;
    for child in children.into_iter().flatten() {
        if db(path, || {
            tx.execute("UPDATE message_body_node SET incoming=incoming-1 WHERE pk=?1 AND incoming>0 AND state='ready'",[child])
        })? != 1
        {
            return Err(bad("retiring directory child lacks counted reference"));
        }
    }
    db(path, || {
        tx.execute(
            "DELETE FROM message_body_node WHERE pk=?1 AND incoming=0 AND state='retiring'",
            [pk],
        )
    })?;
    Ok(true)
}

/// One keyed membership walk, never a recursive population scan.
pub(super) fn lookup(
    dbh: &Connection,
    path: &Path,
    owner: Owner,
    kind: Kind,
    root: Option<i64>,
    key: u64,
    work: &mut Work,
) -> Result<Option<Edge>> {
    let Some(mut pk) = root else { return Ok(None) };
    for height in (0..16u8).rev() {
        let node = load(dbh, path, owner, kind, pk, work)?;
        if node.height != height {
            return Err(bad("directory lookup height"));
        }
        let edge = node.edges[((key >> (height * 4)) & 15) as usize];
        if height == 0 {
            return Ok(edge);
        }
        match edge.map(|e| e.link) {
            Some(Link::Node {
                pk: child,
                kind: child_kind,
                height: child_height,
            }) if child_kind == kind && child_height == height - 1 => pk = child,
            None => return Ok(None),
            _ => return Err(bad("directory lookup shape")),
        }
    }
    Err(bad("directory lookup escaped fixed height"))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> (tempfile::TempDir, crate::Writer, Owner, i64) {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("archive");
        let writer = crate::Writer::open(&root).unwrap();
        writer
            ._index
            .execute("INSERT INTO input_ref(pk,id) VALUES(1,'input')", [])
            .unwrap();
        writer._index.execute("INSERT INTO session(pk,id,origin,native_id,kind,changed) VALUES(1,'opaque-session','test','s','main',0)",[]).unwrap();
        writer
            ._index
            .execute(
                "INSERT INTO message(pk,id,session_pk,seq) VALUES(1,'opaque-message',1,0)",
                [],
            )
            .unwrap();
        writer._index.execute("INSERT INTO message_body(message_pk,session_pk,basis_digest,contributor_revision,removal_epoch,layout,role,written_by,input_pk,parser_version) VALUES(1,1,zeroblob(32),0,0,'flat','assistant','model',1,1)",[]).unwrap();
        let body = writer._index.last_insert_rowid();
        (
            dir,
            writer,
            Owner {
                message: 1,
                session: 1,
            },
            body,
        )
    }
    fn leaf(tx: &Transaction<'_>, ordinal: u64) -> Edge {
        tx.execute("INSERT INTO block(message_pk,session_pk,group_ordinal,local_ordinal,slot_kind,kind) VALUES(1,1,0,?1,'ordinary','text')",[i64::try_from(ordinal).unwrap()]).unwrap();
        Edge {
            link: Link::Block {
                pk: tx.last_insert_rowid(),
                call: None,
            },
            counts: Counts {
                ordinary: 1,
                ..Counts::default()
            },
        }
    }
    fn install(
        tx: &Transaction<'_>,
        file: &Path,
        owner: Owner,
        body: i64,
        root: Option<i64>,
        key: u64,
        leaf: Edge,
    ) -> (i64, Work) {
        let mut work = Work::default();
        let (new, _) = replace(
            tx,
            file,
            owner,
            Kind::Blocks,
            root,
            key,
            Some(leaf),
            &mut work,
        )
        .unwrap();
        acquire(tx, file, owner, new).unwrap();
        // This fixture uses a local block root as a build-frontier lease. Production
        // body group roots use the same acquire/update/release transaction.
        tx.execute("INSERT INTO message_build(message_pk,session_pk,body_pk,basis_digest,contributor_revision,removal_epoch,phase) VALUES(1,1,?1,zeroblob(32),0,0,'layout') ON CONFLICT(message_pk) DO NOTHING",[body]).unwrap();
        let build: i64 = tx
            .query_row("SELECT pk FROM message_build WHERE message_pk=1", [], |r| {
                r.get(0)
            })
            .unwrap();
        tx.execute("INSERT INTO message_build_frontier(build_pk,message_pk,session_pk,depth,node_pk,next_slot) VALUES(?1,1,1,0,?2,0) ON CONFLICT(build_pk,depth) DO UPDATE SET node_pk=excluded.node_pk",params![build,new]).unwrap();
        if let Some(root) = root {
            release(tx, file, owner, root).unwrap();
        }
        (new, work)
    }
    #[test]
    fn one_replacement_copies_fixed_path_and_old_snapshot_survives_retirement() {
        let (_dir, mut writer, owner, body) = fixture();
        let file = writer.root.join("index.db");
        let mut root = None;
        for i in 0..128u64 {
            let tx = writer._index.transaction().unwrap();
            let item = leaf(&tx, i);
            let (next, work) = install(&tx, &file, owner, body, root, i, item);
            assert_eq!(work.nodes_written, 16);
            assert!(work.edges_written <= 256);
            tx.commit().unwrap();
            root = Some(next);
        }
        let old = root.unwrap();
        let reader = writer.read_handle().unwrap();
        reader.index.execute_batch("BEGIN").unwrap();
        let mut work = Work::default();
        let before = lookup(
            &reader.index,
            &file,
            owner,
            Kind::Blocks,
            Some(old),
            0,
            &mut work,
        )
        .unwrap()
        .unwrap();
        assert!(work.nodes_read <= 16 && work.edges_read <= 256);
        let tx = writer._index.transaction().unwrap();
        let item = leaf(&tx, 999);
        let (new, work) = install(&tx, &file, owner, body, root, 0, item);
        assert_eq!(work.nodes_read, 16);
        assert_eq!(work.nodes_written, 16);
        assert!(work.edges_written <= 256);
        assert!(work.references_changed <= 256);
        tx.commit().unwrap();
        let tx = writer._index.transaction().unwrap();
        assert!(collect(&tx, &file, old).unwrap());
        tx.commit().unwrap();
        assert_eq!(
            lookup(
                &reader.index,
                &file,
                owner,
                Kind::Blocks,
                Some(old),
                0,
                &mut Work::default()
            )
            .unwrap(),
            Some(before)
        );
        assert_eq!(
            lookup(
                &writer._index,
                &file,
                owner,
                Kind::Blocks,
                Some(new),
                0,
                &mut Work::default()
            )
            .unwrap(),
            Some(item)
        );
        reader.index.execute_batch("ROLLBACK").unwrap();
        drop(reader);
        drop(writer);
    }
    #[test]
    fn max_depth_abort_drains_one_node_per_transaction_without_collecting_shared_live_path() {
        let (_dir, mut writer, owner, body) = fixture();
        let file = writer.root.join("index.db");
        let tx = writer._index.transaction().unwrap();
        let item = leaf(&tx, 0);
        let (old, _) = install(&tx, &file, owner, body, None, 0, item);
        let build: i64 = tx
            .query_row("SELECT pk FROM message_build WHERE message_pk=1", [], |r| {
                r.get(0)
            })
            .unwrap();
        acquire(&tx, &file, owner, old).unwrap();
        tx.execute("INSERT INTO message_build_frontier(build_pk,message_pk,session_pk,depth,node_pk,next_slot) VALUES(?1,1,1,1,?2,0)",params![build,old]).unwrap();
        tx.commit().unwrap();
        let tx = writer._index.transaction().unwrap();
        let item2 = leaf(&tx, 1);
        let (new, _) = install(&tx, &file, owner, body, Some(old), 1, item2);
        tx.commit().unwrap();
        let tx = writer._index.transaction().unwrap();
        tx.execute(
            "DELETE FROM message_build_frontier WHERE build_pk=?1 AND depth=0",
            [build],
        )
        .unwrap();
        release(&tx, &file, owner, new).unwrap();
        tx.commit().unwrap();
        let mut drained = 0;
        loop {
            let next = writer
                ._index
                .query_row(
                    "SELECT pk FROM message_body_node WHERE incoming=0 ORDER BY pk LIMIT 1",
                    [],
                    |r| r.get::<_, i64>(0),
                )
                .optional()
                .unwrap();
            let Some(next) = next else { break };
            let tx = writer._index.transaction().unwrap();
            assert!(collect(&tx, &file, next).unwrap());
            tx.commit().unwrap();
            drained += 1;
        }
        assert_eq!(drained, 16);
        assert_eq!(
            lookup(
                &writer._index,
                &file,
                owner,
                Kind::Blocks,
                Some(old),
                0,
                &mut Work::default()
            )
            .unwrap(),
            Some(item)
        );
        assert!(
            lookup(
                &writer._index,
                &file,
                owner,
                Kind::Blocks,
                Some(old),
                1,
                &mut Work::default()
            )
            .unwrap()
            .is_none()
        );
        let tx = writer._index.transaction().unwrap();
        tx.execute(
            "DELETE FROM message_build_frontier WHERE build_pk=?1 AND depth=1",
            [build],
        )
        .unwrap();
        release(&tx, &file, owner, old).unwrap();
        tx.commit().unwrap();
        let mut remaining = 0;
        loop {
            let next = writer
                ._index
                .query_row(
                    "SELECT pk FROM message_body_node WHERE incoming=0 ORDER BY pk LIMIT 1",
                    [],
                    |r| r.get::<_, i64>(0),
                )
                .optional()
                .unwrap();
            let Some(next) = next else { break };
            let tx = writer._index.transaction().unwrap();
            assert!(collect(&tx, &file, next).unwrap());
            tx.commit().unwrap();
            remaining += 1;
        }
        assert_eq!(remaining, 16);
        assert_eq!(
            writer
                ._index
                .query_row("SELECT count(*) FROM message_body_node", [], |r| r
                    .get::<_, i64>(0))
                .unwrap(),
            0
        );
    }

    #[test]
    fn lease_vs_collector_and_rollback_preserve_shared_children() {
        let (_dir, mut writer, owner, body) = fixture();
        let file = writer.root.join("index.db");
        let tx = writer._index.transaction().unwrap();
        let item = leaf(&tx, 0);
        let (root, _) = install(&tx, &file, owner, body, None, 0, item);
        tx.commit().unwrap();
        let tx = writer._index.transaction().unwrap();
        assert!(!collect(&tx, &file, root).unwrap());
        tx.commit().unwrap();
        let tx = writer._index.transaction().unwrap();
        let second = leaf(&tx, 1);
        let (next, _) = install(&tx, &file, owner, body, Some(root), 1, second);
        tx.commit().unwrap();
        let child: i64 = writer
            ._index
            .query_row(
                "SELECT child_node_pk FROM message_body_edge WHERE parent_pk=?1 AND slot=0",
                [root],
                |r| r.get(0),
            )
            .unwrap();
        let refs: i64 = writer
            ._index
            .query_row(
                "SELECT incoming FROM message_body_node WHERE pk=?1",
                [child],
                |r| r.get(0),
            )
            .unwrap();
        let tx = writer._index.transaction().unwrap();
        assert!(collect(&tx, &file, root).unwrap());
        tx.rollback().unwrap();
        assert_eq!(
            writer
                ._index
                .query_row(
                    "SELECT incoming FROM message_body_node WHERE pk=?1",
                    [child],
                    |r| r.get::<_, i64>(0)
                )
                .unwrap(),
            refs
        );
        let tx = writer._index.transaction().unwrap();
        assert!(collect(&tx, &file, root).unwrap());
        tx.commit().unwrap();
        assert!(
            lookup(
                &writer._index,
                &file,
                owner,
                Kind::Blocks,
                Some(next),
                0,
                &mut Work::default()
            )
            .unwrap()
            .is_some()
        );
        assert!(
            lookup(
                &writer._index,
                &file,
                owner,
                Kind::Blocks,
                Some(next),
                1,
                &mut Work::default()
            )
            .unwrap()
            .is_some()
        );
        assert!(
            load(
                &writer._index,
                &file,
                owner,
                Kind::Blocks,
                root,
                &mut Work::default()
            )
            .is_err()
        );
    }
}
