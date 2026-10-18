//! Per-message radix-16 trees of fixed height 16, stored as immutable rows in
//! `message_body_node` and `message_body_edge`. A change writes one new
//! root-to-leaf path (16 nodes) and shares every unchanged subtree; nodes are
//! reference-counted and freed by `collect`. The caller owns the transaction.
use crate::{Error, Result, db};
use rusqlite::{Connection, OptionalExtension, Transaction, params};
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    Groups,
    Blocks,
    Fields,
    Operations,
    /// The competing field facts behind one conflicted field slot.
    ConflictFacts,
}
impl Kind {
    fn name(self) -> &'static str {
        match self {
            Self::Groups => "groups",
            Self::Blocks => "blocks",
            Self::Fields => "fields",
            Self::Operations => "operations",
            Self::ConflictFacts => "witnesses",
        }
    }
    fn parse(s: &str) -> Result<Self> {
        match s {
            "groups" => Ok(Self::Groups),
            "blocks" => Ok(Self::Blocks),
            "fields" => Ok(Self::Fields),
            "operations" => Ok(Self::Operations),
            "witnesses" => Ok(Self::ConflictFacts),
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
    pub(super) fn add(self, rhs: Self) -> Result<Self> {
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
    Conflict { slot: i64, root: i64, height: u8 },
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
pub(super) fn summary(
    dbh: &Connection,
    path: &Path,
    owner: Owner,
    kind: Kind,
    root: Option<i64>,
) -> Result<Counts> {
    let Some(pk) = root else {
        return Ok(Counts::default());
    };
    let node = load(dbh, path, owner, kind, pk, &mut Work::default())?;
    if node.height != 15 {
        return Err(bad("directory root height"));
    }
    node.edges
        .iter()
        .flatten()
        .try_fold(Counts::default(), |a, b| a.add(b.counts))
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
    // Statements are cached, but every call still reads and checks the node
    // and all of its edges: owner, state and counts.
    let (height,summary):(u8,Counts)=db(path,||dbh.prepare_cached("SELECT height,ordinary_count,reasoning_count,source_blocks,source_text_blocks,source_fields FROM message_body_node WHERE pk=?1 AND message_pk=?2 AND session_pk=?3 AND kind=?4 AND state='ready'")?.query_row(params![pk,owner.message,owner.session,kind.name()],|r|Ok((r.get(0)?,counts(r,1)?))).optional())?.ok_or_else(||bad("missing or unready directory node"))?;
    work.nodes_read += 1;
    let mut node = Node {
        height,
        edges: [None; 16],
    };
    let mut stmt = db(path, || {
        dbh.prepare_cached("SELECT slot,child_node_pk,child_kind,child_height,block_pk,field_fact_pk,tool_call_pk,association_pk,conflict_slot_pk,ordinary_count,reasoning_count,source_blocks,source_text_blocks,source_fields FROM message_body_edge WHERE parent_pk=?1 ORDER BY slot")
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
        let conflict: Option<i64> = db(path, || row.get(8))?;
        let link = if let Some(slot) = conflict {
            let root = child.ok_or_else(|| bad("conflict edge has no fact subtree"))?;
            let k: String = db(path, || row.get(2))?;
            let h: u8 = db(path, || row.get(3))?;
            if kind != Kind::Fields || height != 0 || k != Kind::ConflictFacts.name() {
                return Err(bad("conflict directory shape"));
            }
            Link::Conflict {
                slot,
                root,
                height: h,
            }
        } else if let Some(child) = child {
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
            counts: db(path, || counts(row, 9))?,
        };
        if matches!(edge.link, Link::Conflict { .. }) && edge.counts != Counts::default() {
            return Err(bad("conflict edge has nonzero counts"));
        }
        total = total.add(edge.counts)?;
        node.edges[slot] = Some(edge);
        work.edges_read += 1;
    }
    if total != summary {
        return Err(bad("directory node counts differ from its edges"));
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
    // Insert the node once, already ready and with its final counts. Any later
    // edge error rolls back the caller's transaction.
    db(path, || {
        tx.prepare_cached("INSERT INTO message_body_node(message_pk,session_pk,kind,height,state,incoming,ordinary_count,reasoning_count,source_blocks,source_text_blocks,source_fields) VALUES(?1,?2,?3,?4,'ready',0,?5,?6,?7,?8,?9)")?.execute(params![owner.message,owner.session,kind.name(),height,total.ordinary,total.reasoning,total.source_blocks,total.source_text_blocks,total.source_fields])
    })?;
    let pk = tx.last_insert_rowid();
    work.nodes_written += 1;
    for (slot, edge) in edges.into_iter().enumerate() {
        let Some(edge) = edge else { continue };
        let (child, k, h, block, field, call, association, conflict) = match edge.link {
            Link::Node { pk, kind, height } => (
                Some(pk),
                Some(kind.name()),
                Some(height),
                None,
                None,
                None,
                None,
                None,
            ),
            Link::Block { pk, call } => (None, None, None, Some(pk), None, call, None, None),
            Link::Field(pk) => (None, None, None, None, Some(pk), None, None, None),
            Link::Operation(pk) => (None, None, None, None, None, None, Some(pk), None),
            Link::Conflict { slot, root, height } => (
                Some(root),
                Some(Kind::ConflictFacts.name()),
                Some(height),
                None,
                None,
                None,
                None,
                Some(slot),
            ),
        };
        if let Some(child) = child {
            if db(path, || {
                tx.prepare_cached("UPDATE message_body_node SET incoming=incoming+1 WHERE pk=?1 AND message_pk=?2 AND session_pk=?3 AND state='ready'")?.execute(params![child,owner.message,owner.session])
            })? != 1
            {
                return Err(bad("cannot acquire missing/retiring directory child"));
            }
            work.references_changed += 1;
        }
        db(path, || {
            tx.prepare_cached("INSERT INTO message_body_edge(parent_pk,message_pk,session_pk,parent_kind,parent_height,slot,child_node_pk,child_kind,child_height,block_pk,field_fact_pk,tool_call_pk,association_pk,conflict_slot_pk,ordinary_count,reasoning_count,source_blocks,source_text_blocks,source_fields) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19)")?.execute(params![pk,owner.message,owner.session,kind.name(),height,slot as i64,child,k,h,block,field,call,association,conflict,edge.counts.ordinary,edge.counts.reasoning,edge.counts.source_blocks,edge.counts.source_text_blocks,edge.counts.source_fields])
        })?;
        work.edges_written += 1;
    }
    Ok((pk, total))
}
/// Write a new path with `leaf` at `key` (`None` removes it) and return the new
/// root. Unchanged siblings are shared. The caller must `acquire` the new root
/// before committing.
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
/// Count a reference to a root from a row that points at it.
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
/// Delete one unreferenced node and release its children, blocks and facts,
/// in the caller's transaction. Children that become unreferenced are left
/// for later calls. Returns false if the node is still referenced or gone.
pub(super) fn collect(tx: &Transaction<'_>, path: &Path, pk: i64) -> Result<bool> {
    if db(path, || {
        tx.execute("UPDATE message_body_node SET state='retiring' WHERE pk=?1 AND incoming=0 AND state IN('building','ready')",[pk])
    })? == 0
    {
        return Ok(false);
    }
    let mut children = [None; 16];
    let mut blocks = [None; 16];
    let mut facts = [None; 16];
    let mut stmt = db(path, || {
        tx.prepare(
            "SELECT slot,child_node_pk,block_pk,field_fact_pk FROM message_body_edge WHERE parent_pk=?1 ORDER BY slot",
        )
    })?;
    let mut rows = db(path, || stmt.query([pk]))?;
    while let Some(row) = db(path, || rows.next())? {
        let slot = usize::from(db(path, || row.get::<_, u8>(0))?);
        if slot >= 16 {
            return Err(bad("directory edge slot out of range"));
        }
        children[slot] = db(path, || row.get::<_, Option<i64>>(1))?;
        blocks[slot] = db(path, || row.get::<_, Option<i64>>(2))?;
        facts[slot] = db(path, || row.get::<_, Option<i64>>(3))?;
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
    for block in blocks.into_iter().flatten() {
        super::blocks::release_unreferenced_block(tx, path, block)?;
    }
    for fact in facts.into_iter().flatten() {
        super::retirement::release_unreferenced_fact(tx, path, fact)?;
    }
    db(path, || {
        tx.execute(
            "DELETE FROM message_body_node WHERE pk=?1 AND incoming=0 AND state='retiring'",
            [pk],
        )
    })?;
    Ok(true)
}

/// The leaf at `key`, found by walking one root-to-leaf path.
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

/// The first leaf with a key above `after`, including leaves with zero counts.
/// Reads at most 31 nodes: the path to the lower bound plus one path down to
/// a minimum. An empty non-root node is an error.
pub(super) fn next(
    dbh: &Connection,
    path: &Path,
    owner: Owner,
    kind: Kind,
    root: Option<i64>,
    after: Option<u64>,
    work: &mut Work,
) -> Result<Option<(u64, Edge)>> {
    let lower = match after {
        Some(key) => match key.checked_add(1) {
            Some(key) => key,
            None => return Ok(None),
        },
        None => 0,
    };
    let Some(root) = root else { return Ok(None) };
    #[derive(Clone, Copy)]
    struct Frame {
        node: Node,
        prefix: u64,
        slot: u8,
    }
    let node = load(dbh, path, owner, kind, root, work)?;
    if node.height != 15 {
        return Err(bad("directory successor root height"));
    }
    let mut stack = [None; 16];
    stack[0] = Some(Frame {
        node,
        prefix: 0,
        slot: 0,
    });
    let mut depth = 0;
    let mut visited = 1u8;
    loop {
        let frame = stack[depth]
            .as_mut()
            .ok_or_else(|| bad("missing successor frame"))?;
        if frame.slot == 16 {
            if depth == 0 {
                return Ok(None);
            }
            stack[depth] = None;
            depth -= 1;
            continue;
        }
        let slot = frame.slot;
        frame.slot += 1;
        let Some(edge) = frame.node.edges[usize::from(slot)] else {
            continue;
        };
        let height = frame.node.height;
        let shift = u32::from(height) * 4;
        let prefix = frame.prefix | (u64::from(slot) << shift);
        let upper = prefix | ((1u64 << shift) - 1);
        if upper < lower {
            continue;
        }
        if height == 0 {
            return Ok(Some((prefix, edge)));
        }
        let Link::Node {
            pk,
            kind: child_kind,
            height: child_height,
        } = edge.link
        else {
            return Err(bad("directory successor shape"));
        };
        if child_kind != kind || child_height != height - 1 || visited >= 31 {
            return Err(bad(
                "directory successor path is invalid or exceeds its fixed bound",
            ));
        }
        let child = load(dbh, path, owner, kind, pk, work)?;
        visited += 1;
        if child.height != child_height || child.edges.iter().all(Option::is_none) {
            return Err(bad(
                "directory successor child is empty or has the wrong height",
            ));
        }
        depth += 1;
        stack[depth] = Some(Frame {
            node: child,
            prefix,
            slot: 0,
        });
    }
}

/// Which count `nth` indexes by.
#[derive(Clone, Copy)]
pub(super) enum Ordinal {
    Ordinary,
    Reasoning,
    All,
}
pub(super) struct Selected {
    pub key: u64,
    pub edge: Edge,
    pub remainder: u64,
    pub skipped: Counts,
}
/// The `index`-th leaf by the chosen count, found from node counts along one
/// path. Keys may be sparse; the index is a position, not a key.
pub(super) fn nth(
    dbh: &Connection,
    path: &Path,
    owner: Owner,
    kind: Kind,
    root: Option<i64>,
    mut index: u64,
    selection: Ordinal,
    work: &mut Work,
) -> Result<Option<Selected>> {
    let Some(mut pk) = root else { return Ok(None) };
    let mut key = 0u64;
    let mut skipped = Counts::default();
    for height in (0..16u8).rev() {
        let node = load(dbh, path, owner, kind, pk, work)?;
        if node.height != height {
            return Err(bad("directory selection height"));
        }
        let mut selected = None;
        for (slot, edge) in node.edges.into_iter().enumerate() {
            let Some(edge) = edge else { continue };
            let count = match selection {
                Ordinal::Reasoning => edge.counts.reasoning,
                Ordinal::Ordinary => edge.counts.ordinary,
                Ordinal::All => edge
                    .counts
                    .ordinary
                    .checked_add(edge.counts.reasoning)
                    .ok_or(Error::Range)?,
            };
            let count = u64::try_from(count).map_err(|_| bad("negative directory count"))?;
            if index < count {
                selected = Some((slot, edge));
                break;
            }
            index = index.checked_sub(count).ok_or(Error::Range)?;
            skipped = skipped.add(edge.counts)?;
        }
        let Some((slot, edge)) = selected else {
            return Ok(None);
        };
        key |= (slot as u64) << (height * 4);
        if height == 0 {
            return Ok(Some(Selected {
                key,
                edge,
                remainder: index,
                skipped,
            }));
        }
        match edge.link {
            Link::Node {
                pk: child,
                kind: child_kind,
                height: child_height,
            } if child_kind == kind && child_height == height - 1 => pk = child,
            _ => return Err(bad("directory selection shape")),
        }
    }
    Err(bad("directory selection escaped fixed height"))
}

#[cfg(test)]
mod tests;
