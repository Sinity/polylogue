//! Durable attributed notes, using the archive writer and existing request identity.
//! The index is never authority for authored intent, revision order or withdrawal.
use crate::{Error, ReadHandle, Result, Writer, client_request, counter, db};
use polylogue_model::{
    BlobHash, ErrorKind, Failure,
    authored::{self, GraphNode, NoteKind, Relation, Revision, Target, WriteNote},
};
use rusqlite::{Connection, OptionalExtension, params};
use serde::Serialize;
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    fs::File,
    path::{Path, PathBuf},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};
const BYTES: usize = 262144;
#[cfg(test)]
mod tests;
fn failure(kind: ErrorKind, message: impl Into<String>) -> Error {
    Error::Authored(Failure::new(kind, message))
}
fn checked_reference(reference: &str) -> Result<(&str, Option<&str>)> {
    authored::note_reference(reference).map_err(Error::Authored)
}
struct Admission(usize);
impl Admission {
    fn add(&mut self, n: usize) -> Result<()> {
        self.0 = self.0.checked_add(n).ok_or(Error::Range)?;
        if self.0 > BYTES {
            return Err(Error::ParserAdmission {
                resource: "authored family metadata",
                required: self.0 as u64,
            });
        }
        Ok(())
    }
}
fn revision(conn: &Connection, path: &Path, id: &str, budget: &mut Admission) -> Result<Revision> {
    let n: Option<i64> = db(path, || {
        conn.query_row("SELECT length(CAST(id AS BLOB))+length(CAST(root AS BLOB))+length(CAST(kind AS BLOB))+coalesce(length(CAST(value AS BLOB)),0)+length(CAST(author AS BLOB))+length(CAST(machine AS BLOB)) FROM note WHERE id=?1",[id],|r|r.get(0)).optional()
    })?;
    budget.add(
        usize::try_from(n.ok_or_else(|| failure(ErrorKind::NotFound, "note revision is absent"))?)
            .map_err(|_| Error::Range)?,
    )?;
    let (id, root, kind, value, author, machine, created_at): (
        String,
        String,
        String,
        String,
        String,
        String,
        i64,
    ) = db(path, || {
        conn.query_row("SELECT id,root,kind,value,author,machine,created_at FROM note WHERE id=?1 AND key IS NULL AND check_json IS NULL AND valid_until IS NULL AND source IS NULL",[id],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?,r.get(5)?,r.get(6)?)))
    })?;
    let kind = match kind.as_str() {
        "note" => NoteKind::Note,
        "correction" => NoteKind::Correction,
        "withdrawal" => NoteKind::Withdrawal,
        _ => {
            return Err(failure(
                ErrorKind::NotYetAvailable,
                "authored family contains an unsupported note kind or field",
            ));
        }
    };
    let mut targets = vec![];
    let mut statement = db(path, || {
        conn.prepare("SELECT length(CAST(ref AS BLOB)),CASE WHEN length(CAST(ref AS BLOB))<=16384 THEN ref END,CASE WHEN length(seen_hash)=32 THEN seen_hash END,anchor_input,anchor_chain,anchor_start,anchor_end,anchor_parser,length(seen_hash) FROM note_target WHERE note_id=?1 ORDER BY ref LIMIT 9")
    })?;
    let mut rows = db(path, || statement.query([&id]))?;
    while let Some(r) = db(path, || rows.next())? {
        if targets.len() == authored::MAX_TARGETS {
            return Err(failure(
                ErrorKind::NotYetAvailable,
                "authored target admission exceeded",
            ));
        }
        budget.add(
            usize::try_from(db(path, || crate::parser_state::unsigned(r, 0))?)
                .map_err(|_| Error::Range)?
                + 32,
        )?;
        let reference: Option<String> = db(path, || r.get(1))?;
        let hash_length: Option<i64> = db(path, || r.get(8))?;
        if hash_length.is_some_and(|n| n != 32) {
            return Err(failure(
                ErrorKind::NotYetAvailable,
                "invalid authored basis length",
            ));
        }
        let bytes: Option<Vec<u8>> = db(path, || r.get(2))?;
        for i in 3..8 {
            if db(path, || r.get_ref(i))? != rusqlite::types::ValueRef::Null {
                return Err(failure(
                    ErrorKind::NotYetAvailable,
                    "anchored authored target presentation is unsupported",
                ));
            }
        }
        let seen_hash = bytes
            .map(|v| {
                v.try_into()
                    .map(BlobHash::from_bytes)
                    .map_err(|_| Error::Invariant("invalid authored basis length".into()))
            })
            .transpose()?;
        targets.push(Target {
            reference: reference.ok_or_else(|| {
                failure(
                    ErrorKind::NotYetAvailable,
                    "authored target identity admission exceeded",
                )
            })?,
            seen_hash,
        });
    }
    drop(rows);
    drop(statement);
    let mut relations = vec![];
    let mut statement = db(path, || {
        conn.prepare("SELECT length(CAST(kind AS BLOB))+length(CAST(other AS BLOB)),CASE WHEN length(CAST(kind AS BLOB))+length(CAST(other AS BLOB))<=17408 THEN kind END,CASE WHEN length(CAST(kind AS BLOB))+length(CAST(other AS BLOB))<=17408 THEN other END FROM note_relation WHERE note_id=?1 ORDER BY kind,other LIMIT 9")
    })?;
    let mut rows = db(path, || statement.query([&id]))?;
    while let Some(r) = db(path, || rows.next())? {
        if relations.len() == 8 {
            return Err(failure(
                ErrorKind::NotYetAvailable,
                "authored relation admission exceeded",
            ));
        }
        budget.add(
            usize::try_from(db(path, || crate::parser_state::unsigned(r, 0))?)
                .map_err(|_| Error::Range)?,
        )?;
        let kind: Option<String> = db(path, || r.get(1))?;
        let other: Option<String> = db(path, || r.get(2))?;
        relations.push(Relation {
            kind: kind.ok_or_else(|| {
                failure(
                    ErrorKind::NotYetAvailable,
                    "authored relation admission exceeded",
                )
            })?,
            other: other.ok_or_else(|| {
                failure(
                    ErrorKind::NotYetAvailable,
                    "authored relation admission exceeded",
                )
            })?,
        });
    }
    Ok(Revision {
        id,
        root,
        kind,
        value,
        author,
        machine,
        created_at,
        targets,
        relations,
    })
}
fn graph(
    conn: &Connection,
    path: &Path,
    root: &str,
) -> Result<(BTreeMap<String, Revision>, BTreeMap<String, GraphNode>)> {
    let mut ids = VecDeque::new();
    let mut statement = db(path, || {
        conn.prepare("SELECT CASE WHEN length(CAST(id AS BLOB))<=128 THEN id END FROM note WHERE root=?1 ORDER BY id LIMIT 129")
    })?;
    for id in db(path, || {
        statement.query_map([root], |r| r.get::<_, Option<String>>(0))
    })? {
        ids.push_back(db(path, || id)?.ok_or_else(|| {
            failure(
                ErrorKind::NotYetAvailable,
                "authored identity admission exceeded",
            )
        })?);
    }
    drop(statement);
    if ids.is_empty() {
        return Err(failure(ErrorKind::NotFound, "note family is absent"));
    }
    let mut expanded = BTreeSet::from([root.to_owned()]);
    let mut notes = BTreeMap::new();
    let mut budget = Admission(0);
    while let Some(id) = ids.pop_front() {
        if notes.contains_key(&id) {
            continue;
        }
        if notes.len() == authored::MAX_GRAPH {
            return Err(failure(
                ErrorKind::NotYetAvailable,
                "authored revision/withdrawal graph exceeds128 nodes",
            ));
        }
        let n = revision(conn, path, &id, &mut budget)?;
        let mut statement = db(path, || {
            conn.prepare("SELECT CASE WHEN length(CAST(note_id AS BLOB))<=128 THEN note_id END FROM note_relation WHERE other=?1 AND kind='withdraws' ORDER BY note_id LIMIT 129")
        })?;
        for other in db(path, || {
            statement.query_map([n.reference()], |r| r.get::<_, Option<String>>(0))
        })? {
            ids.push_back(db(path, || other)?.ok_or_else(|| {
                failure(
                    ErrorKind::NotYetAvailable,
                    "authored identity admission exceeded",
                )
            })?);
        }
        // Include complete related families and both ends of every control.
        // Forward ordinary targets/corrects are not effective-state dependencies.
        if expanded.insert(n.root.clone()) {
            let mut statement = db(path, || {
                conn.prepare("SELECT CASE WHEN length(CAST(id AS BLOB))<=128 THEN id END FROM note WHERE root=?1 ORDER BY id LIMIT 129")
            })?;
            for related in db(path, || {
                statement.query_map([&n.root], |r| r.get::<_, Option<String>>(0))
            })? {
                ids.push_back(db(path, || related)?.ok_or_else(|| {
                    failure(
                        ErrorKind::NotYetAvailable,
                        "authored identity admission exceeded",
                    )
                })?);
            }
        }
        for relation in &n.relations {
            if matches!(relation.kind.as_str(), "revises" | "withdraws") {
                let (parent, pin) = checked_reference(&relation.other)?;
                ids.push_back(pin.unwrap_or(parent).into());
            }
        }
        notes.insert(id, n);
        if ids.len() > authored::MAX_GRAPH * authored::MAX_GRAPH {
            return Err(failure(
                ErrorKind::NotYetAvailable,
                "authored graph work admission exceeded",
            ));
        }
    }
    let mut g = BTreeMap::new();
    for (id, n) in &notes {
        let mut relations = vec![];
        for r in &n.relations {
            if matches!(r.kind.as_str(), "revises" | "withdraws") {
                let (rroot, pin) = checked_reference(&r.other)?;
                relations.push(Relation {
                    kind: r.kind.clone(),
                    other: pin.unwrap_or(rroot).into(),
                });
            }
        }
        g.insert(
            id.clone(),
            GraphNode {
                root: n.root.clone(),
                relations,
            },
        );
    }
    authored::validate_controls(&g).map_err(Error::Authored)?;
    Ok((notes, g))
}
fn exact_id(conn: &Connection, path: &Path, reference: &str, revise: bool) -> Result<String> {
    let (root, pin) = checked_reference(reference)?;
    let (notes, g) = graph(conn, path, root)?;
    if let Some(pin) = pin {
        if notes.get(pin).is_none_or(|n| n.root != root) {
            return Err(failure(
                ErrorKind::NotFound,
                "pinned note is not a revision of this family",
            ));
        }
        return Ok(pin.into());
    }
    if !revise {
        return Ok(root.into());
    }
    let heads = authored::current(root, &g).map_err(Error::Authored)?;
    if heads.len() != 1 {
        return Err(failure(
            ErrorKind::Conflict,
            format!(
                "family has {} current revisions; read note:{root}",
                heads.len()
            ),
        ));
    }
    Ok(heads[0].clone())
}
impl Writer {
    pub fn write_note(&mut self, request: &WriteNote) -> Result<Revision> {
        let intent = request.intent().map_err(Error::Authored)?;
        let hash = client_request::digest(&intent)?;
        let path = self.root.join("archive.db");
        let tx = db(&path, || self.archive.transaction())?;
        if client_request::replay(&tx, &path, &request.request_id, hash)? {
            return revision(&tx, &path, &request.request_id, &mut Admission(0));
        }
        let mut kind = request.kind.unwrap_or(NoteKind::Note);
        let mut root = request.request_id.clone();
        let mut targets = request.targets.clone().unwrap_or_default();
        let mut relations = vec![];
        for r in &request.relations {
            if r.kind == "revises" {
                let id = exact_id(&tx, &path, &r.other, true)?;
                let previous = revision(&tx, &path, &id, &mut Admission(0))?;
                if previous.kind == NoteKind::Withdrawal {
                    return Err(failure(
                        ErrorKind::NotYetAvailable,
                        "revise note/correction only; restore a withdrawal by withdrawing it",
                    ));
                }
                let (_, g) = graph(&tx, &path, &previous.root)?;
                let heads = authored::current(&previous.root, &g).map_err(Error::Authored)?;
                if heads != [id.clone()] {
                    let current = heads
                        .iter()
                        .map(|id| format!("note:{}@{id}", previous.root))
                        .collect::<Vec<_>>();
                    return Err(failure(
                        ErrorKind::Conflict,
                        format!("stale ordinary revision; current revisions: {current:?}"),
                    ));
                }
                if request.kind.is_some_and(|k| k != previous.kind) {
                    return Err(failure(
                        ErrorKind::InvalidQuery,
                        "ordinary revision cannot change note kind",
                    ));
                }
                kind = previous.kind;
                root = previous.root.clone();
                let reference = previous.reference();
                if request.targets.is_none() {
                    targets = previous.targets;
                }
                relations.extend(previous.relations.into_iter().filter(|r| {
                    r.kind == "corrects" && !request.relations.iter().any(|x| x.kind == "corrects")
                }));
                relations.push(Relation {
                    kind: r.kind.clone(),
                    other: reference,
                });
            } else if r.kind == "withdraws" {
                let id = exact_id(&tx, &path, &r.other, false)?;
                let n = revision(&tx, &path, &id, &mut Admission(0))?;
                relations.push(Relation {
                    kind: r.kind.clone(),
                    other: n.reference(),
                });
            } else {
                relations.push(r.clone());
            }
        }
        if kind != NoteKind::Withdrawal && targets.is_empty() {
            return Err(failure(
                ErrorKind::InvalidQuery,
                "authored note requires exact targets",
            ));
        }
        targets.sort_by(|a, b| a.reference.cmp(&b.reference));
        relations.sort_by(|a, b| (&a.kind, &a.other).cmp(&(&b.kind, &b.other)));
        let created_at = request.created_at.unwrap_or(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(|_| Error::Range)?
                .as_millis()
                .try_into()
                .map_err(|_| Error::Range)?,
        );
        let next = counter(&tx, &path)?
            .checked_add(1)
            .filter(|n| *n <= i64::MAX as u64)
            .ok_or(Error::Range)?;
        client_request::insert(&tx, &path, &request.request_id, hash, created_at)?;
        db(&path, || {
            tx.execute("INSERT INTO note(id,root,kind,value,author,machine,created_at,local_seq) VALUES(?1,?2,?3,?4,?5,?6,?7,?8)",params![request.request_id,root,kind.as_str(),request.value,request.author,request.machine,created_at,next as i64])
        })?;
        for t in &targets {
            db(&path, || {
                tx.execute(
                    "INSERT INTO note_target(note_id,ref,seen_hash) VALUES(?1,?2,?3)",
                    params![
                        request.request_id,
                        t.reference,
                        t.seen_hash.map(|h| h.bytes().to_vec())
                    ],
                )
            })?;
        }
        for r in &relations {
            db(&path, || {
                tx.execute(
                    "INSERT INTO note_relation(note_id,kind,other) VALUES(?1,?2,?3)",
                    params![request.request_id, r.kind, r.other],
                )
            })?;
        }
        db(&path, || {
            tx.execute(
                "UPDATE meta SET value=?1 WHERE key='write_counter'",
                [next.to_string()],
            )
        })?;
        db(&path, || tx.commit())?;
        Ok(Revision {
            id: request.request_id.clone(),
            root,
            kind,
            value: request.value.clone(),
            author: request.author.clone(),
            machine: request.machine.clone(),
            created_at,
            targets,
            relations,
        })
    }
}
#[derive(Serialize)]
pub struct NoteState {
    pub revision: Revision,
    pub current: bool,
    pub withdrawn: bool,
}
#[derive(Serialize)]
pub struct Family {
    pub root: String,
    pub conflict: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub matching_target_revisions: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_target_revisions: Option<Vec<String>>,
    pub current: Vec<String>,
    pub history: Vec<String>,
    pub withdrawal_trail: Vec<NoteState>,
    pub shown: Vec<NoteState>,
}
pub struct Snapshot {
    conn: Connection,
    path: PathBuf,
    pub archive_counter: u64,
    pub recovery: Option<polylogue_model::RecoveryQualification>,
    _lock: Arc<File>,
}
impl ReadHandle {
    pub fn authored_snapshot(&self) -> Result<Snapshot> {
        let path = self.root.join("archive.db");
        let conn = crate::open(&path, true, true)?;
        db(&path, || conn.execute_batch("BEGIN DEFERRED"))?;
        let archive_counter = counter(&conn, &path)?;
        let recovery = crate::recovery::qualification(&conn, &path)?;
        Ok(Snapshot {
            conn,
            path,
            archive_counter,
            recovery,
            _lock: Arc::clone(&self._lock),
        })
    }
}
impl Snapshot {
    pub fn family(&self, reference: &str) -> Result<Family> {
        let (root, pin) = checked_reference(reference)?;
        let (notes, g) = graph(&self.conn, &self.path, root)?;
        let heads = authored::current(root, &g).map_err(Error::Authored)?;
        let selected: Vec<String> = match pin {
            Some(id) => {
                if notes.get(id).is_none_or(|n| n.root != root) {
                    return Err(failure(
                        ErrorKind::NotFound,
                        "pinned note is not in this family",
                    ));
                }
                vec![id.into()]
            }
            None => heads.clone(),
        };
        let state = |n: &Revision| -> Result<NoteState> {
            Ok(NoteState {
                revision: n.clone(),
                current: authored::current(&n.root, &g)
                    .map_err(Error::Authored)?
                    .contains(&n.id),
                withdrawn: authored::withdrawn(&n.id, &g).map_err(Error::Authored)?
                    || authored::withdrawn(&n.root, &g).map_err(Error::Authored)?,
            })
        };
        Ok(Family {
            root: root.into(),
            conflict: heads.len() > 1,
            matching_target_revisions: None,
            current_target_revisions: None,
            current: heads.iter().map(|id| format!("note:{root}@{id}")).collect(),
            history: notes
                .values()
                .filter(|n| n.root == root)
                .map(Revision::reference)
                .collect(),
            withdrawal_trail: notes
                .values()
                .filter(|n| n.root != root)
                .map(state)
                .collect::<Result<_>>()?,
            shown: selected
                .iter()
                .map(|id| state(&notes[id]))
                .collect::<Result<_>>()?,
        })
    }
    pub fn backlinks(&self, target: &str) -> Result<Vec<Family>> {
        if !authored::target_reference(target) {
            return Err(failure(
                ErrorKind::InvalidQuery,
                "backlinks require an exact Message/block target",
            ));
        }
        let mut statement = db(&self.path, || {
            self.conn.prepare("SELECT DISTINCT CASE WHEN length(CAST(n.root AS BLOB))<=128 THEN n.root END FROM note_target t JOIN note n ON n.id=t.note_id WHERE t.ref=?1 ORDER BY n.root LIMIT 17")
        })?;
        let roots = db(&self.path, || {
            statement.query_map([target], |r| r.get::<_, Option<String>>(0))
        })?
        .map(|r| {
            db(&self.path, || r)?.ok_or_else(|| {
                failure(
                    ErrorKind::NotYetAvailable,
                    "authored identity admission exceeded",
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;
        if roots.len() > 16 {
            return Err(failure(
                ErrorKind::NotYetAvailable,
                "exact target backlinks exceed16 families",
            ));
        }
        roots
            .into_iter()
            .map(|root| {
                let mut family = self.family(&format!("note:{root}"))?;
                let mut admission = Admission(0);
                let matching = family
                    .history
                    .iter()
                    .filter_map(|reference| {
                        let (_, pin) = checked_reference(reference).ok()?;
                        Some((reference, pin.unwrap()))
                    })
                    .map(|(reference, id)| {
                        Ok((
                            reference.clone(),
                            revision(&self.conn, &self.path, id, &mut admission)?,
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?
                    .into_iter()
                    .filter(|(_, n)| n.targets.iter().any(|t| t.reference == target))
                    .map(|(r, _)| r)
                    .collect::<Vec<_>>();
                family.current_target_revisions = Some(
                    family
                        .current
                        .iter()
                        .filter(|r| matching.contains(r))
                        .cloned()
                        .collect(),
                );
                family.matching_target_revisions = Some(matching);
                Ok(family)
            })
            .collect()
    }
}
