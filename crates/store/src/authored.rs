//! Durable attributed notes, using the archive writer and existing request identity.
//! The index is never authority for authored intent, revision order or withdrawal.
use crate::{Error, ReadHandle, Result, Writer, client_request, counter, db};
use polylogue_model::{
    BlobHash, ErrorKind, Failure,
    authored::{self, NoteKind, Relation, Revision, Target, WriteNote},
};
use rusqlite::{Connection, OptionalExtension, params};
use serde::Serialize;
use std::{
    fs::File,
    path::{Path, PathBuf},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};
const BYTES: usize = 262144;
mod evaluation;
#[cfg(test)]
mod tests;
use evaluation::{Evaluation, default_until};
use std::time::Instant;
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
                resource: "authored exact revision metadata",
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
fn member(conn: &Connection, path: &Path, root: &str, id: &str, cutoff: u64) -> Result<()> {
    let exists: bool = db(path, || {
        conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM note WHERE id=?1 AND root=?2 AND local_seq<=?3)",
            params![id, root, cutoff as i64],
            |r| r.get(0),
        )
    })?;
    if !exists {
        return Err(failure(
            ErrorKind::NotFound,
            "pinned note is not a retained revision of this family at the read basis",
        ));
    }
    Ok(())
}
fn exact_id(
    conn: &Connection,
    path: &Path,
    reference: &str,
    revise: bool,
    until: Instant,
) -> Result<String> {
    let (root, pin) = checked_reference(reference)?;
    let cutoff = counter(conn, path)?;
    if let Some(pin) = pin {
        member(conn, path, root, pin, cutoff)?;
        return Ok(pin.into());
    }
    member(conn, path, root, root, cutoff)?;
    if !revise {
        return Ok(root.into());
    }
    let mut cancel = || false;
    let heads = Evaluation::new(conn, path, root, cutoff, until, &mut cancel)?.heads(root)?;
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
fn association_population(
    conn: &Connection,
    path: &Path,
    target: &str,
    cutoff: u64,
    until: Instant,
    cancel: &mut dyn FnMut() -> bool,
) -> Result<(Connection, tempfile::NamedTempFile)> {
    // Build only the unique family population in bounded indexed source
    // steps. The temporary index spills to owned scratch; no Rust set grows
    // with annotations and no DISTINCT aggregate hides evaluation work.
    let (population, _file) = evaluation::workspace()?;
    population
        .execute_batch("CREATE TABLE family(root TEXT PRIMARY KEY, first_rowid INTEGER UNIQUE);")
        .map_err(|e| failure(ErrorKind::NotYetAvailable, e.to_string()))?;
    let mut pos = 0i64;
    let mut steps = 0u64;
    loop {
        if Instant::now() >= until || cancel() {
            return Err(Error::Cancelled);
        }
        crate::native::reset_scratch_cache(&population).map_err(|e| {
            failure(
                ErrorKind::NotYetAvailable,
                format!("authored association cache reset unavailable: {e}"),
            )
        })?;
        steps += 1;
        if steps > 1_000_000 {
            return Err(failure(
                ErrorKind::NotYetAvailable,
                "authored association evaluation step capacity exceeded",
            ));
        }
        let row: Option<(i64, Option<String>, u64)> = db(path, || {
            conn.query_row("SELECT t.rowid,CASE WHEN length(CAST(n.root AS BLOB))<=128 THEN n.root END,n.local_seq FROM note_target t INDEXED BY note_target_ref JOIN note n ON n.id=t.note_id WHERE t.ref=?1 AND t.rowid>?2 ORDER BY t.rowid LIMIT 1",params![target,pos],|r|Ok((r.get(0)?,r.get(1)?,crate::parser_state::unsigned(r,2)?))).optional()
        })?;
        let Some((row, root, seq)) = row else { break };
        pos = row;
        if seq > cutoff {
            continue;
        }
        let root = root.ok_or_else(|| {
            failure(
                ErrorKind::NotYetAvailable,
                "authored identity admission exceeded",
            )
        })?;
        population
            .execute(
                "INSERT OR IGNORE INTO family(root,first_rowid) VALUES(?1,?2)",
                params![root, row],
            )
            .map_err(|e| {
                failure(
                    ErrorKind::NotYetAvailable,
                    format!("authored association scratch capacity/unavailability: {e}"),
                )
            })?;
    }
    Ok((population, _file))
}
impl Writer {
    /// Archive qualification is independent of derived-tier availability. Read
    /// it before the action; a later status failure cannot relabel a commit.
    pub fn authored_recovery(&self) -> Result<Option<polylogue_model::RecoveryQualification>> {
        crate::recovery::qualification(&self.archive, &self.root.join("archive.db"))
    }
    pub fn write_note(&mut self, request: &WriteNote) -> Result<Revision> {
        self.write_note_until(request, default_until())
    }
    pub fn write_note_until(&mut self, request: &WriteNote, until: Instant) -> Result<Revision> {
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
                let id = exact_id(&tx, &path, &r.other, true, until)?;
                let previous = revision(&tx, &path, &id, &mut Admission(0))?;
                if previous.kind == NoteKind::Withdrawal {
                    return Err(failure(
                        ErrorKind::NotYetAvailable,
                        "revise note/correction only; restore a withdrawal by withdrawing it",
                    ));
                }
                let mut cancel = || false;
                let heads = Evaluation::new(
                    &tx,
                    &path,
                    &previous.root,
                    counter(&tx, &path)?,
                    until,
                    &mut cancel,
                )?
                .heads(&previous.root)?;
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
                let id = exact_id(&tx, &path, &r.other, false, until)?;
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
        // Prospective control evaluation includes every family reached through
        // withdrawals. Failure rolls back note, request identity and counter.
        // Committed replay above deliberately never reruns this admission.
        let mut cancel = || false;
        Evaluation::new(&tx, &path, &root, next, until, &mut cancel)?.qualify_all_families()?;
        // New target membership also has a bounded navigation dependency.
        // Reuse the reader's exact unique-population builder before admitting
        // growth that would make that association traversal unavailable.
        for target in &targets {
            association_population(&tx, &path, &target.reference, next, until, &mut cancel)?;
        }
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
    pub matching_target_revisions_complete: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_target_revisions: Option<Vec<String>>,
    pub current: Vec<String>,
    pub history: Vec<String>,
    pub withdrawal_trail: Vec<NoteState>,
    pub shown: Vec<NoteState>,
    pub status_at_archive_counter: u64,
    pub history_complete: Option<bool>,
    pub withdrawal_trail_complete: Option<bool>,
    pub shown_current_complete: Option<bool>,
    #[serde(skip)]
    pub history_after: Option<i64>,
    #[serde(skip)]
    pub withdrawal_after: Option<i64>,
    #[serde(skip)]
    pub current_after: Option<i64>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PageKind {
    Family,
    History,
    Withdrawals,
    Current,
}
pub struct AssociationPage {
    pub families: Vec<Family>,
    pub after: Option<i64>,
    pub cutoff: u64,
}
pub struct Snapshot {
    conn: Connection,
    path: PathBuf,
    pub archive_counter: u64,
    /// Indexed authored-only watermark, separate from general archive bookkeeping.
    pub authored_sequence: u64,
    pub recovery: Option<polylogue_model::RecoveryQualification>,
    _lock: Arc<File>,
}
impl ReadHandle {
    pub fn authored_snapshot(&self) -> Result<Snapshot> {
        let path = self.root.join("archive.db");
        let conn = crate::open(&path, true, true)?;
        db(&path, || conn.execute_batch("BEGIN DEFERRED"))?;
        let archive_counter = counter(&conn, &path)?;
        let authored_sequence = db(&path, || {
            conn.query_row(
                "SELECT local_seq FROM note INDEXED BY note_local ORDER BY local_seq DESC LIMIT 1",
                [],
                |r| crate::parser_state::unsigned(r, 0),
            )
            .optional()
        })?
        .unwrap_or(0);
        let recovery = crate::recovery::qualification(&conn, &path)?;
        Ok(Snapshot {
            conn,
            path,
            archive_counter,
            authored_sequence,
            recovery,
            _lock: Arc::clone(&self._lock),
        })
    }
}
impl Snapshot {
    pub fn identity(&self) -> [u8; 32] {
        *blake3::hash(self.path.as_os_str().as_encoded_bytes()).as_bytes()
    }
    pub fn exact(&self, reference: &str, cutoff: u64) -> Result<Revision> {
        if cutoff > self.archive_counter {
            return Err(failure(
                ErrorKind::InvalidQuery,
                "unavailable authored read basis",
            ));
        }
        let (root, pin) = checked_reference(reference)?;
        let id = pin.ok_or_else(|| {
            failure(
                ErrorKind::InvalidQuery,
                "value-only reads require an exact pinned revision",
            )
        })?;
        member(&self.conn, &self.path, root, id, cutoff)?;
        revision(&self.conn, &self.path, id, &mut Admission(0))
    }
    pub fn family(&self, reference: &str) -> Result<Family> {
        self.family_page(
            reference,
            self.archive_counter,
            PageKind::Family,
            0,
            default_until(),
            &mut || false,
        )
    }
    pub fn family_page(
        &self,
        reference: &str,
        cutoff: u64,
        kind: PageKind,
        after: i64,
        until: Instant,
        cancel: &mut dyn FnMut() -> bool,
    ) -> Result<Family> {
        if cutoff > self.archive_counter || after < 0 {
            return Err(failure(
                ErrorKind::InvalidQuery,
                "invalid authored page basis",
            ));
        }
        let (root, pin) = checked_reference(reference)?;
        if kind == PageKind::Current && pin.is_some() {
            return Err(failure(
                ErrorKind::InvalidQuery,
                "current pages require a family reference without a revision pin",
            ));
        }
        member(&self.conn, &self.path, root, root, cutoff)?;
        if let Some(pin) = pin {
            member(&self.conn, &self.path, root, pin, cutoff)?;
        }
        let mut eval = Evaluation::new(&self.conn, &self.path, root, cutoff, until, cancel)?;
        let heads = eval.heads(root)?;
        let selected: Vec<String> = match kind {
            PageKind::Family => pin.map_or_else(
                || heads.first().cloned().into_iter().collect(),
                |p| vec![p.into()],
            ),
            PageKind::Current => heads.get(after as usize).cloned().into_iter().collect(),
            _ => vec![],
        };
        let mut state = |id: &str| -> Result<NoteState> {
            if Instant::now() >= until {
                return Err(Error::Cancelled);
            }
            let n = revision(&self.conn, &self.path, id, &mut Admission(0))?;
            let current = eval.is_current(id, &n.root)?;
            let withdrawn = eval.withdrawn(id)? || eval.withdrawn(&n.root)?;
            Ok(NoteState {
                revision: n,
                current,
                withdrawn,
            })
        };
        let shown = selected
            .iter()
            .map(|id| state(id))
            .collect::<Result<Vec<_>>>()?;
        let mut history = vec![];
        let mut history_after = None;
        if matches!(kind, PageKind::Family | PageKind::History) {
            let mut pos = if kind == PageKind::History { after } else { 0 };
            loop {
                eval.checkpoint()?;
                let next: Option<(i64, Option<String>, u64)> = db(&self.path, || {
                    self.conn.query_row("SELECT rowid,CASE WHEN length(CAST(id AS BLOB))<=128 THEN id END,local_seq FROM note INDEXED BY note_root WHERE root=?1 AND rowid>?2 ORDER BY rowid LIMIT 1",params![root,pos],|r|Ok((r.get(0)?,r.get(1)?,crate::parser_state::unsigned(r,2)?))).optional()
                })?;
                let Some((row, id, seq)) = next else { break };
                if seq > cutoff {
                    pos = row;
                    continue;
                }
                if history.len() == 32 {
                    history_after = Some(pos);
                    break;
                }
                let id = id.ok_or_else(|| {
                    failure(
                        ErrorKind::NotYetAvailable,
                        "authored identity admission exceeded",
                    )
                })?;
                history.push(format!("note:{root}@{id}"));
                pos = row;
            }
        }
        let mut withdrawal_trail = vec![];
        let mut withdrawal_after = None;
        if matches!(kind, PageKind::Family | PageKind::Withdrawals) {
            let mut rows = eval.trail(
                root,
                if kind == PageKind::Withdrawals {
                    after
                } else {
                    0
                },
                if kind == PageKind::Withdrawals { 1 } else { 8 },
            )?;
            let limit = if kind == PageKind::Withdrawals { 1 } else { 8 };
            if rows.len() > limit {
                withdrawal_after = Some(rows[limit - 1].0);
                rows.truncate(limit);
            }
            let mut trail_bytes = 0usize;
            let mut last = if kind == PageKind::Withdrawals {
                after
            } else {
                0
            };
            for (row, id) in rows {
                let size: i64 = db(&self.path, || {
                    self.conn.query_row(
                        "SELECT length(CAST(value AS BLOB)) FROM note WHERE id=?1",
                        [&id],
                        |r| r.get(0),
                    )
                })?;
                if kind == PageKind::Family && trail_bytes.saturating_add(size as usize) > 8192 {
                    withdrawal_after = Some(last);
                    break;
                }
                trail_bytes += size as usize;
                last = row;
                let n = revision(&self.conn, &self.path, &id, &mut Admission(0))?;
                let current = eval.is_current(&id, &n.root)?;
                let withdrawn = eval.withdrawn(&id)? || eval.withdrawn(&n.root)?;
                withdrawal_trail.push(NoteState {
                    revision: n,
                    current,
                    withdrawn,
                });
            }
        }
        let current_after = if pin.is_none()
            && matches!(kind, PageKind::Family | PageKind::Current)
            && (if kind == PageKind::Current {
                after as usize + 1
            } else {
                1
            }) < heads.len()
        {
            Some(if kind == PageKind::Current {
                after + 1
            } else {
                1
            })
        } else {
            None
        };
        Ok(Family {
            root: root.into(),
            conflict: heads.len() > 1,
            matching_target_revisions: None,
            matching_target_revisions_complete: None,
            current_target_revisions: None,
            current: heads.iter().map(|id| format!("note:{root}@{id}")).collect(),
            history,
            withdrawal_trail,
            shown,
            status_at_archive_counter: cutoff,
            history_complete: matches!(kind, PageKind::Family | PageKind::History)
                .then_some(history_after.is_none()),
            withdrawal_trail_complete: matches!(kind, PageKind::Family | PageKind::Withdrawals)
                .then_some(withdrawal_after.is_none()),
            shown_current_complete: (pin.is_none()
                && matches!(kind, PageKind::Family | PageKind::Current))
            .then_some(current_after.is_none()),
            history_after,
            withdrawal_after,
            current_after,
        })
    }
    pub fn backlinks_page(
        &self,
        target: &str,
        cutoff: u64,
        after: i64,
        limit: usize,
        until: Instant,
        cancel: &mut dyn FnMut() -> bool,
    ) -> Result<AssociationPage> {
        if !(1..=16).contains(&limit)
            || !authored::target_reference(target)
            || cutoff > self.archive_counter
            || after < 0
        {
            return Err(failure(
                ErrorKind::InvalidQuery,
                "invalid exact target association page",
            ));
        }
        let (population, _file) =
            association_population(&self.conn, &self.path, target, cutoff, until, cancel)?;
        let mut families = vec![];
        let mut next = None;
        let mut pos = after;
        loop {
            if Instant::now() >= until || cancel() {
                return Err(Error::Cancelled);
            }
            crate::native::reset_scratch_cache(&population).map_err(|e| {
                failure(
                    ErrorKind::NotYetAvailable,
                    format!("authored association cache reset unavailable: {e}"),
                )
            })?;
            let row:Option<(i64,String)>=population.query_row("SELECT first_rowid,root FROM family WHERE first_rowid>?1 ORDER BY first_rowid LIMIT 1",[pos],|r|Ok((r.get(0)?,r.get(1)?))).optional().map_err(|e|failure(ErrorKind::NotYetAvailable,e.to_string()))?;
            let Some((row, root)) = row else { break };
            if families.len() == limit {
                next = Some(pos);
                break;
            }
            let mut family = self.family_page(
                &format!("note:{root}"),
                cutoff,
                PageKind::Family,
                0,
                until,
                cancel,
            )?;
            let mut matching = vec![];
            let mut matching_complete = true;
            let mut member_pos = 0i64;
            loop {
                if Instant::now() >= until || cancel() {
                    return Err(Error::Cancelled);
                }
                let m: Option<(i64, String, u64, bool)> = db(&self.path, || {
                    self.conn.query_row("SELECT n.rowid,n.id,n.local_seq,EXISTS(SELECT 1 FROM note_target t WHERE t.note_id=n.id AND t.ref=?3) FROM note n INDEXED BY note_root WHERE n.root=?1 AND n.rowid>?2 ORDER BY n.rowid LIMIT 1",params![root,member_pos,target],|r|Ok((r.get(0)?,r.get(1)?,crate::parser_state::unsigned(r,2)?,r.get(3)?))).optional()
                })?;
                let Some((r, id, seq, matches)) = m else {
                    break;
                };
                member_pos = r;
                if seq <= cutoff && matches {
                    if matching.len() == 32 {
                        matching_complete = false;
                        break;
                    }
                    matching.push(format!("note:{root}@{id}"));
                }
            }
            family.matching_target_revisions = Some(matching);
            family.matching_target_revisions_complete = Some(matching_complete);
            let mut current_matches = vec![];
            for current in &family.current {
                if Instant::now() >= until || cancel() {
                    return Err(Error::Cancelled);
                }
                let (_, pin) = checked_reference(current)?;
                let matches: bool = db(&self.path, || {
                    self.conn.query_row("SELECT EXISTS(SELECT 1 FROM note_target t JOIN note n ON n.id=t.note_id WHERE t.note_id=?1 AND t.ref=?2 AND n.local_seq<=?3)",params![pin.unwrap(),target,cutoff as i64],|r|r.get(0))
                })?;
                if matches {
                    current_matches.push(current.clone());
                }
            }
            family.current_target_revisions = Some(current_matches);
            families.push(family);
            pos = row;
        }
        Ok(AssociationPage {
            families,
            after: next,
            cutoff,
        })
    }
}
