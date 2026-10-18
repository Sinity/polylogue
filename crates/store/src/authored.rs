//! Notes, corrections and withdrawals written by people or agents. They are
//! stored only in `archive.db`, each written with its `client_request` row, and
//! read from an archive snapshot; the index never holds them.
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
const MAX_REVISION_BYTES: usize = 262144;
mod evaluation;
#[cfg(test)]
mod tests;
use evaluation::{Evaluation, default_until};
use std::time::Instant;
fn unavailable(message: impl Into<String>) -> Error {
    failure(ErrorKind::NotYetAvailable, message)
}
fn check_deadline(until: Instant, cancel: &mut dyn FnMut() -> bool) -> Result<()> {
    if Instant::now() >= until || cancel() {
        return Err(Error::Cancelled);
    }
    Ok(())
}
fn failure(kind: ErrorKind, message: impl Into<String>) -> Error {
    Error::Authored(Failure::new(kind, message))
}
fn checked_reference(reference: &str) -> Result<(&str, Option<&str>)> {
    authored::note_reference(reference).map_err(Error::Authored)
}
/// Bytes of one revision's metadata read so far.
struct SizeBudget(usize);
impl SizeBudget {
    fn add(&mut self, n: usize) -> Result<()> {
        self.0 = self.0.checked_add(n).ok_or(Error::Range)?;
        if self.0 > MAX_REVISION_BYTES {
            return Err(Error::ParserAdmission {
                resource: "note revision metadata",
                required: self.0 as u64,
            });
        }
        Ok(())
    }
}
/// Reads one revision with its targets and relations. The size of each part
/// is checked in SQL before the text is loaded.
fn revision(conn: &Connection, path: &Path, id: &str, budget: &mut SizeBudget) -> Result<Revision> {
    let n: Option<i64> = db(path, || {
        conn.query_row(
            "SELECT length(CAST(id AS BLOB)) + length(CAST(root AS BLOB)) \
             + length(CAST(kind AS BLOB)) + coalesce(length(CAST(value AS BLOB)), 0) \
             + length(CAST(author AS BLOB)) + length(CAST(machine AS BLOB)) \
             FROM note WHERE id = ?1",
            [id],
            |r| r.get(0),
        )
        .optional()
    })?;
    budget.add(
        usize::try_from(n.ok_or_else(|| failure(ErrorKind::NotFound, "note revision not found"))?)
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
        conn.query_row(
            "SELECT id, root, kind, value, author, machine, created_at FROM note \
             WHERE id = ?1 AND key IS NULL AND check_json IS NULL \
             AND valid_until IS NULL AND source IS NULL",
            [id],
            |r| {
                Ok((
                    r.get(0)?,
                    r.get(1)?,
                    r.get(2)?,
                    r.get(3)?,
                    r.get(4)?,
                    r.get(5)?,
                    r.get(6)?,
                ))
            },
        )
    })?;
    let kind = match kind.as_str() {
        "note" => NoteKind::Note,
        "correction" => NoteKind::Correction,
        "withdrawal" => NoteKind::Withdrawal,
        _ => return Err(unavailable("note has an unsupported kind or field")),
    };
    let mut targets = vec![];
    let mut statement = db(path, || {
        conn.prepare(
            "SELECT length(CAST(ref AS BLOB)), \
             CASE WHEN length(CAST(ref AS BLOB)) <= 16384 THEN ref END, \
             CASE WHEN length(seen_hash) = 32 THEN seen_hash END, \
             anchor_input, anchor_chain, anchor_start, anchor_end, anchor_parser, \
             length(seen_hash) FROM note_target WHERE note_id = ?1 ORDER BY ref LIMIT 9",
        )
    })?;
    let mut rows = db(path, || statement.query([&id]))?;
    while let Some(r) = db(path, || rows.next())? {
        if targets.len() == authored::MAX_TARGETS {
            return Err(unavailable("note has too many targets"));
        }
        budget.add(
            usize::try_from(db(path, || crate::parser_state::unsigned(r, 0))?)
                .map_err(|_| Error::Range)?
                + 32,
        )?;
        let reference: Option<String> = db(path, || r.get(1))?;
        let hash_length: Option<i64> = db(path, || r.get(8))?;
        if hash_length.is_some_and(|n| n != 32) {
            return Err(unavailable("stored seen_hash is not 32 bytes"));
        }
        let bytes: Option<Vec<u8>> = db(path, || r.get(2))?;
        for i in 3..8 {
            if db(path, || r.get_ref(i))? != rusqlite::types::ValueRef::Null {
                return Err(unavailable("anchored note targets are not supported yet"));
            }
        }
        let seen_hash = bytes
            .map(|v| {
                v.try_into()
                    .map(BlobHash::from_bytes)
                    .map_err(|_| Error::Invariant("stored seen_hash is not 32 bytes".into()))
            })
            .transpose()?;
        targets.push(Target {
            reference: reference.ok_or_else(|| unavailable("note target reference is too long"))?,
            seen_hash,
        });
    }
    drop(rows);
    drop(statement);
    let mut relations = vec![];
    let mut statement = db(path, || {
        conn.prepare(
            "SELECT length(CAST(kind AS BLOB)) + length(CAST(other AS BLOB)), \
             CASE WHEN length(CAST(kind AS BLOB)) + length(CAST(other AS BLOB)) <= 17408 \
             THEN kind END, \
             CASE WHEN length(CAST(kind AS BLOB)) + length(CAST(other AS BLOB)) <= 17408 \
             THEN other END \
             FROM note_relation WHERE note_id = ?1 ORDER BY kind, other LIMIT 9",
        )
    })?;
    let mut rows = db(path, || statement.query([&id]))?;
    while let Some(r) = db(path, || rows.next())? {
        if relations.len() == 8 {
            return Err(unavailable("note has too many relations"));
        }
        budget.add(
            usize::try_from(db(path, || crate::parser_state::unsigned(r, 0))?)
                .map_err(|_| Error::Range)?,
        )?;
        let kind: Option<String> = db(path, || r.get(1))?;
        let other: Option<String> = db(path, || r.get(2))?;
        let (kind, other) = kind
            .zip(other)
            .ok_or_else(|| unavailable("note relation is too long"))?;
        relations.push(Relation { kind, other });
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
/// Fails unless `id` is a revision in the family `root` at archive counter `cutoff`.
fn check_member(conn: &Connection, path: &Path, root: &str, id: &str, cutoff: u64) -> Result<()> {
    let exists: bool = db(path, || {
        conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM note WHERE id = ?1 AND root = ?2 AND local_seq <= ?3)",
            params![id, root, cutoff as i64],
            |r| r.get(0),
        )
    })?;
    if !exists {
        return Err(failure(
            ErrorKind::NotFound,
            "note is not a revision of this family at this archive counter",
        ));
    }
    Ok(())
}
/// The revision ID a reference names: its pin, or the family root. With
/// `revise`, an unpinned reference resolves to the single current revision.
fn resolve_revision(
    conn: &Connection,
    path: &Path,
    reference: &str,
    revise: bool,
    until: Instant,
) -> Result<String> {
    let (root, pin) = checked_reference(reference)?;
    let cutoff = counter(conn, path)?;
    if let Some(pin) = pin {
        check_member(conn, path, root, pin, cutoff)?;
        return Ok(pin.into());
    }
    check_member(conn, path, root, root, cutoff)?;
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
/// Collects the families with a revision that targets `target`, keyed by the
/// first matching `note_target` row, in a temporary on-disk database so that
/// memory does not grow with the number of notes.
fn families_citing(
    conn: &Connection,
    path: &Path,
    target: &str,
    cutoff: u64,
    until: Instant,
    cancel: &mut dyn FnMut() -> bool,
) -> Result<(Connection, tempfile::NamedTempFile)> {
    let (population, _file) = evaluation::workspace()?;
    population
        .execute_batch("CREATE TABLE family(root TEXT PRIMARY KEY, first_rowid INTEGER UNIQUE);")
        .map_err(|e| unavailable(e.to_string()))?;
    let mut pos = 0i64;
    let mut steps = 0u64;
    loop {
        check_deadline(until, cancel)?;
        reset_cache(&population)?;
        steps += 1;
        if steps > 1_000_000 {
            return Err(unavailable("too many steps collecting note backlinks"));
        }
        let row: Option<(i64, Option<String>, u64)> = db(path, || {
            conn.query_row(
                "SELECT t.rowid, CASE WHEN length(CAST(n.root AS BLOB)) <= 128 THEN n.root END, \
                 n.local_seq FROM note_target t INDEXED BY note_target_ref \
                 JOIN note n ON n.id = t.note_id \
                 WHERE t.ref = ?1 AND t.rowid > ?2 ORDER BY t.rowid LIMIT 1",
                params![target, pos],
                |r| Ok((r.get(0)?, r.get(1)?, crate::parser_state::unsigned(r, 2)?)),
            )
            .optional()
        })?;
        let Some((row, root, seq)) = row else { break };
        pos = row;
        if seq > cutoff {
            continue;
        }
        let root = root.ok_or_else(|| unavailable("note ID is too long"))?;
        population
            .execute(
                "INSERT OR IGNORE INTO family(root, first_rowid) VALUES(?1, ?2)",
                params![root, row],
            )
            .map_err(|e| unavailable(format!("note backlinks scratch database failed: {e}")))?;
    }
    Ok((population, _file))
}
/// Drops the scratch database's page cache so its memory stays small.
fn reset_cache(scratch: &Connection) -> Result<()> {
    crate::native::reset_scratch_cache(scratch)
        .map_err(|e| unavailable(format!("note scratch cache reset failed: {e}")))
}
impl Writer {
    /// The archive's recovery record, if any. Read it before writing a note so
    /// that a later status failure cannot change what the reply reports.
    pub fn authored_recovery(&self) -> Result<Option<polylogue_model::RecoveryRecord>> {
        crate::recovery::record(&self.archive, &self.root.join("archive.db"))
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
            return revision(&tx, &path, &request.request_id, &mut SizeBudget(0));
        }
        let mut kind = request.kind.unwrap_or(NoteKind::Note);
        let mut root = request.request_id.clone();
        let mut targets = request.targets.clone().unwrap_or_default();
        let mut relations = vec![];
        for r in &request.relations {
            if r.kind == "revises" {
                let id = resolve_revision(&tx, &path, &r.other, true, until)?;
                let previous = revision(&tx, &path, &id, &mut SizeBudget(0))?;
                if previous.kind == NoteKind::Withdrawal {
                    return Err(unavailable(
                        "only notes and corrections can be revised; withdraw a withdrawal to undo it",
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
                        format!("revision is out of date; current revisions: {current:?}"),
                    ));
                }
                if request.kind.is_some_and(|k| k != previous.kind) {
                    return Err(failure(
                        ErrorKind::InvalidQuery,
                        "a revision cannot change the note kind",
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
                let id = resolve_revision(&tx, &path, &r.other, false, until)?;
                let n = revision(&tx, &path, &id, &mut SizeBudget(0))?;
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
                "a note needs at least one target",
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
            tx.execute(
                "INSERT INTO note(id, root, kind, value, author, machine, created_at, local_seq) \
                 VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    request.request_id,
                    root,
                    kind.as_str(),
                    request.value,
                    request.author,
                    request.machine,
                    created_at,
                    next as i64
                ],
            )
        })?;
        for t in &targets {
            db(&path, || {
                tx.execute(
                    "INSERT INTO note_target(note_id, ref, seen_hash) VALUES(?1, ?2, ?3)",
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
                    "INSERT INTO note_relation(note_id, kind, other) VALUES(?1, ?2, ?3)",
                    params![request.request_id, r.kind, r.other],
                )
            })?;
        }
        db(&path, || {
            tx.execute(
                "UPDATE meta SET value = ?1 WHERE key = 'write_counter'",
                [next.to_string()],
            )
        })?;
        // Refuse a note that would leave a family, or the backlinks of a
        // target, impossible to read; failure rolls back the whole write. A
        // replayed request returned above and is not checked again.
        let mut cancel = || false;
        Evaluation::new(&tx, &path, &root, next, until, &mut cancel)?.check_all_families()?;
        for target in &targets {
            families_citing(&tx, &path, &target.reference, next, until, &mut cancel)?;
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
    /// The highest note `local_seq`; unlike the archive counter it changes
    /// only when a note is written.
    pub authored_sequence: u64,
    pub recovery: Option<polylogue_model::RecoveryRecord>,
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
        let recovery = crate::recovery::record(&conn, &path)?;
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
                "archive counter is newer than this snapshot",
            ));
        }
        let (root, pin) = checked_reference(reference)?;
        let id = pin.ok_or_else(|| {
            failure(
                ErrorKind::InvalidQuery,
                "value reads need a pinned revision (note:ROOT@ID)",
            )
        })?;
        check_member(&self.conn, &self.path, root, id, cutoff)?;
        revision(&self.conn, &self.path, id, &mut SizeBudget(0))
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
            return Err(failure(ErrorKind::InvalidQuery, "invalid page position"));
        }
        let (root, pin) = checked_reference(reference)?;
        if kind == PageKind::Current && pin.is_some() {
            return Err(failure(
                ErrorKind::InvalidQuery,
                "current pages require a family reference without a revision pin",
            ));
        }
        check_member(&self.conn, &self.path, root, root, cutoff)?;
        if let Some(pin) = pin {
            check_member(&self.conn, &self.path, root, pin, cutoff)?;
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
        let mut shown = vec![];
        for id in &selected {
            if Instant::now() >= until {
                return Err(Error::Cancelled);
            }
            shown.push(self.note_state(&mut eval, id)?);
        }
        let mut history = vec![];
        let mut history_after = None;
        if matches!(kind, PageKind::Family | PageKind::History) {
            let mut pos = if kind == PageKind::History { after } else { 0 };
            loop {
                eval.checkpoint()?;
                let next = evaluation::next_in_family(&self.conn, &self.path, root, pos)?;
                let Some((row, id, seq)) = next else { break };
                if seq > cutoff {
                    pos = row;
                    continue;
                }
                if history.len() == 32 {
                    history_after = Some(pos);
                    break;
                }
                let id = id.ok_or_else(|| unavailable("note ID is too long"))?;
                history.push(format!("note:{root}@{id}"));
                pos = row;
            }
        }
        let mut withdrawal_trail = vec![];
        let mut withdrawal_after = None;
        if matches!(kind, PageKind::Family | PageKind::Withdrawals) {
            let (start, limit) = if kind == PageKind::Withdrawals {
                (after, 1)
            } else {
                (0, 8)
            };
            let mut rows = eval.trail(root, start, limit)?;
            if rows.len() > limit {
                withdrawal_after = Some(rows[limit - 1].0);
                rows.truncate(limit);
            }
            let mut trail_bytes = 0usize;
            let mut last = start;
            for (row, id) in rows {
                let size: i64 = db(&self.path, || {
                    self.conn.query_row(
                        "SELECT length(CAST(value AS BLOB)) FROM note WHERE id = ?1",
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
                withdrawal_trail.push(self.note_state(&mut eval, &id)?);
            }
        }
        let next_current = if kind == PageKind::Current {
            after + 1
        } else {
            1
        };
        let current_after = (pin.is_none()
            && matches!(kind, PageKind::Family | PageKind::Current)
            && (next_current as usize) < heads.len())
        .then_some(next_current);
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
    fn note_state(&self, eval: &mut Evaluation, id: &str) -> Result<NoteState> {
        let revision = revision(&self.conn, &self.path, id, &mut SizeBudget(0))?;
        let current = eval.is_current(id, &revision.root)?;
        let withdrawn = eval.withdrawn(id)? || eval.withdrawn(&revision.root)?;
        Ok(NoteState {
            revision,
            current,
            withdrawn,
        })
    }
    /// One page of the note families whose revisions target `target`.
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
                "invalid target backlinks page",
            ));
        }
        let (population, _file) =
            families_citing(&self.conn, &self.path, target, cutoff, until, cancel)?;
        let mut families = vec![];
        let mut next = None;
        let mut pos = after;
        loop {
            check_deadline(until, cancel)?;
            reset_cache(&population)?;
            let row: Option<(i64, String)> = population
                .query_row(
                    "SELECT first_rowid, root FROM family WHERE first_rowid > ?1 \
                     ORDER BY first_rowid LIMIT 1",
                    [pos],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )
                .optional()
                .map_err(|e| unavailable(e.to_string()))?;
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
                check_deadline(until, cancel)?;
                let m: Option<(i64, String, u64, bool)> = db(&self.path, || {
                    self.conn
                        .query_row(
                            "SELECT n.rowid, n.id, n.local_seq, EXISTS(SELECT 1 FROM note_target t \
                             WHERE t.note_id = n.id AND t.ref = ?3) \
                             FROM note n INDEXED BY note_root \
                             WHERE n.root = ?1 AND n.rowid > ?2 ORDER BY n.rowid LIMIT 1",
                            params![root, member_pos, target],
                            |r| {
                                Ok((
                                    r.get(0)?,
                                    r.get(1)?,
                                    crate::parser_state::unsigned(r, 2)?,
                                    r.get(3)?,
                                ))
                            },
                        )
                        .optional()
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
                check_deadline(until, cancel)?;
                let (_, pin) = checked_reference(current)?;
                let matches: bool = db(&self.path, || {
                    self.conn.query_row(
                        "SELECT EXISTS(SELECT 1 FROM note_target t JOIN note n ON n.id = t.note_id \
                         WHERE t.note_id = ?1 AND t.ref = ?2 AND n.local_seq <= ?3)",
                        params![pin.unwrap(), target, cutoff as i64],
                        |r| r.get(0),
                    )
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
