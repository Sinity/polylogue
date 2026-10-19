//! Excision: permanent removal from every file and derived product
//! (architecture 5.6; `27m`, `layg`, `kwsb`). The `excision` row keeps the
//! content from coming back by any route.
//!
//! - **Session:** its rows go from every `index.db` table, its texts from
//!   the search indexes and `embeddings.db` (when no other row holds the same
//!   text), and its bytes from the stored inputs. An input holding only
//!   this session is removed whole; in an input it shares with other
//!   sessions, the byte ranges of its records are overwritten with spaces
//!   of the same length (line breaks kept), so every other record keeps its
//!   offsets. A shared whole document (an export) has no record ranges: the
//!   session's own texts are blanked where they stand as complete JSON
//!   string values, quotes kept, so the document still parses. Notes about
//!   the session go, and briefs that delivered its texts are redacted. The
//!   writer refuses the session's and its messages' rows from then on, and
//!   the loop stops collecting a removed input's paths.
//! - **Input:** the input whole, with every session only it produced.
//! - **Text:** every stored occurrence of a string (a leaked secret), as
//!   written and as JSON escapes it, overwritten with spaces; the index
//!   texts holding it are removed and the inputs that held it are parsed
//!   again from the overwritten bytes.
//!
//! An excised revision keeps its BLAKE3 (the hash of the bytes as first
//! observed); its blobs are what it is verified by afterwards.

use crate::blobs::{Piece, Stored};
use crate::writer::unpack_text;
use crate::{Result, StoreError};
use polylogue_model::{BlobHash, SessionId};
use rusqlite::{OptionalExtension, Transaction, params};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashSet};

/// What to remove.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "scope", content = "value", rename_all = "snake_case")]
pub enum Scope {
    Session(SessionId),
    Input(String),
    Text(String),
}

impl Scope {
    /// `session:<id>`, `input:<id>`, `text:<string>`, or a bare session ID.
    pub fn parse(target: &str) -> std::result::Result<Self, String> {
        let (kind, value) = target.split_once(':').unwrap_or(("", target));
        match kind {
            "input" if !value.is_empty() => Ok(Self::Input(value.to_owned())),
            "text" if !value.is_empty() => Ok(Self::Text(value.to_owned())),
            "session" => SessionId::parse(value)
                .map(Self::Session)
                .map_err(|e| e.to_string()),
            _ => SessionId::parse(target).map(Self::Session).map_err(|_| {
                format!(
                    "not an excision target: {target:?} (session:<id>, input:<id>, text:<string>)"
                )
            }),
        }
    }

    fn label(&self) -> String {
        match self {
            Self::Session(s) => format!("session:{s}"),
            Self::Input(i) => format!("input:{i}"),
            // Never the text itself: the row keeps only its hash.
            Self::Text(t) => format!("text:{}", blake3::hash(t.as_bytes()).to_hex()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExciseRequest {
    /// The client request ID; the excision row's ID.
    pub id: String,
    pub scope: Scope,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// Count what would go without changing anything.
    #[serde(default)]
    pub dry_run: bool,
}

/// What an excision removed (or, on a dry run, would remove).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Excised {
    pub dry_run: bool,
    pub sessions: Vec<String>,
    pub messages: u64,
    pub texts: u64,
    /// Inputs removed whole.
    pub inputs_removed: Vec<String>,
    /// Inputs whose bytes were overwritten in place, and how many spans.
    pub inputs_overwritten: BTreeMap<String, u64>,
    /// Shared whole documents (exports) where some of the session's texts
    /// could not be found as a complete string value and so still stand
    /// (`texts_left` counts them); their rows are gone and refused.
    pub inputs_kept: Vec<String>,
    /// Session texts left in shared whole documents: too short to blank
    /// safely (under 12 bytes) or not stored as a whole string value.
    #[serde(default)]
    pub texts_left: u64,
    /// Notes whose every target was the excised session or its messages.
    #[serde(default)]
    pub notes_removed: u64,
    /// Delivered briefs whose text held the session's texts, blanked.
    #[serde(default)]
    pub deliveries_redacted: u64,
    pub blobs_deleted: u64,
    pub embeddings_deleted: u64,
    /// Paths the loop no longer collects, as `machine:path`.
    pub refused_paths: Vec<String>,
    /// Text scope: files on disk (`machine:path`) that still hold the text.
    /// Polylogue never edits them; a later full re-read of one (a rewrite,
    /// a parser version change) would bring the text back into the index
    /// until the file is scrubbed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub on_disk: Vec<String>,
}

/// The writer's record of what must not come back.
#[derive(Debug, Clone, Default)]
pub struct Refused {
    pub sessions: HashSet<String>,
    pub messages: HashSet<String>,
}

impl Refused {
    /// Loaded from `archive.db` when the writer starts.
    pub fn load(archive: &rusqlite::Connection) -> Result<Self> {
        let mut out = Self::default();
        let mut stmt = archive.prepare("SELECT scope, message_ids_json FROM excision")?;
        let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?;
        for row in rows {
            let (scope, ids) = row?;
            if let Some(s) = scope.strip_prefix("session:") {
                out.sessions.insert(s.to_owned());
            }
            let ids: Vec<String> = serde_json::from_str(&ids).unwrap_or_default();
            out.messages.extend(ids);
        }
        Ok(out)
    }

    pub fn is_empty(&self) -> bool {
        self.sessions.is_empty() && self.messages.is_empty()
    }
}

/// Paths (`(machine, path)`) the loop must never collect again, from every
/// excision's refused locations.
pub fn refused_paths(archive: &rusqlite::Connection) -> Result<Vec<(String, String)>> {
    let mut stmt = archive.prepare("SELECT refused_json FROM excision")?;
    let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
    let mut out = Vec::new();
    for row in rows {
        let refused: RefusedJson = serde_json::from_str(&row?).unwrap_or_default();
        out.extend(refused.locations.into_iter().map(|l| (l.machine, l.path)));
    }
    Ok(out)
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct RefusedJson {
    #[serde(default)]
    locations: Vec<Location>,
    /// BLAKE3 (hex) of every revision and blob the excision removed.
    #[serde(default)]
    revisions: Vec<String>,
    #[serde(default)]
    blobs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
struct Location {
    machine: String,
    path: String,
    device: Option<i64>,
    inode: Option<i64>,
}

/// One byte range in an input's revision chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Span {
    chain: i64,
    start: i64,
    end: i64,
}

/// The two transactions an excision runs in, with the blob store. Files of
/// deleted blobs are removed by the caller after both commit.
pub struct Txs<'a, 'b> {
    pub archive: &'a Transaction<'b>,
    pub index: &'a Transaction<'b>,
    pub embeddings: Option<&'a rusqlite::Connection>,
    pub store: &'a hashed_blobs::Store,
    /// `archive.db`'s write counter for this transaction.
    pub seq: i64,
    /// `index.db`'s write counter for this transaction.
    pub counter: i64,
}

/// Runs an excision. Returns what went and the blob files to delete once
/// both transactions have committed.
pub fn apply(t: &Txs<'_, '_>, req: &ExciseRequest) -> Result<(Excised, Vec<BlobHash>)> {
    let mut done = Excised {
        dry_run: req.dry_run,
        ..Excised::default()
    };
    let mut plan = Plan::default();
    match &req.scope {
        Scope::Session(id) => {
            let pk = session_pk(t.index, &id.to_string())?
                .ok_or_else(|| StoreError::NotFound(format!("session {id}")))?;
            plan.session(t, pk, &id.to_string())?;
        }
        Scope::Input(input) => {
            let pk = input_pk(t.index, input)?;
            let known: bool = t.archive.query_row(
                "SELECT EXISTS (SELECT 1 FROM input WHERE id = ?1)",
                [input],
                |r| r.get(0),
            )?;
            if !known {
                return Err(StoreError::NotFound(format!("input {input}")));
            }
            plan.whole.insert(input.clone());
            if let Some(pk) = pk {
                let sessions: Vec<(i64, String)> = t
                    .index
                    .prepare(
                        "SELECT s.pk, s.id FROM session s WHERE s.pk IN
                             (SELECT session_pk FROM parsed WHERE input_pk = ?1)
                         AND NOT EXISTS (SELECT 1 FROM parsed p WHERE p.session_pk = s.pk
                                         AND p.input_pk <> ?1)",
                    )?
                    .query_map([pk], |r| Ok((r.get(0)?, r.get(1)?)))?
                    .collect::<rusqlite::Result<_>>()?;
                for (spk, sid) in sessions {
                    plan.session(t, spk, &sid)?;
                }
            }
        }
        Scope::Text(text) => {
            plan.text = Some(text.clone());
        }
    }
    plan.apply(t, req, &mut done)
}

#[derive(Default)]
struct Plan {
    /// Sessions to remove: pk and id.
    sessions: Vec<(i64, String)>,
    /// Inputs to remove whole.
    whole: BTreeSet<String>,
    /// Inputs shared with other sessions: the spans to overwrite.
    spans: BTreeMap<String, Vec<Span>>,
    /// Shared inputs whose records have no byte ranges.
    kept: BTreeSet<String>,
    /// Message IDs of the removed sessions, for the writer's refusal.
    message_ids: Vec<String>,
    /// Texts only the removed sessions hold (bodies and titles), for
    /// blanking shared whole documents and delivered briefs.
    own_texts: BTreeSet<String>,
    text: Option<String>,
}

impl Plan {
    fn session(&mut self, t: &Txs<'_, '_>, pk: i64, id: &str) -> Result<()> {
        self.sessions.push((pk, id.to_owned()));
        let ids: Vec<String> = t
            .index
            .prepare("SELECT id FROM message WHERE session_pk = ?1")?
            .query_map([pk], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        self.message_ids.extend(ids);
        // Its own texts: those no other session's block holds.
        let bodies: Vec<Vec<u8>> = t
            .index
            .prepare(
                "SELECT DISTINCT t.body FROM block b JOIN text t ON t.id = b.text_id
                 WHERE b.session_pk = ?1 AND NOT EXISTS
                   (SELECT 1 FROM block o WHERE o.text_id = b.text_id AND o.session_pk <> ?1)",
            )?
            .query_map([pk], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        for body in bodies {
            if let Ok(s) = unpack_text(&body) {
                self.own_texts.insert(s);
            }
        }
        let title: Option<String> = t
            .index
            .query_row(
                "SELECT title FROM session s WHERE pk = ?1 AND NOT EXISTS
               (SELECT 1 FROM session o WHERE o.title = s.title AND o.pk <> ?1)",
                [pk],
                |r| r.get(0),
            )
            .optional()?
            .flatten();
        self.own_texts.extend(title);
        // Every input the session's rows came from.
        let inputs: Vec<(i64, String)> = t
            .index
            .prepare(
                "SELECT pk, id FROM input_ref WHERE pk IN (
                     SELECT input_pk FROM parsed WHERE session_pk = ?1
                     UNION SELECT s.input_pk FROM message_source s JOIN message m ON m.pk = s.message_pk
                           WHERE m.session_pk = ?1
                     UNION SELECT input_pk FROM event WHERE session_pk = ?1)",
            )?
            .query_map([pk], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<rusqlite::Result<_>>()?;
        for (ipk, iid) in inputs {
            let shared: bool = t.index.query_row(
                "SELECT EXISTS (SELECT 1 FROM parsed WHERE input_pk = ?1 AND session_pk <> ?2)
                     OR EXISTS (SELECT 1 FROM message_source s JOIN message m ON m.pk = s.message_pk
                                WHERE s.input_pk = ?1 AND m.session_pk <> ?2)",
                params![ipk, pk],
                |r| r.get(0),
            )?;
            if !shared {
                self.whole.insert(iid);
                continue;
            }
            let mut spans: Vec<Span> = t
                .index
                .prepare(
                    "SELECT s.chain, s.source_start, s.source_end FROM message_source s
                     JOIN message m ON m.pk = s.message_pk
                     WHERE m.session_pk = ?2 AND s.input_pk = ?1 AND s.source_end > s.source_start
                     UNION SELECT source_chain, source_start, source_end FROM event
                     WHERE session_pk = ?2 AND input_pk = ?1 AND source_chain IS NOT NULL
                           AND source_end > source_start",
                )?
                .query_map(params![ipk, pk], |r| {
                    Ok(Span {
                        chain: r.get(0)?,
                        start: r.get(1)?,
                        end: r.get(2)?,
                    })
                })?
                .collect::<rusqlite::Result<_>>()?;
            if spans.is_empty() {
                self.kept.insert(iid);
            } else {
                spans.sort();
                self.spans.entry(iid).or_default().extend(spans);
            }
        }
        Ok(())
    }

    fn apply(
        mut self,
        t: &Txs<'_, '_>,
        req: &ExciseRequest,
        done: &mut Excised,
    ) -> Result<(Excised, Vec<BlobHash>)> {
        let mut refused = RefusedJson::default();
        let mut delete_files = Vec::new();
        let mut removed_texts: BTreeSet<Vec<u8>> = BTreeSet::new();
        if !req.dry_run {
            // First, so the `excised` observations can name it; what it
            // refused is filled in at the end.
            t.archive.execute(
                "INSERT INTO excision(id, scope, reason, created_at, commit_seq)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![req.id, req.scope.label(), req.reason, now_ms(), t.seq],
            )?;
        }

        if let Some(text) = self.text.take() {
            self.text_scope(
                t,
                &text,
                req,
                done,
                &mut refused,
                &mut delete_files,
                &mut removed_texts,
            )?;
        }

        done.sessions = self.sessions.iter().map(|(_, id)| id.clone()).collect();
        done.messages = self.message_ids.len() as u64;
        done.inputs_removed = self.whole.iter().cloned().collect();
        done.inputs_kept = self.kept.iter().cloned().collect();
        for (input, spans) in &self.spans {
            done.inputs_overwritten
                .insert(input.clone(), spans.len() as u64);
        }

        // Session rows, before the inputs, so a blob a removed block named
        // can be checked against what remains.
        let mut session_blobs = BTreeSet::new();
        for (pk, id) in &self.sessions {
            delete_session(
                t,
                *pk,
                id,
                req.dry_run,
                &mut removed_texts,
                &mut session_blobs,
                done,
            )?;
        }

        for input in &self.whole {
            remove_input(t, input, req, &mut refused, done)?;
        }
        for (input, spans) in &self.spans {
            overwrite(t, input, spans, req, &mut refused, &mut delete_files)?;
        }
        if !self.kept.is_empty() {
            let mut still = BTreeSet::new();
            for input in &self.kept {
                let left = blank_document(
                    t,
                    input,
                    &self.own_texts,
                    req,
                    &mut refused,
                    &mut delete_files,
                    done,
                )?;
                done.texts_left += left;
                if left > 0 {
                    still.insert(input.clone());
                }
            }
            done.inputs_kept = still.into_iter().collect();
        }
        if !self.sessions.is_empty() {
            let ids: Vec<String> = self.sessions.iter().map(|(_, id)| id.clone()).collect();
            done.notes_removed = remove_notes(t.archive, &ids, &self.message_ids, req.dry_run)?;
            done.deliveries_redacted =
                redact_deliveries(t.archive, &ids, &self.own_texts, req.dry_run)?;
        }
        // Blobs nothing refers to any more: the removed inputs' and the
        // removed blocks' (inline images, large outputs).
        let mut candidates: BTreeSet<Vec<u8>> = session_blobs;
        candidates.extend(
            refused
                .blobs
                .iter()
                .filter_map(|h| BlobHash::parse(h).ok().map(|b| b.0.to_vec())),
        );
        if !req.dry_run {
            for hash in &candidates {
                let used: bool = t.archive.query_row(
                    "SELECT EXISTS (SELECT 1 FROM extent WHERE blob = ?1)",
                    [hash],
                    |r| r.get(0),
                )? || t.index.query_row(
                    "SELECT EXISTS (SELECT 1 FROM block WHERE blob_hash = ?1)",
                    [hash],
                    |r| r.get(0),
                )?;
                if used {
                    continue;
                }
                let place: Option<String> = t
                    .archive
                    .query_row("SELECT place FROM blob WHERE hash = ?1", [hash], |r| {
                        r.get(0)
                    })
                    .optional()?;
                let Some(place) = place else { continue };
                t.archive
                    .execute("DELETE FROM blob WHERE hash = ?1", [hash])?;
                done.blobs_deleted += 1;
                if place == "file"
                    && let Ok(arr) = <[u8; 32]>::try_from(hash.as_slice())
                {
                    delete_files.push(BlobHash(arr));
                }
            }
        }

        // Embeddings of texts no row holds any more.
        if let Some(e) = t.embeddings {
            for hash in &removed_texts {
                done.embeddings_deleted += if req.dry_run {
                    e.query_row(
                        "SELECT count(*) FROM embedded WHERE text_hash = ?1",
                        [hash],
                        |r| r.get::<_, i64>(0),
                    )? as u64
                } else {
                    let n = e.execute("DELETE FROM embedded WHERE text_hash = ?1", [hash])? as u64;
                    e.execute("DELETE FROM embed_failure WHERE text_hash = ?1", [hash])?;
                    n
                };
            }
        }
        done.texts = removed_texts.len() as u64;

        refused.locations.sort();
        refused.locations.dedup();
        done.refused_paths = refused
            .locations
            .iter()
            .map(|l| format!("{}:{}", l.machine, l.path))
            .collect();

        if !req.dry_run {
            t.archive.execute(
                "UPDATE excision SET refused_json = ?2, message_ids_json = ?3 WHERE id = ?1",
                params![
                    req.id,
                    serde_json::to_string(&refused)?,
                    serde_json::to_string(&self.message_ids)?,
                ],
            )?;
        }
        Ok((done.clone(), delete_files))
    }

    /// Text scope: overwrite every stored occurrence, remove the index
    /// texts that hold it, and name the inputs to parse again.
    #[allow(clippy::too_many_arguments)]
    fn text_scope(
        &mut self,
        t: &Txs<'_, '_>,
        text: &str,
        req: &ExciseRequest,
        done: &mut Excised,
        refused: &mut RefusedJson,
        delete_files: &mut Vec<BlobHash>,
        removed_texts: &mut BTreeSet<Vec<u8>>,
    ) -> Result<()> {
        let needles = needles(text);
        // Stored bytes: every plain blob an extent uses.
        let blobs: Vec<Vec<u8>> = t
            .archive
            .prepare("SELECT DISTINCT blob FROM extent")?
            .query_map([], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        let mut reparse = BTreeSet::new();
        for hash in blobs {
            let Ok(arr) = <[u8; 32]>::try_from(hash.as_slice()) else {
                continue;
            };
            let old = BlobHash(arr);
            let bytes = crate::blobs::read(t.archive, t.store, &old)?;
            let ranges = find_all(&bytes, &needles);
            if ranges.is_empty() {
                continue;
            }
            let inputs: Vec<String> = t
                .archive
                .prepare("SELECT DISTINCT input FROM extent WHERE blob = ?1")?
                .query_map([&hash], |r| r.get(0))?
                .collect::<rusqlite::Result<_>>()?;
            for i in &inputs {
                *done.inputs_overwritten.entry(i.clone()).or_default() += ranges.len() as u64;
                reparse.insert(i.clone());
            }
            if req.dry_run {
                continue;
            }
            if let Some(gone) = replace_blob(t, &old, bytes, &ranges)? {
                refused.blobs.push(hex(&gone.hash));
                delete_files.extend(gone.file);
            }
        }
        // The index is rewritten in place, not parsed again: a live input is
        // read from the file on disk, which still holds the text.
        redact_index(t.index, text, &needles, req.dry_run, removed_texts)?;
        // Files on disk that held it: Polylogue never edits them; the
        // operator scrubs them.
        let mut on_disk = BTreeSet::new();
        for input in &reparse {
            let paths: Vec<(String, String)> = t
                .archive
                .prepare(
                    "SELECT DISTINCT machine, path FROM observation WHERE input = ?1
                     AND path NOT LIKE 'zip:%' AND path NOT LIKE 'copy:%'",
                )?
                .query_map([input], |r| Ok((r.get(0)?, r.get(1)?)))?
                .collect::<rusqlite::Result<_>>()?;
            on_disk.extend(paths.into_iter().map(|(m, p)| format!("{m}:{p}")));
        }
        done.on_disk = on_disk.into_iter().collect();
        Ok(())
    }
}

/// Stored blobs (input bytes, inline or file) that contain `needle`, as
/// written or as JSON escapes it: how an excision is checked afterwards.
pub fn stored_blobs_containing(archive: &crate::Archive, needle: &str) -> Result<Vec<BlobHash>> {
    let conn = crate::files::open_read(&archive.archive_db())?;
    let store = hashed_blobs::Store::open(archive.root.clone(), vec![], 0o600)?;
    let needles = needles(needle);
    let hashes: Vec<Vec<u8>> = conn
        .prepare("SELECT hash FROM blob WHERE state = 'present'")?
        .query_map([], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    let mut out = Vec::new();
    for h in hashes {
        let Ok(arr) = <[u8; 32]>::try_from(h.as_slice()) else {
            continue;
        };
        let hash = BlobHash(arr);
        let bytes = crate::blobs::read(&conn, &store, &hash)?;
        if !find_all(&bytes, &needles).is_empty() {
            out.push(hash);
        }
    }
    Ok(out)
}

/// The same number of spaces as `n` has bytes.
fn blank(n: &[u8]) -> String {
    " ".repeat(n.len())
}

fn replace_all(s: &str, needles: &[Vec<u8>]) -> String {
    let mut out = s.to_owned();
    for n in needles {
        if let Ok(n) = std::str::from_utf8(n) {
            out = out.replace(n, &blank(n.as_bytes()));
        }
    }
    out
}

/// Text scope in `index.db`: every value holding the text gets spaces in
/// its place: texts (with their search entries), block bodies, tool
/// inputs and outputs, tool-call commands and paths, event payloads,
/// titles and provider attributes. Texts that changed are recorded so
/// their embeddings go.
fn redact_index(
    i: &Transaction<'_>,
    text: &str,
    needles: &[Vec<u8>],
    dry_run: bool,
    removed_texts: &mut BTreeSet<Vec<u8>>,
) -> Result<()> {
    let texts: Vec<(i64, Vec<u8>, Vec<u8>, i64)> = i
        .prepare("SELECT id, hash, body, indexed FROM text")?
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))?
        .collect::<rusqlite::Result<_>>()?;
    for (id, hash, body, indexed) in texts {
        let Ok(body) = unpack_text(&body) else {
            continue;
        };
        if !body.contains(text) {
            continue;
        }
        removed_texts.insert(hash);
        if dry_run {
            continue;
        }
        let new_body = body.replace(text, &blank(text.as_bytes()));
        let new_hash = blake3::hash(new_body.as_bytes());
        let existing: Option<i64> = i
            .query_row(
                "SELECT id FROM text WHERE hash = ?1",
                [&new_hash.as_bytes()[..]],
                |r| r.get(0),
            )
            .optional()?;
        let new_id = match existing {
            Some(n) => n,
            None => {
                i.execute(
                    "INSERT INTO text(hash, body, indexed) VALUES (?1, ?2, 0)",
                    params![
                        &new_hash.as_bytes()[..],
                        zstd::bulk::compress(new_body.as_bytes(), 1)?
                    ],
                )?;
                i.last_insert_rowid()
            }
        };
        let folded = polylogue_model::text::fold(&new_body);
        for (bit, table) in [(1, "text_fts"), (2, "output_fts")] {
            if indexed & bit != 0 {
                i.execute(
                    &format!("INSERT INTO {table}(rowid, body) VALUES (?1, ?2)"),
                    params![new_id, folded],
                )?;
            }
        }
        i.execute(
            "UPDATE text SET indexed = indexed | ?2 WHERE id = ?1",
            params![new_id, indexed],
        )?;
        i.execute(
            "UPDATE block SET text_id = ?2 WHERE text_id = ?1",
            params![id, new_id],
        )?;
        remove_text(i, id, indexed)?;
    }
    if dry_run {
        return Ok(());
    }
    // Compressed JSON columns.
    for (table, key, col) in [("block", "pk", "body"), ("event", "pk", "data")] {
        let rows: Vec<(i64, Vec<u8>)> = i
            .prepare(&format!("SELECT {key}, {col} FROM {table}"))?
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<rusqlite::Result<_>>()?;
        for (pk, packed) in rows {
            let Ok(json) = zstd::stream::decode_all(&packed[..]) else {
                continue;
            };
            let Ok(s) = String::from_utf8(json) else {
                continue;
            };
            let new = replace_all(&s, needles);
            if new != s {
                i.execute(
                    &format!("UPDATE {table} SET {col} = ?2 WHERE {key} = ?1"),
                    params![pk, zstd::bulk::compress(new.as_bytes(), 1)?],
                )?;
            }
        }
    }
    // Plain text columns.
    let calls: Vec<(i64, Option<String>, Option<String>)> = i
        .prepare(
            "SELECT pk, command, path FROM tool_call WHERE instr(command, ?1) OR instr(path, ?1)",
        )?
        .query_map([text], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
        .collect::<rusqlite::Result<_>>()?;
    for (pk, command, path) in calls {
        i.execute(
            "INSERT INTO tool_call_trigram(tool_call_trigram, rowid, command, path)
             VALUES ('delete', ?1, ?2, ?3)",
            params![pk, command, path],
        )?;
        let fix = |v: Option<String>| v.map(|s| replace_all(&s, needles));
        let (command, path) = (fix(command), fix(path));
        i.execute(
            "UPDATE tool_call SET command = ?2, path = ?3, command_norm = NULL WHERE pk = ?1",
            params![pk, command, path],
        )?;
        i.execute(
            "INSERT INTO tool_call_trigram(rowid, command, path) VALUES (?1, ?2, ?3)",
            params![pk, command, path],
        )?;
    }
    for (table, col) in [
        ("block", "input_json"),
        ("block", "data_json"),
        ("message", "attrs_json"),
        ("session", "title"),
        ("session", "derived_title"),
        ("session", "attrs_json"),
    ] {
        for n in needles {
            let Ok(n) = std::str::from_utf8(n) else {
                continue;
            };
            i.execute(
                &format!(
                    "UPDATE {table} SET {col} = replace({col}, ?1, ?2) WHERE instr({col}, ?1)"
                ),
                params![n, blank(n.as_bytes())],
            )?;
        }
    }
    // Derived rows may quote the text (work-graph titles, derived titles):
    // every session is derived again from the blanked records.
    i.execute("DELETE FROM derive_state", [])?;
    Ok(())
}

/// The forms a string takes in stored bytes: as written, and as JSON
/// escapes it when it holds quotes, backslashes or control characters.
fn needles(text: &str) -> Vec<Vec<u8>> {
    let mut out = vec![text.as_bytes().to_vec()];
    if let Ok(json) = serde_json::to_string(text) {
        let inner = json.as_bytes()[1..json.len() - 1].to_vec();
        if inner != out[0] {
            out.push(inner);
        }
    }
    out
}

fn find_all(hay: &[u8], needles: &[Vec<u8>]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for n in needles {
        if n.is_empty() || n.len() > hay.len() {
            continue;
        }
        let mut i = 0;
        while let Some(p) = hay[i..].windows(n.len()).position(|w| w == n.as_slice()) {
            out.push((i + p, i + p + n.len()));
            i += p + n.len();
            if i + n.len() > hay.len() {
                break;
            }
        }
    }
    out.sort();
    out
}

/// A blob replaced by an overwritten copy: the old hash, and its file to
/// delete when it was one.
struct Gone {
    hash: [u8; 32],
    file: Option<BlobHash>,
}

/// Overwrites `ranges` of a blob's bytes with spaces (line breaks kept),
/// stores the result, points every extent at it and drops the old row.
fn replace_blob(
    t: &Txs<'_, '_>,
    old: &BlobHash,
    mut bytes: Vec<u8>,
    ranges: &[(usize, usize)],
) -> Result<Option<Gone>> {
    let len = bytes.len();
    for &(s, e) in ranges {
        for b in &mut bytes[s.min(len)..e.min(len)] {
            if *b != b'\n' {
                *b = b' ';
            }
        }
    }
    let piece = Piece::prepare(t.store, &bytes)?;
    if piece.hash == *old {
        return Ok(None);
    }
    let (place, stored_size, data) = match &piece.stored {
        Stored::Inline(z) => ("inline", z.len() as i64, Some(z.as_slice())),
        Stored::File { stored_size } => ("file", *stored_size as i64, None),
    };
    t.archive.execute(
        "INSERT OR IGNORE INTO blob(hash, size, stored_size, place, encoding, state, data)
         VALUES (?1, ?2, ?3, ?4, 'plain', 'present', ?5)",
        params![
            &piece.hash.0[..],
            piece.size as i64,
            stored_size,
            place,
            data
        ],
    )?;
    t.archive.execute(
        "UPDATE extent SET blob = ?2 WHERE blob = ?1",
        params![&old.0[..], &piece.hash.0[..]],
    )?;
    let old_place: Option<String> = t
        .archive
        .query_row(
            "SELECT place FROM blob WHERE hash = ?1",
            [&old.0[..]],
            |r| r.get(0),
        )
        .optional()?;
    t.archive
        .execute("DELETE FROM blob WHERE hash = ?1", [&old.0[..]])?;
    Ok(Some(Gone {
        hash: old.0,
        file: (old_place.as_deref() == Some("file")).then_some(*old),
    }))
}

/// Overwrites a shared input's spans in every blob its chains use.
fn overwrite(
    t: &Txs<'_, '_>,
    input: &str,
    spans: &[Span],
    req: &ExciseRequest,
    refused: &mut RefusedJson,
    delete_files: &mut Vec<BlobHash>,
) -> Result<()> {
    // Blob ranges by blob.
    let mut by_blob: BTreeMap<Vec<u8>, Vec<(usize, usize)>> = BTreeMap::new();
    let mut ext = t.archive.prepare(
        "SELECT offset, length, blob, blob_offset FROM extent
         WHERE input = ?1 AND chain = ?2 AND offset < ?4 AND offset + length > ?3",
    )?;
    for s in spans {
        let rows = ext.query_map(params![input, s.chain, s.start, s.end], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, i64>(1)?,
                r.get::<_, Vec<u8>>(2)?,
                r.get::<_, i64>(3)?,
            ))
        })?;
        for row in rows {
            let (offset, length, blob, blob_offset) = row?;
            let from = s.start.max(offset) - offset + blob_offset;
            let to = s.end.min(offset + length) - offset + blob_offset;
            by_blob
                .entry(blob)
                .or_default()
                .push((from as usize, to as usize));
        }
    }
    if req.dry_run {
        return Ok(());
    }
    let mut spans_json = Vec::new();
    for (hash, ranges) in by_blob {
        let Ok(arr) = <[u8; 32]>::try_from(hash.as_slice()) else {
            continue;
        };
        let old = BlobHash(arr);
        let bytes = crate::blobs::read(t.archive, t.store, &old)?;
        for &(s, e) in &ranges {
            spans_json.push(serde_json::json!([hex(&hash), s, e - s]));
        }
        if let Some(gone) = replace_blob(t, &old, bytes, &ranges)? {
            refused.blobs.push(hex(&gone.hash));
            // The old file goes with the other deletions, after commit.
            delete_files.extend(gone.file);
        }
    }
    insert_excised_observations(t, input, req, Some(&serde_json::Value::Array(spans_json)))?;
    Ok(())
}

/// Shortest session text blanked in a shared whole document: a shorter
/// complete string value ("ok", "user") could be another record's value
/// or the document's own structure.
const MIN_BLANKED: usize = 12;

/// The forms a text takes as a complete JSON string value: as `serde_json`
/// writes it, and with non-ASCII escaped as `\uXXXX` (Python's default).
fn quoted(text: &str) -> Vec<Vec<u8>> {
    let Ok(plain) = serde_json::to_string(text) else {
        return Vec::new();
    };
    let mut ascii = String::with_capacity(plain.len());
    for ch in plain.chars() {
        if ch.is_ascii() {
            ascii.push(ch);
        } else {
            let mut units = [0u16; 2];
            for u in ch.encode_utf16(&mut units) {
                ascii.push_str(&format!("\\u{u:04x}"));
            }
        }
    }
    let mut out = vec![plain.into_bytes()];
    if ascii.as_bytes() != out[0].as_slice() {
        out.push(ascii.into_bytes());
    }
    out
}

/// A shared whole document: blanks the session's own texts where they
/// stand as complete JSON string values, quotes kept. Returns how many of
/// those texts are still in it (too short, or not found whole).
fn blank_document(
    t: &Txs<'_, '_>,
    input: &str,
    texts: &BTreeSet<String>,
    req: &ExciseRequest,
    refused: &mut RefusedJson,
    delete_files: &mut Vec<BlobHash>,
    done: &mut Excised,
) -> Result<u64> {
    let blobs: Vec<Vec<u8>> = t
        .archive
        .prepare("SELECT DISTINCT blob FROM extent WHERE input = ?1")?
        .query_map([input], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    let mut left = texts.iter().filter(|s| s.len() < MIN_BLANKED).count() as u64;
    let wanted: Vec<Vec<Vec<u8>>> = texts
        .iter()
        .filter(|s| s.len() >= MIN_BLANKED)
        .map(|s| quoted(s))
        .collect();
    let mut found = vec![false; wanted.len()];
    let mut spans_json = Vec::new();
    for hash in blobs {
        let Ok(arr) = <[u8; 32]>::try_from(hash.as_slice()) else {
            continue;
        };
        let old = BlobHash(arr);
        let bytes = crate::blobs::read(t.archive, t.store, &old)?;
        let mut ranges = Vec::new();
        for (n, forms) in wanted.iter().enumerate() {
            let hits = find_all(&bytes, forms);
            if !hits.is_empty() {
                found[n] = true;
            }
            // The string's inside only: the quotes keep the document valid.
            ranges.extend(hits.into_iter().map(|(s, e)| (s + 1, e - 1)));
        }
        if ranges.is_empty() {
            continue;
        }
        ranges.sort();
        *done.inputs_overwritten.entry(input.to_owned()).or_default() += ranges.len() as u64;
        if req.dry_run {
            continue;
        }
        for &(s, e) in &ranges {
            spans_json.push(serde_json::json!([hex(&hash), s, e - s]));
        }
        if let Some(gone) = replace_blob(t, &old, bytes, &ranges)? {
            refused.blobs.push(hex(&gone.hash));
            delete_files.extend(gone.file);
        }
    }
    left += found.iter().filter(|f| !**f).count() as u64;
    if !req.dry_run && !spans_json.is_empty() {
        insert_excised_observations(t, input, req, Some(&serde_json::Value::Array(spans_json)))?;
    }
    Ok(left)
}

/// Notes about the removed sessions: targets naming a session or one of its
/// messages go, and a note left with no target goes with its relations.
fn remove_notes(
    a: &Transaction<'_>,
    sessions: &[String],
    messages: &[String],
    dry_run: bool,
) -> Result<u64> {
    let refs: HashSet<String> = sessions
        .iter()
        .map(|s| format!("session:{s}"))
        .chain(messages.iter().map(|m| format!("message:{m}")))
        .collect();
    // Tool calls are named `tool:<session>#<call>`.
    let tools: Vec<String> = sessions.iter().map(|s| format!("tool:{s}#")).collect();
    let about = |r: &str| refs.contains(r) || tools.iter().any(|p| r.starts_with(p.as_str()));
    let mut by_note: BTreeMap<String, (u64, Vec<String>)> = BTreeMap::new();
    let mut stmt = a.prepare("SELECT note_id, ref FROM note_target")?;
    let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?;
    for row in rows {
        let (note, r) = row?;
        let e = by_note.entry(note).or_default();
        e.0 += 1;
        if about(&r) {
            e.1.push(r);
        }
    }
    drop(stmt);
    let mut removed = 0;
    for (note, (all, hit)) in by_note {
        if hit.is_empty() {
            continue;
        }
        let whole = hit.len() as u64 == all;
        if whole {
            removed += 1;
        }
        if dry_run {
            continue;
        }
        for r in &hit {
            a.execute(
                "DELETE FROM note_target WHERE note_id = ?1 AND ref = ?2",
                params![note, r],
            )?;
        }
        if whole {
            a.execute("DELETE FROM note_relation WHERE note_id = ?1", [&note])?;
            a.execute("DELETE FROM note WHERE id = ?1", [&note])?;
        }
    }
    Ok(removed)
}

/// Delivered briefs that named a removed session: the session's own texts
/// in them are blanked, and the content hash follows the new text.
fn redact_deliveries(
    a: &Transaction<'_>,
    sessions: &[String],
    texts: &BTreeSet<String>,
    dry_run: bool,
) -> Result<u64> {
    let rows: Vec<(String, Vec<u8>, String)> = a
        .prepare("SELECT id, body, refs_json FROM delivery")?
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
        .collect::<rusqlite::Result<_>>()?;
    let mut n = 0;
    for (id, body, refs) in rows {
        if !sessions.iter().any(|s| refs.contains(s.as_str())) {
            continue;
        }
        let Ok(text) = unpack_text(&body) else {
            continue;
        };
        let mut new = text.clone();
        for s in texts.iter().filter(|s| s.len() >= MIN_BLANKED) {
            new = new.replace(s.as_str(), &blank(s.as_bytes()));
        }
        if new == text {
            continue;
        }
        n += 1;
        if dry_run {
            continue;
        }
        a.execute(
            "UPDATE delivery SET body = ?2, content_hash = ?3 WHERE id = ?1",
            params![
                id,
                zstd::bulk::compress(new.as_bytes(), 1)?,
                &blake3::hash(new.as_bytes()).as_bytes()[..]
            ],
        )?;
    }
    Ok(n)
}

/// Removes an input whole: its revisions and extents, and records its
/// locations so the loop never collects them again.
fn remove_input(
    t: &Txs<'_, '_>,
    input: &str,
    req: &ExciseRequest,
    refused: &mut RefusedJson,
    done: &mut Excised,
) -> Result<()> {
    let locations: Vec<Location> = t
        .archive
        .prepare(
            "SELECT DISTINCT machine, path, device, inode FROM observation
             WHERE input = ?1 AND path NOT LIKE 'zip:%' AND path NOT LIKE 'copy:%'",
        )?
        .query_map([input], |r| {
            Ok(Location {
                machine: r.get(0)?,
                path: r.get(1)?,
                device: r.get(2)?,
                inode: r.get(3)?,
            })
        })?
        .collect::<rusqlite::Result<_>>()?;
    refused.locations.extend(locations);
    let revisions: Vec<Option<Vec<u8>>> = t
        .archive
        .prepare("SELECT blake3 FROM revision WHERE input = ?1")?
        .query_map([input], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    refused
        .revisions
        .extend(revisions.into_iter().flatten().map(|h| hex(&h)));
    let blobs: Vec<Vec<u8>> = t
        .archive
        .prepare("SELECT DISTINCT blob FROM extent WHERE input = ?1")?
        .query_map([input], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    refused.blobs.extend(blobs.iter().map(|b| hex(b)));
    if req.dry_run {
        return Ok(());
    }
    insert_excised_observations(t, input, req, None)?;
    t.archive
        .execute("DELETE FROM extent WHERE input = ?1", [input])?;
    t.archive
        .execute("DELETE FROM revision WHERE input = ?1", [input])?;
    if let Some(pk) = input_pk(t.index, input)? {
        // Rows of other sessions from this input (an input scope's
        // shared sessions keep their other inputs' rows).
        for table in ["parse_state", "unrecognized", "parsed", "message_source"] {
            t.index
                .execute(&format!("DELETE FROM {table} WHERE input_pk = ?1"), [pk])?;
        }
        t.index
            .execute("UPDATE input_ref SET excised = 1 WHERE pk = ?1", [pk])?;
    }
    let _ = done;
    Ok(())
}

/// One `excised` observation per location of the input, naming the
/// excision.
fn insert_excised_observations(
    t: &Txs<'_, '_>,
    input: &str,
    req: &ExciseRequest,
    spans: Option<&serde_json::Value>,
) -> Result<()> {
    let at = now_ms();
    let rows: Vec<(String, String)> = t
        .archive
        .prepare("SELECT DISTINCT machine, path FROM observation WHERE input = ?1")?
        .query_map([input], |r| Ok((r.get(0)?, r.get(1)?)))?
        .collect::<rusqlite::Result<_>>()?;
    for (n, (machine, path)) in rows.into_iter().enumerate() {
        t.archive.execute(
            "INSERT INTO observation(id, input, machine, path, class, excision, spans_json,
                 polylogue_version, seen_at)
             VALUES (?1, ?2, ?3, ?4, 'excised', ?5, ?6, ?7, ?8)",
            params![
                format!("{at:012x}-{:08x}-{n}-{input}", t.seq),
                input,
                machine,
                path,
                req.id,
                spans.map(ToString::to_string),
                env!("CARGO_PKG_VERSION"),
                at,
            ],
        )?;
    }
    Ok(())
}

/// Removes one session's rows from `index.db`.
fn delete_session(
    t: &Txs<'_, '_>,
    pk: i64,
    id: &str,
    dry_run: bool,
    removed_texts: &mut BTreeSet<Vec<u8>>,
    blobs: &mut BTreeSet<Vec<u8>>,
    done: &mut Excised,
) -> Result<()> {
    let i = t.index;
    let text_ids: Vec<i64> = i
        .prepare(
            "SELECT DISTINCT text_id FROM block WHERE session_pk = ?1 AND text_id IS NOT NULL",
        )?
        .query_map([pk], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    let block_blobs: Vec<Vec<u8>> = i
        .prepare(
            "SELECT DISTINCT blob_hash FROM block WHERE session_pk = ?1 AND blob_hash IS NOT NULL",
        )?
        .query_map([pk], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    blobs.extend(block_blobs);
    if dry_run {
        for tid in text_ids {
            let shared: bool = i.query_row(
                "SELECT EXISTS (SELECT 1 FROM block WHERE text_id = ?1 AND session_pk <> ?2)",
                params![tid, pk],
                |r| r.get(0),
            )?;
            if !shared {
                let hash: Vec<u8> =
                    i.query_row("SELECT hash FROM text WHERE id = ?1", [tid], |r| r.get(0))?;
                removed_texts.insert(hash);
            }
        }
        let _ = done;
        return Ok(());
    }
    // Trigram entries before their tool calls.
    let calls: Vec<(i64, Option<String>, Option<String>)> = i
        .prepare("SELECT pk, command, path FROM tool_call WHERE session_pk = ?1")?
        .query_map([pk], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
        .collect::<rusqlite::Result<_>>()?;
    for (cpk, command, path) in &calls {
        i.execute(
            "INSERT INTO tool_call_trigram(tool_call_trigram, rowid, command, path)
             VALUES ('delete', ?1, ?2, ?3)",
            params![cpk, command, path],
        )?;
    }
    // Every table that names the session, its messages or its tool calls,
    // whichever component added it.
    let tables = columns(i)?;
    for (table, cols) in &tables {
        if table == "session" || table == "message" || table == "tool_call" {
            continue;
        }
        if cols.contains("tool_call_pk") {
            i.execute(
                &format!(
                    "DELETE FROM {table} WHERE tool_call_pk IN
                     (SELECT pk FROM tool_call WHERE session_pk = ?1)"
                ),
                [pk],
            )?;
        }
    }
    for (table, cols) in &tables {
        if table == "session" || table == "message" || table == "tool_call" {
            continue;
        }
        if cols.contains("message_pk") && !cols.contains("session_pk") {
            i.execute(
                &format!(
                    "DELETE FROM {table} WHERE message_pk IN
                     (SELECT pk FROM message WHERE session_pk = ?1)"
                ),
                [pk],
            )?;
        }
    }
    i.execute("DELETE FROM tool_call WHERE session_pk = ?1", [pk])?;
    for (table, cols) in &tables {
        if table == "session" || table == "message" {
            continue;
        }
        if cols.contains("session_pk") {
            i.execute(&format!("DELETE FROM {table} WHERE session_pk = ?1"), [pk])?;
        }
    }
    i.execute(
        "DELETE FROM link WHERE parent_id = ?1 OR child_id = ?1",
        [id],
    )?;
    i.execute("DELETE FROM message WHERE session_pk = ?1", [pk])?;
    i.execute("DELETE FROM session WHERE pk = ?1", [pk])?;
    i.execute(
        "INSERT INTO removed_session(session_id, changed) VALUES (?1, ?2)
         ON CONFLICT(session_id) DO UPDATE SET changed = excluded.changed",
        params![id, t.counter],
    )?;
    // Texts no remaining block holds.
    for tid in text_ids {
        let used: bool = i.query_row(
            "SELECT EXISTS (SELECT 1 FROM block WHERE text_id = ?1)",
            [tid],
            |r| r.get(0),
        )?;
        if used {
            continue;
        }
        let (hash, indexed): (Vec<u8>, i64) =
            i.query_row("SELECT hash, indexed FROM text WHERE id = ?1", [tid], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })?;
        remove_text(i, tid, indexed)?;
        removed_texts.insert(hash);
    }
    Ok(())
}

fn remove_text(i: &Transaction<'_>, id: i64, indexed: i64) -> Result<()> {
    if indexed & 1 != 0 {
        i.execute("DELETE FROM text_fts WHERE rowid = ?1", [id])?;
    }
    if indexed & 2 != 0 {
        i.execute("DELETE FROM output_fts WHERE rowid = ?1", [id])?;
    }
    i.execute("DELETE FROM search_pending WHERE text_id = ?1", [id])?;
    i.execute("DELETE FROM text WHERE id = ?1", [id])?;
    Ok(())
}

/// Ordinary tables in `index.db` and their column names.
fn columns(i: &Transaction<'_>) -> Result<Vec<(String, HashSet<String>)>> {
    let names: Vec<String> = i
        .prepare(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND sql NOT LIKE 'CREATE VIRTUAL%'
             AND name NOT LIKE 'sqlite_%' AND name NOT LIKE '%_fts%' AND name NOT LIKE 'fts_%'
             AND name NOT LIKE 'tool_call_trigram%'",
        )?
        .query_map([], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    let mut out = Vec::new();
    for name in names {
        let cols: HashSet<String> = i
            .prepare(&format!("SELECT name FROM pragma_table_info('{name}')"))?
            .query_map([], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        out.push((name, cols));
    }
    Ok(out)
}

fn session_pk(i: &Transaction<'_>, id: &str) -> Result<Option<i64>> {
    Ok(
        i.query_row("SELECT pk FROM session WHERE id = ?1", [id], |r| r.get(0))
            .optional()?,
    )
}

fn input_pk(i: &Transaction<'_>, id: &str) -> Result<Option<i64>> {
    Ok(
        i.query_row("SELECT pk FROM input_ref WHERE id = ?1", [id], |r| r.get(0))
            .optional()?,
    )
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as i64)
}

#[cfg(test)]
mod tests;
