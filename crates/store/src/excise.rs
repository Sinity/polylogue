//! Excision: permanent removal from every file and derived product
//! (architecture 5.6). The `excision` row keeps the
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
//!   written and as JSON escapes it, overwritten with spaces. The sessions
//!   built from those records lose every index row, and their inputs are
//!   parsed again from the overwritten bytes, so no column list decides
//!   what is cleaned.
//!
//! An excised revision keeps its BLAKE3 (the hash of the bytes as first
//! observed); its blobs are what it is verified by afterwards.

use crate::blobs::{Piece, Stored};
use crate::encoding::unpack_text;
use crate::{Result, StoreError};
use polylogue_model::{BlobHash, SessionId};
use rusqlite::{OptionalExtension, Transaction, params};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashSet};

mod authored;
mod index_rows;
mod stored_bytes;
mod stored_text;
mod text_matches;

use index_rows::*;
use stored_bytes::*;
pub use stored_bytes::{index_columns_containing, stored_blobs_containing};

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
    /// Text scope: authored rows whose payloads were overwritten.
    #[serde(default)]
    pub notes_redacted: u64,
    #[serde(default)]
    pub variants_redacted: u64,
    #[serde(default)]
    pub kept_results_redacted: u64,
    /// Delivered briefs whose text held the session's texts, blanked.
    #[serde(default)]
    pub deliveries_redacted: u64,
    pub blobs_deleted: u64,
    pub embeddings_deleted: u64,
    /// Paths the loop no longer collects, as `machine:path`.
    pub refused_paths: Vec<String>,
    /// Text scope: inputs whose rows went, to be parsed again from their
    /// overwritten stored bytes ([`crate::excise`]); the engine does so
    /// before the excision returns.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inputs_reparsed: Vec<String>,
    /// Text scope: provider locations (`machine:path`) whose retained
    /// history held a match. Current presence and content are not checked.
    /// Polylogue never edits them; a later full read may collect the text
    /// again until the original file is scrubbed.
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    publication: Option<crate::sinex_rows::ExcisionPublication>,
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
pub struct Transactions<'a, 'b> {
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
pub fn apply(t: &Transactions<'_, '_>, req: &ExciseRequest) -> Result<(Excised, Vec<BlobHash>)> {
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
    fn session(&mut self, t: &Transactions<'_, '_>, pk: i64, id: &str) -> Result<()> {
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
                     UNION SELECT es.input_pk FROM event_source es JOIN event e ON e.id=es.event_id WHERE e.session_pk = ?1)",
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
                     UNION SELECT es.chain,es.source_start,es.source_end FROM event_source es
                     JOIN event e ON e.id=es.event_id
                     WHERE e.session_pk = ?2 AND es.input_pk = ?1
                           AND es.source_end > es.source_start",
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
        t: &Transactions<'_, '_>,
        req: &ExciseRequest,
        done: &mut Excised,
    ) -> Result<(Excised, Vec<BlobHash>)> {
        let mut text_matches = self
            .text
            .as_ref()
            .map(|text| stored_text::find(t, text))
            .transpose()?;
        let capture = if req.dry_run {
            None
        } else {
            Some(crate::sinex_rows::Capture::start(t.archive, &req.id)?)
        };
        let mut refused = RefusedJson::default();
        let mut delete_files = Vec::new();
        let mut removed_texts: BTreeSet<Vec<u8>> = BTreeSet::new();
        // Blobs the removed blocks named (inline images, large outputs).
        let mut session_blobs = BTreeSet::new();
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
                text_matches.as_mut().expect("text scope preflight"),
                req,
                done,
                &mut refused,
                &mut delete_files,
                &mut removed_texts,
                &mut session_blobs,
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
        for (pk, id) in &self.sessions {
            delete_session(
                t,
                *pk,
                id,
                req.dry_run,
                &mut removed_texts,
                &mut session_blobs,
            )?;
        }

        for input in &self.whole {
            remove_input(t, input, req, &mut refused, done, &mut removed_texts)?;
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
                    "SELECT EXISTS (SELECT 1 FROM block WHERE blob_hash = ?1) OR EXISTS (SELECT 1 FROM version_block WHERE blob_hash=?1)",
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

        if let Some(capture) = capture {
            let publication = capture.finish(t.archive, t.store)?;
            delete_files.extend(crate::sinex_rows::retired_local(t.archive, &publication)?);
            refused.publication = Some(publication);
        }
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
        t: &Transactions<'_, '_>,
        text: &str,
        matches: &mut stored_text::StoredText,
        req: &ExciseRequest,
        done: &mut Excised,
        refused: &mut RefusedJson,
        _delete_files: &mut Vec<BlobHash>,
        removed_texts: &mut BTreeSet<Vec<u8>>,
        session_blobs: &mut BTreeSet<Vec<u8>>,
    ) -> Result<()> {
        let reparse = matches.inputs();
        done.inputs_overwritten = matches.counts();
        if !req.dry_run {
            matches.overwrite(t, req, refused)?;
        }
        authored::redact(t.archive, text, req.dry_run, done)?;
        // The rows parsed from the text go; the caller parses their inputs
        // again from the overwritten stored bytes.
        done.inputs_reparsed =
            clear_for_reparse(t, text, &reparse, req.dry_run, removed_texts, session_blobs)?;
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
