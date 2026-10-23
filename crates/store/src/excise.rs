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
//!   offsets. Shared JSON records also remove the necessary array separators;
//!   unsupported record boundaries refuse before committing. Notes about
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
use std::collections::{BTreeMap, BTreeSet};

mod authored;
pub(crate) mod blob_cleanup;
mod hashes;
mod index_rows;
mod messages;
mod notes;
mod record_ranges;
pub(crate) mod recovery;
mod refused;
pub use refused::Refused;
pub(crate) use refused::RefusedIds;
mod session_texts;
mod sessions;
mod spans;
mod stored_bytes;
mod stored_text;
mod text_matches;
pub(crate) mod vector_cleanup;
pub(crate) mod vector_policy;

use hashes::RemovalHashes;
use index_rows::*;
use messages::SelectedMessages;
use session_texts::SessionTexts;
use sessions::SelectedSessions;
use spans::Spans;
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

impl crate::Reader<'_> {
    /// Paths (`(machine, path)`) the loop must never collect again, from every
    /// excision's refused locations. Invalid records refuse the read.
    pub fn refused_paths(&self) -> Result<Vec<(String, String)>> {
        let mut stmt = self
            .conn()?
            .prepare("SELECT refused_json FROM a.excision")?;
        let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
        let mut out = Vec::new();
        for row in rows {
            let refused = RefusedJson::decode(&row?)?;
            out.extend(refused.locations.into_iter().map(|l| (l.machine, l.path)));
        }
        Ok(out)
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct RefusedJson {
    /// Exact session selection, including input-scoped removals. Older
    /// records may lack this list; their IDs cannot be inferred from messages.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "refused::session_selection"
    )]
    sessions: Option<Vec<SessionId>>,
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

impl RefusedJson {
    fn decode(value: &str) -> Result<Self> {
        if refused::legacy_empty(value) {
            return Ok(Self::default());
        }
        if !value.trim_start().starts_with('{') {
            return Err(StoreError::Invalid(
                "invalid excision refusal payload".into(),
            ));
        }
        Ok(serde_json::from_str(value)?)
    }
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
/// deleted blobs are recorded as local cleanup intent before commit.
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

/// Runs an excision and records local file cleanup with its canonical rows.
/// The writer replays that cleanup after all database transactions commit.
pub fn apply(t: &Transactions<'_, '_>, req: &ExciseRequest) -> Result<Excised> {
    let mut done = Excised {
        dry_run: req.dry_run,
        ..Excised::default()
    };
    let mut plan = Plan::new(t.index)?;
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
                let mut statement = t.index.prepare(
                    "SELECT s.pk,s.id FROM session s WHERE s.pk IN
                        (SELECT session_pk FROM parsed WHERE input_pk=?1)
                     AND NOT EXISTS (SELECT 1 FROM parsed p WHERE p.session_pk=s.pk AND p.input_pk<>?1)")?;
                let sessions = statement
                    .query_map([pk], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))?;
                for row in sessions {
                    let (spk, sid) = row?;
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

struct Plan<'a> {
    /// Sessions to remove: pk and id.
    sessions: SelectedSessions<'a>,
    /// Inputs to remove whole.
    whole: BTreeSet<String>,
    /// Inputs shared with other sessions: the spans to overwrite.
    spans: Spans,
    /// Shared inputs whose records have no byte ranges.
    kept: BTreeSet<String>,
    /// Message IDs of the removed sessions, for the writer's refusal.
    message_ids: SelectedMessages,
    /// Texts only the removed sessions hold (bodies and titles), for
    /// blanking shared whole documents and delivered briefs.
    own_texts: SessionTexts,
    text: Option<String>,
}

impl<'a> Plan<'a> {
    fn new(index: &'a rusqlite::Connection) -> Result<Self> {
        Ok(Self {
            sessions: SelectedSessions::new(index)?,
            whole: BTreeSet::new(),
            spans: Spans::default(),
            kept: BTreeSet::new(),
            message_ids: SelectedMessages::default(),
            own_texts: SessionTexts::default(),
            text: None,
        })
    }

    fn session(&mut self, t: &Transactions<'_, '_>, pk: i64, id: &str) -> Result<()> {
        self.sessions.insert(pk, id)?;
        let mut messages = t
            .index
            .prepare("SELECT id FROM message WHERE session_pk=?1")?;
        for id in messages.query_map([pk], |r| r.get::<_, String>(0))? {
            self.message_ids.insert(&id?)?;
        }
        // Every input the session's rows came from.
        let mut inputs = t.index.prepare(
            "SELECT pk,id FROM input_ref WHERE pk IN (
                SELECT input_pk FROM parsed WHERE session_pk=?1
                UNION SELECT s.input_pk FROM message_source s JOIN message m ON m.pk=s.message_pk WHERE m.session_pk=?1
                UNION SELECT es.input_pk FROM event_source es JOIN event e ON e.id=es.event_id WHERE e.session_pk=?1)")?;
        let inputs =
            inputs.query_map([pk], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))?;
        for row in inputs {
            let (ipk, iid) = row?;
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
            let mut statement = t.index.prepare(
                "SELECT s.chain,s.source_start,s.source_end FROM message_source s JOIN message m ON m.pk=s.message_pk
                 WHERE m.session_pk=?2 AND s.input_pk=?1 AND s.source_end>s.source_start
                 UNION SELECT es.chain,es.source_start,es.source_end FROM event_source es JOIN event e ON e.id=es.event_id
                 WHERE e.session_pk=?2 AND es.input_pk=?1 AND es.source_end>es.source_start")?;
            let spans = statement.query_map(params![ipk, pk], |r| {
                Ok(Span {
                    chain: r.get(0)?,
                    start: r.get(1)?,
                    end: r.get(2)?,
                })
            })?;
            let mut found = false;
            for span in spans {
                self.spans.insert(&iid, span?)?;
                found = true;
            }
            if !found {
                self.kept.insert(iid);
            }
        }
        Ok(())
    }

    /// Ownership is relative to the complete removal set, not one session
    /// at a time: another selected session is not a surviving owner.
    fn collect_texts(&mut self, t: &Transactions<'_, '_>) -> Result<()> {
        if self.sessions.is_empty()? {
            return Ok(());
        }
        let mut bodies = t.index.prepare(
            "SELECT t.body FROM text t WHERE t.id IN
               (SELECT text_id FROM block WHERE session_pk IN (SELECT session_pk FROM temp.excision_sessions))
             AND NOT EXISTS
               (SELECT 1 FROM block o WHERE o.text_id=t.id
                  AND o.session_pk NOT IN (SELECT session_pk FROM temp.excision_sessions))",
        )?;
        for body in bodies.query_map([], |r| r.get::<_, Vec<u8>>(0))? {
            self.own_texts.insert(&unpack_text(&body?)?)?;
        }
        let mut titles = t.index.prepare(
            "SELECT title FROM session s WHERE pk IN (SELECT session_pk FROM temp.excision_sessions)
               AND title IS NOT NULL AND NOT EXISTS
               (SELECT 1 FROM session o WHERE o.title=s.title
                  AND o.pk NOT IN (SELECT session_pk FROM temp.excision_sessions))",
        )?;
        for title in titles.query_map([], |r| r.get::<_, String>(0))? {
            self.own_texts.insert(&title?)?;
        }
        Ok(())
    }

    fn apply(
        mut self,
        t: &Transactions<'_, '_>,
        req: &ExciseRequest,
        done: &mut Excised,
    ) -> Result<Excised> {
        self.collect_texts(t)?;
        // Removing an input supersedes partial rewrites of its records.
        self.spans.remove_whole(&self.whole)?;
        self.kept.retain(|input| !self.whole.contains(input));
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
        let hashes = RemovalHashes::new(t.index)?;
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
                &hashes,
            )?;
        }

        done.sessions = self.sessions.ids()?;
        refused.sessions = Some(
            done.sessions
                .iter()
                .map(|id| SessionId::parse(id))
                .collect::<std::result::Result<_, _>>()
                .map_err(|error| StoreError::Invalid(error.to_string()))?,
        );
        done.messages = self.message_ids.count()?;
        done.inputs_removed = self.whole.iter().cloned().collect();
        done.inputs_kept = self.kept.iter().cloned().collect();
        self.spans.visit_inputs(|input, count| {
            done.inputs_overwritten.insert(input.to_owned(), count);
            Ok(())
        })?;

        // Session rows, before the inputs, so a blob a removed block named
        // can be checked against what remains.
        self.sessions
            .visit(|pk, id| delete_session(t, pk, id, req.dry_run, &hashes))?;

        for input in &self.whole {
            remove_input(t, input, req, &mut refused, done, &hashes)?;
        }
        self.spans
            .visit_inputs(|input, _| overwrite(t, input, &self.spans, req, &mut refused))?;
        if !self.kept.is_empty() {
            let mut still = BTreeSet::new();
            for input in &self.kept {
                let left = blank_document(t, input, &self.own_texts, req, &mut refused, done)?;
                done.texts_left += left;
                if left > 0 {
                    still.insert(input.clone());
                }
            }
            done.inputs_kept = still.into_iter().collect();
        }
        if !self.sessions.is_empty()? {
            done.notes_removed =
                notes::remove(t.archive, &done.sessions, &self.message_ids, req.dry_run)?;
            done.deliveries_redacted = redact_deliveries(
                t.archive,
                &done.sessions,
                &self.message_ids,
                &self.own_texts,
                req.dry_run,
            )?;
        }
        // Blobs nothing refers to any more: the removed inputs' and the
        // removed blocks' (inline images, large outputs).
        for hash in refused
            .blobs
            .iter()
            .filter_map(|hash| BlobHash::parse(hash).ok())
        {
            hashes.blob_candidate(&hash.0)?;
        }
        if !req.dry_run {
            hashes.visit_blob_candidates(|hash| {
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
                    return Ok(());
                }
                let place: Option<String> = t
                    .archive
                    .query_row("SELECT place FROM blob WHERE hash = ?1", [hash], |r| {
                        r.get(0)
                    })
                    .optional()?;
                let Some(place) = place else { return Ok(()) };
                blob_cleanup::require_no_delta_base(
                    t.archive,
                    &BlobHash(
                        hash.try_into()
                            .map_err(|_| StoreError::Invalid("invalid blob hash".into()))?,
                    ),
                )?;
                t.archive
                    .execute("DELETE FROM blob WHERE hash = ?1", [hash])?;
                done.blobs_deleted += 1;
                if place == "file"
                    && let Ok(arr) = <[u8; 32]>::try_from(hash)
                {
                    blob_cleanup::record(t.archive, &req.id, &BlobHash(arr))?;
                }
                Ok(())
            })?;
        }

        if !req.dry_run {
            t.archive.execute(
                "INSERT INTO vector_deletion_policy VALUES (?1,1,?2)",
                params![req.id, hashes.text_policy_json()?],
            )?;
            hashes.visit_texts(|hash| {
                t.archive.execute(
                    "INSERT INTO pending_vector_excision(text_hash,excision_id) VALUES (?1,?2)
                     ON CONFLICT(text_hash) DO UPDATE SET excision_id=excluded.excision_id",
                    params![hash, req.id],
                )?;
                Ok(())
            })?;
        }

        // Embeddings of texts no row holds any more.
        if let Some(e) = t.embeddings {
            hashes.visit_texts(|hash| {
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
                Ok(())
            })?;
        }
        if !req.dry_run
            && let Some(e) = t.embeddings
        {
            vector_policy::acknowledge(e, t.archive, &req.id)?;
        }
        done.texts = hashes.text_count()?;

        refused.locations.sort();
        refused.locations.dedup();
        done.refused_paths = refused
            .locations
            .iter()
            .map(|l| format!("{}:{}", l.machine, l.path))
            .collect();

        if let Some(capture) = capture {
            let publication = capture.finish(t.archive, t.store)?;
            crate::sinex_rows::visit_retired_local_batches(t.archive, &publication, |hash| {
                blob_cleanup::record(t.archive, &req.id, &hash)
            })?;
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
        if !req.dry_run {
            recovery::acknowledge(t.archive, t.index)?;
        }
        hashes.finish()?;
        self.sessions.finish()?;
        Ok(done.clone())
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
        hashes: &RemovalHashes<'_>,
    ) -> Result<()> {
        let reparse = matches.inputs()?;
        done.inputs_overwritten = matches.counts()?;
        if !req.dry_run {
            matches.overwrite(t, req, refused)?;
        }
        authored::redact(t.archive, text, req.dry_run, done)?;
        // The rows parsed from the text go; the caller parses their inputs
        // again from the overwritten stored bytes.
        done.inputs_reparsed = clear_for_reparse(t, text, &reparse, req.dry_run, hashes)?;
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
