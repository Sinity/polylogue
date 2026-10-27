//! Excision: removing content on purpose (`docs/excision.md`). An excision
//! removes whole stored inputs, with their rows in every database; it never
//! rewrites bytes. Anything it could only remove by rewriting bytes that also
//! hold other content refuses before anything changes.
//!
//! The writer runs it in one pass: [`plan`] decides what goes and refuses
//! what cannot, reading only; [`apply`] writes it in one transaction per
//! database, committed embeddings, archive, index. A dry run rolls [`apply`]
//! back.

use crate::{Result, StoreError};
use polylogue_model::{BlobHash, SessionId};
use rusqlite::{Connection, OptionalExtension, Transaction, params};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub(crate) mod index_rows;
mod refused;

pub(crate) use refused::{RefusedIds, any_excision, matching, raw_import};

/// What to remove.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "target", content = "value", rename_all = "snake_case")]
pub enum ExciseTarget {
    Session(SessionId),
    Document(polylogue_model::DocumentId),
    Input(String),
}

impl ExciseTarget {
    /// `session:<id>`, `document:<id>`, `input:<id>`, or a bare session ID.
    pub fn parse(target: &str) -> std::result::Result<Self, String> {
        let (kind, value) = target.split_once(':').unwrap_or(("", target));
        match kind {
            "document" => polylogue_model::DocumentId::parse(value)
                .map(Self::Document)
                .map_err(|e| e.to_string()),
            "input" if !value.is_empty() => Ok(Self::Input(value.to_owned())),
            "text" => Err(
                "excising a text is not supported: it would rewrite stored bytes that hold other content; excise the session or input that holds it"
                    .into(),
            ),
            "session" => SessionId::parse(value)
                .map(Self::Session)
                .map_err(|e| e.to_string()),
            _ => SessionId::parse(target).map(Self::Session).map_err(|_| {
                format!(
                    "not an excision target: {target:?} (session:<id>, document:<id>, input:<id>)"
                )
            }),
        }
    }

    fn label(&self) -> String {
        match self {
            Self::Session(s) => format!("session:{s}"),
            Self::Document(d) => format!("document:{d}"),
            Self::Input(i) => format!("input:{i}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExciseRequest {
    /// The client request ID; the excision row's ID.
    pub id: String,
    pub target: ExciseTarget,
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
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub documents: Vec<String>,
    /// Texts no index row holds any more.
    pub texts: u64,
    /// The stored inputs removed.
    pub inputs_removed: Vec<String>,
    /// Notes whose every target was removed.
    #[serde(default)]
    pub notes_removed: u64,
    /// Their IDs, which in primary mode name their records in Sinex.
    #[serde(skip)]
    pub(crate) notes: Vec<String>,
    pub blobs_deleted: u64,
    pub embeddings_deleted: u64,
    /// Paths the loop no longer collects, as `machine:path`.
    pub refused_paths: Vec<String>,
}

/// The connections an excision writes through. Blobs it no longer needs
/// are marked `deleting`; the writer deletes their files after the commits.
pub(crate) struct Transactions<'a, 'b> {
    pub archive: &'a Transaction<'b>,
    pub index: &'a Transaction<'b>,
    pub embeddings: Option<&'a Connection>,
    /// `archive.db`'s write counter for this transaction.
    pub seq: i64,
    /// `index.db`'s write counter for this transaction.
    pub counter: i64,
}

/// What an excision removes, decided before anything changes.
#[derive(Debug, Default)]
pub(crate) struct Plan {
    /// Index key and ID of each session to remove.
    sessions: Vec<(i64, String)>,
    documents: Vec<(i64, String)>,
    inputs: BTreeSet<String>,
    /// The same request was recorded by an earlier run: what is left (an
    /// index rebuild, blob files) finishes when the archive opens or after
    /// this run.
    recorded: bool,
}

impl Plan {
    /// The stored inputs it removes.
    pub(crate) fn inputs(&self) -> &BTreeSet<String> {
        &self.inputs
    }

    pub(crate) fn recorded(&self) -> bool {
        self.recorded
    }
}

/// Decides what `request` removes, reading only. Every refusal happens here.
pub(crate) fn plan(
    archive: &Connection,
    index: &Connection,
    request: &ExciseRequest,
) -> Result<Plan> {
    let recorded: Option<String> = archive
        .query_row(
            "SELECT target FROM excision WHERE id = ?1",
            [&request.id],
            |r| r.get(0),
        )
        .optional()?;
    match recorded {
        Some(target) if target == request.target.label() => {
            return Ok(Plan {
                recorded: true,
                ..Plan::default()
            });
        }
        Some(_) => {
            return Err(StoreError::Conflict(format!(
                "excision {} was recorded with another target",
                request.id
            )));
        }
        None => {}
    }
    let mut plan = Plan::default();
    match &request.target {
        ExciseTarget::Session(id) => {
            let id = id.to_string();
            let pk = index_rows::session_pk(index, &id)?
                .ok_or_else(|| StoreError::NotFound(format!("session {id}")))?;
            plan.sessions.push((pk, id));
            plan.inputs
                .extend(index_rows::inputs_of_session(index, pk)?);
        }
        ExciseTarget::Document(id) => {
            let id = id.to_string();
            let pk = index
                .query_row("SELECT pk FROM document WHERE id = ?1", [&id], |r| r.get(0))
                .optional()?
                .ok_or_else(|| StoreError::NotFound(format!("document {id}")))?;
            plan.documents.push((pk, id));
            plan.inputs
                .extend(index_rows::inputs_of_document(index, pk)?);
        }
        ExciseTarget::Input(input) => {
            let known: bool = archive.query_row(
                "SELECT EXISTS(SELECT 1 FROM input WHERE id = ?1)",
                [input],
                |r| r.get(0),
            )?;
            if !known {
                return Err(StoreError::NotFound(format!("input {input}")));
            }
            plan.inputs.insert(input.clone());
            if let Some(pk) = index_rows::input_pk(index, input)? {
                plan.sessions = index_rows::sessions_of_input(index, pk)?;
                plan.documents = index_rows::documents_of_input(index, pk)?;
            }
        }
    }
    let input_named = matches!(request.target, ExciseTarget::Input(_));
    index_rows::refuse_shared(archive, index, &plan, input_named)?;
    Ok(plan)
}

/// Writes the excision `plan` describes.
pub(crate) fn apply(
    t: &Transactions<'_, '_>,
    request: &ExciseRequest,
    plan: &Plan,
) -> Result<Excised> {
    let mut done = Excised {
        dry_run: request.dry_run,
        sessions: plan.sessions.iter().map(|(_, id)| id.clone()).collect(),
        documents: plan.documents.iter().map(|(_, id)| id.clone()).collect(),
        inputs_removed: plan.inputs.iter().cloned().collect(),
        ..Excised::default()
    };
    t.archive.execute(
        "INSERT INTO excision(id, target, reason, created_at, commit_seq)
         VALUES (?1, ?2, ?3, ?4, ?5)",
        params![
            request.id,
            request.target.label(),
            request.reason,
            now_ms(),
            t.seq
        ],
    )?;
    let record = |kind: &str, value: &str| refused::record(t.archive, &request.id, kind, value);

    // Texts whose rows go, so their vectors go too; blobs that may now be
    // unused.
    let mut removed_texts = BTreeSet::new();
    let mut blobs = BTreeSet::new();
    let messages = index_rows::message_ids(t.index, &plan.sessions)?;
    done.messages = messages.len() as u64;
    for id in &messages {
        record("message", id)?;
    }
    for (pk, id) in &plan.sessions {
        record("session", id)?;
        index_rows::remove_session(t, *pk, id, &mut removed_texts, &mut blobs)?;
    }
    let mut references = index_rows::note_references(t.archive, &done.sessions, &messages)?;
    for (pk, id) in &plan.documents {
        record("document", id)?;
        index_rows::remove_document(t, *pk, &mut removed_texts, &mut blobs)?;
        let reference = format!("document:{id}");
        references.push(reference);
    }
    done.notes = index_rows::remove_notes_about(t.archive, &references)?;
    done.notes_removed = done.notes.len() as u64;

    let mut locations = BTreeSet::new();
    for input in &plan.inputs {
        record("input", input)?;
        let removed = index_rows::remove_input(t, &request.id, input)?;
        for blob in removed.blobs {
            record("blob", &blob.to_string())?;
            blobs.insert(blob);
        }
        locations.extend(removed.locations);
    }
    for (machine, path) in &locations {
        record("location", &serde_json::to_string(&(machine, path))?)?;
        done.refused_paths.push(format!("{machine}:{path}"));
    }

    for hash in &blobs {
        if index_rows::blob_unused(t, hash)? && crate::blobs::mark_for_deletion(t.archive, hash)? {
            done.blobs_deleted += 1;
        }
    }
    if let Some(embeddings) = t.embeddings {
        for hash in &removed_texts {
            done.embeddings_deleted +=
                embeddings.execute("DELETE FROM embedded WHERE text_hash = ?1", [hash])? as u64;
            embeddings.execute("DELETE FROM embed_failure WHERE text_hash = ?1", [hash])?;
        }
    }
    done.texts = removed_texts.len() as u64;
    mark_applied(t.archive, t.index)?;
    Ok(done)
}

/// What an already recorded excision removed, from its `excised` rows.
pub(crate) fn recorded(archive: &Connection, request: &ExciseRequest) -> Result<Excised> {
    let values = |kind: &str| -> Result<Vec<String>> {
        Ok(archive
            .prepare(
                "SELECT value FROM excised WHERE excision_id = ?1 AND kind = ?2 ORDER BY value",
            )?
            .query_map(params![request.id, kind], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?)
    };
    let mut refused_paths = Vec::new();
    for location in values("location")? {
        let (machine, path): (String, String) = serde_json::from_str(&location)?;
        refused_paths.push(format!("{machine}:{path}"));
    }
    Ok(Excised {
        sessions: values("session")?,
        messages: values("message")?.len() as u64,
        documents: values("document")?,
        inputs_removed: values("input")?,
        refused_paths,
        ..Excised::default()
    })
}

/// Completes recorded removals before readers can open this index. Each
/// removal uses its recorded identities; unrelated indexed rows stay in place.
pub(crate) fn catch_up(archive: &Connection, index: &Connection) -> Result<()> {
    if !index_behind(archive, index)? {
        return Ok(());
    }
    let previous: i64 = index.pragma_query_value(None, "synchronous", |r| r.get(0))?;
    index.pragma_update(None, "synchronous", 2)?;
    index.pragma_update(None, "secure_delete", true)?;
    let result = (|| {
        let atx = archive.unchecked_transaction()?;
        let tx = index.unchecked_transaction()?;
        let counter = crate::archive::bump_counter(&tx)?;
        let t = Transactions {
            archive: &atx,
            index: &tx,
            embeddings: None,
            seq: 0,
            counter,
        };
        let mut texts = BTreeSet::new();
        let mut blobs = BTreeSet::new();
        let mut statement = atx.prepare("SELECT DISTINCT kind,value FROM excised WHERE kind IN ('session','document','input') ORDER BY CASE kind WHEN 'session' THEN 0 WHEN 'document' THEN 1 ELSE 2 END,value")?;
        let mut rows = statement.query([])?;
        while let Some(row) = rows.next()? {
            let kind: String = row.get(0)?;
            let id: String = row.get(1)?;
            match kind.as_str() {
                "session" => {
                    if let Some(pk) = index_rows::session_pk(&tx, &id)? {
                        index_rows::remove_session(&t, pk, &id, &mut texts, &mut blobs)?;
                    }
                }
                "document" => {
                    let pk: Option<i64> = tx
                        .query_row("SELECT pk FROM document WHERE id=?1", [&id], |r| r.get(0))
                        .optional()?;
                    if let Some(pk) = pk {
                        index_rows::remove_document(&t, pk, &mut texts, &mut blobs)?;
                    }
                }
                "input" => index_rows::remove_index_input(&tx, &id)?,
                _ => unreachable!(),
            }
        }
        mark_applied(&atx, &tx)?;
        tx.commit()?;
        for table in ["text_fts", "output_fts", "tool_call_trigram"] {
            index.execute(
                &format!("INSERT INTO {table}({table}) VALUES ('optimize')"),
                [],
            )?;
        }
        index.query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |_| Ok(()))?;
        Ok(())
    })();
    index.pragma_update(None, "secure_delete", false)?;
    index.pragma_update(None, "synchronous", previous)?;
    result
}

/// Which excisions an `index.db` has applied, committed with their removals.
fn excisions(archive: &Connection) -> Result<String> {
    Ok(archive.query_row(
        "SELECT count(*) || ':' || COALESCE(max(commit_seq), 0) FROM excision",
        [],
        |r| r.get(0),
    )?)
}

/// Records in `index` that it holds every excision `archive` holds.
pub(crate) fn mark_applied(archive: &Connection, index: &Connection) -> Result<()> {
    index.execute(
        "INSERT INTO meta(key, value) VALUES ('excisions', ?1)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        [excisions(archive)?],
    )?;
    Ok(())
}

/// Whether `index` lacks an excision `archive` holds.
pub(crate) fn index_behind(archive: &Connection, index: &Connection) -> Result<bool> {
    let applied: Option<String> = index
        .query_row("SELECT value FROM meta WHERE key = 'excisions'", [], |r| {
            r.get(0)
        })
        .optional()?;
    Ok(applied.as_deref() != Some(excisions(archive)?.as_str()))
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as i64)
}

#[cfg(test)]
mod tests;
