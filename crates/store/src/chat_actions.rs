//! Chat actions: turns the operator asks the browser extension to draft or
//! submit in a chat web page. The request is stored once, with its
//! attachments as blobs; what happened to it is a list of events, and its
//! state is the latest one. The writer applies every change
//! ([`crate::user_writes`]), so two extension instances never claim the
//! same action.

/// The report fields recorded for one chat action event.
pub(crate) struct ChatEventReport<'a> {
    pub request: &'a str,
    pub id: &'a str,
    pub at: i64,
    pub kind: &'a str,
    pub phase: &'a str,
    pub detail: Option<&'a str>,
    pub by: Option<&'a str>,
    pub receipt: Option<&'a serde_json::Value>,
    pub seq: i64,
}

use crate::{Reader, Result, StoreError};
use polylogue_model::BlobHash;
use polylogue_model::JsonExt;
use rusqlite::{Connection, OptionalExtension, params};

use polylogue_model::user::{
    ChatAction, ChatAttachment, ChatEvent, ChatStatus, NewChatAction, chat_outcome,
};

impl Reader<'_> {
    /// Every chat action, oldest first.
    pub fn chat_actions(&self) -> Result<Vec<ChatAction>> {
        let conn = self.conn()?;
        let ids: Vec<String> = conn
            .prepare_cached("SELECT id FROM source_chat_action ORDER BY created_at, id")?
            .query_map([], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        ids.iter()
            .map(|id| {
                read(conn, "source_", id)?
                    .ok_or_else(|| StoreError::NotFound(format!("chat action {id}")))
            })
            .collect()
    }

    pub fn chat_action(&self, id: &str) -> Result<Option<ChatAction>> {
        read(self.conn()?, "source_", id)
    }

    /// A chat action attachment's bytes, checked against their hash.
    pub fn chat_attachment(
        &self,
        action: &str,
        attachment: &str,
    ) -> Result<(ChatAttachment, Vec<u8>)> {
        use std::io::Read;
        let a = self
            .chat_action(action)?
            .ok_or_else(|| refused("unknown_chat_action"))?
            .attachments
            .into_iter()
            .find(|a| a.id == attachment)
            .ok_or_else(|| refused("unknown_chat_action_attachment"))?;
        let store = self.blob_store()?;
        let mut bytes = Vec::new();
        store
            .read_plain(&hashed_blobs::Hash(a.blob.0), a.size, || Ok(()))?
            .read_to_end(&mut bytes)?;
        if BlobHash::of(&bytes) != a.blob {
            return Err(StoreError::BlobCorrupt(a.blob));
        }
        Ok((a, bytes))
    }
}

/// One chat action from the archive tables, whose names start with `table_prefix`
/// (`source_` on the cache connection, empty on a standalone archive).
pub(crate) fn read(conn: &Connection, table_prefix: &str, id: &str) -> Result<Option<ChatAction>> {
    let row = conn
        .prepare_cached(&format!(
            "SELECT request_key, request_hash, request_json, stage_only, needs_approval, created_at, execution
             FROM {table_prefix}chat_action WHERE id = ?1"
        ))?
        .query_row([id], |r| <(String, Vec<u8>, String, bool, bool, i64, String)>::try_from(r))
        .optional()?;
    let Some((request_key, hash, request, stage_only, needs_approval, created_at, execution)) = row
    else {
        return Ok(None);
    };
    let attachments = conn
        .prepare_cached(&format!(
            "SELECT id, blob, name, media_type, size FROM {table_prefix}chat_action_attachment
             WHERE action = ?1 ORDER BY id"
        ))?
        .query_map([id], |r| {
            <(String, Vec<u8>, String, String, u64)>::try_from(r)
        })?
        .map(|row| {
            let (id, blob, name, media_type, size) = row?;
            Ok(ChatAttachment {
                id,
                blob: BlobHash(
                    blob.try_into()
                        .map_err(|_| StoreError::stored(1, "invalid attachment blob"))?,
                ),
                name,
                media_type,
                size,
            })
        })
        .collect::<Result<_>>()?;
    let events = conn
        .prepare_cached(&format!(
            "SELECT at, kind, phase, detail, by, receipt_json, request FROM {table_prefix}chat_action_event
             WHERE action = ?1 ORDER BY position"
        ))?
        .query_map([id], |r| <(i64, String, String, Option<String>, Option<String>, Option<String>, String)>::try_from(r))?
        .map(|row| {
            let (at, kind, phase, detail, by, receipt, request) = row?;
            Ok(ChatEvent {
                request,
                at,
                kind,
                phase,
                detail,
                by,
                receipt: receipt.map(|r| serde_json::from_str(&r)).transpose()?,
            })
        })
        .collect::<Result<_>>()?;
    Ok(Some(ChatAction {
        id: id.to_owned(),
        request_key,
        request_hash: BlobHash(
            hash.try_into()
                .map_err(|_| StoreError::stored(1, "invalid request hash"))?,
        ),
        request: serde_json::from_str(&request)?,
        stage_only,
        needs_approval,
        created_at,
        execution: polylogue_model::authored::RunsHere::from_name(&execution)
            .ok_or_else(|| StoreError::stored(6, "unknown execution eligibility"))?,
        attachments,
        events,
    }))
}

/// A refused change to a chat action, by a stable code.
fn refused(code: &'static str) -> StoreError {
    StoreError::ChatAction(code)
}

/// Stores a new action, or finds the one a resent request made.
pub(crate) fn enqueue(
    tx: &Connection,
    new: &NewChatAction,
    at: i64,
    seq: i64,
    request: &str,
    available: &dyn Fn(&ChatAttachment) -> Result<bool>,
) -> Result<String> {
    let source = crate::source::Source::of(tx)?;
    let known: Option<(String, Vec<u8>)> = tx
        .query_row(
            &source.sql("SELECT id, request_hash FROM @source.chat_action WHERE request_key = ?1"),
            [&new.request_key],
            |r| <(_, _)>::try_from(r),
        )
        .optional()?;
    if let Some((id, hash)) = known {
        return if hash == new.request_hash.0 {
            Ok(id)
        } else {
            Err(refused("idempotency_key_reused"))
        };
    }
    tx.execute(
        &source.sql("INSERT INTO @source.chat_action(id, request_key, request_hash, request_json, stage_only,
             needs_approval, created_at, commit_seq)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)"),
        params![
            new.id,
            new.request_key,
            &new.request_hash.0[..],
            serde_json::to_string(&new.request)?,
            new.stage_only,
            new.needs_approval,
            at,
            seq
        ],
    )?;
    for a in &new.attachments {
        let present = available(a)?;
        if !present {
            return Err(refused("unknown_attachment"));
        }
        tx.execute(
            &source.sql("INSERT INTO @source.chat_action_attachment(action, id, blob, name, media_type, size)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)"),
            params![
                new.id,
                a.id,
                &a.blob.0[..],
                a.name,
                a.media_type,
                i64::try_from(a.size).unwrap_or(i64::MAX)
            ],
        )?;
    }
    let phase = if new.needs_approval {
        "awaiting_approval"
    } else {
        "queued"
    };
    add_event(
        tx,
        ChatEventReport {
            request,
            id: &new.id,
            at,
            kind: phase,
            phase,
            detail: None,
            by: None,
            receipt: None,
            seq,
        },
    )?;
    Ok(new.id.clone())
}

/// Hands the oldest queued action to `by`, unless another action is out.
pub(crate) fn claim(tx: &Connection, by: &str, at: i64, seq: i64, request: &str) -> Result<String> {
    let source = crate::source::Source::of(tx)?;
    let ids: Vec<String> = tx
        .prepare_cached(&source.sql(
            "SELECT id FROM @source.chat_action WHERE execution='live' ORDER BY created_at, id",
        ))?
        .query_map([], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    let mut next = None;
    for id in ids {
        let action = read(tx, source.prefix(), &id)?
            .ok_or_else(|| StoreError::NotFound(format!("chat action {id}")))?;
        match action.status(at).0 {
            ChatStatus::Claimed => return Err(StoreError::NotFound("a queued chat action".into())),
            ChatStatus::Queued if next.is_none() => next = Some(id),
            _ => {}
        }
    }
    let id = next.ok_or_else(|| StoreError::NotFound("a queued chat action".into()))?;
    add_event(
        tx,
        ChatEventReport {
            request,
            id: &id,
            at,
            kind: "claimed",
            phase: "claimed",
            detail: None,
            by: Some(by),
            receipt: None,
            seq,
        },
    )?;
    Ok(id)
}

/// A report from the extension instance holding the claim.
pub(crate) fn report(tx: &Connection, event: ChatEventReport<'_>) -> Result<()> {
    let ChatEventReport {
        request,
        id,
        at,
        kind,
        phase,
        detail,
        by,
        receipt,
        seq,
    } = event;
    let by = by.ok_or_else(|| refused("not_the_claimant"))?;
    let source = crate::source::Source::of(tx)?;
    let action = read(tx, source.prefix(), id)?.ok_or_else(|| refused("unknown_chat_action"))?;
    let next = chat_outcome(kind).ok_or_else(|| refused("unknown_outcome"))?;
    let (status, holder) = action.status(at);
    if status.finished() {
        return Err(refused("action_finished"));
    }
    let late_confirmation =
        next == Some(ChatStatus::Submitted) && action.unconfirmed_submitter(at) == Some(by);
    if holder != Some(by) && !late_confirmation {
        return Err(refused("not_the_claimant"));
    }
    if action.stage_only && next == Some(ChatStatus::Submitted) {
        return Err(refused("stage_only_action_cannot_submit"));
    }
    if late_confirmation {
        // This is the existing trusted extension's captured-turn report, not
        // renewed permission to act or an independent check against the index.
        let field = |key| {
            receipt
                .and_then(|receipt| receipt.get(key))
                .and_then(|value| value.as_str())
                .filter(|value| !value.trim().is_empty())
        };
        let (Some(conversation), Some(_turn)) = (field("conversation"), field("turn")) else {
            return Err(refused("submission_receipt_required"));
        };
        if action
            .request
            .str("conversation")
            .is_some_and(|requested| requested != conversation)
        {
            return Err(refused("submission_conversation_mismatch"));
        }
    }
    add_event(
        tx,
        ChatEventReport {
            request,
            id,
            at,
            kind,
            phase,
            detail,
            by: Some(by),
            receipt,
            seq,
        },
    )
}

/// The operator's approval or refusal of a held submit.
pub(crate) fn decide(
    tx: &Connection,
    id: &str,
    approve: bool,
    by: &str,
    detail: Option<&str>,
    seq: i64,
    request: &polylogue_model::user::UserRequest,
) -> Result<()> {
    let at = request.at.unix_ms();
    let source = crate::source::Source::of(tx)?;
    let action = read(tx, source.prefix(), id)?.ok_or_else(|| refused("unknown_chat_action"))?;
    if action.status(at).0 != ChatStatus::AwaitingApproval {
        return Err(refused("not_awaiting_approval"));
    }
    let (kind, phase) = if approve {
        ("approved", "queued")
    } else {
        ("declined", "declined")
    };
    add_event(
        tx,
        ChatEventReport {
            request: &request.id,
            id,
            at,
            kind,
            phase,
            detail,
            by: Some(by),
            receipt: None,
            seq,
        },
    )
}

fn add_event(tx: &Connection, event: ChatEventReport<'_>) -> Result<()> {
    let ChatEventReport {
        request,
        id,
        at,
        kind,
        phase,
        detail,
        by,
        receipt,
        seq,
    } = event;
    let source = crate::source::Source::of(tx)?;
    tx.execute(
        &source.sql("INSERT INTO @source.chat_action_event(action, position, at, kind, phase, detail, by, receipt_json, request, commit_seq)
         VALUES (?1, (SELECT COALESCE(max(position), 0) + 1 FROM @source.chat_action_event WHERE action = ?1),
                 ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)"),
        params![
            id,
            at,
            kind,
            phase,
            detail,
            by,
            receipt.map(serde_json::to_string).transpose()?,
            request,
            seq
        ],
    )?;
    Ok(())
}

#[cfg(test)]
mod tests;
