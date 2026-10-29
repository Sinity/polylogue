//! Writes people and agents make to `archive.db`: notes and their
//! withdrawals, context deliveries and settings. They go through the one
//! writer thread like everything else, each in its own small transaction,
//! so a note never waits behind a first build for longer than one group.
//!
//! Every write carries a client request ID. The same ID with the same
//! intent writes nothing and returns what the first write created; the
//! same ID with other intent is refused as a conflict.

use crate::{Result, StoreError};
use polylogue_model::{Author, DeliveryOutcome, NewNote, NoteKind, Timestamp};
use rusqlite::{OptionalExtension, Transaction, params};

use polylogue_model::user::{Delivery, UserOutcome, UserRequest, UserWrite};

/// Construct trusted request metadata at the I/O boundary.
pub fn request_now(id: String, write: UserWrite) -> UserRequest {
    UserRequest {
        id,
        machine: machine_id(),
        at: now(),
        write,
    }
}

pub fn now() -> Timestamp {
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    Timestamp::from_unix_ns(i128::try_from(ns).unwrap_or(0))
        .unwrap_or_else(|| Timestamp::from_unix_ms(0).expect("the epoch is a valid timestamp"))
}

/// `/etc/machine-id`, or `unknown` where there is none.
pub fn machine_id() -> String {
    std::fs::read_to_string("/etc/machine-id")
        .map(|s| s.trim().to_owned())
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".into())
}

/// A new UUIDv7 for a request.
pub fn new_request_id() -> String {
    uuid::Uuid::now_v7().to_string()
}

/// Applies one request in `tx`, which the writer commits. `seq` is the
/// archive's write counter for this transaction.
pub(crate) fn apply(tx: &Transaction<'_>, r: &UserRequest, seq: i64) -> Result<UserOutcome> {
    let source = crate::source::Source::of(tx)?;
    let content = serde_json::to_vec(&r.write.intent())?;
    let hash = blake3::hash(&content);
    let seen: Option<Vec<u8>> = tx
        .query_row(
            &source.sql("SELECT content_hash FROM @source.client_request WHERE id = ?1"),
            [&r.id],
            |row| row.get(0),
        )
        .optional()?;
    match seen {
        Some(h) if h.as_slice() == hash.as_bytes() => {
            return Ok(UserOutcome {
                pending: false,
                id: r.id.clone(),
                repeat: true,
            });
        }
        Some(_) => return Err(StoreError::Conflict(r.id.clone())),
        None => {}
    }
    tx.execute(
        &source.sql(
            "INSERT INTO @source.client_request(id, content_hash, created_at, request, commit_seq)
         VALUES (?1, ?2, ?3, ?4, ?5)",
        ),
        params![
            r.id,
            hash.as_bytes().as_slice(),
            r.at.unix_ms(),
            crate::encoding::pack_text(&serde_json::to_string(r)?)?,
            seq
        ],
    )?;
    let at = r.at.unix_ms();
    match &r.write {
        UserWrite::AddNote { note } | UserWrite::AnnotateQuery { note, .. } => {
            add_note(tx, r, note, seq)
        }
        UserWrite::SaveQuery { note } => {
            if note.kind != NoteKind::SavedQuery {
                return Err(StoreError::Invalid("save needs a saved query note".into()));
            }
            add_note(tx, r, note, seq)
        }
        UserWrite::Withdraw {
            note,
            author,
            reason,
        } => {
            let first_note = first_note_of(tx, note)?;
            tx.execute(
                &source.sql("INSERT INTO @source.note(id, first_note, kind, value, author, machine, created_at, commit_seq)
                 VALUES (?1, ?1, 'withdrawal', ?2, ?3, ?4, ?5, ?6)"),
                params![r.id, reason, author.to_string(), r.machine, at, seq],
            )?;
            tx.execute(
                &source.sql("INSERT INTO @source.note_relation(note_id, kind, other) VALUES (?1, 'withdraws', ?2)"),
                params![r.id, format!("note:{first_note}")],
            )?;
            Ok(UserOutcome {
                pending: false,
                id: r.id.clone(),
                repeat: false,
            })
        }
        UserWrite::Deliver(d) | UserWrite::DeliveredAdvice(d) => {
            let body = crate::encoding::pack_text(&d.text)?;
            tx.execute(
                &source.sql("INSERT INTO @source.delivery(id, recipient_kind, recipient_id, sender, event, cwd, at,
                     content_hash, bytes, body, refs_json, left_out_json, commit_seq)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)"),
                params![
                    r.id,
                    d.recipient_kind,
                    d.recipient_id,
                    d.sender,
                    d.event,
                    d.cwd,
                    at,
                    blake3::hash(d.text.as_bytes()).as_bytes().as_slice(),
                    i64::try_from(d.text.len()).unwrap_or(i64::MAX),
                    body,
                    serde_json::to_string(&d.refs)?,
                    serde_json::to_string(&d.left_out)?,
                    seq,
                ],
            )?;
            if matches!(r.write, UserWrite::DeliveredAdvice(_)) {
                tx.execute(&source.sql("INSERT INTO @source.delivery_event(delivery, what, at, detail, commit_seq) VALUES (?1, 'delivered', ?2, ?3, ?4)"),
                    params![r.id, at, "written to the harness hook output", seq])?;
            }
            Ok(UserOutcome {
                pending: false,
                id: r.id.clone(),
                repeat: false,
            })
        }
        UserWrite::DeliveryEvent {
            delivery,
            what,
            detail,
        } => {
            let known: bool = tx
                .prepare_cached(&source.sql("SELECT 1 FROM @source.delivery WHERE id = ?1"))?
                .exists([delivery])?;
            if !known {
                return Err(StoreError::NotFound(format!("delivery {delivery}")));
            }
            tx.execute(
                &source.sql("INSERT OR IGNORE INTO @source.delivery_event(delivery, what, at, detail, commit_seq)
                 VALUES (?1, ?2, ?3, ?4, ?5)"),
                params![delivery, what.as_str(), at, detail, seq],
            )?;
            Ok(UserOutcome {
                pending: false,
                id: delivery.clone(),
                repeat: false,
            })
        }
        UserWrite::KeepResult {
            name,
            query,
            rows_json,
            read_at,
        } => {
            tx.execute(
                &source.sql(
                    "INSERT INTO @source.kept_result(id, name, query, ran_at, read_at, rows_json)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                ),
                params![r.id, name, query, at, read_at, rows_json],
            )?;
            Ok(UserOutcome {
                pending: false,
                id: r.id.clone(),
                repeat: false,
            })
        }
        UserWrite::ChatEnqueue(new) => {
            let id = crate::chat_actions::enqueue(tx, new, at, seq)?;
            Ok(UserOutcome {
                pending: false,
                repeat: id != new.id,
                id,
            })
        }
        UserWrite::ChatClaim { by } => Ok(UserOutcome {
            pending: false,
            id: crate::chat_actions::claim(tx, by, at, seq)?,
            repeat: false,
        }),
        UserWrite::ChatReport {
            action,
            by,
            outcome,
            phase,
            detail,
            receipt,
        } => {
            crate::chat_actions::report(
                tx,
                action,
                by,
                outcome,
                phase,
                detail.as_deref(),
                receipt.as_ref(),
                at,
                seq,
            )?;
            Ok(UserOutcome {
                pending: false,
                id: action.clone(),
                repeat: false,
            })
        }
        UserWrite::ChatDecide {
            action,
            approve,
            by,
            detail,
        } => {
            crate::chat_actions::decide(tx, action, *approve, by, detail.as_deref(), at, seq)?;
            Ok(UserOutcome {
                pending: false,
                id: action.clone(),
                repeat: false,
            })
        }
        UserWrite::BackfillStart(new) => {
            let id = crate::backfill_jobs::start(tx, new, at, seq)?;
            Ok(UserOutcome {
                pending: false,
                repeat: id != new.id,
                id,
            })
        }
        UserWrite::BackfillClaim { by } => Ok(UserOutcome {
            pending: false,
            id: crate::backfill_jobs::claim(tx, by, at, seq)?,
            repeat: false,
        }),
        UserWrite::BackfillReport { job, by, report } => {
            crate::backfill_jobs::report(tx, job, by, report, at, seq)?;
            Ok(UserOutcome {
                pending: false,
                id: job.clone(),
                repeat: false,
            })
        }
        UserWrite::BackfillCancel { job } => {
            crate::backfill_jobs::cancel(tx, job, at, seq)?;
            Ok(UserOutcome {
                pending: false,
                id: job.clone(),
                repeat: false,
            })
        }
        UserWrite::SetSetting {
            key,
            value,
            author,
            machine,
        } => {
            tx.execute(
                &source.sql("INSERT INTO @source.setting(id, machine, key, value, author, set_at, commit_seq)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)"),
                params![
                    r.id,
                    machine.as_deref().unwrap_or("*"),
                    key,
                    value,
                    author.to_string(),
                    at,
                    seq
                ],
            )?;
            Ok(UserOutcome {
                pending: false,
                id: r.id.clone(),
                repeat: false,
            })
        }
    }
}

/// The first note of the note a request adds or withdraws, once applied in
/// `tx`.
pub(crate) fn first_note_of_request(
    tx: &Transaction<'_>,
    r: &UserRequest,
) -> Result<Option<String>> {
    match &r.write {
        UserWrite::AddNote { .. }
        | UserWrite::SaveQuery { .. }
        | UserWrite::AnnotateQuery { .. } => first_note_of(tx, &r.id).map(Some),
        UserWrite::Withdraw { note, .. } => first_note_of(tx, note).map(Some),
        _ => Ok(None),
    }
}

fn first_note_of(tx: &Transaction<'_>, note: &str) -> Result<String> {
    let source = crate::source::Source::of(tx)?;
    tx.query_row(
        &source.sql("SELECT first_note FROM @source.note WHERE id = ?1"),
        [note],
        |r| r.get(0),
    )
    .optional()?
    .ok_or_else(|| StoreError::NotFound(format!("note {note}")))
}

fn add_note(
    tx: &Transaction<'_>,
    r: &UserRequest,
    note: &NewNote,
    seq: i64,
) -> Result<UserOutcome> {
    let source = crate::source::Source::of(tx)?;
    // A saved query or a signal rule is a definition, about nothing in
    // particular; every other note is about something.
    let definition = matches!(note.kind, NoteKind::SavedQuery | NoteKind::Signal);
    if note.targets.is_empty() && !definition {
        return Err(StoreError::Invalid(
            "a note needs at least one target".into(),
        ));
    }
    let first_note = match &note.revises {
        Some(prev) => first_note_of(tx, prev)?,
        None => r.id.clone(),
    };
    tx.execute(
        &source.sql("INSERT INTO @source.note(id, first_note, kind, key, value, author, machine, created_at, valid_until, commit_seq)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)"),
        params![
            r.id,
            first_note,
            note.kind.as_str(),
            note.key,
            note.value,
            note.author.to_string(),
            r.machine,
            r.at.unix_ms(),
            note.valid_until.map(Timestamp::unix_ms),
            seq,
        ],
    )?;
    for target in &note.targets {
        tx.execute(
            &source.sql("INSERT OR IGNORE INTO @source.note_target(note_id, ref) VALUES (?1, ?2)"),
            params![r.id, target.to_string()],
        )?;
    }
    if let Some(prev) = &note.revises {
        tx.execute(
            &source.sql("INSERT INTO @source.note_relation(note_id, kind, other) VALUES (?1, 'revises', ?2)"),
            params![r.id, format!("note:{prev}")],
        )?;
    }
    for (relation, other) in &note.relations {
        tx.execute(
            &source.sql("INSERT OR IGNORE INTO @source.note_relation(note_id, kind, other) VALUES (?1, ?2, ?3)"),
            params![r.id, relation.as_str(), other],
        )?;
    }
    Ok(UserOutcome {
        pending: false,
        id: r.id.clone(),
        repeat: false,
    })
}

#[cfg(test)]
mod tests;
