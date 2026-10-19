//! Writes people and agents make to `archive.db`: notes and their
//! withdrawals, context deliveries and settings. They go through the one
//! writer thread like everything else, each in its own small transaction,
//! so a note never waits behind a first build for longer than one group.
//!
//! Every write carries a client request ID. The same ID with the same
//! content writes nothing and returns what the first write created; the
//! same ID with other content is refused as a conflict.

use crate::{Result, StoreError};
use polylogue_model::{Author, NewNote, Timestamp};
use rusqlite::{OptionalExtension, Transaction, params};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "write", rename_all = "snake_case")]
pub enum UserWrite {
    AddNote {
        note: NewNote,
    },
    /// Withdraws a note family: the newest revision and every older one.
    Withdraw {
        note: String,
        author: Author,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
    /// Context prepared for a recipient, with its exact text.
    Deliver(Delivery),
    /// What happened to a prepared delivery.
    DeliveryEvent {
        delivery: String,
        what: DeliveryOutcome,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },
    SetSetting {
        key: String,
        value: String,
        author: Author,
        /// `None` for an archive-wide key, otherwise this machine.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        machine: Option<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Delivery {
    pub recipient_kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recipient_id: Option<String>,
    pub sender: String,
    pub event: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
    pub text: String,
    pub refs: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub left_out: Vec<LeftOut>,
}

/// A reference or section a delivery did not include, and why.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LeftOut {
    pub what: String,
    pub reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeliveryOutcome {
    Delivered,
    Failed,
    NotSent,
    Used,
}

impl DeliveryOutcome {
    fn as_str(self) -> &'static str {
        match self {
            Self::Delivered => "delivered",
            Self::Failed => "failed",
            Self::NotSent => "not_sent",
            Self::Used => "used",
        }
    }
}

/// What a user write created, or found already created by the same
/// request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UserOutcome {
    /// The note, delivery or setting row the request wrote.
    pub id: String,
    /// The request had already been written; nothing changed.
    pub repeat: bool,
}

/// One user write with the request that carries it.
#[derive(Debug, Clone, PartialEq)]
pub struct UserRequest {
    pub id: String,
    pub machine: String,
    pub at: Timestamp,
    pub write: UserWrite,
}

impl UserRequest {
    /// A request at the current time on this machine.
    pub fn now(id: String, write: UserWrite) -> Self {
        Self {
            id,
            machine: machine_id(),
            at: now(),
            write,
        }
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
    let content = serde_json::to_vec(&r.write)?;
    let hash = blake3::hash(&content);
    let seen: Option<Vec<u8>> = tx
        .query_row(
            "SELECT content_hash FROM client_request WHERE id = ?1",
            [&r.id],
            |row| row.get(0),
        )
        .optional()?;
    match seen {
        Some(h) if h.as_slice() == hash.as_bytes() => {
            return Ok(UserOutcome {
                id: r.id.clone(),
                repeat: true,
            });
        }
        Some(_) => return Err(StoreError::Conflict(r.id.clone())),
        None => {}
    }
    tx.execute(
        "INSERT INTO client_request(id, content_hash, created_at) VALUES (?1, ?2, ?3)",
        params![r.id, hash.as_bytes().as_slice(), r.at.unix_ms()],
    )?;
    let at = r.at.unix_ms();
    match &r.write {
        UserWrite::AddNote { note } => add_note(tx, r, note, seq),
        UserWrite::Withdraw {
            note,
            author,
            reason,
        } => {
            let root = root_of(tx, note)?;
            tx.execute(
                "INSERT INTO note(id, root, kind, value, author, machine, created_at, commit_seq)
                 VALUES (?1, ?1, 'withdrawal', ?2, ?3, ?4, ?5, ?6)",
                params![r.id, reason, author.to_string(), r.machine, at, seq],
            )?;
            tx.execute(
                "INSERT INTO note_relation(note_id, kind, other) VALUES (?1, 'withdraws', ?2)",
                params![r.id, format!("note:{root}")],
            )?;
            Ok(UserOutcome {
                id: r.id.clone(),
                repeat: false,
            })
        }
        UserWrite::Deliver(d) => {
            let body = zstd::bulk::compress(d.text.as_bytes(), 3)?;
            tx.execute(
                "INSERT INTO delivery(id, recipient_kind, recipient_id, sender, event, cwd, at,
                     content_hash, bytes, body, refs_json, left_out_json, commit_seq)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
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
            Ok(UserOutcome {
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
                .prepare_cached("SELECT 1 FROM delivery WHERE id = ?1")?
                .exists([delivery])?;
            if !known {
                return Err(StoreError::NotFound(format!("delivery {delivery}")));
            }
            tx.execute(
                "INSERT OR IGNORE INTO delivery_event(delivery, what, at, detail)
                 VALUES (?1, ?2, ?3, ?4)",
                params![delivery, what.as_str(), at, detail],
            )?;
            Ok(UserOutcome {
                id: delivery.clone(),
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
                "INSERT INTO setting(id, machine, key, value, author, set_at, commit_seq)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
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
                id: r.id.clone(),
                repeat: false,
            })
        }
    }
}

fn root_of(tx: &Transaction<'_>, note: &str) -> Result<String> {
    tx.query_row("SELECT root FROM note WHERE id = ?1", [note], |r| r.get(0))
        .optional()?
        .ok_or_else(|| StoreError::NotFound(format!("note {note}")))
}

fn add_note(
    tx: &Transaction<'_>,
    r: &UserRequest,
    note: &NewNote,
    seq: i64,
) -> Result<UserOutcome> {
    if note.targets.is_empty() {
        return Err(StoreError::Invalid(
            "a note needs at least one target".into(),
        ));
    }
    let root = match &note.revises {
        Some(prev) => root_of(tx, prev)?,
        None => r.id.clone(),
    };
    tx.execute(
        "INSERT INTO note(id, root, kind, key, value, author, machine, created_at, valid_until, commit_seq)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
        params![
            r.id,
            root,
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
            "INSERT OR IGNORE INTO note_target(note_id, ref) VALUES (?1, ?2)",
            params![r.id, target.to_string()],
        )?;
    }
    if let Some(prev) = &note.revises {
        tx.execute(
            "INSERT INTO note_relation(note_id, kind, other) VALUES (?1, 'revises', ?2)",
            params![r.id, format!("note:{prev}")],
        )?;
    }
    for (relation, other) in &note.relations {
        tx.execute(
            "INSERT OR IGNORE INTO note_relation(note_id, kind, other) VALUES (?1, ?2, ?3)",
            params![r.id, relation.as_str(), other],
        )?;
    }
    Ok(UserOutcome {
        id: r.id.clone(),
        repeat: false,
    })
}

#[cfg(test)]
mod tests;
