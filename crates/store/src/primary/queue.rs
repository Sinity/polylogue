//! The producer spool's queue. Encoded payload files never change; this
//! database holds scheduling, acknowledgments and membership.
use crate::ConnExt as _;
use crate::{OptionExt as _, ResultExt as _};
use crate::{Result, StoreError, archive};
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "phase", rename_all = "snake_case")]
pub enum Phase {
    Pending,
    Acknowledged { commit_seq: Option<i64> },
    Refused,
    Conflicted { offered: String },
}

impl Phase {
    pub fn is_pending(&self) -> bool {
        matches!(self, Self::Pending)
    }
    pub fn is_acknowledged(&self) -> bool {
        matches!(self, Self::Acknowledged { .. })
    }
    pub fn is_refused(&self) -> bool {
        matches!(self, Self::Refused | Self::Conflicted { .. })
    }
    pub fn commit_seq(&self) -> Option<i64> {
        match self {
            Self::Acknowledged { commit_seq } => *commit_seq,
            _ => None,
        }
    }
    fn name(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Acknowledged { .. } => "acknowledged",
            Self::Refused => "refused",
            Self::Conflicted { .. } => "conflicted",
        }
    }
}

/// Immutable operation descriptors and the fields queried by scheduling.
/// This contains no original frame, request or source body bytes.
pub struct Operation {
    pub identity: String,
    pub directory: String,
    pub metadata: Vec<u8>,
    pub kind: Kind,
    pub material: Option<String>,
    pub previous_material: Option<String>,
    pub logical_name: Option<String>,
    pub acquisition: Option<String>,
    pub ordinal: Option<u64>,
    pub unresolved: bool,
    pub source: Option<Vec<u8>>,
    pub raw: bool,
    pub raw_generation: Option<Vec<u8>>,
    pub raw_predecessor: Option<String>,
    pub raw_root: bool,
    pub input: Option<String>,
    pub format: Option<String>,
    pub stored_bytes: u64,
    pub completion: Option<Completion>,
    pub captured: bool,
    pub blob: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    CaptureGroupHeader,
    Material,
    Request,
    Events,
    Replay,
    Registration,
    Signals,
    Chunk,
    LargeFrame,
}
impl Kind {
    fn name(self) -> &'static str {
        match self {
            Self::CaptureGroupHeader => "acquisition",
            Self::Material => "material",
            Self::Request => "request",
            Self::Events => "events",
            Self::Replay => "replay",
            Self::Registration => "registration",
            Self::Signals => "signals",
            Self::Chunk => "chunk",
            Self::LargeFrame => "large_frame",
        }
    }
}

#[derive(Debug)]
pub struct Entry {
    pub order: u64,
    pub identity: String,
    pub directory: String,
    pub metadata: Vec<u8>,
    pub phase: Phase,
    pub unresolved: bool,
}

/// An acknowledged operation whose admitted events can now be checked
/// against the confirmed index. Payload files are not read by this query.
pub struct Confirmed {
    pub entry: Entry,
    pub admitted: Vec<String>,
    pub material: bool,
    pub group: bool,
}

#[derive(Clone, Copy, Debug)]
pub enum Completion {
    Request,
    Observation,
    Reading,
}
impl Completion {
    fn name(self) -> &'static str {
        match self {
            Self::Request => "request",
            Self::Observation => "observation",
            Self::Reading => "reading",
        }
    }
}

#[derive(Clone, Copy)]
pub enum Lookup<'a> {
    IdentityPrefix(&'a str),
    Completion(Completion),
    Requests,
    Blobs(Option<&'a str>),
    Captures,
    Name(&'a super::NamedProductSelector),
    GroupHash(&'a str),
    Members(&'a str),
    CaptureGroupHeader(&'a str),
    Source { source: &'a [u8], raw: bool },
    Unattributed,
    ObservedSourceKey(&'a [u8]),
    RawChildren(&'a str),
    RawRoots(&'a [u8]),
}

pub enum Order {
    OldestFirst,
    NewestFirst,
}

pub struct Queue {
    path: PathBuf,
    connection: Connection,
    newest_next: std::cell::Cell<bool>,
}

/// A read transaction fixes the database at the producer backup position. Copying
/// it later does not hold the producer's queue mutex or stop preparation.
pub struct Snapshot {
    connection: Connection,
}
impl Snapshot {
    pub fn write_to(
        self,
        destination: &Path,
        cancelled: &std::sync::atomic::AtomicBool,
    ) -> Result<()> {
        use rusqlite::backup::{Backup, StepResult};
        let check = || {
            if cancelled.load(std::sync::atomic::Ordering::Acquire) {
                Err(StoreError::Invalid("producer backup cancelled".into()))
            } else {
                Ok(())
            }
        };
        check()?;
        let mut target = Connection::open(destination)?;
        {
            let backup = Backup::new(&self.connection, &mut target)?;
            loop {
                check()?;
                match backup.step(128)? {
                    StepResult::Done => break,
                    StepResult::More => {}
                    StepResult::Busy | StepResult::Locked => {
                        std::thread::sleep(std::time::Duration::from_millis(10))
                    }
                    _ => return Err(StoreError::Invalid("unknown producer backup step".into())),
                }
            }
        }
        drop(target);
        std::fs::File::open(destination)?.sync_all()?;
        check()
    }
}

const SCHEMA: &str = "
CREATE TABLE operation (
  ordering INTEGER PRIMARY KEY AUTOINCREMENT,
  identity TEXT NOT NULL UNIQUE,
  directory TEXT NOT NULL,
  metadata BLOB NOT NULL,
  kind TEXT NOT NULL,
  material TEXT UNIQUE,
  previous_material TEXT,
  ready INTEGER NOT NULL DEFAULT 0,
  phase TEXT NOT NULL DEFAULT 'pending',
  position INTEGER,
  offered TEXT,
  admitted TEXT NOT NULL DEFAULT '[]',
  logical_name BLOB,
  acquisition TEXT,
  ordinal INTEGER,
  confirmed_group TEXT,
  group_hash TEXT UNIQUE,
  unresolved INTEGER NOT NULL,
  source BLOB,
  raw INTEGER NOT NULL,
  raw_generation BLOB,
  raw_predecessor TEXT,
  raw_root INTEGER NOT NULL,
  input TEXT,
  format TEXT,
  stored_bytes INTEGER NOT NULL,
  completion TEXT,
  captured INTEGER NOT NULL,
  blob TEXT
) STRICT;
CREATE INDEX queue_completion ON operation(completion,phase,confirmed_group,ordering);
CREATE INDEX queue_blobs ON operation(blob,ordering);
CREATE INDEX queue_requests ON operation(kind,ordering);
CREATE INDEX queue_captures ON operation(captured,ordering);
CREATE INDEX queue_ready ON operation(ready,ordering);
CREATE INDEX queue_material_dependency ON operation(previous_material);
CREATE INDEX queue_confirmed ON operation(phase,position,ordering);
CREATE INDEX queue_names ON operation(logical_name,ordering);
CREATE INDEX queue_members ON operation(confirmed_group,ordering);
CREATE INDEX queue_acquisition ON operation(acquisition,ordinal);
CREATE INDEX queue_source ON operation(source,ordering);
CREATE INDEX queue_raw_generation ON operation(raw_generation,ordering);
CREATE INDEX queue_raw_child ON operation(raw_predecessor,ordering);
CREATE INDEX queue_input ON operation(input,format,ordering);
PRAGMA application_id=1347178801;
PRAGMA user_version=2;
";

const COLUMNS: &str = "ordering,identity,directory,metadata,phase,position,offered,unresolved";

fn entry(row: &rusqlite::Row<'_>) -> rusqlite::Result<Entry> {
    let phase: String = row.get(4)?;
    let phase = match phase.as_str() {
        "pending" => Phase::Pending,
        "acknowledged" => Phase::Acknowledged {
            commit_seq: row.get(5)?,
        },
        "refused" => Phase::Refused,
        "conflicted" => Phase::Conflicted {
            offered: row.get(6)?,
        },
        _ => return Err(rusqlite::Error::InvalidQuery),
    };
    Ok(Entry {
        order: row.get(0)?,
        identity: row.get(1)?,
        directory: row.get(2)?,
        metadata: row.get(3)?,
        phase,
        unresolved: row.get(7)?,
    })
}

/// Identities of parsed frames for one original chain begin with this prefix:
/// `parsed:`, the chain's 64-digit hash and a colon. Sinex numbers readings in
/// the order it admits them, so the frames of one chain are sent one at a time
/// in queue order: a frame is ready only when no earlier frame of its chain is
/// still pending. Frames of different chains proceed concurrently.
pub fn parsed_chain_prefix(root_material: &str, input: &str, chain: i64) -> Result<String> {
    Ok(format!(
        "parsed:{}:",
        blake3::hash(&serde_json::to_vec(&(root_material, input, chain))?).to_hex()
    ))
}
const PARSED_CHAIN_PREFIX_LENGTH: usize = 72;

/// Whether the operation `current` may be sent now: its group members and its
/// previous material are acknowledged, and no earlier frame of its parsed
/// chain is pending.
fn ready_condition() -> String {
    format!(
        "phase='pending' AND unresolved=0 AND kind!='acquisition'
      AND NOT EXISTS(SELECT 1 FROM operation AS member
        WHERE member.confirmed_group=current.identity
          AND (member.phase!='acknowledged' OR member.unresolved!=0))
      AND NOT EXISTS(SELECT 1 FROM operation AS previous
        WHERE previous.material=current.previous_material AND previous.phase!='acknowledged')
      AND NOT (current.identity LIKE 'parsed:%' AND EXISTS(SELECT 1 FROM operation AS earlier
        WHERE earlier.identity>substr(current.identity,1,{n})
          AND earlier.identity<substr(current.identity,1,{m})||';'
          AND earlier.ordering<current.ordering AND earlier.phase='pending'))",
        n = PARSED_CHAIN_PREFIX_LENGTH,
        m = PARSED_CHAIN_PREFIX_LENGTH - 1,
    )
}

fn refresh_ready(connection: &Connection, identity: &str) -> Result<()> {
    connection.execute(
        &format!(
            "UPDATE operation AS current SET ready=({}) WHERE identity=?1",
            ready_condition()
        ),
        [identity],
    )?;
    Ok(())
}

fn insert(connection: &Connection, operation: &Operation, group_hash: Option<&str>) -> Result<()> {
    connection.execute("INSERT INTO operation(identity,directory,metadata,kind,logical_name,acquisition,ordinal,unresolved,source,raw_generation,raw_predecessor,raw_root,input,format,stored_bytes,group_hash,material,previous_material,raw,completion,captured,blob)
          VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19,?20,?21,?22)", params![operation.identity, operation.directory, operation.metadata, operation.kind.name(), operation.logical_name.as_ref().map(|name| name.as_bytes()), operation.acquisition, operation.ordinal, operation.unresolved, operation.source, operation.raw_generation, operation.raw_predecessor, operation.raw_root, operation.input, operation.format, operation.stored_bytes, group_hash, operation.material, operation.previous_material, operation.raw, operation.completion.map(Completion::name), operation.captured, operation.blob])?;
    Ok(())
}

impl Queue {
    pub fn snapshot(&self) -> Result<Snapshot> {
        let connection = archive::open_read(&self.path)?;
        connection.execute_batch("BEGIN")?;
        connection.query_row(
            "SELECT count(*) FROM (SELECT ordering FROM operation LIMIT 1)",
            [],
            |row| row.get::<_, u64>(0),
        )?;
        Ok(Snapshot { connection })
    }
    pub fn open(path: &Path) -> Result<Self> {
        let connection = archive::open_write(path, archive::cache::SETUP)?;
        let version: u32 = connection.pragma_query_value(None, "user_version", |row| row.get(0))?;
        if version == 0 {
            let tables: u64 =
                connection.scalar("SELECT count(*) FROM sqlite_master WHERE type='table'", [])?;
            if tables != 0 {
                return Err(StoreError::Invalid(
                    "producer queue has an unknown format".into(),
                ));
            }
            let transaction = connection.unchecked_transaction()?;
            transaction.execute_batch(SCHEMA)?;
            transaction.commit()?;
        } else if version == 1 {
            return Err(StoreError::Invalid(format!(
                "producer queue {} has format 1; let the earlier build send and confirm every operation, then set the empty queue aside",
                path.display()
            )));
        } else if version != 2
            || connection.pragma_query_value(None, "application_id", |row| row.get::<_, u32>(0))?
                != 1347178801
        {
            return Err(StoreError::Invalid(
                "producer queue has an unknown format".into(),
            ));
        }
        connection.pragma_update(None, "synchronous", "FULL")?;
        Ok(Self {
            path: path.into(),
            connection,
            newest_next: std::cell::Cell::new(true),
        })
    }

    /// The files are synced to disk before the operation enters the queue.
    pub fn prepare(
        &self,
        operation: &Operation,
        group_hash: Option<&str>,
        members: &[String],
    ) -> Result<Entry> {
        let transaction = self.connection.unchecked_transaction()?;
        insert(&transaction, operation, group_hash)?;
        for member in members {
            let changed = transaction.execute("UPDATE operation SET confirmed_group=?1 WHERE identity=?2 AND (confirmed_group IS NULL OR confirmed_group=?1)", params![operation.identity, member])?;
            if changed != 1 {
                return Err(StoreError::Invalid(format!(
                    "capture {member} is absent or belongs to another group"
                )));
            }
        }
        refresh_ready(&transaction, &operation.identity)?;
        transaction.commit()?;
        self.get(&operation.identity)?
            .or_invalid("queued operation disappeared")
    }

    /// Adds a staged capture group to the queue in one commit. The adapter has synced and
    /// moved its immutable directory before this call. Copy queue metadata
    /// directly, without decoding every member or performing one sync per row.
    pub fn publish_capture_group(&self, staged: &Queue, header: &Operation) -> Result<()> {
        if header.kind != Kind::CaptureGroupHeader {
            return Err(StoreError::Invalid(
                "a capture group needs its header".into(),
            ));
        }
        self.connection.execute(
            "ATTACH DATABASE ?1 AS staged",
            [staged.path.as_os_str().as_encoded_bytes()],
        )?;
        let result: Result<()> = (|| {
            let tx = self.connection.unchecked_transaction()?;
            insert(&tx, header, None)?;
            tx.execute("INSERT INTO operation(identity,directory,metadata,kind,material,previous_material,phase,position,offered,admitted,logical_name,acquisition,ordinal,confirmed_group,group_hash,unresolved,source,raw,raw_generation,raw_predecessor,raw_root,input,format,stored_bytes,completion,captured,blob)
                SELECT identity,?1 || '/' || directory,metadata,kind,material,previous_material,phase,position,offered,admitted,logical_name,?2,ordering-1,confirmed_group,group_hash,?3,source,raw,raw_generation,raw_predecessor,raw_root,input,format,stored_bytes,completion,captured,blob FROM staged.operation ORDER BY ordering",
                params![header.directory, header.identity, header.unresolved])?;
            tx.execute(
                &format!(
                    "UPDATE operation AS current SET ready=({}) WHERE acquisition=?1",
                    ready_condition()
                ),
                [&header.identity],
            )?;
            tx.commit()?;
            Ok(())
        })();
        let detached = self.connection.execute("DETACH DATABASE staged", []);
        result?;
        detached?;
        Ok(())
    }

    pub fn remove(&self, identity: &str) -> Result<Vec<String>> {
        let tx = self.connection.unchecked_transaction()?;
        let mut paths = Vec::new();
        let mut groups = Vec::new();
        {
            let mut query = tx.prepare_cached("SELECT directory,acquisition FROM operation WHERE identity=?1 OR confirmed_group=?1")?;
            for row in
                query.query_map([identity], |row| <(String, Option<String>)>::try_from(row))?
            {
                let (path, acquisition) = row?;
                paths.push(path);
                if let Some(group) = acquisition {
                    groups.push(group);
                }
            }
        }
        tx.execute(
            "DELETE FROM operation WHERE identity=?1 OR confirmed_group=?1",
            [identity],
        )?;
        for group in groups {
            if tx.scalar::<bool>(
                "SELECT NOT EXISTS(SELECT 1 FROM operation WHERE acquisition=?1)",
                [&group],
            )? {
                if let Some(path) = tx.scalar_optional::<String>(
                    "SELECT directory FROM operation WHERE identity=?1",
                    [&group],
                )? {
                    paths.push(path);
                }
                tx.execute("DELETE FROM operation WHERE identity=?1", [&group])?;
            }
        }
        tx.commit()?;
        Ok(paths)
    }

    pub fn references_directory(&self, directory: &str) -> Result<bool> {
        self.connection.exists(
            "SELECT EXISTS(SELECT 1 FROM operation WHERE directory=?1)",
            [directory],
        )
    }

    pub fn group_for_member(&self, identity: &str) -> Result<Option<Entry>> {
        Ok(self.connection.query_row(&format!("SELECT {COLUMNS} FROM operation WHERE identity=(SELECT confirmed_group FROM operation WHERE identity=?1)"), [identity], entry).optional()?)
    }

    pub fn member_ordinal(&self, identity: &str) -> Result<Option<u64>> {
        Ok(self
            .connection
            .scalar_optional(
                "SELECT ordinal FROM operation WHERE identity=?1",
                [identity],
            )?
            .flatten())
    }

    pub fn capture_group_members_after(
        &self,
        identity: &str,
        after: Option<u64>,
        limit: usize,
    ) -> Result<Vec<Entry>> {
        let mut query = self.connection.prepare_cached(&format!("SELECT {COLUMNS} FROM operation WHERE acquisition=?1 AND ordinal>=?2 ORDER BY ordinal LIMIT ?3"))?;
        Ok(query
            .query_map(
                params![identity, after.map_or(0, |n| n.saturating_add(1)), limit],
                entry,
            )?
            .collect::<rusqlite::Result<_>>()?)
    }

    pub fn last_order(&self) -> Result<u64> {
        self.connection
            .scalar("SELECT coalesce(max(ordering),0) FROM operation", [])
    }

    pub fn len(&self) -> Result<u64> {
        self.connection.scalar(
            "SELECT count(*) FROM operation WHERE kind!='acquisition'",
            [],
        )
    }

    pub fn is_empty(&self) -> Result<bool> {
        self.len().map(|n| n == 0)
    }

    pub fn get(&self, identity: &str) -> Result<Option<Entry>> {
        Ok(self
            .connection
            .query_row(
                &format!("SELECT {COLUMNS} FROM operation WHERE identity=?1"),
                [identity],
                entry,
            )
            .optional()?)
    }

    /// Select by the confirmed position, not by the number of unrelated
    /// pending operations. An acknowledgment without a position is considered
    /// for a source group only when the confirmed stream caught up. Ungrouped
    /// metadata acknowledgments need no event position.
    pub fn confirmed_after(
        &self,
        applied: i64,
        caught_up: bool,
        after: Option<u64>,
        limit: usize,
    ) -> Result<Vec<Confirmed>> {
        if applied < 0 {
            return Err(StoreError::Invalid("confirmed position is negative".into()));
        }
        let mut query = self.connection.prepare_cached(&format!(
            "SELECT {COLUMNS},admitted,kind,group_hash FROM operation
             WHERE phase='acknowledged' AND ordering>?1
               AND (position<=?2 OR (position IS NULL AND ((group_hash IS NULL AND kind IN ('events','replay','registration')) OR ?3)))
             ORDER BY ordering LIMIT ?4"
        ))?;
        Ok(query
            .query_map(
                params![after.unwrap_or(0), applied, caught_up, limit],
                |row| {
                    let admitted: String = row.get(8)?;
                    let admitted = serde_json::from_str(&admitted).map_err(|error| {
                        rusqlite::Error::FromSqlConversionFailure(
                            8,
                            rusqlite::types::Type::Text,
                            Box::new(error),
                        )
                    })?;
                    let kind: String = row.get(9)?;
                    Ok(Confirmed {
                        entry: entry(row)?,
                        admitted,
                        material: matches!(kind.as_str(), "material" | "request"),
                        group: row.get::<_, Option<String>>(10)?.is_some(),
                    })
                },
            )?
            .collect::<rusqlite::Result<_>>()?)
    }

    pub fn after(&self, after: Option<u64>, limit: usize) -> Result<Vec<Entry>> {
        let mut query = self.connection.prepare_cached(&format!("SELECT {COLUMNS} FROM operation WHERE ordering>?1 AND kind!='acquisition' ORDER BY ordering LIMIT ?2"))?;
        Ok(query
            .query_map(params![after.unwrap_or(0), limit], entry)?
            .collect::<rusqlite::Result<_>>()?)
    }

    pub fn lookup(
        &self,
        lookup: Lookup<'_>,
        after: Option<u64>,
        through: Option<u64>,
        order: Order,
        limit: usize,
    ) -> Result<Vec<Entry>> {
        use rusqlite::types::Value;
        let exclude_refused = matches!(
            &lookup,
            Lookup::Name(_)
                | Lookup::Source { .. }
                | Lookup::Unattributed
                | Lookup::Requests
                | Lookup::IdentityPrefix(_)
        );
        let text = |value: &str| Value::Text(value.into());
        let blob = |value: &[u8]| Value::Blob(value.into());
        let (condition, mut parameters) = match lookup {
            Lookup::IdentityPrefix(prefix) => {
                let mut upper = prefix.to_owned();
                while let Some(last) = upper.pop() {
                    let mut next = u32::from(last) + 1;
                    if (0xD800..=0xDFFF).contains(&next) {
                        next = 0xE000;
                    }
                    if let Some(next) = char::from_u32(next) {
                        upper.push(next);
                        break;
                    }
                }
                (
                    "identity>=?1 AND (?2='' OR identity<?2)",
                    vec![text(prefix), text(&upper)],
                )
            }
            Lookup::Completion(completion) => (
                "completion=?1 AND confirmed_group IS NULL AND (phase='acknowledged' OR (?1='request' AND phase='refused'))",
                vec![text(completion.name())],
            ),
            Lookup::Requests => ("kind='request'", vec![]),
            Lookup::Blobs(hash) => (
                "blob IS NOT NULL AND (?1 IS NULL OR blob=?1)",
                vec![hash.map(text).unwrap_or(Value::Null)],
            ),
            Lookup::Captures => ("captured=1", vec![]),
            Lookup::Name(super::NamedProductSelector::Exact(name)) => {
                ("logical_name=?1", vec![blob(name.as_bytes())])
            }
            Lookup::Name(super::NamedProductSelector::Prefix(prefix)) => {
                let mut upper = prefix.as_bytes().to_vec();
                upper.push(255);
                (
                    "logical_name>=?1 AND logical_name<?2",
                    vec![blob(prefix.as_bytes()), Value::Blob(upper)],
                )
            }
            Lookup::GroupHash(hash) => ("group_hash=?1", vec![text(hash)]),
            Lookup::Members(identity) => ("confirmed_group=?1", vec![text(identity)]),
            Lookup::CaptureGroupHeader(identity) => ("acquisition=?1", vec![text(identity)]),
            Lookup::Source { source, raw } => (
                "source=?1 AND (?2=0 OR raw=1) AND unresolved=0",
                vec![blob(source), Value::Integer(i64::from(raw))],
            ),
            Lookup::Unattributed => (
                "input IS NOT NULL AND source IS NULL AND unresolved=0",
                vec![],
            ),
            Lookup::ObservedSourceKey(generation) => ("raw_generation=?1", vec![blob(generation)]),
            Lookup::RawChildren(previous) => ("raw_predecessor=?1", vec![text(previous)]),
            Lookup::RawRoots(generation) => {
                ("raw_generation=?1 AND raw_root=1", vec![blob(generation)])
            }
        };
        let base = parameters.len();
        let number = |value: u64| {
            i64::try_from(value)
                .map(Value::Integer)
                .or_invalid("queue order is out of range")
        };
        parameters.push(number(after.unwrap_or(0))?);
        parameters.push(number(through.unwrap_or(i64::MAX as u64))?);
        parameters.push(Value::Integer(i64::try_from(limit).unwrap_or(i64::MAX)));
        let sendable = if exclude_refused {
            " AND phase NOT IN ('refused','conflicted') AND NOT EXISTS(SELECT 1 FROM operation AS parent WHERE parent.identity=operation.confirmed_group AND parent.phase IN ('refused','conflicted'))"
        } else {
            ""
        };
        let direction = match order {
            Order::OldestFirst => "ASC",
            Order::NewestFirst => "DESC",
        };
        let sql = format!(
            "SELECT {COLUMNS} FROM operation WHERE {condition}{sendable} AND ordering>?{} AND ordering<=?{} ORDER BY ordering {direction} LIMIT ?{}",
            base + 1,
            base + 2,
            base + 3
        );
        let mut query = self.connection.prepare_cached(&sql)?;
        Ok(query
            .query_map(rusqlite::params_from_iter(parameters), entry)?
            .collect::<rusqlite::Result<_>>()?)
    }

    /// Interleave current capture with the oldest ready backlog. Dependencies
    /// decide which operations are ready; a refused operation never blocks
    /// another source.
    pub fn ready(&self, in_flight: &[String], limit: usize) -> Result<Vec<Entry>> {
        let mut excluded = in_flight.to_vec();
        let mut result = Vec::new();
        for _ in 0..limit {
            let direction = if self.newest_next.get() {
                "DESC"
            } else {
                "ASC"
            };
            let next = self
                .connection
                .query_row(
                    &format!(
                        "SELECT {COLUMNS} FROM operation WHERE ready=1
                  AND identity NOT IN (SELECT value FROM json_each(?1))
                  ORDER BY ordering {direction} LIMIT 1"
                    ),
                    [serde_json::to_string(&excluded)?],
                    entry,
                )
                .optional()?;
            let Some(next) = next else {
                break;
            };
            self.newest_next.set(!self.newest_next.get());
            excluded.push(next.identity.clone());
            result.push(next);
        }
        Ok(result)
    }

    /// Saves the validated, complete acknowledgment once. The encoded
    /// acknowledgment bytes stay with the operation files.
    pub fn acknowledge(
        &self,
        identity: &str,
        phase: &Phase,
        admitted: &[String],
        acknowledgment_bytes: u64,
    ) -> Result<()> {
        if phase.is_pending() {
            return Err(StoreError::Invalid(
                "acknowledgment has no finished outcome".into(),
            ));
        }
        let transaction = self.connection.unchecked_transaction()?;
        let changed = transaction.execute("UPDATE operation SET phase=?1,offered=?2,position=?3,admitted=?4,stored_bytes=stored_bytes+?5,ready=0 WHERE identity=?6 AND phase='pending'", params![phase.name(), match phase { Phase::Conflicted { offered } => Some(offered.as_str()), _ => None }, phase.commit_seq(), serde_json::to_string(admitted)?, acknowledgment_bytes, identity])?;
        if changed != 1 {
            return Err(StoreError::Invalid(
                "acknowledgment has no pending operation".into(),
            ));
        }
        let mut dependents = {
            let mut query = transaction.prepare_cached("SELECT confirmed_group FROM operation WHERE identity=?1 AND confirmed_group IS NOT NULL
          UNION SELECT child.identity FROM operation AS child JOIN operation AS parent ON parent.material=child.previous_material WHERE parent.identity=?1")?;
            query
                .query_map([identity], |row| row.get::<_, String>(0))?
                .collect::<rusqlite::Result<Vec<_>>>()?
        };
        if identity.starts_with("parsed:") && identity.len() > PARSED_CHAIN_PREFIX_LENGTH {
            let prefix = &identity[..PARSED_CHAIN_PREFIX_LENGTH];
            let mut upper = prefix[..PARSED_CHAIN_PREFIX_LENGTH - 1].to_owned();
            upper.push(';');
            let next: Option<String> = transaction
                .scalar_optional("SELECT identity FROM operation WHERE identity>?1 AND identity<?2 AND phase='pending'
                       AND ordering>(SELECT ordering FROM operation WHERE identity=?3)
                     ORDER BY ordering LIMIT 1", params![prefix, upper, identity])?;
            dependents.extend(next);
        }
        for dependent in dependents {
            refresh_ready(&transaction, &dependent)?;
        }
        transaction.commit()?;
        Ok(())
    }

    pub fn counts(&self) -> Result<(u64, u64, u64)> {
        Ok(self.connection.query_row("SELECT coalesce(sum(kind IN ('material','request')),0),coalesce(sum(kind IN ('events','replay')),0),coalesce(sum(stored_bytes),0) FROM operation", [], |row| <(_, _, _)>::try_from(row))?)
    }
}

#[cfg(test)]
mod tests;
