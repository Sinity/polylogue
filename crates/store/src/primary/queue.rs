//! The primary producer's durable queue. Encoded payload files are immutable;
//! this database owns scheduling, acknowledgements and membership.
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
    pub reading_version: Option<String>,
    pub input: Option<String>,
    pub format: Option<String>,
    pub export_keys: Vec<String>,
    pub stored_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Acquisition,
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
            Self::Acquisition => "acquisition",
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

pub enum Lookup<'a> {
    Name(&'a super::NamedProductSelector),
    GroupHash(&'a str),
    Members(&'a str),
    Acquisition(&'a str),
    Source {
        source: &'a [u8],
        raw: bool,
        reading_version: Option<&'a str>,
    },
    Unattributed,
    RawGeneration(&'a [u8]),
    RawChildren(&'a str),
    RawRoots(&'a [u8]),
    Export {
        input: &'a str,
        format: &'a str,
        key: Option<&'a str>,
    },
}

pub enum Order {
    OldestFirst,
    NewestFirst,
}

pub struct Queue {
    path: PathBuf,
    connection: Connection,
}

/// A read transaction fixes the database at the producer backup cut. Copying
/// it later does not hold the producer's queue mutex or stop preparation.
pub struct Snapshot {
    connection: Connection,
}
impl Snapshot {
    pub fn write_to(self, destination: &Path) -> Result<()> {
        self.connection
            .backup(rusqlite::MAIN_DB, destination, None)?;
        std::fs::File::open(destination)?.sync_all()?;
        Ok(())
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
  reading_version TEXT,
  input TEXT,
  format TEXT,
  export_keys TEXT NOT NULL,
  stored_bytes INTEGER NOT NULL
) STRICT;
CREATE INDEX queue_ready ON operation(ready,ordering);
CREATE INDEX queue_material_dependency ON operation(previous_material);
CREATE INDEX queue_confirmed ON operation(phase,position,ordering);
CREATE INDEX queue_names ON operation(logical_name,ordering);
CREATE INDEX queue_members ON operation(confirmed_group,ordering);
CREATE INDEX queue_acquisition ON operation(acquisition,ordinal);
CREATE INDEX queue_source ON operation(source,ordering);
CREATE INDEX queue_raw_generation ON operation(raw_generation,ordering);
CREATE INDEX queue_raw_child ON operation(raw_predecessor,ordering);
CREATE INDEX queue_export_input ON operation(input,format,ordering);
PRAGMA application_id=1347178801;
PRAGMA user_version=1;
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

fn refresh_ready(connection: &Connection, identity: &str) -> Result<()> {
    connection.execute(
        "UPDATE operation AS current SET ready=(
      phase='pending' AND unresolved=0 AND kind!='acquisition'
      AND NOT EXISTS(SELECT 1 FROM operation AS member
        WHERE member.confirmed_group=current.identity
          AND (member.phase!='acknowledged' OR member.unresolved!=0))
      AND NOT EXISTS(SELECT 1 FROM operation AS previous
        WHERE previous.material=current.previous_material AND previous.phase!='acknowledged'))
      WHERE identity=?1",
        [identity],
    )?;
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
            let tables: u64 = connection.query_row(
                "SELECT count(*) FROM sqlite_master WHERE type='table'",
                [],
                |row| row.get(0),
            )?;
            if tables != 0 {
                return Err(StoreError::Invalid(
                    "producer queue has an unknown format".into(),
                ));
            }
            let transaction = connection.unchecked_transaction()?;
            transaction.execute_batch(SCHEMA)?;
            transaction.commit()?;
        } else if version != 1
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
        })
    }

    /// Files have reached durable storage before this queue publication.
    pub fn prepare(
        &self,
        operation: &Operation,
        group_hash: Option<&str>,
        members: &[String],
    ) -> Result<Entry> {
        let transaction = self.connection.unchecked_transaction()?;
        transaction.execute("INSERT INTO operation(identity,directory,metadata,kind,logical_name,acquisition,ordinal,unresolved,source,raw_generation,raw_predecessor,raw_root,reading_version,input,format,export_keys,stored_bytes,group_hash,material,previous_material,raw)
          VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19,?20,?21)", params![operation.identity, operation.directory, operation.metadata, operation.kind.name(), operation.logical_name.as_ref().map(|name| name.as_bytes()), operation.acquisition, operation.ordinal, operation.unresolved, operation.source, operation.raw_generation, operation.raw_predecessor, operation.raw_root, operation.reading_version, operation.input, operation.format, serde_json::to_string(&operation.export_keys)?, operation.stored_bytes, group_hash, operation.material, operation.previous_material, operation.raw])?;
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
            .ok_or_else(|| StoreError::Invalid("published queue operation disappeared".into()))
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
    /// pending operations. An unpositioned acknowledgement is a candidate
    /// only when the caller established that the confirmed stream caught up.
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
               AND (position<=?2 OR (?3 AND position IS NULL))
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
            Lookup::Name(_) | Lookup::Source { .. } | Lookup::Unattributed | Lookup::Export { .. }
        );
        let text = |value: &str| Value::Text(value.into());
        let blob = |value: &[u8]| Value::Blob(value.into());
        let (condition, mut parameters) = match lookup {
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
            Lookup::Acquisition(identity) => ("acquisition=?1", vec![text(identity)]),
            Lookup::Source {
                source,
                raw,
                reading_version,
            } => (
                "source=?1 AND (?2=0 OR raw=1) AND (?3 IS NULL OR reading_version=?3) AND unresolved=0",
                vec![
                    blob(source),
                    Value::Integer(i64::from(raw)),
                    reading_version.map(text).unwrap_or(Value::Null),
                ],
            ),
            Lookup::Unattributed => ("input IS NOT NULL AND source IS NULL", vec![]),
            Lookup::RawGeneration(generation) => ("raw_generation=?1", vec![blob(generation)]),
            Lookup::RawChildren(previous) => ("raw_predecessor=?1", vec![text(previous)]),
            Lookup::RawRoots(generation) => {
                ("raw_generation=?1 AND raw_root=1", vec![blob(generation)])
            }
            Lookup::Export { input, format, key } => (
                "input=?1 AND format=?2 AND (?3 IS NULL OR EXISTS(SELECT 1 FROM json_each(export_keys) WHERE value=?3))",
                vec![
                    text(input),
                    text(format),
                    key.map(text).unwrap_or(Value::Null),
                ],
            ),
        };
        let base = parameters.len();
        let number = |value: u64| {
            i64::try_from(value)
                .map(Value::Integer)
                .map_err(|_| StoreError::Invalid("queue order is out of range".into()))
        };
        parameters.push(number(after.unwrap_or(0))?);
        parameters.push(number(through.unwrap_or(i64::MAX as u64))?);
        parameters.push(Value::Integer(i64::try_from(limit).unwrap_or(i64::MAX)));
        let publishable = if exclude_refused {
            " AND phase NOT IN ('refused','conflicted') AND NOT EXISTS(SELECT 1 FROM operation AS parent WHERE parent.identity=operation.confirmed_group AND parent.phase IN ('refused','conflicted'))"
        } else {
            ""
        };
        let direction = match order {
            Order::OldestFirst => "ASC",
            Order::NewestFirst => "DESC",
        };
        let sql = format!(
            "SELECT {COLUMNS} FROM operation WHERE {condition}{publishable} AND ordering>?{} AND ordering<=?{} ORDER BY ordering {direction} LIMIT ?{}",
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
    /// determine eligibility; a refused operation never blocks another source.
    pub fn ready(&self, in_flight: &[String], limit: usize) -> Result<Vec<Entry>> {
        let mut excluded = in_flight.to_vec();
        let mut result = Vec::new();
        for index in 0..limit {
            let direction = if index % 2 == 0 { "DESC" } else { "ASC" };
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
            excluded.push(next.identity.clone());
            result.push(next);
        }
        Ok(result)
    }

    /// Persist the already validated complete acknowledgement once. Original
    /// encoded acknowledgement bytes stay with the immutable operation files.
    pub fn acknowledge(
        &self,
        identity: &str,
        phase: &Phase,
        admitted: &[String],
        acknowledgement_bytes: u64,
    ) -> Result<()> {
        if phase.is_pending() {
            return Err(StoreError::Invalid(
                "acknowledgement has no finished outcome".into(),
            ));
        }
        let transaction = self.connection.unchecked_transaction()?;
        let changed = transaction.execute("UPDATE operation SET phase=?1,offered=?2,position=?3,admitted=?4,stored_bytes=stored_bytes+?5,ready=0 WHERE identity=?6 AND phase='pending'", params![phase.name(), match phase { Phase::Conflicted { offered } => Some(offered.as_str()), _ => None }, phase.commit_seq(), serde_json::to_string(admitted)?, acknowledgement_bytes, identity])?;
        if changed != 1 {
            return Err(StoreError::Invalid(
                "acknowledgement has no pending operation".into(),
            ));
        }
        let dependents = {
            let mut query = transaction.prepare_cached("SELECT confirmed_group FROM operation WHERE identity=?1 AND confirmed_group IS NOT NULL
          UNION SELECT child.identity FROM operation AS child JOIN operation AS parent ON parent.material=child.previous_material WHERE parent.identity=?1")?;
            query
                .query_map([identity], |row| row.get::<_, String>(0))?
                .collect::<rusqlite::Result<Vec<_>>>()?
        };
        for dependent in dependents {
            refresh_ready(&transaction, &dependent)?;
        }
        transaction.commit()?;
        Ok(())
    }

    pub fn counts(&self) -> Result<(u64, u64, u64)> {
        Ok(self.connection.query_row("SELECT coalesce(sum(kind IN ('material','request')),0),coalesce(sum(kind IN ('events','replay')),0),coalesce(sum(stored_bytes),0) FROM operation", [], |row| Ok((row.get(0)?,row.get(1)?,row.get(2)?)))?)
    }
}

#[cfg(test)]
mod tests;
