//! A build's sessions composed on several threads (docs/architecture.md,
//! one path). The stream's reader groups whole sessions into batches; each
//! composer composes a batch with the composition upkeep uses, into an
//! index of its own held in memory (the sessions' summaries are computed
//! once the build is installed), and hands the batch's rows back with keys that count from one. The writer only appends
//! them, in stream order, adding to each key the keys its table already
//! gave, and the composer's index is rolled back for the next batch.
use super::super::shown::{Composer, Context};
use super::Piece;
use crate::change_summary::ChangeSummary;
use crate::{Result, StoreError};
use rusqlite::types::Value;
use rusqlite::{Connection, OptionalExtension as _, Transaction};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};

/// Messages a batch gathers before it is handed to a composer, at session
/// ends.
pub(super) const BATCH_MESSAGES: usize = 1_000;

/// Tables composition writes that are not taken from the composer: the
/// build file's own triggers fill `session_cwd_member` when a session row
/// is appended, and the build marks every session's summaries due once
/// its sessions are written.
const MADE_BY_TRIGGERS: &[&str] = &["summary_due", "session_cwd_member", "session_lineage"];

/// Whole sessions of the stream, in stream order, numbered from zero.
#[derive(Default)]
pub(super) struct Batch {
    pub seq: u64,
    pub pieces: Vec<Piece>,
    /// What the pieces were decoded from (see `ReadAhead`).
    pub bytes: usize,
    pub messages: usize,
}

/// A batch composed: each table's rows, keyed from one, the events left
/// for the index to compose, and the sessions composed.
pub(super) struct Composed {
    pub seq: u64,
    pub bytes: usize,
    pub rows: Vec<(usize, Vec<Vec<Value>>)>,
    pub late: Vec<(Option<String>, String)>,
    pub sessions: u64,
}

/// What every composer starts from: the build file's inputs (each one's
/// key and newest chain read to its end or waiting), names, links and the
/// write counter the batches are composed under.
pub(super) struct Shared {
    pub inputs: HashMap<String, (i64, Option<i64>)>,
    pub aliases: Arc<super::super::identity::Aliases>,
    pub shared_events: Arc<HashSet<String>>,
    /// The build file, whose content groups composition reads.
    pub build: std::path::PathBuf,
    pub counter: i64,
    /// The rows a session's links are read from (`link_statement`, `session_link`, `prefix_copy`),
    /// as the build file holds them once links are composed.
    pub links: Vec<(String, Vec<Vec<Value>>)>,
}

impl Shared {
    /// What the composers of the build file `index` start from.
    pub(super) fn read(
        index: &Connection,
        aliases: Arc<super::super::identity::Aliases>,
        shared_events: Arc<HashSet<String>>,
        counter: i64,
    ) -> Result<Self> {
        let build = index
            .path()
            .filter(|path| !path.is_empty())
            .map(std::path::PathBuf::from)
            .ok_or_else(|| StoreError::Invalid("a build file has no path".into()))?;
        let inputs = index
            .prepare(&format!(
                "SELECT i.id,i.pk,{} FROM input_ref i",
                super::NEWEST_READ_CHAIN
            ))?
            .query_map([], |r| Ok((r.get(0)?, (r.get(1)?, r.get(2)?))))?
            .collect::<rusqlite::Result<_>>()?;
        let mut links = Vec::new();
        for table in ["link_statement", "session_link", "prefix_copy"] {
            let mut statement = index.prepare(&format!("SELECT * FROM {table}"))?;
            let columns = statement.column_count();
            let rows = statement
                .query_map([], |r| {
                    (0..columns)
                        .map(|n| r.get::<_, Value>(n))
                        .collect::<rusqlite::Result<Vec<_>>>()
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            links.push((table.to_owned(), rows));
        }
        Ok(Self {
            inputs,
            aliases,
            shared_events,
            build,
            counter,
            links,
        })
    }

    fn input(&self, id: &str) -> Result<(i64, Option<i64>)> {
        self.inputs
            .get(id)
            .copied()
            .ok_or_else(|| StoreError::Invalid("a reading names an input the build lacks".into()))
    }
}

/// How a column's value is carried from a composer's index into the build
/// file.
#[derive(Clone, Copy)]
enum Key {
    /// As it is.
    Kept,
    /// The table's own row key.
    Own,
    /// A key of the table at this place in [`Tables`].
    Of(usize),
}

/// One table the composers' rows are taken from.
pub(super) struct Table {
    name: String,
    keys: Vec<Key>,
    /// Whether its row key is `pk`.
    keyed: bool,
    select: String,
    insert: String,
}

/// The tables composition writes, with how their keys are carried: those
/// holding a session's rows (with a session, message, event or block key),
/// the session table first.
pub(super) struct Tables(Vec<Table>);

impl Tables {
    pub(super) fn of(conn: &Connection) -> Result<Self> {
        let mut names: Vec<String> = conn
            .prepare(
                "SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%'
                 AND sql NOT LIKE 'CREATE VIRTUAL TABLE%' ORDER BY name<>'session',name",
            )?
            .query_map([], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        names.retain(|name| !MADE_BY_TRIGGERS.contains(&name.as_str()));
        let columns = |name: &str| -> Result<Vec<String>> {
            Ok(conn
                .prepare(&format!(
                    "SELECT name FROM pragma_table_info('{name}') ORDER BY cid"
                ))?
                .query_map([], |r| r.get(0))?
                .collect::<rusqlite::Result<_>>()?)
        };
        let mut found: Vec<(String, Vec<String>)> = Vec::new();
        for name in names {
            let columns = columns(&name)?;
            let holds_session = name == "session"
                || columns
                    .iter()
                    .any(|c| c == "session_pk" || c == "message_pk");
            if holds_session {
                found.push((name, columns));
            }
        }
        let place = |table: &str| found.iter().position(|(name, _)| name == table);
        let mut tables = Vec::new();
        for (name, columns) in &found {
            let keys = columns
                .iter()
                .map(|column| {
                    let of = match column.as_str() {
                        "pk" => return Ok(Key::Own),
                        "session_pk" => "session",
                        "message_pk" => "message",
                        "event_pk" => "event",
                        "call_block_pk" | "evidence_block_pk" | "result_block_pk" => "block",
                        _ if column.ends_with("_pk") && column != "input_pk" => {
                            return Err(StoreError::internal(format!(
                                "composition writes {name}.{column}, whose keys no composer carries"
                            )));
                        }
                        _ => return Ok(Key::Kept),
                    };
                    place(of).map(Key::Of).ok_or_else(|| {
                        StoreError::internal(format!(
                            "{name}.{column} names a table no composer fills"
                        ))
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let quoted = columns
                .iter()
                .map(|c| format!("\"{c}\""))
                .collect::<Vec<_>>()
                .join(",");
            let marks = (1..=columns.len())
                .map(|n| format!("?{n}"))
                .collect::<Vec<_>>()
                .join(",");
            let keyed = columns.iter().any(|c| c == "pk");
            tables.push(Table {
                name: name.clone(),
                keys,
                keyed,
                select: format!("SELECT {quoted} FROM main.\"{name}\""),
                insert: format!("INSERT INTO main.\"{name}\"({quoted}) VALUES ({marks})"),
            });
        }
        Ok(Self(tables))
    }
}

/// One composer thread's index, held in memory, and its connection to the
/// build file, from which it copies the content groups a batch's readings
/// lie in before composing it (each read a statement of its own, so the
/// writer's commits wait for no composer's batch).
struct Composing {
    conn: Connection,
    build: Connection,
    shared: Arc<Shared>,
    tables: Arc<Tables>,
}

impl Composing {
    fn open(shared: Arc<Shared>, tables: Arc<Tables>) -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        crate::schema::ensure_build_index(&conn)?;
        conn.pragma_update(None, "foreign_keys", false)?;
        crate::meta_set(&conn, "write_counter", shared.counter.to_string())?;
        for (table, rows) in &shared.links {
            let Some(first) = rows.first() else {
                continue;
            };
            let marks = (1..=first.len())
                .map(|n| format!("?{n}"))
                .collect::<Vec<_>>()
                .join(",");
            let mut insert = conn.prepare(&format!("INSERT INTO {table} VALUES ({marks})"))?;
            for row in rows {
                insert.execute(rusqlite::params_from_iter(row))?;
            }
        }
        // Read-write only so SQLite may recover the build file's journal,
        // as the build's other readers may; it writes nothing.
        let build = Connection::open_with_flags(
            &shared.build,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_WRITE | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;
        build.pragma_update(None, "query_only", true)?;
        // A read waits for the writer's transaction, an index creation among
        // them, for as long as it runs.
        build.busy_timeout(std::time::Duration::from_secs(3600))?;
        Ok(Self {
            conn,
            build,
            shared,
            tables,
        })
    }

    /// Composes `batch` in one transaction it then rolls back, and takes
    /// its rows.
    fn compose(&mut self, batch: Batch) -> Result<Composed> {
        let shared = Arc::clone(&self.shared);
        let tx = Transaction::new_unchecked(&self.conn, rusqlite::TransactionBehavior::Immediate)?;
        // A session an event of which follows a message stated under another
        // session is composed from the index (see `MARKS`).
        // The content groups the batch's readings lie in, as the build file
        // holds them: composition reads each reading's content.
        let mut groups = std::collections::BTreeSet::new();
        for piece in &batch.pieces {
            match piece {
                Piece::Message { readings, .. } => {
                    groups.extend(readings.iter().map(|r| r.stored.group));
                }
                Piece::Event(event) => {
                    groups.insert(event.piece.group);
                }
                _ => {}
            }
        }
        {
            let mut read = self
                .build
                .prepare_cached("SELECT hash,body FROM content_group WHERE id=?1")?;
            let mut copy =
                tx.prepare_cached("INSERT INTO content_group(id,hash,body) VALUES (?1,?2,?3)")?;
            for group in groups {
                let (hash, body): (Vec<u8>, Vec<u8>) = read
                    .query_row([group], |r| <(Vec<u8>, Vec<u8>)>::try_from(r))
                    .optional()?
                    .ok_or_else(|| {
                        StoreError::Invalid("a reading names a group the build lacks".into())
                    })?;
                copy.execute(rusqlite::params![group, hash, body])?;
            }
        }
        let changes = ChangeSummary::start(&tx)?;
        let context = Context {
            tx: &tx,
            counter: shared.counter,
            changes: &changes,
        };
        let mut composer = Composer::new(
            &context,
            Arc::clone(&shared.aliases),
            Arc::clone(&shared.shared_events),
        )?;
        let mut kinds = HashMap::new();
        let mut sessions = 0;
        for piece in batch.pieces {
            match piece {
                Piece::Begin => {
                    composer.begin();
                    kinds.clear();
                }
                Piece::Facts { input, chain, kind } => {
                    composer.facts();
                    kinds.insert((shared.input(&input)?.0, chain), kind);
                }
                Piece::Message { id, mut readings } => {
                    for reading in &mut readings {
                        let (pk, finished) = shared.input(&reading.input)?;
                        reading.input_pk = pk;
                        if let Some(given) = &mut reading.given {
                            given.kind = kinds.get(&(pk, reading.chain)).copied().flatten();
                            given.superseded =
                                finished.is_some_and(|newest| newest > reading.chain);
                        }
                    }
                    composer.message(&id, readings)?;
                }
                Piece::Event(mut event) => {
                    event.input_pk = shared.input(&event.input)?.0;
                    composer.event(event);
                }
                Piece::Outcome(mut outcome) => {
                    outcome.input_pk = shared.input(&outcome.input)?.0;
                    composer.outcome(outcome);
                }
                Piece::End(session) => {
                    composer.end(&session)?;
                    sessions += 1;
                }
            }
        }
        let (_, late) = composer.finish();
        super::super::rows::finish_summaries(&tx, &changes, shared.counter)?;
        let mut rows = Vec::new();
        for (n, table) in self.tables.0.iter().enumerate() {
            let mut statement = tx.prepare_cached(&table.select)?;
            let width = statement.column_count();
            let found = statement
                .query_map([], |r| {
                    (0..width)
                        .map(|n| r.get::<_, Value>(n))
                        .collect::<rusqlite::Result<Vec<_>>>()
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            if !found.is_empty() {
                rows.push((n, found));
            }
        }
        tx.rollback()?;
        // The batch's pages go back, so a composer holds one batch at most.
        self.conn.execute_batch("PRAGMA shrink_memory")?;
        Ok(Composed {
            seq: batch.seq,
            bytes: batch.bytes,
            rows,
            late,
            sessions,
        })
    }
}

/// The composer threads and what they hand back.
pub(super) struct Pool {
    results: Receiver<Result<Composed>>,
    /// Batches composed ahead of the one the writer appends next.
    waiting: BTreeMap<u64, Composed>,
    next: u64,
    tables: Arc<Tables>,
    /// The next row key of each keyed table in the build file.
    keys: Vec<i64>,
}

impl Pool {
    /// Starts `count` composers taking batches from `batches`, to append to
    /// the build file `index`.
    pub(super) fn start(
        count: usize,
        shared: Shared,
        batches: Receiver<Result<Batch>>,
        index: &Connection,
    ) -> Result<Self> {
        let template = Connection::open_in_memory()?;
        crate::schema::ensure_build_index(&template)?;
        let tables = Arc::new(Tables::of(&template)?);
        let keys = tables
            .0
            .iter()
            .map(|table| -> Result<i64> {
                if table.keyed {
                    Ok(index.query_row(
                        &format!("SELECT coalesce(max(pk),0) FROM main.\"{}\"", table.name),
                        [],
                        |r| r.get(0),
                    )?)
                } else {
                    Ok(0)
                }
            })
            .collect::<Result<_>>()?;
        let shared = Arc::new(shared);
        let batches = Arc::new(Mutex::new(batches));
        let (sender, results) = std::sync::mpsc::channel();
        for n in 0..count.max(1) {
            let (shared, tables, batches, sender): (_, _, _, Sender<Result<Composed>>) = (
                Arc::clone(&shared),
                Arc::clone(&tables),
                Arc::clone(&batches),
                sender.clone(),
            );
            std::thread::Builder::new()
                .name(format!("polylogue-compose-{n}"))
                .spawn(move || {
                    let mut composing = match Composing::open(shared, tables) {
                        Ok(composing) => composing,
                        Err(error) => {
                            let _ = sender.send(Err(error));
                            return;
                        }
                    };
                    loop {
                        let batch = match batches.lock() {
                            Ok(batches) => batches.recv(),
                            Err(_) => return,
                        };
                        let Ok(batch) = batch else { return };
                        let composed = batch.and_then(|batch| {
                            let _doing = crate::activity::doing("composing sessions");
                            composing.compose(batch)
                        });
                        let failed = composed.is_err();
                        if sender.send(composed).is_err() || failed {
                            return;
                        }
                    }
                })?;
        }
        Ok(Self {
            results,
            waiting: BTreeMap::new(),
            next: 0,
            tables,
            keys,
        })
    }

    /// The next batch in stream order, once composed; `None` once every
    /// batch is.
    pub(super) fn next(&mut self) -> Result<Option<Composed>> {
        loop {
            if let Some(composed) = self.waiting.remove(&self.next) {
                self.next += 1;
                return Ok(Some(composed));
            }
            match self.results.recv() {
                Ok(composed) => {
                    let composed = composed?;
                    self.waiting.insert(composed.seq, composed);
                }
                Err(_) if self.waiting.is_empty() => return Ok(None),
                Err(_) => {
                    return Err(StoreError::internal(
                        "a composer ended before its batch was composed",
                    ));
                }
            }
        }
    }

    /// Appends a composed batch to the build file, each key carried past
    /// the keys its table already holds.
    pub(super) fn append(
        &mut self,
        tx: &Transaction<'_>,
        rows: Vec<(usize, Vec<Vec<Value>>)>,
    ) -> Result<()> {
        let offsets = self.keys.clone();
        for (n, mut rows) in rows {
            let table = &self.tables.0[n];
            let mut insert = tx.prepare_cached(&table.insert)?;
            let mut most = 0;
            for row in &mut rows {
                for (value, key) in row.iter_mut().zip(&table.keys) {
                    if let Value::Integer(local) = value {
                        match key {
                            Key::Own => {
                                most = most.max(*local);
                                *local += offsets[n];
                            }
                            Key::Of(of) => *local += offsets[*of],
                            Key::Kept => {}
                        }
                    }
                }
                insert.execute(rusqlite::params_from_iter(row.iter()))?;
            }
            if table.keyed {
                self.keys[n] = offsets[n] + most;
            }
        }
        Ok(())
    }
}
