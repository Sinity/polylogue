//! The load: every worker's runs merged in one pass into the build file
//! (docs/architecture.md, one path). Inputs come first, then content groups,
//! then texts made one per hash with the parse text rows that hold them,
//! then every parse table in key order, then the marks that decide which
//! sessions are assembled from the index. The merge stops at the session
//! records, which assembly reads ([`super::assemble`]).
use super::worker::{
    COMPOSE, EVENT_ID, FALLBACK, FOLLOWS, GROUPS, GroupKeys, INPUTS, MARKS, MESSAGE_ID,
    ROWS_PER_TRANSACTION, STATES, TABLES, TEXT, TEXT_GROUP, TEXT_HOLDER, TEXTS, Table, decode_row,
    encode_row, key_input, split_id, tables,
};
use super::{Rebuild, RebuildStage};
use crate::sorted_runs::{Merger, Sorter};
use crate::{Result, StoreError};
use rusqlite::types::{Value, ValueRef};
use rusqlite::{Transaction, TransactionBehavior, params};
use std::collections::{BTreeSet, HashSet};

/// What the load leaves for assembly: the rest of the merge, which holds
/// the session records, each message parse's text keys, the sessions
/// assembled from the index and the event IDs several sessions state.
pub(super) struct Loaded {
    pub(super) merged: Merger,
    pub(super) texts: Merger,
    pub(super) fallback: BTreeSet<String>,
    pub(super) shared_events: HashSet<String>,
    pub(super) groups: GroupKeys,
}

/// The merge of the workers' runs, where it stands.
struct Load {
    tables: Vec<Table>,
    next_group: i64,
    /// The keys given to the workers' groups; a group stored before with the
    /// same bytes keeps its key, and what names the new one names it.
    groups: GroupKeys,
    holders: Sorter,
    input: Option<(String, i64)>,
    current_text: Option<(Vec<u8>, i64)>,
    stream: Option<u8>,
    /// Each message parse's text keys, by its key in the session records.
    reading_texts: Sorter,
    /// Sessions assembled from the index, the copy names, the message whose
    /// marks are being read with the sessions stating it so far, the last
    /// event ID's mark and the event IDs several sessions state (see
    /// [`MARKS`]).
    fallback: BTreeSet<String>,
    aliases: Option<crate::writer::identity::Aliases>,
    message: Option<(String, Vec<String>)>,
    last_event: Option<(String, String)>,
    shared_events: HashSet<String>,
}

impl Rebuild {
    /// Loads the runs into the build file: keying texts, then loading
    /// parses. Leaves the rest of the merge for assembly.
    pub(super) fn load(&mut self, runs: Vec<std::path::PathBuf>) -> Result<Loaded> {
        let paths = crate::index::file::Paths::of(&self.archive.root);
        std::fs::create_dir_all(&paths.sorted)?;
        let budget = self.archive.memory_budget();
        // A group's key is never given to another group.
        let next_group: i64 = crate::ConnExt::scalar(
            &self.conn,
            "SELECT max(coalesce((SELECT seq FROM sqlite_sequence WHERE name='content_group'),0),
                 coalesce((SELECT max(id) FROM content_group),0))+1",
            [],
        )?;
        // Text groups whose every text another worker or chain also held
        // are named by nothing once the texts are made one.
        self.conn.execute_batch(
            "CREATE TEMP TABLE IF NOT EXISTS build_text_group(id INTEGER PRIMARY KEY);
             DELETE FROM temp.build_text_group;",
        )?;
        // References are checked by the indexes and assembly after the
        // load; tables arrive in name order, not in reference order.
        self.conn.pragma_update(None, "foreign_keys", false)?;
        let mut merged = Merger::open(runs, &paths.sorted, budget.run_merge_memory())?;
        // The load's own memory: the run readers and the two sorts of text
        // holders.
        self.size_cache(merged.memory() + 2 * budget.side_merge_memory(), false)?;
        let mut load = Load {
            tables: tables(&self.conn)?,
            next_group,
            groups: GroupKeys::default(),
            holders: Sorter::new(
                &paths.sorted,
                "text-holders".into(),
                budget.side_merge_memory(),
            ),
            input: None,
            current_text: None,
            stream: None,
            reading_texts: Sorter::new(
                &paths.sorted,
                "reading-texts".into(),
                budget.side_merge_memory(),
            ),
            fallback: BTreeSet::new(),
            aliases: None,
            message: None,
            last_event: None,
            shared_events: HashSet::new(),
        };
        while self.load_transaction(&mut merged, &mut load)? {}
        crate::fault::check("index.build.loading", "")?;
        let texts = Merger::open(
            load.reading_texts.finish()?,
            &paths.sorted,
            budget.side_merge_memory(),
        )?;
        let mut holders = Merger::open(
            load.holders.finish()?,
            &paths.sorted,
            budget.side_merge_memory(),
        )?;
        self.size_cache(merged.memory() + texts.memory() + holders.memory(), false)?;
        self.stage(RebuildStage::LoadingParses);
        let table = Table::new(
            "parse_text".into(),
            [
                "input_pk", "chain", "document", "item", "position", "text_id",
            ]
            .map(String::from)
            .to_vec(),
            "",
        );
        let mut input = None;
        while self.holders_transaction(&mut holders, &table, &mut input)? {}
        Ok(Loaded {
            merged,
            texts,
            fallback: load.fallback,
            shared_events: load.shared_events,
            groups: load.groups,
        })
    }

    /// Loads up to a transaction's worth of records; false once the runs
    /// are read up to the session records.
    fn load_transaction(&mut self, merged: &mut Merger, load: &mut Load) -> Result<bool> {
        self.archive.stop().check()?;
        let t = Transaction::new_unchecked(&self.conn, TransactionBehavior::Immediate)?;
        let mut written = 0u64;
        let mut values = Vec::new();
        let more = loop {
            if written == ROWS_PER_TRANSACTION {
                break true;
            }
            // The session records are read by assembly.
            if merged
                .peek()?
                .is_none_or(|(key, _)| key.first() == Some(&COMPOSE))
            {
                break false;
            }
            let Some((key, value)) = merged.next()? else {
                break false;
            };
            let (&kind, rest) = key
                .split_first()
                .ok_or_else(|| StoreError::Invalid("a build stream holds an empty key".into()))?;
            if load.stream != Some(kind) {
                if kind >= TABLES {
                    self.stages.start(RebuildStage::LoadingParses);
                }
                load.stream = Some(kind);
            }
            match kind {
                INPUTS => {
                    let row = decode_row(value)?;
                    // Installing an export's assets before the load may
                    // have named an input already.
                    t.prepare_cached(
                        "INSERT INTO input_ref(id,origin,route) VALUES (?1,?2,?3)
                         ON CONFLICT(id) DO UPDATE SET origin=excluded.origin,route=excluded.route
                         WHERE origin IS NOT excluded.origin OR route IS NOT excluded.route",
                    )?
                    .execute(params![key_input(rest)?, row[0], row[1]])?;
                }
                GROUPS => {
                    let worker = u32::from_be_bytes(rest[..4].try_into().expect("four bytes"));
                    let seq = i64::from_be_bytes(rest[4..12].try_into().expect("eight bytes"));
                    load.groups
                        .bases
                        .entry(worker)
                        .or_insert(load.next_group - seq);
                    let Some((&kind, rest)) =
                        value.split_first().filter(|(_, rest)| rest.len() >= 32)
                    else {
                        return Err(StoreError::Invalid("a content group has no hash".into()));
                    };
                    let (hash, body) = rest.split_at(32);
                    let stored = t
                        .prepare_cached(
                            "INSERT INTO content_group(id,hash,body) VALUES (?1,?2,?3)
                             ON CONFLICT(hash) DO NOTHING",
                        )?
                        .execute(params![load.next_group, hash, body])?;
                    if stored == 0 {
                        let held: i64 = t
                            .prepare_cached("SELECT id FROM content_group WHERE hash=?1")?
                            .query_row([hash], |r| r.get(0))?;
                        load.groups.shared.insert(load.next_group, held);
                    } else if kind == TEXT_GROUP {
                        t.prepare_cached("INSERT INTO temp.build_text_group(id) VALUES (?1)")?
                            .execute([load.next_group])?;
                    }
                    load.next_group += 1;
                }
                TEXTS => {
                    let (hash, rest) = rest.split_at(32);
                    if load
                        .current_text
                        .as_ref()
                        .is_none_or(|(same, _)| same.as_slice() != hash)
                    {
                        load.current_text = None;
                    }
                    match rest.first() {
                        Some(&TEXT) => {
                            if load.current_text.is_none() {
                                let row = decode_row(value)?;
                                let Value::Integer(named) = row[0] else {
                                    return Err(StoreError::Invalid(
                                        "a text names no group".into(),
                                    ));
                                };
                                let group = load.groups.key(named)?;
                                t.prepare_cached(
                                    "INSERT INTO text(hash,normalized_hash,group_id,start,length)
                                     VALUES (?1,?2,?3,?4,?5)",
                                )?
                                .execute(params![hash, row[3], group, row[1], row[2]])?;
                                load.current_text = Some((hash.to_vec(), t.last_insert_rowid()));
                            }
                        }
                        Some(&TEXT_HOLDER) => {
                            let (_, id) = load.current_text.as_ref().ok_or_else(|| {
                                StoreError::Invalid("a reading names a text the build lacks".into())
                            })?;
                            let mut row = decode_row(value)?;
                            // A message's text: its parse's key in the
                            // session records, then its block.
                            if let Some(Value::Blob(mut reading)) = row.pop() {
                                let Value::Integer(idx) = row[4] else {
                                    return Err(StoreError::Invalid(
                                        "a reading's text names no block".into(),
                                    ));
                                };
                                reading.extend_from_slice(&(idx as u32).to_be_bytes());
                                load.reading_texts.push(&reading, &id.to_be_bytes())?;
                            }
                            *row.last_mut().expect("a text column") = Value::Integer(*id);
                            values.clear();
                            let refs: Vec<ValueRef<'_>> = row.iter().map(ValueRef::from).collect();
                            encode_row(&refs, &mut values);
                            let mut key = vec![0];
                            key.extend_from_slice(&rest[1..]);
                            load.holders.push(&key, &values)?;
                        }
                        _ => return Err(StoreError::Invalid("a text record of no kind".into())),
                    }
                }
                MARKS => {
                    // Every parse table is loaded by now.
                    let aliases = match &mut load.aliases {
                        Some(aliases) => aliases,
                        None => load
                            .aliases
                            .insert(crate::writer::identity::Aliases::load(&t)?),
                    };
                    let (&mark, rest) = rest
                        .split_first()
                        .ok_or_else(|| StoreError::Invalid("a build mark of no kind".into()))?;
                    match mark {
                        FALLBACK => {
                            load.fallback.insert(split_id(rest)?.0.to_owned());
                        }
                        MESSAGE_ID => {
                            let (message, rest) = split_id(rest)?;
                            let (&how, rest) = rest.split_first().ok_or_else(|| {
                                StoreError::Invalid("a build mark of no kind".into())
                            })?;
                            let session = split_id(rest)?.0;
                            // A message one of whose names is a copy's is
                            // shown as its session decides.
                            let named = crate::writer::identity::message_session(message)
                                .is_some_and(|named| aliases.involve(&named));
                            if named {
                                load.fallback.insert(session.to_owned());
                            }
                            if load.message.as_ref().is_none_or(|(id, _)| id != message) {
                                load.message = Some((message.to_owned(), Vec::new()));
                            }
                            let (_, stating) = load.message.as_mut().expect("set above");
                            match how {
                                // A message stated under several sessions.
                                STATES => {
                                    if stating.iter().any(|other| other != session) {
                                        load.fallback.extend(stating.iter().cloned());
                                        load.fallback.insert(session.to_owned());
                                    }
                                    stating.push(session.to_owned());
                                }
                                // An event following a message: the
                                // composer leaves one that follows another
                                // session's message for the index.
                                FOLLOWS => {}
                                _ => {
                                    return Err(StoreError::Invalid(
                                        "a build mark of no kind".into(),
                                    ));
                                }
                            }
                        }
                        EVENT_ID => {
                            // An event stated under several sessions is
                            // assembled from all its parses once the
                            // session records are.
                            let (event, rest) = split_id(rest)?;
                            let session = split_id(rest)?.0;
                            if load
                                .last_event
                                .as_ref()
                                .is_some_and(|(last, other)| last == event && other != session)
                            {
                                load.shared_events.insert(event.to_owned());
                            }
                            load.last_event = Some((event.to_owned(), session.to_owned()));
                        }
                        _ => return Err(StoreError::Invalid("a build mark of no kind".into())),
                    }
                }
                _ => {
                    let table = load.tables.get(usize::from(kind - TABLES)).ok_or_else(|| {
                        StoreError::Invalid("a build stream names no table".into())
                    })?;
                    let mut row = decode_row(value)?;
                    if let Some(at) = table.input {
                        row[at] = Value::Integer(input_pk(&t, &mut load.input, key_input(rest)?)?);
                    }
                    if let Some(at) = table.group {
                        let Value::Integer(named) = row[at] else {
                            return Err(StoreError::Invalid("a reading row names no group".into()));
                        };
                        row[at] = Value::Integer(load.groups.key(named)?);
                    }
                    insert(&t, table, &row)?;
                }
            }
            written += 1;
        };
        if !more {
            // Text groups whose every text another worker also held.
            t.execute(
                "DELETE FROM content_group WHERE id IN (SELECT id FROM temp.build_text_group)
                 AND id NOT IN (SELECT group_id FROM text)",
                [],
            )?;
            t.execute("DELETE FROM temp.build_text_group", [])?;
        }
        t.commit()?;
        self.count(written);
        Ok(more)
    }

    /// Loads up to a transaction's worth of parse text rows; false once
    /// they are all loaded.
    fn holders_transaction(
        &mut self,
        holders: &mut Merger,
        table: &Table,
        input: &mut Option<(String, i64)>,
    ) -> Result<bool> {
        self.archive.stop().check()?;
        let t = Transaction::new_unchecked(&self.conn, TransactionBehavior::Immediate)?;
        let mut written = 0u64;
        let more = loop {
            if written == ROWS_PER_TRANSACTION {
                break true;
            }
            let Some((key, value)) = holders.next()? else {
                break false;
            };
            let mut row = decode_row(value)?;
            row[0] = Value::Integer(input_pk(&t, input, key_input(&key[1..])?)?);
            insert(&t, table, &row)?;
            written += 1;
        };
        t.commit()?;
        self.count(written);
        Ok(more)
    }
}

/// The build file's key of `id`, looked up once per input as the merge
/// passes it in key order.
fn input_pk(tx: &Transaction<'_>, current: &mut Option<(String, i64)>, id: &str) -> Result<i64> {
    if let Some((held, pk)) = current
        && held == id
    {
        return Ok(*pk);
    }
    let pk: i64 = tx
        .prepare_cached("SELECT pk FROM input_ref WHERE id=?1")?
        .query_row([id], |r| r.get(0))?;
    *current = Some((id.to_owned(), pk));
    Ok(pk)
}

fn insert(tx: &Transaction<'_>, table: &Table, row: &[Value]) -> Result<()> {
    tx.prepare_cached(&table.insert)?
        .execute(rusqlite::params_from_iter(row))?;
    Ok(())
}
