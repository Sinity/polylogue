//! Assembly of a rebuild's sessions (docs/architecture.md, one path): links
//! and documents first, from the build file, so each session is assembled
//! after its links; then every session the session records hold, by a pool
//! of threads ([`pool`]) from the records read in key order; then, from the
//! build file, the sessions whose parses reach other sessions' parses, the
//! events about inputs and the events placed after another session's
//! message. Assembly uses the composition an incremental update uses
//! ([`crate::writer::shown`]).
use super::Rebuild;
use super::load::Loaded;
use super::worker::{
    COMPOSE, EVENT, FACTS, GroupKeys, MESSAGE, OUTCOME, SESSIONS_PER_TRANSACTION, decode_row,
    split_id,
};
use crate::archive::bump_counter;
use crate::sorted_runs::Merger;
use crate::writer::readings::Touched;
use crate::{ConnExt as _, Result, StoreError};
use rusqlite::TransactionBehavior;
use rusqlite::types::Value;
use std::collections::{BTreeSet, HashSet};
use std::sync::Arc;

mod pool;

impl Rebuild {
    /// Assembles every session of the loaded build file on `assemblers`
    /// threads, then makes the assembled tables' indexes that assembly does
    /// not fill as it writes (`after_stream`).
    pub(super) fn assemble(
        &mut self,
        loaded: Loaded,
        after_stream: Vec<String>,
        assemblers: usize,
    ) -> Result<()> {
        let Loaded {
            merged,
            texts,
            mut fallback,
            shared_events,
            groups,
        } = loaded;
        let budget = self.archive.memory_budget();
        let composing = budget.composition_memory();
        // Beside the run readers and the assembly, the full-text fill follows
        // the blocks assembly writes.
        self.size_cache(
            (merged.memory() + texts.memory() + composing + budget.search_fill_memory()) as u64,
            false,
        )?;
        let aliases = Arc::new(crate::writer::identity::Aliases::load(&self.conn)?);
        // The sessions a copy name involves are decided across sessions.
        fallback.extend(
            self.conn
                .prepare(
                    "SELECT alias FROM session_alias WHERE alias<>session_id
                     UNION SELECT session_id FROM session_alias WHERE alias<>session_id",
                )?
                .query_map([], |r| r.get::<_, String>(0))?
                .collect::<rusqlite::Result<Vec<_>>>()?,
        );
        // The session records hold a session for every ID a parse states
        // (its record, a message, an event or a call's outcome), less those
        // assembled from the build file, which their reader passes over.
        let stated: u64 = self.conn.scalar(
            "SELECT count(*) FROM (
                 SELECT DISTINCT session_id FROM session_parse
                 UNION SELECT DISTINCT session_id FROM message_parse
                 UNION SELECT DISTINCT session_id FROM event_parse WHERE session_id IS NOT NULL
                 UNION SELECT DISTINCT session_id FROM call_outcome)",
            [],
        )?;
        let mut passed_over = 0;
        for name in &fallback {
            if self.conn.exists(
                "SELECT EXISTS(SELECT 1 FROM session_parse WHERE session_id=?1)
                     OR EXISTS(SELECT 1 FROM message_parse WHERE session_id=?1)
                     OR EXISTS(SELECT 1 FROM event_parse WHERE session_id=?1)
                     OR EXISTS(SELECT 1 FROM call_outcome WHERE session_id=?1)",
                [name],
            )? {
                passed_over += 1;
            }
        }
        let fallback: Vec<String> = fallback.into_iter().collect();
        let in_stream = stated.saturating_sub(passed_over);
        crate::log!(
            info,
            "index: assembling {in_stream} sessions, then {} across files",
            fallback.len(),
        );
        self.total(in_stream + fallback.len() as u64);
        // Links and documents first, from the build file.
        let mut touched = Touched::default();
        touched.links.extend(
            self.conn
                .prepare("SELECT DISTINCT statement FROM link_parse")?
                .query_map([], |r| r.get::<_, Vec<u8>>(0))?
                .collect::<rusqlite::Result<Vec<_>>>()?,
        );
        touched.documents.extend(
            self.conn
                .prepare("SELECT DISTINCT document_id FROM document_parse")?
                .query_map([], |r| r.get::<_, String>(0))?
                .collect::<rusqlite::Result<Vec<_>>>()?,
        );
        self.compose_from_index(touched)?;
        let deferred = self.assemble_stream(
            merged,
            texts,
            &fallback,
            Arc::clone(&aliases),
            Arc::new(shared_events),
            groups,
            composing / 3,
            assemblers,
        )?;
        // The stream's reader has ended with its runs.
        self.size_cache(0, true)?;
        for sql in after_stream {
            self.archive.stop().check()?;
            self.conn.execute(&sql, [])?;
        }
        self.size_cache(0, false)?;
        // From the build file: the sessions whose parses reach other
        // sessions' parses, a transaction's share at a time; with the first
        // share, the events no session holds and those that follow another
        // session's message.
        let mut deferred = Some(deferred);
        for (n, chunk) in fallback.chunks(SESSIONS_PER_TRANSACTION).enumerate() {
            let mut touched = Touched::default();
            for session in chunk {
                touched.whole.insert(session.clone());
                touched.sessions.insert(session.clone());
                for row in self
                    .conn
                    .prepare_cached(
                        "SELECT session_id,call_id FROM call_outcome WHERE session_id=?1",
                    )?
                    .query_map([session], |r| <(String, String)>::try_from(r))?
                {
                    touched.calls.insert(row?);
                }
            }
            if n == 0 {
                self.events_from_index(&mut touched, deferred.take().unwrap_or_default())?;
            }
            self.compose_from_index(touched)?;
            self.count(chunk.len() as u64);
        }
        if let Some(deferred) = deferred {
            let mut touched = Touched::default();
            self.events_from_index(&mut touched, deferred)?;
            self.compose_from_index(touched)?;
        }
        // Every session's lineage is computed once the rebuild's sessions
        // and links are all written.
        let tx = self
            .conn
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        tx.execute(
            "INSERT INTO summary_due(session_pk,from_order,lineage)
             SELECT pk,X'FF',1 FROM session WHERE true
             ON CONFLICT(session_pk) DO UPDATE SET lineage=1,marked=marked+1",
            [],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// What no session holds, input events, and the events `deferred` names.
    fn events_from_index(
        &self,
        touched: &mut Touched,
        deferred: Vec<(Option<String>, String)>,
    ) -> Result<()> {
        for row in self
            .conn
            .prepare("SELECT event_id FROM event_parse WHERE session_id IS NULL")?
            .query_map([], |r| r.get::<_, String>(0))?
        {
            touched.events.insert((None, row?));
        }
        touched.events.extend(deferred);
        Ok(())
    }

    /// Assembles every session of the session records on the pool and
    /// appends them, a transaction per [`SESSIONS_PER_TRANSACTION`] batches'
    /// worth. Returns the events the pool left for the build file.
    #[allow(clippy::too_many_arguments)]
    fn assemble_stream(
        &mut self,
        merged: Merger,
        texts: Merger,
        fallback: &[String],
        aliases: Arc<crate::writer::identity::Aliases>,
        shared_events: Arc<HashSet<String>>,
        groups: GroupKeys,
        ahead: usize,
        assemblers: usize,
    ) -> Result<Vec<(Option<String>, String)>> {
        crate::fault::check("index.build.composing", "")?;
        // Every batch is assembled under one write counter.
        let tx = self
            .conn
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let counter = bump_counter(&tx)?;
        tx.commit()?;
        let shared = pool::Shared::read(
            &self.conn,
            Arc::clone(&aliases),
            Arc::clone(&shared_events),
            counter,
        )?;
        let skipped: BTreeSet<String> = fallback.iter().cloned().collect();
        let (batches, gate) = read_stream(
            merged,
            texts,
            skipped,
            aliases,
            Arc::new(groups),
            ahead,
            assemblers,
        );
        // A dropped pool lets its reader go.
        struct Closing(Arc<Gate>);
        impl Drop for Closing {
            fn drop(&mut self) {
                self.0.close();
            }
        }
        let closing = Closing(Arc::clone(&gate));
        let mut pool = pool::Pool::start(assemblers, shared, batches, &self.conn)?;
        // A session from the stream refers only to rows the rebuild wrote.
        self.conn.pragma_update(None, "foreign_keys", false)?;
        let mut deferred = Vec::new();
        let result = (|| -> Result<()> {
            loop {
                self.archive.stop().check()?;
                let tx = self
                    .conn
                    .transaction_with_behavior(TransactionBehavior::Immediate)?;
                bump_counter(&tx)?;
                let mut sessions = 0;
                let mut done = false;
                while sessions < SESSIONS_PER_TRANSACTION as u64 * 8 {
                    let Some(composed) = pool.next()? else {
                        done = true;
                        break;
                    };
                    pool.append(&tx, composed.rows)?;
                    gate.release(composed.bytes);
                    deferred.extend(composed.late);
                    sessions += composed.sessions;
                }
                tx.commit()?;
                self.count(sessions);
                if done {
                    return Ok(());
                }
            }
        })();
        self.conn.pragma_update(None, "foreign_keys", true)?;
        drop(closing);
        result?;
        Ok(deferred)
    }

    /// Assembles what `touched` names from the parses in the build file, as
    /// an incremental update does, in one transaction.
    fn compose_from_index(&mut self, touched: Touched) -> Result<()> {
        self.archive.stop().check()?;
        let tx = self
            .conn
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let counter = bump_counter(&tx)?;
        let changes = crate::change_summary::ChangeSummary::start(&tx)?;
        crate::writer::rows::finish(&tx, touched, &changes, counter)?;
        tx.commit()?;
        Ok(())
    }
}

/// What the stream reader hands assembly, in stream order: each session
/// begun, its facts' kinds, each message's parses with their values, its
/// event parses and outcomes, then its end. Input keys, kinds and whether
/// a newer chain supersedes a parse are filled in by assembly, which reads
/// the build file.
pub(super) enum Piece {
    Begin,
    Facts {
        input: String,
        chain: i64,
        kind: Option<polylogue_model::SessionKind>,
    },
    Message {
        id: String,
        readings: Vec<crate::writer::shown::MessageReading>,
    },
    Event(crate::writer::shown::EventRow),
    Outcome(crate::writer::shown::OutcomeRow),
    End(String),
}

/// What the stream's reader has decoded and the rebuild has not yet
/// appended, in bytes, at most a given number: the reader waits while that
/// many are held, unless nothing is (a batch larger than the allowance goes
/// alone), or until assembly is dropped.
pub(super) struct Gate {
    held: std::sync::Mutex<(usize, bool)>,
    room: std::sync::Condvar,
}

impl Gate {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            held: std::sync::Mutex::new((0, false)),
            room: std::sync::Condvar::new(),
        })
    }

    /// Waits until `bytes` more fit in `ahead`; false once closed.
    fn take(&self, bytes: usize, ahead: usize) -> bool {
        let Ok(mut held) = self.held.lock() else {
            return false;
        };
        while !held.1 && held.0 > 0 && held.0 + bytes > ahead {
            match self.room.wait(held) {
                Ok(guard) => held = guard,
                Err(_) => return false,
            }
        }
        held.0 += bytes;
        !held.1
    }

    fn release(&self, bytes: usize) {
        if let Ok(mut held) = self.held.lock() {
            held.0 = held.0.saturating_sub(bytes);
        }
        self.room.notify_all();
    }

    fn close(&self) {
        if let Ok(mut held) = self.held.lock() {
            held.1 = true;
        }
        self.room.notify_all();
    }
}

/// About what a decoded piece holds, from the bytes it was decoded from: a
/// message's decoded value beside its packed blocks.
fn piece_bytes(record: usize) -> usize {
    2 * record + 256
}

/// Reads the session records on a thread of their own, merging them with
/// each message parse's text keys and decoding them, and hands whole
/// sessions to the pool in batches of about [`pool::BATCH_MESSAGES`]
/// messages or a share of `ahead` bytes. Sessions assembled from the build
/// file are passed over.
fn read_stream(
    mut merged: Merger,
    mut texts: Merger,
    skipped: BTreeSet<String>,
    aliases: Arc<crate::writer::identity::Aliases>,
    groups: Arc<GroupKeys>,
    ahead: usize,
    assemblers: usize,
) -> (std::sync::mpsc::Receiver<Result<pool::Batch>>, Arc<Gate>) {
    let (sender, receiver) = std::sync::mpsc::channel();
    let gate = Gate::new();
    let held = Arc::clone(&gate);
    // Enough batches in flight for every assembler and the appender.
    let batch_bytes = (ahead / (2 * assemblers.max(1) + 1)).max(1 << 20);
    let spawned = std::thread::Builder::new()
        .name("polylogue-assemble-stream".into())
        .spawn(move || {
            let mut batch = pool::Batch::default();
            let mut seq = 0;
            let mut ship = |batch: &mut pool::Batch| -> bool {
                let mut full = std::mem::take(batch);
                full.seq = seq;
                seq += 1;
                held.take(full.bytes, ahead) && sender.send(Ok(full)).is_ok()
            };
            let result = (|| -> Result<()> {
                while let Some((key, _)) = merged.peek()? {
                    if key.first() != Some(&COMPOSE) {
                        return Err(StoreError::Invalid(
                            "a build stream holds records after its composition".into(),
                        ));
                    }
                    let session = split_id(&key[1..])?.0.to_owned();
                    if skipped.contains(&session) || aliases.involve(&session) {
                        while merged
                            .peek()?
                            .is_some_and(|(key, _)| of_session(key, &session))
                        {
                            merged.next()?;
                        }
                        continue;
                    }
                    batch.pieces.push(Piece::Begin);
                    let mut add = |piece: Piece, bytes: usize| {
                        if matches!(piece, Piece::Message { .. }) {
                            batch.messages += 1;
                        }
                        batch.bytes += bytes;
                        batch.pieces.push(piece);
                        true
                    };
                    read_session(&mut merged, &mut texts, &groups, &session, &mut add)?;
                    batch.pieces.push(Piece::End(session));
                    if (batch.messages >= pool::BATCH_MESSAGES || batch.bytes >= batch_bytes)
                        && !ship(&mut batch)
                    {
                        return Ok(());
                    }
                }
                if !batch.pieces.is_empty() {
                    ship(&mut batch);
                }
                Ok(())
            })();
            if let Err(error) = result {
                let _ = sender.send(Err(error));
            }
        });
    if let Err(error) = spawned {
        let (sender, failed) = std::sync::mpsc::channel();
        let _ = sender.send(Err(error.into()));
        return (failed, gate);
    }
    (receiver, gate)
}

/// Whether `key` is one of `session`'s records.
fn of_session(key: &[u8], session: &str) -> bool {
    key.len() > session.len() + 1
        && key[0] == COMPOSE
        && &key[1..=session.len()] == session.as_bytes()
        && key[session.len() + 1] == 0
}

/// Decodes one session's records into pieces; false once composition
/// stopped taking them.
fn read_session(
    merged: &mut Merger,
    texts: &mut Merger,
    groups: &GroupKeys,
    session: &str,
    send: &mut impl FnMut(Piece, usize) -> bool,
) -> Result<bool> {
    use crate::writer::shown::{EventRow, Given, MessageReading, OutcomeRow};
    let bad = || StoreError::Invalid("a composition record is malformed".into());
    let mut pending: Option<(String, Vec<MessageReading>)> = None;
    // What the pending message's readings were decoded from.
    let mut pending_bytes = 0;
    let prefix = session.len() + 2;
    while let Some((key, value)) = merged.peek()? {
        if !of_session(key, session) {
            break;
        }
        let (key, value) = (key.to_vec(), value.to_vec());
        merged.next()?;
        let kind = key[prefix];
        let rest = &key[prefix + 1..];
        let row = decode_row(&value)?;
        // (item, input, chain) of a reading; facts name no item.
        let (item, input, chain) = if kind == FACTS {
            let (input, chain) = split_id(rest)?;
            (String::new(), input.to_owned(), chain)
        } else {
            let (item, rest) = split_id(rest)?;
            let (input, chain) = split_id(rest)?;
            (item.to_owned(), input.to_owned(), chain)
        };
        let chain = i64::from_be_bytes(chain.try_into().map_err(|_| bad())?);
        if kind != MESSAGE
            && let Some((id, readings)) = pending.take()
            && !send(
                Piece::Message { id, readings },
                piece_bytes(std::mem::take(&mut pending_bytes)),
            )
        {
            return Ok(false);
        }
        let int = |n: usize| match row.get(n) {
            Some(Value::Integer(v)) => Ok(*v),
            _ => Err(bad()),
        };
        let int_or_null = |n: usize| match row.get(n) {
            Some(Value::Integer(v)) => Ok(Some(*v)),
            Some(Value::Null) => Ok(None),
            _ => Err(bad()),
        };
        let text = |n: usize| match row.get(n) {
            Some(Value::Text(v)) => Ok(v.as_str()),
            _ => Err(bad()),
        };
        let blob = |n: usize| match row.get(n) {
            Some(Value::Blob(v)) => Ok(v.as_slice()),
            _ => Err(bad()),
        };
        let route = |n: usize| {
            polylogue_model::Route::from_name(text(n)?)
                .ok_or_else(|| StoreError::stored(0, "unknown stored route"))
        };
        match kind {
            FACTS => {
                let kind = match row.first() {
                    Some(Value::Text(kind)) => Some(
                        polylogue_model::SessionKind::from_name(kind)
                            .ok_or_else(|| StoreError::stored(0, "unknown stored session kind"))?,
                    ),
                    _ => None,
                };
                if !send(Piece::Facts { input, chain, kind }, 0) {
                    return Ok(false);
                }
            }
            MESSAGE => {
                if pending.as_ref().is_some_and(|(id, _)| *id != item)
                    && let Some((id, readings)) = pending.take()
                    && !send(
                        Piece::Message { id, readings },
                        piece_bytes(std::mem::take(&mut pending_bytes)),
                    )
                {
                    return Ok(false);
                }
                pending_bytes += value.len();
                // Columns: start, end, seq, position, order time, route,
                // completion, content-named, hash, the content's group, start
                // and length, words; then each block's text hash.
                let hash = blob(8)?.to_vec();
                if row.len() < 13 {
                    return Err(bad());
                }
                let texts_of = row[13..]
                    .iter()
                    .map(|text| match text {
                        Value::Blob(hash) => Ok(Some(
                            <[u8; 32]>::try_from(hash.as_slice()).map_err(|_| bad())?,
                        )),
                        Value::Null => Ok(None),
                        _ => Err(bad()),
                    })
                    .collect::<Result<Vec<_>>>()?;
                let blocks = texts_of.len();
                // The text keys the load gave this reading's text rows.
                // Both streams are in key order; the text rows of readings
                // composed from the index are passed over.
                let reading = key.as_slice();
                let mut ids = vec![None; blocks];
                while let Some((text_key, id)) = texts.peek()? {
                    if text_key.len() == reading.len() + 4 && text_key.starts_with(reading) {
                        let idx = u32::from_be_bytes(
                            text_key[reading.len()..].try_into().expect("four bytes"),
                        );
                        let id = i64::from_be_bytes(id.try_into().map_err(|_| bad())?);
                        *ids.get_mut(idx as usize).ok_or_else(bad)? = Some(id);
                    } else if text_key >= reading {
                        break;
                    }
                    texts.next()?;
                }
                polylogue_model::MessageId::parse(&item).map_err(StoreError::invalid)?;
                let reading = MessageReading {
                    input_pk: 0,
                    input,
                    chain,
                    message: item.clone(),
                    session: session.to_owned(),
                    start: int(0)?,
                    end: int(1)?,
                    seq: int(2)?,
                    position: int(3)?,
                    order_time: int_or_null(4)?,
                    route: route(5)?,
                    completion: polylogue_model::Completion::from_name(text(6)?)
                        .ok_or_else(|| StoreError::stored(0, "unknown stored completion"))?,
                    content: int(7)? != 0,
                    hash,
                    stored: crate::content::Stretch {
                        group: groups.key(int(9)?)?,
                        start: int(10)?,
                        length: int(11)?,
                    },
                    given: Some(Given {
                        kind: None,
                        superseded: false,
                        words: int(12)?,
                        texts: texts_of,
                        text_ids: ids,
                    }),
                };
                pending
                    .get_or_insert_with(|| (item, Vec::new()))
                    .1
                    .push(reading);
            }
            EVENT => {
                let bytes = piece_bytes(value.len());
                if !send(
                    Piece::Event(EventRow {
                        input,
                        input_pk: 0,
                        chain,
                        event_id: item,
                        route: route(0)?,
                        hash: blob(1)?.to_vec(),
                        piece: crate::content::Piece {
                            group: groups.key(int(2)?)?,
                            start: int(3)?,
                            length: int(4)?,
                            data_length: Some(int(5)?),
                        },
                        seq: int(6)?,
                        start: int_or_null(7)?,
                        end: int_or_null(8)?,
                    }),
                    bytes,
                ) {
                    return Ok(false);
                }
            }
            OUTCOME => {
                let bytes = piece_bytes(value.len());
                if !send(
                    Piece::Outcome(OutcomeRow {
                        input_pk: 0,
                        chain,
                        position: int(0)?,
                        input,
                        route: route(2)?,
                        call_id: item,
                        body: text(1)?.to_owned(),
                    }),
                    bytes,
                ) {
                    return Ok(false);
                }
            }
            _ => return Err(bad()),
        }
    }
    if let Some((id, readings)) = pending.take() {
        return Ok(send(
            Piece::Message { id, readings },
            piece_bytes(pending_bytes),
        ));
    }
    Ok(true)
}
