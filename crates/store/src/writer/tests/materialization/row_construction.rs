//! A test-only lower bound for constructing a closed cohort away from the
//! serial writer. It does not supply the current rows of a prior cohort.
use super::*;
use crate::archive::open_write;
use rusqlite::{config::DbConfig, types::ValueRef};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::time::Instant;

#[derive(Debug, Default, serde::Serialize)]
struct Phase {
    seconds: f64,
    cpu_seconds: f64,
    process_cpu_seconds: f64,
    io: BTreeMap<String, u64>,
    workers_cpu_seconds: BTreeMap<String, f64>,
}

impl Phase {
    fn add(&mut self, other: Self) {
        self.seconds += other.seconds;
        self.cpu_seconds += other.cpu_seconds;
        self.process_cpu_seconds += other.process_cpu_seconds;
        for (key, value) in other.io {
            *self.io.entry(key).or_default() += value;
        }
        for (name, value) in other.workers_cpu_seconds {
            *self.workers_cpu_seconds.entry(name).or_default() += value;
        }
    }
}

fn scheduled() -> u64 {
    std::fs::read_to_string("/proc/thread-self/schedstat")
        .unwrap()
        .split_whitespace()
        .next()
        .unwrap()
        .parse()
        .unwrap()
}

#[allow(unsafe_code)]
fn process_cpu() -> f64 {
    let mut clock = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: clock is a live output for a supported process CPU clock.
    let code = unsafe { libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, &mut clock) };
    assert_eq!(code, 0);
    clock.tv_sec as f64 + clock.tv_nsec as f64 / 1e9
}

fn process_io() -> BTreeMap<String, u64> {
    std::fs::read_to_string("/proc/self/io")
        .unwrap()
        .lines()
        .map(|line| {
            let (key, value) = line.split_once(':').unwrap();
            (key.to_owned(), value.trim().parse().unwrap())
        })
        .collect()
}

fn measured<T>(work: impl FnOnce() -> T) -> (T, Phase) {
    let workers = crate::timing::workers().len();
    let io = process_io();
    let cpu = scheduled();
    let process = process_cpu();
    let start = Instant::now();
    let value = work();
    let phase = Phase {
        seconds: start.elapsed().as_secs_f64(),
        cpu_seconds: scheduled().saturating_sub(cpu) as f64 / 1e9,
        process_cpu_seconds: process_cpu() - process,
        workers_cpu_seconds: {
            let mut totals = BTreeMap::new();
            for worker in crate::timing::workers().into_iter().skip(workers) {
                *totals.entry(worker.name).or_default() += worker.cpu_seconds.unwrap();
            }
            totals
        },
        io: process_io()
            .into_iter()
            .map(|(key, value)| {
                let before = io.get(&key).copied().unwrap_or(0);
                (key, value.saturating_sub(before))
            })
            .collect(),
    };
    (value, phase)
}

fn quoted(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

#[derive(Debug)]
struct Table {
    name: String,
    columns: Vec<String>,
    order: Vec<String>,
    parents: BTreeSet<String>,
}

fn tables(conn: &Connection, schema: &str) -> Vec<Table> {
    let names: Vec<String> = conn
        .prepare(&format!("PRAGMA {schema}.table_list"))
        .unwrap()
        .query_map([], |row| {
            Ok((row.get::<_, String>(1)?, row.get::<_, String>(2)?))
        })
        .unwrap()
        .map(|row| row.unwrap())
        .filter(|(name, kind)| kind == "table" && !name.starts_with("sqlite_"))
        .map(|(name, _)| name)
        .collect();
    let mut pending = BTreeMap::new();
    for name in names {
        if !conn
            .scalar::<bool>(
                &format!("SELECT EXISTS(SELECT 1 FROM {schema}.{})", quoted(&name)),
                [],
            )
            .unwrap()
        {
            continue;
        }
        let info: Vec<(String, i64)> = conn
            .prepare(&format!("PRAGMA {schema}.table_info({})", quoted(&name)))
            .unwrap()
            .query_map([], |row| Ok((row.get(1)?, row.get(5)?)))
            .unwrap()
            .map(|row| row.unwrap())
            .collect();
        let columns: Vec<_> = info.iter().map(|(name, _)| name.clone()).collect();
        let mut key: Vec<_> = info.into_iter().filter(|(_, n)| *n > 0).collect();
        key.sort_by_key(|(_, n)| *n);
        let order = if key.is_empty() {
            columns.clone()
        } else {
            key.into_iter().map(|(name, _)| name).collect()
        };
        let parents = conn
            .prepare(&format!(
                "PRAGMA {schema}.foreign_key_list({})",
                quoted(&name)
            ))
            .unwrap()
            .query_map([], |row| row.get::<_, String>(2))
            .unwrap()
            .map(|row| row.unwrap())
            .filter(|parent| parent != &name)
            .collect();
        pending.insert(
            name.clone(),
            Table {
                name,
                columns,
                order,
                parents,
            },
        );
    }
    let mut ordered = Vec::new();
    while !pending.is_empty() {
        let name = pending
            .iter()
            .find(|(_, table)| table.parents.iter().all(|p| !pending.contains_key(p)))
            .map(|(name, _)| name.clone())
            .unwrap_or_else(|| panic!("cohort has a foreign-key cycle: {pending:?}"));
        ordered.push(pending.remove(&name).unwrap());
    }
    ordered
}

fn select(schema: &str, table: &Table) -> String {
    let columns = table
        .columns
        .iter()
        .map(|name| quoted(name))
        .collect::<Vec<_>>()
        .join(",");
    let order = table
        .order
        .iter()
        .map(|name| quoted(name))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "SELECT {columns} FROM {schema}.{} ORDER BY {order}",
        quoted(&table.name)
    )
}

fn snapshot(conn: &Connection, schema: &str) -> BTreeMap<String, (u64, String)> {
    tables(conn, schema)
        .into_iter()
        .map(|table| {
            let mut hash = blake3::Hasher::new();
            let mut statement = conn.prepare(&select(schema, &table)).unwrap();
            let columns = statement.column_count();
            let mut rows = statement.query([]).unwrap();
            let mut count = 0u64;
            while let Some(row) = rows.next().unwrap() {
                for column in 0..columns {
                    match row.get_ref(column).unwrap() {
                        ValueRef::Null => hash.update(b"n"),
                        ValueRef::Integer(value) => {
                            hash.update(&[b"i".as_slice(), &value.to_be_bytes()].concat())
                        }
                        ValueRef::Real(value) => {
                            hash.update(&[b"r".as_slice(), &value.to_bits().to_be_bytes()].concat())
                        }
                        ValueRef::Text(value) | ValueRef::Blob(value) => {
                            let tag = if matches!(row.get_ref(column).unwrap(), ValueRef::Text(_)) {
                                b't'
                            } else {
                                b'b'
                            };
                            hash.update(&[tag]);
                            hash.update(&(value.len() as u64).to_be_bytes());
                            hash.update(value)
                        }
                    };
                }
                hash.update(b"\n");
                count += 1;
            }
            (table.name, (count, hash.finalize().to_hex().to_string()))
        })
        .collect()
}

/// The producer already ran the table triggers. Copy their complete effects,
/// including counters, without executing them a second time. Foreign keys and
/// every first-build-maintained index remain enabled on the destination.
fn transfer(conn: &mut Connection, uri: &str, fail_after_copy: bool) -> Result<()> {
    conn.execute("ATTACH DATABASE ?1 AS cohort", [uri])?;
    let source = tables(conn, "cohort");
    let previous = tables(conn, "main");
    let enabled = conn.db_config(DbConfig::SQLITE_DBCONFIG_ENABLE_TRIGGER)?;
    conn.set_db_config(DbConfig::SQLITE_DBCONFIG_ENABLE_TRIGGER, false)?;
    let result = (|| {
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        // This lower bound replaces an entire previously empty model.
        // Remove stale seeded counters as well as inserting the final rows.
        for table in previous.iter().rev() {
            tx.execute(&format!("DELETE FROM main.{}", quoted(&table.name)), [])?;
        }
        for table in &source {
            let columns = table
                .columns
                .iter()
                .map(|name| quoted(name))
                .collect::<Vec<_>>()
                .join(",");
            tx.execute(
                &format!(
                    "INSERT INTO main.{} ({columns}) {}",
                    quoted(&table.name),
                    select("cohort", table)
                ),
                [],
            )?;
        }
        if fail_after_copy {
            // The earlier copies include parse_state. A real FK error must
            // roll back the whole graph and its completed parser positions.
            tx.execute("INSERT INTO message_source(message_pk,input_pk,chain,source_start,source_end,id_kind,session_pk,route,completion,version_pk) VALUES(-1,-1,1,0,1,'native',-1,'live','finished',-1)", [])?;
        }
        tx.commit()?;
        Ok(())
    })();
    conn.set_db_config(DbConfig::SQLITE_DBCONFIG_ENABLE_TRIGGER, enabled)?;
    conn.flush_prepared_statement_cache();
    conn.execute_batch("DETACH DATABASE cohort")?;
    result
}

#[derive(Debug, serde::Serialize)]
struct Construction {
    preparation: Phase,
    rows: Phase,
    bytes: u64,
    cache_bytes: u64,
    messages: u64,
}

fn construct(
    conn: &mut Connection,
    batches: &[Batch],
    budget: crate::archive::MemoryBudget,
) -> Construction {
    crate::chain_progress::open(conn, true).unwrap();
    crate::versions::track_unreferenced(conn).unwrap();
    let upkeep = resolve::IndexUpkeep::AtBuildEnd(Arc::new(
        crate::key_maps::KeyMaps::load_with_budget(conn, budget.keys()).unwrap(),
    ));
    let texts = crate::text_rows::TextIds::new(conn, true, budget.keys() / 3).unwrap();
    let mut preparation = Phase::default();
    let mut phases = Phase::default();
    let mut messages = 0;
    for batch in batches {
        let mut batch = batch.clone();
        let Change::Reparse { chain } = batch.change else {
            panic!("cohort source must already be committed")
        };
        let (_, phase) = measured(|| {
            // Overlapping observations resolve against the preceding commit.
            resolve::prepare_index(conn, &mut batch, chain, &upkeep).unwrap();
        });
        preparation.add(phase);
        let (_, phase) = measured(|| {
            let text_transaction = texts.transaction();
            let tx = conn
                .transaction_with_behavior(TransactionBehavior::Immediate)
                .unwrap();
            let counter = crate::archive::bump_counter(&tx).unwrap();
            let changes = crate::change_summary::ChangeSummary::start(&tx).unwrap();
            for step in &batch.parse {
                let commit = crate::source::Source::reading_commit(
                    &tx,
                    &batch.input.id,
                    chain,
                    step.parsed_upto,
                )
                .unwrap();
                messages += rows::write_rows(
                    &tx,
                    &batch,
                    step,
                    rows::ReadingWrite {
                        texts: &texts,
                        resolution: step.resolution.as_ref().unwrap(),
                        counter,
                        upkeep: &upkeep,
                        source_commit: Some(&commit),
                        changes: &changes,
                        retraction: false,
                        replacement: false,
                    },
                )
                .unwrap();
            }
            if batch.parse.iter().any(|step| step.progress.reached_end()) {
                rows::prune_sessions(&tx, &batch.input.id, chain, counter).unwrap();
            }
            rows::finish(&tx, &changes, &upkeep, &texts, counter).unwrap();
            tx.commit().unwrap();
            text_transaction.commit();
        });
        phases.add(phase);
    }
    crate::chain_progress::mark_current(conn).unwrap();
    Construction {
        preparation,
        rows: phases,
        bytes: (conn.scalar::<u64>("PRAGMA page_count", []).unwrap())
            * conn.scalar::<u64>("PRAGMA page_size", []).unwrap(),
        cache_bytes: cache_bytes(conn),
        messages,
    }
}

#[allow(unsafe_code)]
fn cache_bytes(conn: &Connection) -> u64 {
    let (mut current, mut peak) = (0, 0);
    // SAFETY: the connection remains live and SQLite receives two live i32 outputs.
    let code = unsafe {
        rusqlite::ffi::sqlite3_db_status(
            conn.handle(),
            rusqlite::ffi::SQLITE_DBSTATUS_CACHE_USED,
            &mut current,
            &mut peak,
            0,
        )
    };
    assert_eq!(code, rusqlite::ffi::SQLITE_OK);
    u64::try_from(current).unwrap()
}

#[derive(Clone)]
struct Observation {
    batch: Batch,
    bytes: Vec<u8>,
}

fn observation(input: &str, route: Route, start: u64, messages: Vec<Message>) -> Observation {
    let mut bytes = Vec::new();
    let mut rows = Vec::new();
    let mut entries = Vec::new();
    for message in messages {
        let offset = start + bytes.len() as u64;
        bytes.extend(serde_json::to_vec(&message).unwrap());
        bytes.push(b'\n');
        let end = start + bytes.len() as u64;
        entries.push(Entry {
            key: Key::Response {
                namespace: String::new(),
                session: message.session.to_string(),
                key: message.id.to_string(),
            },
            value: Reference::Response {
                message: message.id.clone(),
                version: message.content_hash().unwrap(),
                offset,
                end,
            },
        });
        let mut value = row(offset, message);
        value.end = end;
        rows.push(value);
    }
    let session = rows[0].message.session.native().as_str().to_owned();
    let mut batch = batch(input, route, start, rows, &[&session]);
    batch.length = start + bytes.len() as u64;
    if route.is_imported() {
        assert_eq!(start, 0);
        batch.input.id = polylogue_model::BlobHash::of(&bytes).to_string();
    }
    let step = &mut batch.parse[0];
    step.parsed_upto = batch.length;
    step.checkpoint = Some(polylogue_model::batch::ParserCheckpoint::new(
        "synthetic=1".into(),
        batch.length,
        Some(serde_json::json!({"_polylogue_context": Delta {
            version: 1,
            base: start,
            through: batch.length,
            entries,
            cleared_states: Vec::new(),
        }})),
    ));
    Observation { batch, bytes }
}

fn prose(number: usize, words: usize) -> String {
    const WORDS: [&str; 16] = [
        "cohort", "parser", "writer", "rust", "query", "source", "index", "lineage", "cargo",
        "record", "commit", "budget", "memory", "native", "fixture", "archive",
    ];
    let mut text = String::from("cohort café parser ");
    for _ in 0..number % 7 {
        text.push_str("cohort ");
    }
    let mut state = number as u64 + 1;
    for _ in 0..words + number % 13 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        text.push_str(WORDS[(state as usize) & 15]);
        text.push_str(&(state >> 32).to_string());
        text.push(' ');
    }
    text
}

fn messages(count: usize, words: usize) -> Vec<Message> {
    (0..count)
        .map(|number| {
            let role = match number % 4 {
                0 | 2 => Role::User,
                _ => Role::Assistant,
            };
            let text = prose(number, words);
            let mut message = text_message(&format!("cohort-{number}"), "s", role, &text, Some("end_turn"));
            message.at = Timestamp::from_unix_ms(1_000 + number as i64);
            if number != 0 {
                message.parent = Some(MessageId::parse(&format!("claude-code:cohort-{}", number - 1)).unwrap());
            }
            match number % 4 {
                1 => {
                    message.stop_reason = Some("tool_use".into());
                    message.blocks = vec![
                        Block::Thinking { text: Some(text), signature: Some(format!("signature-{number}")), redacted: false },
                        serde_json::from_value(serde_json::json!({
                            "type": "tool_call", "call_id": format!("call-{number}"),
                            "tool": "shell", "action": "shell", "input": {"command": format!("cargo check --package fixture-{number}")},
                            "command": format!("cargo check --package fixture-{number}")
                        })).unwrap(),
                    ];
                }
                2 => {
                    message.prompt_kind = Some(PromptKind::Harness);
                    message.stop_reason = None;
                    message.blocks = vec![serde_json::from_value(serde_json::json!({
                        "type": "tool_result", "call_id": format!("call-{}", number - 1),
                        "outcome": {"kind": "ok"}, "output": text,
                        "data": {"exact": number, "flags": [true, null, false]},
                        "reported_duration_ms": 7
                    })).unwrap()];
                }
                3 => {
                    message.blocks = vec![
                        Block::Thinking { text: Some(prose(number ^ 0x55aa, words / 4)), signature: None, redacted: false },
                        Block::Text { text },
                    ];
                }
                _ => {}
            }
            message.attrs.insert("source_record".into(), serde_json::json!(number));
            message
        })
        .collect()
}

struct Cohort {
    original_count: u64,
    sessions: Vec<SessionId>,
    observations: Vec<Observation>,
    probes: Vec<(MessageId, String, Vec<u8>)>,
    sparse: Message,
}

fn cohort(count: usize, words: usize, per_session: usize) -> Cohort {
    assert!(count >= 8 && per_session >= 8);
    let mut original = messages(count, words);
    let mut observations = Vec::new();
    let mut probes = Vec::new();
    let mut sessions = Vec::new();
    let wanted = [
        original[0].id.clone(),
        original[count / 2].id.clone(),
        original[count - 1].id.clone(),
    ];
    for (ordinal, messages) in original.chunks_mut(per_session).enumerate() {
        let name = if ordinal == 0 {
            "s".to_owned()
        } else {
            format!("s-{ordinal}")
        };
        let session = sid(&name);
        sessions.push(session.clone());
        let path = format!("cohort-live-{ordinal}");
        messages[0].parent = None;
        for message in messages.iter_mut() {
            message.session = session.clone();
        }
        let mut start = 0;
        for chunk in messages.chunks(128) {
            let input = observation(&path, Route::Live, start, chunk.to_vec());
            for row in &input.batch.parse[0].output.messages {
                if wanted.contains(&row.message.id) {
                    probes.push((
                        row.message.id.clone(),
                        input.batch.input.id.clone(),
                        input.bytes[(row.offset - start) as usize..(row.end - start) as usize]
                            .to_vec(),
                    ));
                }
            }
            start = input.batch.length;
            observations.push(input);
        }
    }
    // The same native messages retain both provenance claims. This export
    // also fills an otherwise unknown model and usage field group.
    let mut exported = original[..4].to_vec();
    for message in &mut exported {
        message.model = Some("claude-opus-4-1".into());
        message.usage.input = Some(1000);
        message.usage.output = Some(100);
    }
    observations.push(observation("cohort-export", Route::Export, 0, exported));

    let mut sparse = text_message(
        "sparse-cohort",
        "s",
        Role::Assistant,
        "cohort sparse first café",
        Some("end_turn"),
    );
    sparse.blocks.push(Block::Text {
        text: "cohort sparse second parser".into(),
    });
    for ordinal in 0..2 {
        let mut part = sparse.clone();
        part.blocks = vec![part.blocks[ordinal].clone()];
        let mut input = observation(
            &format!("cohort-sparse-{ordinal}"),
            Route::Live,
            0,
            vec![part],
        );
        input.batch.parse[0].checkpoint = None;
        input.batch.parse[0].output.block_selections.insert(
            0,
            polylogue_model::bundle::MessageProjection {
                source_version: sparse.content_hash().unwrap(),
                block_count: 2,
                block_ordinals: vec![ordinal as u64],
            },
        );
        observations.push(input);
    }
    let old = text_message(
        "cohort-rewrite",
        "s",
        Role::Assistant,
        "cohort old parser value",
        Some("end_turn"),
    );
    observations.push(observation(
        "cohort-rewritten",
        Route::Live,
        0,
        vec![old.clone()],
    ));
    let mut new = old;
    new.blocks = vec![Block::Text {
        text: "cohort rewritten café parser value".into(),
    }];
    let mut rewritten = observation("cohort-rewritten", Route::Live, 0, vec![new]);
    rewritten.batch.change = Change::Rewritten;
    observations.push(rewritten);
    Cohort {
        original_count: count as u64,
        sessions,
        observations,
        probes,
        sparse,
    }
}

fn capture(w: &mut TestArchive, fixture: &Cohort) -> Vec<Batch> {
    let batches = fixture
        .observations
        .iter()
        .map(|input| w.commit_source(input.batch.clone(), &input.bytes).unwrap())
        .collect();
    assert!(w.writer.flush_rows().unwrap().errors.is_empty());
    w.writer.stop();
    assert!(w.archive.index_build.building());
    batches
}

fn memory_cohort(main: &Connection, archive: &Archive) -> (String, Connection) {
    let uri = format!(
        "file:row-cohort-{}?mode=memory&cache=shared",
        uuid::Uuid::now_v7()
    );
    let mut memory =
        open_write(Path::new(&uri), archive.memory_budget().writer_cache_mib()).unwrap();
    rusqlite::backup::Backup::new(main, &mut memory)
        .unwrap()
        .run_to_completion(256, std::time::Duration::ZERO, None)
        .unwrap();
    memory
        .execute(
            "ATTACH DATABASE ?1 AS a",
            [format!("file:{}?mode=ro", archive.archive_db().display())],
        )
        .unwrap();
    crate::source::views(&memory).unwrap();
    assert_eq!(
        memory
            .scalar::<u64>("SELECT count(*) FROM message", [])
            .unwrap(),
        0,
        "the closed-cohort bound must not copy a prior parsed index"
    );
    (uri, memory)
}

fn archive(disk: bool) -> TestArchive {
    let directory = if disk {
        // xtask's ordinary temporary directory is tmpfs. The performance
        // cohort must charge real file writes and final sync on both arms.
        let target = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/row-construction");
        std::fs::create_dir_all(&target).unwrap();
        tempfile::Builder::new()
            .prefix("cohort-")
            .tempdir_in(target)
            .unwrap()
    } else {
        tempfile::tempdir().unwrap()
    };
    let (archive, lock) = Archive::open_or_create(&directory.path().join("archive")).unwrap();
    let archive =
        archive.with_memory_budget(crate::archive::MemoryBudget::from_allowance(96 << 20));
    let writer = Writer::start(&archive).unwrap();
    TestArchive {
        writer,
        _lock: lock,
        archive,
        _dir: directory,
    }
}

fn drain(archive: &Archive) {
    for path in [archive.index_db(), archive.archive_db()] {
        let conn = open_write(&path, 8).unwrap();
        let status: (i64, i64, i64) = conn
            .query_row("PRAGMA main.wal_checkpoint(TRUNCATE)", [], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?))
            })
            .unwrap();
        assert_eq!(status.0, 0, "{}", path.display());
        drop(conn);
        std::fs::File::open(path).unwrap().sync_all().unwrap();
    }
    std::fs::File::open(archive.archive_db().parent().unwrap())
        .unwrap()
        .sync_all()
        .unwrap();
}

fn finalize(archive: &Archive) -> Archive {
    let archive = archive.reopened();
    let mut writer = Writer::start(&archive).unwrap();
    writer.derive().unwrap();
    assert!(writer.flush_rows().unwrap().errors.is_empty());
    writer.stop();
    drop(writer);
    assert!(!archive.index_build.building());
    drain(&archive);
    archive
}

#[derive(Debug, PartialEq)]
struct Reading {
    messages: (u64, String),
    citations: Vec<(String, Vec<u8>)>,
    ranks: BTreeMap<String, Vec<(Vec<u8>, f64)>>,
    memberships: Vec<(String, String, i64, i64, i64, Vec<u8>)>,
}

fn reading(archive: &Archive, fixture: &Cohort) -> Reading {
    let pool = ReadPool::open(archive).unwrap();
    let reader = pool.reader().unwrap();
    let mut model_hash = blake3::Hasher::new();
    let mut count = 0u64;
    for session in &fixture.sessions {
        reader
            .visit_messages(session, false, |message| {
                if message.id == MessageId::parse("claude-code:cohort-0").unwrap() {
                    assert_eq!(message.model.as_deref(), Some("claude-opus-4-1"));
                    assert_eq!(message.usage.input, Some(1000));
                    assert_eq!(message.usage.output, Some(100));
                }
                model_hash.update(&message.content_hash().unwrap().0);
                count += 1;
                std::ops::ControlFlow::Continue(())
            })
            .unwrap();
    }
    assert_eq!(count, fixture.original_count + 2);
    let messages = (count, model_hash.finalize().to_hex().to_string());
    let mut citations = Vec::new();
    for (id, input, expected) in &fixture.probes {
        let record = reader
            .stored_record(crate::items::RecordTable::Message, &id.to_string())
            .unwrap()
            .unwrap();
        assert_eq!(&record.input, input);
        assert_eq!(&record.bytes, expected);
        citations.push((record.input, record.bytes));
    }
    let (sparse, layout) = reader
        .message_with_block_selection(&fixture.sparse.id)
        .unwrap()
        .unwrap();
    assert_eq!(sparse, fixture.sparse);
    assert_eq!(layout.block_ordinals, [0, 1]);
    let (rewritten, _) = reader
        .message_with_block_selection(&MessageId::parse("claude-code:cohort-rewrite").unwrap())
        .unwrap()
        .unwrap();
    assert_eq!(rewritten.text(), "cohort rewritten café parser value");
    drop(reader);
    drop(pool);
    let conn = crate::archive::open_read(&archive.index_db()).unwrap();
    assert_eq!(
        conn.scalar::<u64>("SELECT count(*) FROM pragma_foreign_key_check", [])
            .unwrap(),
        0
    );
    let memberships = conn.prepare("SELECT m.id,i.id,s.chain,s.source_start,s.source_end,v.version_hash FROM message_source s JOIN message m ON m.pk=s.message_pk JOIN input_ref i ON i.pk=s.input_pk JOIN message_version v ON v.pk=s.version_pk ORDER BY m.id,i.id,s.chain")
        .unwrap().query_map([], |row| <(_,_,_,_,_,_)>::try_from(row)).unwrap()
        .collect::<rusqlite::Result<Vec<_>>>().unwrap();
    assert!(
        memberships.len() as u64 > messages.0,
        "overlapping observations must survive"
    );
    assert!(
        conn.scalar::<u64>("SELECT count(*) FROM message_route", [])
            .unwrap()
            > 0
    );
    let mut ranks = BTreeMap::new();
    for table in ["text_fts", "output_fts"] {
        for query in ["cohort parser", "cafe"] {
            let found = conn.prepare(&format!("SELECT t.hash,{table}.rank FROM {table} JOIN text t ON t.id={table}.rowid WHERE {table} MATCH ?1 ORDER BY {table}.rank,t.hash"))
                .unwrap().query_map([query], |row| Ok((row.get(0)?,row.get(1)?))).unwrap()
                .collect::<rusqlite::Result<Vec<_>>>().unwrap();
            assert!(!found.is_empty(), "{table}: {query}");
            ranks.insert(format!("{table}:{query}"), found);
        }
    }
    Reading {
        messages,
        citations,
        ranks,
        memberships,
    }
}

#[derive(Debug, serde::Serialize)]
struct Measurement {
    raw: Phase,
    seed: Option<Phase>,
    build: Phase,
    construction: Construction,
    transfer: Option<Phase>,
    finalize: Phase,
    index_bytes: u64,
    archive_bytes: u64,
    cache_mib: i64,
}

fn run(fixture: &Cohort, staged: bool, disk: bool, fail_copy: bool) -> (Reading, Measurement) {
    crate::timing::set_workers_enabled(true);
    let arm = if staged { "staged" } else { "ordinary" };
    let phase = |name: &str| {
        if disk {
            eprintln!("ROW_CONSTRUCTION {arm}: {name}");
        }
    };
    phase("capture originals");
    let mut w = archive(disk);
    let (batches, raw) = measured(|| capture(&mut w, fixture));
    let TestArchive {
        writer,
        archive,
        _lock,
        _dir,
    } = w;
    phase("open captured archive for row construction");
    drop(writer);
    let mut main = open_index_writer(&archive, true).unwrap();
    assert_eq!(
        main.scalar::<u64>("SELECT count(*) FROM message", [])
            .unwrap(),
        0
    );
    assert_eq!(main.scalar::<u64>("PRAGMA foreign_keys", []).unwrap(), 1);
    let budget = archive.memory_budget();
    let mut seed = None;
    let mut copy = None;
    let (construction, build) = if staged {
        phase("prepare private cohort");
        let ((uri, mut memory), preparation) = measured(|| memory_cohort(&main, &archive));
        seed = Some(preparation);
        phase("construct private rows");
        let (construction, build) = measured(|| construct(&mut memory, &batches, budget));
        assert_eq!(
            main.scalar::<u64>("SELECT count(*) FROM parse_state", [])
                .unwrap(),
            0,
            "completed private rows cannot advance the destination checkpoint"
        );
        phase("digest private rows");
        let expected = snapshot(&memory, "main");
        if fail_copy {
            let before = snapshot(&main, "main");
            assert!(
                transfer(&mut main, &uri, true).is_err(),
                "foreign keys must remain enabled"
            );
            assert_eq!(
                snapshot(&main, "main"),
                before,
                "failed copy changed visible data or progress"
            );
            assert_eq!(
                snapshot(&memory, "main"),
                expected,
                "failed copy damaged its cohort"
            );
        }
        phase("transfer private rows");
        let (result, transfer_timing) = measured(|| transfer(&mut main, &uri, false));
        result.unwrap();
        copy = Some(transfer_timing);
        phase("verify transferred row digest");
        assert_eq!(
            snapshot(&main, "main"),
            expected,
            "whole native row graph must transfer exactly"
        );
        drop(memory);
        (construction, build)
    } else {
        phase("construct rows in the destination");
        measured(|| construct(&mut main, &batches, budget))
    };
    phase("finalize and sync the destination");
    drop(main);
    let (archive, finalization) = measured(|| {
        crate::archive::index_build::keep_progress(&archive).unwrap();
        finalize(&archive)
    });
    phase("verify search, membership, and citations");
    let result = reading(&archive, fixture);
    let measurement = Measurement {
        raw,
        seed,
        build,
        construction,
        transfer: copy,
        finalize: finalization,
        index_bytes: std::fs::metadata(archive.index_db()).unwrap().len(),
        archive_bytes: std::fs::metadata(archive.archive_db()).unwrap().len(),
        cache_mib: budget.writer_cache_mib(),
    };
    if disk {
        eprintln!(
            "ROW_CONSTRUCTION {arm} complete {}",
            serde_json::to_string(&measurement).unwrap()
        );
    }
    (result, measurement)
}

#[test]
fn a_closed_native_row_graph_preserves_overlaps_ranks_and_original_citations() {
    let fixture = cohort(16, 16, 8);
    let (ordinary, _) = run(&fixture, false, false, false);
    let (copied, _) = run(&fixture, true, false, false);
    assert_eq!(ordinary, copied);
}

#[test]
fn a_failed_graph_transfer_keeps_the_checkpoint_and_can_retry() {
    let fixture = cohort(8, 8, 8);
    let (ordinary, _) = run(&fixture, false, false, false);
    let (retried, _) = run(&fixture, true, false, true);
    assert_eq!(ordinary, retried);
}

#[cfg(not(debug_assertions))]
#[test]
fn bounded_full_row_construction_reports_its_complete_transfer_cost() {
    const MESSAGES: usize = 32_768;
    const PER_SESSION: usize = 1_024;
    eprintln!("ROW_CONSTRUCTION: prepare {MESSAGES} messages across multiple sessions");
    let fixture = cohort(MESSAGES, 96, PER_SESSION);
    let (ordinary, before) = run(&fixture, false, true, false);
    let (copied, after) = run(&fixture, true, true, false);
    assert_eq!(ordinary, copied);
    let row_cpu = |m: &Measurement| m.construction.rows.cpu_seconds;
    let final_writer = |m: &Measurement| {
        m.finalize
            .workers_cpu_seconds
            .get("polylogue-writer")
            .copied()
            .unwrap()
    };
    let base_writer = row_cpu(&before) + final_writer(&before) + before.finalize.cpu_seconds;
    let staged_writer = after.transfer.as_ref().unwrap().cpu_seconds
        + final_writer(&after)
        + after.finalize.cpu_seconds;
    println!(
        "ROW_CONSTRUCTION {}",
        serde_json::json!({
            "base_messages": MESSAGES,
            "messages_per_session": PER_SESSION,
            "all_observations": fixture.observations.len(),
            "captured_bytes": fixture.observations.iter().map(|o| o.bytes.len() as u64).sum::<u64>(),
            "baseline": before,
            "candidate": after,
            "ordinary_serial_cpu_bound": base_writer,
            "staged_serial_cpu_bound": staged_writer,
            "note": "Actual production row writer and native finalization. Raw originals captured separately. The memory cohort is self-contained; its seed contains no prior messages. Construction and transfer run sequentially, so this is an optimistic serial ownership bound, not observed pipeline overlap or whole-build throughput. Process I/O counters are not device I/O; file sizes and final sync are included."
        })
    );
}
