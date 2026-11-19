//! Temporary physical-layout comparison. Only synthetic rows reach these files.
#![allow(unsafe_code)]
use super::*;
use rusqlite::{Connection, StatementStatus, ffi, types::Value};
use std::{collections::BTreeMap, path::Path};

type Record = Vec<Value>;
#[derive(Clone)]
struct Table {
    columns: Vec<String>,
    rows: Vec<Record>,
    ddl: String,
    indexes: Vec<String>,
}
#[derive(Clone)]
struct State {
    name: &'static str,
    tables: BTreeMap<String, Table>,
}
const PARENTS: &[&str] = &[
    "session",
    "input_ref",
    "message_version",
    "message_block_selection",
    "parse_state",
];
const FOLDED: &[(&str, &str, &str)] = &[
    (
        "claim_version_pk",
        "version_pk",
        "INTEGER NOT NULL REFERENCES message_version(pk)",
    ),
    (
        "claim_id_kind",
        "id_kind",
        "TEXT NOT NULL CHECK(claim_id_kind IN ('native','content'))",
    ),
    ("claim_main", "main", "INTEGER NOT NULL"),
    (
        "claim_route",
        "route",
        "TEXT NOT NULL CHECK(claim_route IN ('live','capture','recorded','export','copy','page_text'))",
    ),
    ("claim_seq", "seq", "INTEGER NOT NULL"),
    (
        "claim_record_position",
        "record_position",
        "INTEGER NOT NULL",
    ),
    ("claim_order_time", "order_time", "INTEGER"),
    ("claim_at", "at", "INTEGER"),
    (
        "claim_completion",
        "completion",
        "TEXT NOT NULL CHECK(claim_completion IN ('finished','unknown','in_progress'))",
    ),
];

fn cpu() -> u64 {
    let mut t = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    assert_eq!(
        unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut t) },
        0
    );
    t.tv_sec as u64 * 1_000_000_000 + t.tv_nsec as u64
}
fn index(table: &Table, column: &str) -> usize {
    table.columns.iter().position(|c| c == column).unwrap()
}
fn rows(conn: &Connection, sql: &str, params: &[Value]) -> Vec<Record> {
    let mut s = conn.prepare_cached(sql).unwrap();
    let n = s.column_count();
    s.query_map(rusqlite::params_from_iter(params), |r| {
        (0..n).map(|i| r.get(i)).collect()
    })
    .unwrap()
    .collect::<rusqlite::Result<Vec<_>>>()
    .unwrap()
}
fn table(conn: &Connection, name: &str) -> Table {
    let columns = rows(conn, &format!("PRAGMA table_info({name})"), &[])
        .into_iter()
        .map(|r| match &r[1] {
            Value::Text(x) => x.clone(),
            _ => unreachable!(),
        })
        .collect::<Vec<_>>();
    let key = rows(conn, &format!("PRAGMA table_info({name})"), &[])
        .into_iter()
        .filter_map(|r| match (&r[1], &r[5]) {
            (Value::Text(name), Value::Integer(n)) if *n > 0 => Some((*n, name.clone())),
            _ => None,
        })
        .collect::<BTreeMap<_, _>>()
        .into_values()
        .collect::<Vec<_>>();
    Table {
        columns,
        rows: rows(conn, &format!("SELECT * FROM {name} ORDER BY {}", key.join(",")), &[]),
        ddl: conn.query_row("SELECT sql FROM sqlite_schema WHERE type='table' AND name=?1", [name], |r| r.get(0)).unwrap(),
        indexes: rows(conn, "SELECT sql FROM sqlite_schema WHERE type='index' AND tbl_name=?1 AND sql IS NOT NULL ORDER BY name", &[Value::Text(name.to_string())]).into_iter().map(|r| match &r[0] { Value::Text(x) => x.clone(), _ => unreachable!() }).collect(),
    }
}
fn snapshot(conn: &Connection, name: &'static str) -> State {
    State {
        name,
        tables: PARENTS
            .iter()
            .chain([&"message", &"message_source"])
            .map(|name| (name.to_string(), table(conn, name)))
            .collect(),
    }
}
fn apply(
    conn: &mut Connection,
    b: &mut Batch,
    upkeep: &resolve::IndexUpkeep,
    texts: &crate::text_rows::TextIds,
) -> usize {
    let input_bytes = serde_json::to_vec(&b.parse[0].output.messages)
        .unwrap()
        .len();
    b.length = b.parse[0]
        .output
        .messages
        .iter()
        .map(|r| r.end.max(r.offset) + 1)
        .max()
        .unwrap_or(b.length)
        .max(b.length);
    b.parse[0].parsed_upto = b.length;
    resolve::prepare_index(conn, b, 1, upkeep).unwrap();
    let tx = conn.transaction().unwrap();
    let changes = crate::change_summary::ChangeSummary::start(&tx).unwrap();
    write_rows(
        &tx,
        b,
        &b.parse[0],
        rows::ReadingWrite {
            texts,
            resolution: b.parse[0].resolution.as_ref().unwrap(),
            counter: 1,
            upkeep,
            source_commit: Some(&fixture_commit()),
            changes: &changes,
            retraction: false,
            replacement: false,
        },
    )
    .unwrap();
    rows::finish(&tx, &changes, upkeep, texts, 1).unwrap();
    tx.commit().unwrap();
    input_bytes
}
fn prose(n: usize) -> Vec<MessageRow> {
    (0..n)
        .map(|n| {
            row(
                n as u64,
                text_message(
                    &format!("m{n}"),
                    "s",
                    Role::Assistant,
                    &format!("message {n}: {}", "synthetic body text ".repeat(64)),
                    Some("end_turn"),
                ),
            )
        })
        .collect()
}
fn states(root: &Path) -> (Vec<State>, usize) {
    let (archive, _lock) = Archive::open_or_create(root).unwrap();
    let mut conn = crate::archive::index_build::open_writer(&archive).unwrap();
    let upkeep =
        resolve::IndexUpkeep::AtBuildEnd(Arc::new(crate::key_maps::KeyMaps::load(&conn).unwrap()));
    let texts =
        crate::text_rows::TextIds::new(&conn, true, archive.memory_budget().keys() / 3).unwrap();
    let mut result = Vec::new();
    let mut input_bytes = 0;
    let mut first = batch("input", Route::Live, 0, prose(1024), &["s"]);
    input_bytes += apply(&mut conn, &mut first, &upkeep, &texts);
    result.push(snapshot(&conn, "principal"));
    let mut extra = prose(256);
    for r in &mut extra {
        r.message.model = Some("synthetic-model".into());
    }
    input_bytes += apply(
        &mut conn,
        &mut batch("other", Route::Capture, 0, extra, &["s"]),
        &upkeep,
        &texts,
    );
    input_bytes += apply(
        &mut conn,
        &mut batch("a-source", Route::Live, 0, prose(128), &["s"]),
        &upkeep,
        &texts,
    );
    result.push(snapshot(&conn, "multi_source"));
    for (input, ordinals) in [("sparse-first", vec![2u64]), ("sparse-second", vec![0, 1])] {
        let mut b = batch(input, Route::Live, 0, vec![], &["s"]);
        for n in 0..16 {
            let mut original = text_message(
                &format!("sparse{n}"),
                "s",
                Role::Assistant,
                "",
                Some("end_turn"),
            );
            original.blocks = (0..3)
                .map(|i| Block::Text {
                    text: format!("sparse {n} original block {i}"),
                })
                .collect();
            let selection = polylogue_model::bundle::MessageProjection {
                source_version: original.content_hash().unwrap(),
                block_count: 3,
                block_ordinals: ordinals.clone(),
            };
            let mut part = original.clone();
            part.blocks = ordinals
                .iter()
                .map(|i| original.blocks[*i as usize].clone())
                .collect();
            b.parse[0].output.messages.push(row(n, part));
            b.parse[0]
                .output
                .block_selections
                .insert(n as usize, selection);
        }
        b.parse[0].counts = counts(16);
        b.parse[0].prepare();
        input_bytes += apply(&mut conn, &mut b, &upkeep, &texts);
    }
    result.push(snapshot(&conn, "sparse"));
    let mut rewritten = prose(768).into_iter().skip(256).collect::<Vec<_>>();
    for r in &mut rewritten {
        r.message.blocks[0] = Block::Text {
            text: format!("rewritten source message {}", r.offset),
        };
    }
    let mut b = batch("input", Route::Live, 0, rewritten, &["s"]);
    b.change = Change::Reparse { chain: 1 };
    b.parse[0].reset = true;
    input_bytes += apply(&mut conn, &mut b, &upkeep, &texts);
    result.push(snapshot(&conn, "rewrite"));
    (result, input_bytes)
}

fn insert_sql(name: &str, columns: &[String]) -> String {
    format!(
        "INSERT INTO {name}({}) VALUES({})",
        columns.join(","),
        (1..=columns.len())
            .map(|i| format!("?{i}"))
            .collect::<Vec<_>>()
            .join(",")
    )
}
fn install(conn: &Connection, name: &str, table: &Table) {
    let sql = insert_sql(name, &table.columns);
    let mut s = conn.prepare_cached(&sql).unwrap();
    for r in &table.rows {
        s.execute(rusqlite::params_from_iter(r)).unwrap();
    }
}
fn physical(state: &State, folded: bool) -> BTreeMap<String, Table> {
    let mut result = state.tables.clone();
    if !folded {
        return result;
    }
    let source = &state.tables["message_source"];
    let messages = &state.tables["message"];
    let mi = index(messages, "pk");
    let input = index(messages, "input_pk");
    let chain = index(messages, "source_chain");
    let s_pk = index(source, "message_pk");
    let s_input = index(source, "input_pk");
    let s_chain = index(source, "chain");
    let mut folded_messages = messages.clone();
    let extra_columns = FOLDED
        .iter()
        .map(|(name, _, ddl)| format!("{name} {ddl}"))
        .collect::<Vec<_>>()
        .join(",");
    // Add columns before the table-level composite UNIQUE constraint.
    folded_messages.ddl = folded_messages.ddl.replace(
        "UNIQUE (pk, session_pk)",
        &format!("{extra_columns}, UNIQUE (pk, session_pk)"),
    );
    assert_ne!(folded_messages.ddl, messages.ddl);
    folded_messages
        .columns
        .extend(FOLDED.iter().map(|(name, ..)| name.to_string()));
    let mut main = std::collections::BTreeSet::new();
    for r in &mut folded_messages.rows {
        let candidates = source
            .rows
            .iter()
            .filter(|s| s[s_pk] == r[mi] && s[s_input] == r[input] && s[s_chain] == r[chain])
            .collect::<Vec<_>>();
        assert_eq!(
            candidates.len(),
            1,
            "every selected message has exactly one main claim"
        );
        let s = candidates[0];
        assert_eq!(
            s[index(source, "session_pk")],
            r[index(messages, "session_pk")]
        );
        assert_eq!(
            s[index(source, "source_start")],
            r[index(messages, "source_start")]
        );
        assert_eq!(
            s[index(source, "source_end")],
            r[index(messages, "source_end")]
        );
        main.insert((
            match r[mi] {
                Value::Integer(x) => x,
                _ => unreachable!(),
            },
            match r[input] {
                Value::Integer(x) => x,
                _ => unreachable!(),
            },
            match r[chain] {
                Value::Integer(x) => x,
                _ => unreachable!(),
            },
        ));
        r.extend(
            FOLDED
                .iter()
                .map(|(_, name, _)| s[index(source, name)].clone()),
        );
    }
    folded_messages.indexes.extend([
        "CREATE INDEX message_inline_input ON message(input_pk,source_chain,source_start)",
        "CREATE INDEX message_inline_order ON message(input_pk,source_chain,session_pk,claim_seq,claim_record_position)",
        "CREATE INDEX message_inline_version ON message(claim_version_pk)",
    ].map(str::to_owned));
    let mut extras = source.clone();
    extras.ddl = extras.ddl.replace("message_source", "message_source_extra");
    extras.indexes = extras
        .indexes
        .iter()
        .map(|sql| sql.replace("message_source", "message_source_extra"))
        .collect();
    extras.rows.retain(|s| {
        !main.contains(&(
            match s[s_pk] {
                Value::Integer(x) => x,
                _ => unreachable!(),
            },
            match s[s_input] {
                Value::Integer(x) => x,
                _ => unreachable!(),
            },
            match s[s_chain] {
                Value::Integer(x) => x,
                _ => unreachable!(),
            },
        ))
    });
    assert_eq!(main.len() + extras.rows.len(), source.rows.len());
    result.insert("message".into(), folded_messages);
    result.remove("message_source");
    result.insert("message_source_extra".into(), extras);
    result
}
fn open(path: &Path, tables: &BTreeMap<String, Table>, folded: bool) -> Connection {
    let conn = crate::archive::open_write(path, 8).unwrap();
    conn.pragma_update(None, "journal_mode", "persist").unwrap();
    conn.pragma_update(None, "synchronous", "off").unwrap();
    conn.pragma_update(None, "temp_store", "file").unwrap();
    for name in PARENTS.iter().copied().chain([
        "message",
        if folded {
            "message_source_extra"
        } else {
            "message_source"
        },
    ]) {
        let t = &tables[name];
        conn.execute_batch(&t.ddl).unwrap();
        for sql in &t.indexes {
            conn.execute_batch(sql).unwrap();
        }
        if PARENTS.contains(&name) {
            install(&conn, name, t);
        }
    }
    assert_eq!(
        conn.query_row("PRAGMA foreign_keys", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        1
    );
    conn
}
fn cache(conn: &Connection) -> BTreeMap<&'static str, i32> {
    [
        ("hits", ffi::SQLITE_DBSTATUS_CACHE_HIT),
        ("misses", ffi::SQLITE_DBSTATUS_CACHE_MISS),
        ("writes", ffi::SQLITE_DBSTATUS_CACHE_WRITE),
    ]
    .into_iter()
    .map(|(name, op)| {
        let (mut value, mut peak) = (0, 0);
        assert_eq!(
            unsafe { ffi::sqlite3_db_status(conn.handle(), op, &mut value, &mut peak, 1) },
            ffi::SQLITE_OK
        );
        (name, value)
    })
    .collect()
}
fn io() -> BTreeMap<String, u64> {
    std::fs::read_to_string("/proc/thread-self/io")
        .unwrap()
        .lines()
        .map(|l| {
            let (k, v) = l.split_once(": ").unwrap();
            (k.to_string(), v.parse().unwrap())
        })
        .collect()
}
fn write(
    conn: &mut Connection,
    tables: &BTreeMap<String, Table>,
    folded: bool,
) -> serde_json::Value {
    let message = &tables["message"];
    let name = if folded {
        "message_source_extra"
    } else {
        "message_source"
    };
    let source = &tables[name];
    let msg_sql = insert_sql("message", &message.columns);
    let src_sql = insert_sql(name, &source.columns);
    let source_key = index(source, "message_pk");
    let mut by_message = BTreeMap::<i64, Vec<&Record>>::new();
    for s in &source.rows {
        if let Value::Integer(pk) = s[source_key] {
            by_message.entry(pk).or_default().push(s);
        }
    }
    conn.flush_prepared_statement_cache();
    cache(conn);
    let before_io = io();
    let before_cpu = cpu();
    let tx = conn.transaction().unwrap();
    {
        let mut messages = tx.prepare_cached(&msg_sql).unwrap();
        let mut sources = tx.prepare_cached(&src_sql).unwrap();
        for m in &message.rows {
            messages.execute(rusqlite::params_from_iter(m)).unwrap();
            let Value::Integer(pk) = m[index(message, "pk")] else {
                unreachable!()
            };
            for s in by_message.get(&pk).into_iter().flatten() {
                sources
                    .execute(rusqlite::params_from_iter(s.iter()))
                    .unwrap();
            }
        }
    }
    tx.commit().unwrap();
    let elapsed = cpu() - before_cpu;
    let after_io = io();
    let status = cache(conn);
    let msg_vm = conn
        .prepare_cached(&msg_sql)
        .unwrap()
        .get_status(StatementStatus::VmStep);
    let src_vm = conn
        .prepare_cached(&src_sql)
        .unwrap()
        .get_status(StatementStatus::VmStep);
    serde_json::json!({"cpu_ns":elapsed,"message_vm":msg_vm,"claim_vm":src_vm,"cache":status,"io":after_io.into_iter().map(|(k,v)|{let d=v-before_io[&k];(k,d)}).collect::<BTreeMap<_,_>>(),"message_rows":message.rows.len(),"extra_claim_rows":source.rows.len()})
}

const SOURCE_COLUMNS: &[&str] = &[
    "message_pk",
    "input_pk",
    "chain",
    "source_start",
    "source_end",
    "id_kind",
    "session_pk",
    "main",
    "route",
    "seq",
    "record_position",
    "order_time",
    "at",
    "completion",
    "version_pk",
];
fn inline(column: &str) -> String {
    match column {
        "message_pk" => "pk".into(),
        "chain" => "source_chain".into(),
        "input_pk" | "source_start" | "source_end" | "session_pk" => column.into(),
        _ => FOLDED
            .iter()
            .find(|(_, source, _)| *source == column)
            .unwrap()
            .0
            .into(),
    }
}
fn claims(folded: bool, predicate: &str) -> String {
    let columns = SOURCE_COLUMNS
        .iter()
        .map(|c| format!("s.{c}"))
        .collect::<Vec<_>>()
        .join(",");
    if !folded {
        return format!("SELECT {columns} FROM message_source s WHERE {predicate}");
    }
    let projected = SOURCE_COLUMNS
        .iter()
        .map(|c| format!("m.{} AS {c}", inline(c)))
        .collect::<Vec<_>>()
        .join(",");
    let mut mapped = predicate.to_string();
    for c in SOURCE_COLUMNS {
        mapped = mapped.replace(&format!("s.{c}"), &format!("m.{}", inline(c)));
    }
    format!(
        "SELECT {projected} FROM message m WHERE {mapped} UNION ALL SELECT {columns} FROM message_source_extra s WHERE {predicate}"
    )
}
fn queries(state: &State, folded: bool) -> Vec<(&'static str, String, Vec<Vec<Value>>)> {
    let pks = state.tables["message"]
        .rows
        .iter()
        .map(|r| vec![r[index(&state.tables["message"], "pk")].clone()])
        .chain([vec![Value::Integer(-1)]])
        .collect::<Vec<_>>();
    let inputs = state.tables["input_ref"]
        .rows
        .iter()
        .map(|r| {
            vec![
                r[index(&state.tables["input_ref"], "pk")].clone(),
                1.into(),
                0.into(),
            ]
        })
        .collect::<Vec<_>>();
    let versions = state.tables["message_version"]
        .rows
        .iter()
        .map(|r| vec![r[index(&state.tables["message_version"], "pk")].clone()])
        .chain([vec![(-1).into()]])
        .collect::<Vec<_>>();
    let selected=if folded {
        "SELECT m.claim_version_pk,p.layout,p.committed,m.session_pk,m.claim_route,m.claim_completion,m.claim_seq,m.claim_at,m.source_start,m.source_end FROM message m LEFT JOIN message_block_selection p ON p.version_pk=m.claim_version_pk WHERE m.pk=?1"
    }else{
        "SELECT s.version_pk,p.layout,p.committed,s.session_pk,s.route,s.completion,s.seq,s.at,s.source_start,s.source_end FROM message m JOIN message_source s ON s.message_pk=m.pk AND s.input_pk=m.input_pk AND s.chain=m.source_chain LEFT JOIN message_block_selection p ON p.version_pk=s.version_pk WHERE m.pk=?1"
    }.to_string();
    let keep=if folded {
        "SELECT EXISTS(SELECT 1 FROM message WHERE claim_version_pk=?1) OR EXISTS(SELECT 1 FROM message_source_extra WHERE version_pk=?1)"
    }else{"SELECT EXISTS(SELECT 1 FROM message_source WHERE version_pk=?1)"}.to_string();
    let trace = format!(
        "WITH sources AS ({}) SELECT s.input_pk,s.chain,s.source_start,s.source_end,s.route,(s.input_pk=m.input_pk AND s.chain=m.source_chain),p.reading_version,p.status,p.parsed_upto FROM sources s JOIN message m ON m.pk=s.message_pk LEFT JOIN parse_state p ON p.input_pk=s.input_pk AND p.chain=s.chain ORDER BY 6 DESC,1,2",
        claims(folded, "s.message_pk=?1")
    );
    vec![
        (
            "all_claims",
            format!("{} ORDER BY 2,3", claims(folded, "s.message_pk=?1")),
            pks.clone(),
        ),
        ("selected", selected, pks.clone()),
        ("trace", trace, pks),
        (
            "input_range",
            format!(
                "{} ORDER BY 4,1",
                claims(
                    folded,
                    "s.input_pk=?1 AND s.chain=?2 AND s.source_start>=?3"
                )
            ),
            inputs.clone(),
        ),
        (
            "source_order",
            format!(
                "{} ORDER BY 10,11,1",
                claims(
                    folded,
                    "s.input_pk=?1 AND s.chain=?2 AND s.session_pk=?3 AND (s.seq,s.record_position)>(?4,?5)"
                )
            ),
            inputs
                .into_iter()
                .map(|mut p| {
                    p.insert(2, 1.into());
                    p.push(0.into());
                    p
                })
                .collect(),
        ),
        ("version_guard", keep, versions),
    ]
}
fn reads(
    conn: &Connection,
    state: &State,
    folded: bool,
    expected: &mut BTreeMap<&'static str, Vec<Vec<Record>>>,
) -> serde_json::Value {
    let mut report = BTreeMap::new();
    for (name, sql, params) in queries(state, folded) {
        conn.flush_prepared_statement_cache();
        cache(conn);
        let start = cpu();
        let result = params
            .iter()
            .map(|p| rows(conn, &sql, p))
            .collect::<Vec<_>>();
        let ns = cpu() - start;
        let vm = conn
            .prepare_cached(&sql)
            .unwrap()
            .get_status(StatementStatus::VmStep);
        let db = cache(conn);
        let plan = rows(conn, &format!("EXPLAIN QUERY PLAN {sql}"), &params[0])
            .into_iter()
            .map(|r| match &r[3] {
                Value::Text(x) => x.clone(),
                _ => unreachable!(),
            })
            .collect::<Vec<_>>();
        if folded {
            assert_eq!(&result, &expected[name], "{} {name}", state.name);
        } else {
            expected.insert(name, result);
        }
        report.insert(name,serde_json::json!({"cpu_ns":ns,"vm":vm,"executions":params.len(),"cache":db,"plan":plan}));
    }
    serde_json::json!(report)
}
fn check(conn: &mut Connection, state: &State, folded: bool) {
    assert_eq!(
        rows(conn, "SELECT * FROM pragma_foreign_key_check", &[]),
        Vec::<Record>::new()
    );
    let mut actual = rows(
        conn,
        &format!("{} ORDER BY 1,2,3", claims(folded, "1")),
        &[],
    );
    let original = &state.tables["message_source"];
    let mut expected = original
        .rows
        .iter()
        .map(|r| {
            SOURCE_COLUMNS
                .iter()
                .map(|c| r[index(original, c)].clone())
                .collect::<Record>()
        })
        .collect::<Vec<_>>();
    // Ordering the diagnostic values is independent of the candidate's query.
    actual.sort_by_key(|r| format!("{r:?}"));
    expected.sort_by_key(|r| format!("{r:?}"));
    assert_eq!(actual, expected, "{} complete source relation", state.name);
    let pk = state.tables["message"].rows[0][0].clone();
    let before = rows(
        conn,
        "SELECT words FROM message WHERE pk=?1",
        std::slice::from_ref(&pk),
    );
    let tx = conn.transaction().unwrap();
    tx.execute(
        "UPDATE message SET words=words+1 WHERE pk=?1",
        rusqlite::params_from_iter([&pk]),
    )
    .unwrap();
    let sql = if folded {
        "UPDATE message SET claim_version_pk=-1 WHERE pk=?1"
    } else {
        "UPDATE message_source SET version_pk=-1 WHERE message_pk=?1"
    };
    assert_eq!(
        tx.execute(sql, rusqlite::params_from_iter([&pk]))
            .unwrap_err()
            .sqlite_error_code(),
        Some(rusqlite::ErrorCode::ConstraintViolation)
    );
    tx.rollback().unwrap();
    assert_eq!(
        rows(conn, "SELECT words FROM message WHERE pk=?1", &[pk]),
        before
    );
    assert_eq!(
        rows(conn, "SELECT * FROM pragma_foreign_key_check", &[]),
        Vec::<Record>::new()
    );
}
fn key(t: &Table, r: &Record) -> Vec<i64> {
    let fields = if t.columns[0] == "message_pk" {
        vec!["message_pk", "input_pk", "chain"]
    } else {
        vec!["pk"]
    };
    fields
        .into_iter()
        .map(|c| match r[index(t, c)] {
            Value::Integer(x) => x,
            _ => unreachable!(),
        })
        .collect()
}
fn delta(
    conn: &mut Connection,
    previous: &BTreeMap<String, Table>,
    next: &BTreeMap<String, Table>,
    folded: bool,
) -> serde_json::Value {
    // Parent/version work is common to both layouts and outside this isolated
    // header/claim comparison. Preserve the real final parent values.
    for name in PARENTS {
        let t = &next[*name];
        let info = rows(conn, &format!("PRAGMA table_info({name})"), &[]);
        let pk = info
            .iter()
            .filter_map(|r| match (&r[1], &r[5]) {
                (Value::Text(n), Value::Integer(k)) if *k > 0 => Some(n.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        let sql = format!(
            "{} ON CONFLICT({}) DO UPDATE SET {}",
            insert_sql(name, &t.columns),
            pk.join(","),
            t.columns
                .iter()
                .filter(|c| !pk.contains(c))
                .map(|c| format!("{c}=excluded.{c}"))
                .collect::<Vec<_>>()
                .join(",")
        );
        let mut stmt = conn.prepare_cached(&sql).unwrap();
        for r in &t.rows {
            stmt.execute(rusqlite::params_from_iter(r)).unwrap();
        }
    }
    let mut operations: Vec<(String, Record)> = Vec::new();
    let mut statements = BTreeMap::<String, u64>::new();
    let source = if folded {
        "message_source_extra"
    } else {
        "message_source"
    };
    // Remove source tuples before removing their message parents.
    for name in [source, "message"] {
        let old = &previous[name];
        let new = &next[name];
        let current = new
            .rows
            .iter()
            .map(|r| key(new, r))
            .collect::<std::collections::BTreeSet<_>>();
        let cols = if name == "message" {
            vec!["pk"]
        } else {
            vec!["message_pk", "input_pk", "chain"]
        };
        let sql = format!(
            "DELETE FROM {name} WHERE {}",
            cols.iter()
                .enumerate()
                .map(|(i, c)| format!("{c}=?{}", i + 1))
                .collect::<Vec<_>>()
                .join(" AND ")
        );
        for r in &old.rows {
            let k = key(old, r);
            if !current.contains(&k) {
                operations.push((sql.clone(), k.into_iter().map(Value::Integer).collect()));
                *statements.entry(sql.clone()).or_default() += 1;
            }
        }
    }
    // Insert or update the final header before its exception tuples.
    for name in ["message", source] {
        let old = &previous[name];
        let new = &next[name];
        let previous = old
            .rows
            .iter()
            .map(|r| (key(old, r), r))
            .collect::<BTreeMap<_, _>>();
        let pkcols = if name == "message" {
            vec!["pk"]
        } else {
            vec!["message_pk", "input_pk", "chain"]
        };
        let insert = insert_sql(name, &new.columns);
        let upsert = format!(
            "{insert} ON CONFLICT({}) DO UPDATE SET {}",
            pkcols.join(","),
            new.columns
                .iter()
                .filter(|c| !pkcols.contains(&c.as_str()))
                .map(|c| format!("{c}=excluded.{c}"))
                .collect::<Vec<_>>()
                .join(",")
        );
        for r in &new.rows {
            let previous = previous.get(&key(new, r));
            if previous.is_some_and(|p| *p == r) {
                continue;
            }
            let sql = if previous.is_some() { &upsert } else { &insert };
            operations.push((sql.clone(), r.clone()));
            *statements.entry(sql.clone()).or_default() += 1;
        }
    }
    // The real writer receives an off-thread prepared choice; do not charge
    // this diagnostic's whole-state diff construction to either serial arm.
    conn.flush_prepared_statement_cache();
    cache(conn);
    let before_io = io();
    let before_cpu = cpu();
    let tx = conn.transaction().unwrap();
    for (sql, row) in operations {
        tx.prepare_cached(&sql)
            .unwrap()
            .execute(rusqlite::params_from_iter(row))
            .unwrap();
    }
    tx.commit().unwrap();
    let elapsed = cpu() - before_cpu;
    let after_io = io();
    let db = cache(conn);
    let vm = statements
        .keys()
        .map(|s| {
            conn.prepare_cached(s)
                .unwrap()
                .get_status(StatementStatus::VmStep) as u64
        })
        .sum::<u64>();
    serde_json::json!({"cpu_ns":elapsed,"vm":vm,"executions":statements.values().sum::<u64>(),"cache":db,"io":after_io.into_iter().map(|(k,v)|{let d=v-before_io[&k];(k,d)}).collect::<BTreeMap<_,_>>()})
}
fn size(path: &Path) -> u64 {
    std::fs::read_dir(path)
        .unwrap()
        .map(|e| {
            let e = e.unwrap();
            if e.file_type().unwrap().is_dir() {
                size(&e.path())
            } else {
                e.metadata().unwrap().len()
            }
        })
        .sum()
}

pub(super) fn run() {
    let root = tempfile::tempdir().unwrap();
    let (states, input_bytes) = states(&root.path().join("production"));
    let mut expected = BTreeMap::new();
    for iteration in 0..2 {
        for folded in if iteration == 0 {
            [false, true]
        } else {
            [true, false]
        } {
            let first = physical(&states[0], folded);
            let path = root.path().join(format!("layout-{iteration}-{folded}.db"));
            let mut conn = open(&path, &first, folded);
            let written = write(&mut conn, &first, folded);
            check(&mut conn, &states[0], folded);
            // Each order uses independently created databases. Oracles are the
            // current schema, queried from rows made by the production writer.
            let mut oracle = if folded {
                expected.get("principal").cloned().unwrap_or_default()
            } else {
                BTreeMap::new()
            };
            let read = reads(&conn, &states[0], folded, &mut oracle);
            if !folded {
                expected.insert("principal", oracle);
            }
            println!(
                "FOLDED_CLAIM {}",
                serde_json::json!({"state":"principal","iteration":iteration,"folded":folded,"write":written,"reads":read})
            );
            let mut previous = first;
            for state in states.iter().skip(1) {
                let next = physical(state, folded);
                let changed = delta(&mut conn, &previous, &next, folded);
                check(&mut conn, state, folded);
                let mut oracle = if folded {
                    expected.get(state.name).cloned().unwrap_or_default()
                } else {
                    BTreeMap::new()
                };
                let read = reads(&conn, state, folded, &mut oracle);
                if !folded {
                    expected.insert(state.name, oracle);
                }
                println!(
                    "FOLDED_CLAIM {}",
                    serde_json::json!({"state":state.name,"iteration":iteration,"folded":folded,"transition":changed,"reads":read})
                );
                previous = next;
            }
            let close = cpu();
            conn.close().unwrap();
            let close_cpu = cpu() - close;
            std::fs::File::open(&path).unwrap().sync_all().unwrap();
            println!(
                "FOLDED_FILES {}",
                serde_json::json!({"iteration":iteration,"folded":folded,"db_bytes":std::fs::metadata(&path).unwrap().len(),"close_cpu_ns":close_cpu})
            );
        }
    }
    let output = size(root.path());
    assert!(input_bytes <= 16 * 1024 * 1024);
    assert!(output <= 128 * 1024 * 1024, "{output}");
    println!("FOLDED_LIMITS input_bytes={input_bytes} output_bytes={output}");
}
