//! Temporary bounded diagnostic. No expanded SQL or captured input is recorded.
#![allow(unsafe_code)]
use super::*;
use rusqlite::{Connection, ffi};
use std::{cell::RefCell, collections::BTreeMap, ffi::CStr};

#[derive(Default)]
struct Counters {
    statements: BTreeMap<String, (u64, u64)>,
    mutations: BTreeMap<String, u64>,
    phases: BTreeMap<&'static str, u64>,
}
thread_local! {
    static COUNTERS: RefCell<Option<Counters>> = const { RefCell::new(None) };
}

fn cpu_ns() -> u64 {
    let mut time = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    assert_eq!(
        unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut time) },
        0
    );
    time.tv_sec as u64 * 1_000_000_000 + time.tv_nsec as u64
}

pub(crate) struct Phase(Option<(&'static str, u64)>);
pub(crate) fn phase(name: &'static str) -> Phase {
    Phase(COUNTERS.with(|c| c.borrow().as_ref().map(|_| (name, cpu_ns()))))
}
impl Drop for Phase {
    fn drop(&mut self) {
        if let Some((name, start)) = self.0 {
            let elapsed = cpu_ns() - start;
            COUNTERS.with(|c| {
                if let Some(c) = &mut *c.borrow_mut() {
                    *c.phases.entry(name).or_default() += elapsed;
                }
            });
        }
    }
}

unsafe extern "C" fn trace(
    code: u32,
    _context: *mut std::ffi::c_void,
    statement: *mut std::ffi::c_void,
    _duration: *mut std::ffi::c_void,
) -> i32 {
    if code != ffi::SQLITE_TRACE_PROFILE as u32 {
        return 0;
    }
    let statement = statement.cast::<ffi::sqlite3_stmt>();
    // Reset after every completed execution, including cached statements. A
    // cumulative cached counter would count old work again. Trigger programs
    // belong to their outer statement; there is only one PROFILE event.
    let steps = unsafe { ffi::sqlite3_stmt_status(statement, ffi::SQLITE_STMTSTATUS_VM_STEP, 1) };
    let sql = unsafe { CStr::from_ptr(ffi::sqlite3_sql(statement)) }.to_string_lossy();
    let sql = sql.split_whitespace().collect::<Vec<_>>().join(" ");
    COUNTERS.with(|c| {
        let mut c = c.borrow_mut();
        if let Some(c) = &mut *c {
            let item = c.statements.entry(sql).or_default();
            item.0 += 1;
            item.1 += steps as u64;
        }
    });
    0
}

fn start(conn: &Connection, trace_sql: bool) {
    conn.flush_prepared_statement_cache();
    COUNTERS.with(|c| *c.borrow_mut() = Some(Counters::default()));
    if trace_sql {
        // This callback receives the real SQLite statement identity and can
        // reset its public counter. It never inspects SQLite's private layout.
        assert_eq!(
            unsafe {
                ffi::sqlite3_trace_v2(
                    conn.handle(),
                    ffi::SQLITE_TRACE_PROFILE as u32,
                    Some(trace),
                    std::ptr::null_mut(),
                )
            },
            ffi::SQLITE_OK
        );
        conn.update_hook(Some(|action, _db: &str, table: &str, _row| {
            COUNTERS.with(|c| {
                let mut c = c.borrow_mut();
                if let Some(c) = &mut *c {
                    *c.mutations
                        .entry(format!("{table}:{action:?}"))
                        .or_default() += 1;
                }
            });
        }))
        .unwrap();
    }
}

fn stop(conn: &Connection, name: &str, traced: bool, cpu: u64, written: u64) {
    assert_eq!(
        unsafe { ffi::sqlite3_trace_v2(conn.handle(), 0, None, std::ptr::null_mut()) },
        ffi::SQLITE_OK
    );
    conn.update_hook(None::<fn(rusqlite::hooks::Action, &str, &str, i64)>)
        .unwrap();
    let c = COUNTERS.with(|c| c.borrow_mut().take().unwrap());
    println!(
        "ROW_DIAGNOSTIC {}",
        serde_json::json!({
            "case": name, "traced": traced, "cpu_ns": cpu, "written": written,
            "statements": c.statements, "mutations": c.mutations, "phase_cpu_ns": c.phases,
        })
    );
}

fn apply_batch(
    conn: &mut Connection,
    b: &mut Batch,
    upkeep: &resolve::IndexUpkeep,
    texts: &crate::text_rows::TextIds,
    name: &str,
    traced: bool,
) {
    b.length = b.parse[0]
        .output
        .messages
        .iter()
        .map(|r| r.end.max(r.offset) + 1)
        .max()
        .unwrap()
        .max(b.length);
    b.parse[0].parsed_upto = b.length;
    resolve::prepare_index(conn, b, 1, upkeep).unwrap();
    start(conn, traced);
    let cpu = cpu_ns();
    let tx = conn.transaction().unwrap();
    let changes = crate::change_summary::ChangeSummary::start(&tx).unwrap();
    let written = write_rows(
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
    {
        let _phase = phase("finish");
        rows::finish(&tx, &changes, upkeep, texts, 1).unwrap();
    }
    tx.commit().unwrap();
    let cpu = cpu_ns() - cpu;
    stop(conn, name, traced, cpu, written);
    assert_eq!(conn.query_row("SELECT parsed_upto FROM parse_state WHERE input_pk=(SELECT pk FROM input_ref WHERE id=?1) AND chain=1", [&b.input.id], |r| r.get::<_,u64>(0)).unwrap(), b.length);
}

fn size(path: &std::path::Path) -> u64 {
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

fn assert_rows(
    conn: &Connection,
    messages: i64,
    tools: i64,
    claims: i64,
) -> Vec<(String, String, i64)> {
    for (table, count) in [
        ("message", messages),
        ("block", messages),
        ("message_source", claims),
        ("tool_call", tools),
    ] {
        assert_eq!(
            conn.query_row(&format!("SELECT count(*) FROM {table}"), [], |r| r
                .get::<_, i64>(0))
                .unwrap(),
            count,
            "{table}"
        );
    }
    assert_eq!(
        conn.query_row("SELECT count(*) FROM pragma_foreign_key_check", [], |r| r
            .get::<_, i64>(
            0
        ))
        .unwrap(),
        0
    );
    assert_eq!(
        conn.query_row(
            "SELECT count(*) FROM block WHERE coalesce(CAST(ordinal AS INTEGER),idx)<>0",
            [],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        0
    );
    let mut stmt = conn.prepare("SELECT m.id,b.kind,coalesce(CAST(b.ordinal AS INTEGER),b.idx) FROM message m JOIN block b ON b.message_pk=m.pk ORDER BY m.id,b.ordinal").unwrap();
    stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
        .unwrap()
        .collect::<std::result::Result<Vec<_>, _>>()
        .unwrap()
}

fn assert_values(conn: &Connection, expected: &[MessageRow]) {
    let mut statement = conn.prepare("SELECT b.body,g.body,t.start,t.length,coalesce(CAST(b.ordinal AS INTEGER),b.idx),m.source_start,m.source_end FROM message m JOIN block b ON b.message_pk=m.pk LEFT JOIN text t ON t.id=b.text_id LEFT JOIN text_group g ON g.id=t.group_id WHERE m.id=?1").unwrap();
    for expected in expected {
        let (block, ordinal, start, end) = statement
            .query_row([expected.message.id.to_string()], |r| {
                let mut block = crate::encoding::unpack_block(&r.get::<_, Vec<u8>>(0)?).unwrap();
                if let Some(text) = crate::text_rows::column(r, 1).unwrap() {
                    crate::encoding::restore_text(&mut block, text);
                }
                Ok((
                    block,
                    r.get::<_, u64>(4)?,
                    r.get::<_, u64>(5)?,
                    r.get::<_, u64>(6)?,
                ))
            })
            .unwrap();
        assert_eq!(block, expected.message.blocks[0]);
        assert_eq!((ordinal, start, end), (0, expected.offset, expected.end));
    }
}

fn prose() -> Vec<MessageRow> {
    (0..1024)
        .map(|n| {
            row(
                n,
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
fn tools(results: bool) -> Vec<MessageRow> {
    (0..512)
        .map(|n| {
            let mut message = text_message(
                &format!("{}{n}", if results { "r" } else { "c" }),
                "s",
                if results { Role::User } else { Role::Assistant },
                "",
                if results { None } else { Some("tool_use") },
            );
            message.blocks = if results {
                vec![Block::ToolResult {
                    reported_duration_ms: None,
                    call_id: Some(format!("call{n}")),
                    outcome: ToolOutcome::Ok,
                    output: Some(format!("result {n}: {}", "synthetic output ".repeat(16))),
                    data: None,
                    attachments: vec![],
                }]
            } else {
                vec![Block::ToolCall {
                    written_text: None,
                    run_by_script: false,
                    call_id: Some(format!("call{n}")),
                    tool: Some("Bash".into()),
                    action: ToolAction::Shell,
                    input: Json::Null,
                    command: Some(format!("printf 'synthetic {n}'")),
                    path: None,
                }]
            };
            row(n + if results { 512 } else { 0 }, message)
        })
        .collect()
}

pub(super) fn run() {
    // Independently check that cached executions are not counted cumulatively.
    let calibration = Connection::open_in_memory().unwrap();
    let mut statement = calibration.prepare("SELECT ?1").unwrap();
    assert_eq!(
        statement.query_row([42], |r| r.get::<_, i64>(0)).unwrap(),
        42
    );
    let one = statement.get_status(rusqlite::StatementStatus::VmStep) as u64;
    drop(statement);
    start(&calibration, true);
    for _ in 0..3 {
        assert_eq!(
            calibration
                .prepare_cached("SELECT ?1")
                .unwrap()
                .query_row([42], |r| r.get::<_, i64>(0))
                .unwrap(),
            42
        );
    }
    COUNTERS.with(|c| {
        assert_eq!(
            c.borrow().as_ref().unwrap().statements["SELECT ?1"],
            (3, 3 * one)
        )
    });
    stop(&calibration, "counter_calibration", true, 0, 0);
    let mut total_input = 0;
    let mut total_files = 0;
    for kind in ["prose", "tools_together", "tools_split"] {
        let mut expected = None;
        for traced in [false, true] {
            let directory = tempfile::tempdir().unwrap();
            let (archive, _lock) =
                Archive::open_or_create(&directory.path().join("archive")).unwrap();
            let mut conn = crate::archive::index_build::open_writer(&archive).unwrap();
            let upkeep = resolve::IndexUpkeep::AtBuildEnd(Arc::new(
                crate::key_maps::KeyMaps::load(&conn).unwrap(),
            ));
            let texts =
                crate::text_rows::TextIds::new(&conn, true, archive.memory_budget().keys() / 3)
                    .unwrap();
            assert_eq!(
                conn.query_row("PRAGMA foreign_keys", [], |r| r.get::<_, i64>(0))
                    .unwrap(),
                1
            );
            assert_eq!(
                conn.query_row(
                    "SELECT count(*) FROM sqlite_schema WHERE name='message_id'",
                    [],
                    |r| r.get::<_, i64>(0)
                )
                .unwrap(),
                0
            );
            if kind == "prose" {
                let rows = prose();
                total_input += 2 * serde_json::to_vec(&rows).unwrap().len() as u64;
                let mut b = batch("input", Route::Live, 0, rows.clone(), &["s"]);
                apply_batch(&mut conn, &mut b, &upkeep, &texts, "prose_fresh", traced);
                assert_rows(&conn, 1024, 0, 1024);
                assert_values(&conn, &prose());
                let mut duplicate = batch("duplicate", Route::Capture, 0, rows, &["s"]);
                apply_batch(
                    &mut conn,
                    &mut duplicate,
                    &upkeep,
                    &texts,
                    "prose_identical_claim",
                    traced,
                );
            } else {
                let calls = tools(false);
                let results = tools(true);
                total_input += serde_json::to_vec(&calls).unwrap().len() as u64
                    + serde_json::to_vec(&results).unwrap().len() as u64;
                if kind == "tools_together" {
                    let mut rows = calls
                        .into_iter()
                        .zip(results)
                        .flat_map(|(a, b)| [a, b])
                        .collect::<Vec<_>>();
                    rows.sort_by_key(|r| r.offset);
                    let mut b = batch("input", Route::Live, 0, rows, &["s"]);
                    apply_batch(&mut conn, &mut b, &upkeep, &texts, "tools_together", traced);
                } else {
                    let mut b = batch("input", Route::Live, 0, calls, &["s"]);
                    apply_batch(
                        &mut conn,
                        &mut b,
                        &upkeep,
                        &texts,
                        "tools_split_calls",
                        traced,
                    );
                    assert_rows(&conn, 512, 512, 512);
                    assert_eq!(conn.query_row("SELECT count(*) FROM tool_call WHERE outcome='unknown' AND result_block_pk IS NULL", [], |r| r.get::<_,i64>(0)).unwrap(), 512);
                    let mut b = batch("input", Route::Live, 512, results, &["s"]);
                    apply_batch(
                        &mut conn,
                        &mut b,
                        &upkeep,
                        &texts,
                        "tools_split_results",
                        traced,
                    );
                }
                assert_eq!(conn.query_row("SELECT count(*) FROM tool_call WHERE outcome='ok' AND result_block_pk IS NOT NULL", [], |r| r.get::<_,i64>(0)).unwrap(), 512);
            }
            if kind == "prose" {
                assert_values(&conn, &prose());
            } else {
                assert_values(&conn, &tools(false));
                assert_values(&conn, &tools(true));
            }
            let session = conn
                .query_row("SELECT pk FROM session WHERE id='claude-code:s'", [], |r| {
                    r.get::<_, i64>(0)
                })
                .unwrap();
            assert_eq!(
                crate::items::stored_entries(&conn, session, 0).unwrap(),
                crate::items::computed_entries(&conn, session).unwrap()
            );
            let shape = assert_rows(
                &conn,
                1024,
                if kind == "prose" { 0 } else { 512 },
                if kind == "prose" { 2048 } else { 1024 },
            );
            if let Some(expected) = &expected {
                assert_eq!(&shape, expected);
            } else {
                expected = Some(shape);
            }
            start(&conn, traced);
            let cpu = cpu_ns();
            crate::text_rows::build_search_indexes(&conn).unwrap();
            conn.execute(
                "INSERT INTO tool_call_trigram(tool_call_trigram) VALUES ('rebuild')",
                [],
            )
            .unwrap();
            for sql in crate::schema::built_at_end_sql().unwrap() {
                conn.execute(&sql, []).unwrap();
            }
            stop(
                &conn,
                &format!("{kind}_deferred_indexes"),
                traced,
                cpu_ns() - cpu,
                0,
            );
            total_files += size(directory.path());
        }
    }
    assert!(total_input <= 16 * 1024 * 1024, "{total_input}");
    assert!(total_files <= 128 * 1024 * 1024, "{total_files}");
    println!("ROW_DIAGNOSTIC_LIMITS input_bytes={total_input} output_bytes={total_files}");
}
