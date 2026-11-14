use super::*;
use crate::archive::open_write;
use rusqlite::hooks::{AuthAction, AuthContext, Authorization};
use std::sync::atomic::{AtomicU64, Ordering};

fn deny_old_projection(context: AuthContext<'_>) -> Authorization {
    match context.action {
        AuthAction::Update {
            table_name: "message",
            ..
        }
        | AuthAction::Delete {
            table_name: "block" | "tool_call" | "message_route",
        } => Authorization::Deny,
        _ => Authorization::Allow,
    }
}

fn apply(
    tx: &Transaction<'_>,
    batch: &Batch,
    step: usize,
    changes: &crate::change_summary::ChangeSummary,
) -> u64 {
    let step = &batch.parse[step];
    write_rows(
        &crate::text_rows::TextIds::new(tx, false).unwrap(),
        tx,
        batch,
        step,
        step.resolution.as_ref().unwrap(),
        1,
        None,
        Some(&fixture_receipt()),
        changes,
    )
    .unwrap()
}

fn check_items(tx: &Transaction<'_>, changes: &crate::change_summary::ChangeSummary) {
    crate::items::refresh(tx, changes).unwrap();
    let session = tx
        .query_row("SELECT pk FROM session WHERE id='claude-code:s'", [], |r| {
            r.get::<_, i64>(0)
        })
        .unwrap();
    let expected = crate::items::computed_entries(tx, session).unwrap();
    let stored = tx
        .prepare("SELECT entry FROM session_item WHERE session_pk=?1 ORDER BY seq")
        .unwrap()
        .query_map([session], |r| r.get::<_, Vec<u8>>(0))
        .unwrap()
        .map(|r| serde_json::from_slice::<polylogue_model::read::IndexEntry>(&r.unwrap()).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(stored, expected);
    assert_eq!(
        tx.query_row("SELECT count(*) FROM pragma_foreign_key_check", [], |r| r
            .get::<_, i64>(
            0
        ))
        .unwrap(),
        0
    );
}

#[test]
fn fresh_selected_messages_install_without_replacing_an_old_projection() {
    let directory = tempfile::tempdir().unwrap();
    let mut conn = open_write(&directory.path().join("index.db"), 1).unwrap();
    crate::schema::create_index(&conn).unwrap();
    conn.set_prepared_statement_cache_capacity(0);
    let tx = conn.transaction().unwrap();
    let mut first = text_message("same", "s", Role::Assistant, "losing body", None);
    first.model = Some("losing model".into());
    let mut last = text_message(
        "same",
        "s",
        Role::Assistant,
        "selected body with four words",
        Some("end_turn"),
    );
    last.model = Some("selected model".into());
    let mut b = batch(
        "input",
        Route::Live,
        0,
        vec![row(7, first), row(29, last.clone())],
        &["s"],
    );
    b.parse[0].repeats_whole_conversations = true;
    resolve::prepare_index(&tx, &mut b, 1, false).unwrap();
    assert_eq!(
        b.parse[0]
            .resolution
            .as_ref()
            .unwrap()
            .choices
            .keep_messages,
        [true, true]
    );
    let changes = crate::change_summary::ChangeSummary::start(&tx).unwrap();
    tx.authorizer(Some(deny_old_projection)).unwrap();
    assert_eq!(apply(&tx, &b, 0, &changes), 1);
    tx.authorizer(None::<fn(AuthContext<'_>) -> Authorization>)
        .unwrap();
    assert_eq!(
        tx.query_row(
            "SELECT model,words,source_start,source_end FROM message",
            [],
            |r| Ok((
                r.get::<_, String>(0)?,
                r.get::<_, i64>(1)?,
                r.get::<_, i64>(2)?,
                r.get::<_, i64>(3)?
            ))
        )
        .unwrap(),
        ("selected model".into(), 5, 29, 29)
    );
    assert_eq!(
        tx.query_row("SELECT count(*) FROM block", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        1
    );
    let text: Vec<u8> = tx
        .query_row(
            "SELECT t.body FROM block b JOIN text t ON t.id=b.text_id",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        crate::encoding::unpack_text(&text).unwrap(),
        "selected body with four words"
    );
    check_items(&tx, &changes);
}

#[test]
fn reserved_message_keys_are_not_creation_results_across_steps() {
    let directory = tempfile::tempdir().unwrap();
    let mut conn = open_write(&directory.path().join("index.db"), 1).unwrap();
    crate::schema::create_index(&conn).unwrap();
    conn.set_prepared_statement_cache_capacity(0);
    let tx = conn.transaction().unwrap();
    let mut b = batch(
        "input",
        Route::Live,
        0,
        vec![row(
            7,
            text_message("same", "s", Role::Assistant, "old body", None),
        )],
        &["s"],
    );
    b.parse[0].progress = polylogue_model::batch::StepProgress::Partway;
    let next = batch(
        "input",
        Route::Live,
        100,
        vec![row(
            129,
            text_message(
                "same",
                "s",
                Role::Assistant,
                "replacement body is different",
                Some("end_turn"),
            ),
        )],
        &["s"],
    );
    b.parse.extend(next.parse);
    b.parse[1].repeats_whole_conversations = true;
    b.length = 200;
    // Both steps reserve the key against the same pre-write snapshot.
    resolve::prepare_index(&tx, &mut b, 1, false).unwrap();
    for step in &b.parse {
        assert!(
            !step
                .resolution
                .as_ref()
                .unwrap()
                .keys
                .get(resolve::Kind::Message, b"claude-code:same")
                .unwrap()
                .existing
        );
        assert_eq!(
            step.resolution.as_ref().unwrap().choices.keep_messages,
            [true]
        );
    }
    let changes = crate::change_summary::ChangeSummary::start(&tx).unwrap();
    tx.authorizer(Some(deny_old_projection)).unwrap();
    assert_eq!(apply(&tx, &b, 0, &changes), 1);
    tx.authorizer(None::<fn(AuthContext<'_>) -> Authorization>)
        .unwrap();
    assert_eq!(apply(&tx, &b, 1, &changes), 1);
    assert_eq!(
        apply(&tx, &b, 1, &changes),
        0,
        "an exact retry keeps the admitted projection"
    );
    assert_eq!(
        tx.query_row(
            "SELECT words,source_start,in_progress FROM message",
            [],
            |r| Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, i64>(1)?,
                r.get::<_, bool>(2)?
            ))
        )
        .unwrap(),
        (4, 129, false)
    );
    assert_eq!(
        tx.query_row("SELECT count(*) FROM block", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        1
    );
    assert_eq!(
        tx.query_row("SELECT count(*) FROM message_source", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        1
    );
    let text: Vec<u8> = tx
        .query_row(
            "SELECT t.body FROM block b JOIN text t ON t.id=b.text_id",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        crate::encoding::unpack_text(&text).unwrap(),
        "replacement body is different"
    );
    check_items(&tx, &changes);
}

#[test]
fn fresh_tool_pairs_and_earlier_event_anchors_keep_the_main_source_sequence() {
    for result_first in [false, true] {
        let directory = tempfile::tempdir().unwrap();
        let mut conn = open_write(&directory.path().join("index.db"), 1).unwrap();
        crate::schema::create_index(&conn).unwrap();
        conn.set_prepared_statement_cache_capacity(0);
        let tx = conn.transaction().unwrap();
        let mut events = batch("events", Route::Live, 0, vec![], &["s"]);
        events.parse[0].output.events.push(Event {
            at_quality: polylogue_model::TimeQuality::Unknown,
            at_precision: None,
            at_until: None,
            session: Some(sid("s")),
            kind: "synthetic.anchor".into(),
            at: None,
            after: Some(MessageId::parse("claude-code:call").unwrap()),
            range: None,
            data: serde_json::json!({}),
        });
        write_test_rows(&tx, &events, &events.parse[0], 1, 1, None).unwrap();
        let mut call = text_message("call", "s", Role::Assistant, "call text", Some("tool_use"));
        call.blocks.push(serde_json::from_value(serde_json::json!({"type":"tool_call","call_id":"tool","tool":"shell","input":{"command":"true"},"command":"true"})).unwrap());
        let mut result = text_message("result", "s", Role::User, "result text", None);
        result.blocks.push(serde_json::from_value(serde_json::json!({"type":"tool_result","call_id":"tool","outcome":{"kind":"ok"},"output":"successful output"})).unwrap());
        let mut records = vec![row(17, call), row(39, result)];
        if result_first {
            records.reverse();
        }
        let mut b = batch("input", Route::Live, 0, records, &["s"]);
        resolve::prepare_index(&tx, &mut b, 1, false).unwrap();
        let changes = crate::change_summary::ChangeSummary::start(&tx).unwrap();
        tx.authorizer(Some(deny_old_projection)).unwrap();
        assert_eq!(apply(&tx, &b, 0, &changes), 2);
        tx.authorizer(None::<fn(AuthContext<'_>) -> Authorization>)
            .unwrap();
        assert_eq!(
            tx.query_row(
                "SELECT count(*) FROM tool_call WHERE result_block_pk IS NOT NULL AND outcome='ok'",
                [],
                |r| r.get::<_, i64>(0)
            )
            .unwrap(),
            1
        );
        assert_eq!(tx.query_row("SELECT seq FROM event WHERE kind='synthetic.anchor' AND after_message_pk IS NOT NULL", [], |r| r.get::<_,i64>(0)).unwrap(), 17);
        check_items(&tx, &changes);
    }
}

thread_local! {
    static STATEMENTS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}
fn count_statement(event: rusqlite::trace::TraceEvent<'_>) {
    if let rusqlite::trace::TraceEvent::Profile(..) = event {
        STATEMENTS.with(|n| n.set(n.get() + 1));
    }
}

fn thread_cpu_ticks() -> u64 {
    let stat = std::fs::read_to_string("/proc/thread-self/stat").unwrap();
    let fields = stat
        .rsplit_once(')')
        .unwrap()
        .1
        .split_whitespace()
        .collect::<Vec<_>>();
    fields[11].parse::<u64>().unwrap() + fields[12].parse::<u64>().unwrap()
}

/// A bounded production row-application measurement; printing counters does
/// not assert a machine-dependent duration or a whole-pipeline rate.
#[test]
fn prepared_message_transaction_work() {
    let directory = tempfile::tempdir().unwrap();
    let mut conn = open_write(&directory.path().join("index.db"), 1).unwrap();
    crate::schema::create_index(&conn).unwrap();
    let rows = (0..256)
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
        .collect();
    let mut b = batch("input", Route::Live, 0, rows, &["s"]);
    b.length = 512;
    b.parse[0].parsed_upto = 512;
    resolve::prepare_index(&conn, &mut b, 1, false).unwrap();
    let steps = Arc::new(AtomicU64::new(0));
    let count = steps.clone();
    conn.progress_handler(
        1,
        Some(move || {
            count.fetch_add(1, Ordering::Relaxed);
            false
        }),
    )
    .unwrap();
    STATEMENTS.with(|n| n.set(0));
    conn.trace_v2(
        rusqlite::trace::TraceEventCodes::SQLITE_TRACE_PROFILE,
        Some(count_statement),
    );
    let cpu = thread_cpu_ticks();
    let start = std::time::Instant::now();
    let tx = conn.transaction().unwrap();
    let changes = crate::change_summary::ChangeSummary::start(&tx).unwrap();
    assert_eq!(apply(&tx, &b, 0, &changes), 256);
    crate::items::refresh(&tx, &changes).unwrap();
    tx.commit().unwrap();
    let elapsed = start.elapsed();
    let cpu = thread_cpu_ticks() - cpu;
    conn.progress_handler(0, None::<fn() -> bool>).unwrap();
    conn.trace_v2(rusqlite::trace::TraceEventCodes::empty(), None);
    println!(
        "prepared transaction: messages=256 statements={} vm_steps={} thread_cpu_ticks={cpu} elapsed_us={}",
        STATEMENTS.with(|n| n.get()),
        steps.load(Ordering::Relaxed),
        elapsed.as_micros()
    );
    assert_eq!(
        conn.query_row("SELECT count(*) FROM block", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        256
    );
}

#[test]
fn early_copy_materialization_then_refresh_keeps_one_body_through_release() {
    let w = test_archive();
    let copy = SessionId::parse("ai-studio:content-ab").unwrap();
    let known = SessionId::parse("ai-studio:drive-file").unwrap();
    let ambiguous = SessionId::parse("ai-studio:other-file").unwrap();
    let mut message = text_message("unused", "s", Role::User, "unchanged copied body", None);
    message.id = MessageId::parse("ai-studio:content-ab~user-0123-0").unwrap();
    message.session = copy.clone();
    let mut copied = batch(
        "copied",
        Route::Capture,
        0,
        vec![row(7, message.clone())],
        &[],
    );
    copied.parse[0]
        .output
        .sessions
        .push(Session::new(copy.clone(), SessionKind::Main));
    w.store_reading(copied, Some(&SOURCE_RECORD)).unwrap();
    assert!(w.writer.flush().unwrap().errors.is_empty());
    let declare = |input: &str, id: &SessionId| {
        let mut b = batch(input, Route::Live, 0, vec![], &[]);
        let mut session = Session::new(id.clone(), SessionKind::Main);
        session.aliases.push(copy.clone());
        b.parse[0].output.sessions.push(session);
        b
    };
    w.store_reading(declare("native", &known), Some(&SOURCE_RECORD))
        .unwrap();
    assert!(w.writer.flush().unwrap().errors.is_empty());
    {
        let pool = ReadPool::open(&w.archive).unwrap();
        let r = pool.reader().unwrap();
        let mut selected = message.clone();
        selected.id = MessageId::parse("ai-studio:drive-file~user-0123-0").unwrap();
        selected.session = known.clone();
        assert_eq!(r.messages(&known, false).unwrap(), [selected]);
        assert!(r.messages(&copy, false).unwrap().is_empty());
        assert_eq!(
            r.query("SELECT count(*) AS n FROM block", &[]).unwrap()[0]["n"],
            1
        );
        let index = r.item_index(&known).unwrap().unwrap();
        assert_eq!(
            index.entries,
            crate::items::computed_entries(r.conn().unwrap(), index.session_pk).unwrap()
        );
    }
    // A second explicit owner releases the copy. Its missing target is
    // installed before the final selected-message refresh in the same step.
    w.store_reading(declare("ambiguous", &ambiguous), Some(&SOURCE_RECORD))
        .unwrap();
    assert!(w.writer.flush().unwrap().errors.is_empty());
    let pool = ReadPool::open(&w.archive).unwrap();
    let r = pool.reader().unwrap();
    assert_eq!(r.messages(&copy, false).unwrap(), [message]);
    assert!(r.messages(&known, false).unwrap().is_empty());
    assert_eq!(
        r.query("SELECT count(*) AS n FROM block", &[]).unwrap()[0]["n"],
        1
    );
    assert_eq!(
        r.query("SELECT count(*) AS n FROM pragma_foreign_key_check", &[])
            .unwrap()[0]["n"],
        0
    );
    let index = r.item_index(&copy).unwrap().unwrap();
    assert_eq!(
        index.entries,
        crate::items::computed_entries(r.conn().unwrap(), index.session_pk).unwrap()
    );
}
