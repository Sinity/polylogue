use super::*;
use rusqlite::trace::{TraceEvent, TraceEventCodes};
use std::cell::Cell;

mod ownership;

#[derive(Clone, Copy, Debug, Default)]
struct Work {
    message_reads: u64,
    text_queries: u64,
    text_rows: u64,
    text_key_queries: u64,
    version_key_queries: u64,
    vm_steps: u64,
}

thread_local! {
    static WORK: Cell<Work> = Cell::new(Work::default());
}

fn family(sql: &str) -> u8 {
    if sql.starts_with("SELECT b.body,p.layout,v.pk") {
        1
    } else if sql.starts_with("SELECT v.idx,g.body,t.start,t.length FROM message_version_text") {
        2
    } else if sql.starts_with("SELECT idx,text_id FROM message_version_text") {
        3
    } else if sql.starts_with("SELECT pk FROM message_version WHERE version_hash") {
        4
    } else {
        0
    }
}

fn trace(event: TraceEvent<'_>) {
    WORK.with(|work| {
        let mut value = work.get();
        match event {
            TraceEvent::Profile(statement, _) => {
                let kind = family(&statement.sql());
                match kind {
                    1 => value.message_reads += 1,
                    2 => value.text_queries += 1,
                    3 => value.text_key_queries += 1,
                    4 => value.version_key_queries += 1,
                    _ => {}
                }
                if kind != 0 {
                    // This fixture disables the statement cache, so each
                    // statement's VM counter belongs to this execution.
                    value.vm_steps +=
                        u64::try_from(statement.get_status(rusqlite::StatementStatus::VmStep))
                            .unwrap();
                }
            }
            TraceEvent::Row(statement) if family(&statement.sql()) == 2 => value.text_rows += 1,
            _ => {}
        }
        work.set(value);
    });
}

fn database() -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    crate::schema::create_index(&conn).unwrap();
    conn.pragma_update(None, "foreign_keys", true).unwrap();
    conn.set_prepared_statement_cache_capacity(0);
    crate::versions::track_unreferenced(&conn).unwrap();
    conn
}

fn reading(input: &str, messages: Vec<Message>) -> Batch {
    batch(
        input,
        Route::Live,
        0,
        messages
            .into_iter()
            .enumerate()
            .map(|(i, m)| row(i as u64, m))
            .collect(),
        &["s"],
    )
}

fn apply(conn: &mut Connection, mut b: Batch) -> (Result<u64>, Work) {
    resolve::prepare_index(conn, &mut b, 1, &resolve::IndexUpkeep::PerTransaction).unwrap();
    apply_prepared(conn, b)
}

fn apply_prepared(conn: &mut Connection, b: Batch) -> (Result<u64>, Work) {
    apply_with_upkeep(conn, b, &resolve::IndexUpkeep::PerTransaction)
}

fn apply_with_upkeep(
    conn: &mut Connection,
    b: Batch,
    upkeep: &resolve::IndexUpkeep,
) -> (Result<u64>, Work) {
    let texts = crate::text_rows::TextIds::new(
        conn,
        upkeep.at_build_end(),
        crate::archive::MemoryBudget::default().keys() / 3,
    )
    .unwrap();
    apply_with_texts(conn, b, upkeep, &texts)
}

fn apply_with_texts(
    conn: &mut Connection,
    b: Batch,
    upkeep: &resolve::IndexUpkeep,
    texts: &crate::text_rows::TextIds,
) -> (Result<u64>, Work) {
    let text_transaction = texts.transaction();
    let tx = conn.transaction().unwrap();
    let changes = crate::change_summary::ChangeSummary::start(&tx).unwrap();
    WORK.with(|w| w.set(Work::default()));
    tx.trace_v2(
        TraceEventCodes::SQLITE_TRACE_PROFILE | TraceEventCodes::SQLITE_TRACE_ROW,
        Some(trace),
    );
    let p = &b.parse[0];
    let result = write_rows(
        &tx,
        &b,
        p,
        rows::ReadingWrite {
            texts,
            resolution: p.resolution.as_ref().unwrap(),
            counter: 1,
            upkeep,
            source_commit: Some(&fixture_commit()),
            changes: &changes,
            retraction: false,
            replacement: false,
        },
    );
    tx.trace_v2(TraceEventCodes::empty(), None);
    let work = WORK.with(Cell::get);
    if result.is_ok() {
        crate::items::refresh(&tx, &changes).unwrap();
        tx.commit().unwrap();
        text_transaction.commit();
    } else {
        tx.rollback().unwrap();
    }
    (result, work)
}
