use super::*;
use crate::fixture::writer::{WriterArchive, message, sid};
use polylogue_model::{Block, Role};
use rusqlite::hooks::{AuthAction, AuthContext, Authorization};

fn structured() -> String {
    use std::fmt::Write;
    let mut value = String::new();
    for n in 0..32768_u64 {
        write!(&mut value, "{:016x}", n.wrapping_mul(0x9e3779b97f4a7c15)).unwrap();
    }
    value
}

fn block() -> Block {
    serde_json::from_value(serde_json::json!({"type":"tool_result","call_id":null,
        "outcome":{"kind":"ok"},"output":"exact visible output","data":{"opaque":structured()}}))
    .unwrap()
}

fn locality(unrelated: usize) -> (u64, u64) {
    let mut w = WriterArchive::new();
    let original = message("one", "s", Role::Assistant, 1000, vec![block()]);
    w.write("s", vec![original.clone()]);
    if unrelated != 0 {
        w.write(
            "unrelated",
            (0..unrelated)
                .map(|n| {
                    message(
                        &format!("m{n}"),
                        "unrelated",
                        Role::User,
                        2000 + n as i64,
                        vec![Block::Text {
                            text: format!("unrelated content {n}"),
                        }],
                    )
                })
                .collect(),
        );
    }
    w.writer.stop();
    let pool = ReadPool::open(&w.archive).unwrap();
    let reader = pool.reader().unwrap();
    let conn = reader.conn().unwrap();
    let pk: i64 = conn
        .query_row(
            "SELECT pk FROM message WHERE id=?1",
            [original.id.to_string()],
            |r| r.get(0),
        )
        .unwrap();
    conn.flush_prepared_statement_cache();
    conn.authorizer(Some(|context: AuthContext<'_>| match context.action {
        AuthAction::Read {
            table_name: "message_version_block",
            column_name: "data",
        } => Authorization::Deny,
        _ => Authorization::Allow,
    }))
    .unwrap();
    reader.take_page_reads().unwrap();
    let unstructured = reader.message_blocks(pk, false).unwrap();
    let text_pages = reader.take_page_reads().unwrap();
    let mut expected = original.blocks.clone();
    let Block::ToolResult { data, .. } = &mut expected[0] else {
        unreachable!()
    };
    *data = None;
    assert_eq!(unstructured, expected);
    assert_eq!(reader.message_blocks(pk, false).unwrap(), expected);
    conn.authorizer(None::<fn(AuthContext<'_>) -> Authorization>)
        .unwrap();
    conn.flush_prepared_statement_cache();
    reader.take_page_reads().unwrap();
    assert_eq!(reader.message_blocks(pk, true).unwrap(), original.blocks);
    let structured_pages = reader.take_page_reads().unwrap();
    assert!(
        structured_pages > text_pages + 8,
        "structured overflow pages were not discriminated: {text_pages}/{structured_pages}"
    );
    assert_eq!(reader.message_blocks(pk, true).unwrap(), original.blocks);
    assert!(
        reader.take_page_reads().unwrap() <= 2,
        "warm structured payload should stay local"
    );
    assert_eq!(
        reader.messages(&sid("s"), false).unwrap(),
        [original.clone()]
    );
    assert_eq!(reader.messages(&sid("s"), false).unwrap(), [original]);
    (text_pages, structured_pages)
}

#[test]
fn canonical_payload_reads_keep_structured_data_separate_and_ignore_unrelated_history() {
    let small = locality(0);
    let large = locality(1024);
    assert!(
        large.0 <= small.0 + 8,
        "unstructured pages grew with unrelated history: {small:?} -> {large:?}"
    );
    assert!(
        large.1 <= small.1 + 8,
        "structured pages grew with unrelated history: {small:?} -> {large:?}"
    );
    println!("canonical payload cache misses small={small:?} unrelated1024={large:?}");
}

#[test]
fn paged_structured_reads_do_not_fetch_the_next_canonical_payload() {
    let mut w = WriterArchive::new();
    let first = message("first", "s", Role::Assistant, 1000, vec![block()]);
    let later = message("later", "s", Role::Assistant, 2000, vec![block()]);
    w.write("s", vec![first.clone(), later.clone()]);
    w.writer.stop();
    let conn =
        crate::archive::open_write(&w.archive.index_db(), crate::archive::cache::SETUP).unwrap();
    assert_eq!(
        conn.execute(
            "UPDATE message_version_block SET data=X'00' WHERE pk IN
        (SELECT b.payload_pk FROM block b JOIN message m ON m.pk=b.message_pk WHERE m.id=?1)",
            [later.id.to_string()]
        )
        .unwrap(),
        1
    );
    drop(conn);
    let pool = ReadPool::open(&w.archive).unwrap();
    let reader = pool.reader().unwrap();
    let mut visited = 0;
    reader
        .visit_messages(&sid("s"), false, |actual| {
            assert_eq!(actual, first);
            visited += 1;
            std::ops::ControlFlow::Break(())
        })
        .unwrap();
    assert_eq!(visited, 1);
    assert!(
        reader.messages(&sid("s"), false).is_err(),
        "the full batch must encounter the later invalid structured payload"
    );
}
