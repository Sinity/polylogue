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

#[cfg(target_os = "linux")]
#[derive(Debug)]
struct ReadWork {
    text_pages: u64,
    structured_pages: u64,
    text_bytes: u64,
    structured_bytes: u64,
    warm_bytes: u64,
    page_size: u64,
}

#[cfg(target_os = "linux")]
fn logical_read_bytes() -> u64 {
    std::fs::read_to_string("/proc/self/io")
        .unwrap()
        .lines()
        .find_map(|line| line.strip_prefix("rchar:"))
        .unwrap()
        .trim()
        .parse()
        .unwrap()
}

#[cfg(target_os = "linux")]
fn locality(unrelated: usize) -> ReadWork {
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
    let data_bytes: i64 = conn
        .query_row(
            "SELECT length(v.data) FROM block b JOIN message_version_block v ON v.pk=b.payload_pk WHERE b.message_pk=?1",
            [pk],
            |r| r.get(0),
        )
        .unwrap();
    let page_size: i64 = conn
        .pragma_query_value(None, "page_size", |r| r.get(0))
        .unwrap();
    assert!(
        data_bytes > page_size * 8,
        "the structured fixture must span overflow pages: {data_bytes}/{page_size}"
    );
    conn.flush_prepared_statement_cache();
    conn.authorizer(Some(|context: AuthContext<'_>| match context.action {
        AuthAction::Read {
            table_name: "message_version_block",
            column_name: "data",
        } => Authorization::Deny,
        _ => Authorization::Allow,
    }))
    .unwrap();
    // Compare cold pager loads with the same configured cache. The later
    // warm repeat keeps its pages; each cold observation releases them first.
    conn.execute_batch("PRAGMA shrink_memory").unwrap();
    reader.take_page_reads().unwrap();
    let before = logical_read_bytes();
    let unstructured = reader.message_blocks(pk, false).unwrap();
    let text_bytes = logical_read_bytes() - before;
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
    conn.execute_batch("PRAGMA shrink_memory").unwrap();
    reader.take_page_reads().unwrap();
    let before = logical_read_bytes();
    assert_eq!(reader.message_blocks(pk, true).unwrap(), original.blocks);
    let structured_bytes = logical_read_bytes() - before;
    let structured_pages = reader.take_page_reads().unwrap();
    // SQLite's direct overflow reads bypass its page cache. Logical file
    // bytes include those reads even when the OS already caches the file.
    assert!(
        text_bytes < data_bytes as u64 / 2,
        "unstructured read fetched structured bytes: {text_bytes}/{data_bytes}"
    );
    assert!(
        structured_bytes > text_bytes + data_bytes as u64 / 2,
        "structured overflow reads were not discriminated: {text_bytes}/{structured_bytes}, stored data={data_bytes}, page={page_size}"
    );
    let before = logical_read_bytes();
    assert_eq!(reader.message_blocks(pk, true).unwrap(), original.blocks);
    let warm_bytes = logical_read_bytes() - before;
    assert!(
        reader.take_page_reads().unwrap() <= 2,
        "warm repeat should not load additional lookup pages"
    );
    assert_eq!(
        reader.messages(&sid("s"), false).unwrap(),
        std::slice::from_ref(&original)
    );
    assert_eq!(reader.messages(&sid("s"), false).unwrap(), [original]);
    ReadWork {
        text_pages,
        structured_pages,
        text_bytes,
        structured_bytes,
        warm_bytes,
        page_size: page_size as u64,
    }
}

#[test]
#[cfg(target_os = "linux")]
fn canonical_payload_reads_keep_structured_data_separate_and_ignore_unrelated_history() {
    let small = locality(0);
    let large = locality(1024);
    assert!(
        large.text_pages <= small.text_pages + 8,
        "unstructured pages grew with unrelated history: {small:?} -> {large:?}"
    );
    assert!(
        large.structured_pages <= small.structured_pages + 8,
        "structured pages grew with unrelated history: {small:?} -> {large:?}"
    );
    for (before, after) in [
        (small.text_bytes, large.text_bytes),
        (small.structured_bytes, large.structured_bytes),
        (small.warm_bytes, large.warm_bytes),
    ] {
        assert!(
            after <= before + 8 * small.page_size,
            "logical reads grew with unrelated history: {small:?} -> {large:?}"
        );
    }
    println!("canonical payload read work small={small:?} unrelated1024={large:?}");
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
