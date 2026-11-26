//! Messages named by their content are shown only from their input's latest
//! chain; messages named by native ID stay shown from an older chain.
use super::*;

fn content_row(offset: u64, id: &str, role: Role, text: &str) -> MessageRow {
    MessageRow {
        page_text: false,
        id_kind: polylogue_model::IdKind::Content,
        ..row(offset, text_message(id, "s", role, text, Some("end_turn")))
    }
}

fn chunks(rows: &[(&str, &str)]) -> Vec<MessageRow> {
    rows.iter()
        .enumerate()
        .map(|(i, (id, text))| {
            let role = if i % 2 == 0 {
                Role::User
            } else {
                Role::Assistant
            };
            content_row(i as u64 * 10, id, role, text)
        })
        .collect()
}

fn store(w: &TestArchive, rows: Vec<MessageRow>, rewritten: bool) {
    let mut b = batch("prompt", Route::Live, 0, rows, &["s"]);
    // A rewrite's bytes differ from the first version's from its first
    // byte: bytes alike would read alike (docs/archive.md, readings).
    let bytes = if rewritten {
        b.change = Change::Rewritten;
        [b'w'; 100]
    } else {
        SOURCE_RECORD
    };
    w.store_reading(b, Some(&bytes)).unwrap();
    assert!(w.writer.flush().unwrap().errors.is_empty());
}

fn shown(w: &TestArchive) -> Vec<String> {
    let pool = ReadPool::open(&w.archive).unwrap();
    pool.reader()
        .unwrap()
        .messages(&sid("s"), false)
        .unwrap()
        .iter()
        .map(|m| m.text())
        .collect()
}

#[test]
fn ai_studio_deleted_chunk_reads_a_c_d() {
    let w = test_archive();
    store(
        &w,
        chunks(&[("a", "A"), ("b", "B"), ("c", "C"), ("d", "D")]),
        false,
    );
    assert_eq!(shown(&w), ["A", "B", "C", "D"]);
    // The prompt is saved again without chunk B; the other chunks keep
    // their content IDs and move to new places.
    let mut rewrite = chunks(&[("a", "A"), ("c", "C"), ("d", "D")]);
    rewrite[1].message.role = Role::User;
    rewrite[2].message.role = Role::Assistant;
    store(&w, rewrite, true);
    assert_eq!(shown(&w), ["A", "C", "D"]);
    let pool = ReadPool::open(&w.archive).unwrap();
    let rows = pool
        .reader()
        .unwrap()
        .query(
            "SELECT count(*) AS n FROM message WHERE id='claude-code:b'",
            &[],
        )
        .unwrap();
    assert_eq!(rows[0]["n"], 0);
}

#[test]
fn rewrite_keeps_native_id_messages_it_no_longer_holds() {
    let w = test_archive();
    let with_native_ids = |rows: Vec<MessageRow>| {
        rows.into_iter()
            .map(|r| MessageRow {
                page_text: false,
                id_kind: polylogue_model::IdKind::Native,
                ..r
            })
            .collect::<Vec<_>>()
    };
    store(
        &w,
        with_native_ids(chunks(&[("a", "A"), ("b", "B")])),
        false,
    );
    store(&w, with_native_ids(chunks(&[("a", "A")])), true);
    assert_eq!(shown(&w), ["A", "B"]);
}

#[test]
fn a_copy_in_another_input_keeps_a_content_named_message() {
    let w = test_archive();
    store(&w, chunks(&[("a", "A"), ("b", "B")]), false);
    let mut copy = batch(
        "export",
        Route::Capture,
        0,
        chunks(&[("a", "A"), ("b", "B")]),
        &["s"],
    );
    copy.parse[0].parse_version = "synthetic=1".into();
    w.store_reading(copy, Some(&SOURCE_RECORD)).unwrap();
    assert!(w.writer.flush().unwrap().errors.is_empty());
    store(&w, chunks(&[("a", "A")]), true);
    assert_eq!(shown(&w), ["A", "B"]);
}

#[test]
fn revision_touches_only_changed_messages() {
    let w = test_archive();
    store(
        &w,
        chunks(&[("a", "A"), ("b", "first draft"), ("c", "C")]),
        false,
    );
    let blocks = |w: &TestArchive| -> Vec<(String, i64)> {
        let pool = ReadPool::open(&w.archive).unwrap();
        pool.reader()
            .unwrap()
            .query(
                "SELECT m.id AS id, b.pk AS pk FROM block b JOIN message m ON m.pk=b.message_pk ORDER BY m.id",
                &[],
            )
            .unwrap()
            .into_iter()
            .map(|row| {
                (
                    row["id"].as_str().unwrap().to_owned(),
                    row["pk"].as_i64().unwrap(),
                )
            })
            .collect()
    };
    let before = blocks(&w);
    // The edited chunk has new content and so a new ID; A and C are the
    // same messages, read again from the new chain.
    store(
        &w,
        chunks(&[("a", "A"), ("b2", "second draft"), ("c", "C")]),
        true,
    );
    assert_eq!(shown(&w), ["A", "second draft", "C"]);
    let after = blocks(&w);
    let pk =
        |rows: &[(String, i64)], id: &str| rows.iter().find(|(m, _)| m == id).map(|(_, pk)| *pk);
    assert_eq!(pk(&after, "claude-code:a"), pk(&before, "claude-code:a"));
    assert_eq!(pk(&after, "claude-code:c"), pk(&before, "claude-code:c"));
    assert_eq!(pk(&after, "claude-code:b"), None);
    assert!(pk(&after, "claude-code:b2").is_some());
}
