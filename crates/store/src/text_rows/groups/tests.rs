use super::*;
use crate::content::GROUP_BYTES;
use crate::text_rows::{remove, resolve};

fn hash(text: &str) -> [u8; 32] {
    *blake3::hash(text.as_bytes()).as_bytes()
}

fn index() -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    crate::schema::create_index(&conn).unwrap();
    conn.pragma_update(None, "foreign_keys", true).unwrap();
    conn
}

fn count(conn: &Connection, table: &str) -> i64 {
    conn.query_row(&format!("SELECT count(*) FROM {table}"), [], |r| r.get(0))
        .unwrap()
}

/// Every prepared text, read back from its group as the writer stores it.
fn read_back(prepared: &PreparedTexts, text: &str) -> String {
    let found = prepared.get(&hash(text)).unwrap();
    slice(
        &found.group.body,
        found.text.start as i64,
        found.text.length as i64,
    )
    .unwrap()
}

#[test]
fn a_step_packs_neighbouring_texts_together_and_a_long_text_alone() {
    let short: Vec<String> = (0..40)
        .map(|n| format!("synthetic reply {n}: the walrus counted the tide"))
        .collect();
    let long = "λ".repeat(GROUP_BYTES);
    let mut texts: Vec<&str> = short.iter().map(String::as_str).collect();
    texts.insert(20, &long);
    texts.push("Łódź café 🦀");
    texts.push(&short[3]);
    let prepared = PreparedTexts::pack(texts.iter().copied()).unwrap();
    // Twenty short texts, the long one alone, then the other twenty and the
    // accented one; the repeated text is prepared once.
    assert_eq!(prepared.groups.len(), 3);
    assert_eq!(prepared.texts.len(), 42);
    let place = |text: &str| prepared.get(&hash(text)).unwrap().text.group;
    assert_eq!(place(&short[0]), place(&short[19]));
    assert_eq!(place(&long), 1);
    assert_eq!(place(&short[20]), place("Łódź café 🦀"));
    for text in texts {
        assert_eq!(read_back(&prepared, text), text);
    }
    let one: usize = short
        .iter()
        .map(|t| zstd::bulk::compress(t.as_bytes(), 3).unwrap().len())
        .sum();
    let grouped = PreparedTexts::pack(short.iter().map(String::as_str)).unwrap();
    assert_eq!(grouped.groups.len(), 1);
    assert!(grouped.groups[0].body.len() * 4 < one);
}

#[test]
fn an_empty_text_is_prepared_in_a_group() {
    for texts in [vec![""], vec!["before", ""], vec!["", "after"]] {
        let prepared = PreparedTexts::pack(texts.iter().copied()).unwrap();
        assert_eq!(prepared.groups.len(), 1);
        for text in texts {
            assert_eq!(read_back(&prepared, text), text);
        }
    }
    let mut prepared = PreparedTexts::pack(["first"]).unwrap();
    prepared.add([""]).unwrap();
    assert_eq!(prepared.groups.len(), 2);
    assert_eq!(read_back(&prepared, ""), "");
}

#[test]
fn added_texts_go_to_new_groups_after_the_prepared_ones() {
    let mut prepared = PreparedTexts::pack(["first", "second"]).unwrap();
    prepared.add(["second", "third"]).unwrap();
    assert_eq!(prepared.groups.len(), 2);
    assert_eq!(prepared.get(&hash("third")).unwrap().text.group, 1);
    assert_eq!(prepared.get(&hash("second")).unwrap().text.group, 0);
    for text in ["first", "second", "third"] {
        assert_eq!(read_back(&prepared, text), text);
    }
}

#[test]
fn prepared_texts_survive_a_staged_step_and_refuse_a_missing_group() {
    let prepared = PreparedTexts::pack(["staged", "Straße"]).unwrap();
    let json = serde_json::to_string(&prepared).unwrap();
    assert_eq!(
        serde_json::from_str::<PreparedTexts>(&json).unwrap(),
        prepared
    );
    let mut stored: serde_json::Value = serde_json::from_str(&json).unwrap();
    stored[1] = serde_json::json!([]);
    assert!(serde_json::from_value::<PreparedTexts>(stored).is_err());
}

#[test]
fn the_writer_stores_a_group_once_and_removes_it_with_its_last_text() {
    let mut conn = index();
    let prepared = PreparedTexts::pack(["kept", "removed", "never new"]).unwrap();
    let tx = conn.transaction().unwrap();
    let kept = resolve(&tx, &hash("kept"), prepared.get(&hash("kept"))).unwrap();
    let removed = resolve(&tx, &hash("removed"), prepared.get(&hash("removed"))).unwrap();
    tx.commit().unwrap();
    assert_eq!(count(&conn, "content_group"), 1);
    assert_eq!(read(&conn, kept).unwrap(), "kept");
    assert_eq!(read(&conn, removed).unwrap(), "removed");

    remove(&conn, removed).unwrap();
    assert_eq!(count(&conn, "content_group"), 1);
    assert_eq!(read(&conn, kept).unwrap(), "kept");
    remove(&conn, kept).unwrap();
    assert_eq!(count(&conn, "content_group"), 0);
    assert_eq!(count(&conn, "text"), 0);
}

#[test]
fn an_equal_group_is_one_row_across_steps_and_a_rolled_back_group_is_stored_again() {
    let mut conn = index();
    let first = PreparedTexts::pack(["shared", "first only"]).unwrap();
    let id = resolve(&conn, &hash("shared"), first.get(&hash("shared"))).unwrap();
    // A later step whose group has the same bytes shares the stored row.
    let same = PreparedTexts::pack(["shared", "first only"]).unwrap();
    let other = resolve(&conn, &hash("first only"), same.get(&hash("first only"))).unwrap();
    assert_eq!(count(&conn, "content_group"), 1);
    assert_eq!(read(&conn, other).unwrap(), "first only");
    let later = PreparedTexts::pack(["shared", "fresh"]).unwrap();
    {
        let tx = conn.transaction().unwrap();
        let again = resolve(&tx, &hash("shared"), later.get(&hash("shared"))).unwrap();
        assert_eq!(again, id);
        assert_eq!(count(&tx, "content_group"), 1);
        resolve(&tx, &hash("fresh"), later.get(&hash("fresh"))).unwrap();
        assert_eq!(count(&tx, "content_group"), 2);
        // Rolled back with its group.
    }
    assert_eq!(count(&conn, "content_group"), 1);
    let fresh = resolve(&conn, &hash("fresh"), later.get(&hash("fresh"))).unwrap();
    assert_eq!(read(&conn, fresh).unwrap(), "fresh");
    assert_eq!(read(&conn, id).unwrap(), "shared");
    assert_eq!(count(&conn, "content_group"), 2);
}

#[test]
fn reads_switching_between_groups_return_each_groups_text() {
    let a = PreparedTexts::pack(["alpha one", "alpha two"]).unwrap();
    let b = PreparedTexts::pack(["beta one"]).unwrap();
    for text in ["alpha one", "beta one", "alpha two", "alpha one"] {
        let prepared = if text.starts_with("alpha") { &a } else { &b };
        assert_eq!(read_back(prepared, text), text);
    }
    let conn = index();
    crate::sql_functions::register(&conn).unwrap();
    let body = &a.get(&hash("alpha two")).unwrap().group.body;
    let start = a.get(&hash("alpha two")).unwrap().text.start as i64;
    let text: String = conn
        .query_row(
            "SELECT pl_text_at(?1, ?2, 9)",
            rusqlite::params![body, start],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(text, "alpha two");
    assert!(slice(body, start, 64).is_err());
    assert!(slice(b"not zstd", 0, 1).is_err());
}

#[test]
fn a_text_outside_its_group_is_an_error_naming_the_column_it_was_read_from() {
    let prepared = PreparedTexts::pack(["alpha"]).unwrap();
    let body = prepared.get(&hash("alpha")).unwrap().group.body.clone();
    let conn = index();
    let error = conn
        .query_row("SELECT 1, 2, ?1, 0, 9999", [body], |r| Ok(joined(r, 2)))
        .unwrap()
        .unwrap_err();
    assert!(
        matches!(
            error,
            StoreError::StoredValue {
                column: Some(2),
                ..
            }
        ),
        "{error}"
    );
}
