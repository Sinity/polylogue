use super::*;

#[test]
fn sorted_json_counts_and_membership_preserve_duplicates_and_opaque_ids() {
    let mut selected = SelectedMessages::default();
    assert_eq!(selected.count().unwrap(), 0);
    assert!(!selected.contains("absent").unwrap());
    assert_eq!(serde_json::to_string(&selected).unwrap(), "[]");
    assert!(selected.0.is_none());
    let mut expected = vec!["z", "a\0", "é", "a:b:c", "a", "a\0", "🦀", ""];
    for id in &expected {
        selected.insert(id).unwrap();
    }
    expected.sort_unstable();
    assert_eq!(selected.count().unwrap(), expected.len() as u64);
    assert_eq!(
        serde_json::to_string(&selected).unwrap(),
        serde_json::to_string(&expected).unwrap()
    );
    for id in &expected {
        assert!(selected.contains(id).unwrap());
    }
    for id in ["a\0z", "a:b", "message:a", "éx"] {
        assert!(!selected.contains(id).unwrap());
    }
}

#[test]
fn large_selection_stays_in_sqlite_and_survives_source_row_deletion() {
    let source = Connection::open_in_memory().unwrap();
    source
        .execute_batch("CREATE TABLE message(id TEXT); BEGIN;")
        .unwrap();
    {
        let mut insert = source.prepare("INSERT INTO message VALUES (?1)").unwrap();
        for n in 0..100_000 {
            insert.execute([format!("opaque:message:{n:08}")]).unwrap();
        }
    }
    source.execute_batch("COMMIT").unwrap();
    let mut selected = SelectedMessages::default();
    {
        let mut stmt = source.prepare("SELECT id FROM message").unwrap();
        for id in stmt.query_map([], |row| row.get::<_, String>(0)).unwrap() {
            selected.insert(&id.unwrap()).unwrap();
        }
    }
    source.execute("DELETE FROM message", []).unwrap();
    assert_eq!(selected.count().unwrap(), 100_000);
    assert!(selected.contains("opaque:message:00099999").unwrap());
    assert!(!selected.contains("opaque:message:00100000").unwrap());
    let db = selected.0.as_ref().unwrap();
    assert_eq!(db.path(), Some(""));
    let bytes: i64 = db
        .query_row(
            "SELECT page_count*page_size FROM pragma_page_count(),pragma_page_size()",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert!(bytes > 1 << 20);
    let plan: String = db
        .query_row(
            "EXPLAIN QUERY PLAN SELECT id FROM selected WHERE id=?1",
            ["opaque:message:00099999"],
            |row| row.get(3),
        )
        .unwrap();
    assert!(plan.contains("selected_id"), "{plan}");
    serde_json::to_writer(std::io::sink(), &selected).unwrap();
    db.execute("DROP TABLE selected", []).unwrap();
    assert!(selected.count().is_err());
    assert!(selected.contains("opaque:message:00099999").is_err());
    assert!(serde_json::to_string(&selected).is_err());
}

#[cfg(feature = "faults")]
#[test]
fn failed_insertion_or_lookup_is_not_an_empty_success() {
    use crate::fault::{self, Action};
    fault::clear();
    let mut selected = SelectedMessages::default();
    fault::arm("excision_messages.insert", "", Action::Fail, 0, 1);
    assert!(selected.insert("gone").is_err());
    assert_eq!(selected.count().unwrap(), 0);
    selected.insert("gone").unwrap();
    fault::arm("excision_messages.contains", "", Action::Fail, 0, 1);
    assert!(selected.contains("gone").is_err());
    assert!(selected.contains("gone").unwrap());
    assert_eq!(selected.count().unwrap(), 1);
    fault::clear();
}
