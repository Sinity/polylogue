use super::*;

fn note(conn: &Connection, id: &str, refs: &[&str]) {
    conn.execute("INSERT INTO note(id,root,kind,value,author,machine,created_at,commit_seq) VALUES (?1,?1,'note','synthetic note','operator','test',1,1)", [id]).unwrap();
    for reference in refs {
        conn.execute("INSERT INTO note_target(note_id,ref,seen_hash,anchor_input,anchor_chain,anchor_start,anchor_end) VALUES (?1,?2,x'010203','input:opaque',7,10,20)",params![id,reference]).unwrap();
    }
    conn.execute(
        "INSERT INTO note_relation VALUES (?1,'derived_from','session:kept')",
        [id],
    )
    .unwrap();
}

#[test]
fn selected_notes_are_removed_or_trimmed_without_losing_surviving_anchors() {
    let mut conn = Connection::open_in_memory().unwrap();
    crate::schema::create_archive(&conn).unwrap();
    let session = "claude-code:gone";
    let opaque = "codex:opaque:\0#";
    let sessions = [session.into(), opaque.into()];
    let mut messages = SelectedMessages::default();
    messages.insert("message:raw").unwrap();
    messages.insert("opaque:id").unwrap();
    let tx = conn.transaction().unwrap();
    for n in 0..10_000 {
        note(&tx, &format!("unrelated-{n}"), &["message:other"]);
    }
    note(&tx, "whole-session", &["session:claude-code:gone"]);
    note(&tx, "whole-message", &["message:opaque:id"]);
    note(&tx, "whole-tool", &["tool:claude-code:gone#call:a"]);
    note(&tx, "whole-opaque", &[&format!("tool:{opaque}#call")]);
    note(
        &tx,
        "partial",
        &["session:claude-code:gone", "session:kept"],
    );
    note(
        &tx,
        "neighbors",
        &[
            "session:claude-code:gone-too",
            "tool:claude-code:gone-too#call",
            "message:opaque:id:other",
        ],
    );
    note(&tx, "raw", &["claude-code:gone"]);
    note(&tx, "empty", &[]);
    assert_eq!(remove(&tx, &sessions, &messages, true).unwrap(), 4);
    let count = |table: &str| -> i64 {
        tx.query_row(&format!("SELECT count(*) FROM {table}"), [], |r| r.get(0))
            .unwrap()
    };
    assert_eq!(count("note"), 10_008);
    assert_eq!(remove(&tx, &sessions, &messages, false).unwrap(), 4);
    assert_eq!(count("note"), 10_004);
    assert_eq!(count("note_relation"), 10_004);
    let kept: (String,Vec<u8>,String,i64,i64,i64) = tx.query_row("SELECT ref,seen_hash,anchor_input,anchor_chain,anchor_start,anchor_end FROM note_target WHERE note_id='partial'",[],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?,r.get(5)?))).unwrap();
    assert_eq!(
        kept,
        (
            "session:kept".into(),
            vec![1, 2, 3],
            "input:opaque".into(),
            7,
            10,
            20
        )
    );
    assert_eq!(remove(&tx, &sessions, &messages, false).unwrap(), 0);
}

#[test]
fn matching_targets_are_disk_backed_and_callback_or_lookup_errors_propagate() {
    let mut targets = Targets::default();
    assert!(targets.0.is_none());
    for n in 0..100_000 {
        targets.insert("one-note", &format!("message:{n}")).unwrap();
    }
    let conn = targets.0.as_ref().unwrap();
    let bytes: i64 = conn
        .query_row(
            "SELECT page_count*page_size FROM pragma_page_count(),pragma_page_size()",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert!(bytes > 1 << 20);
    assert_eq!(conn.path(), Some(""));
    let mut total = 0;
    targets
        .notes(|id, n| {
            assert_eq!(id, "one-note");
            total += n;
            Ok(())
        })
        .unwrap();
    assert_eq!(total, 100_000);
    let mut refs = 0;
    targets
        .refs("one-note", |_| {
            refs += 1;
            Ok(())
        })
        .unwrap();
    assert_eq!(refs, 100_000);
    assert!(
        targets
            .notes(|_, _| Err(StoreError::Invalid("stop".into())))
            .is_err()
    );
    targets
        .0
        .as_ref()
        .unwrap()
        .execute("DROP TABLE target", [])
        .unwrap();
    assert!(targets.notes(|_, _| Ok(())).is_err());
    assert!(targets.refs("one-note", |_| Ok(())).is_err());
    drop(targets);
}

#[cfg(feature = "faults")]
#[test]
fn a_late_staging_read_error_rolls_back_note_and_target_deletions() {
    use crate::fault::{self, Action};
    for (point, skip) in [
        ("excision_targets.insert", 0),
        ("excision_targets.read", 1),
        ("excision_messages.contains", 1),
    ] {
        fault::clear();
        let mut conn = Connection::open_in_memory().unwrap();
        crate::schema::create_archive(&conn).unwrap();
        note(&conn, "one", &["message:gone"]);
        note(&conn, "two", &["message:gone"]);
        let mut messages = SelectedMessages::default();
        messages.insert("gone").unwrap();
        let tx = conn.transaction().unwrap();
        fault::arm(point, "", Action::Fail, skip, 1);
        let error = remove(&tx, &[], &messages, false).unwrap_err();
        assert!(error.to_string().contains(point), "{error}");
        tx.rollback().unwrap();
        assert_eq!(
            conn.query_row("SELECT count(*) FROM note", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            2
        );
        assert_eq!(
            conn.query_row("SELECT count(*) FROM note_target", [], |r| r
                .get::<_, i64>(0))
                .unwrap(),
            2
        );
        assert_eq!(
            conn.query_row("SELECT count(*) FROM note_relation", [], |r| r
                .get::<_, i64>(0))
                .unwrap(),
            2
        );
        let tx = conn.transaction().unwrap();
        assert_eq!(remove(&tx, &[], &messages, false).unwrap(), 2);
    }
    fault::clear();
}
