use super::*;

#[test]
fn corrupt_stored_session_facts_refuse_without_erasure_and_can_retry_after_repair() {
    for corrupt in ["{", "null", "{\"id\":17}"] {
        let w = world();
        let mut first = batch(
            "input",
            0,
            0,
            vec![row(10, text_message("a", "s", Role::User, "kept", None))],
            &["s"],
        );
        first.parse[0].output.sessions[0].tool_version = Some("previous version".into());
        w.writer.send(first).unwrap();
        assert!(w.writer.flush().unwrap().errors.is_empty());
        let conn = crate::archive::open_write(&w.archive.index_db(), crate::archive::cache::SETUP)
            .unwrap();
        let saved: String = conn
            .query_row("SELECT facts FROM session_record", [], |r| r.get(0))
            .unwrap();
        conn.execute("UPDATE session_record SET facts=?1", [corrupt])
            .unwrap();
        let next = batch(
            "input",
            0,
            100,
            vec![row(130, text_message("b", "s", Role::User, "new", None))],
            &["s"],
        );
        w.writer.send(next.clone()).unwrap();
        let report = w.writer.flush().unwrap();
        assert!(
            !report.errors.is_empty(),
            "malformed stored facts must refuse, not become absent: {corrupt}"
        );
        assert_eq!(
            conn.query_row("SELECT facts FROM session_record", [], |r| r
                .get::<_, String>(0))
                .unwrap(),
            corrupt
        );
        assert_eq!(texts(&w, "s"), ["kept"]);
        conn.execute("UPDATE session_record SET facts=?1", [saved])
            .unwrap();
        w.writer.send(next).unwrap();
        assert!(w.writer.flush().unwrap().errors.is_empty());
        assert_eq!(texts(&w, "s"), ["kept", "new"]);
        let reads = ReadPool::open(&w.archive).unwrap();
        assert_eq!(
            reads
                .reader()
                .unwrap()
                .query("SELECT harness_version FROM session", &[])
                .unwrap()[0]["harness_version"],
            "previous version"
        );
    }
}
