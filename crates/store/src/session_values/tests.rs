use crate::fixture::writer::{WriterArchive, call, prose};
use polylogue_model::Message;
use serde_json::{Value as Json, json};

/// A call message naming `model`, with its result.
fn modelled(session: &str, n: u32, tool: &str, model: &str) -> Vec<Message> {
    let mut messages = call(
        session,
        n,
        100 + i64::from(n) * 10,
        tool,
        None,
        Json::Null,
        Ok("done"),
    );
    messages[0].model = Some(model.into());
    messages
}

/// The stored values and first prompt of every session, and the same
/// computed from all the session's rows.
fn stored_and_recounted(archive: &WriterArchive) -> (Vec<Json>, Vec<Json>) {
    let stored = archive.rows(
        "SELECT s.id, v.field, v.value, v.n FROM session_value v JOIN session s ON s.pk = v.session_pk
         WHERE v.n > 0
         UNION ALL SELECT s.id, 'first prompt', f.message_pk, NULL FROM session_first_prompt f
         JOIN session s ON s.pk = f.session_pk
         ORDER BY 1, 2, 3",
    );
    let recounted = archive.rows(
        "SELECT s.id, 'model' AS field, m.model AS value, count(*) AS n FROM message m JOIN session s ON s.pk = m.session_pk
         WHERE m.model IS NOT NULL AND m.model <> '' GROUP BY s.id, m.model
         UNION ALL SELECT s.id, 'tool', t.tool, count(*) FROM tool_call t JOIN session s ON s.pk = t.session_pk
         GROUP BY s.id, t.tool
         UNION ALL SELECT s.id, 'first prompt', (SELECT m.pk FROM message m WHERE m.session_pk = s.pk
           AND m.written_by = 'human' AND m.prompt_kind = 'prose'
           AND EXISTS (SELECT 1 FROM block b JOIN text t ON t.id = b.text_id WHERE b.message_pk = m.pk AND b.kind = 'text')
           ORDER BY m.session_order, m.id LIMIT 1), NULL FROM session s
         ORDER BY 1, 2, 3",
    );
    (stored, recounted)
}

#[test]
fn stored_values_equal_a_recount_after_each_write() {
    let mut archive = WriterArchive::new();
    let check = |archive: &WriterArchive, step: &str| {
        let (stored, recounted) = stored_and_recounted(archive);
        assert_eq!(stored, recounted, "after {step}");
    };
    // A session that starts with calls has no typed prompt until one comes.
    archive.write("s", modelled("s", 1, "Read", "model-a"));
    check(&archive, "a first call");
    let stored = archive.rows("SELECT message_pk FROM session_first_prompt");
    assert_eq!(stored, [json!({"message_pk": null})]);
    archive.write("s", modelled("s", 2, "Bash", "model-b"));
    archive.write("s", prose("s", 3, 500, "a typed prompt"));
    check(&archive, "appends");
    archive.write("s", modelled("s", 4, "Read", "model-a"));
    archive.write("t", prose("t", 1, 100, "another session"));
    check(&archive, "a second session");
    // A rewrite of a stored message takes away what it contributed.
    let mut rewritten = modelled("s", 2, "Grep", "model-c");
    rewritten.extend(prose("s", 0, 50, "an earlier prompt"));
    archive.write_snapshot("s", rewritten);
    check(&archive, "a rewrite");
    let gone = archive.rows(
        "SELECT v.value, v.n FROM session_value v WHERE v.value IN ('Bash', 'model-b') ORDER BY 1",
    );
    assert_eq!(
        gone,
        [
            json!({"value": "Bash", "n": 0}),
            json!({"value": "model-b", "n": 0})
        ],
        "a value the session no longer has keeps its row at 0"
    );
}

#[test]
fn resumed_first_build_recounts_completion_before_install() {
    let mut archive = WriterArchive::new();
    archive.write_underived("s", modelled("s", 1, "Read", "model-a"));
    {
        let index = crate::archive::open_write(
            &archive.archive.index_db(),
            crate::archive::cache::INDEX_WRITER,
        )
        .unwrap();
        let count = |table: &str| {
            index
                .query_row(&format!("SELECT count(*) FROM {table}"), [], |row| {
                    row.get::<_, u64>(0)
                })
                .unwrap()
        };
        assert_eq!(count("session_value"), 0);
        assert_eq!(count("session_first_prompt"), 0);

        // Simulate a retained build that already contains an earlier writer's
        // materialized values. The next append must leave these rows alone.
        let session = index
            .query_row("SELECT pk FROM session", [], |row| row.get(0))
            .unwrap();
        super::rebuild(&index, session, super::write_counter(&index).unwrap()).unwrap();
    }
    archive.write_underived("s", modelled("s", 2, "Read", "model-a"));
    archive.write_underived("s", prose("s", 3, 500, "a typed prompt"));
    archive.write_underived("t", prose("t", 1, 100, "another session"));
    crate::archive::index_build::keep_progress(&archive.archive).unwrap();
    archive.restart();
    assert!(archive.archive.index_build().is_some());
    archive.writer.derive().unwrap();
    assert!(archive.archive.index_build().is_none());
    let (stored, recounted) = stored_and_recounted(&archive);
    assert_eq!(stored, recounted);

    archive.write("s", modelled("s", 3, "Bash", "model-b"));
    let (stored, recounted) = stored_and_recounted(&archive);
    assert_eq!(stored, recounted);
}
