use super::*;
use crate::parse::run;
use serde_json::json;

#[test]
fn names_its_databases_and_their_exports() {
    assert!(is_database(Path::new("/h/.codex/state_5.sqlite")));
    assert!(is_database(Path::new("/h/.codex/queue_1.sqlite")));
    assert!(!is_database(Path::new("/h/.codex/thread_history_1.sqlite")));
    assert!(!is_database(Path::new("/h/.codex/logs_2.sqlite")));
    assert!(!is_database(Path::new("/h/.codex/state_5.sqlite-wal")));
    assert!(recognize(
        Path::new("/a/exports/codex-database/goals_1.sqlite.jsonl"),
        b"{"
    ));
    assert!(!recognize(
        Path::new("/a/exports/codex-database/logs_2.sqlite.jsonl"),
        b"{"
    ));
}

#[test]
fn rows_become_sessions_links_and_events() {
    let lines = [
        json!({"db": "state_5.sqlite", "at_ms": 1, "schema": {"threads": "CREATE TABLE threads (...)"}}),
        json!({"t": "threads", "k": ["019a0000-0000-7000-8000-000000000001"], "h": "00",
               "r": {"id": "019a0000-0000-7000-8000-000000000001", "name": "Fix the build", "cwd": "/w",
                     "thread_source": "user", "source": "cli", "created_at": 1_772_359_200,
                     "created_at_ms": 1_772_359_200_123_i64, "cli_version": "0.160.0", "git_branch": null}}),
        json!({"t": "thread_spawn_edges", "k": ["019a0000-0000-7000-8000-000000000002"], "h": "01",
               "r": {"parent_thread_id": "019a0000-0000-7000-8000-000000000001",
                     "child_thread_id": "019a0000-0000-7000-8000-000000000002", "status": "closed"}}),
        json!({"t": "thread_goals", "k": ["019a0000-0000-7000-8000-000000000001"], "h": "02",
               "r": {"thread_id": "019a0000-0000-7000-8000-000000000001", "objective": "ship it",
                     "status": "complete", "updated_at_ms": 1_772_359_300_000_i64}}),
        json!({"t": "thread_items", "k": ["x"], "h": "03", "r": {"item_json": "{}"}}),
        json!({"t": "threads", "k": ["gone-one"], "gone": true}),
    ];
    let mut bytes = Vec::new();
    for l in &lines {
        bytes.extend(serde_json::to_vec(l).unwrap());
        bytes.push(b'\n');
    }
    let step = run(&Database, &mut NoState::default(), &bytes, 0).unwrap();
    let s = &step.output.sessions[0];
    assert_eq!(s.title.as_ref().unwrap().text, "Fix the build");
    assert_eq!(s.started.unwrap().unix_ms(), 1_772_359_200_123);
    assert_eq!(step.output.links[0].evidence, "thread_spawn_edges");
    assert_eq!(step.output.events[0].kind, "goal");
    assert_eq!(step.counts.skipped.values().sum::<u64>(), 2);
    assert_eq!(step.counts.used.values().sum::<u64>(), 4);
}

#[test]
fn unknown_database_table_has_explicit_schema_identity_without_row_values() {
    let step = crate::parse::run(
        &Database,
        &mut NoState::default(),
        b"{\"t\":\"future_facts\",\"k\":1,\"r\":{\"body\":\"distinctive-private-body\"}}\n",
        0,
    )
    .unwrap();
    let key = serde_json::to_string(&("future_facts", "")).unwrap();
    let shape = &step.counts.wire_shapes[&key];
    assert_eq!(shape.table.as_deref(), Some("future_facts"));
    assert_eq!(shape.unknown_table_count, 1);
    assert_eq!(step.counts.unread_fields["future_facts.body"], 1);
    assert!(
        !serde_json::to_string(&step.counts.wire_shapes)
            .unwrap()
            .contains("distinctive-private-body")
    );
}
