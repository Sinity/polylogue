use super::*;

fn operation(identity: &str) -> Operation {
    Operation {
        identity: identity.into(),
        directory: identity.into(),
        metadata: vec![],
        kind: Kind::Events,
        material: None,
        previous_material: None,
        logical_name: None,
        acquisition: None,
        ordinal: None,
        unresolved: false,
        source: None,
        raw: false,
        raw_generation: None,
        raw_predecessor: None,
        raw_root: false,
        reading_version: None,
        input: None,
        format: None,
        export_keys: vec![],
        stored_bytes: 0,
    }
}

#[test]
fn current_capture_is_ready_before_an_outage_backlog_finishes() {
    let home = tempfile::tempdir().unwrap();
    let queue = Queue::open(&home.path().join("queue.db")).unwrap();
    for n in 0..5000 {
        queue
            .prepare(&operation(&format!("backlog-{n}")), None, &[])
            .unwrap();
    }
    queue.prepare(&operation("current"), None, &[]).unwrap();
    let selected = queue.ready(&[], 2).unwrap();
    assert_eq!(
        selected
            .iter()
            .map(|e| e.identity.as_str())
            .collect::<Vec<_>>(),
        ["current", "backlog-0"]
    );
    let occupied = selected.into_iter().map(|e| e.identity).collect::<Vec<_>>();
    let next = queue.ready(&occupied, 2).unwrap();
    assert!(next.iter().all(|e| !occupied.contains(&e.identity)));
}

#[test]
fn refusal_blocks_its_dependency_but_not_independent_capture() {
    let home = tempfile::tempdir().unwrap();
    let queue = Queue::open(&home.path().join("queue.db")).unwrap();
    let mut first = operation("first");
    first.kind = Kind::Material;
    first.material = Some("material-1".into());
    queue.prepare(&first, None, &[]).unwrap();
    let mut child = operation("child");
    child.previous_material = first.material;
    queue.prepare(&child, None, &[]).unwrap();
    queue
        .acknowledge("first", &Phase::Refused, &[], 10)
        .unwrap();
    queue.prepare(&operation("independent"), None, &[]).unwrap();
    let ready = queue.ready(&[], 10).unwrap();
    assert_eq!(
        ready
            .iter()
            .map(|e| e.identity.as_str())
            .collect::<Vec<_>>(),
        ["independent"]
    );
}

#[test]
fn rejected_group_keeps_members_available_for_a_valid_group() {
    let home = tempfile::tempdir().unwrap();
    let queue = Queue::open(&home.path().join("queue.db")).unwrap();
    queue.prepare(&operation("member"), None, &[]).unwrap();
    assert!(
        queue
            .prepare(
                &operation("invalid"),
                Some("invalid-hash"),
                &["member".into(), "missing".into()]
            )
            .is_err()
    );
    assert!(queue.get("invalid").unwrap().is_none());
    queue
        .prepare(&operation("valid"), Some("valid-hash"), &["member".into()])
        .unwrap();
    assert_eq!(
        queue
            .lookup(Lookup::Members("valid"), None, None, Order::OldestFirst, 10)
            .unwrap()
            .len(),
        1
    );
    queue
        .acknowledge(
            "member",
            &Phase::Acknowledged {
                commit_seq: Some(1),
            },
            &["event-1".into()],
            10,
        )
        .unwrap();
    assert_eq!(queue.ready(&[], 10).unwrap()[0].identity, "valid");
}

#[test]
fn backup_snapshot_keeps_its_queue_boundary_while_preparation_continues() {
    let home = tempfile::tempdir().unwrap();
    let queue = Queue::open(&home.path().join("queue.db")).unwrap();
    queue.prepare(&operation("before"), None, &[]).unwrap();
    let snapshot = queue.snapshot().unwrap();
    queue.prepare(&operation("after"), None, &[]).unwrap();
    let backup = home.path().join("backup.db");
    snapshot.write_to(&backup).unwrap();
    let restored = Queue::open(&backup).unwrap();
    assert!(restored.get("before").unwrap().is_some());
    assert!(restored.get("after").unwrap().is_none());
    assert!(queue.get("after").unwrap().is_some());
}

#[test]
fn confirmed_candidates_keep_event_identity_and_respect_applied_position() {
    let home = tempfile::tempdir().unwrap();
    let queue = Queue::open(&home.path().join("queue.db")).unwrap();
    for name in ["applied", "later", "unpositioned", "pending"] {
        queue.prepare(&operation(name), None, &[]).unwrap();
    }
    queue
        .acknowledge(
            "applied",
            &Phase::Acknowledged {
                commit_seq: Some(10),
            },
            &["event-a".into()],
            1,
        )
        .unwrap();
    queue
        .acknowledge(
            "later",
            &Phase::Acknowledged {
                commit_seq: Some(11),
            },
            &["event-b".into()],
            1,
        )
        .unwrap();
    queue
        .acknowledge(
            "unpositioned",
            &Phase::Acknowledged { commit_seq: None },
            &[],
            1,
        )
        .unwrap();
    let candidates = queue.confirmed_after(10, false, None, 10).unwrap();
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].entry.identity, "applied");
    assert_eq!(candidates[0].admitted, ["event-a"]);
    let candidates = queue
        .confirmed_after(10, true, Some(candidates[0].entry.order), 10)
        .unwrap();
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].entry.identity, "unpositioned");
    assert!(queue.confirmed_after(-1, false, None, 10).is_err());
}
