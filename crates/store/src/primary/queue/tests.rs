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
        completion: None,
        captured: false,
        blob: None,
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
    snapshot
        .write_to(&backup, &std::sync::atomic::AtomicBool::new(false))
        .unwrap();
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

#[test]
fn acquisition_publication_keeps_member_order_and_is_atomic_on_collision() {
    let home = tempfile::tempdir().unwrap();
    let queue = Queue::open(&home.path().join("queue.db")).unwrap();
    queue.prepare(&operation("existing"), None, &[]).unwrap();
    let staged = Queue::open(&home.path().join("staged.db")).unwrap();
    staged.prepare(&operation("first"), None, &[]).unwrap();
    staged.prepare(&operation("second"), None, &[]).unwrap();
    let mut header = operation("acquisition");
    header.kind = Kind::Acquisition;
    queue.publish_acquisition(&staged, &header).unwrap();
    let members = queue
        .lookup(
            Lookup::Acquisition("acquisition"),
            None,
            None,
            Order::OldestFirst,
            10,
        )
        .unwrap();
    assert_eq!(
        members
            .iter()
            .map(|m| m.identity.as_str())
            .collect::<Vec<_>>(),
        ["first", "second"]
    );
    assert_eq!(members[0].directory, "acquisition/first");
    assert!(members[0].order > queue.get("existing").unwrap().unwrap().order);
    let mut colliding = operation("invalid-acquisition");
    colliding.kind = Kind::Acquisition;
    assert!(queue.publish_acquisition(&staged, &colliding).is_err());
    assert!(queue.get("invalid-acquisition").unwrap().is_none());
    assert_eq!(queue.len().unwrap(), 3);
    // An unsuccessful attached transaction leaves the connection usable.
    queue.prepare(&operation("later"), None, &[]).unwrap();
}

#[test]
fn refused_authored_revision_and_its_group_leave_the_stale_base_overlay() {
    let home = tempfile::tempdir().unwrap();
    let queue = Queue::open(&home.path().join("queue.db")).unwrap();
    for identity in ["refused", "grouped", "independent"] {
        let mut operation = operation(identity);
        operation.kind = Kind::Request;
        operation.completion = Some(Completion::Request);
        queue.prepare(&operation, None, &[]).unwrap();
    }
    queue
        .acknowledge("refused", &Phase::Refused, &[], 0)
        .unwrap();
    queue
        .prepare(
            &operation("failed-group"),
            Some("failed-hash"),
            &["grouped".into()],
        )
        .unwrap();
    queue
        .acknowledge("failed-group", &Phase::Refused, &[], 0)
        .unwrap();
    let selected = queue
        .lookup(Lookup::Requests, None, None, Order::OldestFirst, 10)
        .unwrap();
    assert_eq!(
        selected
            .iter()
            .map(|entry| entry.identity.as_str())
            .collect::<Vec<_>>(),
        ["independent"]
    );
    assert!(queue.get("refused").unwrap().is_some());
    assert!(queue.get("grouped").unwrap().is_some());
}

#[test]
fn cancelled_queue_snapshot_never_starts_copying_the_database() {
    let home = tempfile::tempdir().unwrap();
    let queue = Queue::open(&home.path().join("queue.db")).unwrap();
    queue.prepare(&operation("before"), None, &[]).unwrap();
    let destination = home.path().join("cancelled.db");
    assert!(
        queue
            .snapshot()
            .unwrap()
            .write_to(&destination, &std::sync::atomic::AtomicBool::new(true))
            .is_err()
    );
    assert!(!destination.exists());
    assert!(queue.get("before").unwrap().is_some());
}

#[test]
fn one_free_sender_slot_alternates_new_capture_and_old_backlog() {
    let home = tempfile::tempdir().unwrap();
    let queue = Queue::open(&home.path().join("queue.db")).unwrap();
    for identity in ["oldest", "older", "recent"] {
        queue.prepare(&operation(identity), None, &[]).unwrap();
    }
    let first = queue.ready(&[], 1).unwrap().remove(0);
    assert_eq!(first.identity, "recent");
    queue.prepare(&operation("newer"), None, &[]).unwrap();
    let second = queue
        .ready(std::slice::from_ref(&first.identity), 1)
        .unwrap()
        .remove(0);
    assert_eq!(second.identity, "oldest");
    queue.prepare(&operation("newest"), None, &[]).unwrap();
    let third = queue
        .ready(&[first.identity, second.identity], 1)
        .unwrap()
        .remove(0);
    assert_eq!(third.identity, "newest");
}
