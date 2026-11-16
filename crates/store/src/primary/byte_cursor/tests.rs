use super::*;

fn file_state() -> SourceGeneration {
    SourceGeneration::File {
        stream: RawStream {
            source: SourceSelector {
                origin: "synthetic".into(),
                machine: "receiver".into(),
                path_bytes: b"/synthetic/input".to_vec(),
            },
            input: InputKey {
                format: "synthetic".into(),
                id: "input".into(),
                route: polylogue_model::Route::Live,
            },
            role: RawStreamRole::ProviderBytes,
            namespace: None,
        },
        observed: FileSnapshot {
            device: 1,
            inode: 2,
            size: 10,
            modified: FileTime {
                seconds: 3,
                nanos: 4,
            },
            changed: FileTime {
                seconds: 5,
                nanos: 6,
            },
        },
        window: RawRange { start: 0, end: 10 },
    }
}

fn body(state: &SourceGeneration, start: u64, end: u64) -> BodyRange {
    BodyRange {
        generation: state.clone(),
        body: format!("body:{start}:{end}"),
        previous: (start > 0).then(|| format!("body:0:{start}")),
        outcome: Some(CaptureOutcome::Prepared),
        range: RawRange { start, end },
        hash: BlobHash([start as u8; 32]),
        decoded_length: end - start,
        known_total: (end == 10).then_some(10),
    }
}

#[test]
fn holes_and_conflicting_overlaps_do_not_become_maximum_end_cursors() {
    let state = file_state();
    for next in [(6, 10), (4, 10)] {
        let mut check = ByteCursorCheck::new(state.clone()).unwrap();
        check.push(body(&state, 0, 5)).unwrap();
        check.push(body(&state, next.0, next.1)).unwrap();
        assert!(check.finish().is_none());
    }
    let mut check = ByteCursorCheck::new(state.clone()).unwrap();
    check.push(body(&state, 0, 5)).unwrap();
    check.push(body(&state, 5, 10)).unwrap();
    let cursor = check.finish().unwrap();
    assert!(cursor.is_complete_window());
    assert_eq!(cursor.root_body(), "body:0:5");
    assert_eq!(cursor.last_body(), "body:5:10");
}

#[test]
fn tail_window_covers_only_its_back_linked_bytes() {
    let mut state = file_state();
    let SourceGeneration::File { window, .. } = &mut state else {
        unreachable!()
    };
    window.start = 3;
    let mut first = body(&state, 3, 6);
    first.previous = None;
    let mut last = body(&state, 6, 10);
    last.previous = Some(first.body.clone());
    let mut check = ByteCursorCheck::new(state.clone()).unwrap();
    check.push(first.clone()).unwrap();
    let partial = check.finish().unwrap();
    assert_eq!(partial.range(), RawRange { start: 3, end: 6 });
    assert!(!partial.is_complete_window());
    let mut check = ByteCursorCheck::new(state.clone()).unwrap();
    check.push(first.clone()).unwrap();
    check.push(last.clone()).unwrap();
    let complete = check.finish().unwrap();
    assert_eq!(complete.range(), RawRange { start: 3, end: 10 });
    assert!(complete.is_complete_window());
    assert_eq!(complete.root_body(), first.body);
    assert_eq!(complete.last_body(), last.body);
    let mut changed_source = last;
    let SourceGeneration::File { observed, .. } = &mut changed_source.generation else {
        unreachable!()
    };
    observed.changed.nanos += 1;
    let mut check = ByteCursorCheck::new(state).unwrap();
    check.push(first).unwrap();
    check.push(changed_source).unwrap();
    assert!(check.finish().is_none());
}

#[test]
fn equal_file_stamps_do_not_join_different_stream_roles_or_members() {
    let state = file_state();
    for role in [
        RawStreamRole::DecodedMember {
            codec: "zstd".into(),
            member_index: 0,
            path_bytes: b"one".to_vec(),
        },
        RawStreamRole::DecodedMember {
            codec: "zstd".into(),
            member_index: 1,
            path_bytes: b"two".to_vec(),
        },
    ] {
        let mut other = state.clone();
        let SourceGeneration::File { stream, .. } = &mut other else {
            unreachable!()
        };
        stream.role = role;
        let mut check = ByteCursorCheck::new(state.clone()).unwrap();
        check.push(body(&state, 0, 5)).unwrap();
        check.push(body(&other, 5, 10)).unwrap();
        assert!(check.finish().is_none());
    }
}

#[test]
fn database_snapshot_restart_cannot_complete_an_earlier_sampling_window() {
    let stream = RawStream {
        role: RawStreamRole::DatabaseDump,
        ..file_state().stream().clone()
    };
    let state = SourceGeneration::Database {
        stream,
        snapshot: DatabaseSnapshotToken {
            nonce: "00000000-0000-0000-0000-000000000001".into(),
        },
        window: RawRange { start: 0, end: 10 },
    };
    let mut restarted = state.clone();
    let SourceGeneration::Database { snapshot, .. } = &mut restarted else {
        unreachable!()
    };
    snapshot.nonce = "00000000-0000-0000-0000-000000000002".into();
    let mut check = ByteCursorCheck::new(state.clone()).unwrap();
    check.push(body(&state, 0, 5)).unwrap();
    check.push(body(&restarted, 5, 10)).unwrap();
    assert!(check.finish().is_none());
}

#[test]
fn empty_range_needs_a_body_and_repeats_keep_eof() {
    let mut empty = file_state();
    let SourceGeneration::File {
        window, observed, ..
    } = &mut empty
    else {
        unreachable!()
    };
    window.end = 0;
    observed.size = 0;
    assert!(
        ByteCursorCheck::new(empty.clone())
            .unwrap()
            .finish()
            .is_none()
    );
    let mut check = ByteCursorCheck::new(empty.clone()).unwrap();
    let mut empty_body = body(&empty, 0, 0);
    empty_body.known_total = Some(0);
    check.push(empty_body).unwrap();
    assert!(check.finish().unwrap().is_complete_window());

    let state = file_state();
    let mut check = ByteCursorCheck::new(state.clone()).unwrap();
    let mut first = body(&state, 0, 10);
    first.known_total = None;
    check.push(first).unwrap();
    check.push(body(&state, 0, 10)).unwrap();
    assert!(check.finish().unwrap().is_complete_window());
}

#[test]
fn contiguous_ranges_need_the_previous_body() {
    let state = file_state();
    for previous in [None, Some("unrelated-body".into())] {
        let mut check = ByteCursorCheck::new(state.clone()).unwrap();
        check.push(body(&state, 0, 5)).unwrap();
        let mut next = body(&state, 5, 10);
        next.previous = previous;
        check.push(next).unwrap();
        assert!(check.finish().is_none());
    }
}

#[test]
fn provider_window_end_before_opened_file_size_is_not_eof() {
    let mut state = file_state();
    let SourceGeneration::File { observed, .. } = &mut state else {
        unreachable!()
    };
    observed.size = 20;
    let mut check = ByteCursorCheck::new(state.clone()).unwrap();
    check.push(body(&state, 0, 10)).unwrap();
    assert!(check.finish().is_none());
}

#[test]
fn input_route_leaves_stream_and_wire_key_unchanged() {
    let original = file_state();
    let mut imported = original.clone();
    let SourceGeneration::File { stream, .. } = &mut imported else {
        unreachable!()
    };
    stream.input.route = polylogue_model::Route::Export;
    assert_eq!(original, imported);
    assert_eq!(
        serde_json::to_vec(&original).unwrap(),
        serde_json::to_vec(&imported).unwrap()
    );
    let restored: SourceGeneration =
        serde_json::from_slice(&serde_json::to_vec(&imported).unwrap()).unwrap();
    assert_eq!(restored, imported);
}

#[test]
fn cursor_keeps_unresolved_or_unknown_outcome() {
    let state = file_state();
    for outcome in [None, Some(CaptureOutcome::UnresolvedBasisConflict)] {
        let mut check = ByteCursorCheck::new(state.clone()).unwrap();
        let mut first = body(&state, 0, 5);
        first.outcome = outcome;
        check.push(first).unwrap();
        check.push(body(&state, 5, 10)).unwrap();
        let cursor = check.finish().unwrap();
        assert!(cursor.is_complete_window());
        assert_eq!(cursor.outcome(), outcome);
    }
}

#[test]
fn separate_reads_of_equal_bytes_are_ambiguous() {
    let state = file_state();
    let mut check = ByteCursorCheck::new(state.clone()).unwrap();
    let mut old = body(&state, 0, 10);
    old.body = "original-unresolved".into();
    old.outcome = Some(CaptureOutcome::UnresolvedBasisConflict);
    check.push(old).unwrap();
    let mut new = body(&state, 0, 10);
    new.body = "new-observation".into();
    check.push(new).unwrap();
    assert!(check.finish().is_none());
}
