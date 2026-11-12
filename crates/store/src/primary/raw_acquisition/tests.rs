use super::*;

fn generation() -> SourceGeneration {
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

fn body(generation: &SourceGeneration, start: u64, end: u64) -> RawBodyRange {
    RawBodyRange {
        generation: generation.clone(),
        body: format!("body:{start}:{end}"),
        predecessor: (start > 0).then(|| format!("body:0:{start}")),
        disposition: Some(CaptureDisposition::Prepared),
        range: RawRange { start, end },
        hash: BlobHash([start as u8; 32]),
        decoded_length: end - start,
        known_total: (end == 10).then_some(10),
    }
}

#[test]
fn holes_and_conflicting_overlaps_do_not_become_maximum_end_cursors() {
    let generation = generation();
    for next in [(6, 10), (4, 10)] {
        let mut proof = RawAcquisitionProof::new(generation.clone()).unwrap();
        proof.push(body(&generation, 0, 5)).unwrap();
        proof.push(body(&generation, next.0, next.1)).unwrap();
        assert!(proof.finish().is_none());
    }
    let mut proof = RawAcquisitionProof::new(generation.clone()).unwrap();
    proof.push(body(&generation, 0, 5)).unwrap();
    proof.push(body(&generation, 5, 10)).unwrap();
    let cursor = proof.finish().unwrap();
    assert!(cursor.is_complete_window());
    assert_eq!(cursor.root_body(), "body:0:5");
    assert_eq!(cursor.last_body(), "body:5:10");
}

#[test]
fn a_tail_window_proves_only_its_own_predecessor_linked_bytes_and_eof() {
    let mut generation = generation();
    let SourceGeneration::File { window, .. } = &mut generation else {
        unreachable!()
    };
    window.start = 3;
    let mut first = body(&generation, 3, 6);
    first.predecessor = None;
    let mut last = body(&generation, 6, 10);
    last.predecessor = Some(first.body.clone());
    let mut proof = RawAcquisitionProof::new(generation.clone()).unwrap();
    proof.push(first.clone()).unwrap();
    let partial = proof.finish().unwrap();
    assert_eq!(partial.range(), RawRange { start: 3, end: 6 });
    assert!(!partial.is_complete_window());
    let mut proof = RawAcquisitionProof::new(generation.clone()).unwrap();
    proof.push(first.clone()).unwrap();
    proof.push(last.clone()).unwrap();
    let complete = proof.finish().unwrap();
    assert_eq!(complete.range(), RawRange { start: 3, end: 10 });
    assert!(complete.is_complete_window());
    assert_eq!(complete.root_body(), first.body);
    assert_eq!(complete.last_body(), last.body);
    let mut changed_generation = last;
    let SourceGeneration::File { observed, .. } = &mut changed_generation.generation else {
        unreachable!()
    };
    observed.changed.nanos += 1;
    let mut proof = RawAcquisitionProof::new(generation).unwrap();
    proof.push(first).unwrap();
    proof.push(changed_generation).unwrap();
    assert!(proof.finish().is_none());
}

#[test]
fn equal_file_stamps_do_not_join_different_stream_roles_or_members() {
    let generation = generation();
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
        let mut other = generation.clone();
        let SourceGeneration::File { stream, .. } = &mut other else {
            unreachable!()
        };
        stream.role = role;
        let mut proof = RawAcquisitionProof::new(generation.clone()).unwrap();
        proof.push(body(&generation, 0, 5)).unwrap();
        proof.push(body(&other, 5, 10)).unwrap();
        assert!(proof.finish().is_none());
    }
}

#[test]
fn database_snapshot_restart_cannot_complete_an_earlier_sampling_window() {
    let stream = RawStream {
        role: RawStreamRole::DatabaseDump,
        ..generation().stream().clone()
    };
    let generation = SourceGeneration::Database {
        stream,
        snapshot: DatabaseSnapshotToken {
            nonce: "00000000-0000-0000-0000-000000000001".into(),
        },
        window: RawRange { start: 0, end: 10 },
    };
    let mut restarted = generation.clone();
    let SourceGeneration::Database { snapshot, .. } = &mut restarted else {
        unreachable!()
    };
    snapshot.nonce = "00000000-0000-0000-0000-000000000002".into();
    let mut proof = RawAcquisitionProof::new(generation.clone()).unwrap();
    proof.push(body(&generation, 0, 5)).unwrap();
    proof.push(body(&restarted, 5, 10)).unwrap();
    assert!(proof.finish().is_none());
}

#[test]
fn empty_custody_requires_an_actual_body_and_duplicate_ranges_keep_eof_evidence() {
    let mut empty = generation();
    let SourceGeneration::File {
        window, observed, ..
    } = &mut empty
    else {
        unreachable!()
    };
    window.end = 0;
    observed.size = 0;
    assert!(
        RawAcquisitionProof::new(empty.clone())
            .unwrap()
            .finish()
            .is_none()
    );
    let mut proof = RawAcquisitionProof::new(empty.clone()).unwrap();
    let mut actual = body(&empty, 0, 0);
    actual.known_total = Some(0);
    proof.push(actual).unwrap();
    assert!(proof.finish().unwrap().is_complete_window());

    let generation = generation();
    let mut proof = RawAcquisitionProof::new(generation.clone()).unwrap();
    let mut first = body(&generation, 0, 10);
    first.known_total = None;
    proof.push(first).unwrap();
    proof.push(body(&generation, 0, 10)).unwrap();
    assert!(proof.finish().unwrap().is_complete_window());
}

#[test]
fn contiguous_coordinates_without_the_actual_predecessor_do_not_prove_custody() {
    let generation = generation();
    for predecessor in [None, Some("unrelated-body".into())] {
        let mut proof = RawAcquisitionProof::new(generation.clone()).unwrap();
        proof.push(body(&generation, 0, 5)).unwrap();
        let mut next = body(&generation, 5, 10);
        next.predecessor = predecessor;
        proof.push(next).unwrap();
        assert!(proof.finish().is_none());
    }
}

#[test]
fn provider_window_end_before_opened_file_size_is_not_eof() {
    let mut generation = generation();
    let SourceGeneration::File { observed, .. } = &mut generation else {
        unreachable!()
    };
    observed.size = 20;
    let mut proof = RawAcquisitionProof::new(generation.clone()).unwrap();
    proof.push(body(&generation, 0, 10)).unwrap();
    assert!(proof.finish().is_none());
}

#[test]
fn scheduling_rank_does_not_rename_a_raw_stream_or_its_durable_wire_key() {
    let original = generation();
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
fn custody_continuation_preserves_unresolved_or_unknown_disposition() {
    let generation = generation();
    for disposition in [None, Some(CaptureDisposition::UnresolvedBasisConflict)] {
        let mut proof = RawAcquisitionProof::new(generation.clone()).unwrap();
        let mut first = body(&generation, 0, 5);
        first.disposition = disposition;
        proof.push(first).unwrap();
        proof.push(body(&generation, 5, 10)).unwrap();
        let cursor = proof.finish().unwrap();
        assert!(cursor.is_complete_window());
        assert_eq!(cursor.disposition(), disposition);
    }
}

#[test]
fn distinct_original_acquisitions_with_equal_bytes_are_ambiguous_not_disposition_merges() {
    let generation = generation();
    let mut proof = RawAcquisitionProof::new(generation.clone()).unwrap();
    let mut old = body(&generation, 0, 10);
    old.body = "original-unresolved".into();
    old.disposition = Some(CaptureDisposition::UnresolvedBasisConflict);
    proof.push(old).unwrap();
    let mut new = body(&generation, 0, 10);
    new.body = "new-actual-observation".into();
    proof.push(new).unwrap();
    assert!(proof.finish().is_none());
}
