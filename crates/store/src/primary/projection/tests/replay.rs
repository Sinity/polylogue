use super::*;
use crate::primary::*;

fn context() -> NativeFactContext {
    NativeFactContext {
        kind: NativeKind {
            name: "synthetic.input".into(),
            version: 1,
        },
        kind_definition_hash: "a".repeat(64),
        projection: ProjectionIdentity {
            policy_epoch: 7,
            payload_hash: "b".repeat(64),
        },
        origin: ConfirmedOrigin::default(),
    }
}
fn control(control: ReplayControl) -> Confirmed {
    Confirmed {
        position: 0,
        output_index: 0,
        event: String::new(),
        change: ConfirmedChange::Control(Box::new(control)),
        native: None,
    }
}
fn initialize(conn: &Connection) {
    conn.execute(
        "INSERT INTO meta(key,value) VALUES ('confirmed_scope','scope')",
        [],
    )
    .unwrap();
    save_replay(
        conn,
        &ReplayState {
            scope: Some("scope".into()),
            policy_epoch: Some(7),
            ..Default::default()
        },
    )
    .unwrap();
}
fn next() -> Confirmed {
    control(ReplayControl::Page {
        page: ReplayPage {
            position: 20,
            policy_epoch: 7,
            scope: "scope".into(),
            continuation: ReplayContinuation::Next {
                cursor: "signed-page-two".into(),
            },
        },
    })
}
fn complete() -> Confirmed {
    control(ReplayControl::Page {
        page: ReplayPage {
            position: 20,
            policy_epoch: 7,
            scope: "scope".into(),
            continuation: ReplayContinuation::Complete {
                projection: FullProjectionComplete {
                    stamp: "full-projection-basis".into(),
                    position: 20,
                    scope: "scope".into(),
                    catalog: "exact-indexed-catalog".into(),
                },
                raw_inputs: RawInputCoverage::Complete {
                    proof: RawInputProof {
                        stamp: "raw-request-basis".into(),
                        position: 20,
                        policy_epoch: 7,
                        scope: "scope".into(),
                    },
                },
            },
        },
    })
}
fn native_input() -> Confirmed {
    let mut event = input(10, "source");
    event.native = Some(context());
    event
}

#[test]
fn replay_cursor_and_delivery_survive_reopen_without_advancing_application_or_publication() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("index.db");
    let mut index = cache(&path);
    initialize(&index);
    let archive = crate::Archive::on_sinex(dir.path(), &dir.path().join("bytes"));
    let resolver = crate::writer::resolve::Resolver::new(&archive, &index, None).unwrap();
    apply(
        &mut index,
        &resolver,
        &[
            native_input(),
            next(),
            control(ReplayControl::Delivered {
                delivery: NativeDeliveryCursor {
                    stream_instance: "stream".into(),
                    stream_sequence: 18,
                },
            }),
        ],
    )
    .unwrap();
    drop(index);
    let mut index = Connection::open(path).unwrap();
    let state = replay_state(&index).unwrap();
    assert_eq!(state.cursor.unwrap().cursor, "signed-page-two");
    assert_eq!(state.staged_delivery.unwrap().stream_sequence, 18);
    assert_eq!(state.applied_position, None);
    assert_eq!(state.coverage, None);
    assert_eq!(state.publication, None);
    assert_eq!(native_context(&index, "event-source").unwrap(), None);
    assert_eq!(apply(&mut index, &resolver, &[complete()]).unwrap(), 20);
    assert_eq!(
        native_context(&index, "event-source").unwrap(),
        Some(context())
    );
    let state = replay_state(&index).unwrap();
    assert_eq!(state.certified_position(), Some(20));
    assert_eq!(state.publication, None);
    assert!(
        matches!(state.coverage.unwrap().raw_inputs, RawInputCoverage::Complete { proof } if proof.stamp == "raw-request-basis")
    );
    apply(
        &mut index,
        &resolver,
        &[control(ReplayControl::Coverage {
            coverage: RawInputCoverage::Complete {
                proof: RawInputProof {
                    stamp: "actual-write-proof".into(),
                    position: 20,
                    policy_epoch: 7,
                    scope: "scope".into(),
                },
            },
        })],
    )
    .unwrap();
    assert_eq!(
        replay_state(&index).unwrap().publication.unwrap().stamp,
        "actual-write-proof"
    );
}

#[test]
fn missing_original_context_refuses_final_certificate_and_preserves_staged_page() {
    let dir = tempfile::tempdir().unwrap();
    let mut index = cache(&dir.path().join("index.db"));
    initialize(&index);
    let archive = crate::Archive::on_sinex(dir.path(), &dir.path().join("bytes"));
    let resolver = crate::writer::resolve::Resolver::new(&archive, &index, None).unwrap();
    apply(&mut index, &resolver, &[input(10, "source"), next()]).unwrap();
    assert!(apply(&mut index, &resolver, &[complete()]).is_err());
    let state = replay_state(&index).unwrap();
    assert!(state.coverage.is_none());
    assert_eq!(checkpoint(&index).unwrap(), None);
    assert_eq!(state.cursor.unwrap().cursor, "signed-page-two");
    assert_eq!(
        index
            .query_row("SELECT count(*) FROM primary_pending", [], |row| row
                .get::<_, i64>(0))
            .unwrap(),
        1
    );
}

#[test]
fn failed_final_projection_keeps_certificate_hidden_until_the_same_commit_succeeds() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("index.db");
    let mut index = cache(&path);
    initialize(&index);
    let archive = crate::Archive::on_sinex(dir.path(), &dir.path().join("bytes"));
    let resolver = crate::writer::resolve::Resolver::new(&archive, &index, None).unwrap();
    apply(&mut index, &resolver, &[native_input(), next()]).unwrap();
    index.execute_batch("CREATE TRIGGER refuse_source BEFORE INSERT ON source_input BEGIN SELECT raise(ABORT,'refuse projection'); END").unwrap();
    assert!(apply(&mut index, &resolver, &[complete()]).is_err());
    drop(index);
    let mut index = Connection::open(path).unwrap();
    let state = replay_state(&index).unwrap();
    assert!(state.coverage.is_none());
    assert_eq!(state.applied_position, None);
    index.execute_batch("DROP TRIGGER refuse_source").unwrap();
    assert_eq!(apply(&mut index, &resolver, &[]).unwrap(), 20);
    assert_eq!(replay_state(&index).unwrap().certified_position(), Some(20));
}

#[test]
fn native_completion_extends_the_original_projection_basis_and_erasure_fence_only_at_actual_q() {
    let dir = tempfile::tempdir().unwrap();
    let mut index = cache(&dir.path().join("index.db"));
    initialize(&index);
    let archive = crate::Archive::on_sinex(dir.path(), &dir.path().join("bytes"));
    let resolver = crate::writer::resolve::Resolver::new(&archive, &index, None).unwrap();
    apply(&mut index, &resolver, &[native_input(), complete()]).unwrap();
    apply(
        &mut index,
        &resolver,
        &[control(ReplayControl::ErasedThrough { position: 30 })],
    )
    .unwrap();
    assert_eq!(replay_state(&index).unwrap().certified_position(), None);
    for position in [25, 30] {
        apply(
            &mut index,
            &resolver,
            &[control(ReplayControl::Completion {
                position,
                delivered: DeliveredCompletion {
                    stream_instance: "stream".into(),
                    stream_sequence: position as u64,
                },
            })],
        )
        .unwrap();
        let state = replay_state(&index).unwrap();
        assert_eq!(state.applied_position, Some(position));
        assert_eq!(state.coverage.unwrap().projection.position, 20);
        assert_eq!(state.erasure_fence, (position < 30).then_some(30));
    }
    assert_eq!(replay_state(&index).unwrap().certified_position(), Some(30));
}

#[test]
fn original_unknown_origin_cannot_be_promoted_by_later_matching_payload() {
    let dir = tempfile::tempdir().unwrap();
    let mut index = cache(&dir.path().join("index.db"));
    initialize(&index);
    let archive = crate::Archive::on_sinex(dir.path(), &dir.path().join("bytes"));
    let resolver = crate::writer::resolve::Resolver::new(&archive, &index, None).unwrap();
    apply(&mut index, &resolver, &[native_input(), complete()]).unwrap();
    let mut changed = native_input();
    changed.position = 30;
    changed.native.as_mut().unwrap().origin.producer_identity = Some("later-identity".into());
    assert!(
        apply(
            &mut index,
            &resolver,
            &[
                changed,
                control(ReplayControl::Completion {
                    position: 30,
                    delivered: DeliveredCompletion {
                        stream_instance: "stream".into(),
                        stream_sequence: 30
                    },
                })
            ]
        )
        .is_err()
    );
    assert_eq!(checkpoint(&index).unwrap(), Some(20));
    assert_eq!(
        native_context(&index, "event-source").unwrap(),
        Some(context())
    );
}

#[test]
fn invalidated_basis_cannot_be_recertified_by_old_completion_or_publication_proof() {
    let dir = tempfile::tempdir().unwrap();
    let mut index = cache(&dir.path().join("index.db"));
    initialize(&index);
    let archive = crate::Archive::on_sinex(dir.path(), &dir.path().join("bytes"));
    let resolver = crate::writer::resolve::Resolver::new(&archive, &index, None).unwrap();
    apply(&mut index, &resolver, &[native_input(), complete()]).unwrap();
    apply(
        &mut index,
        &resolver,
        &[control(ReplayControl::BasisStatus {
            status: ReplayBasisStatus::Invalidated {
                policy_epoch: Some(8),
                scope: Some("scope".into()),
            },
        })],
    )
    .unwrap();
    for event in [
        complete(),
        control(ReplayControl::Completion {
            position: 30,
            delivered: DeliveredCompletion {
                stream_instance: "stream".into(),
                stream_sequence: 30,
            },
        }),
        control(ReplayControl::Coverage {
            coverage: RawInputCoverage::Complete {
                proof: RawInputProof {
                    stamp: "old-write".into(),
                    position: 20,
                    policy_epoch: 7,
                    scope: "scope".into(),
                },
            },
        }),
    ] {
        assert!(apply(&mut index, &resolver, &[event]).is_err());
        assert_eq!(replay_state(&index).unwrap().certified_position(), None);
        assert_eq!(checkpoint(&index).unwrap(), Some(20));
    }
    assert_eq!(
        native_context(&index, "event-source").unwrap(),
        Some(context())
    );
}

#[test]
fn certified_empty_owner_position_zero_is_distinct_from_unknown_application() {
    let dir = tempfile::tempdir().unwrap();
    let mut index = cache(&dir.path().join("index.db"));
    initialize(&index);
    let archive = crate::Archive::on_sinex(dir.path(), &dir.path().join("bytes"));
    let resolver = crate::writer::resolve::Resolver::new(&archive, &index, None).unwrap();
    assert_eq!(replay_state(&index).unwrap().applied_position, None);
    let event = control(ReplayControl::Page {
        page: ReplayPage {
            position: 0,
            policy_epoch: 7,
            scope: "scope".into(),
            continuation: ReplayContinuation::Complete {
                projection: FullProjectionComplete {
                    stamp: "empty-owner".into(),
                    position: 0,
                    scope: "scope".into(),
                    catalog: "catalog".into(),
                },
                raw_inputs: RawInputCoverage::NotRequested,
            },
        },
    });
    apply(&mut index, &resolver, &[event]).unwrap();
    assert_eq!(replay_state(&index).unwrap().certified_position(), Some(0));
    assert_eq!(checkpoint(&index).unwrap(), Some(0));
}

#[test]
fn offered_write_proof_waits_in_staging_until_its_full_completion_transaction_commits() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("index.db");
    let mut index = cache(&path);
    initialize(&index);
    let archive = crate::Archive::on_sinex(dir.path(), &dir.path().join("bytes"));
    let resolver = crate::writer::resolve::Resolver::new(&archive, &index, None).unwrap();
    index.execute_batch("CREATE TRIGGER refuse_atomic BEFORE INSERT ON source_input BEGIN SELECT raise(ABORT,'refuse'); END").unwrap();
    let offer = control(ReplayControl::Coverage {
        coverage: RawInputCoverage::Complete {
            proof: RawInputProof {
                stamp: "atomic-write-proof".into(),
                position: 20,
                policy_epoch: 7,
                scope: "scope".into(),
            },
        },
    });
    assert!(apply(&mut index, &resolver, &[native_input(), complete(), offer]).is_err());
    assert_eq!(replay_state(&index).unwrap().publication, None);
    assert_eq!(checkpoint(&index).unwrap(), None);
    drop(index);
    let mut index = Connection::open(path).unwrap();
    index.execute_batch("DROP TRIGGER refuse_atomic").unwrap();
    apply(&mut index, &resolver, &[]).unwrap();
    assert_eq!(checkpoint(&index).unwrap(), Some(20));
    assert_eq!(
        replay_state(&index).unwrap().publication.unwrap().stamp,
        "atomic-write-proof"
    );
}

#[test]
fn refusal_after_a_staged_completion_does_not_expose_its_facts_or_discard_them() {
    let dir = tempfile::tempdir().unwrap();
    let mut index = cache(&dir.path().join("index.db"));
    initialize(&index);
    let archive = crate::Archive::on_sinex(dir.path(), &dir.path().join("bytes"));
    let resolver = crate::writer::resolve::Resolver::new(&archive, &index, None).unwrap();
    apply(
        &mut index,
        &resolver,
        &[
            native_input(),
            complete(),
            control(ReplayControl::BasisStatus {
                status: ReplayBasisStatus::Unverified,
            }),
        ],
    )
    .unwrap();
    assert_eq!(checkpoint(&index).unwrap(), None);
    assert_eq!(replay_state(&index).unwrap().certified_position(), None);
    assert_eq!(
        index
            .query_row("SELECT count(*) FROM primary_pending", [], |row| row
                .get::<_, i64>(0))
            .unwrap(),
        1
    );
    assert_eq!(
        index
            .query_row("SELECT count(*) FROM primary_complete", [], |row| row
                .get::<_, i64>(0))
            .unwrap(),
        0
    );
    apply(&mut index, &resolver, &[complete()]).unwrap();
    assert_eq!(replay_state(&index).unwrap().certified_position(), Some(20));
}

#[test]
fn not_requested_retracts_an_offered_write_proof_without_losing_full_projection_completion() {
    let dir = tempfile::tempdir().unwrap();
    let mut index = cache(&dir.path().join("index.db"));
    initialize(&index);
    let archive = crate::Archive::on_sinex(dir.path(), &dir.path().join("bytes"));
    let resolver = crate::writer::resolve::Resolver::new(&archive, &index, None).unwrap();
    apply(
        &mut index,
        &resolver,
        &[
            native_input(),
            complete(),
            control(ReplayControl::Coverage {
                coverage: RawInputCoverage::Complete {
                    proof: RawInputProof {
                        stamp: "withdrawn-write-proof".into(),
                        position: 20,
                        policy_epoch: 7,
                        scope: "scope".into(),
                    },
                },
            }),
            control(ReplayControl::Coverage {
                coverage: RawInputCoverage::NotRequested,
            }),
        ],
    )
    .unwrap();
    let state = replay_state(&index).unwrap();
    assert_eq!(state.certified_position(), Some(20));
    assert_eq!(state.publication, None);
}

#[test]
fn renewed_write_stamp_preserves_the_first_frozen_proof_for_the_same_applied_basis() {
    let dir = tempfile::tempdir().unwrap();
    let mut index = cache(&dir.path().join("index.db"));
    initialize(&index);
    let archive = crate::Archive::on_sinex(dir.path(), &dir.path().join("bytes"));
    let resolver = crate::writer::resolve::Resolver::new(&archive, &index, None).unwrap();
    let offer = |stamp: &str| {
        control(ReplayControl::Coverage {
            coverage: RawInputCoverage::Complete {
                proof: RawInputProof {
                    stamp: stamp.into(),
                    position: 20,
                    policy_epoch: 7,
                    scope: "scope".into(),
                },
            },
        })
    };
    apply(
        &mut index,
        &resolver,
        &[
            native_input(),
            complete(),
            offer("first-owner-proof"),
            offer("renewed-staged-proof"),
        ],
    )
    .unwrap();
    apply(&mut index, &resolver, &[offer("renewed-applied-proof")]).unwrap();
    assert_eq!(
        replay_state(&index).unwrap().publication.unwrap().stamp,
        "first-owner-proof"
    );
}

#[test]
fn durable_read_fence_keeps_an_old_serialized_certificate_unavailable_after_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("index.db");
    let mut index = cache(&path);
    initialize(&index);
    let archive = crate::Archive::on_sinex(dir.path(), &dir.path().join("bytes"));
    let resolver = crate::writer::resolve::Resolver::new(&archive, &index, None).unwrap();
    apply(&mut index, &resolver, &[native_input(), complete()]).unwrap();
    let original = replay_state(&index).unwrap().coverage;
    crate::primary::scope::invalidate(&index, &archive).unwrap();
    drop(index);
    let index = Connection::open(path).unwrap();
    let fenced = replay_state(&index).unwrap();
    assert_eq!(fenced.coverage, original);
    assert_eq!(fenced.certified_position(), None);
    assert_eq!(fenced.publication, None);
    assert_eq!(fenced.applied_position, Some(20));
}
