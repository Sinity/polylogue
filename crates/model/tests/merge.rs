use polylogue_model::merge::*;
use polylogue_model::{BlobHash, Timestamp};
use std::collections::{BTreeMap, BTreeSet};

fn candidate(reference: u64, digest: u8) -> Candidate {
    Candidate {
        reference,
        finish: FinishState::Finished,
        route: RouteKind::Native,
        version_hash: Some(BlobHash::from_bytes([digest; 32])),
        presence: PresenceFacts {
            nonempty_blocks: 1,
            blocks_with_nonempty_text: 1,
            slots: BTreeMap::new(),
        },
    }
}
fn with_slot(mut candidate: Candidate, slot: Slot, value: u64) -> Candidate {
    candidate.presence.slots.insert(
        slot,
        SuppliedValue {
            reference: value,
            source_time: None,
        },
    );
    candidate
}
fn singleton(candidate: &Candidate) -> EffectiveInput {
    fold_input(
        &[Chain {
            current: true,
            revisions: &[Revision {
                length: 1,
                candidate: Some(candidate),
            }],
        }],
        RewritePolicy::Lossy {
            preserve_older_times: false,
        },
    )
    .unwrap()
    .unwrap()
}
fn refs(input: &EffectiveInput) -> BTreeSet<u64> {
    input.body.candidates.clone()
}
fn slot(field: MessageField) -> Slot {
    Slot::Message(field)
}

#[test]
fn source_rank_is_finished_then_route_then_three_mapper_counts_then_digest() {
    let mut native = candidate(1, 1);
    let mut export = candidate(2, 255);
    export.route = RouteKind::AccountExport;
    export.presence.nonempty_blocks = 100;
    assert_eq!(
        refs(
            &select_effective(&[singleton(&export), singleton(&native)])
                .unwrap()
                .unwrap()
        ),
        BTreeSet::from([1])
    );
    native.finish = FinishState::InProgress;
    assert_eq!(
        refs(
            &select_effective(&[singleton(&native), singleton(&export)])
                .unwrap()
                .unwrap()
        ),
        BTreeSet::from([2])
    );
    native.finish = FinishState::Finished;
    export.route = RouteKind::Native;
    export.presence.nonempty_blocks = 2;
    assert_eq!(
        refs(
            &select_effective(&[singleton(&native), singleton(&export)])
                .unwrap()
                .unwrap()
        ),
        BTreeSet::from([2])
    );
    native.presence.nonempty_blocks = 2;
    native.presence.blocks_with_nonempty_text = 2;
    assert_eq!(
        refs(
            &select_effective(&[singleton(&export), singleton(&native)])
                .unwrap()
                .unwrap()
        ),
        BTreeSet::from([1])
    );
    export.presence.blocks_with_nonempty_text = 2;
    native = with_slot(native, slot(MessageField::Model), 0); // supplied known empty/zero reference still counts
    assert_eq!(native.presence.completeness().unwrap().supplied_fields, 1);
    assert_eq!(
        refs(
            &select_effective(&[singleton(&export), singleton(&native)])
                .unwrap()
                .unwrap()
        ),
        BTreeSet::from([1])
    );
}

#[test]
fn current_lower_hash_replaces_old_input_candidate_and_reelects_other_input() {
    let old_a = candidate(1, 240);
    let current_a = candidate(2, 16);
    let b = candidate(3, 128);
    let before = select_effective(&[singleton(&old_a), singleton(&b)])
        .unwrap()
        .unwrap();
    assert_eq!(refs(&before), BTreeSet::from([1]));
    // The forbidden all-candidate comparison cycles: current A > old A by
    // revision, old A > B by digest, B > current A by digest.
    for reversed_revisions in [false, true] {
        let mut revisions = vec![
            Revision {
                length: 1,
                candidate: Some(&old_a),
            },
            Revision {
                length: 2,
                candidate: Some(&current_a),
            },
        ];
        if reversed_revisions {
            revisions.reverse();
        }
        let folded = fold_input(
            &[Chain {
                current: true,
                revisions: &revisions,
            }],
            RewritePolicy::Lossy {
                preserve_older_times: false,
            },
        )
        .unwrap()
        .unwrap();
        assert_eq!(refs(&folded), BTreeSet::from([2]));
        for reversed_inputs in [false, true] {
            let mut inputs = vec![folded.clone(), singleton(&b)];
            if reversed_inputs {
                inputs.reverse();
            }
            assert_eq!(
                refs(&select_effective(&inputs).unwrap().unwrap()),
                BTreeSet::from([3])
            );
        }
    }
}

#[test]
fn older_chains_are_reduced_before_their_content_tie_break() {
    let early_x = candidate(1, 240);
    let later_x = candidate(2, 16);
    let y = candidate(3, 128);
    for reverse in [false, true] {
        let mut x = vec![
            Revision {
                length: 1,
                candidate: Some(&early_x),
            },
            Revision {
                length: 2,
                candidate: Some(&later_x),
            },
        ];
        if reverse {
            x.reverse();
        }
        let y_revisions = [Revision {
            length: 1,
            candidate: Some(&y),
        }];
        let mut chains = vec![
            Chain {
                current: false,
                revisions: &x,
            },
            Chain {
                current: false,
                revisions: &y_revisions,
            },
        ];
        if reverse {
            chains.reverse();
        }
        let folded = fold_input(
            &chains,
            RewritePolicy::Lossy {
                preserve_older_times: false,
            },
        )
        .unwrap()
        .unwrap();
        assert_eq!(refs(&folded), BTreeSet::from([3]));
    }
}

#[test]
fn current_chain_preference_is_only_after_rank_prefix_and_before_digest() {
    let old = candidate(1, 240);
    let current = candidate(2, 16);
    let old_revision = [Revision {
        length: 999,
        candidate: Some(&old),
    }];
    let current_revision = [Revision {
        length: 1,
        candidate: Some(&current),
    }];
    let folded = fold_input(
        &[
            Chain {
                current: false,
                revisions: &old_revision,
            },
            Chain {
                current: true,
                revisions: &current_revision,
            },
        ],
        RewritePolicy::Lossy {
            preserve_older_times: false,
        },
    )
    .unwrap()
    .unwrap();
    assert_eq!(refs(&folded), BTreeSet::from([2]));
    let mut more_complete_old = old.clone();
    more_complete_old.presence.nonempty_blocks = 2;
    let old_revision = [Revision {
        length: 999,
        candidate: Some(&more_complete_old),
    }];
    let folded = fold_input(
        &[
            Chain {
                current: false,
                revisions: &old_revision,
            },
            Chain {
                current: true,
                revisions: &current_revision,
            },
        ],
        RewritePolicy::Lossy {
            preserve_older_times: false,
        },
    )
    .unwrap()
    .unwrap();
    assert_eq!(refs(&folded), BTreeSet::from([1]));
}

#[test]
fn edit_eligibility_precedes_rank_and_excludes_deleted_message_or_field() {
    let old = with_slot(candidate(1, 240), slot(MessageField::Model), 100);
    let current = candidate(2, 16);
    let revisions = [
        Revision {
            length: 10,
            candidate: Some(&old),
        },
        Revision {
            length: 20,
            candidate: Some(&current),
        },
    ];
    let chains = [Chain {
        current: true,
        revisions: &revisions,
    }];
    let edit = fold_input(&chains, RewritePolicy::Edit).unwrap().unwrap();
    assert_eq!(refs(&edit), BTreeSet::from([2]));
    assert!(!edit.slots.contains_key(&slot(MessageField::Model)));
    let lossy = fold_input(
        &chains,
        RewritePolicy::Lossy {
            preserve_older_times: false,
        },
    )
    .unwrap()
    .unwrap();
    assert_eq!(refs(&lossy), BTreeSet::from([1]));
    let deleted = [
        Revision {
            length: 10,
            candidate: Some(&old),
        },
        Revision {
            length: 20,
            candidate: None,
        },
    ];
    assert_eq!(
        fold_input(
            &[Chain {
                current: true,
                revisions: &deleted
            }],
            RewritePolicy::Edit
        )
        .unwrap(),
        None
    );
    assert!(
        fold_input(
            &[Chain {
                current: true,
                revisions: &deleted
            }],
            RewritePolicy::Lossy {
                preserve_older_times: false
            }
        )
        .unwrap()
        .is_some()
    );
    assert_eq!(
        fold_input(
            &[Chain {
                current: false,
                revisions: &revisions
            }],
            RewritePolicy::Edit
        ),
        Err(MergeError::UnresolvedEditHead)
    );
}

#[test]
fn fill_uses_retained_input_facts_without_rescoring_or_filling_parent() {
    let winner = candidate(1, 240);
    let mut donor = with_slot(
        with_slot(candidate(2, 16), slot(MessageField::Model), 20),
        slot(MessageField::Parent),
        21,
    );
    donor.route = RouteKind::AccountExport;
    let first = singleton(&winner);
    let second = singleton(&donor);
    let selected = select_effective(&[first.clone(), second.clone()])
        .unwrap()
        .unwrap();
    assert_eq!(selected.body.rank, first.body.rank);
    assert_eq!(refs(&selected), BTreeSet::from([1]));
    assert!(selected.slots.contains_key(&slot(MessageField::Model)));
    assert!(!selected.slots.contains_key(&slot(MessageField::Parent)));
    assert_eq!(second.body.rank.completeness.supplied_fields, 2);
    let changed = with_slot(candidate(3, 1), slot(MessageField::StopReason), 30);
    let rebuilt = select_effective(&[singleton(&changed), second.clone()])
        .unwrap()
        .unwrap();
    let reversed = select_effective(&[second, singleton(&changed)])
        .unwrap()
        .unwrap();
    assert_eq!(rebuilt, reversed);
    assert!(!rebuilt.slots.contains_key(&slot(MessageField::Parent)));
}

#[test]
fn damaged_times_take_earliest_supplied_older_chain_value_without_rescoring() {
    let mut old = candidate(1, 240);
    old.presence.slots.insert(
        slot(MessageField::At),
        SuppliedValue {
            reference: 10,
            source_time: Some(Timestamp(50)),
        },
    );
    let mut older = candidate(2, 128);
    older.presence.slots.insert(
        slot(MessageField::At),
        SuppliedValue {
            reference: 20,
            source_time: Some(Timestamp(30)),
        },
    );
    let mut current = candidate(3, 16);
    current.presence.slots.insert(
        slot(MessageField::At),
        SuppliedValue {
            reference: 30,
            source_time: Some(Timestamp(0)),
        },
    );
    let a = [Revision {
        length: 1,
        candidate: Some(&old),
    }];
    let b = [Revision {
        length: 1,
        candidate: Some(&older),
    }];
    let c = [Revision {
        length: 1,
        candidate: Some(&current),
    }];
    let chains = [
        Chain {
            current: false,
            revisions: &a,
        },
        Chain {
            current: false,
            revisions: &b,
        },
        Chain {
            current: true,
            revisions: &c,
        },
    ];
    let folded = fold_input(
        &chains,
        RewritePolicy::Lossy {
            preserve_older_times: true,
        },
    )
    .unwrap()
    .unwrap();
    assert_eq!(refs(&folded), BTreeSet::from([3]));
    assert_eq!(
        folded.slots[&slot(MessageField::At)].sources,
        BTreeSet::from([FactRef {
            candidate: 2,
            value: 20
        }])
    );
    assert_eq!(folded.body.rank.completeness.supplied_fields, 1);
}

#[test]
fn equal_evidence_keeps_all_provenance_and_unresolved_facts_never_default() {
    let a = candidate(1, 128);
    let b = candidate(2, 128);
    let merged = select_effective(&[singleton(&a), singleton(&b)])
        .unwrap()
        .unwrap();
    assert_eq!(refs(&merged), BTreeSet::from([1, 2]));
    assert_eq!(
        merged,
        select_effective(&[singleton(&b), singleton(&a)])
            .unwrap()
            .unwrap()
    );
    let mut unknown = a.clone();
    unknown.finish = FinishState::Unknown;
    assert_eq!(unknown.rank().unwrap().finish, FinishState::Unknown);
    unknown.finish = FinishState::Finished;
    unknown.version_hash = None;
    assert_eq!(unknown.rank(), Err(MergeError::UnresolvedEvidence));
    unknown.version_hash = a.version_hash;
    unknown.presence.nonempty_blocks = 0;
    assert_eq!(unknown.rank(), Err(MergeError::InvalidPresence));
}

#[test]
fn unknown_completion_is_eligible_between_positive_finished_and_in_progress() {
    let mut finished = candidate(1, 1);
    finished.route = RouteKind::PageDom;
    let mut unknown = candidate(2, 255);
    unknown.finish = FinishState::Unknown;
    unknown.presence.nonempty_blocks = 100;
    let mut producing = unknown.clone();
    producing.reference = 3;
    producing.finish = FinishState::InProgress;
    producing.presence.nonempty_blocks = 1000;
    for candidates in [
        vec![
            singleton(&finished),
            singleton(&unknown),
            singleton(&producing),
        ],
        vec![
            singleton(&producing),
            singleton(&unknown),
            singleton(&finished),
        ],
    ] {
        let selected = select_effective(&candidates).unwrap().unwrap();
        assert_eq!(refs(&selected), BTreeSet::from([1]));
        assert_eq!(selected.body.rank.finish, FinishState::Finished);
    }
    let selected = select_effective(&[singleton(&producing), singleton(&unknown)])
        .unwrap()
        .unwrap();
    assert_eq!(refs(&selected), BTreeSet::from([2]));
    assert_eq!(selected.body.rank.finish, FinishState::Unknown);
    let mut other_unknown = unknown.clone();
    other_unknown.reference = 4;
    other_unknown.version_hash = Some(BlobHash::from_bytes([1; 32]));
    let selected = select_effective(&[singleton(&other_unknown), singleton(&unknown)])
        .unwrap()
        .unwrap();
    assert_eq!(refs(&selected), BTreeSet::from([2]));
    unknown.version_hash = None;
    assert_eq!(unknown.rank(), Err(MergeError::UnresolvedEvidence));
}

#[test]
fn typed_slot_facts_roundtrip_and_duplicate_slots_are_rejected() {
    let c = with_slot(candidate(1, 128), slot(MessageField::Model), 0);
    let wire = serde_json::to_vec(&c).unwrap();
    assert_eq!(serde_json::from_slice::<Candidate>(&wire).unwrap(), c);
    let selected = singleton(&c);
    let wire = serde_json::to_vec(&selected).unwrap();
    assert_eq!(
        serde_json::from_slice::<EffectiveInput>(&wire).unwrap(),
        selected
    );
    let mut wire = serde_json::to_value(&c).unwrap();
    let slots = wire["presence"]["slots"].as_array_mut().unwrap();
    slots.push(slots[0].clone());
    assert!(serde_json::from_value::<Candidate>(wire).is_err());
}

#[test]
fn archival_edit_ambiguity_retains_other_inputs_until_head_resolves() {
    let a = candidate(1, 240);
    let b = candidate(2, 16);
    let ar = [Revision {
        length: 100,
        candidate: Some(&a),
    }];
    let br = [Revision {
        length: 10,
        candidate: Some(&b),
    }];
    let archives = [
        Chain {
            current: false,
            revisions: &ar,
        },
        Chain {
            current: false,
            revisions: &br,
        },
    ];
    assert_eq!(
        fold_input(&archives, RewritePolicy::Edit),
        Err(MergeError::UnresolvedEditHead)
    );
    assert_eq!(
        fold_input(&archives[..1], RewritePolicy::Edit),
        Err(MergeError::UnresolvedEditHead),
        "one unproved archive is not immutable-head authority"
    );
    let imported = fold_input(
        &[Chain {
            current: true,
            revisions: &ar,
        }],
        RewritePolicy::Edit,
    )
    .unwrap()
    .unwrap();
    assert_eq!(refs(&imported), BTreeSet::from([1]));
    // Other already-effective inputs can still be elected independently.
    assert_eq!(
        refs(&select_effective(&[singleton(&b)]).unwrap().unwrap()),
        BTreeSet::from([2])
    );
    let resolved = fold_input(
        &[
            archives[0],
            Chain {
                current: true,
                revisions: &br,
            },
        ],
        RewritePolicy::Edit,
    )
    .unwrap()
    .unwrap();
    assert_eq!(refs(&resolved), BTreeSet::from([2]));
}

#[test]
fn an_unchanged_candidate_reference_can_be_retained_at_several_revisions() {
    let a = candidate(1, 128);
    let revisions = [
        Revision {
            length: 10,
            candidate: Some(&a),
        },
        Revision {
            length: 20,
            candidate: Some(&a),
        },
    ];
    assert_eq!(
        refs(
            &fold_input(
                &[Chain {
                    current: true,
                    revisions: &revisions
                }],
                RewritePolicy::Lossy {
                    preserve_older_times: false
                }
            )
            .unwrap()
            .unwrap()
        ),
        BTreeSet::from([1])
    );
}

#[test]
fn empty_extra_json_key_is_a_supplied_fact_not_a_schema_error() {
    let c = with_slot(
        candidate(1, 128),
        Slot::Extra {
            block: None,
            key: String::new(),
        },
        0,
    );
    assert_eq!(c.presence.completeness().unwrap().supplied_fields, 1);
    let wire = serde_json::to_vec(&c).unwrap();
    assert_eq!(serde_json::from_slice::<Candidate>(&wire).unwrap(), c);
}

#[test]
fn detached_immutable_copy_can_witness_content_that_a_proved_true_edit_removed() {
    // M/F/N: A1 has a richer message M, field F, and another message N.
    // A2 is a proved true edit, reducing M, removing F and deleting N.
    let mut a1_m = with_slot(candidate(1, 240), slot(MessageField::Model), 10);
    a1_m.presence.nonempty_blocks = 2;
    let a2_m = candidate(2, 16);
    let a1_n = candidate(4, 200);
    let old_m = [Revision {
        length: 100,
        candidate: Some(&a1_m),
    }];
    let new_m = [Revision {
        length: 50,
        candidate: Some(&a2_m),
    }];
    let old_n = [Revision {
        length: 100,
        candidate: Some(&a1_n),
    }];
    let new_n = [Revision {
        length: 50,
        candidate: None,
    }];
    // Grouping is explicit input/chain authority. Byte length across chains is
    // not chronology, and retained old A is not independently eligible.
    let a_m = fold_input(
        &[
            Chain {
                current: false,
                revisions: &old_m,
            },
            Chain {
                current: true,
                revisions: &new_m,
            },
        ],
        RewritePolicy::Edit,
    )
    .unwrap()
    .unwrap();
    assert_eq!(refs(&a_m), BTreeSet::from([2]));
    assert!(!a_m.slots.contains_key(&slot(MessageField::Model)));
    assert_eq!(
        fold_input(
            &[
                Chain {
                    current: false,
                    revisions: &old_n
                },
                Chain {
                    current: true,
                    revisions: &new_n
                }
            ],
            RewritePolicy::Edit
        )
        .unwrap(),
        None
    );
    // Detached B is explicitly a separate immutable input holding A1, not an
    // archival revision of A. Its own current marker asserts no live freshness.
    let mut b_m = a1_m.clone();
    b_m.reference = 3;
    let mut b_n = a1_n.clone();
    b_n.reference = 5;
    let immutable = |candidate: &Candidate| {
        fold_input(
            &[Chain {
                current: true,
                revisions: &[Revision {
                    length: 100,
                    candidate: Some(candidate),
                }],
            }],
            RewritePolicy::Edit,
        )
        .unwrap()
        .unwrap()
    };
    let b_m = immutable(&b_m);
    let b_n = immutable(&b_n);
    for inputs in [
        vec![a_m.clone(), b_m.clone()],
        vec![b_m.clone(), a_m.clone()],
    ] {
        let selected = select_effective(&inputs).unwrap().unwrap();
        assert_eq!(refs(&selected), BTreeSet::from([3]));
        assert!(selected.slots.contains_key(&slot(MessageField::Model)));
    }
    assert_eq!(
        refs(&select_effective(&[b_n]).unwrap().unwrap()),
        BTreeSet::from([5])
    );
    // Rebuilding solely A's proved revision group cannot revive F or N.
    let rebuilt = fold_input(
        &[
            Chain {
                current: true,
                revisions: &new_m,
            },
            Chain {
                current: false,
                revisions: &old_m,
            },
        ],
        RewritePolicy::Edit,
    )
    .unwrap()
    .unwrap();
    assert_eq!(rebuilt, a_m);
}

#[test]
fn equal_content_version_conflicts_are_occupied_and_independent_of_arrival_or_ref_numbers() {
    for ids in [[1, 2, 3], [900, 7, 50]] {
        let a = with_slot(
            with_slot(
                candidate(ids[0], 4),
                slot(MessageField::At),
                ids[0] * 10 + 1,
            ),
            slot(MessageField::CompletedAt),
            ids[0] * 10 + 2,
        );
        let b = with_slot(
            with_slot(
                candidate(ids[1], 4),
                slot(MessageField::At),
                ids[1] * 10 + 1,
            ),
            slot(MessageField::CompletedAt),
            ids[1] * 10 + 2,
        );
        let mut donor = with_slot(
            with_slot(
                candidate(ids[2], 4),
                slot(MessageField::At),
                ids[2] * 10 + 1,
            ),
            slot(MessageField::CompletedAt),
            ids[2] * 10 + 2,
        );
        donor.route = RouteKind::AccountExport;
        let times = BTreeMap::from([
            (ids[0] * 10 + 1, 100i64),
            (ids[0] * 10 + 2, 110),
            (ids[1] * 10 + 1, 200),
            (ids[1] * 10 + 2, 210),
            (ids[2] * 10 + 1, 150),
            (ids[2] * 10 + 2, 150),
        ]);
        for inputs in [
            [singleton(&a), singleton(&b), singleton(&donor)],
            [singleton(&donor), singleton(&b), singleton(&a)],
        ] {
            let merged = select_effective(&inputs).unwrap().unwrap();
            for field in [MessageField::At, MessageField::CompletedAt] {
                let choice = &merged.slots[&slot(field)];
                assert_eq!(choice.sources.len(), 2);
                assert!(choice.sources.iter().all(|r| r.candidate != ids[2]));
                assert_eq!(
                    choice
                        .resolve::<std::convert::Infallible>(|a, b| Ok(
                            times[&a.value] == times[&b.value]
                        ))
                        .unwrap(),
                    FieldResolution::Conflict
                );
            }
            let restored: EffectiveInput =
                serde_json::from_slice(&serde_json::to_vec(&merged).unwrap()).unwrap();
            assert_eq!(restored, merged);
        }
        let remaining = select_effective(&[singleton(&a), singleton(&donor)])
            .unwrap()
            .unwrap();
        assert!(matches!(
            remaining.slots[&slot(MessageField::At)]
                .resolve::<std::convert::Infallible>(|a, b| Ok(times[&a.value] == times[&b.value]))
                .unwrap(),
            FieldResolution::Agreed(_)
        ));
        let mut stronger = with_slot(
            with_slot(candidate(999, 9), slot(MessageField::At), 300),
            slot(MessageField::CompletedAt),
            310,
        );
        stronger.presence.nonempty_blocks = 2;
        let resolved = select_effective(&[singleton(&a), singleton(&b), singleton(&stronger)])
            .unwrap()
            .unwrap();
        assert_eq!(
            resolved.slots[&slot(MessageField::At)].sources,
            BTreeSet::from([FactRef {
                candidate: 999,
                value: 300
            }])
        );
    }
}

#[test]
fn typed_agreement_includes_time_quality_precision_and_known_empty_values() {
    use polylogue_model::{Precision, ProviderTime};
    let refs = [
        FactRef {
            candidate: 1,
            value: 1,
        },
        FactRef {
            candidate: 2,
            value: 2,
        },
    ];
    let choice = SlotChoice {
        sources: refs.into_iter().collect(),
    };
    let a = ProviderTime::Intrinsic {
        at: Timestamp(100),
        precision: Precision::Instant,
        until: None,
    };
    for b in [
        a,
        ProviderTime::Intrinsic {
            at: Timestamp(100),
            precision: Precision::Second,
            until: None,
        },
        ProviderTime::FileMtime {
            at: Timestamp(100),
            precision: Precision::Instant,
            until: None,
        },
    ] {
        let values = BTreeMap::from([(refs[0], a), (refs[1], b)]);
        let result = choice
            .resolve::<std::convert::Infallible>(|l, r| Ok(values[&l] == values[&r]))
            .unwrap();
        assert_eq!(matches!(result, FieldResolution::Agreed(_)), a == b);
    }
    for value in [
        serde_json::json!(false),
        serde_json::json!(0),
        serde_json::json!(""),
        serde_json::json!([]),
        serde_json::json!({}),
    ] {
        let values = BTreeMap::from([(refs[0], value.clone()), (refs[1], value)]);
        assert!(matches!(
            choice
                .resolve::<std::convert::Infallible>(|l, r| Ok(values[&l] == values[&r]))
                .unwrap(),
            FieldResolution::Agreed(_)
        ));
    }
    assert_eq!(
        SlotChoice {
            sources: BTreeSet::new()
        }
        .resolve::<std::convert::Infallible>(|_, _| unreachable!())
        .unwrap(),
        FieldResolution::Missing
    );
}

#[test]
fn field_agreement_resumes_one_witness_at_a_time_and_failure_does_not_invent_conflict() {
    let a = FactRef {
        candidate: 1,
        value: 10,
    };
    let b = FactRef {
        candidate: 2,
        value: 20,
    };
    let c = FactRef {
        candidate: 3,
        value: 30,
    };
    let values = BTreeMap::from([(a, 5), (b, 5), (c, 6)]);
    let mut state = FieldResolution::Missing;
    state
        .observe::<()>(a, |_, _| panic!("first witness needs no comparison"))
        .unwrap();
    let old = state;
    assert_eq!(
        state.observe(b, |_, _| Err("synthetic value read failed")),
        Err("synthetic value read failed")
    );
    assert_eq!(state, old);
    for next in [b, c] {
        state = serde_json::from_slice(&serde_json::to_vec(&state).unwrap()).unwrap();
        state
            .observe::<()>(next, |l, r| Ok(values[&l] == values[&r]))
            .unwrap();
    }
    assert_eq!(state, FieldResolution::Conflict);
    // Further witnesses remain retained by SlotChoice/root ownership; their
    // payload need not be read just to preserve an already known conflict.
    state
        .observe::<()>(a, |_, _| panic!("conflict already established"))
        .unwrap();
}
