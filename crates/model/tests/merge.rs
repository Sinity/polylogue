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
    assert_eq!(unknown.rank(), Err(MergeError::UnknownCompletion));
    unknown.finish = FinishState::Finished;
    unknown.version_hash = None;
    assert_eq!(unknown.rank(), Err(MergeError::UnresolvedEvidence));
    unknown.version_hash = a.version_hash;
    unknown.presence.nonempty_blocks = 0;
    assert_eq!(unknown.rank(), Err(MergeError::InvalidPresence));
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
    let imported = fold_input(&archives[..1], RewritePolicy::Edit)
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
