use super::{
    tests::{basis, source},
    *,
};
use crate::tests::{body, capture, client};
use polylogue_store::primary::{
    FileSnapshot, FileTime, RawRange, RawStream, RawStreamRole, SourceGeneration,
};

fn generation(size: u64) -> SourceGeneration {
    SourceGeneration::File {
        stream: RawStream {
            source: source(b"/home/op/raw.jsonl"),
            input: polylogue_store::InputKey {
                id: "raw-input".into(),
                format: "codex".into(),
                route: polylogue_model::Route::Live,
            },
            role: RawStreamRole::ProviderBytes,
            namespace: None,
        },
        observed: FileSnapshot {
            device: 1,
            inode: 2,
            size,
            modified: FileTime {
                seconds: 3,
                nanos: 4,
            },
            changed: FileTime {
                seconds: 5,
                nanos: 6,
            },
        },
        window: RawRange {
            start: 0,
            end: size,
        },
    }
}

fn acquire_ranges(
    sinex: &Sinex,
    root: &std::path::Path,
    generation: &SourceGeneration,
    intent: &str,
    ranges: &[RawRange],
) -> CaptureReceipt {
    acquire_ranges_on(
        sinex,
        root,
        generation,
        intent,
        ranges,
        basis(&generation.stream().source, None),
        None,
    )
}

fn acquire_ranges_on(
    sinex: &Sinex,
    root: &std::path::Path,
    generation: &SourceGeneration,
    intent: &str,
    ranges: &[RawRange],
    basis: polylogue_store::primary::SourceBasis,
    predecessor: Option<String>,
) -> CaptureReceipt {
    sinex
        .prepare_input_group(
            &CapturePlan {
                required_disposition: None,
                intent: intent.into(),
                bases: vec![basis.clone()],
                count: ranges.len() as u64,
                mode: CaptureMode::Mutable,
            },
            &mut |emit| {
                for (ordinal, range) in ranges.iter().copied().enumerate() {
                    let bytes = vec![b'A' + (ordinal % 26) as u8; range.length()? as usize];
                    let mut input = capture(&bytes);
                    // Logical NoExtent still owns the actual provider body.
                    input.rows.extents.clear();
                    input.rows.input = generation.stream().input.id.clone();
                    input.raw_range = Some(range);
                    input.raw_predecessor = (ordinal == 0).then(|| predecessor.clone()).flatten();
                    input.generation = Some(generation.clone());
                    input.raw_total = (range.end == generation.window().end).then_some(range.end);
                    input.basis = Some(basis.clone());
                    emit(input, vec![body(root, &bytes)])?;
                }
                Ok(())
            },
        )
        .unwrap()
}

#[test]
fn offline_raw_cursor_checks_many_bodies_and_restarts_without_source_or_parser_authority() {
    let root = tempfile::tempdir().unwrap();
    let generation = generation(35);
    let sinex = client(root.path());
    let ranges: Vec<_> = (0..35)
        .map(|start| RawRange {
            start,
            end: start + 1,
        })
        .collect();
    let CaptureReceipt::Accepted { write } =
        acquire_ranges(&sinex, root.path(), &generation, "whole-sample", &ranges)
    else {
        panic!()
    };
    let cursor = sinex
        .raw_acquisition_cursor(&polylogue_store::primary::RawAcquisitionQuery {
            generation: generation.clone(),
            anchor: None,
        })
        .unwrap()
        .unwrap();
    assert_eq!(cursor.range(), RawRange { start: 0, end: 35 });
    assert!(cursor.is_complete_window());
    assert_eq!(cursor.disposition(), Some(CaptureDisposition::Prepared));
    let last = sinex
        .pending_group_headers(&write.identity, Some(33), 1)
        .unwrap();
    assert_eq!(cursor.last_body(), last.headers[0].identity);
    let Capture::Input(header) = &last.headers[0].captured else {
        panic!()
    };
    assert_eq!(header.generation.as_ref(), Some(&generation));
    assert!(last.headers[0].material.is_none());
    assert!(
        sinex
            .queue()
            .unwrap()
            .pending_after(None, 64)
            .unwrap()
            .iter()
            .all(|entry| entry.phase.is_pending())
    );
    drop(sinex);
    let reopened = client(root.path());
    assert_eq!(
        reopened
            .raw_acquisition_cursor(&polylogue_store::primary::RawAcquisitionQuery {
                generation: generation.clone(),
                anchor: None
            })
            .unwrap(),
        Some(cursor)
    );
}

#[test]
fn unresolved_siblings_keep_actual_predecessor_body_refs_and_only_raw_custody_progress() {
    let root = tempfile::tempdir().unwrap();
    let generation = generation(3);
    let sinex = client(root.path());
    tests::acquire(
        &sinex,
        root.path(),
        b"winner",
        basis(&generation.stream().source, None),
        CaptureMode::Mutable,
    )
    .unwrap();
    let accepted_head = sinex
        .pending_source(&generation.stream().source)
        .unwrap()
        .head
        .unwrap()
        .identity;
    let CaptureReceipt::UnresolvedBasisConflict { custody } = acquire_ranges(
        &sinex,
        root.path(),
        &generation,
        "stale-observation",
        &[
            RawRange { start: 0, end: 1 },
            RawRange { start: 1, end: 2 },
            RawRange { start: 2, end: 3 },
        ],
    ) else {
        panic!()
    };
    let page = sinex
        .pending_group_headers(&custody.identity, None, 3)
        .unwrap();
    for pair in page.headers.windows(2) {
        let Capture::Input(next) = &pair[1].captured else {
            panic!()
        };
        assert_eq!(next.raw_predecessor.as_ref(), Some(&pair[0].identity));
    }
    let cursor = sinex
        .raw_acquisition_cursor(&polylogue_store::primary::RawAcquisitionQuery {
            generation: generation.clone(),
            anchor: None,
        })
        .unwrap()
        .unwrap();
    assert!(cursor.is_complete_window());
    assert_eq!(
        cursor.disposition(),
        Some(CaptureDisposition::UnresolvedBasisConflict)
    );
    assert_eq!(cursor.last_body(), page.headers[2].identity);
    assert_eq!(
        sinex
            .pending_source(&generation.stream().source)
            .unwrap()
            .head
            .unwrap()
            .identity,
        accepted_head
    );
    assert_eq!(sinex.pending_captures().unwrap().len(), 1);
    drop(sinex);
    assert_eq!(
        client(root.path())
            .raw_acquisition_cursor(&polylogue_store::primary::RawAcquisitionQuery {
                generation: generation.clone(),
                anchor: None
            })
            .unwrap(),
        Some(cursor)
    );
}

#[test]
fn raw_cursor_refuses_holes_other_generations_and_corrupt_committed_bodies() {
    let root = tempfile::tempdir().unwrap();
    let generation = generation(3);
    let sinex = client(root.path());
    acquire_ranges(
        &sinex,
        root.path(),
        &generation,
        "hole",
        &[RawRange { start: 0, end: 1 }, RawRange { start: 2, end: 3 }],
    );
    assert!(
        sinex
            .raw_acquisition_cursor(&polylogue_store::primary::RawAcquisitionQuery {
                generation: generation.clone(),
                anchor: None
            })
            .unwrap()
            .is_none()
    );
    let mut changed = generation.clone();
    let SourceGeneration::File { observed, .. } = &mut changed else {
        unreachable!()
    };
    observed.changed.nanos += 1;
    assert!(
        sinex
            .raw_acquisition_cursor(&polylogue_store::primary::RawAcquisitionQuery {
                generation: changed.clone(),
                anchor: None
            })
            .unwrap()
            .is_none()
    );

    let complete_root = tempfile::tempdir().unwrap();
    let complete = client(complete_root.path());
    acquire_ranges(
        &complete,
        complete_root.path(),
        &generation,
        "complete",
        &[RawRange { start: 0, end: 3 }],
    );
    let cursor = complete
        .raw_acquisition_cursor(&polylogue_store::primary::RawAcquisitionQuery {
            generation: generation.clone(),
            anchor: None,
        })
        .unwrap()
        .unwrap();
    let (path, _) = complete
        .queue()
        .unwrap()
        .encoded_body(cursor.last_body())
        .unwrap()
        .remove(0);
    let mut file = std::fs::OpenOptions::new().write(true).open(path).unwrap();
    std::io::Write::write_all(&mut file, b"corrupt").unwrap();
    file.sync_all().unwrap();
    assert!(
        complete
            .raw_acquisition_cursor(&polylogue_store::primary::RawAcquisitionQuery {
                generation: generation.clone(),
                anchor: None
            })
            .is_err()
    );
}

fn first_body(sinex: &Sinex, acknowledgment: CaptureReceipt) -> String {
    let group = match acknowledgment {
        CaptureReceipt::Accepted { write } => write.identity,
        CaptureReceipt::UnresolvedBasisConflict { custody } => custody.identity,
    };
    sinex
        .pending_group_headers(&group, None, 1)
        .unwrap()
        .headers
        .remove(0)
        .identity
}

#[test]
fn explicit_body_anchor_keeps_equal_generation_observations_and_dispositions_separate_after_restart()
 {
    let root = tempfile::tempdir().unwrap();
    let generation = generation(3);
    let sinex = client(root.path());
    tests::acquire(
        &sinex,
        root.path(),
        b"winner",
        basis(&generation.stream().source, None),
        CaptureMode::Mutable,
    )
    .unwrap();
    let head = sinex
        .pending_source(&generation.stream().source)
        .unwrap()
        .head
        .unwrap()
        .identity;
    let old = first_body(
        &sinex,
        acquire_ranges(
            &sinex,
            root.path(),
            &generation,
            "old-unresolved",
            &[RawRange { start: 0, end: 3 }],
        ),
    );
    let new = first_body(
        &sinex,
        acquire_ranges_on(
            &sinex,
            root.path(),
            &generation,
            "new-actual",
            &[RawRange { start: 0, end: 3 }],
            basis(&generation.stream().source, Some(head)),
            None,
        ),
    );
    assert!(
        sinex
            .raw_acquisition_cursor(&polylogue_store::primary::RawAcquisitionQuery {
                generation: generation.clone(),
                anchor: None
            })
            .unwrap()
            .is_none()
    );
    let mut selected = Vec::new();
    for (anchor, disposition) in [
        (old, CaptureDisposition::UnresolvedBasisConflict),
        (new, CaptureDisposition::Prepared),
    ] {
        let query = polylogue_store::primary::RawAcquisitionQuery {
            generation: generation.clone(),
            anchor: Some(anchor.clone()),
        };
        let cursor = sinex.raw_acquisition_cursor(&query).unwrap().unwrap();
        assert!(cursor.is_complete_window());
        assert_eq!(cursor.root_body(), anchor);
        assert_eq!(cursor.last_body(), anchor);
        assert_eq!(cursor.disposition(), Some(disposition));
        selected.push((query, cursor));
    }
    drop(sinex);
    let reopened = client(root.path());
    for (query, cursor) in selected {
        assert_eq!(
            reopened.raw_acquisition_cursor(&query).unwrap(),
            Some(cursor)
        );
    }
    let query = polylogue_store::primary::RawAcquisitionQuery {
        generation,
        anchor: None,
    };
    assert!(reopened.raw_acquisition_cursor(&query).unwrap().is_none());
}

#[test]
fn an_explicit_tail_anchor_pins_its_branch_and_never_chooses_a_fork_by_queue_order() {
    let root = tempfile::tempdir().unwrap();
    let generation = generation(3);
    let sinex = client(root.path());
    let start = first_body(
        &sinex,
        acquire_ranges(
            &sinex,
            root.path(),
            &generation,
            "start",
            &[RawRange { start: 0, end: 1 }],
        ),
    );
    let left = first_body(
        &sinex,
        acquire_ranges_on(
            &sinex,
            root.path(),
            &generation,
            "left",
            &[RawRange { start: 1, end: 2 }],
            basis(&generation.stream().source, Some(start.clone())),
            Some(start.clone()),
        ),
    );
    let _right = first_body(
        &sinex,
        acquire_ranges_on(
            &sinex,
            root.path(),
            &generation,
            "right",
            &[RawRange { start: 1, end: 2 }],
            basis(&generation.stream().source, Some(start.clone())),
            Some(start.clone()),
        ),
    );
    let end = first_body(
        &sinex,
        acquire_ranges_on(
            &sinex,
            root.path(),
            &generation,
            "end",
            &[RawRange { start: 2, end: 3 }],
            basis(&generation.stream().source, Some(left.clone())),
            Some(left.clone()),
        ),
    );
    let ambiguous = polylogue_store::primary::RawAcquisitionQuery {
        generation: generation.clone(),
        anchor: Some(start.clone()),
    };
    assert!(sinex.raw_acquisition_cursor(&ambiguous).unwrap().is_none());
    let selected = polylogue_store::primary::RawAcquisitionQuery {
        generation: generation.clone(),
        anchor: Some(left),
    };
    let cursor = sinex.raw_acquisition_cursor(&selected).unwrap().unwrap();
    assert_eq!(cursor.root_body(), start);
    assert_eq!(cursor.last_body(), end);
    assert!(cursor.is_complete_window());
    assert_eq!(cursor.disposition(), Some(CaptureDisposition::Prepared));
    drop(sinex);
    assert_eq!(
        client(root.path())
            .raw_acquisition_cursor(&selected)
            .unwrap(),
        Some(cursor)
    );
}
