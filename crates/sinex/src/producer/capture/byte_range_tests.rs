use super::{
    tests::{source, source_base},
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

fn capture_ranges(
    sinex: &Sinex,
    root: &std::path::Path,
    generation: &SourceGeneration,
    intent: &str,
    ranges: &[RawRange],
) -> CaptureResult {
    capture_ranges_on(
        sinex,
        root,
        generation,
        intent,
        ranges,
        source_base(&generation.stream().source, None),
        None,
    )
}

fn capture_ranges_on(
    sinex: &Sinex,
    root: &std::path::Path,
    generation: &SourceGeneration,
    intent: &str,
    ranges: &[RawRange],
    base: polylogue_store::primary::SourceBasis,
    previous: Option<String>,
) -> CaptureResult {
    sinex
        .prepare_input_group(
            &CapturePlan {
                required_outcome: None,
                intent: intent.into(),
                bases: vec![base.clone()],
                count: ranges.len() as u64,
                mode: CaptureMode::Mutable,
            },
            &mut |emit| {
                for (ordinal, range) in ranges.iter().copied().enumerate() {
                    let bytes = vec![b'A' + (ordinal % 26) as u8; range.length()? as usize];
                    let mut input = capture(&bytes);
                    // A capture with no extents still carries the provider body.
                    input.rows.extents.clear();
                    input.rows.input = generation.stream().input.id.clone();
                    input.raw_range = Some(range);
                    input.raw_predecessor = (ordinal == 0).then(|| previous.clone()).flatten();
                    input.generation = Some(generation.clone());
                    input.raw_total = (range.end == generation.window().end).then_some(range.end);
                    input.basis = Some(base.clone());
                    emit(input, vec![body(root, &bytes)])?;
                }
                Ok(())
            },
        )
        .unwrap()
}

#[test]
fn offline_byte_cursor_checks_many_bodies_and_restarts() {
    let root = tempfile::tempdir().unwrap();
    let generation = generation(35);
    let sinex = client(root.path());
    let ranges: Vec<_> = (0..35)
        .map(|start| RawRange {
            start,
            end: start + 1,
        })
        .collect();
    let CaptureResult::Accepted { write } =
        capture_ranges(&sinex, root.path(), &generation, "whole-sample", &ranges)
    else {
        panic!()
    };
    let cursor = sinex
        .byte_cursor(&polylogue_store::primary::ByteCursorQuery {
            generation: generation.clone(),
            tail_body: None,
        })
        .unwrap()
        .unwrap();
    assert_eq!(cursor.range(), RawRange { start: 0, end: 35 });
    assert!(cursor.is_complete_window());
    assert_eq!(cursor.outcome(), Some(CaptureOutcome::Prepared));
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
            .byte_cursor(&polylogue_store::primary::ByteCursorQuery {
                generation: generation.clone(),
                tail_body: None
            })
            .unwrap(),
        Some(cursor)
    );
}

#[test]
fn unresolved_siblings_keep_previous_body_refs() {
    let root = tempfile::tempdir().unwrap();
    let generation = generation(3);
    let sinex = client(root.path());
    tests::capture_bytes(
        &sinex,
        root.path(),
        b"winner",
        source_base(&generation.stream().source, None),
        CaptureMode::Mutable,
    )
    .unwrap();
    let accepted_latest = sinex
        .pending_source(&generation.stream().source)
        .unwrap()
        .head
        .unwrap()
        .identity;
    let CaptureResult::UnresolvedBasisConflict { spooled } = capture_ranges(
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
        .pending_group_headers(&spooled.identity, None, 3)
        .unwrap();
    for pair in page.headers.windows(2) {
        let Capture::Input(next) = &pair[1].captured else {
            panic!()
        };
        assert_eq!(next.raw_predecessor.as_ref(), Some(&pair[0].identity));
    }
    let cursor = sinex
        .byte_cursor(&polylogue_store::primary::ByteCursorQuery {
            generation: generation.clone(),
            tail_body: None,
        })
        .unwrap()
        .unwrap();
    assert!(cursor.is_complete_window());
    assert_eq!(
        cursor.outcome(),
        Some(CaptureOutcome::UnresolvedBasisConflict)
    );
    assert_eq!(cursor.last_body(), page.headers[2].identity);
    assert_eq!(
        sinex
            .pending_source(&generation.stream().source)
            .unwrap()
            .head
            .unwrap()
            .identity,
        accepted_latest
    );
    assert_eq!(sinex.pending_captures().unwrap().len(), 1);
    drop(sinex);
    assert_eq!(
        client(root.path())
            .byte_cursor(&polylogue_store::primary::ByteCursorQuery {
                generation: generation.clone(),
                tail_body: None
            })
            .unwrap(),
        Some(cursor)
    );
}

#[test]
fn byte_cursor_refuses_holes_changes_and_corrupt_bodies() {
    let root = tempfile::tempdir().unwrap();
    let generation = generation(3);
    let sinex = client(root.path());
    capture_ranges(
        &sinex,
        root.path(),
        &generation,
        "hole",
        &[RawRange { start: 0, end: 1 }, RawRange { start: 2, end: 3 }],
    );
    assert!(
        sinex
            .byte_cursor(&polylogue_store::primary::ByteCursorQuery {
                generation: generation.clone(),
                tail_body: None
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
            .byte_cursor(&polylogue_store::primary::ByteCursorQuery {
                generation: changed.clone(),
                tail_body: None
            })
            .unwrap()
            .is_none()
    );

    let complete_root = tempfile::tempdir().unwrap();
    let complete = client(complete_root.path());
    capture_ranges(
        &complete,
        complete_root.path(),
        &generation,
        "complete",
        &[RawRange { start: 0, end: 3 }],
    );
    let cursor = complete
        .byte_cursor(&polylogue_store::primary::ByteCursorQuery {
            generation: generation.clone(),
            tail_body: None,
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
            .byte_cursor(&polylogue_store::primary::ByteCursorQuery {
                generation: generation.clone(),
                tail_body: None
            })
            .is_err()
    );
}

fn first_body(sinex: &Sinex, result: CaptureResult) -> String {
    let group = match result {
        CaptureResult::Accepted { write } => write.identity,
        CaptureResult::UnresolvedBasisConflict { spooled } => spooled.identity,
    };
    sinex
        .pending_group_headers(&group, None, 1)
        .unwrap()
        .headers
        .remove(0)
        .identity
}

#[test]
fn tail_body_keeps_equal_reads_apart_after_restart() {
    let root = tempfile::tempdir().unwrap();
    let generation = generation(3);
    let sinex = client(root.path());
    tests::capture_bytes(
        &sinex,
        root.path(),
        b"winner",
        source_base(&generation.stream().source, None),
        CaptureMode::Mutable,
    )
    .unwrap();
    let latest = sinex
        .pending_source(&generation.stream().source)
        .unwrap()
        .head
        .unwrap()
        .identity;
    let old = first_body(
        &sinex,
        capture_ranges(
            &sinex,
            root.path(),
            &generation,
            "old-unresolved",
            &[RawRange { start: 0, end: 3 }],
        ),
    );
    let new = first_body(
        &sinex,
        capture_ranges_on(
            &sinex,
            root.path(),
            &generation,
            "new-current",
            &[RawRange { start: 0, end: 3 }],
            source_base(&generation.stream().source, Some(latest)),
            None,
        ),
    );
    assert!(
        sinex
            .byte_cursor(&polylogue_store::primary::ByteCursorQuery {
                generation: generation.clone(),
                tail_body: None
            })
            .unwrap()
            .is_none()
    );
    let mut selected = Vec::new();
    for (tail, outcome) in [
        (old, CaptureOutcome::UnresolvedBasisConflict),
        (new, CaptureOutcome::Prepared),
    ] {
        let query = polylogue_store::primary::ByteCursorQuery {
            generation: generation.clone(),
            tail_body: Some(tail.clone()),
        };
        let cursor = sinex.byte_cursor(&query).unwrap().unwrap();
        assert!(cursor.is_complete_window());
        assert_eq!(cursor.root_body(), tail);
        assert_eq!(cursor.last_body(), tail);
        assert_eq!(cursor.outcome(), Some(outcome));
        selected.push((query, cursor));
    }
    drop(sinex);
    let reopened = client(root.path());
    for (query, cursor) in selected {
        assert_eq!(reopened.byte_cursor(&query).unwrap(), Some(cursor));
    }
    let query = polylogue_store::primary::ByteCursorQuery {
        generation,
        tail_body: None,
    };
    assert!(reopened.byte_cursor(&query).unwrap().is_none());
}

#[test]
fn explicit_tail_body_pins_its_branch() {
    let root = tempfile::tempdir().unwrap();
    let generation = generation(3);
    let sinex = client(root.path());
    let start = first_body(
        &sinex,
        capture_ranges(
            &sinex,
            root.path(),
            &generation,
            "start",
            &[RawRange { start: 0, end: 1 }],
        ),
    );
    let left = first_body(
        &sinex,
        capture_ranges_on(
            &sinex,
            root.path(),
            &generation,
            "left",
            &[RawRange { start: 1, end: 2 }],
            source_base(&generation.stream().source, Some(start.clone())),
            Some(start.clone()),
        ),
    );
    let _right = first_body(
        &sinex,
        capture_ranges_on(
            &sinex,
            root.path(),
            &generation,
            "right",
            &[RawRange { start: 1, end: 2 }],
            source_base(&generation.stream().source, Some(start.clone())),
            Some(start.clone()),
        ),
    );
    let end = first_body(
        &sinex,
        capture_ranges_on(
            &sinex,
            root.path(),
            &generation,
            "end",
            &[RawRange { start: 2, end: 3 }],
            source_base(&generation.stream().source, Some(left.clone())),
            Some(left.clone()),
        ),
    );
    let ambiguous = polylogue_store::primary::ByteCursorQuery {
        generation: generation.clone(),
        tail_body: Some(start.clone()),
    };
    assert!(sinex.byte_cursor(&ambiguous).unwrap().is_none());
    let selected = polylogue_store::primary::ByteCursorQuery {
        generation: generation.clone(),
        tail_body: Some(left),
    };
    let cursor = sinex.byte_cursor(&selected).unwrap().unwrap();
    assert_eq!(cursor.root_body(), start);
    assert_eq!(cursor.last_body(), end);
    assert!(cursor.is_complete_window());
    assert_eq!(cursor.outcome(), Some(CaptureOutcome::Prepared));
    drop(sinex);
    assert_eq!(
        client(root.path()).byte_cursor(&selected).unwrap(),
        Some(cursor)
    );
}
