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
                arrival_rank: 0,
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
    let basis = basis(&generation.stream().source, None);
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
                    let mut input = capture(&bytes, vec![]);
                    // Logical NoExtent still owns the actual provider body.
                    input.rows.extent = None;
                    input.raw_range = Some(range);
                    input.generation = Some(generation.clone());
                    input.raw_total = (range.end == generation.window().end).then_some(range.end);
                    input.basis = Some(basis.clone());
                    emit(input, body(root, &bytes))?;
                }
                Ok(())
            },
        )
        .unwrap()
}

#[test]
fn offline_raw_cursor_spans_pages_and_restarts_without_source_or_parser_authority() {
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
    let cursor = sinex.raw_acquisition_cursor(&generation).unwrap().unwrap();
    assert_eq!(cursor.range(), RawRange { start: 0, end: 35 });
    assert!(cursor.is_complete_stream());
    let last = sinex
        .pending_group_headers(&write.identity, Some(33), 1)
        .unwrap();
    assert_eq!(cursor.last_body(), last.headers[0].identity);
    let Capture::Input(header) = &last.headers[0].captured else {
        panic!()
    };
    assert!(header.parsed.is_empty());
    assert_eq!(header.generation.as_ref(), Some(&generation));
    assert!(last.headers[0].material.is_none());
    assert!(
        sinex
            .queue()
            .unwrap()
            .pending_after(None, 64)
            .unwrap()
            .iter()
            .all(|entry| !entry.acknowledged)
    );
    drop(sinex);
    let reopened = client(root.path());
    assert_eq!(
        reopened.raw_acquisition_cursor(&generation).unwrap(),
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
        .pending_source(&generation.stream().source, "v")
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
    let cursor = sinex.raw_acquisition_cursor(&generation).unwrap().unwrap();
    assert!(cursor.is_complete_stream());
    assert_eq!(cursor.last_body(), page.headers[2].identity);
    assert_eq!(
        sinex
            .pending_source(&generation.stream().source, "v")
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
            .raw_acquisition_cursor(&generation)
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
    assert!(sinex.raw_acquisition_cursor(&generation).unwrap().is_none());
    let mut changed = generation.clone();
    let SourceGeneration::File { observed, .. } = &mut changed else {
        unreachable!()
    };
    observed.changed.nanos += 1;
    assert!(sinex.raw_acquisition_cursor(&changed).unwrap().is_none());

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
        .raw_acquisition_cursor(&generation)
        .unwrap()
        .unwrap();
    let (path, _) = complete
        .queue()
        .unwrap()
        .encoded_body(cursor.last_body())
        .unwrap();
    let mut file = std::fs::OpenOptions::new().write(true).open(path).unwrap();
    std::io::Write::write_all(&mut file, b"corrupt").unwrap();
    file.sync_all().unwrap();
    assert!(complete.raw_acquisition_cursor(&generation).is_err());
}
