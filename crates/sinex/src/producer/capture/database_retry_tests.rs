use super::tests::{accepted, source, source_base};
use super::*;
use crate::tests::{body, capture, client};
use polylogue_store::primary::{
    DatabaseSnapshotToken, RawRange, RawStream, RawStreamRole, SourceGeneration,
};

fn database(bytes: &[u8]) -> CapturedInput {
    let mut input = capture(bytes);
    let window = input.raw_range.unwrap();
    input.generation = Some(SourceGeneration::Database {
        stream: RawStream {
            source: source(b"/home/op/state_5.sqlite"),
            input: polylogue_store::InputKey {
                id: input.rows.input.clone(),
                format: input.rows.format.clone(),
                route: input.route,
            },
            role: RawStreamRole::DatabaseDump,
            namespace: None,
        },
        snapshot: DatabaseSnapshotToken {
            nonce: uuid::Uuid::now_v7().to_string(),
        },
        window,
    });
    input.raw_total = Some(window.end);
    input
}

fn prepare(
    sdk: &Sinex,
    root: &std::path::Path,
    mut input: CapturedInput,
    bytes: &[u8],
    base: polylogue_store::primary::SourceBasis,
) -> CaptureResult {
    input.basis = Some(base.clone());
    sdk.prepare_input_group(
        &CapturePlan {
            intent: blake3::hash(bytes).to_hex().to_string(),
            bases: vec![base],
            count: 1,
            mode: CaptureMode::Mutable,
            required_outcome: None,
        },
        &mut |emit| emit(input.clone(), vec![body(root, bytes)]),
    )
    .unwrap()
}

#[test]
fn a_database_pending_retry_keeps_the_exact_original_nonce_and_body_after_reopen() {
    let root = tempfile::tempdir().unwrap();
    let sdk = client(root.path());
    let input = database(b"A");
    let source = input.generation.as_ref().unwrap().stream().source.clone();
    let first = accepted(prepare(
        &sdk,
        root.path(),
        input.clone(),
        b"A",
        source_base(&source, None),
    ));
    let original = sdk.pending_source(&source).unwrap().head.unwrap();
    drop(sdk);
    let sdk = client(root.path());
    let fresh = database(b"A");
    assert_ne!(fresh.generation, input.generation);
    let retry = accepted(prepare(
        &sdk,
        root.path(),
        fresh.clone(),
        b"A",
        source_base(&source, Some(original.identity.clone())),
    ));
    assert_eq!(retry.identity, first.identity);
    assert_eq!(
        sdk.pending_source(&source).unwrap().head,
        Some(original.clone())
    );
    assert_eq!(sdk.pending_captures().unwrap(), vec![original]);
    // The new read's nonce was not silently made equivalent for ordinary
    // byte-cursor lookup; only the original pending record was reused.
    assert!(
        sdk.byte_cursor(&polylogue_store::primary::ByteCursorQuery {
            generation: fresh.generation.unwrap(),
            tail_body: None,
        })
        .unwrap()
        .is_none()
    );
}

#[test]
fn database_retries_do_not_hide_stream_window_or_stale_source_changes() {
    for change in ["stream", "window", "base"] {
        let root = tempfile::tempdir().unwrap();
        let sdk = client(root.path());
        let mut original = database(b"A");
        if change == "window" {
            original.raw_total = None;
        }
        let source = original
            .generation
            .as_ref()
            .unwrap()
            .stream()
            .source
            .clone();
        accepted(prepare(
            &sdk,
            root.path(),
            original.clone(),
            b"A",
            source_base(&source, None),
        ));
        let first = sdk.pending_source(&source).unwrap().head.unwrap();
        let mut offered = database(b"A");
        let mut base = source_base(&source, Some(first.identity.clone()));
        let mut bytes = b"A".as_slice();
        match change {
            "stream" => {
                let Some(SourceGeneration::Database { stream, .. }) = &mut offered.generation
                else {
                    panic!()
                };
                stream.namespace = Some("other-stream".into());
            }
            "window" => {
                let Some(SourceGeneration::Database { window, .. }) = &mut offered.generation
                else {
                    panic!()
                };
                *window = RawRange { start: 0, end: 2 };
                offered.raw_total = None;
            }
            "base" => {
                // This B read never acquired a source base after A. Its bytes
                // are retained, but it cannot become the next source operation.
                offered = database(b"B");
                bytes = b"B";
                base.pending = None;
            }
            _ => unreachable!(),
        }
        let expected_generation = offered.generation.clone();
        let CaptureResult::UnresolvedBasisConflict { spooled } =
            prepare(&sdk, root.path(), offered, bytes, base)
        else {
            panic!("{change} must retain bytes without accepting the changed read")
        };
        assert_eq!(sdk.pending_source(&source).unwrap().head, Some(first));
        let page = sdk
            .pending_group_headers(&spooled.identity, None, 1)
            .unwrap();
        assert_eq!(page.outcome, Some(CaptureOutcome::UnresolvedBasisConflict));
        let Capture::Input(kept) = &page.headers[0].captured else {
            panic!()
        };
        assert_eq!(kept.generation, expected_generation);
        assert_eq!(kept.hash, BlobHash::of(bytes));
        let mut actual = Vec::new();
        for part in sdk.capture_body(&page.headers[0].identity).unwrap() {
            std::io::Read::read_to_end(
                &mut part
                    .descriptor
                    .open(part.file, None, Default::default())
                    .unwrap(),
                &mut actual,
            )
            .unwrap();
        }
        assert_eq!(actual, bytes);
    }
}

#[test]
fn a_database_changed_then_returned_is_a_new_observation_but_its_repeat_is_not() {
    let root = tempfile::tempdir().unwrap();
    let sdk = client(root.path());
    let source = source(b"/home/op/state_5.sqlite");
    let mut groups = Vec::new();
    for bytes in [b"A".as_slice(), b"B", b"A", b"A"] {
        let base = source_base(
            &source,
            sdk.pending_source(&source)
                .unwrap()
                .head
                .map(|head| head.identity),
        );
        groups.push(accepted(prepare(&sdk, root.path(), database(bytes), bytes, base)).identity);
    }
    assert_ne!(groups[0], groups[2]);
    assert_eq!(groups[2], groups[3]);
    let pending = sdk.pending_captures().unwrap();
    assert_eq!(pending.len(), 3);
    let hashes: Vec<_> = pending
        .iter()
        .map(|pending| {
            let Capture::Input(input) = &pending.captured else {
                panic!()
            };
            input.hash
        })
        .collect();
    assert_eq!(
        hashes,
        [BlobHash::of(b"A"), BlobHash::of(b"B"), BlobHash::of(b"A")]
    );
}
