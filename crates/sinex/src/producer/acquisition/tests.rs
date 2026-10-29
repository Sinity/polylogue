use super::*;
use crate::tests::{body, capture, client};
use polylogue_store::primary::{RawRange, SourceBasis};

fn source(path: &[u8]) -> SourceSelector {
    SourceSelector {
        origin: "codex".into(),
        machine: "fixture-machine".into(),
        path_bytes: path.into(),
    }
}
fn basis(source: &SourceSelector, pending: Option<String>) -> SourceBasis {
    SourceBasis {
        source: source.clone(),
        status: SourceStatus::Ready,
        confirmed: None,
        pending,
    }
}
fn acquire(
    sinex: &Sinex,
    root: &std::path::Path,
    bytes: &[u8],
    basis: SourceBasis,
    mode: CaptureMode,
) -> Result<PreparedWrite> {
    let plan = CapturePlan {
        intent: blake3::hash(bytes).to_hex().to_string(),
        bases: vec![basis.clone()],
        count: 1,
        mode,
    };
    sinex.prepare_input_group(&plan, &mut |emit| {
        let mut input = capture(bytes, Vec::new());
        input.raw_range = Some(RawRange {
            start: 0,
            end: bytes.len() as u64,
        });
        input.basis = Some(basis.clone());
        emit(input, body(root, bytes))
    })
}

#[test]
fn atomic_source_cas_preserves_a_b_a_retry_restart_and_byte_identity() {
    let root = tempfile::tempdir().unwrap();
    let selector = source(b"/home/op/session-\xff.jsonl");
    let sinex = client(root.path());
    let first = acquire(
        &sinex,
        root.path(),
        b"A",
        basis(&selector, None),
        CaptureMode::Mutable,
    )
    .unwrap();
    let first_head = sinex
        .pending_source(&selector, "reading")
        .unwrap()
        .head
        .unwrap()
        .identity;
    assert_eq!(
        acquire(
            &sinex,
            root.path(),
            b"A",
            basis(&selector, None),
            CaptureMode::Mutable
        )
        .unwrap()
        .identity,
        first.identity
    );
    assert!(
        acquire(
            &sinex,
            root.path(),
            b"B",
            basis(&selector, None),
            CaptureMode::Mutable
        )
        .is_err()
    );
    acquire(
        &sinex,
        root.path(),
        b"B",
        basis(&selector, Some(first_head)),
        CaptureMode::Mutable,
    )
    .unwrap();
    let second_head = sinex
        .pending_source(&selector, "reading")
        .unwrap()
        .head
        .unwrap()
        .identity;
    let third = acquire(
        &sinex,
        root.path(),
        b"A",
        basis(&selector, Some(second_head)),
        CaptureMode::Mutable,
    )
    .unwrap();
    assert_ne!(first.identity, third.identity);
    assert!(
        sinex
            .pending_source(&source(b"/home/op/session-\xfe.jsonl"), "reading")
            .unwrap()
            .head
            .is_none()
    );
    drop(sinex);
    let sinex = client(root.path());
    let state = sinex.pending_source(&selector, "reading").unwrap();
    let head = state.head.unwrap();
    assert_eq!(state.status, SourceStatus::Ready);
    assert_eq!(
        acquire(
            &sinex,
            root.path(),
            b"A",
            basis(&selector, Some(head.identity.clone())),
            CaptureMode::Mutable
        )
        .unwrap()
        .identity,
        third.identity
    );
    assert_eq!(sinex.pending_captures().unwrap().len(), 3);
    let body = sinex.capture_body(&head.identity).unwrap();
    let prepared = hashed_blobs::Prepared::from_encoded(
        body.file,
        body.descriptor,
        None,
        hashed_blobs::ReadLimits::default(),
    )
    .unwrap();
    let mut plain = Vec::new();
    std::io::Read::read_to_end(
        &mut prepared
            .reader(None, hashed_blobs::ReadLimits::default())
            .unwrap(),
        &mut plain,
    )
    .unwrap();
    assert_eq!(plain, b"A");
}

#[test]
fn failed_group_is_not_visible_and_immutable_imports_keep_content_identity() {
    let root = tempfile::tempdir().unwrap();
    let sinex = client(root.path());
    let selector = source(b"import");
    let plan = CapturePlan {
        intent: "two-members".into(),
        bases: vec![basis(&selector, None)],
        count: 2,
        mode: CaptureMode::Immutable,
    };
    assert!(
        sinex
            .prepare_input_group(&plan, &mut |emit| {
                let mut input = capture(b"first", Vec::new());
                input.basis = Some(basis(&selector, None));
                emit(input, body(root.path(), b"first"))?;
                Err(polylogue_store::StoreError::Invalid(
                    "second member unavailable".into(),
                ))
            })
            .is_err()
    );
    assert!(sinex.pending_captures().unwrap().is_empty());
    let first = acquire(
        &sinex,
        root.path(),
        b"A",
        basis(&selector, None),
        CaptureMode::Immutable,
    )
    .unwrap();
    acquire(
        &sinex,
        root.path(),
        b"B",
        basis(&selector, None),
        CaptureMode::Immutable,
    )
    .unwrap();
    assert_eq!(
        acquire(
            &sinex,
            root.path(),
            b"A",
            basis(&selector, None),
            CaptureMode::Immutable
        )
        .unwrap()
        .identity,
        first.identity
    );
    assert_eq!(sinex.pending_captures().unwrap().len(), 2);
}

#[test]
fn older_control_capture_is_explicit_when_newer_raw_has_no_parser_state() {
    let root = tempfile::tempdir().unwrap();
    let sinex = client(root.path());
    let selector = source(b"source");
    acquire(
        &sinex,
        root.path(),
        b"A",
        basis(&selector, None),
        CaptureMode::Mutable,
    )
    .unwrap();
    let prior = sinex
        .pending_source(&selector, "v2")
        .unwrap()
        .head
        .unwrap()
        .identity;
    let mut step = polylogue_store::ParseStep::new(
        polylogue_model::Output::default(),
        polylogue_model::RecordCounts::default(),
        0..1,
        true,
    );
    step.checkpoint = Some(polylogue_model::batch::ParserCheckpoint::new(
        "v2".into(),
        1,
        Some(serde_json::json!({"state":"original"})),
    ));
    step.prepare();
    let reading = polylogue_store::primary::CapturedReading::from_step(&step).unwrap();
    let control_basis = basis(&selector, Some(prior));
    sinex
        .prepare_input_group(
            &CapturePlan {
                intent: "control".into(),
                bases: vec![control_basis.clone()],
                count: 1,
                mode: CaptureMode::Mutable,
            },
            &mut |emit| {
                let mut input = capture(b"{}", vec![reading.clone()]);
                input.rows.extent = None;
                input.basis = Some(control_basis.clone());
                emit(input, body(root.path(), b"{}"))
            },
        )
        .unwrap();
    let control_id = sinex
        .pending_source(&selector, "v2")
        .unwrap()
        .head
        .unwrap()
        .identity;
    acquire(
        &sinex,
        root.path(),
        b"B",
        basis(&selector, Some(control_id.clone())),
        CaptureMode::Mutable,
    )
    .unwrap();
    drop(sinex);
    let sinex = client(root.path());
    let state = sinex.pending_source(&selector, "v2").unwrap();
    assert_ne!(state.head.as_ref().unwrap().identity, control_id);
    assert_eq!(state.raw.unwrap().identity, state.head.unwrap().identity);
    let continuation = state.continuation.unwrap();
    assert_eq!(continuation.capture, control_id);
    assert!(continuation.header.parsed.is_empty());
    assert_eq!(
        continuation.checkpoint.parser_state,
        Some(serde_json::json!({"state":"original"}))
    );
    assert!(
        sinex
            .pending_source(&selector, "v3")
            .unwrap()
            .continuation
            .is_none()
    );
}
