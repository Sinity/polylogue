use super::*;
use crate::tests::{body, capture, client};
use polylogue_store::primary::{RawRange, SourceBasis};

pub(super) fn source(path: &[u8]) -> SourceSelector {
    SourceSelector {
        origin: "codex".into(),
        machine: "fixture-machine".into(),
        path_bytes: path.into(),
    }
}
pub(super) fn source_base(source: &SourceSelector, pending: Option<String>) -> SourceBasis {
    SourceBasis {
        boundary: None,
        resolved: None,
        source: source.clone(),
        status: SourceStatus::MissingEvidence,
        confirmed: None,
        pending,
    }
}
pub(super) fn capture_bytes(
    sinex: &Sinex,
    root: &std::path::Path,
    bytes: &[u8],
    base: SourceBasis,
    mode: CaptureMode,
) -> Result<PreparedWrite> {
    let plan = CapturePlan {
        required_outcome: None,
        intent: blake3::hash(bytes).to_hex().to_string(),
        bases: vec![base.clone()],
        count: 1,
        mode,
    };
    let result = sinex.prepare_input_group(&plan, &mut |emit| {
        let mut input = capture(bytes);
        input.raw_range = Some(RawRange {
            start: 0,
            end: bytes.len() as u64,
        });
        input.basis = Some(base.clone());
        emit(input, vec![body(root, bytes)])
    })?;
    match result {
        CaptureResult::Accepted { write } => Ok(write),
        CaptureResult::UnresolvedBasisConflict { .. } => {
            Err(polylogue_store::StoreError::Conflict(
                "original bytes kept without advancing the source".into(),
            )
            .into())
        }
    }
}

fn accepted(result: CaptureResult) -> PreparedWrite {
    let CaptureResult::Accepted { write } = result else {
        panic!("accepted preparation required")
    };
    write
}

#[test]
fn unattributed_spooled_bytes_stay_uncertain() {
    let root = tempfile::tempdir().unwrap();
    let sinex = client(root.path());
    let state = sinex
        .pending_source(&source(b"/synthetic/new-source.jsonl"))
        .unwrap();
    assert_eq!(state.status, SourceStatus::Ready);
    assert!(state.head.is_none() && state.raw.is_none());
    let bytes = b"synthetic unattributed capture";
    sinex
        .prepare_input(&capture(bytes), body(root.path(), bytes))
        .unwrap();
    assert_eq!(
        sinex
            .pending_source(&source(b"/synthetic/new-source.jsonl"))
            .unwrap()
            .status,
        SourceStatus::MissingEvidence
    );
}

#[test]
fn retries_keep_first_outcome_after_downgrade() {
    let root = tempfile::tempdir().unwrap();
    let sinex = client(root.path());
    let selector = source(b"/home/op/receiver-constraint.jsonl");
    let mut plan = CapturePlan {
        intent: "first".into(),
        bases: vec![source_base(&selector, None)],
        count: 1,
        mode: CaptureMode::Mutable,
        required_outcome: None,
    };
    let prepare = |plan: &CapturePlan, bytes: &[u8]| {
        sinex
            .prepare_input_group(plan, &mut |emit| {
                let mut input = capture(bytes);
                input.basis = Some(plan.bases[0].clone());
                emit(input, vec![body(root.path(), bytes)])
            })
            .unwrap()
    };
    let first = accepted(prepare(&plan, b"A"));
    let latest = sinex
        .pending_source(&selector)
        .unwrap()
        .head
        .unwrap()
        .identity;
    plan.required_outcome = Some(CaptureOutcome::UnresolvedBasisConflict);
    assert_eq!(accepted(prepare(&plan, b"A")).identity, first.identity);
    plan.intent = "second".into();
    plan.bases[0].pending = Some(latest.clone());
    let CaptureResult::UnresolvedBasisConflict { spooled } = prepare(&plan, b"B") else {
        panic!()
    };
    let page = sinex
        .pending_group_headers(&spooled.identity, None, 1)
        .unwrap();
    let kept = sinex
        .capture_body(&page.headers[0].identity)
        .unwrap()
        .remove(0);
    let encoded = hashed_blobs::Prepared::from_encoded(
        kept.file,
        kept.descriptor,
        None,
        hashed_blobs::ReadLimits::default(),
    )
    .unwrap();
    let mut reader = encoded
        .reader(None, hashed_blobs::ReadLimits::default())
        .unwrap();
    let mut actual = Vec::new();
    std::io::Read::read_to_end(&mut reader, &mut actual).unwrap();
    assert_eq!(actual, b"B");
    plan.required_outcome = None;
    let CaptureResult::UnresolvedBasisConflict { spooled: retried } = prepare(&plan, b"B") else {
        panic!()
    };
    assert_eq!(retried, spooled);
    assert_eq!(
        sinex
            .pending_source(&selector)
            .unwrap()
            .head
            .unwrap()
            .identity,
        latest
    );
}

#[test]
fn source_swap_keeps_a_b_a_retries_and_byte_identity() {
    let root = tempfile::tempdir().unwrap();
    let selector = source(b"/home/op/session-\xff.jsonl");
    let sinex = client(root.path());
    let first = capture_bytes(
        &sinex,
        root.path(),
        b"A",
        source_base(&selector, None),
        CaptureMode::Mutable,
    )
    .unwrap();
    let first_latest = sinex
        .pending_source(&selector)
        .unwrap()
        .head
        .unwrap()
        .identity;
    assert_eq!(
        capture_bytes(
            &sinex,
            root.path(),
            b"A",
            source_base(&selector, None),
            CaptureMode::Mutable
        )
        .unwrap()
        .identity,
        first.identity
    );
    assert!(
        capture_bytes(
            &sinex,
            root.path(),
            b"B",
            source_base(&selector, None),
            CaptureMode::Mutable
        )
        .is_err()
    );
    capture_bytes(
        &sinex,
        root.path(),
        b"B",
        source_base(&selector, Some(first_latest)),
        CaptureMode::Mutable,
    )
    .unwrap();
    let second_latest = sinex
        .pending_source(&selector)
        .unwrap()
        .head
        .unwrap()
        .identity;
    let third = capture_bytes(
        &sinex,
        root.path(),
        b"A",
        source_base(&selector, Some(second_latest)),
        CaptureMode::Mutable,
    )
    .unwrap();
    assert_ne!(first.identity, third.identity);
    assert!(
        sinex
            .pending_source(&source(b"/home/op/session-\xfe.jsonl"))
            .unwrap()
            .head
            .is_none()
    );
    drop(sinex);
    let sinex = client(root.path());
    let state = sinex.pending_source(&selector).unwrap();
    let latest = state.head.unwrap();
    assert_eq!(state.status, SourceStatus::MissingEvidence);
    assert_eq!(
        capture_bytes(
            &sinex,
            root.path(),
            b"A",
            source_base(&selector, Some(latest.identity.clone())),
            CaptureMode::Mutable
        )
        .unwrap()
        .identity,
        third.identity
    );
    assert_eq!(sinex.pending_captures().unwrap().len(), 3);
    let body = sinex.capture_body(&latest.identity).unwrap().remove(0);
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
        required_outcome: None,
        intent: "two-members".into(),
        bases: vec![source_base(&selector, None)],
        count: 2,
        mode: CaptureMode::Immutable,
    };
    assert!(
        sinex
            .prepare_input_group(&plan, &mut |emit| {
                let mut input = capture(b"first");
                input.basis = Some(source_base(&selector, None));
                emit(input, vec![body(root.path(), b"first")])?;
                Err(polylogue_store::StoreError::Invalid(
                    "second member unavailable".into(),
                ))
            })
            .is_err()
    );
    assert!(sinex.pending_captures().unwrap().is_empty());
    let first = capture_bytes(
        &sinex,
        root.path(),
        b"A",
        source_base(&selector, None),
        CaptureMode::Immutable,
    )
    .unwrap();
    capture_bytes(
        &sinex,
        root.path(),
        b"B",
        source_base(&selector, None),
        CaptureMode::Immutable,
    )
    .unwrap();
    assert_eq!(
        capture_bytes(
            &sinex,
            root.path(),
            b"A",
            source_base(&selector, None),
            CaptureMode::Immutable
        )
        .unwrap()
        .identity,
        first.identity
    );
    assert_eq!(sinex.pending_captures().unwrap().len(), 2);
}

#[test]
fn staging_releases_spool_lock_and_rechecks_source() {
    let root = tempfile::tempdir().unwrap();
    let sinex = client(root.path());
    let selector = source(b"/home/op/mutable.jsonl");
    let original_base = source_base(&selector, None);
    let plan = CapturePlan {
        required_outcome: None,
        intent: "staged-A".into(),
        bases: vec![original_base.clone()],
        count: 1,
        mode: CaptureMode::Mutable,
    };
    let result = sinex.prepare_input_group(&plan, &mut |emit| {
        assert!(
            sinex.spool.try_lock().is_ok(),
            "body staging kept the shared spool lock"
        );
        let mut input = capture(b"A");
        input.basis = Some(original_base.clone());
        emit(input, vec![body(root.path(), b"A")])?;
        capture_bytes(
            &sinex,
            root.path(),
            b"B",
            original_base.clone(),
            CaptureMode::Mutable,
        )?;
        Ok(())
    });
    let CaptureResult::UnresolvedBasisConflict { spooled } = result.unwrap() else {
        panic!("stale capture read must keep its bytes in the spool without advancing the source");
    };
    let kept = sinex
        .pending_group_headers(&spooled.identity, None, 1)
        .unwrap();
    assert_eq!(kept.outcome, Some(CaptureOutcome::UnresolvedBasisConflict));
    assert_eq!(kept.headers.len(), 1);
    let raw = sinex
        .capture_body(&kept.headers[0].identity)
        .unwrap()
        .remove(0);
    let prepared = hashed_blobs::Prepared::from_encoded(
        raw.file,
        raw.descriptor,
        None,
        hashed_blobs::ReadLimits::default(),
    )
    .unwrap();
    let mut restored = Vec::new();
    std::io::Read::read_to_end(
        &mut prepared
            .reader(None, hashed_blobs::ReadLimits::default())
            .unwrap(),
        &mut restored,
    )
    .unwrap();
    assert_eq!(restored, b"A");
    let pending = sinex.pending_captures().unwrap();
    assert_eq!(pending.len(), 1);
    let Capture::Input(input) = &pending[0].captured else {
        panic!("input capture expected")
    };
    assert_eq!(input.hash, BlobHash::of(b"B"));
    assert!(
        sinex
            .pending_group_headers("unknown-group", None, 1)
            .unwrap()
            .headers
            .is_empty()
    );
}

#[test]
fn group_pages_keep_siblings_and_order_after_restart() {
    let root = tempfile::tempdir().unwrap();
    let sinex = client(root.path());
    let selector = source(b"/home/op/database");
    let base = source_base(&selector, None);
    let plan = CapturePlan {
        required_outcome: None,
        intent: "one-provider-transaction".into(),
        bases: vec![base.clone()],
        count: 2,
        mode: CaptureMode::Mutable,
    };
    let first = sinex
        .prepare_input_group(&plan, &mut |emit| {
            for (ordinal, bytes) in [b"first".as_slice(), b"second".as_slice()]
                .into_iter()
                .enumerate()
            {
                let mut input = capture(bytes);
                input.index = u32::try_from(ordinal).unwrap();
                input.basis = Some(base.clone());
                emit(input, vec![body(root.path(), bytes)])?;
            }
            sinex.prepare_blob(
                BlobHash::of(b"independent"),
                11,
                body(root.path(), b"independent"),
            )?;
            Ok(())
        })
        .unwrap();
    let first = accepted(first);
    assert!(
        sinex
            .pending_group_headers(&first.identity, None, 0)
            .is_err()
    );
    let page = sinex
        .pending_group_headers(&first.identity, None, 1)
        .unwrap();
    assert_eq!(page.headers.len(), 1);
    assert_eq!(page.next, Some(0));
    let Capture::Input(header) = &page.headers[0].captured else {
        panic!("input capture expected")
    };
    assert_eq!(header.group.as_ref().unwrap().ordinal, 0);
    let next = sinex
        .pending_group_headers(&first.identity, page.next, 1)
        .unwrap();
    assert_eq!(next.headers.len(), 1);
    assert_eq!(next.next, None);
    let Capture::Input(header) = &next.headers[0].captured else {
        panic!("input capture expected")
    };
    assert_eq!(header.group.as_ref().unwrap().ordinal, 1);
    let orders = {
        let queue = sinex.queue().unwrap();
        let mut after = None;
        let mut orders = vec![];
        loop {
            let page = queue.pending_after(after, 1).unwrap();
            if page.is_empty() {
                break;
            }
            after = Some(page[0].order);
            orders.push(page[0].identity.clone());
        }
        orders
    };
    assert_eq!(orders.len(), 3);
    assert_eq!(orders[1], page.headers[0].identity);
    assert_eq!(orders[2], next.headers[0].identity);
    drop(sinex);
    let sinex = client(root.path());
    let reopened = sinex
        .pending_group_headers(&first.identity, None, 2)
        .unwrap();
    assert_eq!(
        reopened.headers,
        vec![page.headers[0].clone(), next.headers[0].clone()]
    );
    assert_eq!(reopened.next, None);
}

#[test]
fn retry_keeps_first_capture_coordinates() {
    let root = tempfile::tempdir().unwrap();
    let sinex = client(root.path());
    let selector = source(b"/home/op/source");
    let base = source_base(&selector, None);
    let plan = CapturePlan {
        required_outcome: None,
        intent: "same-validated-intent".into(),
        bases: vec![base.clone()],
        count: 1,
        mode: CaptureMode::Mutable,
    };
    let mut first_header = None;
    let first = sinex
        .prepare_input_group(&plan, &mut |emit| {
            let mut input = capture(b"A");
            input.basis = Some(base.clone());
            first_header = Some(input.clone());
            emit(input, vec![body(root.path(), b"A")])
        })
        .unwrap();
    let first = accepted(first);
    sinex
        .prepare_input_group(&plan, &mut |emit| {
            let mut input = first_header.clone().unwrap();
            input.rows.input = "different-tentative-coordinate".into();
            emit(input, vec![body(root.path(), b"A")])
        })
        .unwrap();
    let kept = sinex
        .pending_group_headers(&first.identity, None, 1)
        .unwrap();
    let Capture::Input(original) = sinex.capture(&kept.headers[0].identity).unwrap() else {
        panic!("input capture expected")
    };
    assert_eq!(original.rows.input, first_header.unwrap().rows.input);
}

#[test]
fn unresolved_retry_survives_a_moved_source() {
    let root = tempfile::tempdir().unwrap();
    let sinex = client(root.path());
    let selector = source(b"/home/op/provider.bin");
    let first_base = source_base(&selector, None);
    let plan = CapturePlan {
        required_outcome: None,
        intent: "one-offered-identity".into(),
        bases: vec![first_base.clone()],
        count: 1,
        mode: CaptureMode::Mutable,
    };
    let stage = |bytes: &[u8], no_extent: bool| {
        sinex
            .prepare_input_group(&plan, &mut |emit| {
                let mut input = capture(bytes);
                input.basis = Some(first_base.clone());
                if no_extent {
                    input.rows.extents.clear();
                }
                emit(input, vec![body(root.path(), bytes)])
            })
            .unwrap()
    };
    let accepted_a = accepted(stage(b"A", false));
    let raw_b = b"provider\x00\xff";
    let CaptureResult::UnresolvedBasisConflict { spooled: rejected } = stage(raw_b, true) else {
        panic!("different offered body needs its bytes kept in the spool")
    };
    let kept = sinex
        .pending_group_headers(&rejected.identity, None, 1)
        .unwrap();
    assert_eq!(kept.outcome, Some(CaptureOutcome::UnresolvedBasisConflict));
    let raw_id = kept.headers[0].identity.clone();
    let Capture::Input(header) = &kept.headers[0].captured else {
        panic!("input header expected")
    };
    assert!(header.raw_range.is_some());
    assert!(header.rows.extents.is_empty());
    {
        let queue = sinex.queue().unwrap();
        assert!(queue.get(&raw_id).unwrap().unwrap().unresolved);
        let Ingress::Material { material, .. } = queue.decode_frame(&raw_id).unwrap() else {
            panic!("material expected")
        };
        assert_ne!(material.parser, "polylogue-observation");
    }
    let a_latest = sinex
        .pending_source(&selector)
        .unwrap()
        .head
        .unwrap()
        .identity;
    capture_bytes(
        &sinex,
        root.path(),
        b"C",
        source_base(&selector, Some(a_latest)),
        CaptureMode::Mutable,
    )
    .unwrap();
    let current = sinex
        .pending_source(&selector)
        .unwrap()
        .head
        .unwrap()
        .identity;
    let CaptureResult::UnresolvedBasisConflict { spooled: retry } = stage(raw_b, true) else {
        panic!("unresolved retry must not be promoted")
    };
    assert_eq!(rejected, retry);
    assert_eq!(accepted(stage(b"A", false)).identity, accepted_a.identity);
    assert_eq!(
        sinex
            .pending_source(&selector)
            .unwrap()
            .head
            .unwrap()
            .identity,
        current
    );
    assert_eq!(sinex.pending_captures().unwrap().len(), 2);
    drop(sinex);
    let sinex = client(root.path());
    assert_eq!(
        sinex
            .pending_source(&selector)
            .unwrap()
            .head
            .unwrap()
            .identity,
        current
    );
    let kept = sinex
        .pending_group_headers(&rejected.identity, None, 1)
        .unwrap();
    assert_eq!(kept.outcome, Some(CaptureOutcome::UnresolvedBasisConflict));
    let body = sinex.capture_body(&raw_id).unwrap().remove(0);
    let prepared = hashed_blobs::Prepared::from_encoded(
        body.file,
        body.descriptor,
        None,
        hashed_blobs::ReadLimits::default(),
    )
    .unwrap();
    let mut restored = vec![];
    std::io::Read::read_to_end(
        &mut prepared
            .reader(None, hashed_blobs::ReadLimits::default())
            .unwrap(),
        &mut restored,
    )
    .unwrap();
    assert_eq!(restored, raw_b);
}
