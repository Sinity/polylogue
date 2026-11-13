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
pub(super) fn basis(source: &SourceSelector, pending: Option<String>) -> SourceBasis {
    SourceBasis {
        boundary: None,
        resolved: None,
        source: source.clone(),
        status: SourceStatus::MissingEvidence,
        confirmed: None,
        pending,
    }
}
pub(super) fn acquire(
    sinex: &Sinex,
    root: &std::path::Path,
    bytes: &[u8],
    basis: SourceBasis,
    mode: CaptureMode,
) -> Result<PreparedWrite> {
    let plan = CapturePlan {
        required_disposition: None,
        intent: blake3::hash(bytes).to_hex().to_string(),
        bases: vec![basis.clone()],
        count: 1,
        mode,
    };
    let acknowledgment = sinex.prepare_input_group(&plan, &mut |emit| {
        let mut input = capture(bytes);
        input.raw_range = Some(RawRange {
            start: 0,
            end: bytes.len() as u64,
        });
        input.basis = Some(basis.clone());
        emit(input, vec![body(root, bytes)])
    })?;
    match acknowledgment {
        CaptureReceipt::Accepted { write } => Ok(write),
        CaptureReceipt::UnresolvedBasisConflict { .. } => {
            Err(polylogue_store::StoreError::Conflict(
                "raw bytes retained without source acceptance".into(),
            )
            .into())
        }
    }
}

fn accepted(acknowledgment: CaptureReceipt) -> PreparedWrite {
    let CaptureReceipt::Accepted { write } = acknowledgment else {
        panic!("accepted preparation required")
    };
    write
}

#[test]
fn empty_local_sources_are_known_but_unattributed_custody_stays_uncertain() {
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
fn receiver_downgrade_retains_new_bodies_and_exact_frozen_retries_keep_first_disposition() {
    let root = tempfile::tempdir().unwrap();
    let sinex = client(root.path());
    let selector = source(b"/home/op/receiver-constraint.jsonl");
    let mut plan = CapturePlan {
        intent: "first".into(),
        bases: vec![basis(&selector, None)],
        count: 1,
        mode: CaptureMode::Mutable,
        required_disposition: None,
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
    let head = sinex
        .pending_source(&selector)
        .unwrap()
        .head
        .unwrap()
        .identity;
    plan.required_disposition = Some(CaptureDisposition::UnresolvedBasisConflict);
    assert_eq!(accepted(prepare(&plan, b"A")).identity, first.identity);
    plan.intent = "second".into();
    plan.bases[0].pending = Some(head.clone());
    let CaptureReceipt::UnresolvedBasisConflict { custody } = prepare(&plan, b"B") else {
        panic!()
    };
    let page = sinex
        .pending_group_headers(&custody.identity, None, 1)
        .unwrap();
    let retained = sinex
        .capture_body(&page.headers[0].identity)
        .unwrap()
        .remove(0);
    let encoded = hashed_blobs::Prepared::from_encoded(
        retained.file,
        retained.descriptor,
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
    plan.required_disposition = None;
    let CaptureReceipt::UnresolvedBasisConflict { custody: retried } = prepare(&plan, b"B") else {
        panic!()
    };
    assert_eq!(retried, custody);
    assert_eq!(
        sinex
            .pending_source(&selector)
            .unwrap()
            .head
            .unwrap()
            .identity,
        head
    );
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
        .pending_source(&selector)
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
        .pending_source(&selector)
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
            .pending_source(&source(b"/home/op/session-\xfe.jsonl"))
            .unwrap()
            .head
            .is_none()
    );
    drop(sinex);
    let sinex = client(root.path());
    let state = sinex.pending_source(&selector).unwrap();
    let head = state.head.unwrap();
    assert_eq!(state.status, SourceStatus::MissingEvidence);
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
    let body = sinex.capture_body(&head.identity).unwrap().remove(0);
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
        required_disposition: None,
        intent: "two-members".into(),
        bases: vec![basis(&selector, None)],
        count: 2,
        mode: CaptureMode::Immutable,
    };
    assert!(
        sinex
            .prepare_input_group(&plan, &mut |emit| {
                let mut input = capture(b"first");
                input.basis = Some(basis(&selector, None));
                emit(input, vec![body(root.path(), b"first")])?;
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
fn private_staging_releases_custody_lock_and_rechecks_moved_source_before_publication() {
    let root = tempfile::tempdir().unwrap();
    let sinex = client(root.path());
    let selector = source(b"/home/op/mutable.jsonl");
    let original_basis = basis(&selector, None);
    let plan = CapturePlan {
        required_disposition: None,
        intent: "staged-A".into(),
        bases: vec![original_basis.clone()],
        count: 1,
        mode: CaptureMode::Mutable,
    };
    let result = sinex.prepare_input_group(&plan, &mut |emit| {
        assert!(
            sinex.spool.try_lock().is_ok(),
            "body staging retained the shared custody lock"
        );
        let mut input = capture(b"A");
        input.basis = Some(original_basis.clone());
        emit(input, vec![body(root.path(), b"A")])?;
        acquire(
            &sinex,
            root.path(),
            b"B",
            original_basis.clone(),
            CaptureMode::Mutable,
        )?;
        Ok(())
    });
    let CaptureReceipt::UnresolvedBasisConflict { custody } = result.unwrap() else {
        panic!("stale acquisition must retain raw custody without source acceptance");
    };
    let retained = sinex
        .pending_group_headers(&custody.identity, None, 1)
        .unwrap();
    assert_eq!(
        retained.disposition,
        Some(CaptureDisposition::UnresolvedBasisConflict)
    );
    assert_eq!(retained.headers.len(), 1);
    let raw = sinex
        .capture_body(&retained.headers[0].identity)
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
fn acquisition_pages_preserve_original_siblings_and_relative_orders_after_restart() {
    let root = tempfile::tempdir().unwrap();
    let sinex = client(root.path());
    let selector = source(b"/home/op/database");
    let source_basis = basis(&selector, None);
    let plan = CapturePlan {
        required_disposition: None,
        intent: "one-provider-transaction".into(),
        bases: vec![source_basis.clone()],
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
                input.basis = Some(source_basis.clone());
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
fn exact_retry_retains_first_capture_coordinates_after_cache_coordinates_change() {
    let root = tempfile::tempdir().unwrap();
    let sinex = client(root.path());
    let selector = source(b"/home/op/source");
    let source_basis = basis(&selector, None);
    let plan = CapturePlan {
        required_disposition: None,
        intent: "same-validated-intent".into(),
        bases: vec![source_basis.clone()],
        count: 1,
        mode: CaptureMode::Mutable,
    };
    let mut first_header = None;
    let first = sinex
        .prepare_input_group(&plan, &mut |emit| {
            let mut input = capture(b"A");
            input.basis = Some(source_basis.clone());
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
    let retained = sinex
        .pending_group_headers(&first.identity, None, 1)
        .unwrap();
    let Capture::Input(original) = sinex.capture(&retained.headers[0].identity).unwrap() else {
        panic!("input capture expected")
    };
    assert_eq!(original.rows.input, first_header.unwrap().rows.input);
}

#[test]
fn unresolved_retry_is_stable_after_head_movement_and_keeps_provider_bytes_without_logical_extent()
{
    let root = tempfile::tempdir().unwrap();
    let sinex = client(root.path());
    let selector = source(b"/home/op/provider.bin");
    let frozen_basis = basis(&selector, None);
    let plan = CapturePlan {
        required_disposition: None,
        intent: "one-offered-identity".into(),
        bases: vec![frozen_basis.clone()],
        count: 1,
        mode: CaptureMode::Mutable,
    };
    let stage = |bytes: &[u8], no_extent: bool| {
        sinex
            .prepare_input_group(&plan, &mut |emit| {
                let mut input = capture(bytes);
                input.basis = Some(frozen_basis.clone());
                if no_extent {
                    input.rows.extents.clear();
                }
                emit(input, vec![body(root.path(), bytes)])
            })
            .unwrap()
    };
    let accepted_a = accepted(stage(b"A", false));
    let raw_b = b"provider\x00\xff";
    let CaptureReceipt::UnresolvedBasisConflict { custody: rejected } = stage(raw_b, true) else {
        panic!("different offered body needs raw custody")
    };
    let retained = sinex
        .pending_group_headers(&rejected.identity, None, 1)
        .unwrap();
    assert_eq!(
        retained.disposition,
        Some(CaptureDisposition::UnresolvedBasisConflict)
    );
    let raw_id = retained.headers[0].identity.clone();
    let Capture::Input(header) = &retained.headers[0].captured else {
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
    let a_head = sinex
        .pending_source(&selector)
        .unwrap()
        .head
        .unwrap()
        .identity;
    acquire(
        &sinex,
        root.path(),
        b"C",
        basis(&selector, Some(a_head)),
        CaptureMode::Mutable,
    )
    .unwrap();
    let current = sinex
        .pending_source(&selector)
        .unwrap()
        .head
        .unwrap()
        .identity;
    let CaptureReceipt::UnresolvedBasisConflict { custody: retry } = stage(raw_b, true) else {
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
    let retained = sinex
        .pending_group_headers(&rejected.identity, None, 1)
        .unwrap();
    assert_eq!(
        retained.disposition,
        Some(CaptureDisposition::UnresolvedBasisConflict)
    );
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
