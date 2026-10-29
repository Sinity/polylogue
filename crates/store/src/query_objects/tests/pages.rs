use super::*;
use polylogue_model::{
    BlobHash,
    read::{
        HttpCall, ReadDirection, ReadPosition, ReadRecordKind, ReadRecordLocator, ReadSeek,
        ReadyCall, SelectionLocator,
    },
};

fn position() -> ReadPosition {
    ReadPosition {
        selection: SelectionLocator {
            row: 1,
            record: ReadRecordLocator {
                kind: ReadRecordKind::Message,
                ordinal: 4,
            },
        },
        item: 2,
        character: 31,
        item_hash: BlobHash::of(b"encoded item at the captured snapshot"),
    }
}
pub(super) fn child(root: &QueryRun, id: &str, read_at: ChangePosition) -> QueryRun {
    QueryRun {
        id: run_id(id),
        question: root.question,
        at: Timestamp::from_unix_ms(4000).unwrap(),
        read_at,
        machine: "current-page-machine".into(),
        authorship: compute_authorship(Channel::Web, &CallerEvidence::default(), None),
        binding: RunBinding::Continuation {
            root: root.id.clone(),
            binding_hash: root.binding_hash().unwrap(),
            seek: ReadSeek {
                direction: ReadDirection::After,
                anchor: position(),
            },
            accepted_request: Some(
                serde_json::json!({"id":"actual-current-page","operation":"read","arguments":{"root_run":root.id,"cursor":"opaque-authenticated-cursor-artifact"}}),
            ),
        },
    }
}
fn next(root: &QueryRun) -> ReadyCall {
    ReadyCall {
        cli: format!(
            "polylogue read --root-run {} --cursor opaque-authenticated-cursor-artifact",
            root.id
        ),
        mcp: serde_json::json!({"operation":"read","arguments":{"root_run":root.id,"cursor":"opaque-authenticated-cursor-artifact"}}),
        http: HttpCall {
            method: "POST".into(),
            path: "/read".into(),
            arguments: serde_json::json!({"root_run":root.id,"cursor":"opaque-authenticated-cursor-artifact"}),
        },
    }
}

#[test]
fn each_page_keeps_current_invocation_and_one_immutable_outcome_without_copying_root_options() {
    let archive = WriterArchive::new();
    let object = question("messages");
    archive.writer.query_object(object.clone()).unwrap();
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let mut root = run(
        &object,
        "root-page",
        1000,
        pool.reader().unwrap().change_position().unwrap(),
    );
    let execution = root.binding.original_mut().unwrap();
    execution.accepted_request = Some(
        serde_json::json!({"id":"original-current-correlation","arguments":{"query":"large root intent ".repeat(5000)}}),
    );
    execution.presentation = ExecutionPresentation::Read {
        options: Default::default(),
        delivery: polylogue_model::read::ReadDelivery::Page,
        format: polylogue_model::read::Format::Html,
        page_budget: Some(Default::default()),
        direct_source: None,
    };
    let root_hash = root.binding_hash().unwrap();
    archive.writer.begin_query_run(root.clone()).unwrap();
    let mut first = outcome(&root, 3);
    first.exact = false;
    first.next = Some(next(&root));
    archive
        .writer
        .finish_query_run(first.clone())
        .unwrap()
        .recv()
        .unwrap()
        .unwrap();
    let page = child(
        &root,
        "second-page",
        pool.reader().unwrap().change_position().unwrap(),
    );
    assert_ne!(page.at, root.at);
    assert_ne!(page.authorship, root.authorship);
    assert_eq!(page.binding_hash().unwrap(), root_hash);
    assert!(serde_json::to_vec(&root).unwrap().len() > 25_000);
    assert!(serde_json::to_vec(&page).unwrap().len() < 4096);
    archive.writer.begin_query_run(page.clone()).unwrap();
    let complete = outcome(&page, 2);
    archive
        .writer
        .finish_query_run(complete.clone())
        .unwrap()
        .recv()
        .unwrap()
        .unwrap();
    let reader = pool.reader().unwrap();
    assert_eq!(
        reader.query_run(&page.id, None).unwrap().unwrap().value,
        page
    );
    assert_eq!(
        reader
            .query_run_root(&page.id, None)
            .unwrap()
            .unwrap()
            .value,
        root
    );
    assert_eq!(
        reader
            .query_run_outcome(&root.id, None)
            .unwrap()
            .unwrap()
            .value,
        first
    );
    assert_eq!(
        reader
            .query_run_outcome(&page.id, None)
            .unwrap()
            .unwrap()
            .value,
        complete
    );
    drop(reader);
    let mut overwrite = first;
    overwrite.exact = true;
    overwrite.next = None;
    assert!(
        archive
            .writer
            .finish_query_run(overwrite)
            .unwrap()
            .recv()
            .unwrap()
            .is_err()
    );
}

#[test]
fn child_intent_integrity_survives_outcome_first_order_and_rejects_position_envelope_and_chain_changes()
 {
    let archive = WriterArchive::new();
    let object = question("messages");
    archive.writer.query_object(object.clone()).unwrap();
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let mut root = run(
        &object,
        "integrity-root",
        1000,
        pool.reader().unwrap().change_position().unwrap(),
    );
    root.binding.original_mut().unwrap().presentation = ExecutionPresentation::Read {
        options: Default::default(),
        delivery: Default::default(),
        format: polylogue_model::read::Format::Json,
        page_budget: Some(Default::default()),
        direct_source: None,
    };
    archive.writer.begin_query_run(root.clone()).unwrap();
    let page = child(
        &root,
        "integrity-child",
        pool.reader().unwrap().change_position().unwrap(),
    );
    let observed = outcome(&page, 2);
    archive
        .writer
        .finish_query_run(observed.clone())
        .unwrap()
        .recv()
        .unwrap()
        .unwrap();
    let mut moved = page.clone();
    let RunBinding::Continuation { seek, .. } = &mut moved.binding else {
        unreachable!()
    };
    seek.anchor.character += 1;
    assert_eq!(moved.binding_hash().unwrap(), page.binding_hash().unwrap());
    assert_ne!(moved.record_hash().unwrap(), page.record_hash().unwrap());
    assert!(archive.writer.begin_query_run(moved).is_err());
    archive.writer.begin_query_run(page.clone()).unwrap();
    let mut reversed = page.clone();
    let RunBinding::Continuation { seek, .. } = &mut reversed.binding else {
        unreachable!()
    };
    seek.direction = ReadDirection::Before;
    assert_eq!(
        reversed.binding_hash().unwrap(),
        page.binding_hash().unwrap()
    );
    assert!(archive.writer.begin_query_run(reversed).is_err());
    let mut changed = page.clone();
    let RunBinding::Continuation {
        accepted_request, ..
    } = &mut changed.binding
    else {
        unreachable!()
    };
    *accepted_request = Some(serde_json::json!({"id":"different-current-invocation"}));
    assert!(archive.writer.begin_query_run(changed).is_err());
    let mut bad_outcome = observed;
    bad_outcome.run_hash = BlobHash::of(b"different child intent");
    assert!(
        archive
            .writer
            .finish_query_run(bad_outcome)
            .unwrap()
            .recv()
            .unwrap()
            .is_err()
    );
    let mut chained = child(
        &root,
        "chained-child",
        pool.reader().unwrap().change_position().unwrap(),
    );
    let RunBinding::Continuation { root: parent, .. } = &mut chained.binding else {
        unreachable!()
    };
    *parent = page.id.clone();
    assert!(archive.writer.begin_query_run(chained.clone()).is_err());
    let RunBinding::Continuation { root: parent, .. } = &mut chained.binding else {
        unreachable!()
    };
    *parent = chained.id.clone();
    assert!(archive.writer.begin_query_run(chained).is_err());
}
