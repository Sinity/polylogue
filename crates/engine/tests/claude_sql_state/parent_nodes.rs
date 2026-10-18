//! Relation cells are written atomically with the rest of the parser step.
use super::*;
#[test]
fn parent_nodes_are_stored_atomically_and_survive_commit_and_restart() {
    let raw = line(
        json!({"type":"attachment","uuid":"x","sessionId":"s","parentUuid":"a","attachment":{"type":"synthetic","text":"x"}}),
    );
    let mut w = World::new(raw.as_bytes());
    let original = w.head.clone();
    let snapshot = w.writer().parser_snapshot(&w.head.token).unwrap();
    let view = adapter::View::new(&snapshot, broad()).unwrap();
    let framed = drive_jsonl(
        KeyedRecords::new(FileCtx::main("s"), &view).unwrap(),
        raw.as_bytes(),
        0,
    );
    let mut output = framed.output.unwrap();
    let changes = adapter::encode_step(
        &mut output,
        &w.head,
        framed.parsed_upto,
        &framed.counts,
        broad(),
    )
    .unwrap();
    let node_bytes = changes
        .iter()
        .find(|c| c.namespace == "claude.parent_node.v1")
        .unwrap()
        .value
        .as_ref()
        .unwrap()
        .len();
    assert!(matches!(
        adapter::encode_step(
            &mut output,
            &w.head,
            framed.parsed_upto,
            &framed.counts,
            Budget {
                cells: broad().cells,
                encoded_bytes: node_bytes - 1
            }
        ),
        Err(StoreError::ParserAdmission { .. })
    ));
    drop(view);
    drop(snapshot);
    assert_eq!(
        w.writer()
            .parser_head(original.token.input_pk, original.token.chain)
            .unwrap()
            .unwrap()
            .token,
        original.token
    );
    w.commit(&changes, framed.parsed_upto, &framed.counts);
    w.restart();
    let snapshot = w.writer().parser_snapshot(&w.head.token).unwrap();
    let view = adapter::View::new(&snapshot, broad()).unwrap();
    let node = view.parent_node("s", "\"x\"").unwrap().unwrap();
    assert_eq!(node.parent.as_ref().unwrap().value.as_str(), Some("a"));
    assert_eq!(node.parent.as_ref().unwrap().donor.record.start, 0);
    let tiny = adapter::View::new(
        &snapshot,
        Budget {
            cells: 1,
            encoded_bytes: node_bytes - 1,
        },
    )
    .unwrap();
    assert!(tiny.parent_node("s", "\"x\"").is_err());
    assert!(matches!(
        tiny.take_error(),
        Some(StoreError::ParserAdmission { .. })
    ));
}
