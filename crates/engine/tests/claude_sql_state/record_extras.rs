//! Map size limits use the normal encoder; a failed write advances nothing.
use super::*;
#[test]
fn selected_extra_map_encoding_refuses_small_allowance_and_restarts_after_atomic_commit() {
    let mut value = assistant("a", "m", "r");
    value["effort"] = json!("x".repeat(3000));
    let raw = line(value);
    let mut world = World::new(raw.as_bytes());
    let original = world.head.clone();
    let snapshot = world.writer().parser_snapshot(&world.head.token).unwrap();
    let view = adapter::View::new(&snapshot, broad()).unwrap();
    let framed = drive_jsonl(
        KeyedRecords::new(FileCtx::main("s"), &view).unwrap(),
        raw.as_bytes(),
        0,
    );
    let mut output = framed.output.unwrap();
    let broad_changes = adapter::encode_step(
        &mut output,
        &world.head,
        framed.parsed_upto,
        &framed.counts,
        broad(),
    )
    .unwrap();
    let response_bytes = broad_changes
        .iter()
        .find(|c| c.namespace == "claude.response.v1")
        .unwrap()
        .value
        .as_ref()
        .unwrap()
        .len();
    assert!(response_bytes > 3000);
    assert!(matches!(
        adapter::encode_step(
            &mut output,
            &world.head,
            framed.parsed_upto,
            &framed.counts,
            Budget {
                cells: broad().cells,
                encoded_bytes: response_bytes - 1
            }
        ),
        Err(StoreError::ParserAdmission { .. })
    ));
    drop(view);
    drop(snapshot);
    assert_eq!(
        world
            .writer()
            .parser_head(original.token.input_pk, original.token.chain)
            .unwrap()
            .unwrap()
            .token,
        original.token
    );
    world.commit(&broad_changes, framed.parsed_upto, &framed.counts);
    world.restart();
    let snapshot = world.writer().parser_snapshot(&world.head.token).unwrap();
    let view = adapter::View::new(&snapshot, broad()).unwrap();
    let selected = view
        .response("s", &ResponseKey::new("m", "r"))
        .unwrap()
        .unwrap();
    assert_eq!(selected.record_extras["effort"].raw.len(), 3002);
    assert_eq!(
        selected.record_extras["effort"]
            .fact
            .selection
            .as_ref()
            .unwrap()
            .inputs[0]
            .record
            .start,
        0
    );
    let tiny = adapter::View::new(
        &snapshot,
        Budget {
            cells: broad().cells,
            encoded_bytes: response_bytes - 1,
        },
    )
    .unwrap();
    assert!(tiny.response("s", &ResponseKey::new("m", "r")).is_err());
    assert!(matches!(
        tiny.take_error(),
        Some(StoreError::ParserAdmission { .. })
    ));
}
