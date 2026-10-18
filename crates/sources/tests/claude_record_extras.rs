//! Claude record extras: one per key per response, with their origins.
use polylogue_sources::{
    claude_code::{
        FileCtx, KeyedRecords,
        keyed::{MemoryReadView, ReadView, ResponseCell, ResponseKey},
    },
    frame::drive_jsonl,
};
fn row(uuid: &str, text: &str, extras: &str) -> String {
    format!(
        r#"{{"type":"assistant","uuid":"{uuid}","sessionId":"s","requestId":"r",{extras}"message":{{"id":"response","content":[{{"type":"text","text":"{text}"}}]}}}}
"#
    )
}
fn step(memory: &mut MemoryReadView, raw: &str, base: u64) -> ResponseCell {
    let framed = drive_jsonl(
        KeyedRecords::new(FileCtx::main("s"), &*memory).unwrap(),
        raw.as_bytes(),
        base,
    );
    let mut output = framed.output.unwrap();
    output.header.parsed_upto = framed.parsed_upto;
    memory.apply(output.header, output.delta);
    memory
        .response("s", &ResponseKey::new("response", "r"))
        .unwrap()
        .unwrap()
}
#[test]
fn selected_immediate_extras_preserve_supplied_order_and_chunk_restart_state() {
    let records = [
        row("a", "a", r#""effort":1,"other":null,"#),
        row("b", "b", r#""effort":2,"#),
        row("a", "changed", r#""effort":3,"#),
        row("a", "omitted", ""),
        row("c", "c", ""),
    ];
    let all = records.concat();
    let mut whole = MemoryReadView::default();
    let expected = step(&mut whole, &all, 0);
    let mut memory = MemoryReadView::default();
    let mut offset = 0;
    for row in &records {
        let response = step(&mut memory, row, offset);
        offset += row.len() as u64;
        // These are precisely the response bytes persisted by the SQL adapter.
        let restored: ResponseCell =
            serde_json::from_slice(&serde_json::to_vec(&response).unwrap()).unwrap();
        assert_eq!(restored, response);
    }
    let actual = memory
        .response("s", &ResponseKey::new("response", "r"))
        .unwrap()
        .unwrap();
    assert_eq!(actual, expected);
    assert_eq!(actual.record_extras.len(), 2);
    assert_eq!(actual.record_extras["effort"].item, 0);
    assert_eq!(actual.record_extras["effort"].raw, b"3");
    assert_eq!(actual.record_extras["effort"].fact.supplied_revision, 3);
    assert_eq!(actual.record_extras["other"].raw, b"null");
    let baseline = memory.contribution(&actual.message.id, 1).unwrap().unwrap();
    assert_ne!(
        actual.record_extras["effort"].raw,
        baseline.record_extra_raw("effort").unwrap()
    );
    assert_eq!(
        polylogue_sources::claude_code::keyed::SelectedRecordExtra::correction_tag("effort"),
        "response/record_extra/6566666f7274"
    );
}
#[test]
fn identical_effective_duplicate_is_noop_but_equal_value_resupply_on_changed_record_owns_origin() {
    let a = row("a", "a", r#""effort":1,"#);
    let b = row("b", "b", r#""effort":1,"#);
    let update = row("a", "changed", r#""effort":1,"#);
    let mut memory = MemoryReadView::default();
    let first = step(&mut memory, &(a.clone() + &b), 0);
    let before = (a.len() + b.len()) as u64;
    let selected = step(&mut memory, &update, before);
    assert_eq!(first.record_extras["effort"].item, 1);
    assert_eq!(selected.record_extras["effort"].item, 0);
    assert_eq!(
        selected.record_extras["effort"]
            .fact
            .selection
            .as_ref()
            .unwrap()
            .inputs[0]
            .record
            .start,
        before
    );
    let duplicate = step(&mut memory, &update, before + update.len() as u64);
    assert_eq!(duplicate, selected);
    assert_eq!(selected.record_extras["effort"].raw, b"1");
    // Encoded exact keys remain injective and ASCII, including Unicode.
    use polylogue_sources::claude_code::keyed::SelectedRecordExtra;
    assert_ne!(
        SelectedRecordExtra::correction_tag("a/b"),
        SelectedRecordExtra::correction_tag("a.b")
    );
    assert_eq!(
        SelectedRecordExtra::correction_tag("λ/key"),
        "response/record_extra/cebb2f6b6579"
    );
}
