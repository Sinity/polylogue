use super::*;
use serde_json::json;

#[test]
fn original_rows_keep_text_currency_and_exact_source_ranges() {
    let row = json!({"table":"note","row":{"id":"n1","first_note":"n0","kind":"tag","key":"hidden","value":" { \"x\": 1 } ","author":"agent:s1","machine":"m","created_at":1000,"valid_until":null,"commit_seq":7}});
    let bytes = format!("{}\n", row);
    let step = crate::run(&Rows, &mut State::default(), bytes.as_bytes(), 41).unwrap();
    assert_eq!(step.output.events.len(), 1);
    let event = &step.output.events[0];
    assert_eq!(event.kind, NOTE);
    assert_eq!(event.data, row);
    assert_eq!(event.range, Some((41, 41 + bytes.len() as u64 - 1)));
    assert_eq!(event.at, Timestamp::from_unix_ms(1000));
    assert_eq!(event.at_quality, TimeQuality::Intrinsic);
}

#[test]
fn original_targets_relations_and_other_tables_have_explicit_outcomes() {
    let values = [
        json!({"table":"note_target","row":{"note_id":"n","ref":"note:r"}}),
        json!({"table":"note_relation","row":{"note_id":"w","kind":"withdraws","other":"note:n"}}),
        json!({"table":"meta","row":{"key":"write_counter","value":"5"}}),
    ];
    let bytes = values.iter().map(|v| format!("{v}\n")).collect::<String>();
    let step = crate::run(&Rows, &mut State::default(), bytes.as_bytes(), 0).unwrap();
    assert_eq!(
        step.output
            .events
            .iter()
            .map(|e| e.kind.as_str())
            .collect::<Vec<_>>(),
        [TARGET, RELATION]
    );
    assert_eq!(step.output.events[0].data, values[0]);
    assert_eq!(step.output.events[1].data, values[1]);
    let malformed = b"{\"table\":\"note\",\"row\":{\"id\":\"n\"}}\n";
    assert!(
        crate::run(&Rows, &mut State::default(), malformed, 0)
            .unwrap()
            .output
            .events
            .is_empty()
    );
}
