use super::*;
use crate::parse::run;

#[test]
fn unknown_tables_headers_and_rows_produce_only_scalar_free_wire_facts() {
    let bytes = b"{\"db\":\"tool.db\",\"at_ms\":1,\"schema\":{\"empty_table\":\"CREATE TABLE distinctive_sql_secret(x TEXT)\",\"facts\":\"CREATE TABLE facts(x)\"}}\n{\"t\":\"facts\",\"k\":1,\"r\":{\"body\":\"distinctive_row_secret\",\"n\":7},\"h\":\"hash\"}\n{\"t\":\"facts\",\"k\":1,\"gone\":true}\n";
    let step = run(&RowDump, &mut NoState::default(), bytes, 0).unwrap();
    assert_eq!(step.consumed, bytes.len() as u64);
    assert_eq!(step.output, Output::default());
    assert_eq!(step.counts.records, 3);
    assert_eq!(step.counts.used["schema"], 1);
    assert_eq!(step.counts.counted(), 3);
    let row = &step.counts.wire_shapes[&serde_json::to_string(&("row", "facts", "/body")).unwrap()];
    assert_eq!(row.types["string"], 1);
    assert_eq!(row.unused_count, 1);
    assert_eq!(row.table.as_deref(), Some("facts"));
    let empty =
        &step.counts.wire_shapes[&serde_json::to_string(&("schema", "empty_table", "")).unwrap()];
    assert_eq!(empty.unknown_table_count, 1);
    let encoded = serde_json::to_string(&step.counts.wire_shapes).unwrap();
    assert!(!encoded.contains("distinctive_sql_secret"));
    assert!(!encoded.contains("distinctive_row_secret"));
}

#[test]
fn streamed_append_waits_for_a_complete_row_and_counts_it_once() {
    let first = b"{\"schema\":{}}\n{\"t\":\"facts\",\"r\":{\"n\":";
    let mut state = NoState::default();
    let a = run(&RowDump, &mut state, first, 0).unwrap();
    assert_eq!(a.consumed, 14);
    assert_eq!(a.counts.records, 1);
    let mut tail = first[a.consumed as usize..].to_vec();
    tail.extend_from_slice(b"2}}\n");
    let b = run(&RowDump, &mut state, &tail, a.consumed).unwrap();
    assert_eq!(b.consumed, tail.len() as u64);
    assert_eq!(b.counts.records, 1);
    assert_eq!(
        b.counts.wire_shapes[&serde_json::to_string(&("row", "facts", "/n")).unwrap()].types["number"],
        1
    );
}

#[test]
fn malformed_row_records_are_counted_with_fixed_reasons() {
    let step = run(
        &RowDump,
        &mut NoState::default(),
        b"{\"t\":\"facts\"}\n{\"schema\":false}\n{bad}\n",
        0,
    )
    .unwrap();
    assert_eq!(step.counts.records, 3);
    assert_eq!(step.counts.unreadable.values().sum::<u64>(), 3);
}
