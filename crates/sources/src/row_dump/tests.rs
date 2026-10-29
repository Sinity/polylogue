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

#[test]
fn envelope_and_declared_column_metadata_are_counted_without_repeating_row_leaves() {
    let bytes = br#"{"schema":{"empty_table":"CREATE TABLE empty_table(x TEXT)"},"table_metadata":{"empty_table":{"columns":[{"cid":0,"name":"distinctive_column_name","declared_type":"distinctive_type_name","not_null":false,"primary_key":0,"hidden":1,"future":{"leaf":true}}],"without_rowid":false}}}
{"t":"facts","k":1,"h":"distinctive_hash","r":{"body":"distinctive_body"},"future":{"leaf":true}}
"#;
    let step = run(&RowDump, &mut NoState::default(), bytes, 0).unwrap();
    assert_eq!(step.counts.records, 2);
    assert_eq!(step.counts.counted(), 2);
    let observed = |kind: &str, table: &str, path: &str| {
        &step.counts.wire_shapes[&serde_json::to_string(&(kind, table, path)).unwrap()]
    };
    assert_eq!(observed("row", "facts", "/body").unused_count, 1);
    let envelope = observed("row_envelope", "facts", "/future/leaf");
    assert_eq!(envelope.types["boolean"], 1);
    assert_eq!(envelope.unused_count, 1);
    assert_eq!(observed("row_envelope", "facts", "/k").unused_count, 0);
    assert_eq!(observed("row_envelope", "facts", "").unknown_table_count, 0);
    assert_eq!(observed("row", "facts", "").unknown_table_count, 1);
    assert!(
        observed("table_metadata", "empty_table", "/columns/[]/name")
            .expected
            .contains(&"string".into())
    );
    assert_eq!(
        observed("table_metadata", "empty_table", "/columns/[]/future/leaf").unused_count,
        1
    );
    assert_eq!(
        step.counts
            .wire_shapes
            .values()
            .map(|shape| shape.unused_count)
            .sum::<u64>(),
        3
    );
    let encoded = serde_json::to_string(&step.counts.wire_shapes).unwrap();
    for scalar in [
        "distinctive_column_name",
        "distinctive_type_name",
        "distinctive_hash",
        "distinctive_body",
    ] {
        assert!(!encoded.contains(scalar));
    }
}
