use super::*;
use crate::parse::run;

#[test]
fn row_dump_wire_shapes_hold_no_scalar_values() {
    let bytes = b"{\"db\":\"tool.db\",\"at_ms\":1,\"schema\":{\"empty_table\":\"CREATE TABLE distinctive_sql_secret(x TEXT)\",\"facts\":\"CREATE TABLE facts(x)\"}}\n{\"t\":\"facts\",\"k\":1,\"r\":{\"body\":\"distinctive_row_secret\",\"n\":7},\"at_ms\":1,\"encoding\":\"sqlite\"}\n{\"t\":\"facts\",\"k\":1,\"gone\":true}\n";
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
        b"{\"t\":\"facts\"}\n{\"schema\":false,\"table_metadata\":{\"facts\":{\"future\":{\"leaf\":true}}}}\n{bad}\n",
        0,
    )
    .unwrap();
    assert_eq!(step.counts.records, 3);
    assert_eq!(step.counts.unreadable.len(), 3);
    assert_eq!(
        step.counts.unreadable["row: row-dump record has no row object"],
        1
    );
    assert_eq!(
        step.counts.unreadable["schema: row-dump schema is not an object"],
        1
    );
    let malformed: Vec<_> = step
        .counts
        .unreadable
        .iter()
        .filter(|(key, _)| key.starts_with("?: malformed JSON: "))
        .collect();
    assert_eq!(malformed.len(), 1);
    assert_eq!(*malformed[0].1, 1);
    let shape = &step.counts.wire_shapes
        [&serde_json::to_string(&("table_metadata", "facts", "/future/leaf")).unwrap()];
    assert_eq!(shape.unused_count, 1);
}

#[test]
fn row_wrapper_and_column_metadata_are_counted_once() {
    let bytes = br#"{"schema":{"empty_table":"CREATE TABLE empty_table(x TEXT)"},"table_metadata":{"empty_table":{"columns":[{"cid":0,"name":"distinctive_column_name","declared_type":"distinctive_type_name","not_null":false,"primary_key":0,"hidden":1,"future":{"leaf":true}}],"without_rowid":false}}}
{"t":"facts","k":1,"r":{"body":"distinctive_body"},"at_ms":1,"encoding":"distinctive_encoding","future":{"leaf":true}}
"#;
    let step = run(&RowDump, &mut NoState::default(), bytes, 0).unwrap();
    assert_eq!(step.counts.records, 2);
    assert_eq!(step.counts.counted(), 2);
    let observed = |kind: &str, table: &str, path: &str| {
        &step.counts.wire_shapes[&serde_json::to_string(&(kind, table, path)).unwrap()]
    };
    assert_eq!(observed("row", "facts", "/body").unused_count, 1);
    let header = observed("row_wrapper", "facts", "/future/leaf");
    assert_eq!(header.types["boolean"], 1);
    assert_eq!(header.unused_count, 1);
    assert_eq!(observed("row_wrapper", "facts", "/k").unused_count, 0);
    assert_eq!(observed("row_wrapper", "facts", "").unknown_table_count, 0);
    assert_eq!(observed("row", "facts", "").unknown_table_count, 1);
    assert_eq!(
        observed("table_metadata", "empty_table", "/columns/[]/name").expected,
        vec!["null".to_string(), "string".to_string()]
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
        "distinctive_encoding",
        "distinctive_body",
    ] {
        assert!(!encoded.contains(scalar));
    }
}

#[test]
fn the_envelopes_the_store_writes_are_declared() {
    // The header, row and gone lines as the store's database reader writes
    // them: only the row's own columns, which no parser reads here, are unused.
    let bytes = br#"{"db":"tool.db","at_ms":1,"schema":{"facts":"CREATE TABLE facts(x)"},"table_metadata":{},"source":{"path":"/home/op/tool.db"}}
{"t":"facts","k":[1],"r":{"x":"one"},"at_ms":2,"encoding":"sqlite"}
{"t":"facts","k":[1],"gone":true,"at_ms":3}
"#;
    let step = run(&RowDump, &mut NoState::default(), bytes, 0).unwrap();
    assert_eq!(step.counts.counted(), 3);
    let unused: Vec<_> = step
        .counts
        .wire_shapes
        .iter()
        .filter(|(_, shape)| shape.unused_count > 0)
        .map(|(key, shape)| (key.as_str(), shape.unused_count))
        .collect();
    assert_eq!(unused, [(r#"["row","facts","/x"]"#, 1)]);
    assert_eq!(
        step.counts.unread_fields,
        [("row.x".to_owned(), 1)].into_iter().collect()
    );
}
