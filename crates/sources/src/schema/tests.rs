use super::*;
use serde_json::json;

fn infer(bytes: &[u8], framing: Framing) -> Schema {
    let mut observer = Observer::default();
    observer.read(io::Cursor::new(bytes), framing).unwrap();
    observer.finish("synthetic".into(), vec![])
}

#[test]
fn empty_and_single_input_have_structure_without_values() {
    let empty = infer(b"", Framing::Lines);
    assert_eq!(empty.records, 0);
    assert!(empty.fields.is_empty());
    let one = infer(
        br#"{"id":"private-identifier","count":42,"message":{"content":"private-text"}}"#,
        Framing::Lines,
    );
    assert_eq!(one.records, 1);
    assert_eq!(one.fields["/count"].types["integer"], 1);
    assert!(one.fields["/id"].required);
    let text = serde_json::to_string(&one).unwrap();
    assert!(!text.contains("private-"));
    assert!(!text.contains("enum"));
    assert!(!text.contains("examples"));
}

#[test]
fn nested_arrays_mixed_types_null_and_missing_have_conditional_parent_counts() {
    let schema = infer(
        br#"{"items":[{"text":null},{"text":"x"},{}],"flag":true}
{"items":[],"flag":null}
{"items":[4,[false]]}
"#,
        Framing::Lines,
    );
    let text = &schema.fields["/items/[]/text"];
    assert_eq!((text.present, text.parents, text.missing), (2, 3, 1));
    assert_eq!(
        text.types.keys().map(String::as_str).collect::<Vec<_>>(),
        ["null", "string"]
    );
    assert!(!text.required);
    assert_eq!(schema.fields["/items/[]"].occurrences, 5);
    assert_eq!(schema.fields["/items/[]"].present, 2);
    assert!(schema.fields.contains_key("/items/[]/[]"));
    assert_eq!(schema.fields["/flag"].missing, 1);
}

#[test]
fn ordinary_names_are_reported_and_identifier_keys_are_grouped_without_values() {
    let schema = infer(br#"{"mapping":{"alice":{"text":"secret-a"},"550e8400-e29b-41d4-a716-446655440000":{"text":"secret-b"},"0123456789abcdef":{"count":2},"1234567890":null,"2026-10-27T12:00:00Z":true,"/some/path":false}}
{"mapping":{"bob":{"count":3}}}"#, Framing::Lines);
    assert!(schema.fields.contains_key("/mapping/alice/text"));
    assert!(schema.fields.contains_key("/mapping/bob/count"));
    assert_eq!(schema.fields["/mapping/*"].present, 1);
    assert_eq!(schema.fields["/mapping/*"].occurrences, 5);
    assert_eq!(schema.fields["/mapping/*/text"].missing, 1);
    let output = serde_json::to_string(&schema).unwrap();
    for value in [
        "550e8400",
        "0123456789abcdef",
        "1234567890",
        "2026-10-27",
        "/some/path",
        "secret-",
    ] {
        assert!(!output.contains(value));
    }
    assert!(compare(&schema, &schema).unwrap().changes.is_empty());
}

#[test]
fn sibling_names_across_records_and_observers_are_grouped_after_fifty() {
    let mut combined = Observer::default();
    for i in 0..51 {
        let mut one = Observer::default();
        one.json(
            format!(r#"{{"mapping":{{"entry{i}":{{"text":"hidden"}},"fixed":{{"count":1}}}}}}"#)
                .as_bytes(),
        )
        .unwrap();
        combined.merge(one).unwrap();
    }
    let schema = combined.finish("synthetic".into(), vec![]);
    let field = &schema.fields["/mapping/*"];
    assert_eq!(
        (field.present, field.parents, field.occurrences),
        (51, 51, 102)
    );
    assert_eq!(schema.fields["/mapping/*/text"].missing, 51);
    assert_eq!(schema.fields["/mapping/*/count"].missing, 51);
    assert!(!serde_json::to_string(&schema).unwrap().contains("entry"));
    assert!(compare(&schema, &schema).is_ok());

    let mut observer = Observer::default();
    for i in 0..50 {
        observer
            .json(format!(r#"{{"entry{i}":0}}"#).as_bytes())
            .unwrap();
    }
    assert!(
        observer
            .finish("synthetic".into(), vec![])
            .fields
            .contains_key("/entry49")
    );
}

#[test]
fn forced_names_and_literal_path_markers_keep_their_paths() {
    let mut observer = Observer::new(["a/b".into(), "1234567890".into(), "fixed".into()].into());
    let mut object = serde_json::Map::new();
    for i in 0..51 {
        object.insert(format!("entry{i}"), json!(false));
    }
    object.extend(
        serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(
            r#"{"a/b":1,"*":true,"[]":[],"1234567890":null,"fixed":true}"#,
        )
        .unwrap(),
    );
    observer
        .json(serde_json::to_vec(&object).unwrap().as_slice())
        .unwrap();
    let schema = observer.finish("synthetic".into(), vec![]);
    for path in ["/a~1b", "/1234567890", "/fixed", "/*"] {
        assert!(schema.fields.contains_key(path));
    }
    assert_eq!(schema.fields["/*"].present, 1);
    let schema = infer(br#"{"*":true,"[]":[],"~":false}"#, Framing::Json);
    for path in ["/~2", "/~3", "/~0"] {
        assert!(schema.fields.contains_key(path));
    }
    assert!(compare(&schema, &schema).is_ok());
}

#[test]
fn unreadable_lines_are_counted_and_never_contribute_partial_fields() {
    let schema = infer(b"\n{\"id\":1,\n{}\r\n{\"count\":2}", Framing::Lines);
    assert_eq!((schema.records, schema.unreadable_records), (2, 1));
    assert!(!schema.fields.contains_key("/id"));
    assert_eq!(schema.fields["/count"].missing, 1);
}

#[test]
fn array_exports_stream_records_and_reject_a_malformed_tail() {
    let schema = infer(br#"[{"id":"a"},{"count":1.25}]"#, Framing::Array);
    assert_eq!(schema.records, 2);
    assert_eq!(schema.fields["/count"].types["number"], 1);
    let mut observer = Observer::default();
    assert!(
        observer
            .read(io::Cursor::new(b"[{},"), Framing::Array)
            .is_err()
    );
    assert_eq!(observer.counts(), (0, 0));
}

#[test]
fn long_export_records_have_no_size_cutoff() {
    let bytes = format!(
        "[{{\"text\":\"{}\"}},{{\"count\":1}}]",
        "x".repeat(33 * 1024 * 1024)
    );
    let schema = infer(bytes.as_bytes(), Framing::Array);
    assert_eq!(schema.records, 2);
    assert_eq!(schema.fields["/text"].types["string"], 1);
}

#[test]
fn row_dumps_include_table_column_and_revision_structure_without_sql_or_values() {
    let before = infer(
        br#"{"schema":{"threads":"CREATE TABLE threads (title TEXT)"}}
{"t":"threads","k":"secret-id","r":{"title":"private-title"}}"#,
        Framing::Lines,
    );
    let after = infer(
        br#"{"schema":{"threads":"CREATE TABLE threads (title TEXT, added_column INT)"}}
{"t":"threads","k":"secret-id","r":{"title":"private-title","added_column":1}}"#,
        Framing::Lines,
    );
    assert_eq!(before.fields["/schema/threads"].types["string"], 1);
    assert!(after.fields.contains_key("/r/added_column"));
    let diff = compare(&before, &after).unwrap();
    assert!(
        diff.changes
            .iter()
            .any(|c| c.path == "/r/added_column" && c.kind == ChangeKind::Added)
    );
    let output = serde_json::to_string(&after).unwrap();
    for value in ["CREATE TABLE", "private-title", "secret-id"] {
        assert!(!output.contains(value));
    }
}

#[test]
fn capture_logs_check_digest_and_do_not_accept_a_partial_prefix_as_complete() {
    use polylogue_model::capture_log::{Header, MAGIC};
    let record = br#"{"id":"hidden"}"#;
    let mut bytes = MAGIC.to_vec();
    bytes.extend(Header::of(record).encode());
    bytes.extend(record);
    assert_eq!(infer(&bytes, Framing::CaptureLog).records, 1);
    bytes.pop();
    let mut observer = Observer::default();
    assert!(
        observer
            .read(io::Cursor::new(&bytes), Framing::CaptureLog)
            .is_err()
    );
    assert_eq!(observer.counts(), (0, 0));
    assert_eq!(infer(b"{}", Framing::CaptureLog).records, 1);
}

#[test]
fn recursive_diff_distinguishes_add_remove_types_and_requiredness() {
    let before = infer(
        br#"{"message":{"text":"x","count":1},"id":"a"}
{"message":{"text":"y","count":2},"id":"b"}"#,
        Framing::Lines,
    );
    let after = infer(
        br#"{"message":{"text":null,"flag":true},"id":"c"}
{"message":{"text":"z","flag":true}}"#,
        Framing::Lines,
    );
    let diff = compare(&before, &after).unwrap();
    let changes: Vec<_> = diff
        .changes
        .iter()
        .map(|c| (c.path.as_str(), &c.kind))
        .collect();
    assert_eq!(
        changes,
        [
            ("/id", &ChangeKind::Required),
            ("/message/count", &ChangeKind::Removed),
            ("/message/flag", &ChangeKind::Added),
            ("/message/text", &ChangeKind::Types)
        ]
    );
    assert!(compare(&before, &before).unwrap().changes.is_empty());
    assert_eq!(diff, compare(&before, &after).unwrap());
}

#[test]
fn diff_labels_different_sample_populations_and_refuses_incompatible_rules() {
    let before = infer(b"{}", Framing::Json);
    let mut after = before.clone();
    after.origin = "another".into();
    after.selection.push(Selection {
        input: "another-input".into(),
        revision: Some(1),
        records: 1,
        unreadable_records: 0,
        unreadable: None,
    });
    let diff = compare(&before, &after).unwrap();
    assert!(!diff.comparable);
    assert_eq!(diff.reasons, ["different_origins", "different_samples"]);
    after.rule_version += 1;
    assert!(compare(&before, &after).is_err());
    after = before.clone();
    after.named_fields.insert("forced".into());
    assert!(compare(&before, &after).is_err());
    let mut malformed = serde_json::to_value(&before).unwrap();
    malformed["examples"] = json!(["private"]);
    assert!(serde_json::from_value::<Schema>(malformed).is_err());
}

#[test]
fn diff_refuses_malformed_paths_counts_and_escapes() {
    let before = infer(br#"{"id":1}"#, Framing::Json);
    for mutation in [0, 1, 2, 3] {
        let mut after = before.clone();
        match mutation {
            0 => {
                let field = after.fields.remove("/id").unwrap();
                after.fields.insert("id".into(), field);
            }
            1 => {
                after.fields.get_mut("/id").unwrap().parents += 1;
            }
            2 => {
                let field = after.fields.remove("/id").unwrap();
                after.fields.insert("/invalid~4escape".into(), field);
            }
            _ => {
                after.records += 1;
            }
        }
        assert!(compare(&before, &after).is_err());
    }
}

#[test]
fn array_element_changes_and_null_widening_are_type_findings() {
    let before = infer(br#"{"items":[{"text":"x"}]}"#, Framing::Json);
    let after = infer(br#"{"items":[{"text":null},3]}"#, Framing::Json);
    let changes = compare(&before, &after).unwrap().changes;
    assert!(
        changes
            .iter()
            .any(|c| c.path == "/items/[]" && c.kind == ChangeKind::Types)
    );
    assert!(
        changes
            .iter()
            .any(|c| c.path == "/items/[]/text" && c.kind == ChangeKind::Types)
    );
    assert!(!changes.iter().any(|c| c.kind == ChangeKind::Added));
}

#[test]
fn large_numbers_and_numeric_marker_keys_keep_their_json_types() {
    let schema = infer(br#"{"id":184467440737095516160,"count":1e400,"message":{"$serde_json::private::Number":"1"}}"#,Framing::Json);
    assert_eq!(schema.fields["/id"].types["integer"], 1);
    assert_eq!(schema.fields["/count"].types["number"], 1);
    assert_eq!(schema.fields["/message"].types["object"], 1);
    assert_eq!(
        schema.fields["/message/$serde_json::private::Number"].types["string"],
        1
    );
}

#[test]
fn nested_json_has_no_fixed_depth_cutoff() {
    let bytes = format!("{}0{}", "[".repeat(160), "]".repeat(160));
    let schema = infer(bytes.as_bytes(), Framing::Json);
    assert_eq!(schema.records, 1);
    assert_eq!(schema.fields.len(), 161);
}
