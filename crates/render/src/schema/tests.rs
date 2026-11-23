use super::*;
use serde_json::json;

#[test]
fn plain_and_json_reports_preserve_change_details_and_sample_limits() {
    let data = json!({"rule_version":1,"comparable":false,"before":{"origin":"synthetic","sample":true,"records":2,"unreadable_inputs":0,"unreadable_records":0},"after":{"origin":"synthetic","sample":true,"records":1,"unreadable_inputs":1,"unreadable_records":0},"reasons":["different_samples"],"changes":[{"path":"/text","kind":"types","before":{"types":{"string":2},"required":true},"after":{"types":{"null":1},"required":false}}]});
    let mut bytes = Vec::new();
    render(&data, Format::Terminal, &mut bytes).unwrap();
    let text = String::from_utf8(bytes).unwrap();
    assert_eq!(
        text,
        "Rule version: 1\n\
         before: synthetic (sample: true), 2 records, 0 unreadable inputs, 0 unreadable records\n\
         after: synthetic (sample: true), 1 records, 1 unreadable inputs, 0 unreadable records\n\
         Comparable samples: false\n\
         different_samples\n\
         types /text: types {\"string\":2} -> {\"null\":1}, required true -> false\n"
    );
    let mut bytes = Vec::new();
    render(&data, Format::Json, &mut bytes).unwrap();
    assert_eq!(serde_json::from_slice::<Value>(&bytes).unwrap(), data);
    let mut same = data.clone();
    same["changes"] = json!([]);
    let mut bytes = Vec::new();
    render(&same, Format::Terminal, &mut bytes).unwrap();
    assert_eq!(
        String::from_utf8(bytes).unwrap(),
        "Rule version: 1\n\
         before: synthetic (sample: true), 2 records, 0 unreadable inputs, 0 unreadable records\n\
         after: synthetic (sample: true), 1 records, 1 unreadable inputs, 0 unreadable records\n\
         Comparable samples: false\n\
         different_samples\n\
         No schema differences\n"
    );
}

#[test]
fn plain_reports_escape_controls_in_names_without_changing_json() {
    let sample = json!({"origin":"synthetic\u{1b}[31m","sample":true,"records":1,"unreadable_inputs":0,"unreadable_records":0,"selection":[{"input":"sample\nnext","revision":1}]});
    let data = json!({"rule_version":2,"comparable":true,"before":sample,"after":sample,"reasons":[],"changes":[{"path":"/é\nfield","kind":"added","before":null,"after":{"types":{"string":1},"required":true}}]});
    let mut bytes = Vec::new();
    render(&data, Format::Terminal, &mut bytes).unwrap();
    let text = String::from_utf8(bytes).unwrap();
    assert!(!text.contains('\u{1b}'));
    assert!(text.contains(r"synthetic\x1b[31m"));
    assert!(text.contains(r"sample\nnext"));
    assert!(text.contains(r"/é\nfield"));
    let mut bytes = Vec::new();
    render(&data, Format::Json, &mut bytes).unwrap();
    assert_eq!(serde_json::from_slice::<Value>(&bytes).unwrap(), data);
}

#[test]
fn plain_reports_name_the_table_of_a_tool_database_change() {
    let sample = json!({"origin":"synthetic","sample":true,"records":1,"unreadable_inputs":0,"unreadable_records":0});
    let field = |table: &str| json!({"table":table,"path":"/value","kind":"types","before":{"types":{"string":1},"required":true},"after":{"types":{"null":1},"required":true}});
    let data = json!({"rule_version":2,"comparable":true,"before":sample,"after":sample,"reasons":[],"changes":[field("notes"),field("tags\nx")]});
    let mut bytes = Vec::new();
    render(&data, Format::Terminal, &mut bytes).unwrap();
    let text = String::from_utf8(bytes).unwrap();
    assert_eq!(
        text,
        "Rule version: 2\n\
         before: synthetic (sample: true), 1 records, 0 unreadable inputs, 0 unreadable records\n\
         after: synthetic (sample: true), 1 records, 0 unreadable inputs, 0 unreadable records\n\
         Comparable samples: true\n\
         table notes: types /value: types {\"string\":1} -> {\"null\":1}, required true -> true\n\
         table tags\\nx: types /value: types {\"string\":1} -> {\"null\":1}, required true -> true\n"
    );
}
