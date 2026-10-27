use serde_json::Value;
use std::process::Command;

#[test]
fn public_morrow_tour_matches_current_query_examples() {
    let temp = tempfile::tempdir().unwrap();
    let directory = temp.path().join("tour");
    let operator_archive = temp.path().join("operator-archive");
    let output = Command::new(crate::BIN)
        .args(["--json", "demo", "tour"])
        .arg(&directory)
        .env("POLYLOGUE_ARCHIVE", &operator_archive)
        .output()
        .unwrap();
    assert!(!operator_archive.exists());
    assert!(output.status.success(), "{output:?}");
    let tour: Value = serde_json::from_slice(&output.stdout).unwrap();
    let steps = tour["steps"].as_array().unwrap();
    assert_eq!(steps.len(), 10);
    assert_eq!(tour["first_query_seconds"], steps[0]["seconds"]);
    assert!(tour["preparation_seconds"].as_f64().unwrap() >= 0.0);
    let examples: Value = serde_json::from_str(include_str!(
        "../../../../examples/morrow/expected/main/expected.json"
    ))
    .unwrap();
    assert_eq!(
        steps[0]["result"]["rows"].as_array().unwrap().len(),
        examples["queries"]["13 messages where written_by:human \"chrono\""]
            .as_array()
            .unwrap()
            .len()
    );
    let short = |row: &Value| {
        polylogue_testkit::short(
            row["session"]
                .as_str()
                .or_else(|| row["id"].as_str())
                .unwrap(),
        )
    };
    let mut errors: Vec<_> = steps[2]["result"]["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            (
                short(r),
                r["command"].as_str().unwrap().to_owned(),
                r["exit_code"].as_i64().unwrap(),
            )
        })
        .collect();
    let mut expected_errors: Vec<_> = examples["queries"]["15 tool_calls where outcome:error"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            (
                r[0].as_str().unwrap().to_owned(),
                r[1].as_str().unwrap().to_owned(),
                r[2].as_i64().unwrap(),
            )
        })
        .collect();
    errors.sort();
    expected_errors.sort();
    assert_eq!(errors, expected_errors);
    let mut children: Vec<_> = steps[3]["result"]["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(short)
        .collect();
    children.sort();
    assert_eq!(children, ["#3", "#4"]);
    let unknown: Vec<_> = steps[5]["result"]["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(short)
        .collect();
    let expected_unknown: Vec<_> = examples["queries"]["9 cost:unknown"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_owned())
        .collect();
    assert_eq!(unknown, expected_unknown);
    assert_eq!(steps[8]["result"]["rows"].as_array().unwrap().len(), 1);
    assert!(
        steps[9]["result"]
            .to_string()
            .contains("Use chrono for date parsing.")
    );
    assert!(
        steps
            .iter()
            .all(|s| !s["text"].as_str().unwrap().is_empty())
    );
    assert_eq!(
        std::fs::read_to_string(directory.join("transcript.md")).unwrap(),
        tour["transcript"].as_str().unwrap()
    );
    let marker = directory.join("keep.txt");
    std::fs::write(&marker, "keep this file").unwrap();
    let retry = Command::new(crate::BIN)
        .args(["demo", "tour"])
        .arg(&directory)
        .output()
        .unwrap();
    assert!(!retry.status.success());
    assert_eq!(std::fs::read_to_string(marker).unwrap(), "keep this file");
}
