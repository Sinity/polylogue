use crate::support::{events, messages, parse, sessions};
use polylogue_sources::origin::by_name;
use serde_json::json;
use std::path::Path;

#[test]
fn hermes_atof_reports_are_input_owned_observations_without_conversations() {
    let raw = include_bytes!("../../../../examples/sources/hermes/atof/input.jsonl");
    let path = ".hermes/observability/nemo-relay/atof/events.jsonl";
    let origin = by_name("hermes").unwrap();
    assert!((origin.recognize)(
        Path::new(path),
        &polylogue_sources::document::Head::new(&raw[..64])
    ));
    let parsed = parse("hermes", path, raw);
    assert!(sessions(&parsed).is_empty());
    assert!(messages(&parsed).is_empty());
    let start = events(&parsed, "hermes_tool_start");
    let end = events(&parsed, "hermes_tool_end");
    assert_eq!((start.len(), end.len()), (1, 1));
    assert!(start[0].get("session").is_none());
    assert!(end[0].get("session").is_none());
    assert_eq!(start[0]["data"]["uuid"], end[0]["data"]["uuid"]);
    assert_eq!(start[0]["data"]["data"]["command"], "printf synthetic");
    assert_eq!(end[0]["data"]["data"]["stdout"], "synthetic result\n");
    assert_eq!(end[0]["data"]["data"]["duration_ms"], 1000);
    let approval = events(&parsed, "hermes_approval_response");
    assert_eq!(approval.len(), 1);
    assert_eq!(approval[0]["data"]["data"]["approved"], false);
    assert_eq!(approval[0]["data"]["metadata"]["session_id"], "default");
    assert_eq!(
        approval[0]["data"]["data"]["reason"],
        " synthetic reason λ "
    );
    assert!(approval[0].get("at").is_none());
    assert_eq!(approval[0]["data"]["timestamp"], "not-a-time");
    assert!(!events(&parsed, "hermes_import_diagnostic").is_empty());
}

#[test]
fn hermes_atof_future_versions_and_bad_scope_phases_are_skipped_with_their_reason() {
    for (mut value, reason) in [
        (
            json!({"atof_version":"9","kind":"mark","uuid":"synthetic","timestamp":"2026-01-02T00:00:00Z","name":"future"}),
            "atof: unsupported ATOF version; original record kept",
        ),
        (
            json!({"atof_version":"0.1","kind":"scope","scope_category":"future","category":"tool","uuid":"synthetic","timestamp":"2026-01-02T00:00:00Z","name":"terminal"}),
            "atof: unsupported ATOF envelope or scope phase; original record kept",
        ),
    ] {
        value["data"] = json!({"text":"literal kept evidence"});
        let raw = crate::support::jsonl(&[value]);
        let parsed = parse("hermes", "synthetic.atof.jsonl", &raw);
        assert!(sessions(&parsed).is_empty());
        assert!(messages(&parsed).is_empty());
        assert!(parsed["output"]["events"].as_array().unwrap().is_empty());
        assert_eq!(parsed["counts"]["skipped"], json!({ (reason): 1 }));
    }
}

#[test]
fn verification_rows_are_kept_without_guesses() {
    let rows = [
        json!({"schema":{"version":1},"source":{"machine":"synthetic","path":"/neutral/profile/verification_evidence.db"}}),
        json!({"t":"meta","k":["schema_version"],"r":{"key":"schema_version","value":"1"}}),
        json!({"t":"verification_events","k":[1],"r":{"id":1,"created_at":"2026-01-02T03:04:05Z","session_id":"default","cwd":"/neutral","root":"/neutral","command":" printf synthetic λ ","canonical_command":"printf synthetic","kind":"test","scope":"workspace","status":"passed","exit_code":0,"output_summary":" synthetic output\n"}}),
        json!({"t":"verification_events","k":[2],"r":{"id":2,"created_at":"bad-time","session_id":"native-run","status":"future","exit_code":null,"output_summary":"literal error-looking text"}}),
        json!({"t":"verification_state","k":["default","/neutral"],"r":{"session_id":"default","root":"/neutral","last_event_id":1,"last_edit_at":null,"changed_paths_json":"[\" λ.rs \"]"}}),
    ];
    let raw = crate::support::jsonl(&rows);
    let path = ".hermes/verification_evidence.db.jsonl";
    assert!((by_name("hermes").unwrap().recognize)(
        Path::new(path),
        &polylogue_sources::document::Head::new(&raw[..64])
    ));
    let parsed = parse("hermes", path, &raw);
    assert!(sessions(&parsed).is_empty());
    assert!(messages(&parsed).is_empty());
    let found = events(&parsed, "hermes_verification_event");
    assert_eq!(found.len(), 2);
    for (event, original) in found.iter().zip(&rows[2..4]) {
        assert!(event.get("session").is_none());
        assert_eq!(&event["data"], original);
    }
    assert_eq!(found[0]["at"], "2026-01-02T03:04:05Z");
    assert!(found[1].get("at").is_none());
    assert_eq!(events(&parsed, "hermes_import_diagnostic").len(), 1);
    let state = events(&parsed, "hermes_verification_state");
    assert_eq!(state[0]["data"], rows[4]);
    assert!(state[0].get("at").is_none());
    let export = events(&parsed, "hermes_verification_export");
    assert_eq!(export[0]["data"], rows[0]);
}

#[test]
fn hermes_verification_needs_its_schema_not_markers() {
    for version in [json!(null), json!(1), json!("2")] {
        let raw = crate::support::jsonl(&[
            json!({"t":"verification_events","k":[1],"r":{"id":1,"status":"passed","exit_code":0}}),
            json!({"t":"meta","k":["schema_version"],"r":{"key":"schema_version","value":version}}),
        ]);
        let parsed = parse("hermes", "verification_evidence.db.jsonl", &raw);
        assert!(sessions(&parsed).is_empty());
        assert!(messages(&parsed).is_empty());
        assert!(parsed["output"]["events"].as_array().unwrap().is_empty());
        let reason = "unsupported or absent Hermes verification schema version; original rows kept";
        assert_eq!(
            parsed["counts"]["skipped"],
            json!({
                (format!("verification_events: {reason}")): 1,
                (format!("meta: {reason}")): 1,
            })
        );
    }
    let marker = br#"{"polylogue_artifact":"hermes_verification_evidence_db","verification_db_path":"/neutral/never-open.db"}"#;
    assert!(!(by_name("hermes").unwrap().recognize)(
        Path::new("marker.json"),
        &polylogue_sources::document::Head::new(marker)
    ));
}

#[test]
fn hermes_verification_history_keeps_row_occurrences_under_their_observed_schema() {
    let first = json!({"t":"verification_events","k":[1],"r":{"id":1,"created_at":"2026-01-02T03:04:05Z","session_id":"default","status":"passed","exit_code":0}});
    let state = |at: &str, paths: &str| json!({"t":"verification_state","k":["default","/neutral"],"r":{"session_id":"default","root":"/neutral","last_event_id":1,"last_edit_at":at,"changed_paths_json":paths}});
    let states = [
        state("2026-01-02T03:04:06Z", "[]"),
        state("2026-01-02T03:04:07Z", "[\"neutral.rs\"]"),
    ];
    let mut rows = vec![
        json!({"schema":{"meta":"synthetic","verification_events":"synthetic","verification_state":"synthetic"}}),
        json!({"t":"meta","k":["schema_version"],"r":{"key":"schema_version","value":"1"}}),
        first.clone(),
        states[0].clone(),
        states[1].clone(),
    ];
    for phase in 0..3 {
        if phase == 1 {
            rows.push(
                json!({"t":"meta","k":["schema_version"],"r":{"key":"schema_version","value":"2"}}),
            );
            rows.push(json!({"t":"verification_events","k":[2],"r":{"id":2,"created_at":"2026-01-02T03:04:08Z","session_id":"default","status":"future"}}));
        }
        if phase == 2 {
            rows.push(
                json!({"t":"meta","k":["schema_version"],"r":{"key":"schema_version","value":"1"}}),
            );
            rows.push(json!({"t":"verification_events","k":[3],"r":{"id":3,"created_at":"2026-01-02T03:04:09Z","session_id":"default","status":"passed","exit_code":0}}));
            rows.push(json!({"t":"verification_events","kh":"synthetic-key-digest","gone":true}));
        }
        let raw = crate::support::jsonl(&rows);
        let parsed = parse("hermes", "verification_evidence.db.jsonl", &raw);
        let found = events(&parsed, "hermes_verification_event");
        assert_eq!(
            found.len(),
            if phase == 2 { 2 } else { 1 },
            "the schema applies at each original row occurrence"
        );
        if phase == 2 {
            assert_eq!(found[1]["data"]["r"]["id"], 3);
            let deleted = events(&parsed, "hermes_verification_row_deleted");
            assert_eq!(deleted.len(), 1);
            assert_eq!(deleted[0]["data"], rows.last().unwrap().clone());
        }
        assert_eq!(found[0]["data"], first);
        let observed = events(&parsed, "hermes_verification_state");
        assert_eq!(
            observed.len(),
            2,
            "row versions are recorded observations, not a guessed latest-only table"
        );
        assert_eq!(observed[0]["data"], states[0]);
        assert_eq!(observed[1]["data"], states[1]);
        assert!(sessions(&parsed).is_empty());
        assert!(messages(&parsed).is_empty());
    }
}

#[test]
fn a_deleted_meta_row_clears_the_verification_schema_version() {
    let event = |id: u64| json!({"t":"verification_events","k":[id],"r":{"id":id,"status":"passed","exit_code":0}});
    let raw = crate::support::jsonl(&[
        json!({"t":"meta","k":["schema_version"],"r":{"key":"schema_version","value":"1"}}),
        event(1),
        json!({"t":"meta","k":["schema_version"],"gone":true}),
        event(2),
    ]);
    let parsed = parse("hermes", "verification_evidence.db.jsonl", &raw);
    assert_eq!(events(&parsed, "hermes_verification_event").len(), 1);
    assert_eq!(events(&parsed, "hermes_verification_row_deleted").len(), 1);
    assert!(!parsed["counts"]["skipped"].as_object().unwrap().is_empty());
}
