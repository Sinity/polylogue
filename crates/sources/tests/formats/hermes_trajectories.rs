use crate::support::{events, messages, parse, sessions};
use polylogue_sources::origin::by_name;
use serde_json::{Value, json};
use std::path::Path;

fn trajectory(id: &str, steps: Value) -> Value {
    json!({"schema_version":"ATIF-v1.7","trajectory_id":id,"session_id":"shared-run","agent":{"name":"synthetic-agent","version":"1","model_name":"native-model"},"steps":steps})
}
fn documents(parsed: &Value) -> Vec<&Value> {
    parsed["output"]["documents"]
        .as_array()
        .map(|rows| rows.iter().map(|row| &row["document"]).collect())
        .unwrap_or_default()
}

#[test]
fn atif_transcript_keeps_text_tools_images_and_embedded_scope_without_fabricated_sessions() {
    let mut root = trajectory(
        "root",
        json!([
            {"step_id":1,"source":"user","timestamp":"2026-01-02T03:04:05Z","message":[{"type":"text","text":" exact λ🙂 "},{"type":"image","source":{"media_type":"image/png","path":"images/neutral.png"}}]},
            {"step_id":2,"source":"agent","message":"dispatch","llm_call_count":0,
             "tool_calls":[{"tool_call_id":"a","function_name":"terminal","arguments":{"command":"printf synthetic","supplied":false}},{"tool_call_id":"b","function_name":"read_file","arguments":{"path":"neutral.rs"}}],
             "observation":{"results":[{"source_call_id":"b","content":"second result","extra":{"status":"success"}},{"source_call_id":"a","content":" first result\n"}]}},
            {"step_id":3,"source":"system","message":"","observation":{"results":[{"content":"unowned result","subagent_trajectory_ref":[{"trajectory_id":"child-a"}]}]}}
        ]),
    );
    root["subagent_trajectories"] = json!([
        trajectory(
            "child-a",
            json!([{"step_id":1,"source":"agent","message":"child A","llm_call_count":null}])
        ),
        trajectory(
            "child-b",
            json!([{"step_id":1,"source":"agent","message":"child B","llm_call_count":1}])
        )
    ]);
    let raw = serde_json::to_vec_pretty(&root).unwrap();
    let origin = by_name("hermes").unwrap();
    assert!((origin.recognize)(
        Path::new("trajectory-synthetic.json"),
        &raw[..64]
    ));
    let parsed = parse("hermes", "trajectory-synthetic.json", &raw);
    assert!(sessions(&parsed).is_empty());
    assert!(messages(&parsed).is_empty());
    assert!(parsed["output"]["links"].as_array().unwrap().is_empty());
    let docs = documents(&parsed);
    assert_eq!(docs.len(), 5);
    let ids: std::collections::BTreeSet<_> =
        docs.iter().map(|d| d["id"].as_str().unwrap()).collect();
    assert_eq!(
        ids.len(),
        5,
        "sibling trajectories may share a run ID and restart step IDs"
    );
    for doc in &docs {
        assert_eq!(doc["purpose"], "transcript");
        assert_eq!(doc["authorship"], "unknown");
        assert!(doc.get("session").is_none());
    }
    assert_eq!(docs[0]["blocks"][0]["text"], " exact λ🙂 ");
    let image = &docs[0]["blocks"][1];
    assert_eq!(image["type"], "image");
    assert_eq!(image["direction"], "unknown");
    assert_eq!(image["source_id"], "images/neutral.png");
    assert_eq!(image["media_type"], "image/png");
    assert!(image.get("blob").is_none());
    assert_eq!(docs[0]["source_time"], "2026-01-02T03:04:05Z");
    assert!(docs[0]["metadata"].get("llm_call_count").is_none());
    assert_eq!(docs[1]["metadata"]["llm_call_count"], 0);
    let blocks = docs[1]["blocks"].as_array().unwrap();
    assert_eq!(blocks[1]["call_id"], "a");
    assert_eq!(
        blocks[1]["input"],
        json!({"command":"printf synthetic","supplied":false})
    );
    assert_eq!(blocks[2]["call_id"], "b");
    assert_eq!(blocks[3]["call_id"], "b");
    assert_eq!(blocks[3]["outcome"]["kind"], "unknown");
    assert_eq!(blocks[3]["data"]["extra"]["status"], "success");
    assert_eq!(blocks[4]["call_id"], "a");
    assert_eq!(blocks[4]["output"], " first result\n");
    assert!(docs[2]["blocks"][1]["call_id"].is_null());
    assert_eq!(
        docs[2]["blocks"][1]["data"]["subagent_trajectory_ref"][0]["trajectory_id"],
        "child-a"
    );
    assert!(docs[3]["metadata"].get("llm_call_count").unwrap().is_null());
    assert_eq!(docs[4]["metadata"]["llm_call_count"], 1);
    let facts = events(&parsed, "hermes_trajectory");
    assert_eq!(facts.len(), 3);
    for fact in facts {
        assert!(fact.get("session").is_none());
        let metadata = fact["data"].to_string();
        assert!(!metadata.contains("exact λ") && !metadata.contains("first result"));
    }
}

#[test]
fn atif_future_versions_are_explicitly_unsupported_and_foreign_schema_keys_do_not_claim_hermes() {
    let mut value = trajectory(
        "future",
        json!([{"step_id":1,"source":"agent","message":"stored source text"}]),
    );
    value["schema_version"] = json!("ATIF-v99");
    let parsed = parse(
        "hermes",
        "trajectory-future.json",
        &serde_json::to_vec(&value).unwrap(),
    );
    assert!(documents(&parsed).is_empty());
    assert!(messages(&parsed).is_empty());
    assert!(sessions(&parsed).is_empty());
    assert!(
        parsed["accounting"]["unknown"]
            .as_object()
            .unwrap()
            .keys()
            .any(|s| s.contains("unsupported ATIF"))
    );
    assert!(!(by_name("hermes").unwrap().recognize)(
        Path::new("foreign.json"),
        br#"{"schema_version":"other-format"}"#
    ));
}

#[test]
fn atif_native_timing_and_provider_ids_do_not_become_authorship_or_guessed_time() {
    let value = trajectory(
        "metadata",
        json!([{
            "step_id":1,"source":"agent","message":"readable source text","timestamp":17.5,"llm_call_count":null,
            "tool_calls":[{"tool_call_id":"native-call","function_name":"terminal","arguments":{},"extra":{"provider_data":{"call_id":"provider-call","response_item_id":"provider-item","input":"unpromoted request text"}}}],
            "extra":{"invocation":{"invocation_id":"native-invocation","framework":"synthetic","status":"supplied","start_timestamp":17.5,"end_timestamp":18.75,"input":"unpromoted request text"},"ancestry":{"function_id":"native-function","parent_id":"native-parent"}}
        }]),
    );
    let raw = serde_json::to_vec(&value).unwrap();
    let parsed = parse("hermes", "trajectory-metadata.json", &raw);
    let docs = documents(&parsed);
    assert_eq!(docs.len(), 1);
    let doc = docs[0];
    assert!(doc.get("source_time").is_none());
    assert_eq!(doc["authorship"], "unknown");
    assert!(doc["metadata"].get("llm_call_count").unwrap().is_null());
    assert_eq!(doc["metadata"]["timestamp"], 17.5);
    assert_eq!(doc["metadata"]["invocation"]["start_timestamp"], 17.5);
    assert_eq!(doc["metadata"]["invocation"]["end_timestamp"], 18.75);
    assert_eq!(doc["metadata"]["ancestry"]["parent_id"], "native-parent");
    assert_eq!(
        doc["metadata"]["tool_call_evidence"][0]["provider_data"]["call_id"],
        "provider-call"
    );
    assert!(
        !doc["metadata"]
            .to_string()
            .contains("unpromoted request text")
    );
    assert!(!events(&parsed, "hermes_import_diagnostic").is_empty());
}

#[test]
fn atif_result_content_parts_form_one_ordered_text_stream_without_a_second_text_copy() {
    let first = "-----BEGIN PRIVATE KEY-----\n";
    let last = "synthetic-private-tail\n-----END PRIVATE KEY-----";
    let value = trajectory(
        "result-parts",
        json!([{"step_id":1,"source":"agent","message":"","tool_calls":[{"tool_call_id":"c","function_name":"terminal","arguments":{}}],"observation":{"results":[{"source_call_id":"c","content":[{"type":"text","text":first},{"type":"image","source":{"media_type":"image/png","path":"images/result.png"}},{"type":"text","text":last}],"extra":{"status":"supplied"}}]}}]),
    );
    let parsed = parse(
        "hermes",
        "trajectory-result-parts.json",
        &serde_json::to_vec(&value).unwrap(),
    );
    let doc = documents(&parsed)[0];
    let result = &doc["blocks"][2];
    assert_eq!(result["output"], format!("{first}{last}"));
    assert!(result["data"].get("content").is_none());
    assert_eq!(result["data"]["extra"]["status"], "supplied");
    assert_eq!(result["attachments"][0]["direction"], "unknown");
    assert_eq!(result["attachments"][0]["source_id"], "images/result.png");
}
