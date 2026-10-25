use super::*;
use serde_json::json;

#[test]
fn native_hook_failures_reach_ordinary_mcp_reads_and_budgeted_pages() {
    let daemon = test_daemon::start("human prompt");
    let source = daemon.home.join(".claude/projects/-p/s1.jsonl");
    let summary = json!({"type":"system", "subtype":"stop_hook_summary", "uuid":"hook1",
        "sessionId":"s1", "parentUuid":"u1", "hookErrors":["  synthetic hook failed λ\n"],
        "hookAdditionalContext":[], "preventedContinuation":true, "stopReason":"check blocked"});
    let mut original = std::fs::read(&source).unwrap();
    original.extend(serde_json::to_vec(&summary).unwrap());
    original.push(b'\n');
    std::fs::write(&source, &original).unwrap();
    let id = polylogue_model::SessionId::parse("claude-code:s1").unwrap();
    daemon.wait_for(|_| {
        daemon
            .daemon
            .reads
            .reader()
            .unwrap()
            .events(&id)
            .unwrap()
            .iter()
            .any(|event| event.data == summary)
    });
    let Response::Ok { data: typed } = read(&daemon.daemon, "full", "json", None, None) else {
        panic!("ordinary read failed")
    };
    assert_eq!(typed["messages"].as_array().unwrap().len(), 2);
    assert_eq!(typed["messages"][1]["prompt_kind"], "harness");
    let archive = Archive::at(&daemon.daemon.archive.root);
    let mcp: serde_json::Value = serde_json::from_str(
        &crate::mcp::call(
            &archive,
            "read",
            json!({"session":"claude-code:s1", "format":"markdown"})
                .as_object()
                .unwrap()
                .clone(),
        )
        .unwrap(),
    )
    .unwrap();
    let text = mcp["content"].as_str().unwrap();
    assert!(text.contains("synthetic hook failed λ"), "{text}");
    assert!(text.contains("Stop hook prevented continuation."));
    assert!(text.contains("check blocked"));
    let full = crate::mcp::call(
        &archive,
        "read",
        json!({"session":"claude-code:s1", "budget":10000})
            .as_object()
            .unwrap()
            .clone(),
    )
    .unwrap();
    let full: serde_json::Value = serde_json::from_str(&full).unwrap();
    let mut cursor = None;
    let mut joined = String::new();
    for _ in 0..200 {
        let mut args = json!({"session":"claude-code:s1", "budget":1});
        if let Some(token) = cursor.take() {
            args["cursor"] = token;
        }
        let page = crate::mcp::call(&archive, "read", args.as_object().unwrap().clone()).unwrap();
        let page: serde_json::Value = serde_json::from_str(&page).unwrap();
        joined.push_str(page["text"].as_str().unwrap());
        cursor = page.get("next").filter(|v| v.is_string()).cloned();
        if cursor.is_none() {
            break;
        }
    }
    assert!(cursor.is_none());
    assert_eq!(joined, full["text"].as_str().unwrap());
    assert!(joined.contains("synthetic hook failed λ"));
    assert_eq!(std::fs::read(source).unwrap(), original);
    assert!(
        daemon
            .daemon
            .reads
            .reader()
            .unwrap()
            .events(&id)
            .unwrap()
            .iter()
            .any(|event| event.data == summary)
    );
}
