use super::*;
use polylogue_model::export::Document;
use serde_json::json;

#[test]
fn public_read_and_export_mask_source_fragments_before_views_and_html_pairing() {
    run(&[
        "claude-code:masking",
        "claude-code:hidden",
        "claude-code:aliases",
    ]);
}

#[test]
fn public_tool_aliases_reuse_masked_source_text_without_replaying_it() {
    run(&["claude-code:aliases"]);
}

#[test]
fn public_views_mask_hidden_source_fragments_before_removing_them() {
    run(&["claude-code:hidden"]);
}

fn run(sessions: &[&str]) {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let archive = dir.path().join("archive");
    let config = dir.path().join("config");
    std::fs::create_dir_all(&home).unwrap();
    let env = [
        ("POLYLOGUE_HOME", home.as_path()),
        ("POLYLOGUE_ARCHIVE", archive.as_path()),
        ("XDG_CONFIG_HOME", config.as_path()),
    ];
    let _daemon = start_daemon(&env);
    let seed: Document = serde_json::from_value(json!({
        "format":"polylogue-export", "version":1, "sessions":[
            {"session":{"id":"claude-code:masking", "kind":"main", "title":{"text":"Synthetic masking example", "source":"custom"}}, "messages":[
                {"id":"claude-code:masking/1", "session":"claude-code:masking", "role":"user", "prompt_kind":"prose", "blocks":[{"type":"text", "text":"prefix -----BEGIN PRIVATE KEY-----private-one"}]},
                {"id":"claude-code:masking/2", "session":"claude-code:masking", "role":"assistant", "blocks":[{"type":"text", "text":"private-two-----END PRIVATE KEY----- visible λ🙂"}]},
                {"id":"claude-code:masking/3", "session":"claude-code:masking", "role":"user", "prompt_kind":"prose", "blocks":[{"type":"text", "text":"prefix -----BEGIN PRIVATE KEY-----private-one"}]},
                {"id":"claude-code:masking/4", "session":"claude-code:masking", "role":"assistant", "blocks":[{"type":"text", "text":"private-two-----END PRIVATE KEY----- visible λ🙂"}]}
            ]},
            {"session":{"id":"claude-code:hidden", "kind":"main", "title":{"text":"Synthetic masking example", "source":"custom"}}, "messages":[
                {"id":"claude-code:hidden/1", "session":"claude-code:hidden", "role":"user", "prompt_kind":"prose", "blocks":[
                    {"type":"tool_result", "call_id":null, "outcome":{"kind":"ok"}, "output":"-----BEGIN PRIVATE KEY-----private-hidden"},
                    {"type":"text", "text":"private-hidden-tail-----END PRIVATE KEY----- visible"}
                ]}
            ]},
            {"session":{"id":"claude-code:aliases", "kind":"main", "title":{"text":"Synthetic masking example", "source":"custom"}}, "messages":[
                {"id":"claude-code:aliases/1", "session":"claude-code:aliases", "role":"user", "blocks":[{"type":"text", "text":"-----BEGIN PRIVATE KEY-----private-prefix"}]},
                {"id":"claude-code:aliases/2", "session":"claude-code:aliases", "role":"assistant", "blocks":[
                    {"type":"tool_call", "call_id":"c1", "tool":"Bash", "action":"shell", "command":"private-command-tail-----END PRIVATE KEY----- visible command", "input":{"command":"private-command-tail-----END PRIVATE KEY----- visible command", "notes":"independent notes remain visible", "number":0, "enabled":false, "empty":"", "nil":null}}
                ]},
                {"id":"claude-code:aliases/3", "session":"claude-code:aliases", "role":"user", "blocks":[
                    {"type":"tool_result", "call_id":"c1", "outcome":{"kind":"ok"}, "output":"-----BEGIN PRIVATE KEY-----private-result", "data":{"output":"-----BEGIN PRIVATE KEY-----private-result", "notes":"independent result notes"}}
                ]},
                {"id":"claude-code:aliases/4", "session":"claude-code:aliases", "role":"assistant", "blocks":[{"type":"text", "text":"private-result-tail-----END PRIVATE KEY----- visible result"}]}
            ]}
        ]
    })).unwrap();
    let source = dir.path().join("seed.json");
    let original = seed.to_json().unwrap();
    std::fs::write(&source, &original).unwrap();
    let imported = cli(&env, &["import", source.to_str().unwrap()]);
    assert!(imported.status.success(), "{imported:?}");
    for &session in sessions {
        for format in ["terminal", "markdown", "html", "json"] {
            let read = cli(&env, &["--json", "read", session, "--format", format]);
            assert!(read.status.success(), "{read:?}");
            assert_masked(&read, &format!("read {session} {format}"));
            if format != "terminal" {
                let exported = cli(&env, &["export", session, "--format", format]);
                assert!(exported.status.success(), "{exported:?}");
                assert_masked(&exported, &format!("export {session} {format}"));
            }
        }
    }
    for view in ["dialogue", "prompts"] {
        for format in ["terminal", "markdown", "html", "json"] {
            let out = cli(
                &env,
                &[
                    "--json",
                    "read",
                    "claude-code:hidden",
                    "--view",
                    view,
                    "--format",
                    format,
                ],
            );
            assert!(out.status.success(), "{out:?}");
            assert_masked(&out, &format!("hidden {view} {format}"));
        }
    }
    let out = cli(&env, &["--json", "read", "claude-code:aliases"]);
    let value = json(&out);
    let tool = &value["messages"][1]["blocks"][0];
    assert_eq!(tool["command"], tool["input"]["command"]);
    assert_eq!(tool["input"]["notes"], "independent notes remain visible");
    assert_eq!(tool["input"]["number"], 0);
    assert_eq!(tool["input"]["enabled"], false);
    assert_eq!(tool["input"]["empty"], "");
    assert!(tool["input"]["nil"].is_null());
    let pool = polylogue_store::ReadPool::open(&polylogue_store::Archive::at(&archive)).unwrap();
    let reader = pool.reader().unwrap();
    for session in &seed.sessions {
        assert_eq!(
            reader.messages(&session.session.id, false).unwrap(),
            session.messages
        );
    }
    assert_eq!(std::fs::read_to_string(source).unwrap(), original);
}

fn assert_masked(output: &Output, context: &str) {
    let text = String::from_utf8_lossy(&output.stdout);
    for secret in [
        "private-one",
        "private-two",
        "private-hidden",
        "private-prefix",
        "private-command-tail",
        "private-result",
    ] {
        assert!(!text.contains(secret), "{context}: {secret}: {text}");
    }
    assert!(text.contains("visible"), "{context}: {text}");
}

#[test]
fn public_derived_titles_mask_source_context_before_taking_the_first_line() {
    const SECRET: &str = "sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA-AAAAAAAA";
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let archive = dir.path().join("archive");
    let config = dir.path().join("config");
    std::fs::create_dir_all(&home).unwrap();
    let env = [
        ("POLYLOGUE_HOME", home.as_path()),
        ("POLYLOGUE_ARCHIVE", archive.as_path()),
        ("XDG_CONFIG_HOME", config.as_path()),
    ];
    let _daemon = start_daemon(&env);
    let seed: Document = serde_json::from_value(json!({"format":"polylogue-export", "version":1, "sessions":[
        {"session":{"id":"claude-code:title-context", "kind":"main"}, "messages":[
            {"id":"claude-code:title-context/1", "session":"claude-code:title-context", "role":"user", "prompt_kind":"prose", "blocks":[
                {"type":"tool_result", "call_id":null, "outcome":{"kind":"ok"}, "output":"-----BEGIN PRIVATE KEY-----private-title-head"},
                {"type":"text", "text":"private-title-tail-----END PRIVATE KEY----- visible title"}
            ]}
        ]},
        {"session":{"id":"claude-code:title-prefix", "kind":"main"}, "messages":[
            {"id":"claude-code:title-prefix/1", "session":"claude-code:title-prefix", "role":"assistant", "prompt_kind":"harness", "blocks":[
                {"type":"text", "text":"-----BEGIN PRIVATE KEY-----private-title-prefix"}
            ]},
            {"id":"claude-code:title-prefix/2", "session":"claude-code:title-prefix", "role":"user", "prompt_kind":"prose", "blocks":[
                {"type":"text", "text":"private-title-prefix-tail-----END PRIVATE KEY----- visible prefix title"}
            ]}
        ]},
        {"session":{"id":"claude-code:title-truncated", "kind":"main"}, "messages":[
            {"id":"claude-code:title-truncated/1", "session":"claude-code:title-truncated", "role":"user", "prompt_kind":"prose", "blocks":[
                {"type":"text", "text":format!("{} {SECRET} visible suffix", "x".repeat(40))}
            ]}
        ]}
    ]})).unwrap();
    let source = dir.path().join("titles.json");
    std::fs::write(&source, seed.to_json().unwrap()).unwrap();
    let imported = cli(&env, &["import", source.to_str().unwrap()]);
    assert!(imported.status.success(), "{imported:?}");
    for id in [
        "claude-code:title-context",
        "claude-code:title-prefix",
        "claude-code:title-truncated",
    ] {
        let read = cli(&env, &["--json", "read", id]);
        assert!(read.status.success(), "{read:?}");
        let read = json(&read);
        assert!(read["session"]["title"].is_null());
        let shown = read["display_title"].as_str().unwrap();
        assert!(!shown.contains("private-title"), "{shown}");
        assert!(!shown.contains("sk-ant-api03"), "{shown}");
        if id.ends_with("context") {
            assert_eq!(shown, "visible title");
        }
        if id.ends_with("prefix") {
            assert_eq!(shown, "visible prefix title");
        }
        for format in ["terminal", "markdown", "html"] {
            let read = cli(&env, &["--json", "read", id, "--format", format]);
            let text = String::from_utf8(read.stdout).unwrap();
            assert!(!text.contains("private-title"), "{text}");
            assert!(!text.contains("sk-ant-api03"), "{text}");
        }
    }
    let query = cli(&env, &["--json", "sessions"]);
    assert!(query.status.success(), "{query:?}");
    let query = json(&query);
    assert_eq!(query["rows"].as_array().unwrap().len(), 3);
    for row in query["rows"].as_array().unwrap() {
        let title = row["title"].as_str().unwrap();
        assert!(!title.contains("private-title"), "{row}");
        assert!(!title.contains("sk-ant-api03"), "{row}");
    }
    let pool = polylogue_store::ReadPool::open(&polylogue_store::Archive::at(&archive)).unwrap();
    let reader = pool.reader().unwrap();
    for session in seed.sessions {
        assert!(
            reader
                .session(&session.session.id)
                .unwrap()
                .unwrap()
                .title
                .is_none()
        );
        assert_eq!(
            reader.messages(&session.session.id, false).unwrap(),
            session.messages
        );
    }
}
