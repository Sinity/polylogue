use super::*;
use polylogue_model::export::Document;
use serde_json::json;

const PREFIX: &str = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty";

#[test]
fn snippets_mask_a_whole_private_key_before_shortening_it() {
    run(
        "wholeprobe",
        "claude-code:snippet-whole",
        "claude-code:snippet-whole/1",
    );
}

#[test]
fn snippets_keep_hidden_earlier_source_context() {
    run(
        "prefixprobe",
        "claude-code:snippet-prefix",
        "claude-code:snippet-prefix/2",
    );
}

#[test]
fn output_snippets_use_the_same_source_masking_order() {
    run(
        "outputprobe",
        "claude-code:snippet-output",
        "claude-code:snippet-output/2",
    );
}

fn run(term: &str, expected_session: &str, expected_message: &str) {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let archive = dir.path().join("archive");
    let config = dir.path().join("config");
    std::fs::create_dir_all(&home).unwrap();
    let env = crate::Fixture::new(archive.as_path(), home.as_path(), config.as_path());
    let _daemon = start_daemon(&env);
    let seed: Document = serde_json::from_value(json!({"format":"polylogue-export", "version":1, "sessions":[
        {"session":{"id":"claude-code:snippet-whole", "kind":"main", "title":{"text":"Synthetic snippet", "source":"custom"}}, "messages":[
            {"id":"claude-code:snippet-whole/1", "session":"claude-code:snippet-whole", "role":"user", "prompt_kind":"prose", "blocks":[
                {"type":"text", "text":format!("-----BEGIN PRIVATE KEY----- {PREFIX} wholeprobe -----END PRIVATE KEY----- visible Łódź")}
            ]}
        ]},
        {"session":{"id":"claude-code:snippet-prefix", "kind":"main", "title":{"text":"Synthetic snippet", "source":"custom"}}, "messages":[
            {"id":"claude-code:snippet-prefix/1", "session":"claude-code:snippet-prefix", "role":"assistant", "prompt_kind":"harness", "blocks":[{"type":"text", "text":"-----BEGIN PRIVATE KEY----- private header"}]},
            {"id":"claude-code:snippet-prefix/2", "session":"claude-code:snippet-prefix", "role":"user", "prompt_kind":"prose", "blocks":[{"type":"text", "text":format!("{PREFIX} prefixprobe -----END PRIVATE KEY----- visible Łódź")}]},
            {"id":"claude-code:snippet-prefix/3", "session":"claude-code:snippet-prefix", "role":"assistant", "blocks":[{"type":"text", "text":"independently visible Łódź"}]}
        ]},
        {"session":{"id":"claude-code:snippet-output", "kind":"main", "title":{"text":"Synthetic snippet", "source":"custom"}}, "messages":[
            {"id":"claude-code:snippet-output/1", "session":"claude-code:snippet-output", "role":"assistant", "prompt_kind":"harness", "blocks":[{"type":"text", "text":"-----BEGIN PRIVATE KEY----- private header"}]},
            {"id":"claude-code:snippet-output/2", "session":"claude-code:snippet-output", "role":"user", "blocks":[{"type":"tool_result", "call_id":null, "outcome":{"kind":"ok"}, "output":format!("{PREFIX} outputprobe -----END PRIVATE KEY----- visible output")}]}
        ]}
    ]})).unwrap();
    let source = dir.path().join("snippets.json");
    let original = seed.to_json().unwrap();
    std::fs::write(&source, &original).unwrap();
    let imported = cli(&env, &["import", source.to_str().unwrap()]);
    assert!(imported.status.success(), "{imported:?}");
    wait_status(&env, |status| status["store"]["search_pending"] == 0);
    let query_term = if term == "outputprobe" {
        format!("output:{term}")
    } else {
        term.to_owned()
    };
    let query = cli(&env, &["--json", &query_term]);
    assert!(query.status.success(), "{query:?}");
    let answer = json(&query);
    assert_eq!(answer["count"], 1);
    assert_eq!(answer["rows"][0]["id"], expected_session);
    let snippet = &answer["rows"][0]["snippet"];
    assert_eq!(snippet["session"], expected_session);
    assert_eq!(snippet["message"], expected_message);
    assert!(
        !snippet["text"].as_str().unwrap().contains(term),
        "{answer}"
    );
    assert!(snippet["marks"].as_array().unwrap().is_empty());

    let visible = cli(&env, &["--json", "messages where lodz"]);
    assert!(visible.status.success(), "{visible:?}");
    let visible = json(&visible);
    assert_eq!(visible["count"], 3);
    for row in visible["rows"].as_array().unwrap() {
        let snippet = &row["snippet"];
        let text = snippet["text"].as_str().unwrap();
        assert!(
            !text.contains("wholeprobe") && !text.contains("prefixprobe"),
            "{row}"
        );
        let marks = snippet["marks"].as_array().unwrap();
        assert!(!marks.is_empty(), "{row}");
        for mark in marks {
            let start = mark[0].as_u64().unwrap() as usize;
            let end = mark[1].as_u64().unwrap() as usize;
            assert!(start <= end && end <= text.len());
            assert!(text.is_char_boundary(start) && text.is_char_boundary(end));
            assert_eq!(&text[start..end], "Łódź");
        }
        assert_eq!(snippet["message"], row["id"]);
    }
    let pool = polylogue_store::ReadPool::open(&polylogue_store::Archive::at(&archive)).unwrap();
    let reader = pool.reader().unwrap();
    for session in seed.sessions {
        assert_eq!(
            reader.messages(&session.session.id, false).unwrap(),
            session.messages
        );
    }
    assert_eq!(std::fs::read_to_string(source).unwrap(), original);
}
