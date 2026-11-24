//! A `tool:<session>#<call id>` reference selects that one call in the
//! `tool_calls` unit (docs/references.md).
use super::*;
use crate::fixture::{CallFacts, Fixture, MessageFacts, SessionFacts};
use serde_json::{Value, json};

fn rows(fixture: &Fixture, text: &str) -> Vec<Value> {
    let program = compile(text, &opts(&[])).unwrap_or_else(|error| panic!("{text}: {error}"));
    let reads = fixture.reads().unwrap();
    let reader = reads.reader().unwrap();
    let mut rows = reader.query_rows(&program).unwrap();
    for row in &mut rows {
        crate::query::execute::present(&program, row).unwrap();
    }
    rows
}

#[test]
fn a_tool_call_reference_selects_its_call() {
    let directory = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(directory.path()).unwrap();
    let mut call_ids = vec![];
    for (id, commands) in [
        ("claude-code:a", ["first", "second"]),
        ("claude-code:b", ["third", "fourth"]),
    ] {
        let session = fixture
            .session(SessionFacts {
                id,
                ..Default::default()
            })
            .unwrap();
        let message = fixture
            .message(
                session,
                MessageFacts {
                    id: &format!("{id}-m"),
                    role: "assistant",
                    written_by: "model",
                    ..Default::default()
                },
            )
            .unwrap();
        for (index, command) in commands.into_iter().enumerate() {
            fixture
                .tool_call(
                    session,
                    message,
                    CallFacts {
                        tool: "Bash",
                        action: "shell",
                        command: Some(command),
                        outcome: "ok",
                        ..Default::default()
                    },
                )
                .unwrap();
            // The fixture names a call after its message key and block position.
            call_ids.push(format!("c{message}-{index}"));
        }
    }
    fixture.derive().unwrap();
    assert_eq!(
        rows(
            &fixture,
            &format!(
                "tool_calls where tool:claude-code:a#{} | fields command",
                call_ids[1]
            )
        ),
        [json!({"command": "second"})]
    );
    assert_eq!(
        rows(
            &fixture,
            &format!(
                "tool_calls where tool:claude-code:b#{} | fields command",
                call_ids[2]
            )
        ),
        [json!({"command": "third"})]
    );
    // A call of another session is not selected by this session's reference.
    assert!(
        rows(
            &fixture,
            &format!("tool_calls where tool:claude-code:a#{}", call_ids[2])
        )
        .is_empty()
    );
    // A tool's name still matches by name.
    assert_eq!(rows(&fixture, "tool_calls where tool:Bash").len(), 4);
}
