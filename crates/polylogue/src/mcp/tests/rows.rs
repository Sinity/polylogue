use super::*;

fn invoke(archive: &Archive, tool: Tool, call: Value) -> (String, Answer<Value>, bool) {
    let (text, structured, failed) = call_with_limit(
        archive,
        tool.name(),
        args(call),
        None,
        &CancellationToken::new(),
        None,
    )
    .unwrap();
    (text, serde_json::from_value(structured).unwrap(), failed)
}

#[test]
fn mcp_rows_default_operation() {
    let daemon = test_daemon::start("a prompt for the listing");
    let (text, answer, failed) = invoke(
        &daemon.daemon.archive,
        Tool::Read,
        json!({"arguments":{"selection":"sessions"}}),
    );
    assert!(!failed, "{text}");
    let value = answer.value().unwrap();
    assert_eq!(value["kind"], "listing");
    assert_eq!(value["reply"]["count"], 1);
    assert!(text.contains("claude-code:s1"), "{text}");
}

#[test]
fn mcp_retry_without_request_id_repeats() {
    let daemon = test_daemon::start("a prompt for the note");
    let call = json!({"arguments":{"selection":"session:claude-code:s1","steps":[{"verb":"note","text":"The timeout follows the read."}]}});
    let (_, first, failed) = invoke(&daemon.daemon.archive, Tool::Write, call.clone());
    assert!(!failed, "{first:?}");
    // Moving a retry to a larger tool keeps the home tool's request identity.
    let (_, second, failed) = invoke(&daemon.daemon.archive, Tool::Act, call);
    assert!(!failed, "{second:?}");
    let first = first.value().unwrap();
    let second = second.value().unwrap();
    assert_eq!(
        first["reply"]["steps"][0]["outcome"]["reply"]["reply"]["repeat"],
        false
    );
    assert_eq!(
        second["reply"]["steps"][0]["outcome"]["reply"]["reply"]["repeat"],
        true
    );
    let (_, notes, failed) = invoke(
        &daemon.daemon.archive,
        Tool::Read,
        json!({"arguments":{"selection":"annotations where kind:note"}}),
    );
    assert!(!failed, "{notes:?}");
    assert_eq!(notes.value().unwrap()["reply"]["count"], 1);
}

#[test]
fn read_tool_refuses_write_step() {
    let directory = tempfile::tempdir().unwrap();
    let call = json!({"arguments":{"selection":"session:claude-code:s1","steps":[{"verb":"note","text":"A note."}]},"deadline":null});
    let (text, answer, failed) = invoke(&Archive::at(directory.path()), Tool::Read, call.clone());
    assert!(failed);
    assert_eq!(
        answer.failure().unwrap().message,
        "rows with these arguments needs polylogue_write"
    );
    assert!(text.contains("polylogue_write"), "{text}");
    let next = &answer.envelope.next[0];
    assert_eq!(next.request.arguments, call["arguments"]);
    assert_eq!(next.request.deadline, command_surface::Deadline::None);
    // The label is the corrected call, whole: the tool name, then the call.
    let corrected: Value =
        serde_json::from_str(next.label.strip_prefix("polylogue_write ").unwrap()).unwrap();
    assert_eq!(corrected["arguments"], call["arguments"], "{}", next.label);
}

#[test]
fn write_tool_runs_read_operations() {
    let daemon = test_daemon::start("a prompt for status");
    let (_, answer, failed) = invoke(
        &daemon.daemon.archive,
        Tool::Write,
        json!({"operation":"status","arguments":{}}),
    );
    assert!(!failed, "{answer:?}");
    assert_eq!(answer.value().unwrap()["store"]["messages"], 1);
}

#[test]
fn needs_all_next_call_adds_all() {
    let daemon = test_daemon::start_with_sources_and_workers(
        "first prompt",
        &|home| {
            std::fs::write(
                home.join(".claude/projects/-p/s2.jsonl"),
                polylogue_testkit::claude_user("u2", "s2", "second prompt"),
            )
            .unwrap();
        },
        1,
        2,
    );
    let call = json!({"arguments":{"selection":"sessions","steps":[{"verb":"tag","name":"later"}]},"deadline":null,"request_id":"all-required"});
    let (text, answer, failed) = invoke(&daemon.daemon.archive, Tool::Write, call.clone());
    assert!(failed, "{text}");
    assert_eq!(answer.failure().unwrap().code.as_deref(), Some("needs_all"));
    let next = &answer.envelope.next[0].request;
    let mut expected = call["arguments"].clone();
    expected["all"] = json!(true);
    assert_eq!(next.arguments, expected);
    assert_eq!(next.request_id, "all-required");
    assert_eq!(next.deadline, command_surface::Deadline::None);
    let (text, answer, failed) = invoke(&daemon.daemon.archive, Tool::Write, arguments_for(next));
    assert!(!failed, "{text}");
    assert_eq!(
        answer.value().unwrap()["reply"]["steps"][0]["outcome"]["reply"]["reply"]["targets"],
        2
    );
}

#[test]
fn mcp_step_failure_keeps_undo_and_skipped_steps() {
    let daemon = test_daemon::start("a prompt for step reports");
    let (text, answer, failed) = invoke(
        &daemon.daemon.archive,
        Tool::Write,
        json!({"arguments":{
        "selection":"session:claude-code:s1","steps":[
            {"verb":"note","text":"A note."},
            {"verb":"edit","text":"The session is not a note."},
            {"verb":"read"}
        ]},"request_id":"step-failure"}),
    );
    assert!(failed, "{text}");
    assert_eq!(
        answer.value().unwrap()["reply"]["steps"][2]["outcome"]["status"],
        "not_run"
    );
    let undo = r"printf '%s\n' 'annotation:step-failure-1@step-failure-1' | polylogue --request-id step-failure-1-undo - withdraw";
    let reports = &answer.value().unwrap()["reply"]["steps"];
    assert_eq!(reports[0]["outcome"]["undo"], json!([undo]));
    assert_eq!(reports[1]["outcome"]["error"]["kind"], "bad_request");
    assert_eq!(reports[1]["request_id"], "step-failure-2");
    assert!(text.contains(&format!("undo: {undo}")), "{text}");
    assert!(text.contains("error (bad_request)"), "{text}");
    assert!(text.contains("read not run"), "{text}");
}

#[test]
fn mcp_read_step_renders_archive_text() {
    let daemon = test_daemon::start("a prompt in the shared read renderer");
    let (text, answer, failed) = invoke(
        &daemon.daemon.archive,
        Tool::Read,
        json!({"arguments":{"selection":"session:claude-code:s1","steps":[{"verb":"read"}]}}),
    );
    assert!(!failed, "{answer:?}");
    assert!(text.starts_with(polylogue_render::ARCHIVE_NOTICE), "{text}");
    assert!(
        text.contains("a prompt in the shared read renderer"),
        "{text}"
    );
}

#[test]
fn mcp_follow_returns_paged_call() {
    let daemon = test_daemon::start("a prompt in the finite read after follow");
    let (_, answer, failed) = invoke(
        &daemon.daemon.archive,
        Tool::Read,
        json!({"arguments":{"selection":"session:claude-code:s1","steps":[{"verb":"read","follow":true,"view":"dialogue","budget":128}]}}),
    );
    assert!(failed);
    assert_eq!(
        answer.failure().unwrap().message,
        "a model cannot hold a stream; read again to continue"
    );
    // The same line with each read paged: the typed line, whose defaults
    // (no follow, a page) are left out.
    let next = &answer.envelope.next[0].request;
    assert_eq!(
        next.arguments,
        json!({
            "selection":"session:claude-code:s1",
            "steps":[{"verb":"read","view":"dialogue","budget":128}]
        })
    );
    let (text, read, failed) = invoke(&daemon.daemon.archive, Tool::Read, arguments_for(next));
    assert!(!failed, "{read:?}");
    assert_eq!(
        read.value().unwrap()["reply"]["steps"][0]["outcome"]["reply"]["kind"],
        "read"
    );
    assert!(
        text.contains("a prompt in the finite read after follow"),
        "{text}"
    );
}

#[test]
fn set_listing_is_read_access() {
    let daemon = test_daemon::start("a prompt for setting access");
    let (_, catalog, failed) = invoke(
        &daemon.daemon.archive,
        Tool::Read,
        json!({"operation":"set","arguments":{}}),
    );
    assert!(!failed, "{catalog:?}");
    let catalog = catalog.value().unwrap().as_array().unwrap();
    let listed = catalog
        .iter()
        .find(|setting| setting["key"] == "browser.actions")
        .unwrap();
    assert_eq!(listed["value"], json!({"type":"boolean","value":false}));

    let show = json!({"operation":"set","arguments":{"key":"browser.actions"}});
    let (_, current, failed) = invoke(&daemon.daemon.archive, Tool::Read, show.clone());
    assert!(!failed, "{current:?}");
    let current = current.value().unwrap().clone();
    assert_eq!(current["key"], "browser.actions");
    assert_eq!(current["value"], json!({"type":"boolean","value":false}));

    let call = json!({"operation":"set","arguments":{"key":"browser.actions","value":"on"}});
    let (text, refused, failed) = invoke(&daemon.daemon.archive, Tool::Read, call.clone());
    assert!(failed, "{text}");
    assert_eq!(
        refused.failure().unwrap().message,
        "set with these arguments needs polylogue_write"
    );
    let next = &refused.envelope.next[0];
    assert_eq!(next.request.operation, "set");
    assert_eq!(next.request.arguments, call["arguments"]);
    let corrected: Value =
        serde_json::from_str(next.label.strip_prefix("polylogue_write ").unwrap()).unwrap();
    assert_eq!(corrected["operation"], "set", "{}", next.label);
    assert_eq!(corrected["arguments"], call["arguments"], "{}", next.label);
    let (_, unchanged, failed) = invoke(&daemon.daemon.archive, Tool::Read, show);
    assert!(!failed, "{unchanged:?}");
    assert_eq!(unchanged.value().unwrap(), &current);
}

#[test]
fn import_provider_is_act_access() {
    let directory = tempfile::tempdir().unwrap();
    let archive = Archive::at(directory.path());
    for provider in ["chatgpt", "claude-ai", "grok"] {
        let call = json!({"operation":"import","arguments":{"paths":[provider]}});
        let (text, refused, failed) = invoke(&archive, Tool::Write, call.clone());
        assert!(failed, "{provider}: {text}");
        assert_eq!(
            refused.failure().unwrap().message,
            "import with these arguments needs polylogue_act",
            "{provider}"
        );
        let next = &refused.envelope.next[0];
        assert_eq!(next.request.operation, "import");
        assert_eq!(next.request.arguments, call["arguments"]);
        let corrected: Value =
            serde_json::from_str(next.label.strip_prefix("polylogue_act ").unwrap()).unwrap();
        assert_eq!(corrected["operation"], "import");
        assert_eq!(corrected["arguments"], call["arguments"]);
        let (_, accepted, failed) = invoke(&archive, Tool::Act, corrected);
        assert!(failed);
        assert_eq!(
            accepted.failure().unwrap().code.as_deref(),
            Some("not_running")
        );

        let (text, path, failed) = invoke(
            &archive,
            Tool::Write,
            json!({"operation":"import","arguments":{"paths":[format!("./{provider}")]}}),
        );
        assert!(failed, "{provider}: {text}");
        assert_eq!(path.failure().unwrap().code.as_deref(), Some("not_running"));
    }
}

#[test]
fn mcp_and_cli_print_the_same_step_text() {
    use crate::protocol::line::{StepAnswer, StepOutcome, StepReport, StepsAnswer};
    use polylogue_model::user::{AssertQueryOutcome, KeptPart, SaveReport, SavedPart};
    let at = polylogue_model::Timestamp::from_unix_ms(1_767_225_600_000).unwrap();
    let done = |verb: &str, reply: StepAnswer| StepReport {
        verb: verb.into(),
        request_id: None,
        outcome: StepOutcome::Done {
            reply,
            undo: vec![],
        },
    };
    let plan = crate::resume::Plan {
        session: polylogue_model::SessionId::parse("codex:step-text").unwrap(),
        harness_command: Some(crate::resume::Command {
            program: "codex".into(),
            args: vec!["resume".into(), "step-text".into()],
            cwd: Some("/home/op/project".into()),
        }),
        context_assisted: crate::resume::Command {
            program: "polylogue".into(),
            args: vec!["session:codex:step-text".into(), "brief".into()],
            cwd: None,
        },
    };
    let line = LineAnswer::Steps(StepsAnswer {
        resolved: None,
        steps: vec![
            done(
                "resume",
                StepAnswer::Resume(Box::new(crate::protocol::resume::Answer::Plan(plan))),
            ),
            done(
                "save",
                StepAnswer::Saved(SaveReport {
                    request: "save-request".into(),
                    saved: Some(SavedPart {
                        id: "annotation:open-work@1".into(),
                        name: "open-work".into(),
                        revision: 1,
                        new_revision: true,
                    }),
                    kept: KeptPart {
                        result: "result:kept-1".into(),
                        rows: 3,
                        at,
                    },
                    watch: None,
                    repeat: false,
                    pending: false,
                }),
            ),
            done(
                "tag",
                StepAnswer::WroteEach(AssertQueryOutcome {
                    id: Some("tag-request".into()),
                    repeat: false,
                    pending: false,
                    targets: 2,
                    written: 2,
                    already: vec![],
                }),
            ),
        ],
    });
    let expected = format!(
        "harness command:    cd /home/op/project && codex resume step-text\n\
         with context:       polylogue session:codex:step-text brief\n\
         \nsaved   query:open-work (revision 1): runs again each time you use it\n\
         kept    result:kept-1 (3 rows, {at}), now result:open-work\n\
         \ntagged 2 rows\n"
    );
    assert_eq!(line_text(&line).unwrap(), expected);
    assert_eq!(
        crate::cli::print::line(&line, command_surface::TextStyle::Plain),
        expected.replace("\n\n", "\n")
    );
}
