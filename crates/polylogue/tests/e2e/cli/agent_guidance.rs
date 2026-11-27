use super::*;
use serde_json::{Value, json};
use std::io::Write;
use std::os::unix::net::UnixListener;

#[test]
fn gemini_hooks_install_and_uninstall_through_the_public_command() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("settings.json");
    let original = json!({"model":"synthetic", "hooks":{"BeforeAgent":[{"hooks":[{"type":"command","command":"other"}]}]}});
    std::fs::write(&path, serde_json::to_vec(&original).unwrap()).unwrap();
    let mut installed = None;
    for remove in [false, false, true] {
        let mut command = crate::command(dir.path());
        command.args([
            "install",
            "gemini-cli",
            "--settings",
            path.to_str().unwrap(),
        ]);
        if remove {
            command.arg("--remove");
        }
        let out = command.output().unwrap();
        assert!(out.status.success(), "{out:?}");
        let value: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        if !remove {
            assert_eq!(value["model"], "synthetic");
            assert_eq!(
                value["hooks"]["SessionStart"][0]["hooks"][0]["command"],
                "polylogue internal hook SessionStart --harness gemini-cli"
            );
            assert_eq!(
                value["hooks"]["SessionStart"][0]["hooks"][0]["timeout"],
                10000
            );
            if let Some(previous) = installed.replace(value.clone()) {
                assert_eq!(value, previous);
            }
        } else {
            assert_eq!(value, original);
        }
    }
}

#[test]
fn help_runs_offline_and_the_daemon_serves_the_full_manual() {
    let dir = tempfile::tempdir().unwrap();
    let archive = dir.path().join("archive");
    let fixture = crate::Fixture::new(
        &archive,
        &dir.path().join("home"),
        &dir.path().join("config"),
    );
    let plain = cli(&fixture, &["help"]);
    assert!(plain.status.success(), "{plain:?}");
    let text = String::from_utf8(plain.stdout).unwrap();
    assert!(text.contains("polylogue"));
    assert!(text.contains("read") && text.contains("save"));
    assert!(!archive.exists());
    let _daemon = start_daemon(&fixture);
    let full = cli(&fixture, &["help", "--all"]);
    assert!(full.status.success(), "{full:?}");
    let text = String::from_utf8(full.stdout).unwrap();
    assert!(text.contains("sessions where"));
    // The operation reference follows the guide (docs/agents.md, The skill
    // file): each tool, its operations with their CLI spelling, its steps.
    for heading in [
        "\n## `polylogue_read`\n",
        "\n## `polylogue_write`\n",
        "\n## `polylogue_act`\n",
    ] {
        assert!(text.contains(heading), "{heading}");
    }
    assert!(text.contains("- `status` (CLI `status`): "), "{text}");
    assert!(text.contains("- `dev sql` (CLI `dev sql`): "), "{text}");
    assert!(
        text.contains("- step `tag(name)`: label each row."),
        "{text}"
    );
    let structured = cli(&fixture, &["--json", "help", "--all"]);
    assert!(structured.status.success(), "{structured:?}");
    assert_eq!(
        json(&structured)["manual"].as_str().unwrap().trim_end(),
        text.trim_end()
    );
}

#[test]
fn gemini_start_and_prompt_events_request_context_through_respond() {
    for (event, payload, delivery_event, expected) in [
        (
            "SessionStart",
            json!({"session_id":"current","cwd":"/synthetic/project","source":"resume"}),
            "resume",
            json!({"steps":[{"verb":"brief", "cwd":"/synthetic/project","for":"gemini-cli:current","source":"resume"}]}),
        ),
        (
            "BeforeAgent",
            json!({"session_id":"current","prompt":"Continue https://chatgpt.com/c/previous and inspect src/parser.rs"}),
            "session_brief",
            json!({"selection":"session:chatgpt:previous", "steps":[{"verb":"brief", "for":"gemini-cli:current"}]}),
        ),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let archive = dir.path().join("archive");
        std::fs::create_dir(&archive).unwrap();
        let listener = UnixListener::bind(polylogue_store::Archive::at(&archive).socket()).unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut line = String::new();
            BufReader::new(&stream).read_line(&mut line).unwrap();
            let request: Value = serde_json::from_str(&line).unwrap();
            assert_eq!(request["operation"], "rows");
            assert_eq!(request["arguments"], expected);
            assert!(
                request["request_id"]
                    .as_str()
                    .is_some_and(|id| !id.is_empty())
            );
            // Capture and preparation spend the same entry deadline as this request.
            assert!(
                request["deadline"]
                    .as_u64()
                    .is_some_and(|milliseconds| (1..1_500).contains(&milliseconds)),
                "{request}"
            );
            assert!(request.get("op").is_none());
            let delivery = format!("{}-1", request["request_id"].as_str().unwrap());
            let prepared = json!({
                "event": delivery_event,
                "empty": false,
                "text": "Synthetic context",
                "refs": [],
                "left_out": [],
            });
            let answer = command_surface::Answer::ok(json!({
                "kind":"steps", "reply":{"resolved":null, "steps":[{
                    "verb":"brief", "request_id":delivery,
                    "outcome":{"status":"done", "reply":{"kind":"brief", "reply":{
                        "delivery":delivery, "prepared":prepared
                    }}}
                }]}
            }));
            write!(
                stream,
                "event: answer\ndata: {}\n\n",
                serde_json::to_string(&answer).unwrap()
            )
            .unwrap();
            let (mut stream, _) = listener.accept().unwrap();
            line.clear();
            BufReader::new(&stream).read_line(&mut line).unwrap();
            let request: Value = serde_json::from_str(&line).unwrap();
            assert_eq!(request["operation"], "internal delivery-event");
            assert_eq!(request["arguments"]["delivery"], delivery);
            assert_eq!(request["arguments"]["what"], "delivered");
            write!(
                stream,
                "event: answer\ndata: {}\n\n",
                serde_json::to_string(&command_surface::Answer::ok(json!({}))).unwrap()
            )
            .unwrap();
        });
        let mut child = crate::command(dir.path())
            .args(["internal", "hook", event, "--harness", "gemini-cli"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(&serde_json::to_vec(&payload).unwrap())
            .unwrap();
        let out = child.wait_with_output().unwrap();
        assert!(out.status.success(), "{out:?}");
        assert_eq!(
            gemini_hook_output(&out),
            json!({"hookSpecificOutput":{"hookEventName":event,"additionalContext":"Synthetic context"}})
        );
        server.join().unwrap();
    }
}

/// Hook output follows the harness's own protocol, not a command Answer.
fn gemini_hook_output(output: &Output) -> Value {
    assert!(output.status.success(), "{output:?}");
    let value: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert!(value.get("hookSpecificOutput").is_some(), "{value}");
    value
}
