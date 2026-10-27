//! Query refusals through the real CLI, socket and MCP commands.

use super::*;
use serde_json::{Value, json};
use std::io::Write;
use std::os::unix::net::UnixStream;
use std::process::{ChildStdin, ChildStdout};

struct Mcp {
    _process: Daemon,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
    next: u64,
}

impl Mcp {
    fn start(env: &[(&str, &Path)]) -> Self {
        let mut command = Command::new(crate::BIN);
        command
            .arg("mcp")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        for (key, value) in env {
            command.env(key, value);
        }
        let mut process = Daemon(polylogue_testkit::spawn_daemon(&mut command).unwrap());
        let input = process.0.stdin.take().unwrap();
        let output = BufReader::new(process.0.stdout.take().unwrap());
        let mut mcp = Self {
            _process: process,
            input,
            output,
            next: 0,
        };
        let initialized = mcp.request(
            "initialize",
            json!({
                "protocolVersion": "2025-03-26", "capabilities": {},
                "clientInfo": { "name": "query-tests", "version": "1" }
            }),
        );
        assert!(initialized["error"].is_null(), "{initialized}");
        writeln!(
            mcp.input,
            "{}",
            json!({"jsonrpc":"2.0", "method":"notifications/initialized"})
        )
        .unwrap();
        mcp
    }

    fn request(&mut self, method: &str, params: Value) -> Value {
        self.next += 1;
        writeln!(
            self.input,
            "{}",
            json!({"jsonrpc":"2.0", "id":self.next, "method":method, "params":params})
        )
        .unwrap();
        loop {
            let mut line = String::new();
            assert_ne!(
                self.output.read_line(&mut line).unwrap(),
                0,
                "MCP closed before replying to {method}"
            );
            let reply: Value = serde_json::from_str(&line).unwrap();
            if reply["id"] == self.next {
                return reply;
            }
        }
    }
}

fn failure(error: &Value) -> (String, String) {
    let message = error["message"].as_str().unwrap();
    let suggestion = error["detail"]["suggestion"]
        .as_str()
        .or_else(|| error["suggestion"].as_str())
        .unwrap_or_default();
    (
        error["kind"].as_str().unwrap().into(),
        format!("{message}\ntry: {suggestion}"),
    )
}

fn answer(
    env: &[(&str, &Path)],
    archive: &Path,
    mcp: &mut Mcp,
    surface: &str,
    text: &str,
) -> Result<Value, (String, String)> {
    match surface {
        "cli" => {
            let out = cli(env, &["--json", text]);
            let data = json(&out);
            if out.status.success() {
                Ok(data)
            } else {
                Err(failure(&data["error"]))
            }
        }
        "socket" => {
            let mut stream = UnixStream::connect(archive.join("daemon.sock")).unwrap();
            writeln!(stream, "{}", json!({"op":"query", "text":text, "cursor":null, "page_size":null, "why":false, "session":null, "fields":[]})).unwrap();
            let mut line = String::new();
            BufReader::new(stream).read_line(&mut line).unwrap();
            let response: Value = serde_json::from_str(&line).unwrap();
            if response["result"] == "ok" {
                Ok(response["data"].clone())
            } else {
                Err(failure(&response))
            }
        }
        "mcp" => {
            let reply = mcp.request(
                "tools/call",
                json!({"name":"query", "arguments":{"query":text}}),
            );
            assert!(reply["error"].is_null(), "{reply}");
            let result = &reply["result"];
            let text = result["content"][0]["text"].as_str().unwrap();
            if result["isError"] == true {
                let kind = text
                    .strip_prefix("error (")
                    .and_then(|s| s.split_once(')'))
                    .unwrap()
                    .0;
                Err((kind.into(), text.into()))
            } else {
                Ok(serde_json::from_str(text).unwrap())
            }
        }
        _ => unreachable!(),
    }
}

/// A command word or a near miss of a unit or command at the start of a
/// query, and a value outside a field's closed set, are refused on every
/// surface with what to write instead, never answered with an empty
/// search; the same words quoted, a field comparison, a unit, short words
/// and correct values run as queries.
#[test]
fn every_surface_refuses_command_words_and_values_outside_a_set() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let archive = dir.path().join("archive");
    let config = dir.path().join("config");
    let project = home.join(".claude/projects/-query");
    std::fs::create_dir_all(&project).unwrap();
    std::fs::write(
        project.join("query.jsonl"),
        line("q1", "query", "synthetic walrus prompt"),
    )
    .unwrap();
    let env = [
        ("POLYLOGUE_ARCHIVE", archive.as_path()),
        ("POLYLOGUE_HOME", home.as_path()),
        ("XDG_CONFIG_HOME", config.as_path()),
    ];
    let _daemon = start_daemon(&env);
    wait_status(&env, |s| s["store"]["messages"].as_u64() == Some(1));
    let mut mcp = Mcp::start(&env);
    for surface in ["cli", "socket", "mcp"] {
        for (text, kind, says) in [
            ("brief yesterday", "not_a_query", "\"brief\" yesterday"),
            ("status", "not_a_query", "\"status\""),
            ("messges cargo", "not_a_query", "messages cargo"),
            (
                "sessin where repo:morrow",
                "invalid_query",
                "session where repo:morrow",
            ),
            ("statsu", "not_a_query", "status"),
            ("sessions where outcome:error", "bad_value", "tool_calls"),
            ("tool_calls where outcome:solved", "bad_value", "sessions"),
            ("messages where role:asistant", "bad_value", "assistant"),
            ("kind:continuation", "bad_value", "links"),
            ("kind:mian", "bad_value", "subagent"),
        ] {
            if surface == "cli" && text == "status" {
                let status = answer(&env, &archive, &mut mcp, surface, text).unwrap();
                assert!(status["store"].is_object(), "{status}");
                continue;
            }
            match answer(&env, &archive, &mut mcp, surface, text) {
                Ok(v) => panic!("{surface}: {text} was answered: {v}"),
                Err((got, message)) => {
                    assert_eq!(got, kind, "{surface}: {text}: {message}");
                    assert!(
                        message.contains(says),
                        "{surface}: {text}: {message} does not say {says}"
                    );
                }
            }
        }
        for text in [
            "\"brief\" yesterday",
            "\"status\"",
            "sessions where messages > 0",
            "sessions | group by origin | count",
            "date parsing",
            "bug fix",
            "test output",
            "walrus",
            "tool_calls where outcome:unknown",
            "messages where role:assistant",
            "kind:subagent",
            "kind:*",
        ] {
            if let Err((kind, message)) = answer(&env, &archive, &mut mcp, surface, text) {
                panic!("{surface}: {text} refused: {kind} {message}");
            }
        }
    }
}
