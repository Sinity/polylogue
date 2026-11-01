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
    fn start(env: &crate::Fixture) -> Self {
        let mut command = env.command();
        command
            .arg("mcp")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
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

fn value(answer: command_surface::Answer<Value>) -> Result<Value, command_surface::Failure> {
    match answer {
        command_surface::Answer::Error { error, .. } => Err(error),
        other => Ok(other
            .value()
            .expect("an answered query carries a value")
            .clone()),
    }
}

fn answer(
    env: &crate::Fixture,
    archive: &Path,
    mcp: &mut Mcp,
    surface: &str,
    text: &str,
) -> Result<Value, command_surface::Failure> {
    match surface {
        "cli" => {
            let out = cli(env, &["--json", text]);
            value(serde_json::from_slice(&out.stdout).unwrap())
        }
        "socket" => {
            let mut stream = UnixStream::connect(archive.join("daemon.sock")).unwrap();
            let request = command_surface::Request {
                operation: "query".into(),
                arguments: json!({"query":text}),
                request_id: uuid::Uuid::now_v7().to_string(),
                deadline: command_surface::Deadline::Default,
            };
            writeln!(stream, "{}", serde_json::to_string(&request).unwrap()).unwrap();
            let mut line = String::new();
            BufReader::new(stream).read_line(&mut line).unwrap();
            value(serde_json::from_str(&line).unwrap())
        }
        "mcp" => {
            let reply = mcp.request(
                "tools/call",
                json!({"name":"polylogue_find", "arguments":{
                    "arguments":{"query":text}, "request_id":uuid::Uuid::now_v7().to_string()
                }}),
            );
            assert!(reply["error"].is_null(), "{reply}");
            let result = &reply["result"];
            let answered =
                value(serde_json::from_value(result["structuredContent"].clone()).unwrap());
            assert_eq!(result["isError"], answered.is_err());
            answered
        }
        _ => unreachable!(),
    }
}
fn same_question(actual: &str, expected: &str) {
    let catalog = polylogue_query::catalog::catalog(&[], &[]);
    let identity = |text: &str| {
        let query = query_language::parse::parse(text, &catalog).unwrap();
        query_language::identity::question_id(&query, &catalog, &Default::default()).unwrap()
    };
    assert_eq!(identity(actual), identity(expected));
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
    let env = crate::Fixture::new(archive.as_path(), home.as_path(), config.as_path());
    let _daemon = start_daemon(&env);
    wait_status(&env, |s| s["store"]["messages"].as_u64() == Some(1));
    let mut mcp = Mcp::start(&env);
    for surface in ["cli", "socket", "mcp"] {
        for (text, kind) in [
            ("brief yesterday", command_surface::FailureKind::NotAQuery),
            ("status", command_surface::FailureKind::NotAQuery),
            ("messges cargo", command_surface::FailureKind::NotAQuery),
            (
                "sessin where repo:morrow",
                command_surface::FailureKind::NotAQuery,
            ),
            ("statsu", command_surface::FailureKind::NotAQuery),
        ] {
            if surface == "cli" && text == "status" {
                let status = answer(&env, &archive, &mut mcp, surface, text).unwrap();
                assert!(status["store"].is_object(), "{status}");
                continue;
            }
            let failure = answer(&env, &archive, &mut mcp, surface, text).unwrap_err();
            assert_eq!(failure.kind, kind, "{surface}: {text}: {failure:?}");
            assert_eq!(failure.position.as_ref().unwrap().offset, 0);
            match text {
                "brief yesterday" | "status" => {
                    let word = text.split_whitespace().next().unwrap();
                    assert_eq!(
                        failure.hint.as_deref(),
                        Some(format!("run the `{word}` command").as_str())
                    );
                }
                "statsu" => assert_eq!(failure.suggestion.as_deref(), Some(&json!("status"))),
                _ => {
                    let corrected = failure.suggestion.as_deref().unwrap().as_str().unwrap();
                    let expected = if text == "messges cargo" {
                        "messages where cargo"
                    } else {
                        "sessions where repo:morrow"
                    };
                    same_question(corrected, expected);
                    assert!(answer(&env, &archive, &mut mcp, surface, corrected).is_ok());
                }
            }
        }
        let catalog = polylogue_query::catalog::catalog(&[], &[]);
        for (text, unit, field, rejected) in [
            (
                "sessions where outcome:error",
                "sessions",
                "outcome",
                "error",
            ),
            (
                "tool_calls where outcome:solved",
                "tool_calls",
                "outcome",
                "solved",
            ),
            (
                "messages where role:asistant",
                "messages",
                "role",
                "asistant",
            ),
            ("kind:continuation", "sessions", "kind", "continuation"),
            ("kind:mian", "sessions", "kind", "mian"),
        ] {
            let failure = answer(&env, &archive, &mut mcp, surface, text).unwrap_err();
            assert_eq!(failure.kind, command_surface::FailureKind::BadValue);
            assert_eq!(
                failure.position.as_ref().unwrap().offset,
                text.rfind(':').unwrap() + 1
            );
            let values = &catalog.field(unit, field).unwrap().values;
            let mut expected = format!("`{field}` on `{unit}` is one of: {}", values.join(", "));
            let elsewhere = catalog.units_accepting(rejected);
            if !elsewhere.is_empty() {
                expected.push_str(&format!(
                    "; `{rejected}` is a value of {}",
                    elsewhere.join(", ")
                ));
            }
            assert!(
                failure.message.starts_with(&expected),
                "{surface}: {text}: {failure:?}"
            );
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
            if let Err(failure) = answer(&env, &archive, &mut mcp, surface, text) {
                panic!("{surface}: {text} refused: {failure:?}");
            }
        }
    }
}
