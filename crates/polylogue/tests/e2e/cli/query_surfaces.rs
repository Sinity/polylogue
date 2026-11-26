//! Query refusals through the real CLI, socket and MCP commands.

use super::*;
use serde_json::{Value, json};
use std::io::Write;
use std::os::unix::net::UnixStream;
use std::process::{ChildStdin, ChildStdout};

pub(super) struct Mcp {
    _process: Daemon,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
    next: u64,
}

impl Mcp {
    pub(super) fn start(env: &crate::Fixture) -> Self {
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

    pub(super) fn request(&mut self, method: &str, params: Value) -> Value {
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
        command_surface::Answer {
            outcome: command_surface::Outcome::Error(error),
            ..
        } => Err(error),
        other => Ok(other
            .value()
            .expect("an answered query carries a value")
            .clone()),
    }
}

/// The whole typed answer one surface gave, with the CLI's exit status.
fn surface_answer(
    env: &crate::Fixture,
    archive: &Path,
    mcp: &mut Mcp,
    surface: &str,
    text: &str,
) -> command_surface::Answer<Value> {
    match surface {
        "cli" => {
            let out = cli(env, &["--json", text]);
            let answer: command_surface::Answer<Value> =
                serde_json::from_slice(&out.stdout).unwrap();
            assert_eq!(
                out.status.code(),
                Some(i32::from(answer.exit_code())),
                "{out:?}"
            );
            answer
        }
        "socket" => {
            let mut stream = UnixStream::connect(archive.join("daemon.sock")).unwrap();
            let request = command_surface::Request {
                operation: "rows".into(),
                arguments: json!({"selection":text}),
                request_id: uuid::Uuid::now_v7().to_string(),
                deadline: command_surface::Deadline::Default,
                budget_tokens: Default::default(),
                cursor: None,
                fresh: false,
                format: Default::default(),
            };
            writeln!(stream, "{}", serde_json::to_string(&request).unwrap()).unwrap();
            super::socket_answer(stream)
        }
        "mcp" => {
            let reply = mcp.request(
                "tools/call",
                json!({"name":"polylogue_read", "arguments":{
                    "operation":"rows", "arguments":{"selection":text}, "request_id":uuid::Uuid::now_v7().to_string()
                }}),
            );
            assert!(reply["error"].is_null(), "{reply}");
            let result = &reply["result"];
            let answered: command_surface::Answer<Value> =
                serde_json::from_value(result["structuredContent"].clone()).unwrap();
            assert_eq!(
                result["isError"],
                matches!(
                    answered,
                    command_surface::Answer {
                        outcome: command_surface::Outcome::Error(_),
                        ..
                    }
                )
            );
            answered
        }
        _ => unreachable!(),
    }
}

fn answer(
    env: &crate::Fixture,
    archive: &Path,
    mcp: &mut Mcp,
    surface: &str,
    text: &str,
) -> Result<Value, command_surface::Failure> {
    value(surface_answer(env, archive, mcp, surface, text))
}

#[test]
fn mcp_brief_names_a_delivery_only_when_the_brief_is_not_empty() {
    let dir = tempfile::tempdir().unwrap();
    let env = crate::Fixture::new(
        &dir.path().join("archive"),
        &dir.path().join("home"),
        &dir.path().join("config"),
    );
    let _daemon = start_daemon(&env);
    wait_status(&env, |status| status["store"]["messages"] == 0);
    let mut mcp = Mcp::start(&env);
    for (source, request_id, empty) in [
        ("startup", "native-context-startup", false),
        ("resume", "native-context-resume", true),
    ] {
        let direct: command_surface::Answer<Value> = if source == "startup" {
            let output = cli(
                &env,
                &[
                    "--json",
                    "--request-id",
                    request_id,
                    "brief",
                    "--cwd",
                    "/synthetic",
                ],
            );
            assert!(output.status.success(), "{output:?}");
            serde_json::from_slice(&output.stdout).unwrap()
        } else {
            let mut socket = UnixStream::connect(env.archive().join("daemon.sock")).unwrap();
            let request = command_surface::RunnableRequest::new(
                "rows",
                json!({"steps":[{"verb":"brief","cwd":"/synthetic","source":source}]}),
                request_id,
            );
            writeln!(socket, "{}", serde_json::to_string(&request).unwrap()).unwrap();
            super::socket_answer(socket)
        };
        let brief = |answer| {
            let value = value(answer).unwrap();
            let reports = value["reply"]["steps"].as_array().unwrap();
            assert_eq!(reports.len(), 1, "{value}");
            assert_eq!(reports[0]["outcome"]["status"], "done", "{value}");
            assert_eq!(reports[0]["outcome"]["reply"]["kind"], "brief", "{value}");
            reports[0]["outcome"]["reply"]["reply"].clone()
        };
        let mut expected = brief(direct);
        assert_eq!(expected["delivery"], Value::Null);
        assert_eq!(expected["prepared"]["empty"], empty);
        assert!(
            expected["prepared"]["text"]
                .as_str()
                .unwrap()
                .contains(request_id),
            "{expected}",
        );
        let reply = mcp.request(
            "tools/call",
            json!({"name":"polylogue_read", "arguments":{
                "operation":"rows",
                "arguments":{"steps":[{"verb":"brief", "cwd":"/synthetic", "source":source}]},
                "request_id":request_id,
            }}),
        );
        assert!(reply["error"].is_null(), "{reply}");
        assert_eq!(reply["result"]["isError"], false, "{reply}");
        let mcp: command_surface::Answer<Value> =
            serde_json::from_value(reply["result"]["structuredContent"].clone()).unwrap();
        if !empty {
            expected["delivery"] = Value::String(format!("{request_id}-1"));
        }
        assert_eq!(brief(mcp), expected, "native {source} context");
    }
}

fn same_question(actual: &str, expected: &str) {
    let catalog = polylogue_query::catalog::catalog(&[]);
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
fn every_surface_preserves_query_words_unit_corrections_and_closed_sets() {
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
        for text in ["brief yesterday", "status", "statsu"] {
            let value = answer(&env, &archive, &mut mcp, surface, text).unwrap();
            if surface == "cli" && text == "status" {
                assert!(value["store"].is_object());
            }
        }
        for (text, expected) in [
            ("messges cargo", "messages where cargo"),
            ("sesions where repo:morrow", "sessions where repo:morrow"),
        ] {
            let failure = answer(&env, &archive, &mut mcp, surface, text).unwrap_err();
            assert_eq!(failure.kind, command_surface::FailureKind::NotAQuery);
            assert_eq!(failure.position.as_ref().unwrap().offset, 0);
            let Some(command_surface::Suggestion::Query(corrected)) = failure.suggestion() else {
                panic!("missing typed correction: {failure:?}");
            };
            same_question(corrected, expected);
            assert!(answer(&env, &archive, &mut mcp, surface, corrected).is_ok());
        }
        let catalog = polylogue_query::catalog::catalog(&[]);
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
            "test \"output\"",
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
        // A field name is never a search word; quoted, it is one.
        let failure = answer(&env, &archive, &mut mcp, surface, "test output").unwrap_err();
        assert_eq!(failure.kind, command_surface::FailureKind::InvalidQuery);
        assert_eq!(failure.position.as_ref().unwrap().offset, 5);
    }
}

/// An answer assembled with a missing part is degraded on every surface,
/// keeps its rows, and names the missing part in its notes: the daemon's
/// typed outcome reaches CLI JSON, the socket and MCP structured output
/// unchanged.
#[test]
fn a_degraded_answer_keeps_its_outcome_and_notes_on_every_surface() {
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
    let text = "walrus similar:\"walrus migration\"";
    let mut answers = Vec::new();
    for surface in ["cli", "socket", "mcp"] {
        let answer = surface_answer(&env, &archive, &mut mcp, surface, text);
        let command_surface::Answer {
            outcome:
                command_surface::Outcome::Degraded {
                    value,
                    missing: notes,
                    ..
                },
            ..
        } = &answer
        else {
            panic!("{surface}: {answer:?}");
        };
        assert_eq!(answer.exit_code(), 5, "{surface}");
        assert_eq!(listing(value.clone())["count"], 1, "{surface}: {value}");
        assert!(
            notes.iter().any(|note| note
                == "incomplete: semantic search is off; enable it with `polylogue set embed.provider local`; rows are ranked by text relevance alone"),
            "{surface}: {notes:?}"
        );
        answers.push(notes.clone());
    }
    assert!(
        answers.windows(2).all(|pair| pair[0] == pair[1]),
        "{answers:?}"
    );
}

#[test]
fn unavailable_rankings_and_empty_reads_explain_themselves_on_morrow() {
    let morrow = polylogue_testkit::Morrow::build(polylogue_testkit::Set::Main);
    let env = crate::Fixture::new(
        &morrow.root,
        &morrow.home,
        &morrow.dir.path().join("config"),
    );
    let _daemon = start_daemon(&env);
    wait_status(&env, |status| status["store"]["sessions"] == 8);
    let answer = |selection: &str, read: bool| {
        let mut arguments = vec!["--json", selection];
        if read {
            arguments.push("read");
        }
        let output = cli(&env, &arguments);
        let answer: command_surface::Answer<Value> =
            serde_json::from_slice(&output.stdout).unwrap();
        let failed_step = answer
            .value()
            .and_then(|value| value["reply"]["steps"].as_array())
            .and_then(|steps| steps.first())
            .filter(|step| step["outcome"]["status"] == "failed")
            .map(|step| {
                serde_json::from_value::<command_surface::Failure>(step["outcome"]["error"].clone())
                    .unwrap()
            });
        let expected_exit = failed_step
            .as_ref()
            .map_or(answer.exit_code(), |error| error.kind.exit_code());
        assert_eq!(
            output.status.code(),
            Some(i32::from(expected_exit)),
            "{output:?}"
        );
        answer
    };
    for read in [false, true] {
        let response = answer("similar:\"ambiguous dates\"", read);
        if !read {
            assert_eq!(response.exit_code(), 3, "{response:?}");
        }
        let error = if read {
            read_failure(response)
        } else {
            let command_surface::Outcome::Error(error) = response.outcome else {
                panic!("{response:?}")
            };
            error
        };
        assert_eq!(error.kind.name(), "refused");
        assert_eq!(error.kind.exit_code(), 3);
        assert_eq!(error.code.as_deref(), Some("embeddings_off"));
        assert_eq!(
            error.message,
            "semantic search is off; enable it with `polylogue set embed.provider local`"
        );
        assert_eq!(error.reason().unwrap().name, "embeddings_off");
    }
    let plain = answer("sessions where chrono", false);
    assert_eq!(listing(plain.value().unwrap().clone())["count"], 4);
    let hybrid = answer("sessions where chrono similar:\"ambiguous dates\"", false);
    assert_eq!(hybrid.exit_code(), 5, "{hybrid:?}");
    assert_eq!(
        listing(hybrid.value().unwrap().clone())["rows"],
        listing(plain.value().unwrap().clone())["rows"]
    );
    let expected = "semantic search is off; enable it with `polylogue set embed.provider local`; rows are ranked by text relevance alone";
    assert!(
        matches!(&hybrid.outcome, command_surface::Outcome::Degraded { missing, .. } if missing.iter().any(|note| note == &format!("incomplete: {expected}"))),
        "{hybrid:?}"
    );
    let hybrid = answer("sessions where chrono similar:\"ambiguous dates\"", true);
    assert_eq!(hybrid.exit_code(), 5, "{hybrid:?}");
    assert!(
        matches!(&hybrid.outcome, command_surface::Outcome::Degraded { missing, .. } if missing.iter().any(|note| note == expected)),
        "{hybrid:?}"
    );
    let empty = answer("zzzznothing", true);
    assert!(
        matches!(empty.outcome, command_surface::Outcome::Empty(_)),
        "{empty:?}"
    );
    assert_eq!(empty.exit_code(), 1);
    assert_eq!(empty.envelope.notes, ["no rows matched"]);
    let text = cli(&env, &["zzzznothing", "read"]);
    assert_eq!(text.status.code(), Some(1), "{text:?}");
    assert_eq!(
        String::from_utf8(text.stderr).unwrap().trim(),
        "note: no rows matched"
    );
    let near = answer("near:\"x\"", false);
    assert_eq!(near.exit_code(), 2, "{near:?}");
    let command_surface::Outcome::Error(error) = near.outcome else {
        panic!("{near:?}")
    };
    assert_eq!(error.kind.name(), "unknown_field");
    assert_eq!(error.code.as_deref(), Some("unknown_field"));
    assert_eq!(
        error.hint.as_deref(),
        Some("use similar: for semantic ranking")
    );
    assert_eq!(
        serde_json::to_value(error.suggestion()).unwrap()["query"],
        "similar:\"x\""
    );
    let text_field = answer("messages where text:chrono", false);
    assert_eq!(text_field.exit_code(), 2, "{text_field:?}");
    let command_surface::Outcome::Error(error) = text_field.outcome else {
        panic!("{text_field:?}")
    };
    assert_eq!(error.kind.name(), "unknown_field");
    assert_eq!(error.code.as_deref(), Some("unknown_field"));
    assert_eq!(
        error.hint.as_deref(),
        Some("use quoted words for full-text search, or text ~ \"fragment\" for a substring")
    );
    let measure = answer("sessions | measure nosuch", false);
    assert_eq!(measure.exit_code(), 2, "{measure:?}");
    let command_surface::Outcome::Error(error) = measure.outcome else {
        panic!("{measure:?}")
    };
    assert_eq!(error.kind.name(), "bad_value");
    assert_eq!(error.code.as_deref(), Some("bad_value"));
    assert_eq!(
        error.message,
        "unknown measurement `nosuch`; choose one of compare, daily, drift, facets, failing, outcomes, outlook, process, quota, responses, rhythm, sessions, shell-stages, survival, survival-turns, time, unanswered, usage"
    );
    let origin = answer("origin:nosuch", false);
    assert_eq!(origin.exit_code(), 2, "{origin:?}");
    let command_surface::Outcome::Error(error) = origin.outcome else {
        panic!("{origin:?}")
    };
    assert_eq!(error.kind.name(), "bad_value");
    assert_eq!(error.code.as_deref(), Some("bad_value"));
    assert_eq!(
        error.candidates().first().map(String::as_str),
        Some("codex")
    );
    assert_eq!(
        error.message.lines().next().unwrap(),
        format!(
            "`origin` on `sessions` is one of: {}",
            error.candidates().join(", ")
        )
    );
    let set = |key: &str, value: &str| {
        let output = cli(&env, &["set", key, value]);
        assert!(output.status.success(), "{output:?}");
    };
    set("embed.provider", "local");
    let check_missing = |sentence: &str| {
        for read in [false, true] {
            let response = answer("similar:\"ambiguous dates\"", read);
            assert_eq!(response.exit_code(), 5, "{response:?}");
            let command_surface::Outcome::Degraded { missing, .. } = response.outcome else {
                panic!("{response:?}")
            };
            assert_eq!(
                missing,
                [if read {
                    sentence.to_owned()
                } else {
                    format!("incomplete: {sentence}")
                }]
            );
        }
    };
    check_missing(
        "no vectors are stored for model `qwen3-embedding:0.6b`; build them with `polylogue embed`; rows are ordered by time and unranked",
    );
    set("embed.endpoint", "https://example.invalid/v1");
    check_missing(
        "semantic search configuration is invalid: embed.endpoint https://example.invalid/v1 is not on this machine; the local provider only sends text to a loopback address; rows are ordered by time and unranked",
    );
    set("embed.endpoint", "http://127.0.0.1:11434/v1");
    let path = morrow.root.join("embeddings.db");
    std::fs::rename(&path, morrow.root.join("unreadable-embeddings.db")).unwrap();
    std::fs::create_dir(&path).unwrap();
    check_missing(&format!(
        "the embeddings store cannot be read: unable to open database file: {}; rows are ordered by time and unranked",
        path.display()
    ));
}
