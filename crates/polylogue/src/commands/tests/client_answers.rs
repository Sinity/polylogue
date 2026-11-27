//! Client responses match: the command line, MCP and the HTTP API
//! give the daemon the same request for the same arguments and hand back
//! the complete answer, with the caller details the transport observed and the read positions.

use super::*;
use crate::test_daemon;
use axum::body::{Body, to_bytes};
use polylogue_model::authorship::Authorship;
use std::sync::Arc;
use tower::ServiceExt;

/// Every read command all three surfaces offer, with the arguments it is
/// asked with here.
const ASKED: &[(&str, &[&str])] = &[
    ("rows", &["sessions"]),
    ("rows", &["messages where parser"]),
    ("rows", &["sessions where"]),
    ("help", &["units"]),
    ("dev sql", &["SELECT id FROM v1_sessions ORDER BY id"]),
    ("dev schema infer", &[]),
    ("dev schema unread", &[]),
    ("rows", &["query:running"]),
    ("rows", &["query:waiting"]),
    ("rows", &["query:notices"]),
    ("rows", &["session:claude-code:s1", "read"]),
    ("rows", &["session:claude-code:missing", "read"]),
    ("rows", &["attachment:claude-code:missing#0", "read"]),
    ("rows", &["annotation:missing", "read"]),
    ("rows", &["session:claude-code:s1", "resume"]),
    ("rows", &["brief", "--cwd", "/synthetic"]),
    ("rows", &["session:claude-code:s1", "brief"]),
    ("rows", &["session:claude-code:missing", "brief"]),
    ("rows", &["parser", "brief", "--cwd", "/synthetic"]),
    ("rows", &["chat_actions"]),
    ("rows", &["import_jobs where state:waiting"]),
    (
        "rows",
        &["sessions where since:2024-01-01 until:2024-01-01 | measure responses"],
    ),
];
fn parse_line(argv: &[&str]) -> Result<(&'static Command, command_surface::ParsedCommand), String> {
    let words = argv
        .iter()
        .skip(1)
        .map(|word| (*word).to_owned())
        .collect::<Vec<_>>();
    let parsed = crate::cli::line::parse(
        &words,
        &crate::cli::line::FirstWordCatalog::default(),
        false,
    )
    .map_err(|error| error.message)?;
    if parsed.first == crate::cli::line::FirstWord::Command {
        return crate::cli::parse(argv);
    }
    let row = command("rows").unwrap();
    let request = crate::cli::run_line::request(&parsed).map_err(|error| error.message)?;
    Ok((
        row,
        command_surface::ParsedCommand {
            request,
            kind: declarations().spec_named("rows").unwrap().kind.clone(),
            json: parsed.line.json,
            calling_process: Default::default(),
        },
    ))
}

/// Read commands left out, each with the reason.
const LEFT_OUT: &[(&str, &str)] = &[(
    "status",
    "it answers with the moment it is asked: uptime and the loop's progress",
)];

/// An answer as the client shows it: the daemon's data, or its refusal.
type Answer = Result<Value, String>;

/// The exit code that names an answer's outcome: 0 ok, 1 empty, 5 degraded,
/// or the refusal's code. A refusal before the request is sent has none.
type Exit = Option<u8>;

fn cli_answer(w: &test_daemon::TestDaemon, argv: &[&str], request_id: &str) -> (Answer, Exit) {
    let (c, values) = match parse_line(argv) {
        Ok(parsed) => parsed,
        Err(e) => return (Err(format!("bad_request: {e}")), None),
    };
    let request = crate::client::request(c.name, values.request.arguments, request_id.into());
    let answer = crate::client::call_ready(&w.daemon.archive, &request).unwrap();
    let exit = answer.exit_code();
    let shown = crate::protocol::into_value(answer).map_err(|error| {
        format!(
            "{}: {}",
            error.code.as_deref().unwrap_or_default(),
            error.message
        )
    });
    (shown, Some(exit))
}

/// The MCP answer for the same request through its declared operation.
fn mcp_answer(
    w: &test_daemon::TestDaemon,
    c: &Command,
    values: &command_surface::ParsedCommand,
    request_id: &str,
) -> Answer {
    let names: Vec<&str> = crate::commands::declarations()
        .spec_named(c.name)
        .expect("declared row")
        .arguments
        .iter()
        .filter(|a| !a.calling_process)
        .map(|a| a.name.as_str())
        .collect();
    let arguments = values
        .request
        .arguments
        .as_object()
        .unwrap()
        .iter()
        .filter(|(k, _)| names.contains(&k.as_str()))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let answer =
        crate::mcp::tests::structured_call_as(&w.daemon.archive, c.name, arguments, request_id);
    crate::protocol::into_value(answer).map_err(|error| {
        format!(
            "{}: {}",
            error.code.as_deref().unwrap_or_default(),
            error.message
        )
    })
}

/// Text as a query parameter value.
fn encoded(text: &str) -> String {
    text.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                (b as char).to_string()
            }
            _ => format!("%{b:02X}"),
        })
        .collect()
}

fn http_answer(
    web: &Arc<crate::web::Web>,
    c: &Command,
    values: &command_surface::ParsedCommand,
    request_id: &str,
) -> (Answer, Exit) {
    let names: Vec<&str> = crate::commands::declarations()
        .spec_named(c.name)
        .expect("declared row")
        .arguments
        .iter()
        .filter(|a| !a.calling_process)
        .map(|a| a.name.as_str())
        .collect();
    let arguments: serde_json::Map<String, Value> = values
        .request
        .arguments
        .as_object()
        .unwrap()
        .iter()
        .filter(|(name, _)| names.contains(&name.as_str()))
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect();
    let arguments = Value::Object(arguments);
    let access =
        crate::daemon::handler::policy(c, &arguments, polylogue_model::authorship::Channel::Mcp)
            .unwrap()
            .access;
    let pairs = [
        format!("operation={}", encoded(c.name)),
        format!("arguments={}", encoded(&arguments.to_string())),
        format!("request_id={}", encoded(request_id)),
    ];
    let uri = format!("/api/v1{}?{}", c.http_path(), pairs.join("&"));
    let (request, body) = if access > crate::commands::ToolAccess::Read {
        let body =
            serde_json::json!({"operation":c.name,"arguments":arguments,"request_id":request_id});
        (
            axum::http::Request::post(format!("/api/v1{}", c.http_path())).header(
                axum::http::header::ORIGIN,
                format!("http://127.0.0.1:{}", web.port),
            ),
            Body::from(body.to_string()),
        )
    } else {
        (axum::http::Request::get(uri), Body::empty())
    };
    let request = request
        .header(axum::http::header::CONTENT_TYPE, "application/json")
        .header(axum::http::header::HOST, format!("127.0.0.1:{}", web.port))
        .header(
            axum::http::header::COOKIE,
            format!("polylogue_{}={}", web.key, web.token),
        )
        .body(body)
        .unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let body: Value = runtime.block_on(async {
        let response = crate::web::router(Arc::clone(web))
            .oneshot(request)
            .await
            .unwrap();
        let status = response.status();
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&bytes).unwrap_or_else(|error| {
            panic!(
                "{} HTTP {status} returned invalid JSON: {error}: {}",
                c.name,
                String::from_utf8_lossy(&bytes)
            )
        })
    });
    let answer: command_surface::Answer<Value> = serde_json::from_value(body).unwrap();
    let exit = answer.exit_code();
    let shown = crate::protocol::into_value(answer).map_err(|error| {
        format!(
            "{}: {}",
            error.code.as_deref().unwrap_or_default(),
            error.message
        )
    });
    (shown, Some(exit))
}

// The socket peer is this fixture process; HTTP has the trusted Web
// channel. Expectations come from those boundaries, never from an answer.
fn socket_authorship() -> Authorship {
    let (_client, server) = std::os::unix::net::UnixStream::pair().unwrap();
    let (details, via) = crate::daemon::server::peer(&server, &Default::default()).unwrap();
    assert_eq!(via, command_surface::Via::Cli);
    command_surface::author_of(&details, via, None).into()
}

fn context_authorship_path(name: &str) -> Option<&'static str> {
    match name {
        "rows" => Some("/reply/resolved/context/authorship"),
        _ => None,
    }
}

fn assert_transport_authorship(answer: &Answer, name: &str, expected: &Authorship) {
    if let (Ok(value), Some(path)) = (answer, context_authorship_path(name)) {
        if name == "rows" && value["kind"] != "listing" {
            return;
        }
        assert_eq!(
            value.pointer(path).expect("operation context authorship"),
            &serde_json::to_value(expected).unwrap()
        );
    }
}

fn expected_transport_answer(baseline: &Answer, name: &str, authorship: &Authorship) -> Answer {
    let mut expected = baseline.clone();
    if let (Ok(value), Some(path)) = (&mut expected, context_authorship_path(name)) {
        if name == "rows" && value["kind"] != "listing" {
            return expected;
        }
        *value
            .pointer_mut(path)
            .expect("operation context authorship") = serde_json::to_value(authorship).unwrap();
    }
    expected
}

// A read's first page records its run after the answer has gone
// (docs/reading.md: "handed to the writer without waiting and committed on
// its next turn"), so the previous call's run
// may still be landing. Observe each invocation's descriptor once the
// position has stopped moving, from a fresh reader released before the call.
fn read_position(w: &test_daemon::TestDaemon) -> polylogue_model::ChangePosition {
    let observe = || w.daemon.reads.reader().unwrap().change_position().unwrap();
    let mut last = observe();
    for _ in 0..50 {
        std::thread::sleep(std::time::Duration::from_millis(100));
        let now = observe();
        if now == last {
            return now;
        }
        last = now;
    }
    panic!("the change position kept moving: {last:?}");
}

fn assert_read_page(answer: &Answer, position: &polylogue_model::ChangePosition) {
    let value = answer.as_ref().expect("session read succeeds");
    let page = crate::test_answers::read(command_surface::Answer::ok(value.clone()));
    assert_eq!(
        &page.snapshot.read_position, position,
        "the page must report the independently observed position"
    );
    assert!(
        page.document.footer.next.is_none(),
        "small full-read fixture fits one page"
    );
    let body = page
        .document
        .items
        .iter()
        .filter_map(|item| match &item.kind {
            polylogue_model::read::ItemKind::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<String>();
    assert_eq!(body, "look at the parser");
}

#[test]
fn internal_completion_http_returns_candidates() {
    let daemon = test_daemon::start("synthetic HTTP completion");
    let web = Arc::new(crate::web::Web {
        daemon: Arc::clone(&daemon.daemon),
        key: "synthetic".into(),
        token: "synthetic".into(),
        port: 8765,
    });
    let (command, arguments) = crate::cli::parse(&[
        "polylogue",
        "internal",
        "complete",
        "tool_calls where outcome:err",
    ])
    .unwrap();
    assert_eq!(command.http_path(), "/internal/complete");
    let (answer, exit) = http_answer(&web, command, &arguments, "internal-http-completion");
    assert_eq!(exit, Some(0));
    let candidates = answer.unwrap();
    // The outcomes are ok, error, pending and unknown; only `error` starts with `err`.
    let values: Vec<&str> = candidates
        .as_array()
        .unwrap()
        .iter()
        .map(|candidate| candidate["value"].as_str().unwrap())
        .collect();
    assert_eq!(values, ["error"], "{candidates}");
}

#[test]
fn every_client_answers_the_same_request_identically() {
    let offered: Vec<&str> = COMMANDS
        .iter()
        .filter(|c| c.tool.is_some() && c.http && !c.writes && matches!(c.run, Run::Daemon(_)))
        .map(|c| c.name)
        .collect();
    for name in &offered {
        assert!(
            ASKED.iter().any(|(asked, _)| asked == name)
                || LEFT_OUT.iter().any(|(left, _)| left == name),
            "the client comparison does not ask `{name}`"
        );
    }
    let w = test_daemon::start_with_clock(
        "look at the parser",
        polylogue_model::WallClock::fixed("2026-03-01T12:00:00Z".parse().unwrap()),
    );
    let web = Arc::new(crate::web::Web {
        daemon: Arc::clone(&w.daemon),
        key: "abc123def456".into(),
        token: "t0ken".into(),
        port: 8765,
    });
    let socket_author = socket_authorship();
    let web_author: Authorship = command_surface::author_of(
        &command_surface::CallerEvidence::default(),
        command_surface::Via::Web,
        None,
    )
    .into();
    for (index, (name, args)) in ASKED.iter().enumerate() {
        let request_id = format!("client-parity-{index}");
        let command = command(name).unwrap();
        let words = command.cli_words();
        let mut argv = vec!["polylogue"];
        if command.cli != Cli::Line {
            argv.extend(words.iter().map(String::as_str));
        }
        argv.extend(*args);
        let (c, values) = parse_line(&argv).unwrap();
        let cli_position = (args.last() == Some(&"read")).then(|| read_position(&w));
        let (cli, cli_exit) = cli_answer(&w, &argv, &request_id);
        let owned = crate::daemon::handler::policy(
            c,
            &values.request.arguments,
            polylogue_model::authorship::Channel::Mcp,
        )
        .unwrap()
        .access
            > crate::commands::ToolAccess::Read;
        let stored = |id: &str| w.daemon.reads.reader().unwrap().user_request(id).unwrap();
        let original =
            owned.then(|| stored(&request_id).expect("the owned line retains its accepted intent"));
        let mcp_position = (args.last() == Some(&"read")).then(|| read_position(&w));
        let mcp = mcp_answer(&w, c, &values, &request_id);
        if let Some(original) = &original {
            assert_eq!(original.authorship, socket_author);
            assert_eq!(
                stored(&request_id).as_ref(),
                Some(original),
                "same-author replay preserves the exact accepted body"
            );
            let (same, exit) = cli_answer(&w, &argv, &request_id);
            assert_eq!((&same, exit), (&cli, cli_exit));
            assert_eq!(stored(&request_id).as_ref(), Some(original));
        }
        let http_position = (args.last() == Some(&"read")).then(|| read_position(&w));
        let http_id = if owned {
            let (refused, _) = http_answer(&web, c, &values, &request_id);
            assert_eq!(
                refused,
                Err(format!(
                    "conflict: request {request_id} was already used for different content"
                )),
                "a different author cannot take over an accepted request"
            );
            assert_eq!(stored(&request_id), original);
            format!("{request_id}-web")
        } else {
            request_id.clone()
        };
        let (http, http_exit) = http_answer(&web, c, &values, &http_id);
        if let Some(original) = &original {
            let independent = stored(&http_id).expect("the web action owns its own request");
            assert_eq!(independent.machine, original.machine);
            assert_eq!(independent.authorship, web_author);
            for (accepted, id) in [(original, &request_id), (&independent, &http_id)] {
                let polylogue_model::user::UserWrite::ResolveLine { request, .. } = &accepted.write
                else {
                    panic!("the accepted line")
                };
                let request: command_surface::RunnableRequest =
                    serde_json::from_value(request.clone()).unwrap();
                assert_eq!(&request.request_id, id);
                assert_eq!(
                    serde_json::from_value::<crate::protocol::line::Rows>(request.arguments)
                        .unwrap(),
                    serde_json::from_value::<crate::protocol::line::Rows>(
                        values.request.arguments.clone()
                    )
                    .unwrap(),
                    "each channel accepted the same semantic line"
                );
            }
            let (again, exit) = http_answer(&web, c, &values, &http_id);
            assert_eq!((&again, exit), (&http, http_exit));
            assert_eq!(stored(&http_id).as_ref(), Some(&independent));
        }
        if cli_exit.is_some() {
            assert_eq!(
                cli_exit,
                http_exit,
                "`{}`: the command line and HTTP name different outcomes; CLI: {cli:?}; HTTP: {http:?}",
                argv.join(" ")
            );
        }
        assert_transport_authorship(&cli, name, &socket_author);
        assert_transport_authorship(&mcp, name, &socket_author);
        assert_transport_authorship(&http, name, &web_author);
        if args.last() == Some(&"read") && args.first() == Some(&"session:claude-code:s1") {
            assert_read_page(&cli, cli_position.as_ref().unwrap());
            assert_read_page(&mcp, mcp_position.as_ref().unwrap());
            assert_read_page(&http, http_position.as_ref().unwrap());
            // Keep the complete raw answer. Only its expected per-invocation
            // descriptor differs, and that value was observed outside the answer.
            let mut expected = cli.as_ref().unwrap().clone();
            expected["reply"]["steps"][0]["outcome"]["reply"]["reply"]["snapshot"]["read_position"] =
                serde_json::to_value(mcp_position.unwrap()).unwrap();
            assert_eq!(Ok(expected.clone()), mcp, "whole MCP read answer");
            expected["reply"]["steps"][0]["outcome"]["reply"]["reply"]["snapshot"]["read_position"] =
                serde_json::to_value(http_position.unwrap()).unwrap();
            assert_eq!(Ok(expected), http, "whole HTTP read answer");
            continue;
        }
        if args.contains(&"brief")
            && let Ok(value) = &cli
        {
            // Session's protocol handler runs in this test process. The
            // daemon observes that socket peer as CLI, so it does not record
            // a model delivery. The MCP process control is in e2e.
            assert_eq!(
                value["reply"]["steps"][0]["outcome"]["reply"]["reply"]["delivery"],
                Value::Null
            );
            assert_eq!(
                http.as_ref().unwrap()["reply"]["steps"][0]["outcome"]["reply"]["reply"]["delivery"],
                Value::Null
            );
            assert_eq!(
                mcp.as_ref().unwrap()["reply"]["steps"][0]["outcome"]["reply"]["reply"]["delivery"],
                Value::Null
            );
        }
        assert_eq!(
            cli,
            mcp,
            "`{}`: the command line and MCP differ",
            argv.join(" ")
        );
        let mut expected_http = expected_transport_answer(&cli, name, &web_author);
        if owned {
            let value = expected_http.as_mut().expect("the independent resume plan");
            assert_eq!(
                value["reply"]["steps"][0]["request_id"],
                format!("{request_id}-1")
            );
            value["reply"]["steps"][0]["request_id"] = serde_json::json!(format!("{http_id}-1"));
        }
        assert_eq!(
            expected_http,
            http,
            "`{}`: the command line and HTTP differ",
            argv.join(" ")
        );
    }
}
