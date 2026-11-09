//! Client responses match: the command line, MCP and the HTTP API
//! give the daemon the same request for the same arguments and hand back
//! the complete answer, with native transport provenance and observed read positions.

use super::*;
use crate::test_daemon;
use axum::body::{Body, to_bytes};
use polylogue_model::authorship::{Authorship, Channel};
use std::sync::Arc;
use tower::ServiceExt;

/// Every read command all three surfaces offer, with the arguments it is
/// asked with here.
const ASKED: &[(&str, &[&str])] = &[
    ("query", &["sessions"]),
    ("query", &["messages where parser"]),
    ("query", &["sessions where"]),
    ("explain", &["syntax"]),
    ("sql", &["SELECT id FROM v1_sessions ORDER BY id"]),
    ("schema infer", &[]),
    ("schema unread", &[]),
    (
        "complete",
        &["sessions where ori | count", "--position", "18"],
    ),
    ("list", &[]),
    ("read", &["session:claude-code:s1"]),
    ("read", &["session:claude-code:missing"]),
    ("attachment", &["claude-code:missing", "0"]),
    ("read", &["assertion:missing"]),
    ("continue", &["claude-code:s1"]),
    ("brief", &["--cwd", "/synthetic", "--source", "startup"]),
    (
        "advisory",
        &[
            "Review parser.rs",
            "--cwd",
            "/synthetic",
            "--session",
            "codex:parity",
        ],
    ),
    ("takeover", &["claude-code:s1"]),
    ("takeover", &["claude-code:missing"]),
    ("recall", &["parser", "--cwd", "/synthetic"]),
    ("chat list", &[]),
    ("backfill list", &[]),
    ("timeline", &["--date", "2024-01-01"]),
    ("cost", &["--date", "2024-01-01"]),
];

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
    let cwd = std::env::current_dir().unwrap();
    let (c, _, arguments) = match crate::cli::request_arguments(argv, &cwd) {
        Ok(parsed) => parsed,
        Err(e) => return (Err(format!("bad_request: {e}")), None),
    };
    let request = crate::client::request(c.name, Value::Object(arguments), request_id.into());
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

/// The MCP answer for the same request through its declared operation;
/// `complete` also has the protocol's completion method.
fn mcp_answer(
    w: &test_daemon::TestDaemon,
    c: &Command,
    values: &Values,
    request_id: &str,
) -> Answer {
    if c.name == "complete" {
        return crate::mcp::tests::complete(
            &w.daemon.archive,
            &values.text("text").unwrap_or_default(),
            values.integer("position"),
        );
    }
    let names: Vec<&str> = c.args_for(Surface::Mcp).map(|a| a.name).collect();
    let arguments = values
        .0
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
    values: &Values,
    request_id: &str,
) -> (Answer, Exit) {
    let mut pairs: Vec<String> = Vec::new();
    let names: Vec<&str> = c.args_for(Surface::Http).map(|a| a.name).collect();
    for (name, value) in values
        .0
        .iter()
        .filter(|(name, _)| names.contains(&name.as_str()))
    {
        let texts = match value {
            Value::Array(items) => items
                .iter()
                .map(|item| {
                    item.as_str()
                        .map_or_else(|| item.to_string(), str::to_owned)
                })
                .collect(),
            Value::String(text) => vec![text.clone()],
            Value::Bool(false) | Value::Null => Vec::new(),
            other => vec![other.to_string()],
        };
        pairs.extend(texts.iter().map(|t| format!("{name}={}", encoded(t))));
    }
    let arguments: serde_json::Map<String, Value> = values
        .0
        .iter()
        .filter(|(name, _)| names.contains(&name.as_str()))
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect();
    pairs = vec![
        format!("operation={}", encoded(c.name)),
        format!(
            "arguments={}",
            encoded(&Value::Object(arguments).to_string())
        ),
        format!("request_id={}", encoded(request_id)),
    ];
    let uri = format!("/api/v1{}?{}", c.http_path(), pairs.join("&"));
    let request = axum::http::Request::get(uri)
        .header(axum::http::header::HOST, format!("127.0.0.1:{}", web.port))
        .header(
            axum::http::header::COOKIE,
            format!("polylogue_{}={}", web.key, web.token),
        )
        .body(Body::empty())
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
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    });
    let answer: command_surface::Answer<Value> = serde_json::from_value(body).unwrap();
    let exit = answer.exit_code();
    let shown = crate::protocol::into_value(answer)
        .map_err(|error| format!("{}: {}", error.code.unwrap(), error.message));
    (shown, Some(exit))
}

// The socket peer is this actual fixture process; HTTP has the trusted Web
// channel. Expectations come from those boundaries, never from an answer.
fn socket_authorship(w: &test_daemon::TestDaemon) -> Authorship {
    let (_client, server) = std::os::unix::net::UnixStream::pair().unwrap();
    let (evidence, via) = crate::protocol::caller::peer(
        &server,
        &Default::default(),
        w.daemon.config.requests.machine(),
    )
    .unwrap();
    assert_eq!(via, Channel::Cli);
    polylogue_model::authorship::compute_authorship(via, &evidence, None)
}

fn context_authorship_path(name: &str) -> Option<&'static str> {
    match name {
        "query" | "list" => Some("/resolved/context/authorship"),
        "timeline" | "cost" => Some("/context/authorship"),
        _ => None,
    }
}

fn assert_transport_authorship(answer: &Answer, name: &str, expected: &Authorship) {
    if let (Ok(value), Some(path)) = (answer, context_authorship_path(name)) {
        assert_eq!(
            value.pointer(path).expect("operation context authorship"),
            &serde_json::to_value(expected).unwrap()
        );
    }
}

fn expected_transport_answer(baseline: &Answer, name: &str, authorship: &Authorship) -> Answer {
    let mut expected = baseline.clone();
    if let (Ok(value), Some(path)) = (&mut expected, context_authorship_path(name)) {
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
// position has settled, from a fresh native reader released before the call.
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
    let value = answer.as_ref().expect("canonical session read succeeds");
    let page: polylogue_model::read::ReadPage = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(
        &page.snapshot.read_at, position,
        "the page must report the independently observed native descriptor"
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
        operations: Arc::clone(w.operations()),
        key: "abc123def456".into(),
        token: "t0ken".into(),
        port: 8765,
    });
    let socket_author = socket_authorship(&w);
    let web_author =
        polylogue_model::authorship::compute_authorship(Channel::Web, &Default::default(), None);
    for (index, (name, args)) in ASKED.iter().enumerate() {
        let request_id = format!("client-parity-{index}");
        let command = command(name).unwrap();
        let words = command.cli_words();
        let mut argv = vec!["polylogue"];
        if command.cli != Cli::NoSubcommand {
            argv.extend(words.iter().map(String::as_str));
        }
        argv.extend(*args);
        let (c, values) = crate::cli::parse(&argv).unwrap();
        let cli_position = (*name == "read").then(|| read_position(&w));
        let (cli, cli_exit) = cli_answer(&w, &argv, &request_id);
        let mcp_position = (*name == "read").then(|| read_position(&w));
        let mcp = mcp_answer(&w, c, &values, &request_id);
        let http_position = (*name == "read").then(|| read_position(&w));
        let (http, http_exit) = http_answer(&web, c, &values, &request_id);
        if cli_exit.is_some() {
            assert_eq!(
                cli_exit,
                http_exit,
                "`{}`: the command line and HTTP name different outcomes",
                argv.join(" ")
            );
        }
        assert_transport_authorship(&cli, name, &socket_author);
        assert_transport_authorship(&mcp, name, &socket_author);
        assert_transport_authorship(&http, name, &web_author);
        if *name == "read" && args.first() == Some(&"session:claude-code:s1") {
            assert_read_page(&cli, cli_position.as_ref().unwrap());
            assert_read_page(&mcp, mcp_position.as_ref().unwrap());
            assert_read_page(&http, http_position.as_ref().unwrap());
            // Keep the complete raw answer. Only its expected per-invocation
            // descriptor differs, and that value was observed outside the answer.
            let mut expected = cli.as_ref().unwrap().clone();
            expected["snapshot"]["read_at"] = serde_json::to_value(mcp_position.unwrap()).unwrap();
            assert_eq!(Ok(expected.clone()), mcp, "whole MCP read answer");
            expected["snapshot"]["read_at"] = serde_json::to_value(http_position.unwrap()).unwrap();
            assert_eq!(Ok(expected), http, "whole HTTP read answer");
            continue;
        }
        if matches!(*name, "brief" | "takeover" | "recall" | "advisory")
            && let Ok(value) = &cli
        {
            // Session's protocol handler runs in this test process. The
            // daemon observes that native peer as CLI, so it does not record
            // a model delivery. The native MCP process control is in e2e.
            assert_eq!(value["delivery"], Value::Null);
            assert_eq!(http.as_ref().unwrap()["delivery"], Value::Null);
            assert_eq!(mcp.as_ref().unwrap()["delivery"], Value::Null);
        }
        assert_eq!(
            cli,
            mcp,
            "`{}`: the command line and MCP differ",
            argv.join(" ")
        );
        assert_eq!(
            expected_transport_answer(&cli, name, &web_author),
            http,
            "`{}`: the command line and HTTP differ",
            argv.join(" ")
        );
    }
}
