//! Client responses match: the command line, MCP and the HTTP API
//! give the daemon the same request for the same arguments and hand back
//! the same answer, refusals included.

use super::*;
use crate::daemon::Response;
use crate::test_daemon;
use axum::body::{Body, to_bytes};
use std::sync::Arc;
use tower::ServiceExt;

/// Every read command all three surfaces offer, with the arguments it is
/// asked with here.
const ASKED: &[(&str, &[&str])] = &[
    ("query", &["sessions"]),
    ("query", &["messages where parser"]),
    ("query", &["sessions where"]),
    ("explain", &["syntax"]),
    ("schema infer", &[]),
    ("schema unread", &[]),
    (
        "complete",
        &["sessions where ori | count", "--position", "18"],
    ),
    ("list", &[]),
    ("read", &["session:claude-code:s1"]),
    ("outline", &["claude-code:s1"]),
    ("read", &["session:claude-code:missing"]),
    ("attachment", &["claude-code:missing", "0"]),
    ("note list", &[]),
    ("note read", &["missing"]),
    ("continue", &["claude-code:s1"]),
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

fn cli_answer(w: &test_daemon::TestDaemon, argv: &[&str]) -> Answer {
    let (c, values) = crate::cli::parse(argv).unwrap();
    let (Action::Request(build) | Action::Stream { build, .. }) = c.action else {
        panic!("{} is local", c.name)
    };
    let caller = Caller {
        surface: Surface::Cli,
        cwd: std::env::current_dir().ok(),
        json: true,
    };
    let request = build(c, &values, &caller).map_err(|e| format!("bad_request: {e}"))?;
    match crate::client::send(&w.daemon.archive.socket(), &request).unwrap() {
        Response::Ok { data, .. } => Ok(data),
        Response::Error { kind, message, .. } => Err(format!("{kind}: {message}")),
    }
}

fn mcp_answer(w: &test_daemon::TestDaemon, c: &Command, values: &Values) -> Answer {
    let names: Vec<&str> = c.args_for(Surface::Mcp).map(|a| a.name).collect();
    let arguments = values
        .0
        .iter()
        .filter(|(k, _)| names.contains(&k.as_str()))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    match crate::mcp::tests::call(&w.daemon.archive, &c.mcp_name(), arguments) {
        Ok(text) => Ok(serde_json::from_str(&text).unwrap()),
        // `error (kind): message`, with the corrected query on a last line
        // of its own when there is one.
        Err(text) => {
            let text = text
                .rsplit_once("\ntry: ")
                .map_or(text.as_str(), |(t, _)| t);
            let (kind, message) = text
                .strip_prefix("error (")
                .and_then(|rest| rest.split_once("): "))
                .unwrap_or_else(|| panic!("{text}"));
            Err(format!("{kind}: {message}"))
        }
    }
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

fn http_answer(web: &Arc<crate::web::Web>, c: &Command, values: &Values) -> Answer {
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
    match answer {
        command_surface::Answer::Error { error, .. } => {
            Err(format!("{}: {}", error.code.unwrap(), error.message))
        }
        answer => Ok(answer.value().unwrap().clone()),
    }
}

// A full read records its own Run. Observe each invocation's descriptor from a
// fresh native reader before making the call, releasing it before transport I/O.
fn read_position(w: &test_daemon::TestDaemon) -> polylogue_model::ChangePosition {
    w.daemon.reads.reader().unwrap().change_position().unwrap()
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
        .filter(|c| c.mcp && c.http && !c.writes && matches!(c.action, Action::Request(_)))
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
    for (name, args) in ASKED {
        let command = command(name).unwrap();
        let words = command.cli_words();
        let mut argv = vec!["polylogue"];
        if command.cli != Cli::NoSubcommand {
            argv.extend(words.iter().map(String::as_str));
        }
        argv.extend(*args);
        let (c, values) = crate::cli::parse(&argv).unwrap();
        let cli_position = (*name == "read").then(|| read_position(&w));
        let cli = cli_answer(&w, &argv);
        let mcp_position = (*name == "read").then(|| read_position(&w));
        let mcp = mcp_answer(&w, c, &values);
        let http_position = (*name == "read").then(|| read_position(&w));
        let http = http_answer(&web, c, &values);
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
        assert_eq!(
            cli,
            mcp,
            "`{}`: the command line and MCP differ",
            argv.join(" ")
        );
        assert_eq!(
            cli,
            http,
            "`{}`: the command line and HTTP differ",
            argv.join(" ")
        );
    }
}
