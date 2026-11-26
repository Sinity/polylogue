//! One table of lines over the Morrow archive, run through the CLI with
//! `--json`, the socket, HTTP and MCP: every surface gives the same rows,
//! totals and refusals, because one daemon function answers them all
//! (docs/latency.md, the daemon's states; docs/interfaces.md, operation
//! table).

use super::*;
use serde_json::{Value, json};
use std::io::Write;
use std::os::unix::net::UnixStream;

/// Each line and what Morrow answers it with (examples/morrow/expected).
const LINES: &[&str] = &[
    // Session #5 alone is in the morrow repository and started on or after the day.
    "repo:morrow since:2026-10-05",
    // Four sessions mention chrono.
    "chrono",
    "sessions | group by origin | count",
    "session:codex:01a20000-0005-7000-8000-000000000005",
    "tool_calls where outcome:error",
    // Refusals: a misspelt unit and a value outside a closed set.
    "messges cargo",
    "sessions where outcome:error",
];

/// What every surface must agree on: the rows and their total, or the
/// refusal's kind, code and message.
fn agreed(answer: &command_surface::Answer<Value>) -> Value {
    match &answer.outcome {
        command_surface::Outcome::Error(error) => json!({
            "refused": error.kind.name(),
            "code": error.code,
            "message": error.message,
        }),
        _ => {
            let value = answer.value().unwrap();
            json!({
                "outcome": answer.exit_code(),
                "kind": value["kind"],
                "rows": value["reply"]["rows"],
                "count": value["reply"]["count"],
            })
        }
    }
}

/// The daemon's HTTP port and token, from its runtime directory.
fn http_access(env_root: &Path) -> (u16, String) {
    let runtime = env_root.join("runtime/polylogue");
    let until = Instant::now() + READY_DEADLINE;
    loop {
        if let Ok(entries) = std::fs::read_dir(&runtime) {
            for entry in entries {
                let path = entry.unwrap().path();
                if let (Ok(port), Ok(token)) = (
                    std::fs::read_to_string(path.join("port")),
                    std::fs::read_to_string(path.join("token")),
                ) {
                    return (port.trim().parse().unwrap(), token.trim().to_owned());
                }
            }
        }
        assert!(
            Instant::now() < until,
            "the daemon wrote no HTTP access files"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn http_answer(port: u16, token: &str, line: &str) -> command_surface::Answer<Value> {
    let mut peer = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
    let selection: String = line
        .bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                (b as char).to_string()
            }
            _ => format!("%{b:02X}"),
        })
        .collect();
    write!(
        peer,
        "GET /api/v1/rows?selection={selection} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nAuthorization: Bearer {token}\r\nConnection: close\r\n\r\n"
    )
    .unwrap();
    let mut response = String::new();
    peer.read_to_string(&mut response).unwrap();
    let (head, body) = response.split_once("\r\n\r\n").unwrap();
    assert!(head.contains("content-type: application/json"), "{head}");
    serde_json::from_str(body).unwrap()
}

fn socket_answer_of(archive: &Path, line: &str) -> command_surface::Answer<Value> {
    let mut stream = UnixStream::connect(archive.join("daemon.sock")).unwrap();
    let request = command_surface::RunnableRequest::new(
        "rows",
        json!({"selection": line}),
        uuid::Uuid::now_v7().to_string(),
    );
    writeln!(stream, "{}", serde_json::to_string(&request).unwrap()).unwrap();
    super::socket_answer(stream)
}

fn cli_answer(env: &crate::Fixture, line: &str) -> command_surface::Answer<Value> {
    let out = cli(env, &["--json", line]);
    let answer: command_surface::Answer<Value> = serde_json::from_slice(&out.stdout)
        .unwrap_or_else(|error| panic!("{line}: {error}: {out:?}"));
    assert_eq!(
        out.status.code(),
        Some(i32::from(answer.exit_code())),
        "{out:?}"
    );
    answer
}

fn mcp_answer(mcp: &mut super::query_surfaces::Mcp, line: &str) -> command_surface::Answer<Value> {
    let reply = mcp.request(
        "tools/call",
        json!({"name":"polylogue_read", "arguments":{
            "operation":"rows", "arguments":{"selection":line},
            "request_id":uuid::Uuid::now_v7().to_string()
        }}),
    );
    assert!(reply["error"].is_null(), "{reply}");
    serde_json::from_value(reply["result"]["structuredContent"].clone()).unwrap()
}

#[test]
fn every_surface_gives_the_same_rows_totals_and_refusals_on_morrow() {
    let temp = tempfile::tempdir().unwrap();
    let operator = crate::Fixture::new(
        &temp.path().join("operator-archive"),
        &temp.path().join("operator-home"),
        &temp.path().join("config"),
    );
    let directory = temp.path().join("morrow");
    let mut demo = Command::new(crate::BIN);
    operator.environment(&mut demo);
    let built = demo.arg("demo").arg(&directory).output().unwrap();
    assert!(built.status.success(), "{built:?}");
    let env = crate::Fixture::new(
        &directory.join("archive"),
        &directory.join("home"),
        &directory.join("config"),
    );
    let _daemon = start_daemon(&env);
    let status = wait_status(&env, |_| true);
    assert_eq!(status["store"]["sessions"], 8);
    let (port, token) = http_access(&directory);
    let mut mcp = super::query_surfaces::Mcp::start(&env);
    for line in LINES {
        let expected = agreed(&cli_answer(&env, line));
        for (surface, answer) in [
            ("socket", socket_answer_of(env.archive(), line)),
            ("http", http_answer(port, &token, line)),
            ("mcp", mcp_answer(&mut mcp, line)),
        ] {
            assert_eq!(agreed(&answer), expected, "{surface}: {line}");
        }
    }
    let ids = |line: &str| {
        let answer = cli_answer(&env, line);
        agreed(&answer)["rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row["id"].as_str().unwrap().to_owned())
            .collect::<Vec<_>>()
    };
    assert_eq!(
        ids("repo:morrow since:2026-10-05"),
        ["codex:01a20000-0005-7000-8000-000000000005"]
    );
    assert_eq!(ids("chrono").len(), 4);
    assert_eq!(
        agreed(&cli_answer(&env, "messges cargo"))["refused"],
        "not_a_query"
    );
    assert_eq!(
        agreed(&cli_answer(&env, "sessions where outcome:error"))["refused"],
        "bad_value"
    );
}
