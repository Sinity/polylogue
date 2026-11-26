use super::*;
use serde_json::{Value, json};

fn wire(lines: &[Value]) -> Vec<u8> {
    let mut out = Vec::new();
    for data in lines {
        let name = match data["type"].as_str().unwrap() {
            "change" => "record",
            name => name,
        };
        crate::protocol::stream::event(&mut out, name, data).unwrap();
    }
    out
}

fn example() -> Vec<Value> {
    vec![
        json!({"type":"begin","position":"g:1:1","generation":"g","replacement":false,"generation_changed":false}),
        json!({"type":"change","position":"g:1:1","component":"archive","change":{"kind":"note","ref":"a","seq":2}}),
        json!({"type":"change","position":"g:1:1","component":"archive","change":{"kind":"note","ref":"b","seq":2}}),
        json!({"type":"change","position":"g:1:1","component":"index","change":{"kind":"session","ref":"test:a","seq":2}}),
        json!({"type":"change","position":"g:1:1","component":"index","change":{"kind":"session","ref":"test:b","seq":2}}),
        json!({"type":"complete","position":"g:2:2","count":4,"replacement":false}),
    ]
}

fn completed(bytes: &[u8]) -> Vec<Value> {
    bytes
        .split_inclusive(|b| *b == b'\n')
        .filter(|line| line.ends_with(b"\n"))
        .filter_map(|line| serde_json::from_slice::<Value>(line).ok())
        .filter(|line| line["type"] == "complete")
        .collect()
}

#[test]
fn every_stopped_prefix_keeps_the_previous_position_and_retry_delivers_every_line() {
    let lines = example();
    for prefix in 0..lines.len() {
        let mut output = Vec::new();
        assert!(receive(&wire(&lines[..prefix])[..], &mut output, Some("g:1:1")).is_err());
        assert!(completed(&output).is_empty(), "prefix {prefix}");
        let mut retry = Vec::new();
        assert_eq!(
            receive(&wire(&lines)[..], &mut retry, Some("g:1:1")).unwrap(),
            0
        );
        assert_eq!(completed(&retry)[0]["position"], "g:2:2");
        assert_eq!(
            retry
                .split(|b| *b == b'\n')
                .filter(|line| !line.is_empty())
                .count(),
            6
        );
    }
    let encoded = wire(&lines);
    for cutoff in 0..encoded.len() {
        let mut output = Vec::new();
        assert!(
            receive(&encoded[..cutoff], &mut output, Some("g:1:1")).is_err(),
            "cutoff {cutoff}"
        );
        assert!(completed(&output).is_empty(), "cutoff {cutoff}");
    }
}

struct Stops {
    bytes: Vec<u8>,
    left: usize,
}
impl Write for Stops {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.left == 0 {
            return Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "synthetic output stop",
            ));
        }
        let n = self.left.min(bytes.len());
        self.bytes.extend_from_slice(&bytes[..n]);
        self.left -= n;
        Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn stopped_output_never_forwards_a_complete_new_checkpoint() {
    let input = wire(&example());
    let mut full = Vec::new();
    assert_eq!(receive(&input[..], &mut full, Some("g:1:1")).unwrap(), 0);
    for cutoff in 0..full.len() {
        let mut output = Stops {
            bytes: Vec::new(),
            left: cutoff,
        };
        assert!(receive(&input[..], &mut output, Some("g:1:1")).is_err());
        assert!(completed(&output.bytes).is_empty(), "cutoff {cutoff}");
    }
}

#[test]
fn inconsistent_completion_or_premature_event_advancement_refuses_before_the_new_checkpoint() {
    for mutant in [
        "count",
        "generation",
        "replacement",
        "row-position",
        "no-header",
    ] {
        let mut lines = example();
        match mutant {
            "count" => lines.last_mut().unwrap()["count"] = json!(3),
            "generation" => lines.last_mut().unwrap()["position"] = json!("other:2:2"),
            "replacement" => lines.last_mut().unwrap()["replacement"] = json!(true),
            "row-position" => lines[1]["position"] = json!("g:2:2"),
            "no-header" => {
                lines.remove(0);
            }
            _ => unreachable!(),
        }
        let mut output = Vec::new();
        assert!(
            receive(&wire(&lines)[..], &mut output, Some("g:1:1")).is_err(),
            "{mutant}"
        );
        assert!(completed(&output).is_empty(), "{mutant}");
    }
}

#[test]
fn the_command_socket_and_cli_receiver_deliver_one_finite_batch() {
    let daemon = crate::test_daemon::start("synthetic feed source");
    let (_, values) = crate::cli::parse(&["polylogue", "dev", "feed"]).unwrap();
    let request = request(&values).unwrap();
    assert_eq!(request.arguments, serde_json::json!({}));
    let mut output = Vec::new();
    assert_eq!(
        send(&daemon.daemon.archive, &request, &mut output, None).unwrap(),
        0
    );
    let entries: Vec<Value> = output
        .split(|b| *b == b'\n')
        .filter(|s| !s.is_empty())
        .map(|s| serde_json::from_slice(s).unwrap())
        .collect();
    assert_eq!(entries[0]["replacement"], true);
    assert!(
        entries
            .iter()
            .any(|line| line["change"]["ref"] == "claude-code:s1")
    );
    let last = entries.last().unwrap();
    assert_eq!(last["type"], "complete");
    let checkpoint = last["position"].as_str().unwrap().to_owned();
    let request = crate::client::request(
        "dev feed",
        serde_json::json!({ "after": checkpoint }),
        "changes-again".into(),
    );
    let mut again = Vec::new();
    assert_eq!(
        send(&daemon.daemon.archive, &request, &mut again, None).unwrap(),
        0
    );
    let end = completed(&again);
    assert_eq!(end.len(), 1);
    assert_eq!(end[0]["count"], 0);
}

#[test]
fn a_receiver_stages_the_replacement_references_until_complete_after_a_rebuild() {
    use polylogue_testkit::{ClaudeHome, claude_user};
    use std::io::Read;
    fn prepared(home: &ClaudeHome, since: Option<&str>) -> Vec<u8> {
        let pool = home.reads();
        let reader = pool.reader().unwrap();
        let mut file = crate::daemon::query::feed::prepare_with(
            &reader,
            since,
            &query_language::time::Clock::new(
                "2026-10-07T10:00:00Z".parse().unwrap(),
                query_language::time::DaySetting::new(
                    jiff::tz::TimeZone::UTC,
                    jiff::civil::Time::constant(4, 0, 0, 0),
                )
                .unwrap(),
            ),
            tempfile::tempfile().unwrap(),
        )
        .unwrap();
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).unwrap();
        bytes
    }
    // What `receive` forwards, as JSON lines; None when the batch is not
    // complete.
    fn delivered(wire: &[u8], since: Option<&str>) -> Option<Vec<Value>> {
        let mut bytes = Vec::new();
        if receive(wire, &mut bytes, since).ok()? != 0 {
            return None;
        }
        Some(
            bytes
                .split(|b| *b == b'\n')
                .filter(|s| !s.is_empty())
                .map(|s| serde_json::from_slice(s).unwrap())
                .collect(),
        )
    }
    // The sessions a batch lists (docs/operations.md, The change feed: a full
    // listing names every current reference).
    fn sessions(lines: &[Value]) -> Vec<&str> {
        lines
            .iter()
            .filter(|line| line["change"]["kind"] == "session")
            .map(|line| line["change"]["ref"].as_str().unwrap())
            .collect()
    }
    let home = ClaudeHome::new();
    for id in ["a", "b"] {
        home.append(
            &format!("{id}.jsonl"),
            &claude_user(&format!("{id}1"), id, "synthetic reference"),
        );
    }
    let (mut engine, lock) = home.engine(64 << 20);
    assert!(engine.run_pass().unwrap().errors.is_empty());
    let first = delivered(&prepared(&home, None), None).unwrap();
    assert_eq!(first[0]["replacement"], true);
    assert_eq!(sessions(&first), ["claude-code:a", "claude-code:b"]);
    let checkpoint = first.last().unwrap()["position"]
        .as_str()
        .unwrap()
        .to_owned();
    drop(engine);
    drop(lock);
    polylogue_testkit::delete_index(&home.root);
    let (mut engine, _lock) = home.engine(64 << 20);
    assert!(engine.run_pass().unwrap().errors.is_empty());
    let output = prepared(&home, Some(&checkpoint));
    let positions: Vec<_> = output
        .iter()
        .enumerate()
        .filter(|(_, b)| **b == b'\n')
        .map(|(i, _)| i + 1)
        .collect();
    for prefix in positions.iter().take(positions.len() - 1) {
        assert!(delivered(&output[..*prefix], Some(&checkpoint)).is_none());
    }
    // The index was replaced since the checkpoint: the batch is a full
    // listing of the current sessions, and holds no removals.
    let current = delivered(&output, Some(&checkpoint)).unwrap();
    assert_eq!(current[0]["replacement"], true);
    assert_eq!(sessions(&current), ["claude-code:a", "claude-code:b"]);
    assert!(
        current
            .iter()
            .all(|line| line["change"]["kind"] != "removed")
    );
    let next = current.last().unwrap()["position"].as_str().unwrap();
    assert_ne!(
        ChangePosition::parse(next).unwrap().generation,
        ChangePosition::parse(&checkpoint).unwrap().generation
    );
}

#[test]
fn a_prepared_feed_survives_daemon_close_without_holding_its_reader() {
    let daemon = crate::test_daemon::start("synthetic finite delivery");
    let mut wire = Vec::new();
    let operation = daemon
        .daemon
        .operation_context(&polylogue_reading::operation::Context::worker_authorship(
            "finite-feed-fixture",
        ))
        .unwrap();
    <crate::daemon::service::feeds::Changes as crate::daemon::handler::Feed>::produce(
        serde_json::from_value(serde_json::json!({})).unwrap(),
        &daemon.daemon,
        &operation,
        &mut wire,
    )
    .unwrap();
    drop(daemon);
    let mut output = Vec::new();
    assert_eq!(receive(&wire[..], &mut output, None).unwrap(), 0);
    assert_eq!(completed(&output).len(), 1);
}

#[test]
fn failed_final_flush_is_an_unknown_delivery_outcome_and_can_replay_from_the_prior_position() {
    struct FlushFails(Vec<u8>);
    impl Write for FlushFails {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Err(io::Error::other("synthetic flush failure"))
        }
    }
    let input = wire(&example());
    let mut output = FlushFails(Vec::new());
    assert!(receive(&input[..], &mut output, Some("g:1:1")).is_err());
    // Bytes may have escaped. Failure does not prove the receiver rejected
    // them; replay sends the same IDs and the previous completed position.
    let mut replay = Vec::new();
    assert_eq!(receive(&input[..], &mut replay, Some("g:1:1")).unwrap(), 0);
    assert_eq!(replay, output.0);
}

#[test]
fn feed_refuses_query_page_and_field_flags_instead_of_ignoring_global_arguments() {
    let root = tempfile::tempdir().unwrap();
    let engine = crate::test_daemon::engine_config(root.path().join("home"));
    let startup = crate::config::Startup::for_test(engine, root.path());
    // The feed declares no query flags: the parser refuses them.
    for flag in ["--fields", "--why"] {
        let error = crate::cli::command()
            .try_get_matches_from(["polylogue", "dev", "feed", flag, "ref"])
            .unwrap_err();
        assert_eq!(error.exit_code(), 2, "{flag}");
    }
    for suffix in [vec!["--cursor", "opaque"], vec!["--budget", "none"]] {
        let mut argv = vec!["polylogue", "dev", "feed"];
        argv.extend(suffix);
        let matches = crate::cli::command().try_get_matches_from(&argv).unwrap();
        assert_eq!(
            crate::cli::run(&matches, &startup),
            ExitCode::from(2),
            "{argv:?}"
        );
        assert!(!startup.config.archive.join("daemon.sock").exists());
    }
    let error = crate::cli::command()
        .try_get_matches_from(["polylogue", "dev", "feed", "--fresh"])
        .unwrap_err();
    assert_eq!(error.exit_code(), 2);
}

fn request(v: &command_surface::ParsedCommand) -> Result<command_surface::RunnableRequest, String> {
    Ok(crate::client::request(
        "dev feed",
        v.request.arguments.clone(),
        "feed".into(),
    ))
}

#[test]
fn a_change_feed_deadline_uses_timeout_exit_code() {
    let mut output = Vec::new();
    let mut wire = Vec::new();
    crate::protocol::stream::error(
        &mut wire,
        crate::daemon::Failure::deadline_exceeded("read deadline exceeded"),
    )
    .unwrap();
    assert_eq!(receive(&wire[..], &mut output, None).unwrap(), 7);
}
