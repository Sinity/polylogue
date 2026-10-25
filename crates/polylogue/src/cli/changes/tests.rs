use super::*;
use serde_json::{Value, json};

fn wire(lines: &[Value]) -> Vec<u8> {
    let mut out = Vec::new();
    for data in lines {
        serde_json::to_writer(&mut out, &Response::Ok { data: data.clone() }).unwrap();
        out.push(b'\n');
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
        assert!(receive(&wire(&lines)[..], &mut retry, Some("g:1:1")).unwrap());
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
    assert!(receive(&input[..], &mut full, Some("g:1:1")).unwrap());
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
    let world = crate::test_daemon::start("synthetic feed source");
    let (_, values) = crate::cli::parse(&["polylogue", "changes"]).unwrap();
    let request = super::request(&values).unwrap();
    assert!(matches!(request, Request::Changes { since: None }));
    let mut output = Vec::new();
    assert!(send(&world.daemon.archive, &request, &mut output).unwrap());
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
    let request = Request::Changes {
        since: Some(checkpoint),
    };
    let mut again = Vec::new();
    assert!(send(&world.daemon.archive, &request, &mut again).unwrap());
    let end = completed(&again);
    assert_eq!(end.len(), 1);
    assert_eq!(end[0]["count"], 0);
}

#[test]
fn a_receiver_stages_the_replacement_references_until_complete_after_a_rebuild() {
    use polylogue_testkit::{ClaudeHome, claude_user};
    use std::collections::BTreeSet;
    use std::io::Read;
    fn prepared(home: &ClaudeHome, since: Option<&str>) -> Vec<u8> {
        let pool = home.reads();
        let reader = pool.reader().unwrap();
        let mut file =
            crate::daemon::query::feed::prepare(&reader, since, 0, tempfile::tempfile().unwrap())
                .unwrap();
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).unwrap();
        bytes
    }
    fn apply_if_complete(
        wire: &[u8],
        old: &BTreeSet<String>,
        since: Option<&str>,
    ) -> Option<(BTreeSet<String>, String)> {
        let mut bytes = Vec::new();
        if !receive(wire, &mut bytes, since).ok()? {
            return None;
        }
        let lines: Vec<Value> = bytes
            .split(|b| *b == b'\n')
            .filter(|s| !s.is_empty())
            .map(|s| serde_json::from_slice(s).unwrap())
            .collect();
        let mut staged = if lines[0]["replacement"] == true {
            BTreeSet::new()
        } else {
            old.clone()
        };
        for line in &lines {
            let change = &line["change"];
            if change["kind"] == "session" || change["kind"] == "note" {
                staged.insert(format!(
                    "{}:{}",
                    change["kind"].as_str().unwrap(),
                    change["ref"].as_str().unwrap()
                ));
            }
            if change["kind"] == "removed" {
                staged.remove(&format!("session:{}", change["ref"].as_str().unwrap()));
            }
        }
        Some((
            staged,
            lines.last().unwrap()["position"]
                .as_str()
                .unwrap()
                .to_owned(),
        ))
    }
    let home = ClaudeHome::new();
    for id in ["a", "b"] {
        home.append(
            &format!("{id}.jsonl"),
            &claude_user(&format!("{id}1"), id, "synthetic reference"),
        );
    }
    let (mut engine, lock) = home.engine(64 << 20);
    assert!(engine.converge().unwrap().errors.is_empty());
    let (old, checkpoint) =
        apply_if_complete(&prepared(&home, None), &BTreeSet::new(), None).unwrap();
    assert_eq!(
        old,
        BTreeSet::from([
            "session:claude-code:a".into(),
            "session:claude-code:b".into()
        ])
    );
    drop(engine);
    drop(lock);
    polylogue_testkit::delete_index(&home.root);
    let (mut engine, _lock) = home.engine(64 << 20);
    assert!(engine.converge().unwrap().errors.is_empty());
    let output = prepared(&home, Some(&checkpoint));
    let positions: Vec<_> = output
        .iter()
        .enumerate()
        .filter(|(_, b)| **b == b'\n')
        .map(|(i, _)| i + 1)
        .collect();
    let mut prior = old.clone();
    prior.insert("session:claude-code:stale-local-reference".into());
    prior.insert("note:stale-note-reference".into());
    for prefix in positions.iter().take(positions.len() - 1) {
        assert!(apply_if_complete(&output[..*prefix], &prior, Some(&checkpoint)).is_none());
        assert!(prior.contains("session:claude-code:stale-local-reference"));
    }
    let (current, next) = apply_if_complete(&output, &prior, Some(&checkpoint)).unwrap();
    assert_eq!(current, old);
    assert_ne!(
        ChangePosition::parse(&next).unwrap().generation,
        ChangePosition::parse(&checkpoint).unwrap().generation
    );
}

#[test]
fn a_prepared_feed_survives_daemon_close_without_holding_its_reader() {
    use std::io::Read;
    let world = crate::test_daemon::start("synthetic finite delivery");
    let mut file = crate::daemon::service::change_feed(&world.daemon, None).unwrap();
    drop(world);
    let mut wire = Vec::new();
    file.read_to_end(&mut wire).unwrap();
    let mut output = Vec::new();
    assert!(receive(&wire[..], &mut output, None).unwrap());
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
    assert!(receive(&input[..], &mut replay, Some("g:1:1")).unwrap());
    assert_eq!(replay, output.0);
}

#[test]
fn feed_refuses_query_page_and_field_flags_instead_of_ignoring_global_arguments() {
    for suffix in [
        vec!["--cursor", "opaque"],
        vec!["--fields", "ref"],
        vec!["--why"],
    ] {
        let mut argv = vec!["polylogue", "changes"];
        argv.extend(suffix);
        let (_, values) = crate::cli::parse(&argv).unwrap();
        assert!(super::request(&values).is_err(), "{argv:?}");
    }
}
