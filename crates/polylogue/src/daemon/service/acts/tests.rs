use super::*;
use crate::test_daemon;

fn at(text: &str) -> polylogue_model::WallClock {
    polylogue_model::WallClock::fixed(text.parse().unwrap())
}

fn s1() -> SessionId {
    SessionId::parse("claude-code:s1").unwrap()
}

fn invocation(d: &test_daemon::TestDaemon) -> Invocation {
    d.daemon
        .invocation(&crate::operation::Context::worker_authorship("acts-test"))
}

/// A prepared ping to `s1`, as a ping to a running session leaves it.
fn prepared(d: &test_daemon::TestDaemon, id: &str, text: &str) {
    author(
        &d.daemon,
        &invocation(d),
        id.into(),
        UserWrite::Deliver(Delivery {
            recipient_kind: "session".into(),
            recipient_id: Some("claude-code:s1".into()),
            sender: DeliverySender::Ping.as_str().into(),
            event: DeliveryEvent::Ping.as_str().into(),
            cwd: None,
            text: text.into(),
            refs: vec![],
            left_out: vec![],
        }),
    )
    .unwrap();
}

fn claim_ids(d: &test_daemon::TestDaemon) -> Vec<String> {
    let answer = handle(
        Request::Claim {
            session: s1(),
            event: "PostToolUse".into(),
        },
        &d.daemon,
        &invocation(d),
    );
    crate::test_answers::ok(answer)["deliveries"]
        .as_array()
        .unwrap()
        .iter()
        .map(|i| i["id"].as_str().unwrap().to_owned())
        .collect()
}

/// Concurrent claims of one session hand each pending delivery over
/// exactly once and lower the flag.
#[test]
fn concurrent_claims_hand_each_delivery_over_once() {
    let d = test_daemon::start_with_clock("mailbox", at("2026-03-01T10:05:00Z"));
    prepared(&d, "ping-a", "first");
    prepared(&d, "ping-b", "second");
    crate::mailbox::raise(d.daemon.acts.mail(), &s1()).unwrap();
    let claimed: Vec<String> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..4).map(|_| scope.spawn(|| claim_ids(&d))).collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect()
    });
    let mut sorted = claimed.clone();
    sorted.sort();
    assert_eq!(sorted, ["ping-a", "ping-b"]);
    assert!(!crate::mailbox::raised(d.daemon.acts.mail(), &s1()));
    assert!(
        claim_ids(&d).is_empty(),
        "a claimed delivery is never sent again"
    );
    // The hook records `delivered` once it has written the text; until
    // then the claim leaves no outcome, and nothing hands it out again.
    let reader = d.daemon.reads.reader().unwrap();
    let (row, _) = reader.delivery("ping-a").unwrap().unwrap();
    assert!(row.events.is_empty());
}

/// A ping is refused, and the refusal recorded, when no channel reaches
/// the session: a live ping to a session that ended, or a session whose
/// harness shows no hook.
#[test]
fn a_ping_with_no_channel_is_refused_and_recorded() {
    let ping = |d: &test_daemon::TestDaemon, live: bool| {
        crate::test_answers::ok(handle(
            Request::Ping {
                request_id: polylogue_store::user_writes::new_request_id(),
                target: "@[claude-code:s1]".into(),
                text: "rebase after my commit".into(),
                live,
                about: None,
            },
            &d.daemon,
            &invocation(d),
        ))
    };
    let ended = test_daemon::start_with_clock("ended", at("2026-03-02T10:00:00Z"));
    let answer = ping(&ended, true);
    assert_eq!(answer["deliveries"][0]["outcome"], "not_sent");
    assert_eq!(answer["deliveries"][0]["reason"], "session_not_live");
    let delivery = answer["deliveries"][0]["delivery"].as_str().unwrap();
    let reader = ended.daemon.reads.reader().unwrap();
    let (row, text) = reader.delivery(delivery).unwrap().unwrap();
    assert_eq!(row.events, ["not_sent"]);
    assert!(text.contains("rebase after my commit"));
    assert!(text.contains(&format!("delivery:{delivery}")));
    drop(reader);
    // Running by its transcript's growth, but no hook evidence: no channel.
    let running = test_daemon::start_with_clock("running", at("2026-03-01T10:05:00Z"));
    let answer = ping(&running, false);
    assert_eq!(answer["deliveries"][0]["reason"], "unsupported");
    assert!(!crate::mailbox::raised(running.daemon.acts.mail(), &s1()));
}

/// A broadcast resolves its query once, when sent: one ping assertion
/// about each recipient and one delivery each.
#[test]
fn a_broadcast_resolves_its_recipients_when_sent() {
    let d = test_daemon::start_with_clock("broadcast", at("2026-03-02T10:00:00Z"));
    let answer = crate::test_answers::ok(handle(
        Request::Ping {
            request_id: polylogue_store::user_writes::new_request_id(),
            target: "@[sessions]".into(),
            text: "pause pushes".into(),
            live: true,
            about: None,
        },
        &d.daemon,
        &invocation(&d),
    ));
    let deliveries = answer["deliveries"].as_array().unwrap();
    assert_eq!(deliveries.len(), 1);
    assert_eq!(deliveries[0]["session"], "claude-code:s1");
    assert!(answer["ping"].as_str().is_some());
}

/// Without a pending permission request there is nothing to answer.
#[test]
fn an_answer_needs_a_pending_request() {
    let d = test_daemon::start_with_clock("answer", at("2026-03-01T10:05:00Z"));
    let answer = handle(
        Request::Answer {
            request_id: polylogue_store::user_writes::new_request_id(),
            session: s1(),
            approve: true,
            text: None,
            standing: true,
        },
        &d.daemon,
        &invocation(&d),
    );
    let value = serde_json::to_value(&answer).unwrap();
    assert_eq!(value["error"]["kind"], "refused");
}

/// A fixture Codex app-server behind `codex app-server proxy`: each
/// `thread/start` makes thread `t<N>`; each `turn/start` is written to
/// `turns`.
fn fake_codex(dir: &std::path::Path) -> std::path::PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let program = dir.join("codex");
    std::fs::write(
        &program,
        r#"#!/bin/sh
dir=$(dirname "$0")
while IFS= read -r line; do
  id=$(printf '%s' "$line" | sed -n 's/.*"id":\([0-9]*\).*/\1/p')
  [ -z "$id" ] && continue
  case "$line" in
    *'"thread/start"'*)
      n=$(cat "$dir/threads" 2>/dev/null || echo 0); n=$((n+1)); echo "$n" > "$dir/threads"
      printf '{"id":%s,"result":{"thread":{"id":"t%s"}}}\n' "$id" "$n" ;;
    *'"turn/start"'*) printf '%s\n' "$line" >> "$dir/turns"; printf '{"id":%s,"result":{}}\n' "$id" ;;
    *) printf '{"id":%s,"result":{}}\n' "$id" ;;
  esac
done
"#,
    )
    .unwrap();
    std::fs::set_permissions(&program, std::fs::Permissions::from_mode(0o755)).unwrap();
    program
}

/// The first turns the fixture app-server was given, in order.
fn turns(dir: &std::path::Path) -> Vec<String> {
    std::fs::read_to_string(dir.join("turns"))
        .unwrap_or_default()
        .lines()
        .map(|l| {
            let v: Value = serde_json::from_str(l).unwrap();
            v["params"]["input"][0]["text"].as_str().unwrap().to_owned()
        })
        .collect()
}

fn hosted_codex(d: &test_daemon::TestDaemon) -> tempfile::TempDir {
    let fixture = tempfile::tempdir().unwrap();
    let program = fake_codex(fixture.path());
    d.daemon
        .acts
        .set_codex(program, fixture.path().join("app-server-control.sock"));
    fixture
}

/// A notice to a session that is not running is recorded `not_sent` with
/// sender `signal` and event `notice`, and nothing resumes the session.
#[test]
fn a_notice_to_a_stopped_session_is_not_sent_and_starts_nothing() {
    let d = test_daemon::start_with_clock("notice", at("2026-03-02T10:00:00Z"));
    let sent = notice(
        &d.daemon,
        "main-force-pushed",
        "@[claude-code:s1]",
        "main moved",
    )
    .unwrap();
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0]["outcome"], "not_sent");
    assert_eq!(sent[0]["reason"], "session_not_live");
    let delivery = sent[0]["delivery"].as_str().unwrap();
    let reader = d.daemon.reads.reader().unwrap();
    let (row, text) = reader.delivery(delivery).unwrap().unwrap();
    assert_eq!(
        (row.sender.as_str(), row.event.as_str()),
        ("signal", "notice")
    );
    assert_eq!(row.events, ["not_sent"]);
    assert!(text.contains("main moved") && text.contains(&format!("delivery:{delivery}")));
    // Nothing else reached the session: no resumed run, no flag.
    assert_eq!(reader.deliveries_to("claude-code:s1").unwrap().len(), 1);
    drop(reader);
    assert!(!crate::mailbox::raised(d.daemon.acts.mail(), &s1()));
}

/// A hand-off starts a new hosted session in another harness, never a
/// resume or fork of the source: its first turn opens with the pack's
/// delivery reference, and the `pack` delivery names the source it was
/// briefed from and the message it continues from. `--print` starts
/// nothing, and a harness with no control protocol is refused.
#[test]
fn a_hand_off_starts_a_new_session_briefed_from_the_source() {
    let d = test_daemon::start_with_clock("hand-off source text", at("2026-03-01T10:05:00Z"));
    let fixture = hosted_codex(&d);
    let work = tempfile::tempdir().unwrap();
    let hand_off = |to: &str, print: bool| {
        handle(
            Request::HandOff {
                request_id: polylogue_store::user_writes::new_request_id(),
                session: s1(),
                to: to.into(),
                cwd: Some(work.path().to_owned()),
                at: Some("message:claude-code:u1".into()),
                stop_source: false,
                print,
            },
            &d.daemon,
            &invocation(&d),
        )
    };
    let printed = crate::test_answers::ok(hand_off("codex", true));
    assert!(
        printed["pack"]
            .as_str()
            .unwrap()
            .contains("Hand-off from claude-code:s1")
    );
    assert!(turns(fixture.path()).is_empty(), "print starts nothing");
    let refused = serde_json::to_value(hand_off("chatgpt", false)).unwrap();
    assert_eq!(refused["error"]["kind"], "refused");
    let answer = crate::test_answers::ok(hand_off("codex", false));
    assert_eq!(answer["session"], "codex:t1");
    assert_eq!(answer["briefed_from"], "claude-code:s1");
    let delivery = answer["delivery"].as_str().unwrap();
    let first = turns(fixture.path());
    assert_eq!(first.len(), 1);
    assert!(first[0].starts_with(&format!(
        "delivery:{delivery}\nHand-off from claude-code:s1"
    )));
    let reader = d.daemon.reads.reader().unwrap();
    let (row, _) = reader.delivery(delivery).unwrap().unwrap();
    assert_eq!((row.event.as_str(), row.sender.as_str()), ("pack", "act"));
    assert_eq!(row.recipient_id.as_deref(), Some("codex:t1"));
    assert_eq!(row.events, ["delivered"]);
    drop(reader);
    assert!(
        d.daemon
            .acts
            .is_hosted(&SessionId::parse("codex:t1").unwrap())
    );
    assert!(
        crate::hosting::records(&d.daemon.acts.hosts)
            .iter()
            .any(|(_, r)| r.session == "codex:t1")
    );
}

/// A role address resolves to a running instance, else starts one where
/// the role last ran; a role that does not exist is not found, and a
/// role never seen running has no directory to start in.
#[test]
fn role_addresses_and_role_starts() {
    use polylogue_model::assertion::{AssertionKind, NewAssertion};
    let d = test_daemon::start_with_clock("roles", at("2026-03-01T10:05:00Z"));
    let fixture = hosted_codex(&d);
    let work = tempfile::tempdir().unwrap();
    let ping = |target: &str| {
        serde_json::to_value(handle(
            Request::Ping {
                request_id: polylogue_store::user_writes::new_request_id(),
                target: target.into(),
                text: "review my diff".into(),
                live: false,
                about: None,
            },
            &d.daemon,
            &invocation(&d),
        ))
        .unwrap()
    };
    assert_eq!(ping("@[role:reviewer]")["error"]["kind"], "not_found");
    author(
        &d.daemon,
        &invocation(&d),
        polylogue_store::user_writes::new_request_id(),
        UserWrite::Assert {
            assertions: vec![NewAssertion {
                kind: AssertionKind::Role,
                key: Some("reviewer".into()),
                value: Some(
                    json!({"instructions": "Review diffs for data safety.", "harness": "codex"}),
                ),
                valid_until: None,
                links: vec![],
            }],
        },
    )
    .unwrap();
    let refused = ping("@[role:reviewer]");
    assert!(
        refused["error"]["message"]
            .as_str()
            .unwrap()
            .contains("no_live_instance")
    );
    let started = crate::test_answers::ok(handle(
        Request::Start {
            request_id: polylogue_store::user_writes::new_request_id(),
            harness: None,
            role: Some("reviewer".into()),
            cwd: work.path().to_owned(),
            text: "look at the store".into(),
        },
        &d.daemon,
        &invocation(&d),
    ));
    assert_eq!(started["session"], "codex:t1");
    assert_eq!(started["role"], "reviewer");
    let first = &turns(fixture.path())[0];
    assert!(first.contains("You run as the role `reviewer`"));
    assert!(first.contains("Review diffs for data safety.") && first.contains("look at the store"));
    let reader = d.daemon.reads.reader().unwrap();
    let instances = reader.sessions_started_with("role:reviewer").unwrap();
    assert_eq!(instances.len(), 1);
    assert_eq!(instances[0].0, "codex:t1");
    drop(reader);
    // The instance runs on its hosted connection (no live evidence has
    // arrived yet), so a ping reaches it rather than starting another.
    let pinged = crate::test_answers::ok(handle(
        Request::Ping {
            request_id: polylogue_store::user_writes::new_request_id(),
            target: "@[role:reviewer]".into(),
            text: "review my diff".into(),
            live: false,
            about: None,
        },
        &d.daemon,
        &invocation(&d),
    ));
    assert_eq!(pinged["deliveries"][0]["session"], "codex:t1");
    assert_eq!(pinged["deliveries"][0]["channel"], "protocol");
    assert_eq!(pinged["deliveries"][0]["outcome"], "delivered");
    assert!(turns(fixture.path())[1].contains("review my diff"));
    // Once the daemon lets go of it, the role has no running instance, and
    // a ping starts one where the last one ran.
    d.daemon
        .acts
        .release_host(&SessionId::parse("codex:t1").unwrap());
    let restarted = crate::test_answers::ok(handle(
        Request::Ping {
            request_id: polylogue_store::user_writes::new_request_id(),
            target: "@[role:reviewer]".into(),
            text: "one more".into(),
            live: false,
            about: None,
        },
        &d.daemon,
        &invocation(&d),
    ));
    assert_eq!(restarted["deliveries"][0]["session"], "codex:t2");
    assert!(turns(fixture.path())[2].contains("one more"));
    // A role's harness and an explicit other harness conflict.
    let conflict = serde_json::to_value(handle(
        Request::Start {
            request_id: polylogue_store::user_writes::new_request_id(),
            harness: Some("hermes".into()),
            role: Some("reviewer".into()),
            cwd: work.path().to_owned(),
            text: "x".into(),
        },
        &d.daemon,
        &invocation(&d),
    ))
    .unwrap();
    assert_eq!(conflict["error"]["kind"], "bad_value");
}
