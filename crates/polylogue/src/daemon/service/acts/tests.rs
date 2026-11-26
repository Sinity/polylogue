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
        .invocation(&polylogue_reading::operation::Context::worker_authorship(
            "acts-test",
        ))
}

/// A prepared message to `s1`, as a message to a running session leaves it.
fn prepared(d: &test_daemon::TestDaemon, id: &str, text: &str) {
    author(
        &d.daemon,
        &invocation(d),
        id.into(),
        UserWrite::Deliver(Delivery {
            recipient_kind: "session".into(),
            recipient_id: Some("claude-code:s1".into()),
            sender: DeliverySender::Send,
            event: DeliveryEvent::Message,
            cwd: None,
            text: text.into(),
            refs: vec![],
            left_out: vec![],
        }),
    )
    .unwrap();
}

fn claim_ids(d: &test_daemon::TestDaemon) -> Vec<String> {
    let answer =
        crate::test_daemon::Call::new("internal mail-claim", serde_json::json!({"session": s1()}))
            .authorship(invocation(d).authorship)
            .answer(&d.daemon);
    crate::test_answers::json(answer)["deliveries"]
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
    prepared(&d, "message-a", "first");
    prepared(&d, "message-b", "second");
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
    assert_eq!(sorted, ["message-a", "message-b"]);
    assert!(!crate::mailbox::raised(d.daemon.acts.mail(), &s1()));
    assert!(
        claim_ids(&d).is_empty(),
        "a claimed delivery is never sent again"
    );
    // The hook records `delivered` once it has written the text; until
    // then the claim leaves no outcome, and nothing hands it out again.
    let reader = d.daemon.reads.reader().unwrap();
    let (row, _) = reader.delivery("message-a").unwrap().unwrap();
    assert!(row.events.is_empty());
}

/// A message is refused, and the refusal recorded, when no channel reaches
/// the session: a live message to a session that ended, or a session whose
/// harness shows no hook.
#[test]
fn a_message_with_no_channel_is_refused_and_recorded() {
    let send = |d: &test_daemon::TestDaemon, live: bool| {
        crate::test_answers::json(
            super::send(
                &d.daemon,
                &invocation(d),
                &polylogue_store::user_writes::new_request_id(),
                "@[claude-code:s1]",
                "rebase after my commit",
                live,
                None,
            )
            .unwrap(),
        )
    };
    let ended = test_daemon::start_with_clock("ended", at("2026-03-02T10:00:00Z"));
    let answer = send(&ended, true);
    assert_eq!(answer["deliveries"][0]["outcome"], "not_sent");
    assert_eq!(answer["deliveries"][0]["reason"], "session_not_live");
    let delivery = answer["deliveries"][0]["delivery"].as_str().unwrap();
    ended.wait_for(|status| status["pending_writes"]["delivery_outcomes"] == 0);
    let reader = ended.daemon.reads.reader().unwrap();
    let (row, text) = reader.delivery(delivery).unwrap().unwrap();
    assert_eq!(row.events, ["not_sent"]);
    assert!(text.contains("rebase after my commit"));
    assert!(text.contains(&format!("delivery:{delivery}")));
    drop(reader);
    // Running by its transcript's growth, but no hook event: no channel.
    let running = test_daemon::start_with_clock("running", at("2026-03-01T10:05:00Z"));
    let answer = send(&running, false);
    assert_eq!(answer["deliveries"][0]["reason"], "unsupported");
    assert!(!crate::mailbox::raised(running.daemon.acts.mail(), &s1()));
}

/// A broadcast resolves its query once, when sent: one message annotation
/// about each recipient and one delivery each.
#[test]
fn a_broadcast_resolves_its_recipients_when_sent() {
    let d = test_daemon::start_with_clock("broadcast", at("2026-03-02T10:00:00Z"));
    let answer = crate::test_answers::json(
        super::send(
            &d.daemon,
            &invocation(&d),
            &polylogue_store::user_writes::new_request_id(),
            "@[sessions]",
            "pause pushes",
            true,
            None,
        )
        .unwrap(),
    );
    let deliveries = answer["deliveries"].as_array().unwrap();
    assert_eq!(deliveries.len(), 1);
    assert_eq!(deliveries[0]["session"], "claude-code:s1");
    assert!(answer["message"].as_str().is_some());
}

/// Without a pending permission request there is nothing to answer.
#[test]
fn an_answer_needs_a_pending_request() {
    let d = test_daemon::start_with_clock("answer", at("2026-03-01T10:05:00Z"));
    let answer = super::answer(
        &d.daemon,
        &invocation(&d),
        &polylogue_store::user_writes::new_request_id(),
        &s1(),
        true,
        None,
        true,
    )
    .unwrap();
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

/// `resume --in` starts a new hosted session in another harness, never a
/// resume or fork of the source: its first turn opens with the brief's
/// delivery reference, and names the source it was briefed from and the
/// message it continues from. Preparing the briefing starts nothing, and a
/// harness with no control protocol is refused.
#[test]
fn resume_in_starts_a_new_session_briefed_from_the_source() {
    let d = test_daemon::start_with_clock("hand-off source text", at("2026-03-01T10:05:00Z"));
    let fixture = hosted_codex(&d);
    let work = tempfile::tempdir().unwrap();
    let request_id = polylogue_store::user_writes::new_request_id();
    let source = s1();
    let continuation = |harness| Continuation {
        request_id: &request_id,
        source: &source,
        harness,
        cwd: Some(work.path()),
        at: Some("message:claude-code:u1"),
        stop_source: false,
    };
    let printed = super::briefing(&d.daemon, &invocation(&d), &continuation("codex")).unwrap();
    assert!(printed.text.starts_with("Hand-off from claude-code:s1"));
    assert_eq!(printed.cwd.as_deref(), Some(work.path()));
    assert!(
        turns(fixture.path()).is_empty(),
        "a briefing starts nothing"
    );
    let refused = serde_json::to_value(
        super::resume_in(&d.daemon, &invocation(&d), &continuation("chatgpt")).unwrap(),
    )
    .unwrap();
    assert_eq!(refused["error"]["kind"], "refused");
    let answer = crate::test_answers::json(
        super::resume_in(&d.daemon, &invocation(&d), &continuation("codex")).unwrap(),
    );
    assert_eq!(answer["session"], "codex:t1");
    assert_eq!(answer["briefed_from"], "claude-code:s1");
    let delivery = answer["delivery"].as_str().unwrap();
    let first = turns(fixture.path());
    assert_eq!(first.len(), 1);
    assert!(first[0].starts_with(&format!(
        "delivery:{delivery}\nHand-off from claude-code:s1"
    )));
    d.wait_for(|status| status["pending_writes"]["delivery_outcomes"] == 0);
    let reader = d.daemon.reads.reader().unwrap();
    let (row, _) = reader.delivery(delivery).unwrap().unwrap();
    assert_eq!(
        (row.event.as_str(), row.sender.as_str()),
        ("first_turn", "act")
    );
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
            .into_iter()
            .map(Result::unwrap)
            .any(|(_, r)| r.session == "codex:t1")
    );
}

/// A role address resolves to a running instance, else starts one where
/// the role last ran; a role that does not exist is not found, and a
/// role never seen running has no directory to start in.
#[test]
fn role_addresses_and_role_starts() {
    use polylogue_model::annotation::{AnnotationKind, NewAnnotation};
    let d = test_daemon::start_with_clock("roles", at("2026-03-01T10:05:00Z"));
    let fixture = hosted_codex(&d);
    let work = tempfile::tempdir().unwrap();
    let send = |target: &str| {
        serde_json::to_value(
            super::send(
                &d.daemon,
                &invocation(&d),
                &polylogue_store::user_writes::new_request_id(),
                target,
                "review my diff",
                false,
                None,
            )
            .unwrap_or_else(|failure| failure.answer()),
        )
        .unwrap()
    };
    assert_eq!(send("@[role:reviewer]")["error"]["kind"], "not_found");
    author(
        &d.daemon,
        &invocation(&d),
        polylogue_store::user_writes::new_request_id(),
        UserWrite::Assert {
            annotations: vec![NewAnnotation {
                kind: AnnotationKind::Role,
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
    let refused = send("@[role:reviewer]");
    assert_eq!(refused["error"]["kind"], "refused");
    assert_eq!(
        refused["error"]["message"],
        "no_live_instance: the role reviewer has no running session and none ran before to take a directory from; start one with `polylogue start --role reviewer --cwd DIR TEXT`"
    );
    let started = crate::test_answers::ok(crate::test_daemon::Call::new("start", serde_json::json!({"role": Some("reviewer"), "cwd": work.path(), "text": "look at the store"})).authorship(invocation(&d).authorship).answer(&d.daemon));
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
    // The instance runs on its hosted connection (no live events have
    // arrived yet), so a message reaches it rather than starting another.
    let pinged = crate::test_answers::json(
        super::send(
            &d.daemon,
            &invocation(&d),
            &polylogue_store::user_writes::new_request_id(),
            "@[role:reviewer]",
            "review my diff",
            false,
            None,
        )
        .unwrap(),
    );
    assert_eq!(pinged["deliveries"][0]["session"], "codex:t1");
    assert_eq!(pinged["deliveries"][0]["channel"], "protocol");
    assert_eq!(pinged["deliveries"][0]["outcome"], "delivered");
    assert!(turns(fixture.path())[1].contains("review my diff"));
    // Once the daemon lets go of it, the role has no running instance, and
    // a message starts one where the last one ran.
    d.daemon
        .acts
        .release_host(&SessionId::parse("codex:t1").unwrap());
    let restarted = crate::test_answers::json(
        super::send(
            &d.daemon,
            &invocation(&d),
            &polylogue_store::user_writes::new_request_id(),
            "@[role:reviewer]",
            "one more",
            false,
            None,
        )
        .unwrap(),
    );
    assert_eq!(restarted["deliveries"][0]["session"], "codex:t2");
    assert!(turns(fixture.path())[2].contains("one more"));
    // A role's harness and an explicit other harness conflict.
    let conflict = serde_json::to_value(crate::test_daemon::Call::new("start", serde_json::json!({"in": Some("hermes"), "role": Some("reviewer"), "cwd": work.path(), "text": "x"})).authorship(invocation(&d).authorship).answer(&d.daemon))
    .unwrap();
    assert_eq!(conflict["error"]["kind"], "bad_value");
}

/// Markup is not parsed yet, so the pack names the MCP operation that
/// records an answer.
#[test]
fn message_text_names_the_write_tool() {
    let answer = "Answer with Polylogue's MCP `rows` operation (tool `polylogue_write`): `selection` and `replies_to` both name `annotation:p1@p1`; use a `note` step with `as: finding` and your answer as `text`. Reach the sender with a `send` step (tool `polylogue_act`).";
    assert_eq!(
        message_text("d1", "p1", "operator", "rebase please", None),
        format!("Message from operator (delivery:d1, annotation:p1):\nrebase please\n{answer}")
    );
    assert_eq!(
        message_text("d1", "p1", "operator", "rebase please", Some("session:x")),
        format!(
            "Message from operator (delivery:d1, annotation:p1):\nrebase please\nAbout: session:x\n{answer}"
        )
    );
}

/// A message that starts a new role instance tells it how to answer.
#[test]
fn new_instance_ping_text_says_how_to_reply() {
    let text = new_instance_message_text("d1", "operator", "rebase please", Some("session:x"));
    assert_eq!(
        text,
        "Message from operator:\nrebase please\nAbout: session:x\nAnswer with a note replying to `delivery:d1` (`replies_to`, tool `polylogue_write`), or reach the sender with a `send` step (tool `polylogue_act`)."
    );
}

/// An answer written the way the pack says is an annotation linked to the message.
#[test]
fn an_authored_finding_replies_to_the_message() {
    use polylogue_model::annotation::{AnnotationKind, LinkRole, NewAnnotation, NewAnnotationLink};
    let d = test_daemon::start_with_clock("answer through assert", at("2026-03-02T10:00:00Z"));
    let sent = crate::test_answers::json(
        super::send(
            &d.daemon,
            &invocation(&d),
            &polylogue_store::user_writes::new_request_id(),
            "@[claude-code:s1]",
            "is the rebase done?",
            true,
            None,
        )
        .unwrap(),
    );
    let message = sent["message"].as_str().unwrap().to_owned();
    let delivery = sent["deliveries"][0]["delivery"].as_str().unwrap();
    let reader = d.daemon.reads.reader().unwrap();
    let (_, text) = reader.delivery(delivery).unwrap().unwrap();
    drop(reader);
    let instruction =
        format!("`selection` and `replies_to` both name `annotation:{message}@{message}`");
    assert!(text.contains(&instruction), "{text}");
    let answer = NewAnnotation {
        kind: AnnotationKind::Finding,
        key: None,
        value: Some(serde_json::json!("rebased onto master")),
        valid_until: None,
        links: [LinkRole::About, LinkRole::RepliesTo]
            .into_iter()
            .map(|role| NewAnnotationLink {
                role,
                reference: polylogue_model::Reference::parse(&format!(
                    "annotation:{message}@{message}"
                ))
                .unwrap(),
            })
            .collect(),
    };
    let written = author(
        &d.daemon,
        &invocation(&d),
        "answer".into(),
        UserWrite::Assert {
            annotations: vec![answer],
        },
    )
    .unwrap();
    let id = written.id.as_str();
    let reader = d.daemon.reads.reader().unwrap();
    let stored = reader
        .annotation(&polylogue_model::RevisionReference::exact(
            id.into(),
            id.into(),
        ))
        .unwrap()
        .unwrap();
    assert!(
        stored.fact.links.iter().any(|link| link.role == LinkRole::RepliesTo
            && matches!(&link.reference, polylogue_model::Reference::Annotation(target) if target.root == message)),
        "{:?}",
        stored.fact.links
    );
}

/// A headless resume runs only in the session's recorded directory: with
/// none recorded, or one that no longer exists, it is `not_sent` by name
/// and nothing runs in the daemon's own directory.
#[test]
fn resume_with_missing_directory_is_refused_by_name() {
    let d = test_daemon::start_with_clock("no directory", at("2026-03-02T10:00:00Z"));
    std::fs::write(
        d.home.join(".claude/projects/-p/s9.jsonl"),
        format!(
            "{}\n",
            serde_json::json!({
                "type": "user", "uuid": "u9", "sessionId": "s9",
                "timestamp": "2026-03-01T10:00:00Z", "cwd": "/home/op/gone",
                "message": {"role": "user", "content": "plan the otter release"}
            })
        ),
    )
    .unwrap();
    d.wait_for(|status| status["store"]["messages"] == 2);
    let detail = |session: &str, request: &str| {
        let answer = crate::test_answers::ok(
            super::send(
                &d.daemon,
                &invocation(&d),
                request,
                session,
                "rebase",
                false,
                None,
            )
            .unwrap(),
        );
        let row = &answer.deliveries[0];
        assert_eq!(row.outcome, ActOutcome::NotSent);
        let delivery = row.delivery.as_deref().unwrap();
        d.wait_for(|status| status["pending_writes"]["delivery_outcomes"] == 0);
        let reader = d.daemon.reads.reader().unwrap();
        let rows = reader
            .query(
                "SELECT outcome, detail FROM delivery_event WHERE delivery=?1",
                &[polylogue_store::SqlValue::Text(delivery.into())],
            )
            .unwrap();
        assert_eq!(rows.len(), 1, "{rows:?}");
        assert_eq!(rows[0]["outcome"], "not_sent");
        rows[0]["detail"].as_str().unwrap().to_owned()
    };
    assert_eq!(
        detail("claude-code:s1", "resume-unknown"),
        "directory_unknown"
    );
    assert_eq!(
        detail("claude-code:s9", "resume-missing"),
        "directory_missing(/home/op/gone)"
    );
}
