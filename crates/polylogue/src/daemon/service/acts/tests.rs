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

/// law: pl.live.mailbox-once
/// Concurrent claims of one session hand each pending delivery over
/// exactly once, oldest first, and lower the flag.
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
    assert!(claim_ids(&d).is_empty(), "a claimed delivery is never sent again");
    let reader = d.daemon.reads.reader().unwrap();
    let (row, _) = reader.delivery("ping-a").unwrap().unwrap();
    assert_eq!(row.events, ["delivered"]);
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

/// law: pl.live.broadcast-frozen
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
