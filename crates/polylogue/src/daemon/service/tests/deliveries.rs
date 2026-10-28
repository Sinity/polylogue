//! A delivery is recorded once per request: a hook that resends its request
//! after a lost answer gets the same delivery back.

use super::*;
use crate::briefing::Ask;

fn takeover(d: &test_daemon::TestDaemon, request_id: &str) -> serde_json::Value {
    let request = Request::Context {
        request_id: request_id.into(),
        ask: Ask::Takeover {
            session: polylogue_model::SessionId::parse("claude-code:s1").unwrap(),
            for_session: None,
        },
        sender: Some(polylogue_model::DeliverySender::Hook.as_str().into()),
    };
    match handle(request, &d.daemon) {
        Response::Ok { data, .. } => data,
        other => panic!("{other:?}"),
    }
}

fn deliveries(d: &test_daemon::TestDaemon) -> u64 {
    let request = Request::Query {
        text: "deliveries".into(),
        cursor: None,
        page_size: None,
        why: false,
        session: None,
        fields: vec![],
    };
    match handle(request, &d.daemon) {
        Response::Ok { data, .. } => data["count"].as_u64().unwrap(),
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_resent_context_request_is_one_delivery() {
    let d = test_daemon::start("the walrus migration");
    let request_id = polylogue_store::user_writes::new_request_id();
    let first = takeover(&d, &request_id);
    let delivery = first["delivery"].as_str().unwrap().to_owned();
    let again = takeover(&d, &request_id);
    assert_eq!(again["delivery"], delivery.as_str());
    assert_eq!(deliveries(&d), 1);
    let other = takeover(&d, &polylogue_store::user_writes::new_request_id());
    assert_ne!(other["delivery"], delivery.as_str());
    assert_eq!(deliveries(&d), 2);
}

#[test]
fn an_accepted_prepared_receipt_crosses_the_read_deadline_without_a_false_refusal() {
    use polylogue_store::read::limit::ReadLimit;
    use std::time::{Duration, Instant};
    let d = test_daemon::start("receipt handoff");
    let blocker = std::fs::File::open(d.daemon.archive.archive_db()).unwrap();
    // Readers may start; the writer waits only at the register commit boundary.
    blocker.lock_shared().unwrap();
    let id = polylogue_store::user_writes::new_request_id();
    let limit = ReadLimit::new(Duration::from_secs(2), Default::default());
    let request = Request::Context {
        request_id: id.clone(),
        ask: Ask::Takeover {
            session: polylogue_model::SessionId::parse("claude-code:s1").unwrap(),
            for_session: None,
        },
        sender: Some("mcp".into()),
    };
    let (response, handed_off) = std::thread::scope(|scope| {
        let worker = scope.spawn(|| super::super::handle_with_limit(request, &d.daemon, &limit));
        let until = Instant::now() + Duration::from_secs(5);
        while limit.stage() != Some("record delivery")
            && !worker.is_finished()
            && Instant::now() < until
        {
            std::thread::yield_now();
        }
        let handed_off = limit.stage() == Some("record delivery");
        if handed_off {
            while limit.check().is_ok() {
                std::thread::yield_now();
            }
        }
        drop(blocker);
        (worker.join().unwrap(), handed_off)
    });
    assert!(
        handed_off,
        "context never handed off its receipt: {response:?}"
    );
    assert!(matches!(
        limit.check(),
        Err(polylogue_store::StoreError::ReadDeadlineExceeded)
    ));
    let Response::Ok { data, .. } = response else {
        panic!("accepted receipt was reported as a refusal: {response:?}");
    };
    assert_eq!(data["delivery"], id);
    let reader = d.daemon.reads.reader().unwrap();
    let saved = reader.user_request(&id).unwrap().unwrap();
    assert!(matches!(
        saved.write,
        polylogue_store::UserWrite::Deliver(_)
    ));
    let events = reader
        .query(
            "SELECT count(*) AS n FROM delivery_event WHERE delivery=?1",
            &[polylogue_store::SqlValue::Text(id)],
        )
        .unwrap();
    assert_eq!(events[0]["n"], 0, "prepared is not a delivered event");
}
