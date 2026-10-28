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
