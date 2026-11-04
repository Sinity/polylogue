use super::*;
use polylogue_model::{
    authored::{AuthoredEffect, PreparedRequest},
    signal::{Checkpoint, Commit},
    user::{UserOutcome, UserWrite},
};

struct Calendar;
impl shared_signals::Calendar for Calendar {
    fn logical_day(&self, _: i64) -> String {
        "synthetic-day".into()
    }
    fn minute(&self, _: i64) -> u16 {
        0
    }
    fn next_at(&self, after: i64, _: u16) -> i64 {
        after + 1
    }
}

#[test]
fn signal_state_is_an_ordinary_authored_request_held_in_the_spool() {
    let root = tempfile::tempdir().unwrap();
    let sinex = client(root.path());
    let checkpoint = Checkpoint {
        owner: "fixture-machine".into(),
        evaluator: shared_signals::Evaluator::new(0, vec![], &Calendar).unwrap(),
        rules: vec![],
        host: None,
        notifications: None,
    };
    let prepared = PreparedRequest {
        request: polylogue_store::fixture::authored::operator_request(
            "signal-fixture".into(),
            UserWrite::SignalCommit(Box::new(Commit {
                prior: None,
                checkpoint,
                outbox: vec![],
                events: vec![],
            })),
        ),
        outcome: UserOutcome {
            id: "signal-fixture".into(),
            pending: false,
            repeat: false,
        },
        effect: AuthoredEffect::default(),
    };
    let outcome = sinex.prepare_request(&prepared).unwrap();
    assert_eq!(outcome.id, "signal-fixture");
    assert!(outcome.pending);
    let mut pending = Vec::new();
    polylogue_store::primary::Sinex::visit_pending_requests(&sinex, &mut |request| {
        pending.push(request.clone());
        Ok(())
    })
    .unwrap();
    assert_eq!(
        pending,
        [polylogue_model::authored::PendingAuthored::Live { prepared }]
    );
}
