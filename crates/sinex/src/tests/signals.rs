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
fn generic_source_paths_refuse_live_and_restored_signal_state_before_spool_publication() {
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
        original: None,
        request: polylogue_store::fixture::authored::operator_request(
            "signal-fixture".into(),
            UserWrite::SignalCommit(Box::new(Commit {
                prior: None,
                checkpoint: checkpoint.clone(),
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
    assert!(matches!(
        sinex.prepare_request(&prepared),
        Err(crate::Error::Refused { .. })
    ));
    let attribution = polylogue_model::authored::ImportAttribution {
        machine: "fixture-machine".into(),
        input: "bundle".into(),
        chain: 1,
        actor: polylogue_model::Author::Operator,
        at: polylogue_model::Timestamp::from_unix_ms(1).unwrap(),
    };
    for product in [
        Product::Request(prepared.clone()),
        Product::RestoredRequest {
            prepared,
            claims: vec![attribution.clone()],
            current: attribution,
        },
    ] {
        let write = ProductWrite {
            product,
            location: None,
            parents: vec![],
            machine: None,
            output_index: 0,
        };
        assert!(matches!(
            sinex.prepare_named_products("signal-state", std::slice::from_ref(&write)),
            Err(crate::Error::Refused { .. })
        ));
        assert!(matches!(
            sinex.prepare_products(std::slice::from_ref(&write)),
            Err(crate::Error::Refused { .. })
        ));
        assert!(matches!(
            sinex.complete_captures(&[], &[write]),
            Err(crate::Error::Refused { .. })
        ));
    }
    let mut ordinary = PreparedRequest {
        original: None,
        request: polylogue_store::fixture::authored::operator_request(
            "ordinary-fixture".into(),
            UserWrite::SetSetting {
                key: "example".into(),
                value: "value".into(),
                machine: None,
            },
        ),
        outcome: UserOutcome {
            id: "ordinary-fixture".into(),
            pending: false,
            repeat: false,
        },
        effect: AuthoredEffect::default(),
    };
    ordinary.effect.signal_checkpoints.push(checkpoint);
    assert!(matches!(
        sinex.prepare_request(&ordinary),
        Err(crate::Error::Refused { .. })
    ));
    assert!(
        sinex
            .queue()
            .unwrap()
            .pending_after(None, 1)
            .unwrap()
            .is_empty()
    );
}
