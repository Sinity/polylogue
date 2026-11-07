use super::*;
use crate::primary::authored::{apply, prepare, restore};
use polylogue_model::{
    Author, Timestamp, authored::PendingAuthored, signal::Event, user::UserWrite,
};
use shared_signals::{Calendar, DeliveryOutcome, DeliveryReceipt, Evaluator};

struct CalendarFixture;
impl Calendar for CalendarFixture {
    fn logical_day(&self, at: i64) -> String {
        (at / 86_400_000).to_string()
    }
    fn minute(&self, at: i64) -> u16 {
        ((at / 60_000) % 1440) as u16
    }
    fn next_at(&self, after: i64, minute: u16) -> i64 {
        after
            + (i64::from(minute) + 1440 - i64::from(self.minute(after))) % 1440 * 60_000
            + 86_400_000
    }
}
fn checkpoint_value() -> Checkpoint {
    Checkpoint {
        owner: "host".into(),
        evaluator: Evaluator::new(0, vec![], &CalendarFixture).unwrap(),
        rules: vec![],
        host: None,
        notifications: None,
    }
}
fn cache() -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    conn.pragma_update(None, "foreign_keys", true).unwrap();
    conn.execute_batch(&crate::source::schema().unwrap().ddl)
        .unwrap();
    conn
}
fn request(id: &str, commit: Commit) -> UserRequest {
    UserRequest {
        id: id.into(),
        machine: "host".into(),
        authorship: polylogue_model::authorship::Authorship {
            author: Author::Script("signal-fixture".into()),
            via: polylogue_model::authorship::Channel::Daemon,
        },
        at: Timestamp::from_unix_ms(100).unwrap(),
        write: UserWrite::SignalCommit(Box::new(commit)),
    }
}
fn prepare_pending(
    conn: &Connection,
    r: &UserRequest,
    pending: &[polylogue_model::authored::PreparedRequest],
) -> Result<polylogue_model::authored::PreparedRequest> {
    prepare(
        conn,
        r,
        None,
        None,
        |visit| {
            for p in pending {
                visit(&PendingAuthored::Live {
                    prepared: p.clone(),
                })?;
            }
            Ok(())
        },
        |_| Ok(()),
    )
}
fn initial() -> Commit {
    Commit {
        prior: None,
        checkpoint: checkpoint_value(),
        outbox: vec![crate::fixture::signals::outbox(
            "rule", "session", 100, false,
        )],
        events: vec![],
    }
}
fn admit(conn: &mut Connection, p: &polylogue_model::authored::PreparedRequest, seq: i64) {
    let tx = conn.transaction().unwrap();
    apply(&tx, p, seq).unwrap();
    tx.commit().unwrap();
}

#[test]
fn signal_checkpoint_outbox_and_receipt_freeze_once_across_pending_and_restart() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("source.db");
    let mut conn = Connection::open(&path).unwrap();
    conn.execute_batch(&crate::source::schema().unwrap().ddl)
        .unwrap();
    let first = initial();
    let id = first.outbox[0].metadata.delivery.id().to_owned();
    let p = prepare_pending(&conn, &request("first", first.clone()), &[]).unwrap();
    assert!(checkpoint(&conn, Source::Cache, "host").unwrap().is_none());
    let mut next = first.clone();
    next.prior = Some(first.checkpoint.identity().unwrap());
    next.outbox.clear();
    next.events = vec![Event::Outlet {
        receipt: DeliveryReceipt {
            delivery_id: id.clone(),
            at: 101,
            outlet: "value".into(),
            outcome: DeliveryOutcome::Unknown,
        },
    }];
    let q = prepare_pending(
        &conn,
        &request("next", next.clone()),
        std::slice::from_ref(&p),
    )
    .unwrap();
    assert_eq!(q.effect.signal_events.len(), 1);
    assert!(
        prepare_pending(
            &conn,
            &request("stale", first.clone()),
            std::slice::from_ref(&p)
        )
        .is_err()
    );
    admit(&mut conn, &p, 1);
    admit(&mut conn, &q, 2);
    drop(conn);
    let conn = Connection::open(&path).unwrap();
    let reopened = prepare_pending(&conn, &request("next", next.clone()), &[]).unwrap();
    assert_eq!(reopened, q);
    let mut conflict = next;
    conflict.events.clear();
    assert!(prepare_pending(&conn, &request("next", conflict), &[]).is_err());
    let events = events(&conn, Source::Cache, &id).unwrap();
    assert_eq!(events[0].position, 1);
    assert!(
        matches!(&events[0].event,Event::Outlet{receipt} if receipt.outcome==DeliveryOutcome::Unknown)
    );
}

#[test]
fn signal_historical_reverse_snapshots_preserve_receipts_without_authorizing_effects() {
    let mut origin = cache();
    let first = initial();
    let id = first.outbox[0].metadata.delivery.id().to_owned();
    let p = prepare_pending(&origin, &request("first", first.clone()), &[]).unwrap();
    admit(&mut origin, &p, 1);
    let mut next = first.clone();
    next.prior = Some(first.checkpoint.identity().unwrap());
    next.outbox.clear();
    next.events = vec![Event::Outlet {
        receipt: DeliveryReceipt {
            delivery_id: id.clone(),
            at: 101,
            outlet: "value".into(),
            outcome: DeliveryOutcome::Unknown,
        },
    }];
    let q = prepare_pending(&origin, &request("next", next), &[]).unwrap();
    let mut target = cache();
    for (seq, prepared) in [(1, &q), (2, &p), (3, &q)] {
        let tx = target.transaction().unwrap();
        restore(&tx, prepared, seq).unwrap();
        tx.commit().unwrap();
    }
    assert!(
        checkpoint(&target, Source::Cache, "host")
            .unwrap()
            .is_none()
    );
    assert_eq!(
        metadata(&target, Source::Cache, &id)
            .unwrap()
            .unwrap()
            .execution,
        ExecutionEligibility::Historical
    );
    assert_eq!(
        events(&target, Source::Cache, &id).unwrap(),
        q.effect.signal_events
    );
    let tx = target.transaction().unwrap();
    apply(&tx, &p, 4).unwrap();
    tx.commit().unwrap();
    assert_eq!(
        metadata(&target, Source::Cache, &id)
            .unwrap()
            .unwrap()
            .execution,
        ExecutionEligibility::Historical
    );
    let mut corrupt = q.clone();
    corrupt.effect.signal_events[0].position = 2;
    assert!(super::validate(&corrupt).is_err());
}

#[test]
fn signal_source_schema_is_shared_by_ledger_and_cache_and_rollback_is_atomic() {
    let mut conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(crate::schema::ARCHIVE).unwrap();
    let r = request("rolled-back", initial());
    {
        let tx = conn.transaction().unwrap();
        crate::user_writes::apply(&tx, &r, 1).unwrap();
    }
    assert!(checkpoint(&conn, Source::Ledger, "host").unwrap().is_none());
    assert_eq!(
        conn.query_row("SELECT count(*) FROM delivery", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        0
    );
    let mut committed = initial();
    committed.outbox[0].metadata.notice.delivery_id = "forged".into();
    let tx = conn.transaction().unwrap();
    assert!(crate::user_writes::apply(&tx, &request("invalid", committed), 2).is_err());
}

#[test]
fn signal_numeric_host_decision_survives_tagged_request_round_trip() {
    let mut commit = initial();
    let delivery = commit.outbox[0].metadata.delivery.clone();
    commit.events.push(Event::HostDecision {
        delivery: Box::new(delivery),
    });
    let request = request("host-decision", commit);
    let encoded = serde_json::to_vec(&request).unwrap();
    let decoded: UserRequest = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(decoded, request);
    let conn = cache();
    let prepared = prepare_pending(&conn, &request, &[]).unwrap();
    super::validate(&prepared).unwrap();
}

#[test]
fn signal_outbox_notice_must_follow_its_route_decision() {
    let mut conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(crate::schema::ARCHIVE).unwrap();
    let mut shown_on_desktop = initial();
    shown_on_desktop.outbox[0].metadata.notice.outlets.desktop = true;
    let mut current = initial();
    current.outbox[0].metadata.notice.current = true;
    for (id, commit) in [("outlets", shown_on_desktop), ("current", current)] {
        let tx = conn.transaction().unwrap();
        assert!(crate::user_writes::apply(&tx, &request(id, commit), 1).is_err());
    }
    let tx = conn.transaction().unwrap();
    crate::user_writes::apply(&tx, &request("agreeing", initial()), 1).unwrap();
}
