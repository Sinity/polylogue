use super::*;
use polylogue_model::{
    Author, BlobHash, Timestamp,
    historical::{Intent, OriginalBytes, SourceSchema},
};

fn record() -> HistoricalAcceptedRequest {
    let intent = b"exact predecessor semantic identity".to_vec();
    let outcome = UserOutcome {
        id: "old-fact".into(),
        repeat: false,
        pending: false,
    };
    HistoricalAcceptedRequest {
        hash_evidence: polylogue_model::historical::AcceptedHashEvidence::Observed,
        native: Default::default(),
        id: "old-request".into(),
        source: SourceSchema::PolylogueArchive { version: 4 },
        content_hash: BlobHash::of(&intent),
        original: Some(OriginalBytes {
            cached: vec![],
            request: Some(
                br#"{"id":"old-request","write":{"write":"withdraw","note":"old-fact"}}"#.to_vec(),
            ),
            intent: Some(intent),
            outcome: Some(serde_json::to_vec(&outcome).unwrap()),
            effect: None,
            records: Vec::new(),
            imports: None,
        }),
        intent: Some(Intent::Withdrawal {
            note: "old-fact".into(),
            reason: None,
        }),
        machine: None,
        at: None,
        authorship: None,
        outcome: Some(outcome),
        conserved_effects: None,
        facts: vec![],
        imports: Default::default(),
    }
}
fn ingress(at: i64) -> ImportAttribution {
    ImportAttribution {
        machine: "receiver".into(),
        input: "synthetic-bundle".into(),
        chain: 1,
        actor: Author::Operator,
        at: Timestamp::from_unix_ms(at).unwrap(),
    }
}
fn archive() -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(crate::schema::ARCHIVE).unwrap();

    conn
}

#[test]
fn historical_restore_retains_original_outcome_bytes_and_unknown_original_metadata() {
    let mut conn = archive();
    let original = record();
    let tx = conn.transaction().unwrap();
    restore(&tx, &original, &[], &ingress(20), 7).unwrap();
    tx.commit().unwrap();
    let mut expected = original.clone();
    expected.imports.merge(&[], &ingress(20));
    assert_eq!(get(&conn, &original.id).unwrap(), Some(expected.clone()));
    let at: Option<i64> = conn
        .query_row("SELECT created_at FROM client_request", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(at, None);
    let tx = conn.transaction().unwrap();
    restore(&tx, &original, &[], &ingress(10), 99).unwrap();
    tx.commit().unwrap();
    assert_eq!(get(&conn, &original.id).unwrap(), Some(expected));
    let counter: i64 = conn
        .query_row("SELECT commit_seq FROM client_request", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(counter, 7);
    let mut native = Vec::new();
    super::super::visit(&conn, &mut |prepared, _| {
        native.push(prepared.clone());
        Ok(())
    })
    .unwrap();
    assert!(native.is_empty());
    let mut historical = Vec::new();
    visit(&conn, &mut |record| {
        historical.push(record);
        Ok(())
    })
    .unwrap();
    assert_eq!(historical.len(), 1);
    assert_eq!(historical[0].original, original.original);
    assert_eq!(historical[0].outcome, original.outcome);
    let scratch = archive();
    super::super::records::copy(&conn, &scratch).unwrap();
    assert_eq!(
        get(&scratch, &original.id).unwrap(),
        get(&conn, &original.id).unwrap()
    );
    let copied_counter: i64 = scratch
        .query_row("SELECT commit_seq FROM client_request", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(copied_counter, 0);
}

#[test]
fn historical_identity_cannot_be_reaccepted_or_replaced_with_different_evidence() {
    let mut conn = archive();
    let original = record();
    let tx = conn.transaction().unwrap();
    restore(&tx, &original, &[], &ingress(1), 1).unwrap();
    tx.commit().unwrap();
    let native = UserRequest {
        id: original.id.clone(),
        machine: "receiver".into(),
        at: ingress(1).at,
        authorship: polylogue_model::authorship::Authorship {
            author: Author::Operator,
            via: polylogue_model::authorship::Channel::Cli,
        },
        write: polylogue_model::user::UserWrite::WithdrawAssertions {
            targets: vec![],
            reason: None,
        },
    };
    assert!(matches!(
        super::super::known_request(&conn, &native),
        Err(StoreError::Conflict(_))
    ));
    let mut changed = original.clone();
    changed.machine = Some("invented-origin".into());
    let tx = conn.transaction().unwrap();
    assert!(matches!(
        restore(&tx, &changed, &[], &ingress(2), 2),
        Err(StoreError::Conflict(_))
    ));
    tx.rollback().unwrap();
    let stored = get(&conn, &original.id).unwrap().unwrap();
    assert_eq!(stored.machine, None);
    assert_eq!(stored.authorship, None);
    assert_eq!(stored.imports.observed, [ingress(1)]);
}
