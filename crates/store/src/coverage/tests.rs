use super::*;
use polylogue_model::{
    Timestamp,
    capture::{Accessibility, RootCoverage},
};
#[test]
fn completed_snapshots_replace_all_roots_without_treating_source_time_as_authority() {
    let db = Connection::open_in_memory().unwrap();
    db.execute_batch(SCHEMA).unwrap();
    let mut fact = Coverage {
        version: 1,
        machine: "synthetic".into(),
        observed_at: Timestamp::from_unix_ms(2000).unwrap(),
        roots: vec![RootCoverage {
            path_bytes: b"/tool".to_vec(),
            status: Accessibility::Accessible,
            entries: Vec::new(),
            unavailable_paths: Vec::new(),
        }],
    };
    apply(&db, &fact).unwrap();
    fact.observed_at = Timestamp::from_unix_ms(1000).unwrap();
    fact.roots[0].status = Accessibility::Unavailable;
    fact.roots[0].unavailable_paths.push(b"/tool".to_vec());
    apply(&db, &fact).unwrap();
    let payload: String = db
        .query_row("SELECT payload_json FROM capture_coverage", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert_eq!(serde_json::from_str::<Coverage>(&payload).unwrap(), fact);
    fact.roots.push(fact.roots[0].clone());
    assert!(apply(&db, &fact).is_err());
    clear(&db).unwrap();
    assert_eq!(
        db.query_row("SELECT count(*) FROM capture_coverage", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        0
    );
}
