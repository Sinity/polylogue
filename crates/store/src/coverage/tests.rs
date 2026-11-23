use super::*;
use polylogue_model::{
    Timestamp,
    capture::{Accessibility, RootCoverage},
};
#[test]
fn completed_walks_replace_all_roots_regardless_of_source_time() {
    let db = Connection::open_in_memory().unwrap();
    db.execute_batch(SCHEMA).unwrap();
    let root = |path: &[u8], status| RootCoverage {
        path_bytes: path.to_vec(),
        status,
        entries: Vec::new(),
        unavailable_paths: Vec::new(),
    };
    let mut fact = Coverage {
        version: 1,
        machine: "synthetic".into(),
        observed_at: Timestamp::from_unix_ms(2000).unwrap(),
        roots: vec![
            root(b"/tool", Accessibility::Accessible),
            root(b"/other", Accessibility::Accessible),
        ],
    };
    apply(&db, &fact).unwrap();
    // A later walk with an older time and fewer roots replaces the whole
    // payload; nothing of the earlier walk's roots is kept.
    fact.observed_at = Timestamp::from_unix_ms(1000).unwrap();
    fact.roots = vec![root(b"/tool", Accessibility::Unavailable)];
    fact.roots[0].unavailable_paths.push(b"/tool".to_vec());
    apply(&db, &fact).unwrap();
    let stored = || -> Coverage {
        let payload: String = db
            .query_row("SELECT payload_json FROM capture_coverage", [], |r| {
                r.get(0)
            })
            .unwrap();
        serde_json::from_str(&payload).unwrap()
    };
    assert_eq!(stored(), fact);
    let mut rejected = fact.clone();
    rejected.roots.push(rejected.roots[0].clone());
    assert!(matches!(
        apply(&db, &rejected),
        Err(StoreError::Invalid(message)) if message == "coverage roots must be nonempty and unique"
    ));
    assert_eq!(stored(), fact, "a rejected walk leaves the stored one");
    clear(&db).unwrap();
    assert_eq!(
        db.query_row("SELECT count(*) FROM capture_coverage", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        0
    );
}
