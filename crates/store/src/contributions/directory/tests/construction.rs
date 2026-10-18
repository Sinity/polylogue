use super::*;
use rusqlite::hooks::{AuthAction, AuthContext, Authorization};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

#[derive(Debug)]
struct Measurement {
    compilations: [usize; 3],
    changes: u64,
    work: Vec<Work>,
    node_rows: i64,
    edge_rows: i64,
}
fn construct(cache: bool) -> Measurement {
    let (_dir, mut writer, owner, body) = fixture();
    if !cache {
        writer._index.set_prepared_statement_cache_capacity(0);
    }
    let counts = Arc::new([
        AtomicUsize::new(0),
        AtomicUsize::new(0),
        AtomicUsize::new(0),
    ]);
    let seen = Arc::clone(&counts);
    writer
        ._index
        .authorizer(Some(move |c: AuthContext<'_>| {
            let slot = match c.action {
                AuthAction::Insert {
                    table_name: "message_body_node",
                } => Some(0),
                AuthAction::Insert {
                    table_name: "message_body_edge",
                } => Some(1),
                AuthAction::Update {
                    table_name: "message_body_node",
                    column_name: "state",
                } => Some(2),
                _ => None,
            };
            if let Some(i) = slot {
                seen[i].fetch_add(1, Ordering::Relaxed);
            }
            Authorization::Allow
        }))
        .unwrap();
    let file = writer.root.join("index.db");
    let before = writer._index.total_changes();
    let mut root = None;
    let mut work = Vec::new();
    for i in 0..4 {
        let tx = writer._index.transaction().unwrap();
        let edge = leaf(&tx, i);
        let (next, w) = install(&tx, &file, owner, body, root, i, edge);
        tx.commit().unwrap();
        root = Some(next);
        work.push(w);
    }
    let changes = writer._index.total_changes() - before;
    let compilations = std::array::from_fn(|i| counts[i].load(Ordering::Relaxed));
    for i in 0..4 {
        assert!(
            lookup(
                &writer._index,
                &file,
                owner,
                Kind::Blocks,
                root,
                i,
                &mut Work::default()
            )
            .unwrap()
            .is_some()
        );
    }
    assert_eq!(
        summary(&writer._index, &file, owner, Kind::Blocks, root)
            .unwrap()
            .ordinary,
        4
    );
    assert_eq!(
        writer
            ._index
            .query_row(
                "SELECT count(*) FROM message_body_node WHERE state!='ready'",
                [],
                |r| r.get::<_, i64>(0)
            )
            .unwrap(),
        0
    );
    Measurement {
        compilations,
        changes,
        work,
        node_rows: writer
            ._index
            .query_row("SELECT count(*) FROM message_body_node", [], |r| r.get(0))
            .unwrap(),
        edge_rows: writer
            ._index
            .query_row("SELECT count(*) FROM message_body_edge", [], |r| r.get(0))
            .unwrap(),
    }
}
#[test]
fn construction_reuses_fixed_write_programs_without_changing_execution_or_work() {
    let uncached = construct(false);
    let cached = construct(true);
    eprintln!("construction uncached={uncached:?}; cached={cached:?}");
    assert_eq!(uncached.changes, cached.changes);
    assert_eq!(uncached.work, cached.work);
    assert_eq!(
        (uncached.node_rows, uncached.edge_rows),
        (cached.node_rows, cached.edge_rows)
    );
    assert_eq!(cached.node_rows, 64);
    assert_eq!(uncached.compilations[0], 64);
    assert_eq!(uncached.compilations[1], cached.edge_rows as usize);
    assert_eq!(cached.compilations, [1, 1, 0]);
}

#[test]
fn complete_ready_node_has_one_insert_and_no_later_state_transition() {
    let (_dir, mut writer, owner, _) = fixture();
    let file = writer.root.join("index.db");
    writer._index.execute_batch("CREATE TRIGGER forbid_node_state_change BEFORE UPDATE OF state ON message_body_node BEGIN SELECT RAISE(ABORT,'directory construction must not transition a ready node'); END;").unwrap();
    let tx = writer._index.transaction().unwrap();
    let edge = leaf(&tx, 0);
    let before = tx.total_changes();
    let mut work = Work::default();
    let (root, counts) = replace(
        &tx,
        &file,
        owner,
        Kind::Blocks,
        None,
        0,
        Some(edge),
        &mut work,
    )
    .unwrap();
    assert_eq!(work.nodes_written, 16);
    assert_eq!(work.edges_written, 16);
    assert_eq!(work.references_changed, 15);
    // 16 node INSERTs +16 edge INSERTs +15 child acquisitions; no16 state UPDATEs.
    assert_eq!(tx.total_changes() - before, 47);
    assert_eq!(counts.ordinary, 1);
    assert!(
        lookup(
            &tx,
            &file,
            owner,
            Kind::Blocks,
            Some(root),
            0,
            &mut Work::default()
        )
        .unwrap()
        .is_some()
    );
    tx.rollback().unwrap();
}
