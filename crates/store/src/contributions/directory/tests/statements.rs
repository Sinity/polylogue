use super::*;
use rusqlite::hooks::{AuthAction, AuthContext, Authorization};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

#[test]
fn repeated_directory_walks_reuse_compilation_but_keep_exact_results_and_work() {
    let (_dir, mut writer, owner, body) = fixture();
    let path = writer.root.join("index.db");
    let mut root = None;
    let mut expected = Vec::new();
    for (ordinal, key) in [0, 17, 4096, u64::MAX].into_iter().enumerate() {
        let tx = writer._index.transaction().unwrap();
        let edge = leaf(&tx, ordinal as u64);
        root = Some(install(&tx, &path, owner, body, root, key, edge).0);
        tx.commit().unwrap();
        expected.push((key, edge));
    }
    writer._index.flush_prepared_statement_cache();
    let compilations = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&compilations);
    writer
        ._index
        .authorizer(Some(move |context: AuthContext<'_>| {
            if matches!(
                context.action,
                AuthAction::Read {
                    table_name: "message_body_node" | "message_body_edge",
                    ..
                }
            ) {
                observed.fetch_add(1, Ordering::Relaxed);
            }
            Authorization::Allow
        }))
        .unwrap();
    let writes = writer._index.total_changes();
    let mut baseline = Vec::new();
    for &(key, edge) in &expected {
        let mut work = Work::default();
        assert_eq!(
            lookup(
                &writer._index,
                &path,
                owner,
                Kind::Blocks,
                root,
                key,
                &mut work
            )
            .unwrap(),
            Some(edge)
        );
        assert_eq!(work.nodes_read, 16);
        assert_eq!(work.nodes_written, 0);
        baseline.push(work);
    }
    // The authorizer runs at SQLite statement compilation, not each execution.
    // Both templates were used; further complete walks must not recompile them.
    let warm = compilations.load(Ordering::Relaxed);
    assert!(warm > 0);
    for _ in 0..20 {
        for (&(key, edge), expected_work) in expected.iter().zip(&baseline) {
            let mut work = Work::default();
            assert_eq!(
                lookup(
                    &writer._index,
                    &path,
                    owner,
                    Kind::Blocks,
                    root,
                    key,
                    &mut work
                )
                .unwrap(),
                Some(edge)
            );
            assert_eq!(&work, expected_work);
        }
    }
    assert_eq!(compilations.load(Ordering::Relaxed), warm);
    assert_eq!(writer._index.total_changes(), writes);
    // A cold cache must compile again: this distinguishes a vacuous counter.
    writer._index.flush_prepared_statement_cache();
    lookup(
        &writer._index,
        &path,
        owner,
        Kind::Blocks,
        root,
        0,
        &mut Work::default(),
    )
    .unwrap();
    assert!(compilations.load(Ordering::Relaxed) > warm);
}

#[test]
fn cached_directory_statements_reread_changed_rows_and_keep_all_shape_guards() {
    for change in [
        "UPDATE message_body_node SET ordinary_count=ordinary_count+1 WHERE pk=?1",
        "UPDATE message_body_node SET height=14 WHERE pk=?1",
        "UPDATE message_body_node SET state='building' WHERE pk=?1",
        "DELETE FROM message_body_edge WHERE parent_pk=?1",
        "UPDATE message_body_edge SET child_height=0 WHERE parent_pk=?1",
    ] {
        let (_dir, mut writer, owner, body) = fixture();
        let path = writer.root.join("index.db");
        let tx = writer._index.transaction().unwrap();
        let edge = leaf(&tx, 0);
        let root = install(&tx, &path, owner, body, None, 0, edge).0;
        tx.commit().unwrap();
        assert_eq!(
            lookup(
                &writer._index,
                &path,
                owner,
                Kind::Blocks,
                Some(root),
                0,
                &mut Work::default()
            )
            .unwrap(),
            Some(edge)
        );
        assert!(
            lookup(
                &writer._index,
                &path,
                Owner {
                    message: owner.message + 1,
                    ..owner
                },
                Kind::Blocks,
                Some(root),
                0,
                &mut Work::default()
            )
            .is_err()
        );
        assert!(
            lookup(
                &writer._index,
                &path,
                owner,
                Kind::Fields,
                Some(root),
                0,
                &mut Work::default()
            )
            .is_err()
        );
        // Deliberate at-rest corruption bypasses FK and CHECK shape protection;
        // the cached SELECTs must still fetch and reject the changed rows.
        writer
            ._index
            .execute_batch("PRAGMA foreign_keys=OFF; PRAGMA ignore_check_constraints=ON")
            .unwrap();
        writer._index.execute(change, [root]).unwrap();
        assert!(
            matches!(
                lookup(
                    &writer._index,
                    &path,
                    owner,
                    Kind::Blocks,
                    Some(root),
                    0,
                    &mut Work::default()
                ),
                Err(Error::Invariant(_))
            ),
            "{change}"
        );
    }
}
