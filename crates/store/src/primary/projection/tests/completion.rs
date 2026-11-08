//! Completion observes only actual completed source changes, in owner position units.
use super::*;
fn fact(position: i64, event: &str, product: Product) -> Confirmed {
    Confirmed {
        position,
        event: event.into(),
        output_index: 0,
        native: None,
        change: ConfirmedChange::Live(Box::new(product)),
    }
}
fn lifecycle(position: i64, event: &str, change: ConfirmedChange) -> Confirmed {
    Confirmed {
        position,
        event: event.into(),
        output_index: 0,
        native: None,
        change,
    }
}
#[test]
fn native_authored_replacement_and_query_recovery_use_only_completed_owner_positions() {
    let directory = tempfile::tempdir().unwrap();
    let (archive, _lock) = crate::Archive::open_primary(
        &directory.path().join("cache"),
        &directory.path().join("bytes"),
    )
    .unwrap();
    let mut index = Connection::open(archive.index_db()).unwrap();
    index
        .execute_batch("DROP INDEX primary_fact_changed")
        .unwrap();
    crate::schema::ensure_primary_index(&index).unwrap();
    let resolver = crate::writer::resolve::Resolver::new(&archive, &index, None).unwrap();
    crate::primary::scope::reset(&mut index, &archive, &resolver, "scope-a").unwrap();
    let a = setting(&index, "request-a", "drive.ai-studio");
    apply(
        &mut index,
        &resolver,
        &[fact(10, "request-a", a), barrier(20)],
    )
    .unwrap();
    let reads = crate::ReadPool::open(&archive).unwrap();
    let before = reads.reader().unwrap().change_position().unwrap();
    assert_eq!(before.archive, 20);
    assert_ne!(
        before.index, before.archive,
        "the source cursor is not the index transaction counter"
    );
    let b = setting(&index, "request-b", "browser.actions");
    apply(
        &mut index,
        &resolver,
        &[fact(21, "request-b", b), barrier(21)],
    )
    .unwrap();
    assert!(
        !reads
            .reader()
            .unwrap()
            .completion_authored_replaced(Some(&before))
            .unwrap()
    );
    assert_eq!(
        reads
            .reader()
            .unwrap()
            .completion_setting_keys(Some(before.archive))
            .unwrap(),
        ["browser.actions"]
    );
    let before = reads.reader().unwrap().change_position().unwrap();
    apply(
        &mut index,
        &resolver,
        &[lifecycle(30, "request-a", ConfirmedChange::Archived)],
    )
    .unwrap();
    assert!(
        !reads
            .reader()
            .unwrap()
            .completion_authored_replaced(Some(&before))
            .unwrap(),
        "staging is not publication"
    );
    apply(&mut index, &resolver, &[barrier(30)]).unwrap();
    assert!(
        reads
            .reader()
            .unwrap()
            .completion_authored_replaced(Some(&before))
            .unwrap()
    );
    assert_eq!(
        reads
            .reader()
            .unwrap()
            .completion_setting_keys(None)
            .unwrap(),
        ["browser.actions"]
    );
    let before = reads.reader().unwrap().change_position().unwrap();
    apply(
        &mut index,
        &resolver,
        &[
            lifecycle(40, "request-a", ConfirmedChange::Restored),
            barrier(40),
        ],
    )
    .unwrap();
    assert!(
        reads
            .reader()
            .unwrap()
            .completion_authored_replaced(Some(&before))
            .unwrap()
    );
    assert_eq!(
        reads
            .reader()
            .unwrap()
            .completion_setting_keys(None)
            .unwrap(),
        ["browser.actions", "drive.ai-studio"]
    );
    let before = reads.reader().unwrap().change_position().unwrap();
    apply(
        &mut index,
        &resolver,
        &[
            lifecycle(45, "request-a", ConfirmedChange::Tombstoned),
            barrier(45),
        ],
    )
    .unwrap();
    assert!(
        reads
            .reader()
            .unwrap()
            .completion_authored_replaced(Some(&before))
            .unwrap(),
        "tombstone must retain category identity without its erased body"
    );
    assert_eq!(
        reads
            .reader()
            .unwrap()
            .completion_setting_keys(None)
            .unwrap(),
        ["browser.actions"]
    );
    let object =
        polylogue_query::plan::prepare("sessions", &polylogue_query::Bindings::default(), |_| {
            Ok(())
        })
        .unwrap()
        .query_object()
        .unwrap();
    let id = object.id.to_string();
    let before = reads.reader().unwrap().change_position().unwrap();
    apply(
        &mut index,
        &resolver,
        &[
            fact(
                50,
                "query",
                Product::Query {
                    object: Box::new(object),
                },
            ),
            barrier(60),
        ],
    )
    .unwrap();
    assert_eq!(
        index
            .query_row(
                "SELECT commit_seq FROM source_query WHERE hash=?1",
                [&id],
                |row| row.get::<_, i64>(0)
            )
            .unwrap(),
        50
    );
    assert!(
        reads
            .reader()
            .unwrap()
            .completion_changed_queries(before.archive)
            .unwrap()
            .contains(&id)
    );
    let before = reads.reader().unwrap().change_position().unwrap();
    assert_eq!(before.archive, 60);
    apply(
        &mut index,
        &resolver,
        &[
            lifecycle(70, "query", ConfirmedChange::Archived),
            barrier(70),
        ],
    )
    .unwrap();
    assert!(
        reads
            .reader()
            .unwrap()
            .completion_changed_queries(before.archive)
            .unwrap()
            .contains(&id)
    );
    let before = reads.reader().unwrap().change_position().unwrap();
    apply(
        &mut index,
        &resolver,
        &[
            lifecycle(80, "query", ConfirmedChange::Restored),
            barrier(80),
        ],
    )
    .unwrap();
    assert!(
        reads
            .reader()
            .unwrap()
            .completion_changed_queries(before.archive)
            .unwrap()
            .contains(&id)
    );
    assert_eq!(
        index
            .query_row(
                "SELECT commit_seq FROM source_query WHERE hash=?1",
                [&id],
                |row| row.get::<_, i64>(0)
            )
            .unwrap(),
        50,
        "restoration does not replace original admission"
    );
    let before = reads.reader().unwrap().change_position().unwrap();
    crate::primary::scope::reset(&mut index, &archive, &resolver, "scope-b").unwrap();
    let reader = reads.reader().unwrap();
    assert_ne!(
        reader.change_position().unwrap().generation,
        before.generation
    );
    assert!(reader.completion_authored_replaced(Some(&before)).is_err());
    assert!(reader.completion_setting_keys(None).unwrap().is_empty());
}
