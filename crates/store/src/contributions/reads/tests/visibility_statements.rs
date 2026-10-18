use super::*;
use rusqlite::hooks::{AuthAction, AuthContext, Authorization};
use std::sync::atomic::{AtomicUsize, Ordering};

fn meter(connection: &Connection) -> (Arc<AtomicUsize>, Arc<AtomicUsize>) {
    connection.flush_prepared_statement_cache();
    let compiled = Arc::new(AtomicUsize::new(0));
    let instructions = Arc::new(AtomicUsize::new(0));
    let count = Arc::clone(&compiled);
    connection
        .authorizer(Some(move |context: AuthContext<'_>| {
            if matches!(context.action, AuthAction::Select) {
                count.fetch_add(1, Ordering::Relaxed);
            }
            Authorization::Allow
        }))
        .unwrap();
    let count = Arc::clone(&instructions);
    connection
        .progress_handler(
            1,
            Some(move || {
                count.fetch_add(1, Ordering::Relaxed);
                false
            }),
        )
        .unwrap();
    (compiled, instructions)
}

#[test]
fn visibility_statements_reuse_compilation_not_results_or_execution_work() {
    let mut f = Fixture::new();
    f.ready("a", 0, 1, "model", false, &[]);
    let reader = f.writer.read_handle().unwrap();
    let catalog = reader.session_catalog(65536).unwrap();
    let view = catalog.session(&f.session, 65536).unwrap().unwrap();
    let SessionStep::Ready { message, .. } = view.next(None, 65536).unwrap() else {
        panic!("ready")
    };
    let read = locked(&catalog.read).unwrap();
    let (index_compiled, index_work) = meter(&read.visibility);
    let (archive_compiled, archive_work) = meter(read.archive_visibility.as_ref().unwrap());
    let writes = (
        read.visibility.total_changes(),
        read.archive_visibility.as_ref().unwrap().total_changes(),
    );
    drop(read);
    let check = || {
        catalog.visible().unwrap();
        view.visible().unwrap();
        message.visible().unwrap();
    };
    check();
    let compiled = (
        index_compiled.load(Ordering::Relaxed),
        archive_compiled.load(Ordering::Relaxed),
    );
    assert!(compiled.0 > 0 && compiled.1 > 0);
    // The same four SQL templates must still execute on every call. VM work
    // counts distinguish statement reuse from skipping current authority reads.
    let work = (
        index_work.load(Ordering::Relaxed),
        archive_work.load(Ordering::Relaxed),
    );
    assert!(work.0 > 0 && work.1 > 0);
    for round in 2..=21 {
        check();
        assert_eq!(
            (
                index_compiled.load(Ordering::Relaxed),
                archive_compiled.load(Ordering::Relaxed)
            ),
            compiled
        );
        assert_eq!(
            (
                index_work.load(Ordering::Relaxed),
                archive_work.load(Ordering::Relaxed)
            ),
            (work.0 * round, work.1 * round)
        );
    }
    let read = locked(&catalog.read).unwrap();
    assert_eq!(
        (
            read.visibility.total_changes(),
            read.archive_visibility.as_ref().unwrap().total_changes()
        ),
        writes
    );
    read.visibility.flush_prepared_statement_cache();
    read.archive_visibility
        .as_ref()
        .unwrap()
        .flush_prepared_statement_cache();
    drop(read);
    check();
    assert!(index_compiled.load(Ordering::Relaxed) > compiled.0);
    assert!(archive_compiled.load(Ordering::Relaxed) > compiled.1);
}

#[test]
fn warm_visibility_statements_observe_post_pin_revocation() {
    for change in ["hidden", "epoch", "generation", "excision"] {
        let mut f = Fixture::new();
        f.ready("a", 0, 1, "model", false, &[]);
        let reader = f.writer.read_handle().unwrap();
        let catalog = reader.session_catalog(65536).unwrap();
        let view = catalog.session(&f.session, 65536).unwrap().unwrap();
        let SessionStep::Ready { message, .. } = view.next(None, 65536).unwrap() else {
            panic!("ready")
        };
        catalog.visible().unwrap();
        view.visible().unwrap();
        message.visible().unwrap();
        match change {
            "hidden" => {
                f.writer
                    ._index
                    .execute("UPDATE message SET hidden=1", [])
                    .unwrap();
            }
            "epoch" => {
                f.writer
                    ._index
                    .execute("UPDATE message SET removal_epoch=removal_epoch+1", [])
                    .unwrap();
            }
            "generation" => {
                f.writer
                    ._index
                    .execute(
                        "UPDATE meta SET value='different' WHERE key='generation'",
                        [],
                    )
                    .unwrap();
            }
            "excision" => {
                f.writer.archive.execute("INSERT INTO excision(id,scope,created_at,local_seq) VALUES('synthetic','session:synthetic',0,1)", []).unwrap();
            }
            _ => unreachable!(),
        }
        if change == "excision" {
            assert!(matches!(
                catalog.visible(),
                Err(Error::ExcisionCheckRequired)
            ));
            assert!(matches!(view.visible(), Err(Error::ExcisionCheckRequired)));
        } else {
            assert!(matches!(
                message.visible(),
                Err(Error::MessagePredecessorChanged)
            ));
            if change == "generation" {
                assert!(matches!(
                    catalog.visible(),
                    Err(Error::MessagePredecessorChanged)
                ));
                assert!(matches!(
                    view.visible(),
                    Err(Error::MessagePredecessorChanged)
                ));
            }
        }
    }
}
