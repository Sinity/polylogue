use super::*;

fn setting(id: &str) -> polylogue_model::user::UserRequest {
    crate::fixture::authored::operator_request(
        id.into(),
        polylogue_model::user::UserWrite::SetSetting {
            key: "browser.actions".into(),
            value: "on".into(),
            machine: None,
        },
    )
}

#[test]
fn queued_read_action_rejects_an_obsolete_generation_without_committing() {
    let archive = test_archive();
    let token = archive.archive.scope_token().unwrap();
    token.check().unwrap();
    archive.archive.read_scope.begin_reset().unwrap();
    archive.archive.read_scope.finish_reset();
    assert!(matches!(
        archive
            .writer
            .archive()
            .user(crate::writer::UserSubmission {
                request: setting("stale"),
                scope: Some(token),
                preparation: None,
                parent: None
            }),
        Err(StoreError::ScopeChanged)
    ));
    let source = Connection::open(archive.archive.archive_db()).unwrap();
    let count: i64 = source
        .query_row(
            "SELECT count(*) FROM client_request WHERE id='stale'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(count, 0);
    archive
        .writer
        .archive()
        .user(crate::writer::UserSubmission {
            request: setting("current"),
            scope: Some(archive.archive.scope_token().unwrap()),
            preparation: None,
            parent: None,
        })
        .unwrap();
    let count: i64 = source
        .query_row(
            "SELECT count(*) FROM client_request WHERE id='current'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(count, 1);
}

#[test]
fn a_current_token_from_another_archive_cannot_authorize_an_action() {
    let target = test_archive();
    let other = test_archive();
    assert!(matches!(
        target.writer.archive().user(crate::writer::UserSubmission {
            request: setting("foreign"),
            scope: Some(other.archive.scope_token().unwrap()),
            preparation: None,
            parent: None
        }),
        Err(StoreError::ScopeChanged)
    ));
    let source = Connection::open(target.archive.archive_db()).unwrap();
    let count: i64 = source
        .query_row("SELECT count(*) FROM client_request", [], |row| row.get(0))
        .unwrap();
    assert_eq!(count, 0);
}

#[test]
fn a_full_user_queue_returns_the_entire_submission() {
    let archive = test_archive();
    let sender = archive.writer.sender();
    let pause = sender.pause_user_dispatch_for_test("hold-full-queue", DispatchPhase::Owned);
    let held = sender.clone();
    let holding = std::thread::spawn(move || held.archive().user(setting("hold-full-queue")));
    pause.wait().unwrap();
    let mut queued = Vec::new();
    for index in 0..crate::archive::writer::USER_QUEUE {
        match sender
            .archive()
            .try_submit(setting(&format!("queued-{index}")))
            .unwrap()
        {
            Queued::Committing(result) => queued.push(result),
            Queued::Full(_) => panic!("queue filled before its capacity"),
        }
    }
    let scope = archive.archive.scope_token().unwrap();
    let parent = ParentLifetime::capture(None, crate::cancellation::Cancellation::default());
    let request = setting("returned-submission");
    let submission = UserSubmission {
        request: request.clone(),
        scope: Some(scope.clone()),
        preparation: Some(crate::user_writes::PreparationContext::new(vec![
            "codex".into(),
        ])),
        parent: Some(Arc::clone(&parent)),
    };
    let Queued::Full(returned) = sender.archive().try_submit(submission).unwrap() else {
        panic!("a full queue accepted another submission");
    };
    assert_eq!(returned.request, request);
    assert!(returned.scope.as_ref().unwrap().same(&scope));
    assert_eq!(
        returned.preparation.as_ref().unwrap().bindings().origins,
        ["codex"]
    );
    assert!(Arc::ptr_eq(returned.parent.as_ref().unwrap(), &parent));
    pause.release();
    holding.join().unwrap().unwrap();
    for result in queued {
        result.recv().unwrap().unwrap();
    }
    sender.archive().user(*returned).unwrap();
}
