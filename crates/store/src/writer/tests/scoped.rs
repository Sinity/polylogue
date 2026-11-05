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
            .sender()
            .unwrap()
            .user_from_read(setting("stale"), token),
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
        .user_from_read(setting("current"), archive.archive.scope_token().unwrap())
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
        target
            .writer
            .user_from_read(setting("foreign"), other.archive.scope_token().unwrap()),
        Err(StoreError::ScopeChanged)
    ));
    let source = Connection::open(target.archive.archive_db()).unwrap();
    let count: i64 = source
        .query_row("SELECT count(*) FROM client_request", [], |row| row.get(0))
        .unwrap();
    assert_eq!(count, 0);
}
