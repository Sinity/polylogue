use super::*;

#[test]
fn a_flag_is_raised_once_and_lowered_idempotently() {
    let dir = tempfile::tempdir().unwrap();
    let session = SessionId::parse("claude-code:0199a1b2-3c4d-7e8f-9012-345678901234").unwrap();
    assert!(!raised(dir.path(), &session));
    raise(dir.path(), &session).unwrap();
    raise(dir.path(), &session).unwrap();
    assert!(raised(dir.path(), &session));
    assert!(
        flag(dir.path(), &session).ends_with("claude-code/0199a1b2-3c4d-7e8f-9012-345678901234")
    );
    lower(dir.path(), &session).unwrap();
    lower(dir.path(), &session).unwrap();
    assert!(!raised(dir.path(), &session));
}
