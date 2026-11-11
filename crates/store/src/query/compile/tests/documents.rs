use super::*;
#[test]
fn documents_search_their_own_content_and_refuse_unproved_session_scope() {
    let program = c("documents where purpose:memory authorship:unknown \"supplied words\"");
    assert_eq!(program.unit, Unit::Documents);
    assert!(statements(&program).contains("document_block"));
    assert!(!statements(&program).contains("written_by"));
    let error = compile("sessions | documents where purpose:context", &opts(&[])).unwrap_err();
    assert_eq!(error.kind(), "invalid_query");
    assert!(
        error
            .to_string()
            .contains("no implicit session association")
    );
}
