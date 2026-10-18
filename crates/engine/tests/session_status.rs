use polylogue_engine::session_status::SessionStatus;

#[test]
fn known_text_repairs_are_advisory_even_beside_an_independent_incompleteness() {
    let mut status = SessionStatus::default();
    status.text_repairs = 2;
    assert!(status.complete());
    assert_eq!(status.notes().len(), 1);
    assert!(status.incomplete_reasons().is_empty());
    status.record_errors = 1;
    assert!(!status.complete());
    assert_eq!(status.notes().len(), 2);
    let missing = status.incomplete_reasons();
    assert_eq!(missing.len(), 1);
    assert!(missing[0].contains("undecodable complete records"));
}
