use polylogue_engine::session_qualification::Qualification;

#[test]
fn known_text_repairs_are_advisory_even_beside_an_independent_incompleteness() {
    let mut qualification = Qualification::default();
    qualification.text_repairs = 2;
    assert!(qualification.complete());
    assert_eq!(qualification.notes().len(), 1);
    assert!(qualification.incomplete_reasons().is_empty());
    qualification.record_errors = 1;
    assert!(!qualification.complete());
    assert_eq!(qualification.notes().len(), 2);
    let missing = qualification.incomplete_reasons();
    assert_eq!(missing.len(), 1);
    assert!(missing[0].contains("undecodable complete records"));
}
