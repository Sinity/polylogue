use super::*;

struct EmptyDocument;

impl DocumentParser for EmptyDocument {
    fn document(&self, _: &[u8], _: &mut Output, _: &mut Accounting) {}
}

#[test]
fn document_parser_requires_the_whole_input() {
    let parser = Parse::Document(Box::new(EmptyDocument));
    assert!(parser.is_document());
    assert!(matches!(
        parser.step(None, b"{}", 12),
        Err(ParseError::NotWhole(12))
    ));
    let (step, carry) = parser.step(None, b"{}", 0).unwrap();
    assert_eq!(step.consumed, 2);
    assert_eq!(carry, Json::Null);
}
