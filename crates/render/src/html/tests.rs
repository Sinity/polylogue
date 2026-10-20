use super::*;
use crate::{Format, tests::conversation};

#[test]
fn exported_document_uses_the_same_content_as_the_reader() {
    let messages = conversation();
    let items: Vec<_> = messages.iter().map(Rendered::Message).collect();
    for view in [View::Full, View::Dialogue, View::Prompts, View::Actions] {
        let body = fragment(&items, view);
        let mut document = Vec::new();
        crate::render(&items, Format::Html, view, &mut document).unwrap();
        let document = String::from_utf8(document).unwrap();
        assert!(document.contains(&body));
        assert!(!body.contains("sk-ant-api03-AAAA"));
    }
}

#[test]
fn calls_show_their_outcome_and_result_once() {
    let messages = conversation();
    let items: Vec<_> = messages.iter().map(Rendered::Message).collect();
    let body = fragment(&items, View::Actions);
    assert!(body.contains("error (exit 101)"));
    assert!(body.contains("cargo test"));
    assert_eq!(body.matches("test failed").count(), 1);
    assert!(body.contains("<details class=\"tool\">"));
    assert!(!body.contains("Running them."));
}

#[test]
fn attribute_text_and_ids_are_encoded_without_interpreting_markup() {
    assert_eq!(escape("<&\"'"), "&lt;&amp;&quot;&#39;");
    assert_eq!(
        encode_path_segment("codex:a/b?c#dé"),
        "codex%3Aa%2Fb%3Fc%23d%C3%A9"
    );
}
