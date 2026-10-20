use super::*;
use crate::{Format, tests::conversation};

#[test]
fn document_and_reader_keep_the_selected_view_and_mask_credentials() {
    let messages = conversation();
    let items: Vec<_> = messages.iter().map(Rendered::Message).collect();
    for (view, prompt, assistant, thinking, tools) in [
        (View::Full, true, true, true, true),
        (View::Dialogue, true, true, true, false),
        (View::Prompts, true, false, false, false),
        (View::Actions, false, false, false, true),
    ] {
        let mut document = Vec::new();
        crate::render(&items, Format::Html, view, &mut document).unwrap();
        for body in [fragment(&items, view), String::from_utf8(document).unwrap()] {
            assert_eq!(body.contains("run the tests"), prompt);
            assert_eq!(body.contains("Running them."), assistant);
            assert_eq!(body.contains("<summary>thinking</summary>"), thinking);
            assert_eq!(body.contains("test failed"), tools);
            assert_eq!(body.contains("cargo test"), tools);
            assert!(!body.contains("sk-ant-api03-AAAA"));
        }
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
