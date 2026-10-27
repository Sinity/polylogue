use super::*;
fn conversation() -> Vec<polylogue_model::Message> {
    let mut messages = crate::tests::conversation();
    let mut masker = polylogue_model::masking::MessageMasker::default();
    for message in &mut messages {
        masker.message(message);
    }
    messages
}

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
        crate::html::document(&items, view, &mut document).unwrap();
        for body in [content(&items, view), String::from_utf8(document).unwrap()] {
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
    let body = content(&items, View::Actions);
    assert!(body.contains("error (exit 101)"));
    assert!(body.contains("cargo test"));
    assert_eq!(body.matches("test failed").count(), 1);
    assert!(body.contains("<details class=\"tool\">"));
    assert!(!body.contains("Running them."));
}

#[test]
fn paged_calls_show_the_recorded_outcome_name_and_command_preview() {
    let slice = crate::model_pages::BlockSlice {
        message: None,
        block: 0,
        kind: "tool_call".into(),
        label: Some("shell <tool>".into()),
        outcome: Some(polylogue_model::CallOutcome::Error),
        text: "printf '<value>'\nsecond line".into(),
        from: 0,
        until: "printf '<value>'\nsecond line".chars().count(),
        more: true,
    };
    let mut out = Vec::new();
    page_slices(&[slice], &mut out).unwrap();
    let body = String::from_utf8(out).unwrap();
    let summary = body.split("</summary>").next().unwrap();
    assert!(summary.contains("outcome error\">error</span> <b>shell &lt;tool&gt;</b> <code>printf &#39;&lt;value&gt;&#39;</code>"));
    assert!(!summary.contains("second line"));
    assert!(body.contains("<pre>printf &#39;&lt;value&gt;&#39;\nsecond line</pre>"));
    assert!(body.contains("Continues on the next page"));
}

#[test]
fn a_page_starting_inside_a_block_says_so() {
    let mut slice = crate::model_pages::BlockSlice {
        message: None,
        block: 0,
        kind: "text".into(),
        label: None,
        outcome: None,
        text: "tail".into(),
        from: 8,
        until: 12,
        more: false,
    };
    let mut out = Vec::new();
    page_slices(std::slice::from_ref(&slice), &mut out).unwrap();
    assert!(
        String::from_utf8(out)
            .unwrap()
            .contains("starts within a source block")
    );
    slice.from = 0;
    let mut out = Vec::new();
    page_slices(&[slice], &mut out).unwrap();
    assert!(
        !String::from_utf8(out)
            .unwrap()
            .contains("starts within a source block")
    );
}

#[test]
fn attribute_text_and_ids_are_encoded_as_text_not_markup() {
    assert_eq!(escape("<&\"'"), "&lt;&amp;&quot;&#39;");
    assert_eq!(
        encode_path_segment("codex:a/b?c#dé"),
        "codex%3Aa%2Fb%3Fc%23d%C3%A9"
    );
}

fn content(items: &[Rendered<'_>], view: View) -> String {
    let mut out = Vec::new();
    fragment(items, view, &mut out).unwrap();
    String::from_utf8(out).unwrap()
}

#[test]
fn write_failures_are_returned() {
    struct Broken;
    impl std::io::Write for Broken {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            Err(std::io::ErrorKind::BrokenPipe.into())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let messages = conversation();
    let items: Vec<_> = messages.iter().map(Rendered::Message).collect();
    assert_eq!(
        fragment(&items, View::Full, &mut Broken)
            .unwrap_err()
            .kind(),
        std::io::ErrorKind::BrokenPipe
    );
}
