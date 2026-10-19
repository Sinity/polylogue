use super::*;

#[test]
fn an_imported_file_keeps_the_origin_whose_layout_it_sits_in() {
    // Claude Code accepts any JSON-lines file on import; a Grok CLI history
    // copied with its directories still reads as Grok CLI.
    let head = br#"{"type":"system","content":"You are a CLI."}"#;
    let grok = Path::new("/backup/.grok/sessions/%2Fhome/01a0/chat_history.jsonl");
    assert_eq!(recognize_import(grok, head).unwrap().name, "grok-cli");
    let loose = Path::new("/backup/s1.jsonl");
    assert_eq!(recognize_import(loose, head).unwrap().name, "claude-code");
}
