use super::*;
#[test]
fn request_snapshots_keep_new_messages_and_repeat_a_shared_prefix_once() {
    let bytes=b"TO LLM 2026-01-01T10:00:00\n-------\nSYSTEM instruction\n-------\nUSER first\nLLM RESPONSE 2026-01-01T10:00:01\nASSISTANT answer\nTO LLM 2026-01-01T10:00:02\n-------\nSYSTEM instruction\n-------\nUSER first\n-------\nASSISTANT answer\n-------\nUSER continue\nUSER second line\nLLM RESPONSE 2026-01-01T10:00:03\nASSISTANT more\n";
    let step =
        crate::export::fixture::parse(&mut open(Path::new(HISTORY)).unwrap(), bytes).unwrap();
    assert_eq!(step.output.sessions.len(), 1);
    assert_eq!(
        step.output
            .messages
            .iter()
            .map(|row| row.message.text())
            .collect::<Vec<_>>(),
        [
            "instruction",
            "first",
            "answer",
            "continue\nsecond line",
            "more"
        ]
    );
    assert!(
        step.output
            .messages
            .iter()
            .all(|row| row.message.at.is_none())
    );
    assert_eq!(
        step.output.messages[3].message.parent.as_ref(),
        Some(&step.output.messages[2].message.id)
    );
    assert_eq!(step.output.events[1].data["copied_prefix_messages"], 3);
}
#[test]
fn markdown_is_a_companion_document_with_exact_text() {
    let bytes = b"# aider chat started at 2026-01-01 10:00:00\n\n#### first\n\nanswer\n";
    let step =
        crate::export::fixture::parse(&mut open(Path::new(COMPANION)).unwrap(), bytes).unwrap();
    assert!(step.output.sessions.is_empty());
    assert_eq!(step.output.documents.len(), 1);
    assert_eq!(
        step.output.documents[0].document.blocks[0].text(),
        Some(std::str::from_utf8(bytes).unwrap())
    );
}
#[test]
fn aider_ids_from_content_anchor() {
    let ids = |bytes: &[u8]| -> Vec<String> {
        crate::export::fixture::parse(&mut open(Path::new(HISTORY)).unwrap(), bytes)
            .unwrap()
            .output
            .messages
            .iter()
            .map(|row| {
                assert_eq!(row.id_kind, polylogue_model::IdKind::Content);
                row.message.id.to_string()
            })
            .collect()
    };
    let one = b"TO LLM 2026-01-01T10:00:00\n-------\nSYSTEM instruction\n-------\nUSER first\nLLM RESPONSE 2026-01-01T10:00:01\nASSISTANT answer\n";
    // The same conversation logged later, with a request the first log did
    // not have before it: the session is named by its first request, so
    // that stays, and each message keeps its ID wherever it now sits.
    let two = b"TO LLM 2026-01-01T10:00:00\n-------\nSYSTEM instruction\n-------\nUSER first\nLLM RESPONSE 2026-01-01T10:00:01\nASSISTANT answer\nTO LLM 2026-01-01T10:05:00\n-------\nSYSTEM instruction\n-------\nUSER first\n-------\nASSISTANT answer\n-------\nUSER first\nLLM RESPONSE 2026-01-01T10:05:01\nASSISTANT answer\n";
    let first = ids(one);
    let second = ids(two);
    assert_eq!(first.len(), 3);
    assert_eq!(second[..3], first[..]);
    // A repeated prompt and answer are the next occurrence of the same
    // content, not the first one again.
    assert_eq!(second.len(), 5);
    assert_ne!(second[3], second[1]);
    assert_ne!(second[4], second[2]);
    assert!(second[3].ends_with("-1") && second[1].ends_with("-0"));
}
