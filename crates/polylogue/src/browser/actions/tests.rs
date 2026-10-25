use super::*;

fn request(conversation: Option<&str>, submit: bool) -> ChatRequest {
    ChatRequest {
        provider: "chatgpt".into(),
        conversation: conversation.map(str::to_owned),
        project: None,
        text: "summarize this".into(),
        model: "X".into(),
        effort: None,
        submit,
    }
}

#[test]
fn a_request_is_checked_before_it_is_stored() {
    let new = new_action("k1", &request(None, false), vec![]).unwrap();
    assert!(new.stage_only && !new.needs_approval);
    assert_eq!(new.request_key, "k1");
    let held = new_action("k2", &request(Some("c1"), true), vec![]).unwrap();
    assert!(held.needs_approval && !held.stage_only);
    let started = new_action("k3", &request(None, true), vec![]).unwrap();
    assert!(!started.needs_approval);

    let mut other = request(None, false);
    other.provider = "grok".into();
    assert_eq!(
        new_action("k", &other, vec![]).unwrap_err(),
        Refused(400, "unsupported_provider")
    );
    let mut blank = request(None, false);
    blank.model = " ".into();
    assert_eq!(
        new_action("k", &blank, vec![]).unwrap_err(),
        Refused(400, "model_required")
    );

    let file = ChatFile {
        name: "../escape".into(),
        media_type: "text/plain".into(),
        bytes: b"bytes".to_vec(),
    };
    let blob = BlobHash::of(&file.bytes);
    assert_eq!(
        new_action("k", &request(None, false), vec![attachment(&file, blob)]).unwrap_err(),
        Refused(400, "attachment_name_not_a_basename")
    );
    let file = ChatFile {
        name: "notes.txt".into(),
        ..file
    };
    let attached = new_action("k", &request(None, false), vec![attachment(&file, blob)]).unwrap();
    assert_eq!(attached.attachments[0].size, 5);
    // The same request with other files is other content under the same key.
    assert_ne!(
        attached.request_hash,
        new_action("k", &request(None, false), vec![])
            .unwrap()
            .request_hash
    );
}
