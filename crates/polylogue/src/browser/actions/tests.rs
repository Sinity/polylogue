use super::*;

fn request(conversation: Option<&str>, submit: bool) -> ChatRequest {
    ChatRequest {
        provider: "chatgpt".into(),
        conversation: conversation.map(str::to_owned),
        project: None,
        text: "summarize this".into(),
        model: Some("X".into()),
        effort: None,
        submit,
    }
}

#[test]
fn a_request_is_checked_and_shaped_into_an_action_row() {
    let new = new_action("k1", &request(None, false), vec![], false).unwrap();
    assert!(new.stage_only && !new.needs_approval);
    assert_eq!(new.request_key, "k1");
    let held = new_action("k2", &request(Some("c1"), true), vec![], true).unwrap();
    assert!(held.needs_approval && !held.stage_only);
    let started = new_action("k3", &request(None, true), vec![], false).unwrap();
    assert!(!started.needs_approval);

    let mut other = request(None, false);
    other.provider = "grok".into();
    assert_eq!(
        new_action("k", &other, vec![], false).unwrap_err(),
        Refused(polylogue_model::Code::UnsupportedProvider)
    );
    // The page keeps its selected model, but an effort is chosen in the
    // model's menu.
    let mut unnamed = request(None, false);
    unnamed.model = None;
    assert!(new_action("k", &unnamed, vec![], false).is_ok());
    unnamed.effort = Some("Standard".into());
    assert_eq!(
        new_action("k", &unnamed, vec![], false).unwrap_err(),
        Refused(polylogue_model::Code::ModelRequired)
    );

    let file = ChatFile {
        name: "../escape".into(),
        media_type: "text/plain".into(),
        bytes: b"bytes".to_vec(),
    };
    let blob = BlobHash::of(&file.bytes);
    assert_eq!(
        new_action(
            "k",
            &request(None, false),
            vec![attachment(&file, blob)],
            false
        )
        .unwrap_err(),
        Refused(polylogue_model::Code::AttachmentNameNotABasename)
    );
    let file = ChatFile {
        name: "notes.txt".into(),
        ..file
    };
    let attached = new_action(
        "k",
        &request(None, false),
        vec![attachment(&file, blob)],
        false,
    )
    .unwrap();
    assert_eq!(attached.attachments[0].size, 5);
    // The same request with other files is other content under the same key.
    assert_ne!(
        attached.request_hash,
        new_action("k", &request(None, false), vec![], false)
            .unwrap()
            .request_hash
    );
}
