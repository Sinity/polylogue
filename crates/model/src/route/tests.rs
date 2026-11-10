use super::*;
use crate::{Block, MessageId, PromptKind, SessionId, TimeQuality, Timestamp, Usage};

fn message(role: Role) -> Message {
    serde_json::from_value(serde_json::json!({
        "id": "claude-code:m1",
        "session": "claude-code:s1",
        "role": role,
        "blocks": [],
    }))
    .unwrap()
}

/// A message with every field set, so taking every group from it must
/// reproduce it: no field is left out of the groups.
fn full() -> Message {
    let mut m = message(Role::Assistant);
    m.id = MessageId::parse("claude-code:m1").unwrap();
    m.session = SessionId::parse("claude-code:s1").unwrap();
    m.parent = Some(MessageId::parse("claude-code:m0").unwrap());
    m.at = Timestamp::from_unix_ms(1_000);
    m.at_quality = TimeQuality::Intrinsic;
    m.model = Some("model-a".into());
    m.prompt_kind = Some(PromptKind::Prose);
    m.blocks = vec![Block::Text { text: "a".into() }];
    m.usage = Usage {
        output: Some(3),
        ..Usage::default()
    };
    m.stop_reason = Some("end_turn".into());
    m.reported_cost_usd = Some(serde_json::Number::from(1));
    m.duration_ms = Some(9);
    m.production.phase = Some("final".into());
    m.attrs.insert("k".into(), serde_json::json!(1));
    m
}

#[test]
fn every_message_field_belongs_to_a_group() {
    let from = full();
    let mut shown = message(Role::User);
    for group in FieldGroup::ALL {
        assert!(group.present(&from), "{group:?}");
        assert!(!group.present(&shown) || *group == FieldGroup::Content);
        group.take(&mut shown, &from);
    }
    assert_eq!(shown, from);
}

#[test]
fn page_text_is_last_for_every_group() {
    for group in FieldGroup::ALL {
        for route in Route::ALL {
            if *route != Route::PageText {
                assert!(group.rank(*route) < group.rank(Route::PageText));
            }
        }
    }
}

#[test]
fn completion_has_three_values() {
    let user = message(Role::User);
    assert_eq!(Completion::of(&user, Route::Live), Completion::Finished);
    let mut answer = message(Role::Assistant);
    assert_eq!(Completion::of(&answer, Route::Live), Completion::InProgress);
    assert_eq!(
        Completion::of(&answer, Route::PageText),
        Completion::Unknown
    );
    answer.stop_reason = Some("end_turn".into());
    assert_eq!(Completion::of(&answer, Route::Live), Completion::Finished);
    let mut export = message(Role::Assistant);
    export.id = MessageId::parse("chatgpt:m1").unwrap();
    assert_eq!(Completion::of(&export, Route::Export), Completion::Unknown);
    assert!(Completion::Finished < Completion::Unknown);
    assert!(Completion::Unknown < Completion::InProgress);
}
