use super::*;
use crate::{Block, MessageId, Precision, PromptKind, SessionId, TimeQuality, Timestamp, Usage};

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
    m.at_precision = Some(Precision::Bounded);
    m.at_until = Timestamp::from_unix_ms(2_000);
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
        assert!(!group.present(&shown), "{group:?}");
        group.take(&mut shown, &from);
    }
    assert_eq!(shown, from);
}

#[test]
fn every_group_ranks_routes_in_the_documented_order() {
    // docs/sessions.md, one conversation through several routes.
    for group in FieldGroup::ALL {
        let mut order = Route::ALL.to_vec();
        order.sort_by_key(|route| group.rank(*route));
        assert_eq!(
            order,
            [
                Route::Live,
                Route::Capture,
                Route::Recorded,
                Route::Export,
                Route::Copy,
                Route::PageText
            ],
            "{group:?}"
        );
    }
}

#[test]
fn message_completion_has_three_values() {
    let user = message(Role::User);
    assert_eq!(
        MessageCompletion::of(&user, Route::Live),
        MessageCompletion::Finished
    );
    let mut answer = message(Role::Assistant);
    assert_eq!(
        MessageCompletion::of(&answer, Route::Live),
        MessageCompletion::InProgress
    );
    assert_eq!(
        MessageCompletion::of(&answer, Route::PageText),
        MessageCompletion::Unknown
    );
    answer.stop_reason = Some("end_turn".into());
    assert_eq!(
        MessageCompletion::of(&answer, Route::Live),
        MessageCompletion::Finished
    );
    let mut export = message(Role::Assistant);
    export.id = MessageId::parse("chatgpt:m1").unwrap();
    assert_eq!(
        MessageCompletion::of(&export, Route::Export),
        MessageCompletion::Unknown
    );
    assert!(MessageCompletion::Finished < MessageCompletion::Unknown);
    assert!(MessageCompletion::Unknown < MessageCompletion::InProgress);
}
