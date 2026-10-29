use super::*;
use crate::{
    Block, Message, MessageId, PromptKind, Role, TimeEvidence, Usage,
    bundle::{MessageRecord, Record},
};

fn message(role: Role, prompt_kind: Option<PromptKind>) -> Message {
    Message {
        id: MessageId::parse("test:session:message").unwrap(),
        session: crate::SessionId::parse("test:session").unwrap(),
        parent: None,
        role,
        at: None,
        at_quality: crate::TimeQuality::Unknown,
        at_precision: None,
        at_until: None,
        model: None,
        prompt_kind,
        blocks: vec![
            Block::Text {
                text: "visible λ".into(),
            },
            Block::Thinking {
                text: Some("private thought".into()),
                signature: None,
                redacted: false,
            },
            Block::Text {
                text: "later".into(),
            },
        ],
        usage: Usage::default(),
        stop_reason: None,
        reported_cost_usd: None,
        duration_ms: None,
        production: Default::default(),
        attrs: serde_json::Map::from_iter([(
            "provider_field".into(),
            serde_json::json!({"keep":"exact"}),
        )]),
    }
}
fn record(message: Message) -> Record {
    let source_version = message.content_hash().unwrap();
    Record::Message(Box::new(MessageRecord {
        reference: format!("message:{}", message.id),
        commit: 3,
        order: 7,
        source_version,
        block_count: message.blocks.len() as u64,
        block_ordinals: (0..message.blocks.len() as u64).collect(),
        message,
    }))
}
#[test]
fn dialogue_is_dialogue_and_original_positions_survive() {
    let original = record(message(Role::Assistant, None));
    let Record::Message(before) = original.clone() else {
        unreachable!()
    };
    let options = ReadOptions {
        view: View::Dialogue,
        ..Default::default()
    };
    let evidence = ProjectionEvidence {
        branch: BranchState::Active,
        ..Default::default()
    };
    let Record::Message(after) = project_record(original, &options, &evidence)
        .unwrap()
        .unwrap()
    else {
        unreachable!()
    };
    assert_eq!(after.block_ordinals, vec![0, 2]);
    assert_eq!(after.block_count, 3);
    assert_eq!(after.source_version, before.source_version);
    assert_eq!(after.message.attrs, before.message.attrs);
    assert_eq!(after.message.production, before.message.production);
    assert_eq!(after.message.time_evidence(), TimeEvidence::UNKNOWN);
    assert!(
        project_record(
            record(message(Role::User, Some(PromptKind::Harness))),
            &options,
            &evidence
        )
        .unwrap()
        .is_none()
    );
}
#[test]
fn context_requires_prepared_membership_and_does_not_inherit_current_branch_filter() {
    let options = ReadOptions {
        view: View::Context,
        ..Default::default()
    };
    assert!(matches!(
        project_record(
            record(message(Role::Assistant, None)),
            &options,
            &ProjectionEvidence::default()
        ),
        Err(ReadError::MissingEvidence { .. })
    ));
    let evidence = ProjectionEvidence {
        inherited: true,
        branch: BranchState::Abandoned {
            n: Some(1),
            total: Some(2),
        },
        context: Some(ContextEvidence {
            selected: true,
            complete: false,
            reason: Some("latest summary absent".into()),
            evicted: vec![],
            not_recorded: vec![],
        }),
        ..Default::default()
    };
    assert!(
        project_record(record(message(Role::Assistant, None)), &options, &evidence)
            .unwrap()
            .is_some()
    );
}
#[test]
fn view_names_have_no_predecessor_aliases_and_options_name_the_supported_views() {
    for name in ["full", "prompts", "actions", "unknown"] {
        assert!(View::parse(name).is_none());
    }
    assert_eq!(View::ALL.len(), View::NAMES.len());
    let invalid = ReadOptions {
        view: View::Dialogue,
        thinking: Some(Expansion::Full),
        ..Default::default()
    };
    let error = invalid.validate().unwrap_err().to_string();
    assert!(error.contains("thinking"));
    assert!(error.contains("transcript, context"));
}
#[test]
fn value_states_never_serialize_unknown_as_zero() {
    let unknown = serde_json::to_value(ValueState::<u64>::Unknown {
        reason: Some("not measured".into()),
    })
    .unwrap();
    assert_eq!(unknown["state"], "unknown");
    assert!(unknown.get("value").is_none());
    let zero = serde_json::to_value(ValueState::known(0)).unwrap();
    assert_eq!(zero["value"], 0);
    assert!(
        serde_json::from_value::<ValueState<u64>>(serde_json::json!({"state":"unknown","value":0}))
            .is_err()
    );
}
#[test]
fn estimator_uses_utf8_and_long_runs_consistently() {
    assert_eq!(estimate_tokens(""), 0);
    assert_eq!(estimate_tokens("a b c"), 4);
    assert_eq!(estimate_tokens("λ"), 2);
    assert_eq!(estimate_tokens(&"x".repeat(80)), 20);
    assert_eq!(TOKEN_ESTIMATOR, "utf8-words-v1");
}

#[test]
fn encrypted_only_thinking_is_omitted_without_rewriting_its_source_identity() {
    let mut source = message(Role::Assistant, None);
    source.blocks.push(Block::Thinking {
        text: None,
        signature: Some("opaque-provider-reasoning".into()),
        redacted: false,
    });
    let source = record(source);
    let Record::Message(before) = source.clone() else {
        unreachable!()
    };
    let after = project_record(
        source,
        &ReadOptions::default(),
        &ProjectionEvidence {
            branch: BranchState::Active,
            ..Default::default()
        },
    )
    .unwrap()
    .unwrap();
    let Record::Message(after) = after else {
        unreachable!()
    };
    assert_eq!(after.block_ordinals, vec![0, 1, 2]);
    assert_eq!(after.block_count, 4);
    assert_eq!(after.source_version, before.source_version);
    assert!(
        after
            .message
            .blocks
            .iter()
            .any(|block| matches!(block, Block::Thinking { text: Some(_), .. }))
    );
}

#[test]
fn document_dialogue_refuses_by_target_without_projecting_its_positions() {
    let document = crate::Document::new(
        crate::DocumentId::parse("test:document").unwrap(),
        crate::DocumentPurpose::Context,
    );
    let row = crate::bundle::DocumentRecord {
        reference: format!("document:{}", document.id),
        commit: 1,
        document,
    };
    let error = project_record(
        Record::Document(row),
        &ReadOptions {
            view: View::Dialogue,
            ..Default::default()
        },
        &ProjectionEvidence::default(),
    )
    .unwrap_err();
    assert!(
        matches!(error,ReadError::UnsupportedTarget{accepted_views,..} if accepted_views==vec![View::Transcript,View::Raw])
    );
}
