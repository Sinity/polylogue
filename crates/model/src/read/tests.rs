use super::*;
use crate::{
    Block, Message, MessageId, Precision, PromptKind, Role, TimeAccuracy, Timestamp, Usage,
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
        injections: Vec::new(),
        bindings: Vec::new(),
        copied_for: Vec::new(),
        message,
    }))
}
#[test]
fn dialogue_is_dialogue_and_original_positions_survive() {
    let mut timed = message(Role::Assistant, None);
    timed
        .set_time(
            Timestamp::from_unix_ms(1_000),
            TimeAccuracy::intrinsic(Precision::Instant),
        )
        .unwrap();
    let original = record(timed);
    let Record::Message(before) = original.clone() else {
        unreachable!()
    };
    let options = ReadOptions {
        view: View::Dialogue,
        ..Default::default()
    };
    let facts = LayoutFacts {
        branch: BranchState::Active,
        ..Default::default()
    };
    let Record::Message(after) = project_record(original, &options, &facts).unwrap().unwrap()
    else {
        unreachable!()
    };
    assert_eq!(after.block_ordinals, vec![0, 2]);
    assert_eq!(after.block_count, 3);
    assert_eq!(after.source_version, before.source_version);
    assert_eq!(after.message.attrs, before.message.attrs);
    assert_eq!(after.message.production, before.message.production);
    assert_eq!(after.message.at, Timestamp::from_unix_ms(1_000));
    assert_eq!(
        after.message.time_accuracy(),
        TimeAccuracy::intrinsic(Precision::Instant)
    );
    assert!(
        project_record(
            record(message(Role::User, Some(PromptKind::Harness))),
            &options,
            &facts
        )
        .unwrap()
        .is_none()
    );
}
#[test]
fn context_view_needs_prepared_membership() {
    let options = ReadOptions {
        view: View::Context,
        ..Default::default()
    };
    assert!(matches!(
        project_record(
            record(message(Role::Assistant, None)),
            &options,
            &LayoutFacts::default()
        ),
        Err(ReadError::MissingPreparedData { .. })
    ));
    let facts = LayoutFacts {
        inherited: true,
        branch: BranchState::Abandoned {
            n: Some(1),
            total: Some(2),
        },
        context: Some(ContextState {
            selected: true,
            complete: false,
            reason: Some("latest summary absent".into()),
            evicted: vec![],
            not_recorded: vec![],
        }),
        ..Default::default()
    };
    let Record::Message(row) =
        project_record(record(message(Role::Assistant, None)), &options, &facts)
            .unwrap()
            .unwrap()
    else {
        unreachable!()
    };
    // The context view keeps every block kind, thinking included.
    assert_eq!(row.block_ordinals, [0, 1, 2]);
}
#[test]
fn unknown_view_names_are_refused_and_options_name_the_supported_views() {
    assert!(View::parse("unknown").is_none());
    assert_eq!(
        "unknown".parse::<View>().unwrap_err().to_string(),
        "unknown view \"unknown\"; views: transcript, dialogue, overview, context, tree, changes, live, raw, trace"
    );
    let invalid = ReadOptions {
        view: View::Dialogue,
        show: vec!["thinking:full".into()],
        ..Default::default()
    };
    let error = invalid.validate().unwrap_err().to_string();
    assert_eq!(
        error,
        "view dialogue does not accept thinking; accepted by transcript, context"
    );
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
    // 17 characters count as three words, so the word term (6) beats the byte term (5).
    assert_eq!(estimate_tokens(&format!("{} a", "x".repeat(17))), 6);
    assert_eq!(TOKEN_ESTIMATOR, "utf8-words-v1");
}

#[test]
fn encrypted_only_thinking_is_omitted() {
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
        &LayoutFacts {
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
        &LayoutFacts::default(),
    )
    .unwrap_err();
    assert!(
        matches!(error,ReadError::UnsupportedTarget{accepted_views,..} if accepted_views==vec![View::Transcript,View::Raw])
    );
}

#[test]
fn expansion_headers_is_the_only_public_collapsed_spelling() {
    assert_eq!(
        serde_json::to_string(&Expansion::Collapsed).unwrap(),
        "\"headers\""
    );
    assert_eq!(
        serde_json::from_str::<Expansion>("\"headers\"").unwrap(),
        Expansion::Collapsed
    );
    assert!(serde_json::from_str::<Expansion>("\"collapsed\"").is_err());
}

#[test]
fn layout_keeps_selected_hook_injections_only() {
    use crate::injection::{InjectionEvidence, StatedHookField};
    let mut original = record(message(Role::Assistant, None));
    let Record::Message(row) = &mut original else {
        unreachable!()
    };
    let field = |n: u64| StatedHookField {
        input: format!("synthetic-input-{n}"),
        chain: 1,
        start: n + 1,
        end: n + 2,
        pointer: format!("/message/content/{n}"),
    };
    row.injections = (0..3)
        .map(|block| InjectionEvidence {
            message: row.message.id.clone(),
            source_version: row.source_version,
            block,
            observed: Some(field(block)),
            declared: vec![field(block + 4)],
        })
        .collect();
    let before = row.clone();
    let projected = project_record(
        original,
        &ReadOptions {
            view: View::Dialogue,
            ..Default::default()
        },
        &LayoutFacts::default(),
    )
    .unwrap()
    .unwrap();
    let Record::Message(after) = projected else {
        unreachable!()
    };
    assert_eq!(after.block_ordinals, [0, 2]);
    assert_eq!(
        after.injections,
        [before.injections[0].clone(), before.injections[2].clone()]
    );
    assert_eq!(after.source_version, before.source_version);
    assert_eq!(after.block_count, before.block_count);
    assert_eq!(after.message.attrs, before.message.attrs);
    assert_eq!(
        after.message.blocks,
        [
            before.message.blocks[0].clone(),
            before.message.blocks[2].clone()
        ]
    );
    after.validate().unwrap();
}

#[test]
fn layout_keeps_slots_and_ambiguous_same_hash_sources() {
    use crate::attachment_binding::{
        AttachmentBinding, AttachmentCandidate, StatedAttachmentSource,
    };
    let attachment = |source: &str| crate::Attachment {
        source_id: Some(source.into()),
        ..Default::default()
    };
    let mut source = message(Role::Assistant, None);
    source.blocks = vec![
        Block::File(attachment("file:source")),
        Block::ToolResult {
            call_id: None,
            reported_duration_ms: None,
            outcome: crate::ToolOutcome::Ok,
            output: Some("result".into()),
            data: None,
            attachments: vec![attachment("result:first"), attachment("result:second")],
        },
        Block::Text {
            text: "kept".into(),
        },
    ];
    let mut original = record(source);
    let Record::Message(row) = &mut original else {
        unreachable!()
    };
    let candidate = |name: &str| AttachmentCandidate {
        hash: crate::BlobHash::of(b"same"),
        size: 4,
        sources: vec![StatedAttachmentSource {
            namespace: None,
            input: name.into(),
            chain: 1,
            source_version: format!("historical:{name}"),
            members: vec![format!("members/{name}")],
            names: vec![name.into()],
            media_types: vec!["application/octet-stream".into()],
            declared_sizes: vec![9],
        }],
    };
    row.bindings = [
        (0, 0, "file:source"),
        (1, 0, "result:first"),
        (1, 1, "result:second"),
    ]
    .into_iter()
    .map(|(block, attachment, source_id)| AttachmentBinding {
        source_version: row.source_version,
        namespace: crate::attachment_binding::AttachmentNamespace {
            input: "synthetic-import".into(),
            chain: 1,
            import_id: "original".into(),
        },
        exported_from: crate::attachment_binding::AttachmentNamespace {
            input: "synthetic-export".into(),
            chain: 2,
            import_id: "current".into(),
        },
        prior_exports: vec![],
        block,
        attachment,
        source_id: source_id.into(),
        candidates: vec![candidate("first"), candidate("second")],
    })
    .collect();
    let before = row.clone();
    original.validate().unwrap();
    let projected = project_record(
        original.clone(),
        &ReadOptions {
            hide: vec!["tools".into()],
            ..Default::default()
        },
        &LayoutFacts::default(),
    )
    .unwrap()
    .unwrap();
    let Record::Message(after) = projected else {
        unreachable!()
    };
    assert_eq!(after.block_ordinals, [0, 2]);
    assert_eq!(after.bindings, [before.bindings[0].clone()]);
    assert_eq!(after.bindings[0].candidates.len(), 2);
    assert_eq!(
        after.bindings[0].candidates[0].hash,
        after.bindings[0].candidates[1].hash
    );
    assert_ne!(
        after.bindings[0].candidates[0].sources,
        after.bindings[0].candidates[1].sources
    );
    assert_eq!(after.source_version, before.source_version);
    after.validate().unwrap();
    let dialogue = project_record(
        original,
        &ReadOptions {
            view: View::Dialogue,
            ..Default::default()
        },
        &LayoutFacts::default(),
    )
    .unwrap()
    .unwrap();
    let Record::Message(dialogue) = dialogue else {
        unreachable!()
    };
    assert_eq!(dialogue.block_ordinals, [2]);
    assert!(dialogue.bindings.is_empty());
    dialogue.validate().unwrap();
}

fn shown(show: &[&str], hide: &[&str]) -> ReadOptions {
    ReadOptions {
        show: show.iter().map(|word| (*word).to_owned()).collect(),
        hide: hide.iter().map(|word| (*word).to_owned()).collect(),
        ..ReadOptions::default()
    }
}

#[test]
fn show_and_hide_words_expand_through_one_table() {
    let defaults = ReadOptions::default();
    assert_eq!(defaults.tools(), Expansion::Collapsed);
    assert_eq!(defaults.thinking(), Expansion::Collapsed);
    assert_eq!(defaults.harness(), Expansion::Collapsed);
    assert_eq!(defaults.output_lines(), 100);
    assert!(!defaults.inherited() && !defaults.branches() && !defaults.turns());
    assert!(defaults.annotations() && !defaults.context_meter());
    assert_eq!(defaults.spawns(), SpawnDisplay::Card);
    let options = shown(
        &["tools:full", "thinking", "inherited", "turns", "context"],
        &["annotations", "spawns"],
    );
    options.validate().unwrap();
    assert_eq!(options.tools(), Expansion::Full);
    assert_eq!(options.output_lines(), 0);
    assert_eq!(options.thinking(), Expansion::Full);
    assert_eq!(options.harness(), Expansion::Collapsed);
    assert!(options.inherited() && options.turns() && options.context_meter());
    assert!(!options.annotations() && options.spawns() == SpawnDisplay::Hidden);
    // The options keep the words the caller wrote.
    assert_eq!(
        serde_json::to_value(&options).unwrap(),
        serde_json::json!({
            "view": "transcript",
            "show": ["tools:full", "thinking", "inherited", "turns", "context"],
            "hide": ["annotations", "spawns"],
        })
    );
    // A hide word wins over a show word for the same kind; the last show
    // word wins over an earlier one.
    let later = shown(&["tools:line", "tools:7", "thinking:line"], &["thinking"]);
    assert_eq!(later.tools(), Expansion::Full);
    assert_eq!(later.output_lines(), 7);
    assert_eq!(later.thinking(), Expansion::None);
    for kind in ShowKind::ALL {
        assert_eq!(ShowKind::parse(kind.name()), Some(kind));
    }
    for (show, hide) in [(&["tools:loud"][..], &[][..]), (&[], &["nothing"])] {
        assert!(matches!(
            shown(show, hide).validate(),
            Err(ReadError::UnknownShowWord { .. })
        ));
    }
}

#[test]
fn spawn_levels_and_help_come_from_the_show_table() {
    for (word, expected) in [
        ("spawns", SpawnDisplay::Inline { depth: None }),
        ("spawns:full", SpawnDisplay::Inline { depth: None }),
        ("spawns:card", SpawnDisplay::Card),
        ("spawns:2", SpawnDisplay::Inline { depth: Some(2) }),
        (
            "spawns:4294967296",
            SpawnDisplay::Inline {
                depth: Some(u32::MAX),
            },
        ),
    ] {
        let options = shown(&[word], &[]);
        options.validate().unwrap();
        assert_eq!(options.spawns(), expected, "{word}");
    }
    for refused in ["spawns:N", "spawns:0", "spawns:deep", "tools:0"] {
        assert!(
            matches!(
                shown(&[refused], &[]).validate(),
                Err(ReadError::UnknownShowWord { .. })
            ),
            "{refused}"
        );
    }
    assert_eq!(
        ShowKind::HELP,
        "Item kinds to show, each with an optional level (the fullest when omitted): tools[:line|full|N] thinking[:line|full] harness[:line|full] inherited[:full] branches[:full] spawns[:card|full|N] annotations[:header] turns[:dividers] context[:meter]"
    );
    let inline_context = ReadOptions {
        view: View::Context,
        ..shown(&["spawns:full"], &[])
    };
    assert!(matches!(
        inline_context.validate(),
        Err(ReadError::UnsupportedOption { .. })
    ));
}

#[test]
fn the_context_meter_is_refused_where_no_message_header_is_read() {
    for view in [View::Overview, View::Tree, View::Live, View::Trace] {
        let options = ReadOptions {
            view,
            ..shown(&["context"], &[])
        };
        assert!(
            matches!(
                options.validate(),
                Err(ReadError::UnsupportedOption { ref option, .. }) if option == "context_meter"
            ),
            "{view}"
        );
    }
    for view in [View::Transcript, View::Dialogue, View::Context] {
        let options = ReadOptions {
            view,
            ..shown(&["context"], &[])
        };
        assert!(options.validate().is_ok(), "{view}");
    }
}

#[test]
fn injected_and_summary_messages_are_harness_and_tool_only_rows_are_tools() {
    for kind in [PromptKind::InjectedContext, PromptKind::CompactionSummary] {
        let message = message(Role::User, Some(kind));
        assert_eq!(crate::WrittenBy::of(&message), crate::WrittenBy::Harness);
        assert!(!View::Dialogue.keeps_message(&message));
        assert!(!crate::is_prompt(message.role, message.prompt_kind));
    }
    let mut tool = message(Role::User, Some(PromptKind::Harness));
    tool.blocks = vec![Block::ToolResult {
        reported_duration_ms: None,
        call_id: Some("call".into()),
        outcome: crate::ToolOutcome::Ok,
        output: Some("result".into()),
        data: None,
        attachments: vec![],
    }];
    assert_eq!(crate::WrittenBy::of(&tool), crate::WrittenBy::Tool);
    tool.blocks.push(Block::Text {
        text: "runtime feedback".into(),
    });
    assert_eq!(crate::WrittenBy::of(&tool), crate::WrittenBy::Harness);
}

#[test]
fn follow_up_holds_one_request_per_direction() {
    use command_surface::{Deadline, Format, Request, TokenBudget};
    let request = Request {
        operation: "read".into(),
        arguments: serde_json::json!({"target":"session:test:one","view":"transcript","alternates":["en","de"]}),
        request_id: "next-child".into(),
        deadline: Deadline::None,
        budget_tokens: TokenBudget::Unlimited,
        cursor: Some("opaque-next-cursor".into()),
        fresh: true,
        format: Format::Json,
    };
    let footer = ReadFooter {
        next: Some(ReadyCall {
            request: request.clone(),
        }),
        previous: Some(ReadyCall {
            request: Request {
                request_id: "previous-child".into(),
                cursor: Some("opaque-previous-cursor".into()),
                ..request
            },
        }),
        ..Default::default()
    };
    let encoded = serde_json::to_string(&footer).unwrap();
    let value: serde_json::Value = serde_json::from_str(&encoded).unwrap();
    assert_eq!(
        value["next"],
        serde_json::json!({"operation":"read","arguments":{"target":"session:test:one","view":"transcript","alternates":["en","de"]},"request_id":"next-child","deadline":null,"budget_tokens":null,"cursor":"opaque-next-cursor","fresh":true,"format":"json"})
    );
    assert_eq!(value["previous"]["request_id"], "previous-child");
    assert_eq!(value["previous"]["cursor"], "opaque-previous-cursor");
    assert_eq!(encoded.matches("opaque-next-cursor").count(), 1);
    assert_eq!(encoded.matches("opaque-previous-cursor").count(), 1);
    assert_eq!(
        serde_json::from_str::<ReadFooter>(&encoded).unwrap(),
        footer
    );
}

#[test]
fn full_tool_output_and_numeric_line_limits_expand_from_their_word() {
    for (word, lines) in [
        ("tools:full", 0),
        ("tools:2", 2),
        ("tools:7", 7),
        ("tools:100", 100),
        ("tools:18446744073709551615", u64::MAX),
    ] {
        let options = shown(&[word], &[]);
        options.validate().unwrap();
        assert_eq!(options.tools(), Expansion::Full, "{word}");
        assert_eq!(options.output_lines(), lines, "{word}");
        assert_eq!(
            serde_json::to_value(&options).unwrap()["show"],
            serde_json::json!([word])
        );
    }
}

#[test]
fn read_at_is_a_time_or_a_position_in_one_string() {
    // A message word is no place to read at: it stays the caller's words,
    // which the reading refuses with the line that selects the message.
    let message: ReadOptions =
        serde_json::from_value(serde_json::json!({"view":"context","at":"claude-code:u1"}))
            .unwrap();
    assert_eq!(
        message.at,
        Some(ReadAt::TimeExpression("claude-code:u1".into()))
    );
    assert_eq!(
        serde_json::to_value(&message).unwrap()["at"],
        "claude-code:u1"
    );
    let historical = serde_json::json!({"at":"generation:2:3"});
    let options: ReadOptions = serde_json::from_value(historical).unwrap();
    options.validate().unwrap();
    assert_eq!(
        options.annotation_position(),
        Some(&crate::ChangePosition {
            index_instance: "generation".into(),
            index: 2,
            archive: 3
        })
    );
    assert_eq!(
        serde_json::to_value(&options).unwrap()["at"],
        "generation:2:3"
    );
    let schema = schemars::schema_for!(ReadOptions).to_value();
    assert_eq!(
        schema["properties"]["at"],
        serde_json::json!({
            "description":"Annotations and pages as they stood at this time or position",
            "type":["string","null"]
        })
    );
}

#[test]
fn annotation_times_keep_the_same_read_option_string_codec() {
    for text in [
        "2026-11-21T10:00:00Z",
        "1h",
        "08:00",
        "08:00:00",
        "2026-11-21T09:00:00",
    ] {
        let options: ReadOptions = serde_json::from_value(serde_json::json!({"at":text})).unwrap();
        options.validate().unwrap();
        assert_eq!(serde_json::to_value(&options).unwrap()["at"], text);
        assert!(options.annotation_position().is_none());
    }
}

#[test]
fn read_token_choices_preserve_the_current_stored_page_caps() {
    let default = PageBudget::default();
    for (choice, tokens) in [
        (ReadBudget::Default, 4000),
        (ReadBudget::Tokens(0), 1),
        (ReadBudget::Tokens(73), 73),
        (ReadBudget::Tokens(u64::MAX), u64::MAX),
        (ReadBudget::Unlimited, u64::MAX),
    ] {
        let resolved = default.with_tokens(choice);
        assert_eq!(resolved.tokens.get(), tokens);
        assert_eq!(resolved.bytes, default.bytes);
        assert_eq!(resolved.items, default.items);
        let encoded = serde_json::to_value(resolved).unwrap();
        assert_eq!(
            encoded,
            serde_json::json!({"items":50,"tokens":tokens,"bytes":21000})
        );
        assert_eq!(
            serde_json::from_value::<PageBudget>(encoded).unwrap(),
            resolved
        );
    }
}
