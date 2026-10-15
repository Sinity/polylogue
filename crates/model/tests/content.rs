use polylogue_model::*;
use serde_json::{Value, json};

fn session(native: &str) -> Session {
    let origin = Origin::new("future.provider").unwrap();
    let id = SessionId::new(&origin, native).unwrap();
    Session::new(id, origin, native, SessionKind::Main).unwrap()
}
fn message(session: &Session, native: &str) -> Message {
    Message::new(
        MessageId::new(session.id.clone(), native).unwrap(),
        IdKind::Provider,
        Role::Assistant,
        WrittenBy::Unknown,
    )
}
fn roundtrip<T>(value: &T) -> T
where
    T: serde::Serialize + serde::de::DeserializeOwned,
{
    serde_json::from_slice(&serde_json::to_vec(value).unwrap()).unwrap()
}

#[test]
fn unknown_does_not_become_false_zero_empty_or_epoch() {
    let mut session = session("a:b");
    let mut msg = message(&session, "m:n");
    assert_eq!(session.temporary, None);
    assert_eq!(session.models, None);
    assert_eq!(msg.on_active_path, None);
    assert_eq!(msg.at, None);
    assert_eq!(msg.version_hash, None);
    assert_eq!(msg.written_by, WrittenBy::Unknown);
    session.temporary = Some(false);
    session.models = Some(Default::default());
    msg.on_active_path = Some(false);
    msg.in_progress = Some(false);
    msg.copy_origin = Some(CopyOrigin::Unknown);
    msg.at = Some(ProviderTime::Unknown);
    msg.usage = Some(Usage {
        input: Some(0),
        output: Some(u64::MAX),
        provider_cost_nusd: Some(Usd(i64::MAX)),
        ..Usage::default()
    });
    msg.push(
        BlockSlot::Item(0),
        BlockContent::Text {
            text: String::new(),
        },
    )
    .unwrap();
    let parsed = Parsed {
        sessions: vec![ParsedSession {
            session,
            messages: vec![msg],
            events: vec![],
        }],
        ..Parsed::default()
    };
    parsed.validate().unwrap();
    assert_eq!(roundtrip(&parsed), parsed);
    let wire = serde_json::to_value(&parsed).unwrap();
    assert_eq!(wire["sessions"][0]["messages"][0]["on_active_path"], false);
    assert_eq!(wire["sessions"][0]["messages"][0]["usage"]["input"], 0);
    assert_eq!(
        wire["sessions"][0]["messages"][0]["usage"]["provider_cost_nusd"],
        i64::MAX
    );
}

#[test]
fn ids_keep_separate_relations_even_when_opaque_text_disagrees() {
    let a = session("native:with:colons");
    let mid = MessageId::from_stored("opaque:foreign:layout", a.id.clone()).unwrap();
    let id = BlockId::from_stored("also:opaque", mid.clone(), BlockSlot::Reasoning(7)).unwrap();
    let decoded: BlockId = roundtrip(&id);
    assert_eq!(decoded.message().session(), &a.id);
    assert_eq!(decoded.message(), &mid);
    assert_eq!(decoded.as_str(), "also:opaque");
    assert_eq!(decoded.slot(), &BlockSlot::Reasoning(7));
    assert_eq!(decoded.index(), Some(7));
    assert_ne!(
        BlockId::new(mid.clone(), 0),
        BlockId::reasoning(mid.clone(), 0)
    );
    assert!(BlockId::attachment(mid.clone(), "").is_err());
    assert!(BlockId::from_stored("", mid.clone(), BlockSlot::Item(0)).is_err());
    let mut wire = serde_json::to_value(&id).unwrap();
    wire["id"] = json!("");
    assert!(serde_json::from_value::<BlockId>(wire).is_err());
    let mut wire = serde_json::to_value(&id).unwrap();
    wire["slot"] = json!({"attachment":""});
    assert!(serde_json::from_value::<BlockId>(wire).is_err());
    assert!(Session::new(a.id.clone(), a.origin.clone(), "", SessionKind::Chat).is_err());
    let mut wire = serde_json::to_value(a).unwrap();
    wire["native_id"] = json!("");
    assert!(serde_json::from_value::<Session>(wire).is_err());
}

#[test]
fn insertion_and_envelope_validation_reject_wrong_relations_and_duplicates() {
    let a = session("a");
    let b = session("b");
    let mut msg = message(&a, "m");
    msg.push(
        BlockSlot::Item(0),
        BlockContent::Text {
            text: "text".into(),
        },
    )
    .unwrap();
    msg.push(
        BlockSlot::Reasoning(0),
        BlockContent::Thinking(Thinking::default()),
    )
    .unwrap();
    assert_eq!(
        msg.push(
            BlockSlot::Item(0),
            BlockContent::Text {
                text: "duplicate".into()
            }
        )
        .unwrap_err(),
        ModelError::DuplicateBlock
    );
    let foreign = Block {
        id: BlockId::new(message(&b, "m").id, 1),
        content: BlockContent::Text {
            text: "foreign".into(),
        },
    };
    assert_eq!(
        msg.insert_block(foreign).unwrap_err(),
        ModelError::BlockMessage
    );
    msg.validate().unwrap();
    let mut parsed = ParsedSession {
        session: b,
        messages: vec![msg.clone()],
        events: vec![],
    };
    assert_eq!(parsed.validate(), Err(ModelError::MessageSession));
    parsed.session = a;
    parsed.messages.push(msg);
    assert_eq!(parsed.validate(), Err(ModelError::DuplicateMessage));
    parsed.messages.pop();
    let duplicate = parsed.messages[0].blocks[0].clone();
    parsed.messages[0].blocks.push(duplicate);
    assert_eq!(parsed.validate(), Err(ModelError::DuplicateBlock));
}

#[test]
fn tool_result_requires_status_and_preserves_empty_output_and_unknown_reasons() {
    assert!(serde_json::from_value::<ToolResult>(json!({"text":""})).is_err());
    assert!(serde_json::from_value::<ToolResult>(json!({"outcome":"unknown"})).is_err());
    assert!(
        serde_json::from_value::<ToolResult>(json!({"outcome":"ok","outcome_reason":"bad"}))
            .is_err()
    );
    assert!(
        serde_json::from_value::<ToolResult>(
            json!({"outcome":"ok","text":"","blob":BlobHash::of(b"x")})
        )
        .is_err()
    );
    for outcome in [
        ToolOutcome::Ok,
        ToolOutcome::Error { reason: None },
        ToolOutcome::Error {
            reason: Some(ErrorReason::Denied("policy:new".into())),
        },
        ToolOutcome::Unknown {
            reason: UnknownReason::Other("future_reason".into()),
        },
    ] {
        for output in [
            None,
            Some(ToolOutput::Text(String::new())),
            Some(ToolOutput::Blob(BlobHash::of(b""))),
        ] {
            let result = ToolResult {
                call_id: None,
                outcome: outcome.clone(),
                exit_code: Some(0),
                duration_ms: Some(0),
                output,
            };
            assert_eq!(roundtrip(&result), result);
        }
    }
    assert!(
        !ToolOutcome::Unknown {
            reason: UnknownReason::NoResult
        }
        .is_success()
    );
    assert_eq!(
        UnknownReason::parse("future_reason").as_str(),
        "future_reason"
    );
    assert_eq!(ErrorReason::parse("future_error").code(), "future_error");
}

#[test]
fn absent_and_explicit_json_null_and_false_stay_distinct() {
    let absent: ToolCall = serde_json::from_value(json!({})).unwrap();
    let null: ToolCall = serde_json::from_value(json!({"input":null,"tool":""})).unwrap();
    assert_eq!(absent.input, None);
    assert_eq!(null.input, Some(Value::Null));
    assert_eq!(roundtrip(&null), null);
    let mut msg = message(&session("json"), "m");
    msg.attrs = Some(Value::Null);
    msg.push(BlockSlot::Item(0), BlockContent::ToolCall(null.clone()))
        .unwrap();
    assert_eq!(roundtrip(&msg), msg);
    assert_eq!(null.tool, Some(String::new()));
    let a = Thinking::default();
    let b = Thinking {
        encrypted: Some(false),
        text: Some(String::new()),
        ..Thinking::default()
    };
    assert_ne!(a, b);
    assert_eq!(roundtrip(&b), b);
    let attachment = Attachment::default();
    assert_eq!(attachment.direction, AttachmentDirection::Unknown);
    assert_eq!(roundtrip(&attachment), attachment);
}

#[test]
fn link_constructor_and_validation_enforce_kind_and_supplied_relations() {
    let a = session("a");
    let b = session("b");
    assert_eq!(
        Link::new(
            LinkKind::Subagent,
            LinkHow::Resume,
            a.id.clone(),
            b.id.clone(),
            "thread_spawn"
        )
        .unwrap_err(),
        ModelError::LinkKind
    );
    assert_eq!(
        Link::new(
            LinkKind::Subagent,
            LinkHow::Fresh,
            a.id.clone(),
            a.id.clone(),
            "thread_spawn"
        )
        .unwrap_err(),
        ModelError::SelfLink
    );
    let mut link = Link::new(
        LinkKind::Subagent,
        LinkHow::Copied,
        a.id.clone(),
        b.id.clone(),
        "thread_spawn",
    )
    .unwrap();
    link.spawned_by = Some(BlockId::new(message(&a, "spawn:call").id, 0));
    link.result_use = Some(ResultUse::Unknown);
    assert_eq!(roundtrip(&link), link);
    link.validate().unwrap();
    link.branch_point = Some(message(&b, "wrong").id);
    assert_eq!(link.validate(), Err(ModelError::MessageSession));
}

#[test]
fn record_outcome_is_copy_for_copy_ignore_reasons() {
    let outcome = RecordOutcome::Ignored(3_u8);
    let _copy = outcome;
    assert_eq!(outcome, RecordOutcome::Ignored(3));
}

#[test]
fn source_version_digest_uses_validated_hex_wire_without_changing_policy() {
    let mut msg = message(&session("version"), "m");
    assert_eq!(msg.version_hash, None);
    let digest = BlobHash::from_bytes([0xab; 32]);
    msg.version_hash = Some(digest);
    let mut wire = serde_json::to_value(&msg).unwrap();
    assert_eq!(wire["version_hash"], "ab".repeat(32));
    assert_eq!(roundtrip(&msg), msg);
    wire["version_hash"] = json!([171, 171]);
    assert!(serde_json::from_value::<Message>(wire).is_err());
}
