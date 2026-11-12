use super::*;
use polylogue_model::{
    Block, Message, MessageId, Role, SessionId, TimeQuality, Usage,
    bundle::{MessageRecord, Record},
    read::*,
};
use serde_json::json;
use std::{
    io::{self, Write},
    num::{NonZeroU64, NonZeroUsize},
};
fn message(blocks: Vec<Block>) -> Record {
    let message = Message {
        id: MessageId::parse("test:session:message").unwrap(),
        session: SessionId::parse("test:session").unwrap(),
        parent: None,
        role: Role::Assistant,
        at: None,
        at_quality: TimeQuality::Unknown,
        at_precision: None,
        at_until: None,
        model: None,
        prompt_kind: None,
        blocks,
        usage: Usage::default(),
        stop_reason: None,
        reported_cost_usd: None,
        duration_ms: None,
        production: Default::default(),
        attrs: serde_json::Map::from_iter([("native".into(), json!({"exact":[3,null,"λ"]}))]),
    };
    Record::Message(Box::new(MessageRecord {
        reference: format!("message:{}", message.id),
        commit: 1,
        order: 0,
        source_version: message.content_hash().unwrap(),
        block_count: message.blocks.len() as u64,
        block_ordinals: (0..message.blocks.len() as u64).collect(),
        injections: Vec::new(),
        bindings: Vec::new(),
        message,
    }))
}
fn header() -> ReadHeader {
    ReadHeader {
        target: "session:test:session".into(),
        view: View::Transcript,
        title: ValueState::NotRecorded,
        kind: HeaderKind::Record {
            record_type: "test".into(),
        },
        figures: vec![],
        zone: None,
    }
}
fn document(items: Vec<ReadItem>) -> ReadDocument {
    ReadDocument {
        header: header(),
        items,
        footer: ReadFooter::default(),
    }
}
fn text(text: &str) -> ReadItem {
    ReadItem {
        source: Source::record("block:test:session:message#3"),
        kind: ItemKind::Text { text: text.into() },
    }
}
fn encode(doc: &ReadDocument, format: Format) -> String {
    let mut out = vec![];
    encode_document(doc, format, EncodeOptions::default(), &mut out).unwrap();
    String::from_utf8(out).unwrap()
}
fn layout(record: Record, options: &ReadOptions, evidence: &ProjectionEvidence) -> Vec<ReadItem> {
    let record = project_record(record, options, evidence).unwrap().unwrap();
    layout_record(&record, options, evidence)
        .unwrap()
        .into_iter()
        .filter_map(|p| {
            if let ReadPart::Item(item) = p {
                Some(item)
            } else {
                None
            }
        })
        .collect()
}
#[test]
fn formats_use_one_layout_and_archived_text_survives_json() {
    let payload = "# operator text\n<script>alert('archive')</script>\nsecret-like bearer abcλ🙂\n```\nend\x1b[2J";
    let items = layout(
        message(vec![Block::Text {
            text: payload.into(),
        }]),
        &ReadOptions::default(),
        &ProjectionEvidence::default(),
    );
    let doc = document(items);
    let encoded = encode(&doc, Format::Json);
    assert_eq!(serde_json::from_str::<ReadDocument>(&encoded).unwrap(), doc);
    let lines = encode(&doc, Format::Jsonl);
    let parts: Vec<ReadPart> = lines
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(parts, doc.clone().into_parts().collect::<Vec<_>>());
    let terminal = encode(&doc, Format::Terminal);
    assert!(terminal.contains("bearer abcλ🙂"));
    assert!(terminal.contains("\\x1b[2J"));
    assert!(!terminal.contains('\x1b'));
    let markdown = encode(&doc, Format::Markdown);
    assert!(markdown.contains("\\# operator text"));
    assert!(markdown.contains("&lt;script&gt;"));
    assert!(!markdown.contains("<script>"));
    let html = encode(&doc, Format::Html);
    assert!(html.contains("&lt;script&gt;"));
    assert!(!html.contains("<script>"));
    assert!(html.contains("data-reference=\"block:test:session:message#0\""));
}
#[test]
fn dialogue_has_no_thinking_harness_or_tools_and_keeps_original_ordinals() {
    let options = ReadOptions {
        view: View::Dialogue,
        ..Default::default()
    };
    let items = layout(
        message(vec![
            Block::Thinking {
                text: Some("hidden".into()),
                signature: None,
                redacted: false,
            },
            Block::Text {
                text: "spoken".into(),
            },
        ]),
        &options,
        &ProjectionEvidence::default(),
    );
    assert_eq!(items.len(), 2);
    assert_eq!(items[1].source.block, Some(1));
    assert!(matches!(&items[1].kind,ItemKind::Text{text} if text=="spoken"));
    let mut record = message(vec![Block::Text {
        text: "harness text".into(),
    }]);
    if let Record::Message(row) = &mut record {
        row.message.prompt_kind = Some(polylogue_model::PromptKind::Harness);
        row.source_version = row.message.content_hash().unwrap();
    }
    assert!(
        project_record(record, &options, &ProjectionEvidence::default())
            .unwrap()
            .is_none()
    );
}
#[test]
fn tool_pair_uses_declared_relationship_outside_current_record() {
    let call = Block::ToolCall {
        written_text: None,
        call_id: Some("call-x".into()),
        tool: Some("Bash".into()),
        action: polylogue_model::ToolAction::Shell,
        input: json!({"argv":["cargo","test"]}),
        command: Some("cargo test".into()),
        path: None,
        run_by_script: false,
    };
    let result = Block::ToolResult {
        reported_duration_ms: Some(700),
        call_id: Some("call-x".into()),
        outcome: polylogue_model::ToolOutcome::Error {
            reason: Some(polylogue_model::ErrorReason::ExitCode(0)),
        },
        output: Some("failure remains error".into()),
        data: Some(json!({"exit_code":0})),
        attachments: vec![],
    };
    let source = Source::record("block:test:earlier:call#7");
    let mut evidence = ProjectionEvidence::default();
    evidence.tool_pairs.insert(
        0,
        ToolPair {
            emit_at: "block:test:session:message#0".into(),
            call: SourceBlock {
                source: source.clone(),
                block: call,
            },
            result: Some(SourceBlock {
                source: Source::record("block:test:session:message#0"),
                block: result.clone(),
            }),
            running: false,
            duration_ms: ValueState::default(),
            error_line: Some("first reported failure".into()),
            child: None,
            edit: None,
        },
    );
    let options = ReadOptions {
        tools: Some(Expansion::Full),
        ..Default::default()
    };
    let items = layout(message(vec![result]), &options, &evidence);
    let ItemKind::Call { call } = &items[1].kind else {
        panic!("call")
    };
    assert_eq!(items[1].source, source);
    assert_eq!(call.outcome, ValueState::known("error".into()));
    assert_eq!(call.exit, Some(0));
    assert_eq!(call.duration_ms, ValueState::known(700));
    assert_eq!(call.output.as_deref(), Some("failure remains error"));
    assert!(matches!(&items[2].kind,ItemKind::Other {data,..} if data==&json!({"exit_code":0})));
}
#[test]
fn unknown_call_and_empty_output_are_distinct() {
    let options = ReadOptions {
        tools: Some(Expansion::Full),
        ..Default::default()
    };
    let result = Block::ToolResult {
        reported_duration_ms: None,
        call_id: Some("orphan".into()),
        outcome: polylogue_model::ToolOutcome::Unknown {
            reason: polylogue_model::UnknownReason::StatusFieldUnreadable,
        },
        output: Some(String::new()),
        data: None,
        attachments: vec![],
    };
    let items = layout(
        message(vec![result]),
        &options,
        &ProjectionEvidence::default(),
    );
    let ItemKind::Call { call } = &items[1].kind else {
        panic!()
    };
    assert!(matches!(call.outcome, ValueState::Unknown { .. }));
    assert_eq!(call.output.as_deref(), Some(""));
    assert!(matches!(call.tool, ValueState::Unknown { .. }));
}
#[test]
fn variants_change_layout_only_and_stale_is_opt_in() {
    let original = message(vec![Block::Text {
        text: "original".into(),
    }]);
    let mut evidence = ProjectionEvidence::default();
    evidence
        .variants
        .insert(
            polylogue_model::Reference::parse("block:test:session:message#0").unwrap(),
            VariantText {
                mark: VariantMark {
                    reference: "variant:test".into(),
                    key: "en".into(),
                    original: "block:test:session:message#0".into(),
                    stale: true,
                    show_original: None,
                },
                text: "variant".into(),
                relations: vec![polylogue_model::VariantRelation::Translates],
                alignment: vec![],
            },
        )
        .unwrap();
    let options = ReadOptions::default();
    let projected = project_record(original.clone(), &options, &evidence)
        .unwrap()
        .unwrap();
    assert_eq!(projected, original);
    let items = layout_record(&projected, &options, &evidence).unwrap();
    assert!(
        matches!(&items[1],ReadPart::Item(ReadItem{kind:ItemKind::Text{text},..}) if text=="original")
    );
    let items = layout_record(
        &projected,
        &ReadOptions {
            variants: vec!["en".into()],
            ..options
        },
        &evidence,
    )
    .unwrap();
    assert!(
        items.iter().any(|item| matches!(item,ReadPart::Item(ReadItem{kind:ItemKind::Text{text},..}) if text=="variant"))
    );
    assert_eq!(projected, original);
}
#[test]
fn specialized_views_require_typed_evidence_and_never_fall_back_to_transcript() {
    let record = message(vec![Block::Text {
        text: "not a summary".into(),
    }]);
    for view in [View::Summary, View::Links, View::Changes] {
        let parts = layout_record(
            &record,
            &ReadOptions {
                view,
                ..Default::default()
            },
            &ProjectionEvidence::default(),
        )
        .unwrap();
        assert!(!format!("{parts:?}").contains("not a summary"));
    }
    for view in [View::Raw, View::Live] {
        assert!(matches!(
            layout_record(
                &record,
                &ReadOptions {
                    view,
                    ..Default::default()
                },
                &ProjectionEvidence::default()
            ),
            Err(ReadError::MissingEvidence { .. })
        ));
    }
    let evidence = ProjectionEvidence {
        raw: vec![RawRecord {
            reference: "message:test:session:message".into(),
            input: "input:abc".into(),
            media_type: "application/octet-stream".into(),
            bytes: vec![0, 255, 10],
        }],
        ..Default::default()
    };
    let parts = layout_record(
        &record,
        &ReadOptions {
            view: View::Raw,
            ..Default::default()
        },
        &evidence,
    )
    .unwrap();
    assert!(
        matches!(&parts[0],ReadPart::Item(ReadItem{kind:ItemKind::Raw{record},..}) if record.bytes==vec![0,255,10])
    );
}
#[test]
fn value_states_keep_zero_empty_missing_unavailable_and_partial() {
    assert_eq!(crate::encode::value_text(&ValueState::known(0)), "0");
    assert_eq!(crate::encode::value_text(&ValueState::known("")), "");
    for state in [
        ValueState::<String>::default(),
        ValueState::NotRecorded,
        ValueState::Unavailable {
            reason: "grant".into(),
        },
        ValueState::Pending,
        ValueState::Error {
            reason: "provider".into(),
        },
        ValueState::Partial {
            value: "0".into(),
            unknown: 2,
        },
        ValueState::Inferred { value: "0".into() },
        ValueState::Stale {
            value: None,
            reason: "changed".into(),
        },
        ValueState::Inherited {
            value: "source".into(),
        },
    ] {
        let text = crate::encode::value_text(&state);
        assert_ne!(text, "0");
        assert!(html::value(&state).into_string().contains("data-state="));
    }
}
#[test]
fn markdown_fences_are_longer_than_any_stored_run() {
    let fenced = crate::encode::fence("```\n``````\n<details>\n", "json");
    assert!(fenced.starts_with("```````json\n"));
    assert!(fenced.ends_with("```````\n"));
}
#[test]
fn archived_html_and_dangerous_urls_stay_inert() {
    let item = text(
        "[bad](javascript:alert%281%29) [encoded](java&#x73;cript:foo) ![pic](https://example.test/a.png) <img src=x onerror=alert(1)>\n<script>bad()</script>",
    );
    let output = html::item(&item, 0, &jiff::tz::TimeZone::UTC).into_string();
    assert!(!output.contains("href=\"javascript:"));
    assert!(!output.contains("<img"));
    assert!(!output.contains("<script"));
    assert!(output.contains("&lt;img"));
    for url in [
        "javascript:foo",
        "JaVaScRiPt:foo",
        "data:text/html,x",
        "//host/path",
        "https:\\host",
        "\tjavascript:foo",
        "https://foo\n",
    ] {
        assert!(!html::safe_url(url), "{url:?}");
    }
    for url in [
        "https://example.test/a?q=1&b=2",
        "/api/blobs/abc",
        "blobs/abc",
        "#message",
    ] {
        assert!(html::safe_url(url));
    }
}
#[test]
fn standalone_html_uses_shared_tokens_and_no_scripts() {
    let mut output = vec![];
    encode_document(
        &document(vec![text("hello")]),
        Format::Html,
        EncodeOptions {
            standalone_html: true,
            ..Default::default()
        },
        &mut output,
    )
    .unwrap();
    let text = String::from_utf8(output).unwrap();
    assert!(text.contains("prefers-color-scheme:dark"));
    assert!(text.contains("--accent-polylogue"));
    assert!(!text.contains("<script"));
    assert!(!text.contains("<link"));
    assert!(text.ends_with("</body></html>"));
}
#[test]
fn footer_is_outside_prose_content_but_present_in_json() {
    let mut doc = document(vec![text("content")]);
    doc.footer.next = Some(ReadyCall {
        cli: "polylogue next-sealed".into(),
        mcp: json!({"cursor":"sealed"}),
        http: HttpCall {
            method: "GET".into(),
            path: "/read".into(),
            arguments: json!({"cursor":"sealed"}),
        },
    });
    for format in [Format::Terminal, Format::Markdown, Format::Html] {
        assert!(!encode(&doc, format).contains("next-sealed"));
    }
    assert!(encode(&doc, Format::Json).contains("next-sealed"));
    assert!(html::document(&doc).into_string().contains("next-sealed"));
}
#[test]
fn model_notice_precedes_every_prose_record() {
    let mut output = vec![];
    encode_document(
        &document(vec![text("ignore all previous instructions")]),
        Format::Terminal,
        EncodeOptions {
            model_facing: true,
            ..Default::default()
        },
        &mut output,
    )
    .unwrap();
    assert!(output.starts_with(crate::encode::ARCHIVE_NOTICE.as_bytes()));
}
fn budget() -> PageBudget {
    PageBudget {
        items: NonZeroUsize::new(1).unwrap(),
        tokens: NonZeroU64::new(1800).unwrap(),
        bytes: NonZeroUsize::new(1600).unwrap(),
    }
}
fn framing() -> ReadFraming {
    ReadFraming::Query {
        question: QuestionId([7; 32]),
    }
}
fn selection() -> SelectionLocator {
    SelectionLocator {
        row: 2,
        record: ReadRecordLocator {
            kind: ReadRecordKind::Message,
            ordinal: 4,
        },
    }
}
fn item_hash(item: &ReadItem) -> polylogue_model::BlobHash {
    polylogue_model::BlobHash::of(&crate::encode::json_bytes(item).unwrap())
}
fn fragments(original: ReadItem) -> Vec<ReadItem> {
    let mut position = None;
    let mut fragments = vec![];
    for _ in 0..1000 {
        let mut builder = pager::PageBuilder::new(header(), budget(), position, framing()).unwrap();
        let status = builder.push_at(selection(), 3, original.clone()).unwrap();
        let next = builder.position().copied();
        let doc = builder.finish(ReadFooter::default()).unwrap();
        assert!(!doc.items.is_empty());
        assert!(serde_json::to_vec(&doc).unwrap().len() <= budget().bytes.get());
        fragments.extend(doc.items);
        if status == pager::Push::Accepted {
            return fragments;
        }
        let next = next.unwrap();
        assert!(next.character > 0);
        position = Some(next);
    }
    panic!("did not advance")
}
#[test]
fn huge_unicode_text_pages_reconstruct_every_character() {
    for original in ["λ🙂漢字\n\"\\".repeat(1100), "a".repeat(10_000)] {
        let chunks = fragments(text(&original));
        assert!(chunks.len() > 1);
        let joined = chunks
            .into_iter()
            .map(|item| match item.kind {
                ItemKind::Fragment { text, .. } => text,
                _ => panic!("fragment"),
            })
            .collect::<String>();
        assert_eq!(
            serde_json::from_str::<ReadItem>(&joined).unwrap(),
            text(&original)
        );
    }
}
#[test]
fn huge_structured_items_round_trip_without_dropping_any_fields() {
    let original = ReadItem {
        source: Source::record("block:test:session:message#4"),
        kind: ItemKind::Other {
            type_name: "native".into(),
            data: json!({"all":"λ\n\"".repeat(2000),"provider":{"nested":[0,null,false]}}),
        },
    };
    let chunks = fragments(original.clone());
    let json = chunks
        .into_iter()
        .map(|item| match item.kind {
            ItemKind::Fragment { text, .. } => text,
            _ => panic!(),
        })
        .collect::<String>();
    assert_eq!(serde_json::from_str::<ReadItem>(&json).unwrap(), original);
}
#[test]
fn page_cut_uses_canonical_record_position_not_paired_display_source() {
    let mut builder = pager::PageBuilder::new(header(), budget(), None, framing()).unwrap();
    assert_eq!(
        builder.push_at(selection(), 0, text("first")).unwrap(),
        pager::Push::Accepted
    );
    assert_eq!(
        builder.push_at(selection(), 1, text("second")).unwrap(),
        pager::Push::Full
    );
    assert_eq!(
        builder.position(),
        Some(&ItemPlace {
            selection: selection(),
            item: 1,
            character: 0,
            item_hash: item_hash(&text("second"))
        })
    );
}
#[test]
fn missing_or_invalid_cursor_is_an_error_never_a_shifted_page() {
    let mut builder = pager::PageBuilder::new(
        header(),
        budget(),
        Some(ItemPlace {
            selection: SelectionLocator {
                row: 99,
                ..selection()
            },
            item: 3,
            character: 1,
            item_hash: item_hash(&text("text")),
        }),
        framing(),
    )
    .unwrap();
    assert_eq!(
        builder.push_at(selection(), 3, text("text")),
        Err(pager::PageError::InvalidPosition)
    );
}
#[test]
fn lazy_layout_stops_before_visiting_next_block() {
    let record = message(
        (0..1000)
            .map(|_| Block::Text {
                text: "block".repeat(50),
            })
            .collect(),
    );
    let mut visited = 0;
    visit_record(
        &record,
        &ReadOptions::default(),
        &ProjectionEvidence::default(),
        |_| {
            visited += 1;
            false
        },
    )
    .unwrap();
    assert_eq!(visited, 1);
}
struct CountSink {
    bytes: usize,
    max_write: usize,
}
impl Write for CountSink {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.bytes += bytes.len();
        self.max_write = self.max_write.max(bytes.len());
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
#[test]
fn streaming_export_does_not_collect_the_session() {
    let mut out = CountSink {
        bytes: 0,
        max_write: 0,
    };
    let mut encoder = Encoder::new(
        &header(),
        Format::Markdown,
        EncodeOptions::default(),
        &mut out,
    )
    .unwrap();
    for _ in 0..100_000 {
        encoder.push(&text("same bounded record λ")).unwrap();
    }
    encoder.finish(&ReadFooter::default()).unwrap();
    assert!(out.bytes > 2_000_000);
    assert!(out.max_write < 1024);
}

#[test]
fn summary_cards_show_header_facts_without_source_instructions() {
    let mut session = polylogue_model::Session::new(
        SessionId::parse("test:session").unwrap(),
        Default::default(),
    );
    session.instructions = Some("HARNESS_INSTRUCTIONS_NOT_SUMMARY".into());
    let record = Record::Session(polylogue_model::bundle::SessionRecord {
        reference: "session:test:session".into(),
        commit: 1,
        session,
    });
    let items = layout(
        record,
        &ReadOptions {
            view: View::Summary,
            ..Default::default()
        },
        &ProjectionEvidence::default(),
    );
    for format in [Format::Terminal, Format::Markdown, Format::Html] {
        assert!(
            !encode(&document(items.clone()), format).contains("HARNESS_INSTRUCTIONS_NOT_SUMMARY")
        );
    }
}
#[test]
fn html_export_has_no_network_font_dependency() {
    assert!(!html::export_css().contains("/assets/"));
}
#[test]
fn declared_tool_pairs_refuse_conflicting_ids() {
    let call = Block::ToolCall {
        written_text: None,
        call_id: Some("a".into()),
        tool: None,
        action: Default::default(),
        input: json!({}),
        command: None,
        path: None,
        run_by_script: false,
    };
    let result = Block::ToolResult {
        reported_duration_ms: None,
        call_id: Some("b".into()),
        outcome: polylogue_model::ToolOutcome::Ok,
        output: None,
        data: None,
        attachments: vec![],
    };
    let pair = ToolPair {
        emit_at: "block:test#0".into(),
        call: SourceBlock {
            source: Source::record("block:test#0"),
            block: call,
        },
        result: Some(SourceBlock {
            source: Source::record("block:test#1"),
            block: result,
        }),
        running: false,
        duration_ms: ValueState::default(),
        error_line: None,
        child: None,
        edit: None,
    };
    assert!(pair.validate().is_err());
}

fn result_output(output: String, options: ReadOptions) -> (Record, Box<CallItem>) {
    let record = message(vec![Block::ToolResult {
        reported_duration_ms: None,
        call_id: Some("orphan".into()),
        outcome: polylogue_model::ToolOutcome::Ok,
        output: Some(output),
        data: None,
        attachments: vec![],
    }]);
    let items = layout(record.clone(), &options, &ProjectionEvidence::default());
    let ItemKind::Call { call } = items[1].kind.clone() else {
        panic!()
    };
    (record, call)
}
#[test]
fn expanded_tool_output_has_exact_omission_counts_and_zero_means_all() {
    let original = (0..301)
        .map(|n| format!("line {n} λ\n"))
        .collect::<String>();
    let options = ReadOptions {
        tools: Some(Expansion::Full),
        ..Default::default()
    };
    let (record, call) = result_output(original.clone(), options.clone());
    let excerpt = call.output_excerpt.unwrap();
    assert!(matches!(
        &excerpt.parts[1],
        OutputPart::Omission { lines: 101, .. }
    ));
    assert!(matches!(&excerpt.parts[0],OutputPart::Text {text,..} if text.starts_with("line 0")));
    assert!(matches!(&excerpt.parts[2],OutputPart::Text {text,..} if text.starts_with("line 201")));
    assert_eq!(
        excerpt
            .parts
            .iter()
            .map(|part| match part {
                OutputPart::Text { text, .. } => text.chars().count() as u64,
                OutputPart::Omission { characters, .. } => *characters,
            })
            .sum::<u64>(),
        original.chars().count() as u64
    );
    assert!(call.output.is_none());
    let Record::Message(row) = record else {
        panic!()
    };
    assert!(
        matches!(&row.message.blocks[0],Block::ToolResult{output:Some(output),..} if output==&original)
    );
    let (_, call) = result_output(
        original.clone(),
        ReadOptions {
            output_lines: 0,
            ..options
        },
    );
    assert_eq!(call.output.as_deref(), Some(original.as_str()));
    assert!(call.output_excerpt.is_none());
}
#[test]
fn a_single_long_output_line_is_shortened_at_unicode_boundaries() {
    let original = "λ🙂".repeat(5000);
    let (_, call) = result_output(
        original.clone(),
        ReadOptions {
            tools: Some(Expansion::Full),
            ..Default::default()
        },
    );
    let excerpt = call.output_excerpt.unwrap();
    assert!(matches!(&excerpt.parts[0],OutputPart::Text {text,..} if text.chars().count()==2000));
    assert!(matches!(&excerpt.parts[2],OutputPart::Text {text,..} if text.chars().count()==2000));
    assert!(matches!(
        &excerpt.parts[1],
        OutputPart::Omission {
            characters: 6000,
            lines: 0,
            ..
        }
    ));
}
#[test]
fn multiple_result_attachments_keep_individual_availability_and_source() {
    let attachments = vec![
        polylogue_model::Attachment {
            name: Some("first".into()),
            ..Default::default()
        },
        polylogue_model::Attachment {
            name: Some("second".into()),
            ..Default::default()
        },
    ];
    let record = message(vec![Block::ToolResult {
        reported_duration_ms: None,
        call_id: None,
        outcome: polylogue_model::ToolOutcome::Ok,
        output: None,
        data: None,
        attachments,
    }]);
    let mut evidence = ProjectionEvidence::default();
    evidence.attachments.insert(
        "block:test:session:message#0".into(),
        std::collections::BTreeMap::from([(
            0,
            vec![
                AttachmentEvidence {
                    availability: ValueState::known("present".into()),
                    ..Default::default()
                },
                AttachmentEvidence {
                    availability: ValueState::Unavailable {
                        reason: "source missing".into(),
                    },
                    ..Default::default()
                },
            ],
        )]),
    );
    let items = layout(record, &ReadOptions::default(), &evidence);
    assert!(
        matches!(&items[2].kind,ItemKind::Attachment{availability:ValueState::Value{value},..} if value=="present")
    );
    assert!(
        matches!(&items[3].kind,ItemKind::Attachment{availability:ValueState::Unavailable{reason},..} if reason=="source missing")
    );
}
#[test]
fn inline_images_need_exact_hash_safe_raster_type_and_size() {
    let bytes = vec![137, 80, 78, 71];
    let mut attachment = polylogue_model::Attachment {
        name: Some("sample".into()),
        media_type: Some("image/png".into()),
        blob: Some(polylogue_model::BlobHash::of(&bytes)),
        ..Default::default()
    };
    let make = |attachment: polylogue_model::Attachment| ReadItem {
        source: Source::record("block:test#0"),
        kind: ItemKind::Attachment {
            attachment: Box::new(attachment),
            supplied_by: ValueState::default(),
            availability: ValueState::known("present".into()),
            link: Some("blobs/hash".into()),
            inline_bytes: Some(bytes.clone()),
            verified_hash: None,
            open: None,
        },
    };
    assert!(
        html::item(&make(attachment.clone()), 100, &jiff::tz::TimeZone::UTC)
            .into_string()
            .contains("data:image/png;base64,")
    );
    assert!(
        !html::item(&make(attachment.clone()), 3, &jiff::tz::TimeZone::UTC)
            .into_string()
            .contains("<img")
    );
    assert!(
        !html::item(&make(attachment.clone()), 0, &jiff::tz::TimeZone::UTC)
            .into_string()
            .contains("<img")
    );
    attachment.media_type = Some("image/svg+xml".into());
    assert!(
        !html::item(&make(attachment.clone()), 100, &jiff::tz::TimeZone::UTC)
            .into_string()
            .contains("<img")
    );
    attachment.media_type = Some("text/html".into());
    assert!(
        !html::item(&make(attachment.clone()), 100, &jiff::tz::TimeZone::UTC)
            .into_string()
            .contains("<img")
    );
    attachment.media_type = Some("image/png".into());
    attachment.blob = Some(polylogue_model::BlobHash::of(b"different"));
    assert!(
        !html::item(&make(attachment), 100, &jiff::tz::TimeZone::UTC)
            .into_string()
            .contains("<img")
    );
}

#[test]
fn hand_written_format_goldens_agree_on_the_typed_document() {
    let mut doc = document(vec![
        text("# kept <text>\nline λ"),
        ReadItem {
            source: text("").source,
            kind: ItemKind::Thinking {
                words: ValueState::known(0),
                content: ThinkingContent::Collapsed,
            },
        },
    ]);
    doc.header.target = "message:test:session:message".into();
    doc.header.title = ValueState::known("Example".into());
    doc.header.kind = HeaderKind::Record {
        record_type: "message".into(),
    };
    assert_eq!(
        encode(&doc, Format::Terminal),
        include_str!("../tests/goldens/simple.terminal")
    );
    assert_eq!(
        encode(&doc, Format::Markdown),
        include_str!("../tests/goldens/simple.markdown")
    );
    assert_eq!(
        encode(&doc, Format::Html),
        include_str!("../tests/goldens/simple.html").trim_end()
    );
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&encode(&doc, Format::Json)).unwrap(),
        serde_json::from_str::<serde_json::Value>(include_str!("../tests/goldens/simple.json"))
            .unwrap()
    );
}

#[test]
fn every_encoder_exposes_controls_without_losing_json_source_values() {
    let source = "\x1b]52;c;data\x07\u{9b}2J\u{202e}x\r\u{2066}";
    let doc = document(vec![text(source)]);
    for format in [
        Format::Terminal,
        Format::Markdown,
        Format::Html,
        Format::Json,
        Format::Jsonl,
    ] {
        let output = encode(&doc, format);
        assert!(
            !output
                .chars()
                .any(|c| c.is_control() && c != '\n' && c != '\t'),
            "{format:?}"
        );
        assert!(!output.contains('\u{202e}'));
        assert!(!output.contains('\u{2066}'));
    }
    assert_eq!(
        serde_json::from_str::<ReadDocument>(&encode(&doc, Format::Json)).unwrap(),
        doc
    );
}
#[test]
fn document_items_keep_a_runnable_document_reference_and_separate_original_position() {
    let mut doc = polylogue_model::Document::new(
        polylogue_model::DocumentId::parse("test:document").unwrap(),
        polylogue_model::DocumentPurpose::Transcript,
    );
    doc.blocks = vec![
        Block::Text { text: "one".into() },
        Block::Text { text: "two".into() },
    ];
    let record = Record::Document(polylogue_model::bundle::DocumentRecord {
        reference: "document:test:document".into(),
        commit: 1,
        document: doc,
    });
    let items = layout(
        record,
        &ReadOptions::default(),
        &ProjectionEvidence::default(),
    );
    assert_eq!(items[1].source.reference, "document:test:document");
    assert_eq!(items[1].source.block, Some(1));
}
struct BrokenPipe {
    writes: usize,
}
impl Write for BrokenPipe {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.writes += 1;
        if self.writes == 1 {
            return Ok(bytes.len());
        }
        Err(io::ErrorKind::BrokenPipe.into())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
#[test]
fn json_encoding_preserves_the_output_error_kind() {
    let mut sink = BrokenPipe { writes: 0 };
    let error = Encoder::new(&header(), Format::Json, EncodeOptions::default(), &mut sink)
        .err()
        .unwrap();
    assert_eq!(error.kind(), io::ErrorKind::BrokenPipe);
}

#[test]
fn each_retained_long_tool_line_has_its_own_explicit_omission() {
    let original = format!("{}\nshort middle\n{}\n", "a".repeat(5000), "λ".repeat(5000));
    let (_, call) = result_output(
        original,
        ReadOptions {
            tools: Some(Expansion::Full),
            ..Default::default()
        },
    );
    let excerpt = call.output_excerpt.unwrap();
    assert_eq!(
        excerpt
            .parts
            .iter()
            .filter(|part| matches!(part, OutputPart::Omission { lines: 0, .. }))
            .count(),
        2
    );
    assert!(
        excerpt
            .parts
            .iter()
            .any(|part| matches!(part,OutputPart::Text {text,..} if text.contains("short middle")))
    );
}

#[test]
fn empty_thinking_runs_compact_without_reindexing_later_text() {
    let record = message(vec![
        Block::Thinking {
            text: None,
            signature: None,
            redacted: false,
        },
        Block::Thinking {
            text: Some(String::new()),
            signature: None,
            redacted: false,
        },
        Block::Text {
            text: "after".into(),
        },
    ]);
    let items = layout(
        record,
        &ReadOptions::default(),
        &ProjectionEvidence::default(),
    );
    assert_eq!(items.len(), 3);
    assert!(matches!(
        &items[1].kind,
        ItemKind::Thinking {
            content: ThinkingContent::Absent { blocks: 2 },
            ..
        }
    ));
    assert_eq!(items[2].source.block, Some(2));
}
#[test]
fn orphan_result_names_its_recorded_call_id_in_every_prose_format() {
    let (_, call) = result_output("result".into(), ReadOptions::default());
    assert_eq!(call.call_id.as_deref(), Some("orphan"));
    let doc = document(vec![ReadItem {
        source: Source::record("block:test#0"),
        kind: ItemKind::Call { call },
    }]);
    for format in [Format::Terminal, Format::Markdown, Format::Html] {
        assert!(encode(&doc, format).contains("result for unknown call orphan"));
    }
}
#[test]
fn budgeted_pages_name_the_canonical_token_estimator() {
    let mut builder = pager::PageBuilder::new(header(), budget(), None, framing()).unwrap();
    builder.push_at(selection(), 0, text("item")).unwrap();
    let doc = builder.finish(ReadFooter::default()).unwrap();
    assert_eq!(doc.footer.estimator.as_deref(), Some(TOKEN_ESTIMATOR));
}

#[test]
fn a_recorded_call_with_unknown_tool_name_is_not_an_orphan_result() {
    let call = Block::ToolCall {
        written_text: None,
        call_id: Some("known-call".into()),
        tool: None,
        action: Default::default(),
        input: json!({}),
        command: None,
        path: None,
        run_by_script: false,
    };
    let items = layout(
        message(vec![call]),
        &ReadOptions::default(),
        &ProjectionEvidence::default(),
    );
    let ItemKind::Call { call } = &items[1].kind else {
        panic!()
    };
    assert!(!call.orphan);
    assert_eq!(call.call_id.as_deref(), Some("known-call"));
    assert!(!crate::encode::call_label(call).contains("unknown call"));
}

#[test]
fn document_attachments_at_different_blocks_do_not_share_availability() {
    let mut document = polylogue_model::Document::new(
        polylogue_model::DocumentId::parse("test:document").unwrap(),
        polylogue_model::DocumentPurpose::Transcript,
    );
    document.blocks = vec![
        Block::File(Default::default()),
        Block::File(Default::default()),
    ];
    let record = Record::Document(polylogue_model::bundle::DocumentRecord {
        reference: "document:test:document".into(),
        commit: 1,
        document,
    });
    let mut evidence = ProjectionEvidence::default();
    evidence.attachments.insert(
        "document:test:document".into(),
        std::collections::BTreeMap::from([
            (
                0,
                vec![AttachmentEvidence {
                    availability: ValueState::known("present".into()),
                    ..Default::default()
                }],
            ),
            (
                1,
                vec![AttachmentEvidence {
                    availability: ValueState::Unavailable {
                        reason: "missing".into(),
                    },
                    ..Default::default()
                }],
            ),
        ]),
    );
    let items = layout(record, &ReadOptions::default(), &evidence);
    assert!(
        matches!(&items[0].kind,ItemKind::Attachment{availability:ValueState::Value{value},..} if value=="present")
    );
    assert!(
        matches!(&items[1].kind,ItemKind::Attachment{availability:ValueState::Unavailable{reason},..} if reason=="missing")
    );
    assert_eq!(items[0].source.reference, items[1].source.reference);
    assert_ne!(items[0].source.block, items[1].source.block);
}

#[test]
fn format_flags_and_public_json_schema_use_one_closed_vocabulary() {
    let schema = serde_json::to_value(schemars::schema_for!(Format)).unwrap();
    assert_eq!(schema["enum"], serde_json::json!(Format::NAMES));
    for name in Format::NAMES {
        let format = Format::parse(name).unwrap();
        assert_eq!(format.to_string(), *name);
        assert_eq!(
            serde_json::to_value(format).unwrap(),
            serde_json::json!(name)
        );
        assert_eq!(
            serde_json::from_value::<Format>(serde_json::json!(name)).unwrap(),
            format
        );
    }
    assert!(Format::parse("md").is_none());
    assert!(serde_json::from_value::<Format>(serde_json::json!("md")).is_err());
}

#[test]
fn encoded_control_expansion_is_included_in_the_page_byte_budget() {
    let original = text(&"\u{9b}\u{202e}\"\\".repeat(4000));
    let mut position = None;
    let mut restored = String::new();
    for _ in 0..1000 {
        let mut builder = pager::PageBuilder::new(header(), budget(), position, framing()).unwrap();
        let state = builder.push_at(selection(), 0, original.clone()).unwrap();
        let next = builder.position().copied();
        let document = builder.finish(ReadFooter::default()).unwrap();
        let encoded = encode(&document, Format::Json);
        assert!(encoded.len() <= budget().bytes.get());
        for item in document.items {
            match item.kind {
                ItemKind::Fragment { text, .. } | ItemKind::Text { text } => {
                    restored.push_str(&text)
                }
                _ => panic!(),
            }
        }
        if state == pager::Push::Accepted {
            break;
        }
        position = next;
    }
    assert_eq!(
        serde_json::from_str::<ReadItem>(&restored).unwrap(),
        original
    );
}
#[test]
fn variant_marks_only_describe_visible_substitutions() {
    let record = message(vec![Block::Text {
        text: "visible".into(),
    }]);
    let mut evidence = ProjectionEvidence::default();
    evidence
        .variants
        .insert(
            polylogue_model::Reference::parse("block:test:session:message#7").unwrap(),
            VariantText {
                mark: VariantMark {
                    reference: "variant:test".into(),
                    key: "en".into(),
                    original: "block:test:session:message#7".into(),
                    stale: false,
                    show_original: None,
                },
                text: "not selected".into(),
                relations: vec![polylogue_model::VariantRelation::Translates],
                alignment: vec![],
            },
        )
        .unwrap();
    let items = layout(record, &ReadOptions::default(), &evidence);
    assert!(matches!(&items[0].kind,ItemKind::MessageHeader{header} if header.variant.is_none()));
}
#[test]
fn paired_off_page_result_attachments_use_their_own_original_ordinal() {
    let call = Block::ToolCall {
        written_text: None,
        call_id: Some("c".into()),
        tool: Some("capture".into()),
        action: Default::default(),
        input: json!({}),
        command: None,
        path: None,
        run_by_script: false,
    };
    let result = Block::ToolResult {
        call_id: Some("c".into()),
        reported_duration_ms: None,
        outcome: polylogue_model::ToolOutcome::Ok,
        output: None,
        data: None,
        attachments: vec![Default::default()],
    };
    let call_source = Source::block(
        &MessageId::parse("test:session:message").unwrap(),
        &SessionId::parse("test:session").unwrap(),
        0,
        false,
    );
    let result_source = Source::block(
        &MessageId::parse("test:offpage:result").unwrap(),
        &SessionId::parse("test:offpage").unwrap(),
        7,
        false,
    );
    let mut evidence = ProjectionEvidence::default();
    evidence.tool_pairs.insert(
        0,
        ToolPair {
            emit_at: call_source.reference.clone(),
            call: SourceBlock {
                source: call_source,
                block: call.clone(),
            },
            result: Some(SourceBlock {
                source: result_source.clone(),
                block: result,
            }),
            running: false,
            duration_ms: Default::default(),
            error_line: None,
            child: None,
            edit: None,
        },
    );
    evidence.attachments.insert(
        result_source.reference.clone(),
        std::collections::BTreeMap::from([
            (
                0,
                vec![AttachmentEvidence {
                    availability: ValueState::Unavailable {
                        reason: "wrong ordinal".into(),
                    },
                    ..Default::default()
                }],
            ),
            (
                7,
                vec![AttachmentEvidence {
                    availability: ValueState::known("present".into()),
                    ..Default::default()
                }],
            ),
        ]),
    );
    let items = layout(message(vec![call]), &ReadOptions::default(), &evidence);
    assert!(
        matches!(&items[2],ReadItem{source,kind:ItemKind::Attachment{availability:ValueState::Value{value},..}} if source==&result_source && value=="present")
    );
}

#[test]
fn one_token_pages_and_accessible_empty_text_always_make_progress() {
    let mut limits = budget();
    limits.tokens = NonZeroU64::new(1).unwrap();
    let mut builder = pager::PageBuilder::new(header(), limits, None, framing()).unwrap();
    assert_eq!(
        builder.push_at(selection(), 0, text("λ🙂more")).unwrap(),
        pager::Push::Full
    );
    let position = builder.position().unwrap();
    assert!(position.character > 0);
    let document = builder.finish(ReadFooter::default()).unwrap();
    assert!(!document.items.is_empty());
    let mut builder = pager::PageBuilder::new(header(), limits, None, framing()).unwrap();
    assert_eq!(
        builder.push_at(selection(), 0, text("")).unwrap(),
        pager::Push::Accepted
    );
    assert_eq!(
        builder.finish(ReadFooter::default()).unwrap().items,
        vec![text("")]
    );
}
fn large_footer(position: Option<&ItemPlace>) -> Result<ReadFooter, pager::PageError> {
    let mut footer = ReadFooter {
        referenced_by: vec![ReferencedBy {
            reference: "x".repeat(600),
            role: "evidence".into(),
            open: None,
        }],
        ..Default::default()
    };
    footer.next = position.map(|position| ReadyCall {
        cli: format!(
            "read {} {} {}",
            position.selection.row, position.item, position.character
        ),
        mcp: json!({"position":position}),
        http: HttpCall {
            method: "GET".into(),
            path: "/api/v1/read".into(),
            arguments: json!({"position":position}),
        },
    });
    Ok(footer)
}
#[test]
fn final_footer_size_can_trim_items_and_regenerate_the_exact_boundary() {
    let mut limits = budget();
    limits.items = NonZeroUsize::new(3).unwrap();
    let mut builder = pager::PageBuilder::new(header(), limits, None, framing()).unwrap();
    for n in 0..3 {
        assert_eq!(
            builder
                .push_at(selection(), n, text(&"a".repeat(100)))
                .unwrap(),
            pager::Push::Accepted
        );
    }
    assert_eq!(
        builder.push_at(selection(), 3, text("next")).unwrap(),
        pager::Push::Full
    );
    let mut attempts = 0;
    let mut last = None;
    let doc = builder
        .finish_with(|position| {
            attempts += 1;
            last = position.copied();
            large_footer(position)
        })
        .unwrap();
    assert!(attempts > 1);
    assert!(!doc.items.is_empty());
    assert!(doc.items.len() < 3);
    let position = last.as_ref().unwrap();
    assert_eq!(position.item, doc.items.len() as u64);
    assert_eq!(
        doc.footer.next.as_ref().unwrap().mcp["position"],
        json!(position)
    );
    assert!(encode(&doc, Format::Json).len() <= limits.bytes.get());
}
#[test]
fn final_footer_size_can_shorten_a_single_item_without_skipping_its_middle() {
    let mut limits = budget();
    limits.bytes = NonZeroUsize::new(1800).unwrap();
    limits.items = NonZeroUsize::new(2).unwrap();
    let mut builder = pager::PageBuilder::new(header(), limits, None, framing()).unwrap();
    assert_eq!(
        builder
            .push_at(selection(), 0, text(&"λ".repeat(300)))
            .unwrap(),
        pager::Push::Accepted
    );
    assert_eq!(
        builder
            .push_at(selection(), 1, text(&"n".repeat(600)))
            .unwrap(),
        pager::Push::Full
    );
    let mut last = None;
    let doc = builder
        .finish_with(|position| {
            last = position.copied();
            large_footer(position)
        })
        .unwrap();
    let ItemKind::Fragment {
        text: payload,
        position,
        until,
        total,
        ..
    } = &doc.items[0].kind
    else {
        panic!()
    };
    assert_eq!(position.character, 0);
    assert_eq!(
        *total,
        String::from_utf8(crate::encode::json_bytes(&text(&"λ".repeat(300))).unwrap())
            .unwrap()
            .chars()
            .count() as u64
    );
    assert_eq!(payload.chars().count() as u64, *until);
    assert_eq!(last.as_ref().unwrap().character, *until);
    assert_eq!(last.as_ref().unwrap().item, 0);
    assert!(encode(&doc, Format::Json).len() <= limits.bytes.get());
}

fn text_variant(original: &str, text: &str, alignment: Vec<TextAlignment>) -> VariantText {
    VariantText {
        mark: VariantMark {
            reference: "variant:synthetic@revision".into(),
            key: "en".into(),
            original: original.into(),
            stale: false,
            show_original: None,
        },
        relations: vec![polylogue_model::VariantRelation::Translates],
        text: text.into(),
        alignment,
    }
}
#[test]
fn variants_replace_thinking_and_remote_tool_output_at_their_exact_sources() {
    let call = Block::ToolCall {
        written_text: None,
        call_id: Some("c".into()),
        tool: Some("test".into()),
        action: Default::default(),
        input: json!({}),
        command: None,
        path: None,
        run_by_script: false,
    };
    let original = message(vec![
        Block::Thinking {
            text: Some("original thinking".into()),
            redacted: false,
            signature: Some("opaque".into()),
        },
        call.clone(),
    ]);
    let call_source = Source::block(
        &MessageId::parse("test:session:message").unwrap(),
        &SessionId::parse("test:session").unwrap(),
        1,
        false,
    );
    let result_source = Source::block(
        &MessageId::parse("test:remote:result").unwrap(),
        &SessionId::parse("test:remote").unwrap(),
        7,
        false,
    );
    let mut evidence = ProjectionEvidence::default();
    for (reference, text) in [
        ("block:test:session:message#0", "translated thinking"),
        (result_source.reference.as_str(), "translated result"),
    ] {
        evidence
            .variants
            .insert(
                polylogue_model::Reference::parse(reference).unwrap(),
                text_variant(reference, text, vec![]),
            )
            .unwrap();
    }
    evidence.tool_pairs.insert(
        1,
        ToolPair {
            emit_at: call_source.reference.clone(),
            call: SourceBlock {
                source: call_source,
                block: call,
            },
            result: Some(SourceBlock {
                source: result_source.clone(),
                block: Block::ToolResult {
                    call_id: Some("c".into()),
                    reported_duration_ms: None,
                    outcome: polylogue_model::ToolOutcome::Ok,
                    output: Some("original result".into()),
                    data: None,
                    attachments: vec![],
                },
            }),
            running: false,
            duration_ms: Default::default(),
            error_line: None,
            child: None,
            edit: None,
        },
    );
    let options = ReadOptions {
        thinking: Some(Expansion::Full),
        tools: Some(Expansion::Full),
        ..Default::default()
    };
    assert_eq!(
        project_record(original.clone(), &options, &evidence)
            .unwrap()
            .unwrap(),
        original
    );
    let items = layout(original, &options, &evidence);
    assert!(items.iter().any(|i| matches!(&i.kind, ItemKind::Thinking { content: ThinkingContent::Text { text }, .. } if text == "translated thinking")));
    assert!(items.iter().any(|i| matches!(&i.kind, ItemKind::Call { call } if call.output.as_deref()==Some("translated result") && call.result_source.as_ref()==Some(&result_source))));
    let doc = document(items);
    for format in [
        Format::Terminal,
        Format::Markdown,
        Format::Html,
        Format::Json,
        Format::Jsonl,
    ] {
        let encoded = encode(&doc, format);
        assert!(encoded.contains("translated thinking"));
        assert!(encoded.contains("translated result"));
        assert!(!encoded.contains("original thinking"));
        assert!(!encoded.contains("original result"));
        assert!(encoded.contains("variant"));
    }
}
#[test]
fn document_overlay_uses_original_body_ordinal_and_byte_boundaries() {
    let mut doc = polylogue_model::Document::new(
        polylogue_model::DocumentId::parse("test:document").unwrap(),
        polylogue_model::DocumentPurpose::Transcript,
    );
    doc.blocks = vec![
        Block::Text {
            text: "first".into(),
        },
        Block::Text {
            text: "aλz".into()
        },
    ];
    let source = Source {
        block: Some(1),
        ..Source::record("document:test:document")
    };
    let mut evidence = ProjectionEvidence::default();
    let reference = polylogue_model::Reference::parse("document:test:document").unwrap();
    evidence
        .variants
        .insert(
            reference.clone(),
            text_variant(
                &reference.to_string(),
                "replacement",
                vec![TextAlignment {
                    source: source.clone(),
                    start: 1,
                    end: 3,
                }],
            ),
        )
        .unwrap();
    let items = layout(
        Record::Document(polylogue_model::bundle::DocumentRecord {
            reference: reference.to_string(),
            commit: 1,
            document: doc,
        }),
        &ReadOptions::default(),
        &evidence,
    );
    assert!(matches!(&items[0].kind,ItemKind::Text{text} if text=="first"));
    assert!(
        items.iter().any(|i| i.source == source
            && matches!(&i.kind,ItemKind::Text{text} if text=="areplacementz"))
    );
}

#[test]
fn supplied_attachment_action_survives_layout_and_all_encoders() {
    let source = Source::block(
        &MessageId::parse("test:session:message").unwrap(),
        &SessionId::parse("test:session").unwrap(),
        9,
        false,
    );
    let action = ReadyCall {
        cli: "polylogue read --request '{\"target\":\"attachment:synthetic\"}'".into(),
        mcp: json!({"target":"attachment:synthetic"}),
        http: HttpCall {
            method: "GET".into(),
            path: "/api/v1/read".into(),
            arguments: json!({"operation":"read","arguments":{"target":"attachment:synthetic"},"request_id":"same","deadline":"default"}),
        },
    };
    let mut evidence = ProjectionEvidence::default();
    evidence
        .attachments
        .entry(source.reference.clone())
        .or_default()
        .insert(
            9,
            vec![AttachmentEvidence {
                open: Some(Box::new(action.clone())),
                ..Default::default()
            }],
        );
    let mut items = vec![];
    layout::visit_source_block(
        &SourceBlock {
            source: source.clone(),
            block: Block::File(Default::default()),
        },
        &ReadOptions::default(),
        &evidence,
        |item| {
            items.push(item);
            true
        },
    )
    .unwrap();
    assert!(
        matches!(&items[0].kind,ItemKind::Attachment{open:Some(open),..} if open.as_ref()==&action)
    );
    for format in [
        Format::Terminal,
        Format::Markdown,
        Format::Html,
        Format::Json,
        Format::Jsonl,
    ] {
        let output = encode(&document(items.clone()), format);
        assert!(
            output.contains("attachment:synthetic"),
            "{format}: {output}"
        );
        if format == Format::Html {
            assert!(output.contains("href=\"/api/v1/read?"));
        }
    }
    assert_eq!(
        html::DEFAULT_INLINE_IMAGE_BYTES,
        EncodeOptions::default().inline_image_bytes
    );
    assert!(html::inline_media_type("image/png"));
    assert!(!html::inline_media_type("image/svg+xml"));
    assert!(!html::inline_media_type("text/html"));
}

#[test]
fn a_paired_result_header_marks_only_the_pair_actually_emitted_here() {
    let result = Block::ToolResult {
        call_id: Some("c".into()),
        reported_duration_ms: None,
        outcome: polylogue_model::ToolOutcome::Ok,
        output: Some("original result".into()),
        data: None,
        attachments: vec![],
    };
    let result_source = Source::block(
        &MessageId::parse("test:session:message").unwrap(),
        &SessionId::parse("test:session").unwrap(),
        0,
        false,
    );
    let call_source = Source::block(
        &MessageId::parse("test:other:call").unwrap(),
        &SessionId::parse("test:other").unwrap(),
        3,
        false,
    );
    let mut evidence = ProjectionEvidence::default();
    evidence
        .variants
        .insert(
            polylogue_model::Reference::parse(&result_source.reference).unwrap(),
            text_variant(&result_source.reference, "replacement result", vec![]),
        )
        .unwrap();
    evidence.tool_pairs.insert(
        0,
        ToolPair {
            emit_at: call_source.reference.clone(),
            call: SourceBlock {
                source: call_source,
                block: Block::ToolCall {
                    written_text: None,
                    call_id: Some("c".into()),
                    tool: Some("test".into()),
                    action: Default::default(),
                    input: json!({}),
                    command: None,
                    path: None,
                    run_by_script: false,
                },
            },
            result: Some(SourceBlock {
                source: result_source.clone(),
                block: result.clone(),
            }),
            running: false,
            duration_ms: Default::default(),
            error_line: None,
            child: None,
            edit: None,
        },
    );
    let options = ReadOptions {
        tools: Some(Expansion::Full),
        ..Default::default()
    };
    let hidden = layout(message(vec![result.clone()]), &options, &evidence);
    assert_eq!(hidden.len(), 1);
    assert!(matches!(&hidden[0].kind,ItemKind::MessageHeader{header} if header.variant.is_none()));
    evidence.tool_pairs.get_mut(&0).unwrap().emit_at = result_source.reference;
    let shown = layout(message(vec![result]), &options, &evidence);
    assert!(matches!(&shown[0].kind,ItemKind::MessageHeader{header} if header.variant.is_some()));
    assert!(shown.iter().any(|item|matches!(&item.kind,ItemKind::Call{call} if call.output.as_deref()==Some("replacement result"))));
}

#[test]
fn a_separator_only_span_keeps_its_exact_reference_and_emits_text_once() {
    let original = message(vec![
        Block::Text { text: "a".into() },
        Block::Text { text: "b".into() },
    ]);
    let span = "span:message:test:session:message#b=1-3";
    let source = Source::block(
        &MessageId::parse("test:session:message").unwrap(),
        &SessionId::parse("test:session").unwrap(),
        0,
        false,
    );
    let mut evidence = ProjectionEvidence::default();
    evidence
        .variants
        .insert(
            polylogue_model::Reference::parse(span).unwrap(),
            text_variant(
                span,
                "RENDITIONSEPARATOR",
                vec![TextAlignment {
                    source,
                    start: 1,
                    end: 1,
                }],
            ),
        )
        .unwrap();
    let projected = project_record(original.clone(), &ReadOptions::default(), &evidence)
        .unwrap()
        .unwrap();
    assert_eq!(projected, original);
    let items = layout(projected, &ReadOptions::default(), &evidence);
    assert!(
        matches!(&items[0].kind,ItemKind::MessageHeader{header} if header.variant.as_ref().is_some_and(|variant|variant.original==span))
    );
    let texts = items
        .iter()
        .filter_map(|item| {
            if let ItemKind::Text { text } = &item.kind {
                Some(text.as_str())
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    assert_eq!(texts, ["aRENDITIONSEPARATOR", "b"]);
    for format in [
        Format::Terminal,
        Format::Markdown,
        Format::Html,
        Format::Json,
        Format::Jsonl,
    ] {
        assert_eq!(
            encode(&document(items.clone()), format)
                .matches("RENDITIONSEPARATOR")
                .count(),
            1
        );
    }
}

#[test]
fn accessible_empty_record_renditions_have_root_anchors_without_canonical_blocks() {
    for original in [
        message(vec![]),
        Record::Document(polylogue_model::bundle::DocumentRecord {
            reference: "document:test:document".into(),
            commit: 1,
            document: polylogue_model::Document::new(
                polylogue_model::DocumentId::parse("test:document").unwrap(),
                polylogue_model::DocumentPurpose::Transcript,
            ),
        }),
    ] {
        let reference = original.reference().unwrap().to_owned();
        let key = if reference.starts_with("message:") {
            format!("span:{reference}#b=0-0")
        } else {
            reference.clone()
        };
        let mut variant = text_variant(
            &key,
            "expanded empty source",
            vec![TextAlignment {
                source: Source::record(&reference),
                start: 0,
                end: 0,
            }],
        );
        variant.relations = vec![polylogue_model::VariantRelation::Expands];
        let mut evidence = ProjectionEvidence::default();
        evidence
            .variants
            .insert(polylogue_model::Reference::parse(&key).unwrap(), variant)
            .unwrap();
        let projected = project_record(original.clone(), &ReadOptions::default(), &evidence)
            .unwrap()
            .unwrap();
        assert_eq!(projected, original);
        let items = layout(projected, &ReadOptions::default(), &evidence);
        assert!(items.iter().any(|item| item.source.reference == reference
            && item.source.block.is_none()
            && matches!(&item.kind,ItemKind::Text{text} if text=="expanded empty source")));
        for format in [
            Format::Terminal,
            Format::Markdown,
            Format::Html,
            Format::Json,
            Format::Jsonl,
        ] {
            assert_eq!(
                encode(&document(items.clone()), format)
                    .matches("expanded empty source")
                    .count(),
                1
            );
        }
    }
}

#[test]
fn verified_bound_rasters_inline_without_overriding_canonical_attachment_hashes() {
    let bytes = vec![137, 80, 78, 71];
    let hash = polylogue_model::BlobHash::of(&bytes);
    let attachment = polylogue_model::Attachment {
        source_id: Some("bound:source".into()),
        media_type: Some("image/png".into()),
        ..Default::default()
    };
    let source = Source::block(
        &MessageId::parse("test:session:message").unwrap(),
        &SessionId::parse("test:session").unwrap(),
        0,
        false,
    );
    let mut evidence = ProjectionEvidence::default();
    evidence
        .attachments
        .entry(source.reference.clone())
        .or_default()
        .insert(
            0,
            vec![AttachmentEvidence {
                inline_bytes: Some(bytes.clone()),
                verified_hash: Some(hash),
                availability: ValueState::known("present".into()),
                ..Default::default()
            }],
        );
    let block = SourceBlock {
        source: source.clone(),
        block: Block::Image(attachment.clone()),
    };
    let render = |block: &SourceBlock, evidence: &ProjectionEvidence| {
        let mut items = vec![];
        let result = visit_source_block(block, &ReadOptions::default(), evidence, |item| {
            items.push(item);
            true
        });
        result.map(|()| items)
    };
    let items = render(&block, &evidence).unwrap();
    assert!(
        matches!(&items[0].kind,ItemKind::Attachment{attachment,verified_hash:Some(value),..} if attachment.blob.is_none() && *value==hash)
    );
    assert!(
        html::item(&items[0], 100, &jiff::tz::TimeZone::UTC)
            .into_string()
            .contains("data:image/png;base64,")
    );
    assert_eq!(block.block, Block::Image(attachment));
    let mut svg = items[0].clone();
    if let ItemKind::Attachment { attachment, .. } = &mut svg.kind {
        attachment.media_type = Some("image/svg+xml".into());
    }
    assert!(
        !html::item(&svg, 100, &jiff::tz::TimeZone::UTC)
            .into_string()
            .contains("data:image/svg")
    );
    let entry = evidence
        .attachments
        .get_mut(&source.reference)
        .unwrap()
        .get_mut(&0)
        .unwrap()
        .first_mut()
        .unwrap();
    entry.verified_hash = Some(polylogue_model::BlobHash::of(b"wrong"));
    assert!(render(&block, &evidence).is_err());
    let entry = evidence
        .attachments
        .get_mut(&source.reference)
        .unwrap()
        .get_mut(&0)
        .unwrap()
        .first_mut()
        .unwrap();
    entry.verified_hash = None;
    assert!(render(&block, &evidence).is_err());
}

#[test]
fn unbounded_native_source_is_inside_lossless_bounded_frames() {
    let original = ReadItem {
        source: Source {
            reference: format!("message:{}", "λ/native:".repeat(4000)),
            message: Some(MessageId::parse(&"natural:".repeat(3000)).unwrap()),
            block: Some(998),
            source_session: Some(SessionId::parse(&"session:".repeat(3000)).unwrap()),
            inherited: true,
        },
        kind: ItemKind::Other {
            type_name: "provider_record".into(),
            data: json!({"time":null,"provider":{"exact":["\u{9b}",false,7]}}),
        },
    };
    let hash = item_hash(&original);
    let mut encoded = String::new();
    let mut offset = 0;
    for chunk in fragments(original.clone()) {
        assert_eq!(chunk.source, framing().source());
        let ItemKind::Fragment {
            text,
            position,
            until,
            total,
        } = chunk.kind
        else {
            panic!()
        };
        assert_eq!(position.item_hash, hash);
        assert_eq!(position.selection, selection());
        assert_eq!(position.character, offset);
        assert!(until > offset && until <= total);
        offset = until;
        encoded.push_str(&text);
    }
    assert_eq!(polylogue_model::BlobHash::of(encoded.as_bytes()), hash);
    assert_eq!(
        serde_json::from_str::<ReadItem>(&encoded).unwrap(),
        original
    );
}

#[test]
fn resumed_item_digest_rejects_late_evidence_changes_before_emitting_a_fragment() {
    let original = text(&"visible".repeat(1000));
    let mut first = pager::PageBuilder::new(header(), budget(), None, framing()).unwrap();
    assert_eq!(
        first.push_at(selection(), 0, original.clone()).unwrap(),
        pager::Push::Full
    );
    let position = *first.position().unwrap();
    let mut changed =
        pager::PageBuilder::new(header(), budget(), Some(position), framing()).unwrap();
    assert_eq!(
        changed.push_at(
            selection(),
            0,
            text("changed availability or provider fields")
        ),
        Err(pager::PageError::ItemChanged)
    );
    let mut shifted =
        pager::PageBuilder::new(header(), budget(), Some(position), framing()).unwrap();
    assert_eq!(
        shifted.push_at(selection(), 1, original.clone()),
        Err(pager::PageError::InvalidPosition)
    );
    let mut other_record =
        pager::PageBuilder::new(header(), budget(), Some(position), framing()).unwrap();
    let next_record = SelectionLocator {
        record: ReadRecordLocator {
            ordinal: 5,
            ..selection().record
        },
        ..selection()
    };
    assert_eq!(
        other_record.push_at(next_record, 0, original),
        Err(pager::PageError::InvalidPosition)
    );
    let missing = pager::PageBuilder::new(header(), budget(), Some(position), framing()).unwrap();
    assert!(matches!(
        missing.finish(ReadFooter::default()),
        Err(pager::PageError::InvalidPosition)
    ));
}

#[test]
fn prose_encoders_label_json_fragments_as_encoded_transport() {
    let fragment = fragments(text(&"\u{9b}\n".repeat(2000))).remove(0);
    let doc = document(vec![fragment]);
    for format in [Format::Terminal, Format::Markdown, Format::Html] {
        assert!(encode(&doc, format).contains("Encoded ReadItem JSON fragment"));
    }
}

#[test]
fn authored_selection_preserves_the_complete_frozen_request_and_explicit_coverage() {
    use polylogue_model::{
        Author, Reference, Timestamp,
        assertion::{
            AssertionFact, AssertionKind, AssertionLink, LinkRole, NewAssertion, NewAssertionLink,
        },
        authored::{AuthoredEffect, PreparedRequest},
        authorship::{Authorship, Channel},
        bundle::AuthoredRecord,
        user::{UserOutcome, UserRequest, UserWrite},
    };
    let at = Timestamp::from_unix_ms(123).unwrap();
    let target = Reference::parse("session:codex:subject").unwrap();
    let statements: Vec<_> = ["one", "two"]
        .into_iter()
        .map(|text| NewAssertion {
            kind: AssertionKind::Note,
            key: None,
            value: Some(serde_json::json!(format!("entire {text}"))),
            valid_until: None,
            links: vec![NewAssertionLink {
                role: LinkRole::About,
                reference: target.clone(),
            }],
        })
        .collect();
    let request = UserRequest {
        id: "synthetic".into(),
        machine: "test-machine".into(),
        at,
        authorship: Authorship {
            author: Author::Operator,
            via: Channel::Cli,
        },
        write: UserWrite::Assert {
            assertions: statements.clone(),
        },
    };
    let facts = statements
        .into_iter()
        .enumerate()
        .map(|(n, statement)| {
            let id = format!("synthetic.{n}");
            AssertionFact::from_request(
                id.clone(),
                id,
                statement,
                vec![AssertionLink {
                    role: LinkRole::About,
                    reference: target.clone(),
                    seen_hash: None,
                    quote: None,
                }],
                &request,
            )
        })
        .collect();
    let row = AuthoredRecord {
        reference: "request:synthetic".into(),
        commit: 9,
        prepared: PreparedRequest {
            request,
            outcome: UserOutcome {
                id: "synthetic.0".into(),
                repeat: false,
                pending: false,
            },
            effect: AuthoredEffect {
                assertions: facts,
                ..Default::default()
            },
        },
    };
    let record = Record::Authored(Box::new(row.clone()));
    record.validate().unwrap();
    let options = ReadOptions {
        assertions: true,
        ..Default::default()
    };
    let selected = AuthoredSelection {
        selected: polylogue_model::Reference::Delivery("selected".into()),
        coverage: AuthoredCoverage::WholeRequest,
    };
    let evidence = ProjectionEvidence {
        authored: Some(selected.clone()),
        ..Default::default()
    };
    let projected = project_record(record.clone(), &options, &evidence)
        .unwrap()
        .unwrap();
    assert_eq!(projected, record);
    let items = layout(projected, &options, &evidence);
    let ItemKind::Other { type_name, data } = &items[0].kind else {
        panic!()
    };
    assert_eq!(type_name, "authored");
    assert_eq!(
        serde_json::from_value::<AuthoredRecord>(data["record"].clone()).unwrap(),
        row
    );
    assert_eq!(
        serde_json::from_value::<AuthoredSelection>(data["selection"].clone()).unwrap(),
        selected
    );
    assert_eq!(data["selection"]["coverage"], "whole_request");
    let doc = document(items);
    let restored: ReadDocument = serde_json::from_str(&encode(&doc, Format::Json)).unwrap();
    assert_eq!(restored, doc);
}

fn selected_fragment(text: &str, start: u64) -> Record {
    selected_fragment_in(
        polylogue_model::Reference::Message(MessageId::parse("test:session:message").unwrap()),
        text,
        start,
    )
}
fn selected_fragment_in(container: polylogue_model::Reference, text: &str, start: u64) -> Record {
    use polylogue_model::{
        BlobHash, Reference, SpanUnit,
        bundle::{
            TextCoverage, TextFragment, TextFragmentProvenance, TextFragmentRecord,
            TextSourceVersion, TextSpanOrigin,
        },
    };
    let end = start + text.len() as u64;
    let span = Reference::Span {
        container: Box::new(container.clone()),
        unit: SpanUnit::Bytes,
        start,
        end,
    };
    Record::TextFragment(Box::new(TextFragmentRecord {
        reference: span.to_string(),
        commit: 7,
        fragment: TextFragment {
            container: container.clone(),
            source_version: TextSourceVersion::ContentHash {
                record: container,
                hash: BlobHash::of(b"retained whole-source claim"),
            },
            coverage: TextCoverage {
                start,
                end,
                total_bytes: None,
            },
            text: text.into(),
            text_hash: BlobHash::of(text.as_bytes()),
            provenance: TextFragmentProvenance {
                spans: vec![TextSpanOrigin {
                    reference: span,
                    start,
                    end,
                }],
                sources: vec![],
                imports: vec![],
            },
        },
    }))
}
fn known_fragment_origin(kind: polylogue_model::BlockKind, start: u64, end: u64) -> FragmentOrigin {
    FragmentOrigin {
        message: Some(FragmentMessageOrigin {
            role: Role::Assistant,
            prompt_kind: None,
        }),
        bodies: vec![FragmentBodyOrigin { start, end, kind }],
    }
}
#[test]
fn partial_source_claims_do_not_invent_message_roles_or_dialogue_visibility() {
    use polylogue_model::BlockKind;
    let record = selected_fragment("<script>\u{9b}quote</script>", 9);
    record.validate().unwrap();
    let claimed = known_fragment_origin(BlockKind::Text, 9, 42);
    for origin in [
        ValueState::default(),
        ValueState::Unavailable {
            reason: "source unavailable".into(),
        },
        ValueState::Partial {
            value: claimed.clone(),
            unknown: 1,
        },
        ValueState::Inferred {
            value: claimed.clone(),
        },
        ValueState::Stale {
            value: Some(claimed.clone()),
            reason: "source changed".into(),
        },
        ValueState::Inherited { value: claimed },
    ] {
        let evidence = ProjectionEvidence {
            fragment_origin: origin.clone(),
            ..Default::default()
        };
        let projected = project_record(record.clone(), &ReadOptions::default(), &evidence)
            .unwrap()
            .unwrap();
        assert_eq!(projected, record);
        let items = layout(projected, &ReadOptions::default(), &evidence);
        assert_eq!(items.len(), 1);
        assert!(
            matches!(&items[0].kind,ItemKind::TextFragment {origin:actual,..} if actual==&origin)
        );
        let doc = document(items);
        let encoded = encode(&doc, Format::Json);
        assert_eq!(serde_json::from_str::<ReadDocument>(&encoded).unwrap(), doc);
        let parts: Vec<ReadPart> = encode(&doc, Format::Jsonl)
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(parts, doc.clone().into_parts().collect::<Vec<_>>());
        let html = encode(&doc, Format::Html);
        assert!(html.contains("Selected source text"));
        assert!(html.contains("&lt;script&gt;"));
        assert!(!html.contains("<script>"));
        assert!(encode(&doc, Format::Markdown).contains("Source classification:"));
        for view in [View::Dialogue, View::Context] {
            assert!(matches!(
                project_record(
                    record.clone(),
                    &ReadOptions {
                        view,
                        ..Default::default()
                    },
                    &evidence
                ),
                Err(ReadError::MissingEvidence { .. })
            ));
        }
    }
}
#[test]
fn known_fragment_origins_apply_existing_view_and_expansion_rules_without_mutation() {
    use polylogue_model::BlockKind;
    let record = selected_fragment("selected", 12);
    for (kind, option) in [
        (BlockKind::Thinking, "thinking"),
        (BlockKind::ToolResult, "tools"),
    ] {
        let evidence = ProjectionEvidence {
            fragment_origin: ValueState::known(known_fragment_origin(kind, 10, 30)),
            ..Default::default()
        };
        assert!(matches!(
            project_record(record.clone(), &ReadOptions::default(), &evidence),
            Err(ReadError::RequiredExpansion { .. })
        ));
        assert!(matches!(
            project_record(
                record.clone(),
                &ReadOptions {
                    view: View::Dialogue,
                    ..Default::default()
                },
                &evidence
            ),
            Err(ReadError::UnsupportedTarget { .. })
        ));
        let mut full = ReadOptions::default();
        if option == "thinking" {
            full.thinking = Some(Expansion::Full)
        } else {
            full.tools = Some(Expansion::Full)
        };
        assert_eq!(
            project_record(record.clone(), &full, &evidence).unwrap(),
            Some(record.clone())
        );
        let items = layout(record.clone(), &full, &evidence);
        assert!(matches!(&items[0].kind, ItemKind::TextFragment { .. }));
    }
    let text = ProjectionEvidence {
        fragment_origin: ValueState::known(known_fragment_origin(BlockKind::Text, 10, 30)),
        ..Default::default()
    };
    assert_eq!(
        project_record(
            record.clone(),
            &ReadOptions {
                view: View::Dialogue,
                ..Default::default()
            },
            &text
        )
        .unwrap(),
        Some(record.clone())
    );
    let mut harness = text.clone();
    let ValueState::Value { value: origin } = &mut harness.fragment_origin else {
        panic!()
    };
    origin.message.as_mut().unwrap().prompt_kind = Some(polylogue_model::PromptKind::Harness);
    assert!(matches!(
        project_record(record.clone(), &ReadOptions::default(), &harness),
        Err(ReadError::RequiredExpansion { .. })
    ));
    assert_eq!(
        project_record(
            record.clone(),
            &ReadOptions {
                harness: Some(Expansion::Full),
                ..Default::default()
            },
            &harness
        )
        .unwrap(),
        Some(record.clone())
    );
    assert!(matches!(
        project_record(
            record.clone(),
            &ReadOptions {
                view: View::Dialogue,
                ..Default::default()
            },
            &harness
        ),
        Err(ReadError::UnsupportedTarget { .. })
    ));
    let record = selected_fragment_in(
        polylogue_model::Reference::Document(
            polylogue_model::DocumentId::parse("test:document").unwrap(),
        ),
        "selected",
        12,
    );
    let mut document = text;
    let ValueState::Value { value: origin } = &mut document.fragment_origin else {
        panic!()
    };
    origin.message = None;
    assert!(matches!(
        project_record(
            record,
            &ReadOptions {
                view: View::Dialogue,
                ..Default::default()
            },
            &document
        ),
        Err(ReadError::UnsupportedTarget { .. })
    ));
}
#[test]
fn fragment_separator_ranges_require_complete_original_adjacent_body_evidence() {
    use polylogue_model::BlockKind;
    let record = selected_fragment("\n\n", 2);
    let mut origin = FragmentOrigin {
        message: Some(FragmentMessageOrigin {
            role: Role::Assistant,
            prompt_kind: None,
        }),
        bodies: vec![
            FragmentBodyOrigin {
                start: 0,
                end: 2,
                kind: BlockKind::Text,
            },
            FragmentBodyOrigin {
                start: 4,
                end: 9,
                kind: BlockKind::Thinking,
            },
        ],
    };
    let dialogue = ReadOptions {
        view: View::Dialogue,
        ..Default::default()
    };
    assert_eq!(
        project_record(
            record.clone(),
            &dialogue,
            &ProjectionEvidence {
                fragment_origin: ValueState::known(origin.clone()),
                ..Default::default()
            }
        )
        .unwrap(),
        Some(record.clone())
    );
    origin.bodies[1].start = 5;
    assert!(matches!(
        project_record(
            record,
            &dialogue,
            &ProjectionEvidence {
                fragment_origin: ValueState::known(origin),
                ..Default::default()
            }
        ),
        Err(ReadError::InvalidEvidence(_))
    ));
    let empty = selected_fragment_in(
        polylogue_model::Reference::Document(
            polylogue_model::DocumentId::parse("test:empty").unwrap(),
        ),
        "",
        0,
    );
    let bodyless = ProjectionEvidence {
        fragment_origin: ValueState::known(FragmentOrigin {
            message: None,
            bodies: vec![],
        }),
        ..Default::default()
    };
    assert_eq!(
        project_record(empty.clone(), &ReadOptions::default(), &bodyless).unwrap(),
        Some(empty.clone())
    );
    assert!(matches!(
        project_record(empty, &dialogue, &bodyless),
        Err(ReadError::UnsupportedTarget { .. })
    ));
}

fn before_anchor(item: &ReadItem, ordinal: u64, character: u64) -> ItemPlace {
    ItemPlace {
        selection: selection(),
        item: ordinal,
        character,
        item_hash: item_hash(item),
    }
}
#[test]
fn before_unicode_fragments_reconstruct_complete_json_without_overlap_or_gaps() {
    let original = ReadItem {
        source: Source::record(format!("block:test:{}#3", "opaque:λ".repeat(1000))),
        kind: ItemKind::Other {
            type_name: "native".into(),
            data: json!({"text":"λ🙂漢字\n\"\\\u{9b}".repeat(1000),"provider":{"exact":[0,null,false]}}),
        },
    };
    let encoded = String::from_utf8(crate::encode::json_bytes(&original).unwrap()).unwrap();
    let mut anchor = before_anchor(&original, 3, encoded.chars().count() as u64);
    let mut chunks = Vec::new();
    for _ in 0..1000 {
        let mut page =
            pager::PageBuilder::new_before(header(), budget(), anchor, framing()).unwrap();
        assert_eq!(page.upper_position(), Some(&anchor));
        let status = page.push_at(selection(), 3, original.clone()).unwrap();
        let lower = *page.first_position().unwrap();
        let previous = page.position().copied();
        assert_eq!(page.upper_position(), Some(&anchor));
        let doc = page.finish(ReadFooter::default()).unwrap();
        assert!(encode(&doc, Format::Json).len() <= budget().bytes.get());
        let [item] = doc.items.as_slice() else {
            panic!()
        };
        assert_eq!(item.source, framing().source());
        let ItemKind::Fragment {
            text,
            position,
            until,
            total,
        } = &item.kind
        else {
            panic!()
        };
        assert_eq!(*position, lower);
        assert_eq!(position.item_hash, item_hash(&original));
        assert_eq!(*until, anchor.character);
        assert_eq!(*total, encoded.chars().count() as u64);
        assert_eq!(text.chars().count() as u64, until - position.character);
        assert!(position.character < *until);
        chunks.push(text.clone());
        if status == pager::Push::Accepted {
            assert_eq!(lower.character, 0);
            break;
        }
        assert_eq!(previous, Some(lower));
        anchor = lower;
    }
    chunks.reverse();
    let restored = chunks.concat();
    assert_eq!(restored, encoded);
    assert_eq!(
        serde_json::from_str::<ReadItem>(&restored).unwrap(),
        original
    );
}
#[test]
fn before_zero_excludes_the_verified_anchor_and_outputs_descending_input_forward() {
    let upper_item = text("anchor");
    let upper = before_anchor(&upper_item, 5, 0);
    let mut limits = budget();
    limits.items = NonZeroUsize::new(2).unwrap();
    let mut page = pager::PageBuilder::new_before(header(), limits, upper, framing()).unwrap();
    assert_eq!(
        page.push_at(selection(), 5, upper_item.clone()).unwrap(),
        pager::Push::Accepted
    );
    assert_eq!(page.first_position(), None);
    assert_eq!(
        page.push_at(selection(), 4, text("four")).unwrap(),
        pager::Push::Accepted
    );
    assert_eq!(
        page.push_at(selection(), 3, text("three")).unwrap(),
        pager::Push::Accepted
    );
    assert_eq!(
        page.push_at(selection(), 2, text("two")).unwrap(),
        pager::Push::Full
    );
    assert_eq!(page.first_position().unwrap().item, 3);
    assert_eq!(page.position(), page.first_position());
    assert_eq!(page.upper_position(), Some(&upper));
    let doc = page.finish(ReadFooter::default()).unwrap();
    assert_eq!(doc.items, vec![text("three"), text("four")]);
    let mut changed = pager::PageBuilder::new_before(header(), limits, upper, framing()).unwrap();
    assert_eq!(
        changed.push_at(selection(), 5, text("late availability changed")),
        Err(pager::PageError::ItemChanged)
    );
    let mut missing = pager::PageBuilder::new_before(header(), limits, upper, framing()).unwrap();
    assert_eq!(
        missing.push_at(selection(), 4, upper_item.clone()),
        Err(pager::PageError::InvalidPosition)
    );
    let mut duplicate = pager::PageBuilder::new_before(header(), limits, upper, framing()).unwrap();
    duplicate.push_at(selection(), 5, upper_item).unwrap();
    assert_eq!(
        duplicate.push_at(selection(), 5, text("same ordinal")),
        Err(pager::PageError::InvalidPosition)
    );
}
#[test]
fn before_footer_trim_evicts_earliest_output_and_regenerates_its_lower_anchor() {
    let upper_item = text("anchor");
    let upper = before_anchor(&upper_item, 3, 0);
    let mut limits = budget();
    limits.items = NonZeroUsize::new(3).unwrap();
    limits.bytes = NonZeroUsize::new(2200).unwrap();
    let mut page = pager::PageBuilder::new_before(header(), limits, upper, framing()).unwrap();
    page.push_at(selection(), 3, upper_item).unwrap();
    for n in (0..3).rev() {
        assert_eq!(
            page.push_at(selection(), n, text(&format!("{n}{}", "a".repeat(350))))
                .unwrap(),
            pager::Push::Accepted
        );
    }
    assert_eq!(page.first_position().unwrap().item, 0);
    assert_eq!(page.upper_position(), Some(&upper));
    let earlier_row = SelectionLocator {
        row: selection().row - 1,
        ..selection()
    };
    assert_eq!(
        page.push_at(earlier_row, 0, text("earlier row")).unwrap(),
        pager::Push::Full
    );
    let initial_lower = *page.first_position().unwrap();
    assert_eq!(page.position(), Some(&initial_lower));
    let mut initial_footer = large_footer(Some(&initial_lower)).unwrap();
    initial_footer.shown = 3;
    initial_footer.estimator = Some(TOKEN_ESTIMATOR.into());
    let untrimmed = ReadDocument {
        header: header(),
        items: (0..3)
            .map(|n| text(&format!("{n}{}", "a".repeat(350))))
            .collect(),
        footer: initial_footer,
    };
    assert!(
        encode(&untrimmed, Format::Json).len() > limits.bytes.get(),
        "fixture must require footer trimming"
    );
    let mut attempts = 0;
    let mut final_lower = None;
    let doc = page
        .finish_with(|lower| {
            attempts += 1;
            final_lower = lower.copied();
            large_footer(lower)
        })
        .unwrap();
    assert!(attempts > 1);
    assert!(!doc.items.is_empty() && doc.items.len() < 3);
    let lower = final_lower.unwrap();
    assert_eq!(lower.item, 3 - doc.items.len() as u64);
    let ItemKind::Text { text: first } = &doc.items[0].kind else {
        panic!()
    };
    assert!(first.starts_with(&lower.item.to_string()));
    let ItemKind::Text { text: last } = &doc.items.last().unwrap().kind else {
        panic!()
    };
    assert!(last.starts_with('2'));
    assert_eq!(
        doc.footer.next.as_ref().unwrap().mcp["position"],
        json!(lower)
    );
    assert!(encode(&doc, Format::Json).len() <= limits.bytes.get());
}
#[test]
fn before_footer_trim_advances_only_the_left_fragment_edge() {
    let original = text(&"λ🙂".repeat(150));
    let encoded = String::from_utf8(crate::encode::json_bytes(&original).unwrap()).unwrap();
    let upper = before_anchor(&original, 1, encoded.chars().count() as u64);
    let mut limits = budget();
    limits.items = NonZeroUsize::new(2).unwrap();
    limits.bytes = NonZeroUsize::new(1900).unwrap();
    let mut page = pager::PageBuilder::new_before(header(), limits, upper, framing()).unwrap();
    assert_eq!(
        page.push_at(selection(), 1, original.clone()).unwrap(),
        pager::Push::Accepted
    );
    assert_eq!(
        page.push_at(selection(), 0, text(&"n".repeat(1000)))
            .unwrap(),
        pager::Push::Full
    );
    assert_eq!(page.upper_position(), Some(&upper));
    let mut final_lower = None;
    let doc = page
        .finish_with(|lower| {
            final_lower = lower.copied();
            large_footer(lower)
        })
        .unwrap();
    let ItemKind::Fragment {
        text,
        position,
        until,
        total,
    } = &doc.items[0].kind
    else {
        panic!()
    };
    assert!(position.character > 0);
    assert_eq!(*until, upper.character);
    assert_eq!(*total, upper.character);
    assert_eq!(position.item_hash, upper.item_hash);
    assert_eq!(final_lower, Some(*position));
    assert_eq!(
        text,
        &encoded
            .chars()
            .skip(position.character as usize)
            .collect::<String>()
    );
    assert!(encode(&doc, Format::Json).len() <= limits.bytes.get());
}

#[test]
fn before_input_must_descend_across_rows_families_and_record_ordinals() {
    let anchor_item = text("anchor");
    let anchor_selection = SelectionLocator {
        row: 2,
        record: ReadRecordLocator {
            kind: ReadRecordKind::Event,
            ordinal: 4,
        },
    };
    let anchor = ItemPlace {
        selection: anchor_selection,
        item: 5,
        character: 0,
        item_hash: item_hash(&anchor_item),
    };
    for (selection, ordinal) in [
        (
            SelectionLocator {
                row: 3,
                ..anchor_selection
            },
            0,
        ),
        (
            SelectionLocator {
                record: ReadRecordLocator {
                    kind: ReadRecordKind::Link,
                    ordinal: 0,
                },
                ..anchor_selection
            },
            0,
        ),
        (
            SelectionLocator {
                record: ReadRecordLocator {
                    ordinal: 5,
                    ..anchor_selection.record
                },
                ..anchor_selection
            },
            0,
        ),
        (anchor_selection, 6),
    ] {
        let mut page =
            pager::PageBuilder::new_before(header(), budget(), anchor, framing()).unwrap();
        page.push_at(anchor_selection, 5, anchor_item.clone())
            .unwrap();
        assert_eq!(
            page.push_at(selection, ordinal, text("out of order")),
            Err(pager::PageError::InvalidPosition)
        );
    }
    let mut limits = budget();
    limits.items = NonZeroUsize::new(8).unwrap();
    limits.bytes = NonZeroUsize::new(5000).unwrap();
    let mut page = pager::PageBuilder::new_before(header(), limits, anchor, framing()).unwrap();
    page.push_at(anchor_selection, 5, anchor_item).unwrap();
    for selection in [
        SelectionLocator {
            row: 2,
            record: ReadRecordLocator {
                kind: ReadRecordKind::Message,
                ordinal: 99,
            },
        },
        SelectionLocator {
            row: 2,
            record: ReadRecordLocator {
                kind: ReadRecordKind::Session,
                ordinal: 100,
            },
        },
        SelectionLocator {
            row: 1,
            record: ReadRecordLocator {
                kind: ReadRecordKind::Link,
                ordinal: 100,
            },
        },
    ] {
        assert_eq!(
            page.push_at(selection, 9, text("earlier")).unwrap(),
            pager::Push::Accepted
        );
    }
    assert_eq!(page.first_position().unwrap().selection.row, 1);
    let later_family = SelectionLocator {
        row: 1,
        record: ReadRecordLocator {
            kind: ReadRecordKind::Document,
            ordinal: 0,
        },
    };
    assert_eq!(
        page.push_at(later_family, 0, text("later family in same row")),
        Err(pager::PageError::InvalidPosition)
    );
}

#[test]
fn typed_lineage_issues_fragment_losslessly_and_escape_prose_markup() {
    let original = ReadItem {
        source: Source::record("session:test:child"),
        kind: ItemKind::LinkIssue {
            issue: ParentIssue::MissingParent {
                child: SessionId::parse("test:child").unwrap(),
                parent: SessionId::parse(&format!("test:{}", "λ🙂<script>\u{1b}".repeat(400)))
                    .unwrap(),
            },
        },
    };
    let parts = fragments(original.clone());
    assert!(parts.len() > 1);
    let encoded = parts
        .into_iter()
        .map(|part| match part.kind {
            ItemKind::Fragment { text, .. } => text,
            _ => panic!("large issue must use ordinary JSON fragments"),
        })
        .collect::<String>();
    assert_eq!(
        serde_json::from_str::<ReadItem>(&encoded).unwrap(),
        original
    );
    let document = ReadDocument {
        header: header(),
        items: vec![original],
        footer: ReadFooter::default(),
    };
    let html = encode(&document, Format::Html);
    assert!(!html.contains("<script>"));
    assert!(html.contains("&lt;script&gt;"));
    let terminal = encode(&document, Format::Terminal);
    assert!(!terminal.contains('\u{1b}'));
    assert!(terminal.contains("Parents incomplete"));
}

#[test]
fn html_relationship_links_encode_only_exact_recorded_session_references() {
    let render = |from: &str, to: &str| {
        crate::html::item(
            &ReadItem {
                source: Source::record("link:synthetic"),
                kind: ItemKind::Card {
                    card: Box::new(Card::LinkEdge {
                        from: from.into(),
                        to: to.into(),
                        relation: "continuation".into(),
                        how: Some("fork <marker>".into()),
                        cross_link: false,
                    }),
                },
            },
            1000,
            &jiff::tz::TimeZone::UTC,
        )
        .into_string()
    };
    let html = render(
        "session:claude-code:parent",
        "session:claude-code:child\"<&",
    );
    assert!(html.contains("href=\"/s/claude-code%3Aparent\""), "{html}");
    assert!(
        html.contains("href=\"/s/claude-code%3Achild%22%3C%26\""),
        "{html}"
    );
    assert!(
        html.contains("session:claude-code:child&quot;&lt;&amp;"),
        "{html}"
    );
    assert!(
        html.contains("continuation (fork &lt;marker&gt;)"),
        "{html}"
    );
    let inert = render("javascript:alert(1)", "message:claude-code:u1");
    assert!(!inert.contains("href="), "{inert}");
    assert!(inert.contains("javascript:alert(1)") && inert.contains("message:claude-code:u1"));
}

#[test]
fn times_render_local_with_quality() {
    let mut record = message(vec![Block::Text {
        text: "hello".into(),
    }]);
    if let Record::Message(m) = &mut record {
        m.message.at =
            Some(polylogue_model::Timestamp::parse_rfc3339("2026-11-09T08:12:00Z").unwrap());
        m.message.at_quality = TimeQuality::Intrinsic;
        m.message.at_precision = Some(polylogue_model::Precision::Instant);
        m.source_version = m.message.content_hash().unwrap();
    }
    let options = ReadOptions::default();
    let items = layout(record, &options, &ProjectionEvidence::default());
    let mut doc = ReadDocument {
        header: header(),
        items,
        footer: ReadFooter::default(),
    };
    doc.header.zone = Some("Europe/Warsaw".into());
    for format in [Format::Terminal, Format::Markdown, Format::Html] {
        let text = encode(&doc, format);
        assert!(
            text.contains("2026-11-09 09:12 +01:00 (intrinsic, instant)"),
            "{format:?}: {text}"
        );
    }
    if let ItemKind::MessageHeader { header } = &mut doc.items[0].kind {
        header.time.evidence = polylogue_model::TimeEvidence {
            quality: TimeQuality::Captured,
            precision: Some(polylogue_model::Precision::Bounded),
            until: Some(polylogue_model::Timestamp::parse_rfc3339("2026-11-09T08:13:00Z").unwrap()),
        };
    } else {
        panic!("message header");
    }
    let mut session = polylogue_model::Session::new(
        SessionId::parse("test:time-session").unwrap(),
        polylogue_model::SessionKind::Main,
    );
    session
        .set_time(
            Some(polylogue_model::Timestamp::parse_rfc3339("2026-11-09T08:12:00Z").unwrap()),
            polylogue_model::TimeEvidence {
                quality: TimeQuality::Captured,
                precision: Some(polylogue_model::Precision::Bounded),
                until: Some(
                    polylogue_model::Timestamp::parse_rfc3339("2026-11-09T08:13:00Z").unwrap(),
                ),
            },
        )
        .unwrap();
    doc.items.push(ReadItem {
        source: Source::record("session:test:time-session"),
        kind: ItemKind::Card {
            card: Box::new(Card::Session {
                session: Box::new(session),
                open: None,
                lineage_complete: ValueState::NotApplicable,
            }),
        },
    });
    for format in [Format::Terminal, Format::Markdown, Format::Html] {
        let text = encode(&doc, format);
        assert_eq!(
            text.matches(
                "2026-11-09 09:12 +01:00 (captured, bounded until 2026-11-09 09:13 +01:00)"
            )
            .count(),
            2,
            "message and session card in {format:?}: {text}"
        );
    }
    let text = encode(&doc, Format::Terminal);
    assert!(text.contains("Times Europe/Warsaw."), "{text}");
    doc.header.zone = None;
    let text = encode(&doc, Format::Terminal);
    assert!(!text.contains("Times "), "{text}");
    assert!(text.contains("2026-11-09 08:12 +00:00"), "{text}");
}

/// A document covering every author, every indicator and every tool
/// outcome, for the format goldens.
fn every_kind_document() -> ReadDocument {
    use polylogue_model::{Author, Origin, ToolOutcome, UnknownReason};
    let source = Source::record("block:test:session:message#0");
    let annotations = [
        Author::Operator,
        Author::Agent(None),
        Author::Agent(Some(SessionId::parse("codex:agent-session").unwrap())),
        Author::AgentOrigin(Origin::new("codex").unwrap()),
        Author::Unattributed,
        Author::Script("nightly".into()),
    ]
    .into_iter()
    .map(|author| Annotation {
        reference: "assertion:a1@a1".into(),
        author,
        kind: "note".into(),
        value: json!("checked"),
        open: None,
    })
    .collect();
    let mut items = vec![ReadItem {
        source: source.clone(),
        kind: ItemKind::MessageHeader {
            header: Box::new(MessageHeader {
                time: TimeValue::unknown(),
                author: ValueState::known("model".into()),
                model: ValueState::NotRecorded,
                assertions: annotations,
                variant: None,
                inherited: false,
                branch: Default::default(),
                context: None,
            }),
        },
    }];
    for indicator in [
        IndicatorKind::Continues,
        IndicatorKind::Branches,
        IndicatorKind::Compacted,
        IndicatorKind::Microcompacted,
        IndicatorKind::RateLimited,
        IndicatorKind::ModelSwitch,
        IndicatorKind::AwaySummary,
        IndicatorKind::Harness,
        IndicatorKind::Turn,
        IndicatorKind::LinkIssue,
        IndicatorKind::Stale,
        IndicatorKind::Variant,
    ] {
        items.push(ReadItem {
            source: source.clone(),
            kind: ItemKind::Indicator {
                indicator,
                text: "as recorded".into(),
                expand: None,
            },
        });
    }
    let options = ReadOptions {
        tools: Some(Expansion::Full),
        ..Default::default()
    };
    for (n, outcome) in [
        ToolOutcome::Ok,
        ToolOutcome::Error {
            reason: Some(polylogue_model::ErrorReason::ExitCode(2)),
        },
        ToolOutcome::Unknown {
            reason: UnknownReason::StatusFieldUnreadable,
        },
        ToolOutcome::Unknown {
            reason: UnknownReason::RunningInBackground,
        },
    ]
    .into_iter()
    .enumerate()
    {
        let result = Block::ToolResult {
            reported_duration_ms: None,
            call_id: Some(format!("call-{n}")),
            outcome,
            output: Some("out".into()),
            data: None,
            attachments: vec![],
        };
        items.extend(
            layout(
                message(vec![result]),
                &options,
                &ProjectionEvidence::default(),
            )
            .into_iter()
            .filter(|item| matches!(item.kind, ItemKind::Call { .. })),
        );
    }
    let mut doc = document(items);
    doc.footer.coverage = vec![Coverage {
        target: TargetKind::TextFragment,
        view: View::Transcript,
        omitted: 2,
        accepted_views: vec![View::Transcript],
    }];
    doc
}

#[test]
fn format_goldens_terminal() {
    assert_eq!(
        encode(&every_kind_document(), Format::Terminal),
        include_str!("../tests/goldens/every_kind.terminal")
    );
}

#[test]
fn format_goldens_markdown() {
    // The document ends with a fence and a blank line; the file cannot.
    assert_eq!(
        encode(&every_kind_document(), Format::Markdown).trim_end(),
        include_str!("../tests/goldens/every_kind.markdown").trim_end()
    );
}

#[test]
fn format_goldens_html() {
    assert_eq!(
        encode(&every_kind_document(), Format::Html),
        include_str!("../tests/goldens/every_kind.html").trim_end()
    );
}

#[test]
fn no_debug_formatting_in_rendered_output() {
    let doc = every_kind_document();
    for format in [Format::Terminal, Format::Markdown, Format::Html] {
        let output = encode(&doc, format);
        for debug in [
            "Some(",
            "None",
            "SessionId(",
            "Agent(",
            "AgentOrigin",
            "Script(",
            "Unattributed",
            "StatusFieldUnreadable",
            "RunningInBackground",
            "RateLimited",
            "ModelSwitch",
            "AwaySummary",
            "LinkIssue",
            "TextFragment",
        ] {
            assert!(
                !output.contains(debug),
                "{format:?} shows {debug}:\n{output}"
            );
        }
    }
    let mut session = polylogue_model::Session::new(
        SessionId::parse("test:session").unwrap(),
        polylogue_model::SessionKind::Main,
    );
    session
        .set_time(
            Some(polylogue_model::Timestamp::from_unix_ms(1_000).unwrap()),
            polylogue_model::TimeEvidence {
                quality: TimeQuality::Captured,
                precision: Some(polylogue_model::Precision::Minute),
                until: None,
            },
        )
        .unwrap();
    let card = crate::encode::session_lines(&session, &jiff::tz::TimeZone::UTC).join("\n");
    assert!(card.contains("(captured, minute)"), "{card}");
    for (error, words) in [
        (crate::pager::PageError::ItemChanged, "changed"),
        (crate::pager::PageError::AlreadyFull, "full"),
    ] {
        assert!(error.to_string().contains(words));
    }
}

#[test]
fn markdown_escapes_only_heading_markers() {
    let doc = document(vec![text(
        "# heading-like line\n   ## indented\n    # code indent\nTitle\n===\n---\n- a list item\nuse foo_bar(*x*) [link](u) a.b! {c} `tick` 1+1 a|b\n<b>bold</b> & co",
    )]);
    let markdown = encode(&doc, Format::Markdown);
    let body: Vec<&str> = markdown.lines().skip(3).collect();
    assert_eq!(
        body,
        [
            "\\# heading-like line",
            "   \\## indented",
            "    # code indent",
            "Title",
            "\\===",
            "\\---",
            "- a list item",
            "use foo_bar(*x*) [link](u) a.b! {c} `tick` 1+1 a|b",
            "&lt;b&gt;bold&lt;/b&gt; &amp; co",
            "",
        ]
    );
}
