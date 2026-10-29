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
    evidence.variants.insert(
        0,
        VariantText {
            mark: VariantMark {
                reference: "variant:test".into(),
                key: "en".into(),
                original: "block:test:session:message#0".into(),
                stale: true,
                show_original: None,
            },
            text: "variant".into(),
        },
    );
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
        matches!(&items[1],ReadPart::Item(ReadItem{kind:ItemKind::Text{text},..}) if text=="variant")
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
    let output = html::item(&item).into_string();
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
fn budget() -> pager::Budget {
    pager::Budget {
        items: NonZeroUsize::new(1).unwrap(),
        tokens: NonZeroU64::new(1800).unwrap(),
        bytes: NonZeroUsize::new(1600).unwrap(),
    }
}
fn fragments(original: ReadItem) -> Vec<ReadItem> {
    let mut position = None;
    let mut fragments = vec![];
    for _ in 0..1000 {
        let mut builder = pager::PageBuilder::new(header(), budget(), position).unwrap();
        let status = builder
            .push_at("message:test:session:message", 3, original.clone())
            .unwrap();
        let next = builder.position().cloned();
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
                ItemKind::Fragment {
                    encoding: FragmentEncoding::Text,
                    text,
                    ..
                } => text,
                _ => panic!("fragment"),
            })
            .collect::<String>();
        assert_eq!(joined, original);
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
            ItemKind::Fragment {
                encoding: FragmentEncoding::ReadItemJson,
                text,
                ..
            } => text,
            _ => panic!(),
        })
        .collect::<String>();
    assert_eq!(serde_json::from_str::<ReadItem>(&json).unwrap(), original);
}
#[test]
fn page_cut_uses_canonical_record_position_not_paired_display_source() {
    let mut builder = pager::PageBuilder::new(header(), budget(), None).unwrap();
    assert_eq!(
        builder
            .push_at("message:canonical", 0, text("first"))
            .unwrap(),
        pager::Push::Accepted
    );
    assert_eq!(
        builder
            .push_at("message:canonical", 1, text("second"))
            .unwrap(),
        pager::Push::Full
    );
    assert_eq!(
        builder.position(),
        Some(&ReadPosition {
            reference: "message:canonical".into(),
            item: 1,
            character: 0
        })
    );
}
#[test]
fn missing_or_invalid_cursor_is_an_error_never_a_shifted_page() {
    let mut builder = pager::PageBuilder::new(
        header(),
        budget(),
        Some(ReadPosition {
            reference: "message:other".into(),
            item: 3,
            character: 1,
        }),
    )
    .unwrap();
    assert_eq!(
        builder.push_at("message:canonical", 3, text("text")),
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
    assert_eq!(excerpt.omitted_lines, 101);
    assert!(excerpt.head.starts_with("line 0"));
    assert!(excerpt.tail.starts_with("line 201"));
    assert_eq!(
        excerpt.head.chars().count() as u64
            + excerpt.omitted_characters
            + excerpt.tail.chars().count() as u64,
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
    assert_eq!(excerpt.head.chars().count(), 2000);
    assert_eq!(excerpt.tail.chars().count(), 2000);
    assert_eq!(excerpt.omitted_characters, 6000);
    assert_eq!(excerpt.omitted_lines, 0);
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
        },
    };
    assert!(
        html::item_with_inline_limit(&make(attachment.clone()), 100)
            .into_string()
            .contains("data:image/png;base64,")
    );
    assert!(
        !html::item_with_inline_limit(&make(attachment.clone()), 3)
            .into_string()
            .contains("<img")
    );
    assert!(
        !html::item(&make(attachment.clone()))
            .into_string()
            .contains("<img")
    );
    attachment.media_type = Some("image/svg+xml".into());
    assert!(
        !html::item_with_inline_limit(&make(attachment.clone()), 100)
            .into_string()
            .contains("<img")
    );
    attachment.media_type = Some("text/html".into());
    assert!(
        !html::item_with_inline_limit(&make(attachment.clone()), 100)
            .into_string()
            .contains("<img")
    );
    attachment.media_type = Some("image/png".into());
    attachment.blob = Some(polylogue_model::BlobHash::of(b"different"));
    assert!(
        !html::item_with_inline_limit(&make(attachment), 100)
            .into_string()
            .contains("<img")
    );
}
