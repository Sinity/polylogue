use super::*;
use crate::{
    BlobHash, MessageId, Reference, SpanUnit,
    bundle::{
        TextCoverage, TextFragment, TextFragmentProvenance, TextSourceVersion, TextSpanOrigin,
    },
    read::ContextState,
};

fn body(start: u64, end: u64, kind: BlockKind) -> FragmentBodyOrigin {
    FragmentBodyOrigin { start, end, kind }
}

fn origin(bodies: Vec<FragmentBodyOrigin>) -> FragmentOrigin {
    FragmentOrigin {
        message: Some(FragmentMessageOrigin {
            role: Role::Assistant,
            prompt_kind: None,
        }),
        bodies,
    }
}

fn reason(result: Result<(), ReadError>) -> String {
    result.unwrap_err().to_string()
}

/// Two bodies in one message: bytes 2..4 are the one separator between them.
fn two_bodies() -> FragmentOrigin {
    origin(vec![
        body(0, 2, BlockKind::Text),
        body(4, 9, BlockKind::Thinking),
    ])
}

#[test]
fn an_origin_must_cover_the_selected_range_with_adjacent_text_bodies() {
    let whole = two_bodies();
    whole.validate(0, 9).unwrap();
    whole.validate(1, 7).unwrap();
    whole.validate(3, 3).unwrap();

    let prefix = "fragment origin does not match the selected source recipe: ";
    assert_eq!(
        reason(whole.validate(5, 3)),
        format!("invalid prepared read data: {prefix}the selected range is reversed")
    );
    assert_eq!(
        reason(origin(vec![]).validate(0, 4)),
        format!("invalid prepared read data: {prefix}no body covers the selected range")
    );
    origin(vec![]).validate(0, 0).unwrap();

    // A leading gap: nothing proves what lies before the first body.
    let leading = origin(vec![body(1, 9, BlockKind::Text)]);
    assert_eq!(
        reason(leading.validate(0, 9)),
        format!(
            "invalid prepared read data: {prefix}the first body starts after the selected range"
        )
    );
    // A trailing gap.
    assert_eq!(
        reason(whole.validate(0, 10)),
        format!("invalid prepared read data: {prefix}the last body ends before the selected range")
    );
    // A kind that is not source text.
    let tool_call = origin(vec![body(0, 9, BlockKind::ToolCall)]);
    assert_eq!(
        reason(tool_call.validate(0, 9)),
        format!("invalid prepared read data: {prefix}a body is not a text kind")
    );
    // Bodies that are not exactly one separator apart.
    let mut gapped = two_bodies();
    gapped.bodies[1].start = 5;
    assert_eq!(
        reason(gapped.validate(0, 9)),
        format!(
            "invalid prepared read data: {prefix}bodies are not separated by exactly one separator"
        )
    );
    // A body whose own range is reversed.
    let reversed = origin(vec![body(3, 2, BlockKind::Text)]);
    assert_eq!(
        reason(reversed.validate(3, 2)),
        format!("invalid prepared read data: {prefix}the selected range is reversed")
    );
    let reversed = origin(vec![
        body(0, 4, BlockKind::Text),
        body(6, 5, BlockKind::Text),
    ]);
    assert_eq!(
        reason(reversed.validate(0, 5)),
        format!("invalid prepared read data: {prefix}a body range is reversed")
    );
}

fn fragment_record(text: &str) -> TextFragmentRecord {
    let container = Reference::Message(MessageId::parse("test:session:message").unwrap());
    let end = text.len() as u64;
    let span = Reference::Span {
        container: Box::new(container.clone()),
        unit: SpanUnit::Bytes,
        start: 0,
        end,
    };
    TextFragmentRecord {
        reference: span.to_string(),
        commit: 7,
        fragment: TextFragment {
            container: container.clone(),
            source_version: TextSourceVersion::ContentHash {
                record: container,
                hash: BlobHash::of(b"whole source"),
            },
            coverage: TextCoverage {
                start: 0,
                end,
                total_bytes: None,
            },
            text: text.into(),
            text_hash: BlobHash::of(text.as_bytes()),
            provenance: TextFragmentProvenance {
                spans: vec![TextSpanOrigin {
                    reference: span,
                    start: 0,
                    end,
                }],
                sources: vec![],
                imports: vec![],
            },
        },
    }
}

#[test]
fn a_context_reading_shows_a_fragment_only_of_a_selected_message() {
    let record = fragment_record("hello");
    let options = ReadOptions {
        view: View::Context,
        ..ReadOptions::default()
    };
    let facts = |context| LayoutFacts {
        fragment_origin: ValueState::known(origin(vec![body(0, 5, BlockKind::Text)])),
        context,
        ..LayoutFacts::default()
    };
    let state = |selected| ContextState {
        selected,
        complete: true,
        reason: None,
        evicted: vec![],
        not_recorded: vec![],
    };
    assert!(matches!(
        project_fragment(&record, &options, &facts(None)),
        Err(ReadError::MissingPreparedData { .. })
    ));
    assert!(matches!(
        project_fragment(&record, &options, &facts(Some(state(false)))),
        Err(ReadError::UnsupportedTarget { .. })
    ));
    project_fragment(&record, &options, &facts(Some(state(true)))).unwrap();
}

#[test]
fn an_origin_reads_as_its_message_and_distinct_body_kinds() {
    let origin = FragmentOrigin {
        message: Some(FragmentMessageOrigin {
            role: Role::User,
            prompt_kind: Some(PromptKind::Prose),
        }),
        bodies: vec![
            body(0, 4, BlockKind::Text),
            body(5, 9, BlockKind::ToolResult),
            body(10, 12, BlockKind::Text),
        ],
    };
    assert_eq!(
        origin.to_string(),
        "user message (prose) · text, tool_result"
    );
    let document = FragmentOrigin {
        message: None,
        bodies: vec![],
    };
    assert_eq!(document.to_string(), "document");
}
