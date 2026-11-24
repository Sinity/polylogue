use super::*;
#[test]
fn seek_keeps_ordinals_and_separator_positions() {
    let body = SourceTextBody {
        block: 7,
        field: TextField::ToolOutput,
    };
    let after = SourceTextBody {
        block: 11,
        field: TextField::Thinking,
    };
    let seek = SourceSpanSeek {
        start: SourceTextPosition {
            piece: SourceTextPiece::Body { body },
            byte: 3,
        },
        end: SourceTextPosition {
            piece: SourceTextPiece::Separator {
                before: body,
                after,
            },
            byte: 2,
        },
    };
    let container = Reference::parse("message:codex:example").unwrap();
    let coverage = TextCoverage {
        start: 30,
        end: 39,
        total_bytes: None,
    };
    seek.validate(&container, &coverage).unwrap();
    assert_eq!(
        serde_json::from_str::<SourceSpanSeek>(&serde_json::to_string(&seek).unwrap()).unwrap(),
        seek
    );
    let mut bad = seek;
    bad.end.byte = 3;
    assert_eq!(
        bad.validate(&container, &coverage),
        Err(TextFragmentError::Origin)
    );
    let reversed = SourceSpanSeek {
        start: SourceTextPosition {
            piece: SourceTextPiece::Body { body: after },
            byte: 0,
        },
        end: seek.start,
    };
    assert_eq!(
        reversed.validate(&container, &coverage),
        Err(TextFragmentError::Origin)
    );
    let reversed_separator = SourceSpanSeek {
        start: seek.end,
        end: seek.start,
    };
    assert_eq!(
        reversed_separator.validate(&container, &coverage),
        Err(TextFragmentError::Origin)
    );
    let block = Reference::parse("block:codex:example#7").unwrap();
    assert_eq!(
        seek.validate(&block, &coverage),
        Err(TextFragmentError::Origin)
    );
    let inside_body = |block: u64| {
        let position = |byte| SourceTextPosition {
            piece: SourceTextPiece::Body {
                body: SourceTextBody {
                    block,
                    field: TextField::ToolOutput,
                },
            },
            byte,
        };
        SourceSpanSeek {
            start: position(0),
            end: position(9),
        }
    };
    assert_eq!(inside_body(7).validate(&block, &coverage), Ok(()));
    assert_eq!(
        inside_body(11).validate(&block, &coverage),
        Err(TextFragmentError::Origin)
    );
}
#[test]
fn accessible_empty_invents_no_body_or_coverage() {
    let position = SourceTextPosition {
        piece: SourceTextPiece::Empty,
        byte: 0,
    };
    let seek = SourceSpanSeek {
        start: position,
        end: position,
    };
    let container = Reference::parse("document:codex:example").unwrap();
    seek.validate(
        &container,
        &TextCoverage {
            start: 0,
            end: 0,
            total_bytes: Some(0),
        },
    )
    .unwrap();
    assert_eq!(
        seek.validate(
            &container,
            &TextCoverage {
                start: 0,
                end: 1,
                total_bytes: None
            }
        ),
        Err(TextFragmentError::Origin)
    );
    assert!(
        serde_json::from_str::<SourceTextBody>(r#"{"block":7,"field":"text","body":"unselected"}"#)
            .is_err()
    );
}
