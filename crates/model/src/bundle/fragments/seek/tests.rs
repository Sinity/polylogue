use super::*;
#[test]
fn native_seek_coordinates_preserve_actual_ordinals_and_explicit_separator_positions() {
    let body = NativeTextBody {
        block: 7,
        field: TextField::ToolOutput,
    };
    let after = NativeTextBody {
        block: 11,
        field: TextField::Thinking,
    };
    let seek = NativeSpanSeek {
        start: NativeTextPosition {
            piece: NativeTextPiece::Body { body },
            byte: 3,
        },
        end: NativeTextPosition {
            piece: NativeTextPiece::Separator {
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
        serde_json::from_str::<NativeSpanSeek>(&serde_json::to_string(&seek).unwrap()).unwrap(),
        seek
    );
    let mut bad = seek;
    bad.end.byte = 3;
    assert!(bad.validate(&container, &coverage).is_err());
    let reversed = NativeSpanSeek {
        start: NativeTextPosition {
            piece: NativeTextPiece::Body { body: after },
            byte: 0,
        },
        end: seek.start,
    };
    assert!(reversed.validate(&container, &coverage).is_err());
    let reversed_separator = NativeSpanSeek {
        start: seek.end,
        end: seek.start,
    };
    assert!(reversed_separator.validate(&container, &coverage).is_err());
    assert!(
        seek.validate(
            &Reference::parse("block:codex:example#7").unwrap(),
            &coverage
        )
        .is_err()
    );
}
#[test]
fn accessible_empty_has_no_invented_body_and_cannot_name_nonempty_coverage() {
    let position = NativeTextPosition {
        piece: NativeTextPiece::Empty,
        byte: 0,
    };
    let seek = NativeSpanSeek {
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
    assert!(
        seek.validate(
            &container,
            &TextCoverage {
                start: 0,
                end: 1,
                total_bytes: None
            }
        )
        .is_err()
    );
    assert!(
        serde_json::from_str::<NativeTextBody>(r#"{"block":7,"field":"text","body":"unselected"}"#)
            .is_err()
    );
}
