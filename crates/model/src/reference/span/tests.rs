use super::*;

#[test]
fn original_span_bounds_are_independent_of_chunk_boundaries() {
    let text = "a💡ż\n尾";
    for (unit, start, end) in [
        (SpanUnit::Bytes, 1, 7),
        (SpanUnit::CodePoints, 1, 3),
        (SpanUnit::Utf16, 1, 4),
    ] {
        for split in (0..=text.len()).filter(|split| text.is_char_boundary(*split)) {
            let mut span = SpanBounds::new(unit, start, end).unwrap();
            span.push(&text[..split]).unwrap();
            span.push(&text[split..]).unwrap();
            assert_eq!(span.finish().unwrap(), 1..7);
        }
    }
}
#[test]
fn span_boundaries_reject_incomplete_characters_and_absent_positions() {
    assert!(matches!(
        SpanBounds::new(SpanUnit::Bytes, 2, 1),
        Err(SpanError::Reversed)
    ));
    for (unit, start, end) in [(SpanUnit::Bytes, 2, 5), (SpanUnit::Utf16, 2, 3)] {
        let mut span = SpanBounds::new(unit, start, end).unwrap();
        assert_eq!(span.push("a💡ż"), Err(SpanError::SplitCharacter));
    }
    let mut span = SpanBounds::new(SpanUnit::CodePoints, 0, 4).unwrap();
    span.push("a💡ż").unwrap();
    assert_eq!(span.finish(), Err(SpanError::OutOfRange));
    assert_eq!(
        SpanBounds::new(SpanUnit::Utf16, 0, 0).unwrap().finish(),
        Ok(0..0)
    );
}
#[test]
fn original_body_offsets_preserve_empty_bodies_and_separator_positions() {
    let mut offsets = TextBodyOffsets::default();
    assert_eq!(offsets.push(0).unwrap(), 0..0);
    assert_eq!(offsets.push(5).unwrap(), 2..7);
    assert_eq!(offsets.push(0).unwrap(), 9..9);
    let mut span = SpanBounds::new(SpanUnit::Utf16, 0, 5).unwrap();
    for chunk in ["", TEXT_BODY_SEPARATOR, "a💡", TEXT_BODY_SEPARATOR, ""] {
        span.push(chunk).unwrap();
    }
    assert_eq!(span.finish().unwrap(), 0..7);
}

#[test]
fn span_body_alignment_preserves_separators_and_zero_width_insertions() {
    assert_eq!(
        span_bodies(4..8, [5, 2]).unwrap(),
        vec![
            TextSpanPiece::Body {
                body_index: 0,
                start: 4,
                end: 5
            },
            TextSpanPiece::Separator {
                after_body: 0,
                start: 0,
                end: 2
            },
            TextSpanPiece::Body {
                body_index: 1,
                start: 0,
                end: 1
            },
        ]
    );
    assert_eq!(
        span_bodies(5..6, [5, 2]).unwrap(),
        vec![TextSpanPiece::Separator {
            after_body: 0,
            start: 0,
            end: 1
        }]
    );
    assert_eq!(
        span_bodies(5..5, [5, 2]).unwrap(),
        vec![TextSpanPiece::Body {
            body_index: 0,
            start: 5,
            end: 5
        }]
    );
    assert_eq!(
        span_bodies(6..6, [5, 2]).unwrap(),
        vec![TextSpanPiece::Separator {
            after_body: 0,
            start: 1,
            end: 1
        }]
    );
    assert_eq!(
        span_bodies(7..7, [5, 2]).unwrap(),
        vec![TextSpanPiece::Body {
            body_index: 1,
            start: 0,
            end: 0
        }]
    );
    assert_eq!(span_bodies(0..0, []).unwrap(), vec![]);
}

#[test]
fn every_resolved_byte_belongs_to_a_body_or_an_explicit_separator() {
    for lengths in [vec![], vec![0], vec![0, 0], vec![1, 2], vec![3, 0, 4]] {
        let total = lengths.iter().sum::<u64>() + 2 * lengths.len().saturating_sub(1) as u64;
        for start in 0..=total {
            for end in start..=total {
                let pieces = span_bodies(start..end, lengths.iter().copied()).unwrap();
                let covered: u64 = pieces
                    .iter()
                    .map(|piece| match piece {
                        TextSpanPiece::Body { start, end, .. } => end - start,
                        TextSpanPiece::Separator { start, end, .. } => (end - start) as u64,
                    })
                    .sum();
                assert_eq!(covered, end - start, "{lengths:?} {start}..{end}");
            }
        }
    }
}
