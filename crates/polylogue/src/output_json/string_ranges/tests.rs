use super::*;
use crate::output_json::Encoder;
use polylogue_model::RawJson;
use std::{cell::Cell, time::Duration};

/// A deadline no test reaches, for tests that are not about deadlines; a short
/// one fails slow debug builds on a loaded machine.
fn no_deadline() -> Instant {
    Instant::now() + Duration::from_secs(24 * 60 * 60)
}
fn streamed(raw: &[u8], piece: usize) -> io::Result<Vec<u8>> {
    let mut encoder = Encoder::new(Vec::new(), no_deadline());
    encoder.string_ranges(
        |offset| {
            let start = offset as usize;
            let end = (start + piece).min(raw.len());
            Ok((raw[start..end].to_vec(), end == raw.len()))
        },
        &mut || false,
    )?;
    encoder.finish()
}
fn ordinary(raw: &[u8]) -> io::Result<Vec<u8>> {
    let value = RawJson::parse(raw).unwrap();
    let mut encoder = Encoder::new(Vec::new(), no_deadline());
    encoder.raw_json(&value, false)?;
    encoder.finish()
}
#[test]
fn string_ranges_mask_like_the_whole_value_with_escapes_and_utf8() {
    let value = format!(
        "{}\n\\\"🦀 é\t{}",
        "neutral ".repeat(17000),
        "tail".repeat(1000)
    );
    let raw = serde_json::to_vec(&value).unwrap();
    assert!(raw.len() > 131072);
    for piece in [1, 7, 4096] {
        assert_eq!(streamed(&raw, piece).unwrap(), ordinary(&raw).unwrap());
    }
    // Reuse the privacy library's tested credential form across range/escape boundaries.
    let raw = br#""before ghp_abcdefghijklmnopqrstuvwxyz1234567890 after\nend""#;
    for piece in [1, 7, 17] {
        assert_eq!(streamed(raw, piece).unwrap(), ordinary(raw).unwrap());
    }
}
#[test]
fn native_utf16_ranges_preserve_units_and_refuse_uncertain_masking() {
    for raw in [
        br#""a\ud800b\udc00c\ud83e\udd80\ufffd""#.as_slice(),
        br#""\ud800\ud800\udc00""#.as_slice(),
    ] {
        let canonical = RawJson::parse(raw).unwrap().canonical();
        for piece in [1, 3, 7] {
            assert_eq!(
                streamed(&canonical, piece).unwrap(),
                ordinary(&canonical).unwrap()
            );
        }
    }
    let large = format!(
        "\"{}\\ud800{}\\udc00\"",
        "neutral".repeat(10000),
        "tail".repeat(17000)
    );
    assert!(large.len() > 131072);
    assert_eq!(
        streamed(large.as_bytes(), 4096).unwrap(),
        ordinary(large.as_bytes()).unwrap()
    );
    let raw = br#""\ud800 ghp_abcdefghijklmnopqrstuvwxyz1234567890""#;
    assert!(ordinary(raw).is_err());
    assert!(streamed(raw, 3).is_err());
}
#[test]
fn string_ranges_reject_nonstring_malformed_trailing_and_zero_progress() {
    assert_eq!(streamed(b"\"\"", 1).unwrap(), b"\"\"");
    for raw in [
        b"{}".as_slice(),
        b"\"a\"null",
        b"\"\\u00\"",
        b"\"\\x\"",
        b"\"a\n\"",
        b"\"a",
        b"\"\xff\"",
    ] {
        assert!(streamed(raw, 1).is_err(), "{raw:?}");
    }
    let mut encoder = Encoder::new(Vec::new(), no_deadline());
    assert!(
        encoder
            .string_ranges(|_| Ok((vec![], false)), &mut || false)
            .is_err()
    );
}
#[test]
fn string_ranges_keep_deadline_cancellation_and_field_read_errors() {
    let raw = serde_json::to_vec(&"neutral".repeat(20000)).unwrap();
    let polls = Cell::new(0);
    let mut encoder = Encoder::new(Vec::new(), no_deadline());
    let error = encoder
        .string_ranges(
            |offset| {
                let start = offset as usize;
                let end = (start + 4096).min(raw.len());
                Ok((raw[start..end].to_vec(), end == raw.len()))
            },
            &mut || {
                polls.set(polls.get() + 1);
                polls.get() > 2
            },
        )
        .unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::ConnectionAborted);
    let mut encoder = Encoder::new(Vec::new(), Instant::now() - Duration::from_millis(1));
    assert_eq!(
        encoder
            .string_ranges(|_| panic!("expired before read"), &mut || false)
            .unwrap_err()
            .kind(),
        io::ErrorKind::TimedOut
    );
    let mut encoder = Encoder::new(Vec::new(), no_deadline());
    assert_eq!(
        encoder
            .string_ranges(
                |_| Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "synthetic visibility loss"
                )),
                &mut || false
            )
            .unwrap_err()
            .kind(),
        io::ErrorKind::PermissionDenied
    );
}
