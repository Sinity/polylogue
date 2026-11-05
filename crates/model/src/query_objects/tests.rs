use super::*;
use crate::{
    BlobHash, Reference, SpanUnit,
    bundle::fragments::seek::{NativeTextBody, NativeTextPiece, NativeTextPosition, TextField},
};

fn resolved() -> NormalizedReadSpan {
    let original = Reference::parse("span:message:codex:synthetic#c=1-3").unwrap();
    NormalizedReadSpan::new(
        TextSpanOrigin {
            reference: original,
            start: 4,
            end: 9,
        },
        TextSourceVersion::ContentHash {
            record: Reference::parse("message:codex:synthetic").unwrap(),
            hash: BlobHash::of(b"retained source identity, not selected text"),
        },
        NativeSpanSeek {
            start: NativeTextPosition {
                piece: NativeTextPiece::Body {
                    body: NativeTextBody {
                        block: 7,
                        field: TextField::Text,
                    },
                },
                byte: 1,
            },
            end: NativeTextPosition {
                piece: NativeTextPiece::Body {
                    body: NativeTextBody {
                        block: 7,
                        field: TextField::Text,
                    },
                },
                byte: 6,
            },
        },
    )
    .unwrap()
}

#[test]
fn normalized_span_round_trips_declared_identity_without_copied_source_facts() {
    let value = resolved();
    let encoded = serde_json::to_value(&value).unwrap();
    let decoded: NormalizedReadSpan = serde_json::from_value(encoded.clone()).unwrap();
    decoded.validate().unwrap();
    assert_eq!(decoded, value);
    assert_eq!(
        decoded.coverage(),
        TextCoverage {
            start: 4,
            end: 9,
            total_bytes: None
        }
    );
    for field in ["text", "header", "verified_source_hash"] {
        let mut copied = encoded.clone();
        copied
            .as_object_mut()
            .unwrap()
            .insert(field.into(), serde_json::json!("untrusted copied fact"));
        assert!(serde_json::from_value::<NormalizedReadSpan>(copied).is_err());
    }
}

#[test]
fn normalized_span_rejects_changed_owner_original_coordinates_and_native_endpoints() {
    let value = resolved();
    let mut wrong_owner = value.clone();
    wrong_owner.version = TextSourceVersion::ContentHash {
        record: Reference::parse("message:codex:other").unwrap(),
        hash: BlobHash::of(b"other retained identity"),
    };
    assert_eq!(
        wrong_owner.validate(),
        Err(TextFragmentError::SourceVersion)
    );
    let mut wrong_coordinates = value.clone();
    let Reference::Span { unit, .. } = &mut wrong_coordinates.origin.reference else {
        unreachable!()
    };
    *unit = SpanUnit::Bytes;
    assert_eq!(wrong_coordinates.validate(), Err(TextFragmentError::Origin));
    let mut wrong_native = value.clone();
    wrong_native.seek.end.piece = NativeTextPiece::Body {
        body: NativeTextBody {
            block: 3,
            field: TextField::Text,
        },
    };
    assert_eq!(wrong_native.validate(), Err(TextFragmentError::Origin));
    let mut wrong_container = value;
    wrong_container.origin.reference = Reference::parse("message:codex:synthetic").unwrap();
    assert_eq!(wrong_container.validate(), Err(TextFragmentError::Origin));
}

#[test]
fn fragment_only_span_binding_keeps_membership_without_inventing_native_seek() {
    let native = resolved();
    let value = ReadSpanBinding::Fragments {
        origin: native.origin,
        version: native.version,
        membership: BlobHash::of(b"actual indexed grain metadata"),
    };
    value.validate().unwrap();
    let encoded = serde_json::to_value(&value).unwrap();
    let decoded: ReadSpanBinding = serde_json::from_value(encoded.clone()).unwrap();
    assert_eq!(decoded, value);
    let mut invented = encoded;
    invented
        .as_object_mut()
        .unwrap()
        .insert("seek".into(), serde_json::json!({"start":0,"end":0}));
    assert!(serde_json::from_value::<ReadSpanBinding>(invented).is_err());
}
