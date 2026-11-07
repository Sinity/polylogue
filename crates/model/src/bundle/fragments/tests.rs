use super::*;

fn fragment(start: u64, text: &str) -> TextFragment {
    let container = Reference::Message(crate::MessageId::parse("codex:source").unwrap());
    let end = start + text.len() as u64;
    TextFragment {
        source_version: TextSourceVersion::ContentHash {
            record: container.clone(),
            hash: BlobHash::of(b"canonical source identity"),
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
                reference: Reference::Span {
                    container: Box::new(container.clone()),
                    unit: SpanUnit::Bytes,
                    start,
                    end,
                },
                start,
                end,
            }],
            sources: vec![],
            imports: vec![],
        },
        container,
    }
}

#[test]
fn text_fragment_overlap_union_is_commutative_idempotent_and_keeps_gaps_unknown() {
    let left = fragment(0, "A😀");
    let right = fragment(1, "😀B");
    assert_eq!(left.validate_overlap(&right).unwrap(), Some(()));
    let merged = left.merge(&right).unwrap().unwrap();
    assert_eq!(merged.text, "A😀B");
    assert_eq!(
        merged.coverage,
        TextCoverage {
            start: 0,
            end: 6,
            total_bytes: None
        }
    );
    assert_eq!(merged, right.merge(&left).unwrap().unwrap());
    assert_eq!(merged, merged.merge(&merged).unwrap().unwrap());
    assert_eq!(merged.provenance.spans.len(), 2);
    assert!(fragment(0, "A").merge(&fragment(2, "B")).unwrap().is_none());
    assert_eq!(merged.coverage.total_bytes, None);
}

#[test]
fn text_fragment_rejects_conflicting_overlap_and_known_totals_without_selecting_a_version() {
    assert_eq!(
        fragment(0, "A😀").validate_overlap(&fragment(1, "xxxxB")),
        Err(TextFragmentError::Conflict)
    );
    let mut left = fragment(0, "A");
    let mut right = fragment(2, "B");
    left.coverage.total_bytes = Some(10);
    right.coverage.total_bytes = Some(11);
    assert_eq!(left.merge(&right), Err(TextFragmentError::Conflict));
    let mut different = fragment(0, "A");
    different.source_version = TextSourceVersion::ContentHash {
        record: different.container.clone(),
        hash: BlobHash::of(b"another version"),
    };
    assert!(fragment(0, "A").merge(&different).unwrap().is_none());
}

#[test]
fn text_fragment_never_verifies_a_source_hash_from_selected_bytes() {
    let value = fragment(7, "body");
    assert!(value.validate().is_ok());
    assert_ne!(
        value.text_hash,
        match value.source_version {
            TextSourceVersion::ContentHash { hash, .. } => hash,
            _ => unreachable!(),
        }
    );
    let mut check = value.check_source(&value.source_version).unwrap();
    assert_eq!(
        check.push("prefix WRONG tail"),
        Err(TextFragmentError::CompleteSource)
    );
    let mut different = value.source_version.clone();
    if let TextSourceVersion::ContentHash { hash, .. } = &mut different {
        *hash = BlobHash::of(b"different canonical value");
    }
    assert!(matches!(
        value.check_source(&different),
        Err(TextFragmentError::SourceVersion)
    ));
}

#[test]
fn text_fragment_complete_source_checks_selected_bytes_and_original_utf16_coordinates_streamingly()
{
    let mut value = fragment(1, "😀");
    value.coverage.total_bytes = Some(6);
    value.provenance.spans[0].reference = Reference::Span {
        container: Box::new(value.container.clone()),
        unit: SpanUnit::Utf16,
        start: 1,
        end: 3,
    };
    let mut check = value.check_source(&value.source_version).unwrap();
    for text in ["A", "😀", "B"] {
        check.push(text).unwrap();
    }
    check.finish().unwrap();
    value.provenance.spans[0].reference = Reference::Span {
        container: Box::new(value.container.clone()),
        unit: SpanUnit::Utf16,
        start: 1,
        end: 2,
    };
    // A snippet cannot attest its unavailable original prefix or surrogate boundary.
    assert!(value.validate().is_ok());
    let mut check = value.check_source(&value.source_version).unwrap();
    check.push("A").unwrap();
    assert_eq!(check.push("😀"), Err(TextFragmentError::CharacterBoundary));
}

#[test]
fn text_fragment_absolute_boundary_stays_unattested_until_complete_source_exists() {
    let value = fragment(2, "X");
    assert!(value.validate().is_ok());
    let mut check = value.check_source(&value.source_version).unwrap();
    assert_eq!(
        check.push("A😀B"),
        Err(TextFragmentError::CharacterBoundary)
    );
    assert_eq!(
        fragment(0, "A😀").validate_overlap(&fragment(2, "X")),
        Err(TextFragmentError::CharacterBoundary)
    );
}

#[test]
fn text_fragment_empty_selection_and_large_coordinates_do_not_invent_outside_bytes() {
    let value = fragment(3, "");
    let mut check = value.check_source(&value.source_version).unwrap();
    check.push("abc").unwrap();
    check.finish().unwrap();
    assert_eq!(value.coverage.total_bytes, None);
    let merged = fragment(u64::MAX - 2, "A")
        .merge(&fragment(u64::MAX - 1, "B"))
        .unwrap()
        .unwrap();
    assert_eq!(merged.coverage.start, u64::MAX - 2);
    assert_eq!(merged.coverage.end, u64::MAX);
    assert_eq!(merged.text, "AB");
}

#[test]
fn text_fragment_separator_coverage_uses_the_existing_document_recipe() {
    let mut value = fragment(1, "\n\n");
    value.container = Reference::Document(crate::DocumentId::parse("codex:document").unwrap());
    value.source_version = TextSourceVersion::ContentHash {
        record: value.container.clone(),
        hash: BlobHash::of(b"document value identity"),
    };
    value.provenance.spans[0].reference = value.reference();
    let mut check = value.check_source(&value.source_version).unwrap();
    for text in ["A", crate::TEXT_BODY_SEPARATOR, "B"] {
        check.push(text).unwrap();
    }
    check.finish().unwrap();
    let record = TextFragmentRecord {
        reference: value.reference().to_string(),
        commit: 99,
        fragment: value,
    };
    record.validate().unwrap();
    let encoded = serde_json::to_vec(&record).unwrap();
    assert_eq!(record, serde_json::from_slice(&encoded).unwrap());
}

#[test]
fn text_fragment_partial_selection_cannot_carry_the_full_unselected_body_or_headers() {
    let mut value = fragment(7, "KEEP");
    value.text = "SECRET KEEP SECRET".into();
    value.coverage.start = 0;
    value.coverage.end = value.text.len() as u64;
    value.text_hash = BlobHash::of(value.text.as_bytes());
    assert_eq!(value.validate(), Err(TextFragmentError::Origin));
    let value = fragment(7, "KEEP");
    let encoded = serde_json::to_value(&value).unwrap();
    assert!(encoded.get("message").is_none());
    assert!(encoded.get("block").is_none());
    assert!(encoded.get("verified").is_none());
    assert!(encoded["coverage"].get("total_bytes").is_none());
    let mut injected = encoded;
    injected
        .as_object_mut()
        .unwrap()
        .insert("verified".into(), serde_json::json!(true));
    assert!(serde_json::from_value::<TextFragment>(injected).is_err());
}

#[test]
fn text_fragment_exact_revision_never_resolves_a_current_root() {
    let mut value = fragment(0, "x");
    value.container = Reference::Assertion(crate::RevisionReference {
        root: "assertion".into(),
        revision: None,
    });
    value.source_version = TextSourceVersion::ExactRevision {
        reference: value.container.clone(),
    };
    value.provenance.spans[0].reference = value.reference();
    assert_eq!(value.validate(), Err(TextFragmentError::SourceVersion));
    value.container = Reference::Assertion(crate::RevisionReference::exact(
        "assertion".into(),
        "revision".into(),
    ));
    value.source_version = TextSourceVersion::ExactRevision {
        reference: value.container.clone(),
    };
    value.provenance.spans[0].reference = value.reference();
    assert!(value.validate().is_ok());
}

#[test]
fn portable_fragment_envelope_keeps_versioned_identity_and_sender_commit() {
    let selected = fragment(1, "😀");
    let record = crate::bundle::Record::TextFragment(Box::new(TextFragmentRecord {
        reference: selected.reference().to_string(),
        commit: 17,
        fragment: selected.clone(),
    }));
    record.validate().unwrap();
    assert_eq!(record.reference(), Some("span:message:codex:source#b=1-5"));
    assert_eq!(record.commit(), Some(17));
    let wire = serde_json::to_value(&record).unwrap();
    assert_eq!(wire["record"], "text_fragment");
    assert_eq!(
        serde_json::from_value::<crate::bundle::Record>(wire).unwrap(),
        record
    );
    let mut claims = selected.clone();
    claims.coverage.total_bytes = Some(8);
    claims.provenance.spans.clear();
    assert_eq!(claims.content_key(), selected.content_key());
    let mut different_version = selected.clone();
    different_version.source_version = TextSourceVersion::ContentHash {
        record: selected.container.clone(),
        hash: BlobHash::of(b"another complete canonical source"),
    };
    assert_ne!(different_version.content_key(), selected.content_key());
    let conflict = fragment(1, "xxxx");
    assert_ne!(conflict.content_key(), selected.content_key());
}
