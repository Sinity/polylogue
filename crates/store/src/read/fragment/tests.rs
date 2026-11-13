use super::*;
use crate::fixture::writer::{WriterArchive, message};
use polylogue_model::bundle::fragments::{
    TextCoverage, TextFragment, TextFragmentProvenance, TextSpanOrigin,
};
use polylogue_model::{Block, MessageRow, Output, Role, Session, SessionKind, batch::FragmentRow};

fn record(start: u64, text: &str, total: Option<u64>, version: BlobHash) -> TextFragmentRecord {
    let container = Reference::parse("message:claude-code:s-m").unwrap();
    let end = start + text.len() as u64;
    let fragment = TextFragment {
        source_version: TextSourceVersion::ContentHash {
            record: container.clone(),
            hash: version,
        },
        container: container.clone(),
        coverage: TextCoverage {
            start,
            end,
            total_bytes: total,
        },
        text: text.into(),
        text_hash: BlobHash::of(text.as_bytes()),
        provenance: TextFragmentProvenance {
            spans: vec![TextSpanOrigin {
                reference: Reference::Span {
                    container: Box::new(container),
                    unit: SpanUnit::Bytes,
                    start,
                    end,
                },
                start,
                end,
            }],
            sources: vec![polylogue_model::bundle::SourceClaim {
                input: "foreign-input".into(),
                source: "provider".into(),
                source_chain: 7,
                source_start: Some(400),
                source_end: Some(900),
            }],
            imports: vec![],
        },
    };
    TextFragmentRecord {
        reference: fragment.reference().to_string(),
        commit: 900,
        fragment,
    }
}

fn capture(w: &mut WriterArchive, input: &str, record: TextFragmentRecord, admitted: Option<u64>) {
    capture_rows(w, input, record, admitted, false);
}

fn capture_rows(
    w: &mut WriterArchive,
    input: &str,
    record: TextFragmentRecord,
    admitted: Option<u64>,
    historical: bool,
) {
    let mut bytes = serde_json::to_vec(&record).unwrap();
    let end = bytes.len() as u64;
    bytes.push(b'\n');
    let output = Output {
        fragments: vec![FragmentRow {
            offset: 0,
            end,
            record,
        }],
        ..Output::default()
    };
    let mut batch =
        crate::fixture::writer::step(input, "polylogue-bundle", "/p/bundle.jsonl", output, 0, 1);
    batch.length = bytes.len() as u64;
    batch.parse[0].parsed_upto = batch.length;
    let mut raw = batch.clone();
    raw.parse.clear();
    let blob = w.archive.prepare_bytes(bytes.as_slice()).unwrap();
    let stored = w.writer.store_bytes(blob, raw).unwrap();
    assert!(!historical || admitted.is_none());
    batch.change = crate::Change::Reparse {
        chain: stored.chain,
    };
    batch.observation = None;
    w.writer.send(batch).unwrap();
    assert!(w.writer.flush().unwrap().errors.is_empty());
    // Reader fixtures vary retained admission evidence after the ordinary
    // writer has allocated the input and committed its fragment membership.
    if historical {
        crate::archive::open_write(&w.archive.archive_db(), 1)
            .unwrap()
            .execute("DELETE FROM extent WHERE input=?1", [input])
            .unwrap();
    }
    if historical || admitted.is_some() {
        let index = crate::archive::open_write(&w.archive.index_db(), 1).unwrap();
        let changed = index.execute(
            "UPDATE text_fragment_source SET admitted=?2 WHERE input_pk=(SELECT pk FROM input_ref WHERE id=?1)",
            rusqlite::params![input, admitted.map(crate::fragment::receipt).transpose().unwrap()],
        ).unwrap();
        assert_eq!(changed, 1);
    }
}

fn source() -> polylogue_model::Message {
    message(
        "m",
        "s",
        Role::User,
        1,
        vec![Block::Text {
            text: "héllo".into(),
        }],
    )
}
fn read(
    reader: &Reader<'_>,
    version: Option<&TextSourceVersion>,
    start: u64,
    end: u64,
) -> (FragmentAvailability, String) {
    let container = Reference::parse("message:claude-code:s-m").unwrap();
    let mut text = String::new();
    let availability = reader
        .visit_fragment_text(&container, version, start, end, |slice| {
            text.push_str(slice);
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    (availability, text)
}

#[test]
fn fragment_only_utf8_union_preserves_unknown_outside_and_never_attests_a_full_hash() {
    let mut w = WriterArchive::new();
    let hash = BlobHash::of(b"unavailable complete provider record");
    capture(
        &mut w,
        "fragment-left",
        record(0, "hé", Some(6), hash),
        Some(130),
    );
    capture(
        &mut w,
        "fragment-right",
        record(3, "llo", Some(6), hash),
        Some(140),
    );
    capture(
        &mut w,
        "fragment-duplicate",
        record(0, "hé", Some(6), hash),
        Some(150),
    );
    let pool = crate::ReadPool::open(&w.archive).unwrap();
    let reader = pool.reader().unwrap();
    let (availability, text) = read(&reader, None, 0, 6);
    assert!(matches!(
        availability,
        FragmentAvailability::Declared { .. }
    ));
    assert_eq!(text, "héllo");
    let selected = reader
        .select_fragment_span(
            &Reference::parse("message:claude-code:s-m").unwrap(),
            None,
            0,
            6,
        )
        .unwrap();
    assert_eq!(
        selected.coverage,
        TextCoverage {
            start: 0,
            end: 6,
            total_bytes: Some(6)
        }
    );
    assert_eq!(selected.admitted, Some(150));
    assert!(
        reader
            .message_with_projection(&source().id)
            .unwrap()
            .is_none()
    );
    assert_eq!(
        read(&reader, None, 0, 7),
        (FragmentAvailability::Unknown, String::new())
    );
    let mut emitted = false;
    assert!(
        reader
            .visit_fragment_text(
                &Reference::parse("message:claude-code:s-m").unwrap(),
                None,
                2,
                3,
                |_| {
                    emitted = true;
                    Ok(ControlFlow::Continue(()))
                }
            )
            .is_err()
    );
    assert!(!emitted);
    let mut sources = Vec::new();
    reader
        .visit_fragment_claims(
            &Reference::parse("message:claude-code:s-m").unwrap(),
            None,
            |claim| {
                sources.push((claim.input, claim.admitted, claim.record.commit));
                ControlFlow::Continue(())
            },
        )
        .unwrap();
    sources.sort();
    assert_eq!(
        sources,
        vec![
            ("fragment-duplicate".into(), Some(150), 900),
            ("fragment-left".into(), Some(130), 900),
            ("fragment-right".into(), Some(140), 900)
        ]
    );
}

#[test]
fn contradictory_claims_and_versions_remain_exportable_without_selecting_an_arrival_winner() {
    let mut w = WriterArchive::new();
    let version = BlobHash::of(b"version one");
    capture(&mut w, "first", record(0, "hello", Some(5), version), None);
    capture(&mut w, "second", record(1, "XX", Some(5), version), None);
    let other = BlobHash::of(b"version two");
    capture(&mut w, "other", record(0, "newer", Some(5), other), None);
    let pool = crate::ReadPool::open(&w.archive).unwrap();
    let reader = pool.reader().unwrap();
    assert_eq!(
        read(&reader, None, 0, 5),
        (FragmentAvailability::Ambiguous, String::new())
    );
    let owner = Reference::parse("message:claude-code:s-m").unwrap();
    let first = TextSourceVersion::ContentHash {
        record: owner.clone(),
        hash: version,
    };
    let (availability, text) = read(&reader, Some(&first), 0, 5);
    assert_eq!(
        availability,
        FragmentAvailability::Conflict {
            version: first,
            reason: FragmentConflict::Overlap
        }
    );
    assert!(text.is_empty());
    let second = TextSourceVersion::ContentHash {
        record: owner.clone(),
        hash: other,
    };
    assert_eq!(
        read(&reader, Some(&second), 0, 5),
        (
            FragmentAvailability::Declared { version: second },
            "newer".into()
        )
    );
    let mut claims = 0;
    reader
        .visit_fragment_claims(&owner, None, |_| {
            claims += 1;
            ControlFlow::Continue(())
        })
        .unwrap();
    assert_eq!(claims, 3);
}

#[test]
fn complete_canonical_source_refutes_foreign_claim_without_losing_it_or_its_receipt() {
    let mut w = WriterArchive::new();
    let source = source();
    let version = source.content_hash().unwrap();
    capture(
        &mut w,
        "foreign-fragment",
        record(0, "WRONG!", Some(6), version),
        Some(130),
    );
    w.write("s", vec![source.clone()]);
    let owner = Reference::Message(source.id.clone());
    let expected = FragmentAvailability::Canonical {
        version: TextSourceVersion::ContentHash {
            record: owner.clone(),
            hash: version,
        },
        claims: FragmentClaims::Conflict(FragmentConflict::CanonicalSource),
    };
    let pool = crate::ReadPool::open(&w.archive).unwrap();
    {
        let reader = pool.reader().unwrap();
        assert_eq!(
            read(&reader, None, 0, 6),
            (expected.clone(), "héllo".into())
        );
        assert_eq!(
            reader
                .message_with_projection(&source.id)
                .unwrap()
                .unwrap()
                .0,
            source
        );
    }
    crate::archive::open_write(&w.archive.archive_db(), 1)
        .unwrap()
        .execute("DELETE FROM extent WHERE input='foreign-fragment'", [])
        .unwrap();
    let reader = pool.reader().unwrap();
    assert_eq!(read(&reader, None, 0, 6), (expected, "héllo".into()));
    let mut retained = Vec::new();
    reader
        .visit_fragment_claims(&owner, None, |claim| {
            retained.push(claim);
            ControlFlow::Continue(())
        })
        .unwrap();
    assert_eq!(retained.len(), 1);
    assert_eq!(retained[0].admitted, Some(130));
    assert_eq!(retained[0].record.fragment.text, "WRONG!");
    assert_eq!(
        retained[0].record.fragment.provenance.sources[0].input,
        "foreign-input"
    );
}

#[test]
fn sparse_message_metadata_and_complete_text_coverage_do_not_grant_canonical_attestation() {
    let mut w = WriterArchive::new();
    let mut full = source();
    full.blocks.push(Block::ToolCall {
        written_text: None,
        run_by_script: false,
        call_id: Some("tool".into()),
        tool: Some("provider-tool".into()),
        action: Default::default(),
        input: serde_json::json!({"unavailable":"provider fields"}),
        command: None,
        path: None,
    });
    let version = full.content_hash().unwrap();
    let mut partial = full.clone();
    partial.blocks.pop();
    let mut output = Output {
        sessions: vec![Session::new(full.session.clone(), SessionKind::Main)],
        messages: vec![MessageRow {
            page_text: false,
            id_kind: polylogue_model::IdKind::Native,
            offset: 0,
            end: 1,
            message: partial,
        }],
        ..Output::default()
    };
    output.message_projections.insert(
        0,
        polylogue_model::bundle::MessageProjection {
            source_version: version,
            block_count: 2,
            block_ordinals: vec![0],
        },
    );
    w.write_output(
        "partial-message",
        "polylogue-bundle",
        "/p/partial.jsonl",
        output,
        1,
    );
    capture(
        &mut w,
        "all-text",
        record(0, "héllo", Some(6), version),
        None,
    );
    let pool = crate::ReadPool::open(&w.archive).unwrap();
    let reader = pool.reader().unwrap();
    assert!(matches!(
        read(&reader, None, 0, 6).0,
        FragmentAvailability::Declared { .. }
    ));
    assert_eq!(
        reader
            .message_with_projection(&full.id)
            .unwrap()
            .unwrap()
            .1
            .block_ordinals,
        vec![0]
    );
}

#[test]
fn contradictory_totals_conflict_even_when_claim_intervals_are_disjoint() {
    let mut w = WriterArchive::new();
    let version = BlobHash::of(b"same complete value");
    capture(
        &mut w,
        "total-ten",
        record(0, "ab", Some(10), version),
        None,
    );
    capture(
        &mut w,
        "total-eleven",
        record(5, "fg", Some(11), version),
        None,
    );
    let pool = crate::ReadPool::open(&w.archive).unwrap();
    let reader = pool.reader().unwrap();
    let (availability, text) = read(&reader, None, 0, 2);
    assert!(matches!(
        availability,
        FragmentAvailability::Conflict {
            reason: FragmentConflict::Total,
            ..
        }
    ));
    assert!(text.is_empty());
}

#[test]
fn complete_source_checks_original_normalization_before_attesting_matching_selected_bytes() {
    let mut w = WriterArchive::new();
    let source = source();
    let version = source.content_hash().unwrap();
    let mut claim = record(0, "hé", Some(6), version);
    claim.fragment.provenance.spans[0].reference = Reference::Span {
        container: Box::new(Reference::Message(source.id.clone())),
        unit: SpanUnit::CodePoints,
        start: 0,
        end: 3,
    };
    // The sender claims these three characters normalize to bytes 0..3;
    // complete text proves they cover bytes 0..4 instead.
    capture(&mut w, "wrong-normalization", claim, None);
    w.write("s", vec![source]);
    let pool = crate::ReadPool::open(&w.archive).unwrap();
    let reader = pool.reader().unwrap();
    let (availability, text) = read(&reader, None, 0, 3);
    assert!(matches!(
        availability,
        FragmentAvailability::Canonical {
            claims: FragmentClaims::Conflict(FragmentConflict::CanonicalSource),
            ..
        }
    ));
    assert_eq!(text, "hé");
}

#[test]
fn matching_full_document_attests_fragments_without_using_a_message_container() {
    use polylogue_model::{Document, DocumentAuthorship, DocumentId, DocumentPurpose, DocumentRow};
    let mut w = WriterArchive::new();
    let document = Document {
        id: DocumentId::parse("claude-code:doc").unwrap(),
        purpose: DocumentPurpose::Context,
        title: None,
        source_time: None,
        source_time_quality: polylogue_model::TimeQuality::Unknown,
        source_time_precision: None,
        source_time_until: None,
        authorship: DocumentAuthorship::Unknown,
        blocks: vec![Block::Text {
            text: "héllo".into(),
        }],
        metadata: polylogue_model::Json::Null,
    };
    let version = BlobHash(
        crate::encoding::RowPacker::new()
            .unwrap()
            .pack_document(&document)
            .unwrap()
            .hash,
    );
    let container = Reference::Document(document.id.clone());
    let mut claim = record(0, "hé", Some(6), version);
    claim.fragment.container = container.clone();
    claim.fragment.source_version = TextSourceVersion::ContentHash {
        record: container.clone(),
        hash: version,
    };
    claim.fragment.provenance.spans[0].reference = Reference::Span {
        container: Box::new(container.clone()),
        unit: SpanUnit::Bytes,
        start: 0,
        end: 3,
    };
    claim.reference = claim.fragment.reference().to_string();
    capture(&mut w, "document-fragment", claim, None);
    w.write_output(
        "native-document",
        "synthetic-provider",
        "/p/document",
        Output {
            documents: vec![DocumentRow {
                offset: 0,
                end: 1,
                document,
            }],
            ..Output::default()
        },
        1,
    );
    let pool = crate::ReadPool::open(&w.archive).unwrap();
    let reader = pool.reader().unwrap();
    let mut text = String::new();
    let available = reader
        .visit_fragment_text(&container, None, 0, 3, |chunk| {
            text.push_str(chunk);
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    assert!(matches!(
        available,
        FragmentAvailability::Canonical {
            claims: FragmentClaims::Compatible,
            ..
        }
    ));
    assert_eq!(text, "hé");
    let selected = reader.select_fragment_span(&container, None, 0, 3).unwrap();
    let Reference::Document(id) = &container else {
        unreachable!()
    };
    assert_eq!(selected.admitted, reader.document_commit(id).unwrap());
    assert_eq!(selected.coverage.total_bytes, Some(6));
}

#[test]
fn fragment_selection_keeps_unknown_receipt_instead_of_using_sender_or_read_position() {
    let mut w = WriterArchive::new();
    let version = BlobHash::of(b"unavailable complete source");
    capture(
        &mut w,
        "proven",
        record(0, "hello", Some(5), version),
        Some(130),
    );
    capture_rows(
        &mut w,
        "historical",
        record(0, "hello", Some(5), version),
        None,
        true,
    );
    let pool = crate::ReadPool::open(&w.archive).unwrap();
    let reader = pool.reader().unwrap();
    let selected = reader
        .select_fragment_span(
            &Reference::parse("message:claude-code:s-m").unwrap(),
            None,
            0,
            5,
        )
        .unwrap();
    assert!(matches!(
        selected.availability,
        FragmentAvailability::Declared { .. }
    ));
    assert_eq!(selected.coverage.total_bytes, Some(5));
    assert_eq!(selected.admitted, None);
    assert_eq!(read(&reader, None, 0, 5).1, "hello");
}

#[test]
fn exact_variant_part_bytes_do_not_borrow_the_earlier_header_admission() {
    use polylogue_model::{Variant, VariantEffect, VariantKind, VariantPart, VariantRelation};
    let w = WriterArchive::new();
    let conn = crate::archive::open_write(&w.archive.archive_db(), 1).unwrap();
    let header = Variant {
        id: "variant-rev".into(),
        root: "variant-root".into(),
        base: None,
        source: "message:codex:source".into(),
        source_hash: BlobHash::of(b"source"),
        kind: VariantKind::Translation,
        from_lang: Some("en".into()),
        to_lang: Some("pl".into()),
        author: polylogue_model::Author::Agent(None),
        via: "mcp".into(),
        method: None,
        machine: "machine".into(),
        created_at: polylogue_model::Timestamp::from_unix_ms(1).unwrap(),
        commit_seq: 999,
    };
    crate::variants::apply(
        &conn,
        &VariantEffect {
            variants: vec![header],
            variant_parts: vec![],
        },
        10,
    )
    .unwrap();
    crate::variants::apply(
        &conn,
        &VariantEffect {
            variants: vec![],
            variant_parts: vec![VariantPart {
                variant: "variant-rev".into(),
                n: 0,
                of: "block:codex:source#0".into(),
                of_hash: BlobHash::of(b"original"),
                relation: VariantRelation::Translates,
                body: "late".into(),
            }],
        },
        20,
    )
    .unwrap();
    let pool = crate::ReadPool::open(&w.archive).unwrap();
    let reader = pool.reader().unwrap();
    assert_eq!(
        reader.variant("variant-rev").unwrap().unwrap().commit_seq,
        10
    );
    let selected = reader
        .select_fragment_span(
            &Reference::parse("variant:variant-root@variant-rev#0").unwrap(),
            None,
            0,
            4,
        )
        .unwrap();
    assert!(matches!(
        selected.availability,
        FragmentAvailability::Canonical { .. }
    ));
    assert_eq!(selected.coverage.total_bytes, Some(4));
    assert_eq!(selected.admitted, None);
}

#[test]
fn fragment_claim_ordinal_skips_poisoned_prefix_before_decoding() {
    let mut w = WriterArchive::new();
    let version = BlobHash::of(b"complete value");
    capture(
        &mut w,
        "earlier",
        record(0, "ab", Some(4), version),
        Some(130),
    );
    capture(
        &mut w,
        "later",
        record(2, "cd", Some(4), version),
        Some(140),
    );
    let conn = crate::archive::open_write(&w.archive.index_db(), 1).unwrap();
    conn.execute(
        "UPDATE text_fragment SET record=x'00' WHERE start=?1",
        [0u64.to_be_bytes().as_slice()],
    )
    .unwrap();
    let pool = crate::ReadPool::open(&w.archive).unwrap();
    let reader = pool.reader().unwrap();
    let owner = Reference::parse("message:claude-code:s-m").unwrap();
    assert!(
        reader
            .visit_fragment_claims(&owner, None, |_| ControlFlow::Continue(()))
            .is_err()
    );
    let mut found = Vec::new();
    reader
        .visit_fragment_claims_from_ordinal(&owner, None, 1, |claim| {
            found.push(claim.record.fragment.text);
            ControlFlow::Break(())
        })
        .unwrap();
    assert_eq!(found, vec!["cd"]);
    for ordinal in [2, u64::MAX] {
        reader
            .visit_fragment_claims_from_ordinal(&owner, None, ordinal, |_| panic!("past claim end"))
            .unwrap();
    }
}

#[test]
fn fragment_text_delivery_stops_successfully_and_preserves_consumer_errors() {
    let mut w = WriterArchive::new();
    let version = BlobHash::of(b"unavailable source");
    capture(&mut w, "left", record(0, "hé", Some(6), version), Some(130));
    capture(
        &mut w,
        "right",
        record(3, "llo", Some(6), version),
        Some(140),
    );
    let owner = Reference::parse("message:claude-code:s-m").unwrap();
    let pool = crate::ReadPool::open(&w.archive).unwrap();
    {
        let reader = pool.reader().unwrap();
        let mut pieces = Vec::new();
        let available = reader
            .visit_fragment_text(&owner, None, 0, 6, |text| {
                pieces.push(text.to_string());
                Ok(ControlFlow::Break(()))
            })
            .unwrap();
        assert!(matches!(available, FragmentAvailability::Declared { .. }));
        assert_eq!(pieces, vec!["hé"]);
        assert!(
            matches!(reader.visit_fragment_text(&owner,None,0,6,|_|Err(StoreError::Invalid("consumer refusal".into()))),Err(StoreError::Invalid(reason)) if reason=="consumer refusal")
        );
    }
    let mut full = source();
    full.blocks = vec![
        Block::Text { text: "hé".into() },
        Block::Text { text: "llo".into() },
    ];
    w.write("s", vec![full]);
    let reader = pool.reader().unwrap();
    let mut pieces = Vec::new();
    let available = reader
        .visit_fragment_text(&owner, None, 0, 8, |text| {
            pieces.push(text.to_string());
            Ok(ControlFlow::Break(()))
        })
        .unwrap();
    assert!(matches!(available, FragmentAvailability::Canonical { .. }));
    assert_eq!(pieces, vec!["hé"]);
}
