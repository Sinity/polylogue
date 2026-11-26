use super::*;
use polylogue_model::annotation::{AnnotationKind, LinkRole, NewAnnotation, NewAnnotationLink};
use polylogue_model::{Reference, RevisionReference};

fn original_bytes(row: &serde_json::Value, field: &str) -> Vec<u8> {
    let hex = row[field].as_str().unwrap();
    assert_eq!(hex.len() % 2, 0);
    hex.as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

#[test]
fn old_prepared_notes_restore_without_new_text_interpretation() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("pre-markdown-authored.json")).unwrap();
    let originals: Vec<PreparedRequest> =
        serde_json::from_value(fixture["prepared"].clone()).unwrap();
    let packed = fixture["packed_client_requests"].as_array().unwrap();
    assert_eq!(originals.len(), 2);
    for original in &originals {
        let row = packed
            .iter()
            .find(|row| row["id"].as_str() == Some(original.request.id.as_str()))
            .unwrap();
        assert_eq!(
            crate::encoding::unpack_json::<UserRequest>(&original_bytes(row, "request_zstd_hex"))
                .unwrap(),
            original.request
        );
        assert_eq!(
            crate::encoding::unpack_json::<UserOutcome>(&original_bytes(row, "outcome_zstd_hex"))
                .unwrap(),
            original.outcome
        );
        assert_eq!(
            crate::encoding::unpack_json::<polylogue_model::authored::AuthoredEffect>(
                &original_bytes(row, "effect_zstd_hex")
            )
            .unwrap(),
            original.effect
        );
        assert_eq!(
            blake3::Hash::from_bytes(
                crate::authored_records::request_hash(&original.request).unwrap()
            )
            .to_hex()
            .to_string()
            .to_uppercase(),
            row["content_hash_hex"].as_str().unwrap()
        );
    }
    let destination = crate::fixture::writer::WriterArchive::new();
    for original in &originals {
        destination
            .writer
            .restore_authored(original.clone())
            .unwrap();
    }
    let pool = crate::ReadPool::open(&destination.archive).unwrap();
    let position = pool.reader().unwrap().change_position().unwrap();
    for original in &originals {
        assert!(
            destination
                .writer
                .restore_authored(original.clone())
                .unwrap()
                .repeat
        );
    }
    let reader = pool.reader().unwrap();
    assert_eq!(reader.change_position().unwrap(), position);
    let mut restored = vec![];
    reader
        .visit_authored(&mut |record| {
            restored.push(record.clone());
            Ok(())
        })
        .unwrap();
    assert_eq!(restored, originals);
    for original in &originals {
        for fact in &original.effect.annotations {
            let stored = reader
                .annotation(&RevisionReference {
                    root: fact.root.clone(),
                    revision: Some(fact.id.clone()),
                })
                .unwrap()
                .unwrap();
            assert_eq!(stored.fact, *fact);
        }
    }
    let stored_hashes = reader.query(
        "SELECT id,hex(content_hash) AS content_hash_hex FROM source_client_request ORDER BY id", &[]
    ).unwrap();
    for row in stored_hashes {
        let old = packed.iter().find(|old| old["id"] == row["id"]).unwrap();
        assert_eq!(row["content_hash_hex"], old["content_hash_hex"]);
    }

    let live = crate::fixture::writer::WriterArchive::new();
    for original in originals {
        let text = original.effect.annotations[0].value.clone().unwrap();
        let request = crate::fixture::authored::operator_request(
            format!("current-{}", original.request.id),
            UserWrite::Assert {
                annotations: vec![NewAnnotation {
                    kind: AnnotationKind::Note,
                    key: None,
                    value: Some(text.clone()),
                    valid_until: None,
                    links: vec![NewAnnotationLink {
                        role: LinkRole::About,
                        reference: Reference::parse("session:codex:fixture").unwrap(),
                    }],
                }],
            },
        );
        let outcome = live.writer.user(request);
        if original.request.id == "pre-markdown-link" {
            let outcome = outcome.unwrap();
            let pool = crate::ReadPool::open(&live.archive).unwrap();
            let fact = pool
                .reader()
                .unwrap()
                .annotation(&RevisionReference {
                    root: outcome.id.clone(),
                    revision: Some(outcome.id),
                })
                .unwrap()
                .unwrap()
                .fact;
            assert_eq!(fact.value, Some(text));
            assert_eq!(
                fact.links
                    .iter()
                    .map(|link| (link.role, link.reference.to_string()))
                    .collect::<std::collections::BTreeSet<_>>(),
                std::collections::BTreeSet::from([
                    (LinkRole::About, "session:codex:fixture".into()),
                    (LinkRole::Mentions, "url:https://example.test/source".into()),
                ])
            );
        } else {
            match outcome {
                Err(StoreError::Selection { code, .. }) => {
                    assert_eq!(code, polylogue_model::Code::NotBuilt)
                }
                other => panic!("a live session citation must keep its refusal: {other:?}"),
            }
        }
    }
}
