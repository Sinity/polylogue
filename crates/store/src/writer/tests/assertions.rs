use super::*;
use polylogue_model::{
    Author, Reference,
    assertion::{AssertionFact, AssertionKind, AssertionLink, LinkRole},
    authored::ImportAttribution,
};

fn fact(root: &str, id: &str, kind: AssertionKind, links: &[(LinkRole, &str)]) -> AssertionFact {
    AssertionFact {
        id: id.into(),
        root: root.into(),
        kind,
        key: None,
        value: Some(serde_json::json!(id)),
        author: Author::Unattributed,
        via: None,
        machine: None,
        created_at: None,
        valid_until: None,
        links: links
            .iter()
            .map(|(role, reference)| AssertionLink {
                role: *role,
                reference: Reference::parse(reference).unwrap(),
                seen_hash: None,
            })
            .collect(),
    }
}

fn restore(archive: &TestArchive, fact: AssertionFact) {
    archive
        .writer
        .restore_assertion(
            fact,
            vec![],
            ImportAttribution {
                input: "synthetic-import".into(),
                chain: 1,
                machine: "receiver".into(),
                actor: Author::Operator,
                at: Timestamp::from_unix_ms(1).unwrap(),
            },
        )
        .unwrap();
}

fn selected(reader: &crate::Reader<'_>, text: &str) -> Vec<String> {
    let options = polylogue_query::Options {
        clock: query_language::time::Clock::new(
            "2026-10-29T12:00:00Z".parse().unwrap(),
            query_language::time::DaySetting::new(
                jiff::tz::TimeZone::get("UTC").unwrap(),
                jiff::civil::time(4, 0, 0, 0),
            )
            .unwrap(),
        ),
        origins: vec!["codex".into()],
        command_words: vec![],
        page_size: None,
        random_seed: None,
        embeddings: false,
        self_session: None,
        saved: Default::default(),
        fields: vec![],
        kept: Default::default(),
        chat_pages: vec![],
    };
    let selection = reader.prepare_selection(text, &options).unwrap();
    let program = selection.program();
    let mut ids: Vec<_> = reader
        .query(&program.sql, &crate::selection::values(&program.params))
        .unwrap()
        .into_iter()
        .map(|row| row["id"].as_str().unwrap().to_owned())
        .collect();
    ids.sort();
    ids
}

#[test]
fn canonical_query_and_typed_about_share_forks_withdrawal_and_snapshot_retirement() {
    let archive = test_archive();
    let about = (LinkRole::About, "session:codex:subject");
    // The base arrives last; its larger local admission cannot replace heads.
    for id in ["right", "left"] {
        restore(
            &archive,
            fact(
                "family",
                id,
                AssertionKind::Note,
                &[about, (LinkRole::Revises, "assertion:family@base")],
            ),
        );
    }
    restore(
        &archive,
        fact("family", "base", AssertionKind::Note, &[about]),
    );
    let reads = ReadPool::open(&archive.archive).unwrap();
    let pinned = reads.reader().unwrap();
    assert_eq!(selected(&pinned, "assertions"), ["left", "right"]);
    assert_eq!(
        selected(&pinned, "assertions where conflict:true"),
        ["left", "right"]
    );
    let mut associated = Vec::new();
    let flow = pinned
        .visit_assertions_about(&Reference::parse(about.1).unwrap(), |row| {
            associated.push(row.fact.id);
            Ok(std::ops::ControlFlow::Continue(()))
        })
        .unwrap();
    assert!(matches!(flow, std::ops::ControlFlow::Continue(())));
    assert_eq!(associated, ["left", "right"]);
    restore(
        &archive,
        fact(
            "withdrawal",
            "withdrawal",
            AssertionKind::Withdrawal,
            &[(LinkRole::Withdraws, "assertion:family@left")],
        ),
    );
    assert_eq!(selected(&pinned, "assertions"), ["left", "right"]);
    drop(pinned);
    let fresh = reads.reader().unwrap();
    assert_eq!(selected(&fresh, "assertions"), ["right"]);
    assert!(selected(&fresh, "assertions where conflict:true").is_empty());
    let exact = fresh
        .assertion(&polylogue_model::RevisionReference::exact(
            "family".into(),
            "left".into(),
        ))
        .unwrap()
        .unwrap();
    assert_eq!(exact.fact.created_at, None);
    assert_eq!(exact.fact.author, Author::Unattributed);
    assert_eq!(exact.fact.value, Some(serde_json::json!("left")));
}

#[test]
fn historical_bundle_restore_preserves_original_bytes_outcome_and_whole_effect_membership() {
    use polylogue_model::historical::{ConservedAccepted, HistoricalRecord, SourceSchema};
    let request = br#"{"id":"old-action","machine":"sender","at":"1970-01-01T00:00:00.001Z","write":{"write":"add_note","note":{"kind":"note","value":"recorded text","targets":["repo:synthetic"]}}}"#;
    let intent = br#"["sender",{"write":"add_note","note":{"kind":"note","value":"recorded text","targets":["repo:synthetic"]}}]"#;
    let outcome = br#"{"id":"old-action","repeat":false,"pending":false}"#;
    let effect = br#"{"notes":[{"note":{"id":"old-action","first_note":"old-action","kind":"note","value":"recorded text","author":"operator","created_at":"1970-01-01T00:00:00.001Z","targets":["repo:synthetic"]},"machine":"sender","revises":[]}],"withdrawals":[],"settings":[],"variants":{"variants":[],"variant_parts":[]},"kept_results":[],"deliveries":[],"chats":[],"backfills":[]}"#;
    let ConservedAccepted::Historical(record) = crate::assertions::conversion::conserve_request(
        SourceSchema::PolylogueArchive { version: 4 },
        request,
        Some(outcome),
        Some(effect),
        Some(BlobHash::of(intent)),
        None,
    )
    .unwrap() else {
        panic!("retired action must remain historical")
    };
    let first = test_archive();
    let current = ImportAttribution {
        input: "historical-package".into(),
        chain: 1,
        machine: "receiver".into(),
        actor: Author::Operator,
        at: Timestamp::from_unix_ms(10).unwrap(),
    };
    let restored = first
        .writer
        .restore_history(
            HistoricalRecord::Accepted(record.clone()),
            vec![],
            current.clone(),
        )
        .unwrap();
    assert!(!restored.repeat && !restored.pending);
    let mut later = current.clone();
    later.at = Timestamp::from_unix_ms(20).unwrap();
    assert!(
        first
            .writer
            .restore_history(HistoricalRecord::Accepted(record.clone()), vec![], later)
            .unwrap()
            .repeat
    );
    let reads = ReadPool::open(&first.archive).unwrap();
    let reader = reads.reader().unwrap();
    let selected = Reference::parse("assertion:old-action@old-action").unwrap();
    let Some(crate::read::authored_evidence::AuthoredReadEvidence::Historical {
        record: exported,
        ..
    }) = reader.authored_record(&selected).unwrap()
    else {
        panic!("whole historical accepted evidence missing")
    };
    let wire = polylogue_model::bundle::Record::Historical(exported.clone());
    wire.validate().unwrap();
    let decoded: polylogue_model::bundle::Record =
        serde_json::from_slice(&serde_json::to_vec(&wire).unwrap()).unwrap();
    assert_eq!(decoded, wire);
    let HistoricalRecord::Accepted(original) = &exported.evidence else {
        panic!("wrong historical grain")
    };
    assert_eq!(
        original.original.as_ref().unwrap().request.as_deref(),
        Some(request.as_slice())
    );
    assert_eq!(
        original.original.as_ref().unwrap().effect.as_deref(),
        Some(effect.as_slice())
    );
    assert_eq!(
        original.original.as_ref().unwrap().outcome.as_deref(),
        Some(outcome.as_slice())
    );
    assert_eq!(original.authorship, None);
    assert_eq!(original.imports.observed, [current]);
    let second = test_archive();
    second
        .writer
        .restore_history(
            exported.evidence.clone(),
            exported.imports.clone(),
            ImportAttribution {
                input: "second-package".into(),
                chain: 1,
                machine: "second-receiver".into(),
                actor: Author::Operator,
                at: Timestamp::from_unix_ms(30).unwrap(),
            },
        )
        .unwrap();
    let reads = ReadPool::open(&second.archive).unwrap();
    let reader = reads.reader().unwrap();
    let copied = reader
        .historical_accepted_request("old-action")
        .unwrap()
        .unwrap();
    assert_eq!(copied.original, original.original);
    assert_eq!(copied.outcome, original.outcome);
    assert_eq!(copied.conserved_effects, original.conserved_effects);
    assert_eq!(copied.imports.observed.len(), 1);
    assert!(
        copied
            .imports
            .declared
            .contains(&original.imports.observed[0])
    );
    assert!(
        reader
            .assertion(&polylogue_model::RevisionReference::exact(
                "old-action".into(),
                "old-action".into()
            ))
            .is_err(),
        "historical values never become validated current-kind inputs"
    );
}

#[test]
fn explicit_native_revision_uses_historical_recorded_text_and_current_value_validation() {
    use polylogue_model::{
        assertion::{AssertionPatch, Change},
        authorship::{Authorship, Channel},
        historical::{
            HistoricalAssertionEvidence, HistoricalRecord, NoteKind, RecordedKind, SourceSchema,
        },
        user::{UserRequest, UserWrite},
    };
    let archive = test_archive();
    let fact = HistoricalAssertionEvidence {
        id: "old-base".into(),
        root: "old-family".into(),
        source: SourceSchema::PolylogueArchive { version: 4 },
        kind: RecordedKind::Known(NoteKind::Note),
        key: None,
        value: Some("original text".into()),
        recorded_author: None,
        machine: None,
        created_at: None,
        valid_until: None,
        targets: vec![],
        relations: vec![],
        links: vec![],
        predecessor: None,
    };
    archive
        .writer
        .restore_history(
            HistoricalRecord::Assertion(Box::new(fact.clone())),
            vec![],
            ImportAttribution {
                input: "source".into(),
                chain: 1,
                machine: "receiver".into(),
                actor: Author::Operator,
                at: Timestamp::from_unix_ms(1).unwrap(),
            },
        )
        .unwrap();
    let request = UserRequest {
        id: "native-revision".into(),
        machine: "actual-machine".into(),
        at: Timestamp::from_unix_ms(2).unwrap(),
        authorship: Authorship {
            author: Author::Operator,
            via: Channel::Cli,
        },
        write: UserWrite::ReviseAssertions {
            bases: vec![polylogue_model::RevisionReference::exact(
                "old-family".into(),
                "old-base".into(),
            )],
            patch: AssertionPatch {
                value: Change::Set(serde_json::json!("current text")),
                links: Some(vec![polylogue_model::assertion::NewAssertionLink {
                    role: LinkRole::About,
                    reference: Reference::parse("session:claude-code:synthetic").unwrap(),
                }]),
                ..Default::default()
            },
        },
    };
    archive.writer.user(request.clone()).unwrap();
    let pool = ReadPool::open(&archive.archive).unwrap();
    let reader = pool.reader().unwrap();
    let current = reader
        .assertion(&polylogue_model::RevisionReference {
            root: "old-family".into(),
            revision: None,
        })
        .unwrap()
        .unwrap();
    assert_eq!(current.fact.id, "native-revision");
    assert_eq!(current.fact.machine.as_deref(), Some("actual-machine"));
    assert_eq!(current.fact.via, Some(Channel::Cli));
    assert_eq!(current.fact.value, Some(serde_json::json!("current text")));
    assert_eq!(
        current
            .fact
            .links
            .iter()
            .find(|link| link.role == LinkRole::Revises)
            .unwrap()
            .seen_hash,
        Some(BlobHash::of(b"original text"))
    );
    let old = reader
        .assertion_evidence(&polylogue_model::RevisionReference::exact(
            "old-family".into(),
            "old-base".into(),
        ))
        .unwrap()
        .unwrap();
    assert!(
        matches!(old, polylogue_model::historical::AssertionEvidence::Historical { fact: old, .. } if old == fact)
    );
    let mut stale = request;
    stale.id = "stale-revision".into();
    assert!(matches!(
        archive.writer.user(stale),
        Err(StoreError::Stale { .. })
    ));
}

#[test]
fn exact_no_effect_request_exports_its_original_action_outcome_and_provenance() {
    use polylogue_model::{
        assertion::NewAssertion,
        authorship::{Authorship, Channel},
        user::{UserRequest, UserWrite},
    };
    let archive = test_archive();
    let request = |id: &str| UserRequest {
        id: id.into(),
        machine: "actual-machine".into(),
        at: Timestamp::from_unix_ms(1).unwrap(),
        authorship: Authorship {
            author: Author::Operator,
            via: Channel::Cli,
        },
        write: UserWrite::Assert {
            assertions: vec![NewAssertion {
                kind: AssertionKind::Tag,
                key: Some("synthetic-tag".into()),
                value: None,
                valid_until: None,
                links: vec![polylogue_model::assertion::NewAssertionLink {
                    role: LinkRole::About,
                    reference: Reference::parse("session:claude-code:synthetic").unwrap(),
                }],
            }],
        },
    };
    archive.writer.user(request("tag-base")).unwrap();
    let original = request("no:effect@#..request");
    archive.writer.user(original.clone()).unwrap();
    let pool = ReadPool::open(&archive.archive).unwrap();
    let reader = pool.reader().unwrap();
    let reference = Reference::Request(original.id.clone());
    let Some(crate::read::authored_evidence::AuthoredReadEvidence::Local { selected, record }) =
        reader.authored_record(&reference).unwrap()
    else {
        panic!("exact no-effect accepted request missing")
    };
    assert_eq!(selected, reference);
    assert_eq!(record.prepared.request, original);
    assert!(record.prepared.effect.assertions.is_empty());
    assert_eq!(record.prepared.outcome.id, "tag-base");
    assert!(record.commit > 0);
    let wire = polylogue_model::bundle::Record::Authored(record.clone());
    wire.validate().unwrap();
    let decoded: polylogue_model::bundle::Record =
        serde_json::from_slice(&serde_json::to_vec(&wire).unwrap()).unwrap();
    assert_eq!(decoded, wire);
    let other = test_archive();
    other
        .writer
        .restore_authored(
            record.prepared.clone(),
            record.imports.clone(),
            ImportAttribution {
                input: "no-effect-package".into(),
                chain: 1,
                machine: "receiver".into(),
                actor: Author::Operator,
                at: Timestamp::from_unix_ms(2).unwrap(),
            },
        )
        .unwrap();
    let pool = ReadPool::open(&other.archive).unwrap();
    let reader = pool.reader().unwrap();
    let Some(crate::read::authored_evidence::AuthoredReadEvidence::Local {
        record: copied, ..
    }) = reader.authored_record(&reference).unwrap()
    else {
        panic!("import lost the no-effect request")
    };
    assert_eq!(copied.prepared, record.prepared);
}
