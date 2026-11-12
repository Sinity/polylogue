use super::*;
use polylogue_model::{
    Author, Reference,
    assertion::{AssertionFact, AssertionKind, AssertionLink, LinkRole},
};

fn fact(root: &str, id: &str, kind: AssertionKind, links: &[(LinkRole, &str)]) -> AssertionFact {
    AssertionFact {
        id: id.into(),
        root: root.into(),
        kind,
        key: None,
        value: Some(serde_json::json!(id)),
        author: Author::Unattributed,
        via: polylogue_model::authorship::Channel::Import,
        machine: "synthetic-host".into(),
        created_at: Some(polylogue_model::Timestamp::from_unix_ms(1).unwrap()),
        valid_until: None,
        links: links
            .iter()
            .map(|(role, reference)| AssertionLink {
                role: *role,
                reference: Reference::parse(reference).unwrap(),
                seen_hash: None,
                quote: None,
            })
            .collect(),
    }
}

fn restore(archive: &TestArchive, fact: AssertionFact) {
    archive.writer.restore_assertion(fact).unwrap();
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
        .query_rows(program)
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
        .visit_assertion_references_linked(
            &Reference::parse(about.1).unwrap(),
            &[polylogue_model::assertion::LinkRole::About],
            |row| {
                associated.push(row.revision.expect("linked revisions are exact"));
                Ok(std::ops::ControlFlow::Continue(()))
            },
        )
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
    assert_eq!(exact.fact.created_at, Timestamp::from_unix_ms(1));
    assert_eq!(exact.fact.author, Author::Unattributed);
    assert_eq!(exact.fact.value, Some(serde_json::json!("left")));
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
        .restore_authored(record.prepared.clone())
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

#[test]
fn unknown_authored_time_keeps_currency_queries_and_distinct_source_memberships() {
    use polylogue_model::{RevisionReference, authored::ImportAttribution};
    let archive = test_archive();
    let mut undated = fact(
        "undated",
        "undated",
        AssertionKind::Note,
        &[(LinkRole::About, "session:codex:subject")],
    );
    undated.created_at = None;
    let ingress = |input: &str, at: i64| ImportAttribution {
        input: input.into(),
        chain: 1,
        machine: "receiving-host".into(),
        actor: Author::Unattributed,
        at: Timestamp::from_unix_ms(at).unwrap(),
    };
    assert!(
        !archive
            .writer
            .restore_assertion_from(undated.clone(), ingress("physical-one", 50))
            .unwrap()
            .repeat
    );
    assert!(
        archive
            .writer
            .restore_assertion_from(undated.clone(), ingress("physical-two", 60))
            .unwrap()
            .repeat
    );
    assert!(
        archive
            .writer
            .restore_assertion_from(undated.clone(), ingress("physical-one", 70))
            .unwrap()
            .repeat
    );
    let mut dated = undated.clone();
    dated.id = "dated".into();
    dated.root = "dated".into();
    dated.created_at = Timestamp::from_unix_ms(5);
    archive.writer.restore_assertion(dated.clone()).unwrap();
    let mut conflicting = dated.clone();
    conflicting.created_at = None;
    assert!(archive.writer.restore_assertion(conflicting).is_err());
    let reads = ReadPool::open(&archive.archive).unwrap();
    let reader = reads.reader().unwrap();
    assert_eq!(selected(&reader, "assertions"), ["dated", "undated"]);
    assert_eq!(
        selected(&reader, "assertions where created_at:unknown"),
        ["undated"]
    );
    assert_eq!(
        selected(&reader, "assertions where created_at:known"),
        ["dated"]
    );
    let reference = RevisionReference::exact("undated".into(), "undated".into());
    let restored = reader.assertion(&reference).unwrap().unwrap();
    assert_eq!(restored.fact, undated);
    let ingresses = reader.assertion_imports(&reference).unwrap().unwrap();
    assert_eq!(
        ingresses,
        [ingress("physical-one", 50), ingress("physical-two", 60)]
    );
    assert_eq!(
        reader
            .assertion(&RevisionReference::exact("dated".into(), "dated".into()))
            .unwrap()
            .unwrap()
            .fact,
        dated
    );
}
