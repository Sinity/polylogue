use super::*;
use polylogue_model::{
    Author, Reference,
    annotation::{AnnotationFact, AnnotationKind, AnnotationLink, LinkRole},
};

fn fact(root: &str, id: &str, kind: AnnotationKind, links: &[(LinkRole, &str)]) -> AnnotationFact {
    AnnotationFact {
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
            .map(|(role, reference)| AnnotationLink {
                role: *role,
                reference: Reference::parse(reference).unwrap(),
                seen_hash: None,
                quote: None,
            })
            .collect(),
    }
}

fn restore(archive: &TestArchive, fact: AnnotationFact) {
    archive
        .writer
        .archive()
        .restore_annotation(fact, None)
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
        page_size: None,
        random_seed: None,
        embeddings: false,
        self_session: None,
        here: None,
        seen_marks_of: None,
        saved: Default::default(),
        fields: vec![],
        kept: Default::default(),
        provider_catalog: None,
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
fn query_and_typed_about_share_forks_and_withdrawal() {
    let archive = test_archive();
    let about = (LinkRole::About, "session:codex:subject");
    // The base arrives last; its larger local commit position cannot replace heads.
    for id in ["right", "left"] {
        restore(
            &archive,
            fact(
                "family",
                id,
                AnnotationKind::Note,
                &[about, (LinkRole::Revises, "annotation:family@base")],
            ),
        );
    }
    restore(
        &archive,
        fact("family", "base", AnnotationKind::Note, &[about]),
    );
    let reads = ReadPool::open(&archive.archive).unwrap();
    let pinned = reads.reader().unwrap();
    assert_eq!(selected(&pinned, "annotations"), ["left", "right"]);
    assert_eq!(
        selected(&pinned, "annotations where conflict:true"),
        ["left", "right"]
    );
    let mut associated = Vec::new();
    let flow = pinned
        .visit_annotation_references_linked(
            &Reference::parse(about.1).unwrap(),
            &[polylogue_model::annotation::LinkRole::About],
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
            AnnotationKind::Withdrawal,
            &[(LinkRole::Withdraws, "annotation:family@left")],
        ),
    );
    assert_eq!(selected(&pinned, "annotations"), ["left", "right"]);
    drop(pinned);
    let fresh = reads.reader().unwrap();
    assert_eq!(selected(&fresh, "annotations"), ["right"]);
    assert!(selected(&fresh, "annotations where conflict:true").is_empty());
    let mut associated = Vec::new();
    let flow = fresh
        .visit_annotation_references_linked(
            &Reference::parse(about.1).unwrap(),
            &[polylogue_model::annotation::LinkRole::About],
            |row| {
                associated.push(row.revision.expect("linked revisions are exact"));
                Ok(std::ops::ControlFlow::Continue(()))
            },
        )
        .unwrap();
    assert!(matches!(flow, std::ops::ControlFlow::Continue(())));
    assert_eq!(associated, ["right"]);
    let exact = fresh
        .annotation(&polylogue_model::RevisionReference::exact(
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
fn no_effect_request_exports_its_outcome_and_source() {
    use polylogue_model::{
        annotation::NewAnnotation,
        authorship::{Authorship, Channel},
        user::{UserRequest, UserWrite},
    };
    let archive = test_archive();
    let request = |id: &str| UserRequest {
        id: id.into(),
        machine: "requesting-machine".into(),
        at: Timestamp::from_unix_ms(1).unwrap(),
        authorship: Authorship {
            author: Author::Operator,
            via: Channel::Cli,
        },
        write: UserWrite::Assert {
            annotations: vec![NewAnnotation {
                kind: AnnotationKind::Tag,
                key: Some("synthetic-tag".into()),
                value: None,
                valid_until: None,
                links: vec![polylogue_model::annotation::NewAnnotationLink {
                    role: LinkRole::About,
                    reference: Reference::parse("session:claude-code:synthetic").unwrap(),
                }],
            }],
        },
    };
    archive.writer.archive().user(request("tag-base")).unwrap();
    let original = request("no:effect@#..request");
    archive.writer.archive().user(original.clone()).unwrap();
    let pool = ReadPool::open(&archive.archive).unwrap();
    let reader = pool.reader().unwrap();
    let reference = Reference::Request(original.id.clone());
    let Some(crate::read::authored_read::AuthoredRead::Local { selected, record }) =
        reader.authored_record(&reference).unwrap()
    else {
        panic!("no-effect accepted request missing")
    };
    assert_eq!(selected, reference);
    assert_eq!(record.prepared.request, original);
    assert!(record.prepared.effect.annotations.is_empty());
    assert_eq!(record.prepared.outcome.id, "tag-base");
    let wire = polylogue_model::bundle::Record::Authored(record.clone());
    wire.validate().unwrap();
    let decoded: polylogue_model::bundle::Record =
        serde_json::from_slice(&serde_json::to_vec(&wire).unwrap()).unwrap();
    assert_eq!(decoded, wire);
    let other = test_archive();
    other
        .writer
        .archive()
        .restore_authored(record.prepared.clone())
        .unwrap();
    let pool = ReadPool::open(&other.archive).unwrap();
    let reader = pool.reader().unwrap();
    let Some(crate::read::authored_read::AuthoredRead::Local { record: copied, .. }) =
        reader.authored_record(&reference).unwrap()
    else {
        panic!("import lost the no-effect request")
    };
    assert_eq!(copied.prepared, record.prepared);
}

#[test]
fn unknown_authored_time_keeps_current_revision_queries() {
    use polylogue_model::{RevisionReference, authored::ImportAttribution};
    let archive = test_archive();
    let mut undated = fact(
        "undated",
        "undated",
        AnnotationKind::Note,
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
            .archive()
            .restore_annotation(undated.clone(), Some(ingress("input-one", 50)))
            .unwrap()
            .repeat
    );
    assert!(
        archive
            .writer
            .archive()
            .restore_annotation(undated.clone(), Some(ingress("input-two", 60)))
            .unwrap()
            .repeat
    );
    assert!(
        archive
            .writer
            .archive()
            .restore_annotation(undated.clone(), Some(ingress("input-one", 70)))
            .unwrap()
            .repeat
    );
    let mut dated = undated.clone();
    dated.id = "dated".into();
    dated.root = "dated".into();
    dated.created_at = Timestamp::from_unix_ms(5);
    archive
        .writer
        .archive()
        .restore_annotation(dated.clone(), None)
        .unwrap();
    let mut conflicting = dated.clone();
    conflicting.created_at = None;
    assert!(matches!(
        archive.writer.archive().restore_annotation(conflicting, None),
        Err(StoreError::Conflict(id)) if id == "dated"
    ));
    let reads = ReadPool::open(&archive.archive).unwrap();
    let reader = reads.reader().unwrap();
    assert_eq!(selected(&reader, "annotations"), ["dated", "undated"]);
    assert_eq!(
        selected(&reader, "annotations where created_at:unknown"),
        ["undated"]
    );
    assert_eq!(
        selected(&reader, "annotations where created_at:known"),
        ["dated"]
    );
    let reference = RevisionReference::exact("undated".into(), "undated".into());
    let restored = reader.annotation(&reference).unwrap().unwrap();
    assert_eq!(restored.fact, undated);
    let ingresses = reader.annotation_imports(&reference).unwrap().unwrap();
    assert_eq!(
        ingresses,
        [ingress("input-one", 50), ingress("input-two", 60)]
    );
    assert_eq!(
        reader
            .annotation(&RevisionReference::exact("dated".into(), "dated".into()))
            .unwrap()
            .unwrap()
            .fact,
        dated
    );
}

#[test]
fn private_readers_hydrate_authored_membership_in_temp_tables() {
    let archive = test_archive();
    restore(
        &archive,
        fact(
            "kept",
            "kept",
            AnnotationKind::Note,
            &[(LinkRole::About, "session:claude-code:synthetic")],
        ),
    );
    let pool = ReadPool::open(&archive.archive).unwrap();
    let reader = pool.reader().unwrap();
    let reference = Reference::Annotation(polylogue_model::RevisionReference::exact(
        "kept".into(),
        "kept".into(),
    ));
    // Imported legacy facts have no immutable request association, but their
    // normal authored read must still hydrate membership before reporting it.
    assert!(matches!(
        reader.authored_record(&reference).unwrap(),
        Some(crate::read::authored_read::AuthoredRead::Incomplete {
            issue: crate::read::authored_read::AuthoredReadIssue::MissingAssociation
        })
    ));
    assert_eq!(
        reader
            .query("SELECT singleton FROM temp.authored_membership_ready", &[])
            .unwrap()[0]["singleton"],
        1
    );
}
