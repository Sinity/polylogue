use super::*;
use polylogue_model::{
    Author, Timestamp,
    assertion::{
        AssertionFact, AssertionKind, AssertionLink, LinkRole, NewAssertion, NewAssertionLink,
    },
    authored::AuthoredEffect,
    authorship::{Authorship, Channel},
    user::{UserOutcome, UserRequest, UserWrite},
};

fn prepared() -> PreparedRequest {
    let about = Reference::parse("repo:synthetic").unwrap();
    let fact = AssertionFact {
        id: "native-request".into(),
        root: "native-request".into(),
        kind: AssertionKind::Note,
        key: None,
        value: Some(serde_json::json!("original source text")),
        author: Author::Operator,
        via: Some(Channel::Cli),
        machine: Some("synthetic".into()),
        created_at: None,
        valid_until: None,
        links: vec![AssertionLink {
            role: LinkRole::About,
            reference: about.clone(),
            seen_hash: None,
        }],
    };
    PreparedRequest {
        original: None,
        request: UserRequest {
            id: fact.id.clone(),
            machine: "synthetic".into(),
            at: Timestamp::from_unix_ms(1).unwrap(),
            authorship: Authorship {
                author: Author::Operator,
                via: Channel::Cli,
            },
            write: UserWrite::Assert {
                assertions: vec![NewAssertion {
                    kind: fact.kind,
                    key: None,
                    value: fact.value.clone(),
                    valid_until: None,
                    links: vec![NewAssertionLink {
                        role: LinkRole::About,
                        reference: about,
                    }],
                }],
            },
        },
        outcome: UserOutcome {
            id: fact.id.clone(),
            repeat: false,
            pending: false,
        },
        effect: AuthoredEffect {
            assertions: vec![fact],
            ..Default::default()
        },
    }
}

#[test]
fn restored_fact_before_whole_request_keeps_exact_effect_membership() {
    let mut conn = Connection::open_in_memory().unwrap();
    let schema = crate::source::from_archive(crate::schema::ARCHIVE).unwrap();
    conn.execute_batch(&schema.ddl).unwrap();
    let mut prepared = prepared();
    let original_effect = serde_json::to_vec(&prepared.effect).unwrap();
    prepared.original = Some(polylogue_model::historical::AcceptedEncoding {
        hash_evidence: polylogue_model::historical::AcceptedHashEvidence::Observed,
        native: Default::default(),
        source: polylogue_model::historical::SourceSchema::PolylogueArchive { version: 4 },
        content_hash: polylogue_model::BlobHash(
            crate::primary::authored::request_hash(&prepared.request).unwrap(),
        ),
        original: polylogue_model::historical::OriginalBytes {
            cached: vec![],
            request: Some(serde_json::to_vec(&prepared.request).unwrap()),
            intent: Some(
                serde_json::to_vec(&(
                    &prepared.request.machine,
                    &prepared.request.authorship,
                    prepared.request.write.intent(),
                ))
                .unwrap(),
            ),
            outcome: Some(serde_json::to_vec(&prepared.outcome).unwrap()),
            effect: Some(original_effect.clone()),
            records: Vec::new(),
            imports: None,
        },
    });
    crate::assertions::apply(&conn, &prepared.effect.assertions, 2).unwrap();
    let tx = conn.transaction().unwrap();
    crate::primary::authored::apply(&tx, &prepared, 9).unwrap();
    tx.commit().unwrap();
    let frozen = crate::primary::authored::known_request(&conn, &prepared.request)
        .unwrap()
        .unwrap();
    assert_eq!(frozen, prepared);
    assert_eq!(
        frozen.original.as_ref().unwrap().original.effect.as_ref(),
        Some(&original_effect)
    );
    let counters:(i64,i64)=conn.query_row("SELECT a.commit_seq,r.commit_seq FROM source_assertion a JOIN source_client_request r ON a.id=r.id",[],|row|Ok((row.get(0)?,row.get(1)?))).unwrap();
    assert_eq!(counters, (2, 9));
    conn.execute_batch("BEGIN").unwrap();
    prepare_membership(&conn, &mut || Ok(())).unwrap();
    let request:String=conn.query_row("SELECT request FROM temp.authored_effect_request WHERE kind='assertion' AND id='native-request' AND member='native-request'",[],|row|row.get(0)).unwrap();
    assert_eq!(request, prepared.request.id);
    let target = Target::Assertion {
        id: "native-request".into(),
        root: "native-request".into(),
        selected: Reference::Assertion(RevisionReference::exact(
            "native-request".into(),
            "native-request".into(),
        )),
    };
    assert!(target.matches(&prepared));
    let other = Target::Assertion {
        id: "native-request".into(),
        root: "other-root".into(),
        selected: Reference::Assertion(RevisionReference::exact(
            "other-root".into(),
            "native-request".into(),
        )),
    };
    assert!(!other.matches(&prepared));
    conn.execute_batch("ROLLBACK").unwrap();
    assert!(
        !conn
            .prepare_cached("SELECT 1 FROM sqlite_temp_schema WHERE name='authored_effect_request'")
            .unwrap()
            .exists([])
            .unwrap()
    );
}
