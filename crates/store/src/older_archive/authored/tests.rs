use super::super::{OlderArchive, fixture};
use super::*;

fn standard() -> (tempfile::TempDir, OlderArchive) {
    let dir = tempfile::tempdir().unwrap();
    let standard = fixture::standard(&dir.path().join("older"));
    let older = OlderArchive::open(&standard.root, &dir.path().join("copies")).unwrap();
    (dir, older)
}

fn exact(root: &str, id: &str) -> Reference {
    Reference::Assertion(RevisionReference::exact(root.into(), id.into()))
}

#[test]
fn notes_become_assertions_with_their_identity_revisions_and_withdrawals() {
    let (_dir, older) = standard();
    let notes = older.notes().unwrap();
    let roots = roots(&notes);
    let mapped: BTreeMap<String, Mapped<AssertionFact>> = notes
        .iter()
        .map(|n| (n.id.clone(), note(n, &roots)))
        .collect();
    let first = mapped["n1"].as_ref().unwrap();
    assert_eq!((first.id.as_str(), first.root.as_str()), ("n1", "n1"));
    assert_eq!(first.kind, AssertionKind::Note);
    assert_eq!(first.value, Some(serde_json::json!("a first note")));
    assert_eq!(first.author, Author::Operator);
    assert_eq!(first.via, Channel::Import);
    assert_eq!(first.created_at, Timestamp::from_unix_ms(10_000).unwrap());
    assert_eq!(
        first.links,
        [AssertionLink {
            role: LinkRole::About,
            reference: Reference::parse("session:claude-code:one").unwrap(),
            seen_hash: None,
            quote: None,
        }]
    );
    let revised = mapped["n2"].as_ref().unwrap();
    assert_eq!(revised.root, "n1");
    assert!(revised.links.contains(&AssertionLink {
        role: LinkRole::Revises,
        reference: exact("n1", "n1"),
        seen_hash: None,
        quote: None,
    }));
    let tag = mapped["t1"].as_ref().unwrap();
    assert_eq!(tag.key.as_deref(), Some("topic"));
    assert_eq!(tag.value, Some(serde_json::json!({"value": "storage"})));
    let withdrawal = mapped["w1"].as_ref().unwrap();
    assert_eq!(
        withdrawal.links,
        [AssertionLink {
            role: LinkRole::Withdraws,
            reference: Reference::Assertion(RevisionReference {
                root: "t1".into(),
                revision: None,
            }),
            seen_hash: None,
            quote: None,
        }]
    );
    let reason = mapped["x1"].as_ref().unwrap_err();
    assert!(reason.contains("distillate"), "{reason}");
}

#[test]
fn an_older_hidden_tag_becomes_the_hidden_session_mark() {
    let (_dir, older) = standard();
    let notes = older.notes().unwrap();
    let mut hidden = notes.iter().find(|note| note.id == "t1").unwrap().clone();
    hidden.key = Some("hidden".into());
    hidden.value = Some("kept out of ordinary reads".into());
    let fact = note(&hidden, &roots(&notes)).unwrap();
    assert_eq!(fact.kind, AssertionKind::Hidden);
    assert_eq!(fact.key, None);
    assert_eq!(
        fact.value,
        Some(serde_json::json!("kept out of ordinary reads"))
    );
    assert_eq!(fact.id, hidden.id);
    assert_eq!(fact.root, hidden.first_note);
}

#[test]
fn a_relation_to_another_family_names_that_note_exactly() {
    let note_row = |id: &str, first: &str, relations: Vec<(String, String)>| Note {
        id: id.into(),
        first_note: first.into(),
        kind: "note".into(),
        key: None,
        value: Some("text".into()),
        author: "operator".into(),
        machine: "m".into(),
        created_at: 1,
        valid_until: None,
        targets: vec![],
        relations,
    };
    let notes = vec![
        note_row("a1", "a1", vec![]),
        note_row("a2", "a1", vec![("revises".into(), "note:a1".into())]),
        note_row("b1", "b1", vec![("answers".into(), "note:a2".into())]),
    ];
    let roots = roots(&notes);
    let answer = note(&notes[2], &roots).unwrap();
    assert_eq!(answer.links.len(), 2);
    assert!(
        answer
            .links
            .iter()
            .any(|link| link.role == LinkRole::Answers && link.reference == exact("a1", "a2"))
    );
    assert!(
        answer
            .links
            .iter()
            .any(|link| link.role == LinkRole::About && link.reference == exact("a1", "a2"))
    );
    let unknown = note_row("c1", "c1", vec![("on_page".into(), "note:a1".into())]);
    assert!(note(&unknown, &roots).unwrap_err().contains("on_page"));
}

#[test]
fn a_later_setting_value_revises_the_one_before_it_on_the_same_machine() {
    let (_dir, older) = standard();
    let mapped = settings(&older.settings().unwrap(), older.machines());
    let ids: Vec<_> = mapped.iter().map(|(id, _)| id.as_str()).collect();
    assert_eq!(ids, ["s1", "s2", "s3"]);
    let s1 = mapped[0].1.as_ref().unwrap();
    let s2 = mapped[1].1.as_ref().unwrap();
    let s3 = mapped[2].1.as_ref().unwrap();
    assert_eq!(s1.root, "s1");
    assert!(s1.links.is_empty());
    assert_eq!(s2.root, "s1");
    assert_eq!(s2.links[0].role, LinkRole::Revises);
    assert_eq!(s2.links[0].reference, exact("s1", "s1"));
    // An archive-wide value records the one machine the archive knew.
    assert_eq!(s1.machine, older.machines()[0].id);
    assert_eq!(
        s1.value.as_ref().unwrap()["machine"],
        serde_json::Value::Null
    );
    assert_eq!(s3.root, "s3");
    assert_eq!(
        s3.value.as_ref().unwrap()["machine"],
        serde_json::json!(older.machines()[0].id)
    );

    let two = [
        older.machines()[0].clone(),
        Machine {
            id: "another".into(),
            host: "laptop".into(),
            first_seen: 1,
        },
    ];
    let mapped = settings(&older.settings().unwrap(), &two);
    assert!(
        mapped[0]
            .1
            .as_ref()
            .unwrap_err()
            .contains("records no machine")
    );
    assert!(mapped[2].1.is_ok());
}

#[test]
fn deliveries_events_and_kept_results_become_the_requests_that_wrote_them() {
    let (_dir, older) = standard();
    let rows = requests(
        &older.requests().unwrap(),
        &older.deliveries().unwrap(),
        &older.delivery_events().unwrap(),
        &older.kept_results().unwrap(),
    );
    let by_id: BTreeMap<&str, &MappedRequest> =
        rows.requests.iter().map(|r| (r.id.as_str(), r)).collect();
    for note_request in ["n1", "n2", "t1", "w1"] {
        let mapped = by_id[note_request];
        assert_eq!(mapped.class, Some(RequestClass::Note));
        assert!(matches!(mapped.request, Ok(None)));
    }

    let d1 = by_id["d1"].request.as_ref().unwrap().as_ref().unwrap();
    assert!(matches!(d1.request.write, UserWrite::Deliver(_)));
    assert_eq!(d1.request.at, Timestamp::from_unix_ms(19_000).unwrap());
    assert_eq!(d1.outcome.id, "d1");
    let change = &d1.effect.deliveries[0];
    assert_eq!(
        change.delivery.prepared.text,
        "synthetic context for a session"
    );
    assert_eq!(change.delivery.events.len(), 1);
    assert_eq!(change.delivery.events[0].request, "e1");
    assert!(change.added_events.is_empty());

    let e1 = by_id["e1"].request.as_ref().unwrap().as_ref().unwrap();
    assert_eq!(e1.outcome.id, "d1");
    assert_eq!(e1.effect.deliveries[0].added_events.len(), 1);
    // The second report of the same event wrote no row of its own.
    let e2 = by_id["e2"].request.as_ref().unwrap().as_ref().unwrap();
    assert!(e2.effect.deliveries[0].added_events.is_empty());

    let d2 = by_id["d2"].request.as_ref().unwrap().as_ref().unwrap();
    assert!(matches!(d2.request.write, UserWrite::DeliveredAdvice(_)));
    assert_eq!(d2.effect.deliveries[0].added_events[0].request, "d2");

    let k1 = by_id["k1"].request.as_ref().unwrap().as_ref().unwrap();
    assert_eq!(k1.effect.kept_results[0].query, "sessions");

    for prepared in [d1, e1, e2, d2, k1] {
        crate::primary::authored::validate(prepared).unwrap();
    }
    assert_eq!(rows.deliveries.len(), 1);
    assert_eq!(rows.deliveries[0].0, "d3");
    assert!(rows.deliveries[0].1.contains("no request"));
    assert!(rows.kept_results.is_empty());
}

#[test]
fn a_request_that_disagrees_with_its_row_refuses_both() {
    let dir = tempfile::tempdir().unwrap();
    let standard = fixture::standard(&dir.path().join("older"));
    let conn = rusqlite::Connection::open(standard.root.join("archive.db")).unwrap();
    conn.execute("UPDATE kept_result SET query = 'other' WHERE id = 'k1'", [])
        .unwrap();
    drop(conn);
    let older = OlderArchive::open(&standard.root, &dir.path().join("copies")).unwrap();
    let rows = requests(
        &older.requests().unwrap(),
        &older.deliveries().unwrap(),
        &older.delivery_events().unwrap(),
        &older.kept_results().unwrap(),
    );
    let k1 = rows.requests.iter().find(|r| r.id == "k1").unwrap();
    assert!(k1.request.as_ref().unwrap_err().contains("disagree"));
    assert_eq!(rows.kept_results.len(), 1);
    assert!(rows.kept_results[0].1.contains("disagree"));
}
