use super::*;
use crate::predecessor::fixture;
use polylogue_model::assertion::LinkRole;

fn mapped() -> (
    Vec<Row>,
    Vec<(String, Mapped<AssertionFact>)>,
    RevisionReference,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("user.db");
    fixture::user_db(&path);
    let db = UserDb::open(&path, dir.path()).unwrap();
    let rows = db.assertions().unwrap();
    let pass = RevisionReference::exact("pass-1".into(), "pass-1".into());
    let facts = facts(&rows, "machine-1", &pass, &db.result_set_members().unwrap());
    (rows, facts, pass)
}

fn fact<'a>(facts: &'a [(String, Mapped<AssertionFact>)], id: &str) -> &'a AssertionFact {
    facts
        .iter()
        .find(|(row, _)| row == id)
        .unwrap()
        .1
        .as_ref()
        .unwrap()
}

fn links(fact: &AssertionFact, role: LinkRole) -> Vec<String> {
    fact.links
        .iter()
        .filter(|link| link.role == role)
        .map(|link| link.reference.to_string())
        .collect()
}

#[test]
fn every_predecessor_form_maps_with_its_author_and_history() {
    let (rows, facts, pass) = mapped();
    assert_eq!(rows.len(), 12);
    assert_eq!(facts.len(), 12);

    let open = fact(&facts, fixture::CANDIDATE);
    assert_eq!(open.kind, AssertionKind::Question);
    assert_eq!(open.key.as_deref(), Some("duplicate:1"));
    assert_eq!(open.author, Author::Script(fixture::DETECTOR.into()));
    assert_eq!(open.via, Channel::Import);
    assert_eq!(open.machine, "machine-1");
    assert_eq!(open.created_at.unix_ms(), 1_000);
    assert_eq!(
        links(open, LinkRole::About),
        ["session:chatgpt:00000000-0000-4000-8000-00000000aaaa"]
    );
    assert_eq!(
        links(open, LinkRole::Origin),
        [Reference::Assertion(pass).to_string()]
    );
    let value = open.value.as_ref().unwrap();
    assert_eq!(value["text"], "two captures of one conversation disagree");
    assert_eq!(value["predecessor"]["assertion_id"], fixture::CANDIDATE);
    assert_eq!(value["predecessor"]["status"], "candidate");

    // The promotion is a revision of the detector's request, by whoever ran
    // the command under the operator's account name.
    let promoted = fact(&facts, fixture::PROMOTED);
    assert_eq!(promoted.kind, AssertionKind::Question);
    assert_eq!(promoted.root, fixture::ACCEPTED);
    assert_eq!(promoted.author, Author::Unattributed);
    let accepted = format!("assertion:{0}@{0}", fixture::ACCEPTED);
    assert_eq!(links(promoted, LinkRole::Revises), [accepted.clone()]);
    assert_eq!(links(promoted, LinkRole::Evidence), [accepted.clone()]);

    let accept = fact(&facts, fixture::ACCEPT);
    assert_eq!(accept.kind, AssertionKind::Judgment);
    assert_eq!(accept.key.as_deref(), Some("accepted"));
    assert_eq!(accept.root, fixture::ACCEPT);
    assert_eq!(links(accept, LinkRole::About), [accepted.clone()]);
    assert_eq!(
        links(accept, LinkRole::Evidence),
        [
            accepted,
            format!("assertion:{}@{}", fixture::ACCEPTED, fixture::PROMOTED)
        ]
    );
    assert_eq!(
        accept.value.as_ref().unwrap()["reason"],
        "the bytes are identical"
    );
    let reject = fact(&facts, fixture::REJECT);
    assert_eq!(reject.key.as_deref(), Some("rejected"));
    assert_eq!(
        links(reject, LinkRole::About),
        [format!("assertion:{0}@{0}", fixture::REJECTED)]
    );

    let blocker = fact(&facts, fixture::BLOCKER);
    assert_eq!(blocker.kind, AssertionKind::Problem);
    assert_eq!(blocker.key.as_deref(), Some("raw/1"));
    assert_eq!(
        blocker.value.as_ref().unwrap()["text"],
        "a competing copy blocks the merge"
    );

    // A finding about an analysis is about the rows it was drawn from;
    // references with no counterpart stay in its predecessor fields.
    let finding = fact(&facts, fixture::FINDING);
    assert_eq!(finding.kind, AssertionKind::Finding);
    assert_eq!(finding.author, Author::Script("run:followups-1".into()));
    assert_eq!(
        links(finding, LinkRole::About),
        [
            "message:claude-code:00000000-0000-4000-8000-00000000bbbb:m-1",
            "message:claude-code:00000000-0000-4000-8000-00000000bbbb:m-2"
        ]
    );
    assert!(links(finding, LinkRole::Evidence).is_empty());
    assert_eq!(
        finding.value.as_ref().unwrap()["predecessor"]["evidence_refs_json"],
        r#"["query:00","result-set:finding-0001"]"#
    );

    let note = fact(&facts, fixture::NOTE);
    assert_eq!(note.kind, AssertionKind::Note);
    assert_eq!(note.key, None);
    assert_eq!(
        note.author,
        Author::Agent(Some(
            SessionId::parse("codex:00000000-0000-7000-8000-00000000c0de").unwrap()
        ))
    );
    assert_eq!(
        note.valid_until,
        Timestamp::parse_rfc3339("2026-07-12T00:00:00Z")
    );
    assert_eq!(
        links(note, LinkRole::Evidence),
        ["session:codex:00000000-0000-7000-8000-00000000c0de"]
    );

    let refused = &facts.iter().find(|(id, _)| id == fixture::MARK).unwrap().1;
    assert_eq!(
        refused.as_ref().unwrap_err(),
        "assertion kind mark has no counterpart here"
    );
}

#[test]
fn every_row_follows_the_rows_it_names() {
    let (_, facts, _) = mapped();
    let position = |id: &str| facts.iter().position(|(row, _)| row == id).unwrap();
    assert!(position(fixture::ACCEPTED) < position(fixture::PROMOTED));
    assert!(position(fixture::PROMOTED) < position(fixture::ACCEPT));
    assert!(position(fixture::REJECTED) < position(fixture::REJECT));
}

#[test]
fn a_target_with_no_counterpart_refuses_its_row() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("user.db");
    fixture::user_db(&path);
    let db = UserDb::open(&path, dir.path()).unwrap();
    let rows = db.assertions().unwrap();
    let pass = RevisionReference::exact("pass-1".into(), "pass-1".into());
    // Without the result set, the finding's analysis target has no counterpart.
    let facts = facts(&rows, "machine-1", &pass, &BTreeMap::new());
    let finding = &facts
        .iter()
        .find(|(id, _)| id == fixture::FINDING)
        .unwrap()
        .1;
    assert_eq!(
        finding.as_ref().unwrap_err(),
        "its target analysis:followups has no counterpart here"
    );
}

#[test]
fn every_row_of_every_table_is_kept_verbatim() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("user.db");
    fixture::user_db(&path);
    let db = UserDb::open(&path, dir.path()).unwrap();
    let mut out = Vec::new();
    assert_eq!(db.write_rows(&mut out).unwrap(), 12 + 1 + 2);
    let lines: Vec<Json> = out
        .split(|b| *b == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice(line).unwrap())
        .collect();
    assert_eq!(lines[0]["table"], "annotation_schemas");
    assert_eq!(lines[0]["row"]["schema_id"], "seed.example");
    assert!(lines.iter().any(|line| line["table"] == "assertions"
        && line["row"]["assertion_id"] == fixture::MARK
        && line["row"]["body_text"] == "starred"));
    assert_eq!(
        db.row_counts().unwrap(),
        BTreeMap::from([
            ("annotation_schemas".to_owned(), 1),
            ("assertions".to_owned(), 12),
            ("result_set_members".to_owned(), 2)
        ])
    );
}

#[test]
fn a_database_without_the_predecessors_assertions_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("other.db");
    Connection::open(&path)
        .unwrap()
        .execute_batch("CREATE TABLE assertions (id TEXT)")
        .unwrap();
    let error = UserDb::open(&path, dir.path()).err().unwrap().to_string();
    assert!(error.contains("not the predecessor's user.db"), "{error}");
}

#[test]
fn the_pass_is_the_same_for_the_same_rows() {
    let (rows, _, _) = mapped();
    let first = pass("pass-1", &rows, "machine-1").unwrap();
    assert_eq!(first, pass("pass-1", &rows, "machine-1").unwrap());
    assert_eq!(first.kind, AssertionKind::Pass);
    assert_eq!(first.created_at.unix_ms(), 6_000);
    assert_eq!(first.value.as_ref().unwrap()["counts"]["assertions"], 12);
}
