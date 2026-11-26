use super::*;
use crate::archive::{cache, open_write};
use crate::fixture::{Fixture, SessionFacts};
use rusqlite::params;

fn session(f: &mut Fixture, id: &str) -> i64 {
    f.session(SessionFacts {
        id,
        kind: "main",
        ..Default::default()
    })
    .unwrap()
}
fn read(f: &Fixture, id: &str) -> SessionLineage {
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    reader
        .lineage_root(&SessionId::parse(id).unwrap())
        .unwrap()
        .unwrap()
}
fn rooted(root: &str, depth: u64) -> SessionLineage {
    SessionLineage {
        root: Some(SessionId::parse(root).unwrap()),
        depth: Some(depth),
        root_pending: false,
        issue: None,
    }
}
fn db(f: &Fixture) -> Connection {
    open_write(&f.archive.index_db(), cache::SETUP).unwrap()
}

#[test]
fn arrival_order_permutation_oracle() {
    for order in [[0, 1, 2, 3], [3, 2, 1, 0], [1, 3, 0, 2], [2, 0, 3, 1]] {
        let dir = tempfile::tempdir().unwrap();
        let mut f = Fixture::new(dir.path()).unwrap();
        for id in [
            "test:root",
            "test:spawn",
            "test:fork",
            "test:continuation",
            "test:resume",
            "test:reference",
            "test:copy",
        ] {
            session(&mut f, id);
        }
        let edges = [
            ("test:root", "test:spawn", "spawn"),
            ("test:spawn", "test:fork", "fork"),
            ("test:fork", "test:continuation", "continuation"),
            ("test:continuation", "test:resume", "resume"),
        ];
        for i in order {
            let (p, c, k) = edges[i];
            f.link(p, c, k, None).unwrap();
            f.derive().unwrap();
        }
        f.link("test:root", "test:reference", "reference", Some("mention"))
            .unwrap();
        db(&f).execute("INSERT INTO shared_prefix(input_pk,chain,parent_id,child_id,explicit_members) SELECT pk,0,'test:root','test:copy',0 FROM input_ref LIMIT 1",[]).unwrap();
        for (id, depth) in [
            ("root", 0),
            ("spawn", 1),
            ("fork", 2),
            ("continuation", 3),
            ("resume", 4),
        ] {
            assert_eq!(read(&f, &format!("test:{id}")), rooted("test:root", depth));
        }
        for id in ["test:reference", "test:copy"] {
            assert_eq!(read(&f, id), rooted(id, 0));
        }
        f.derive().unwrap();
        assert_eq!(read(&f, "test:resume"), rooted("test:root", 4));
    }
}

#[test]
fn missing_parent_arrival_stub_resolution_and_retraction_invalidate_descendants() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    for id in ["test:child", "test:leaf"] {
        session(&mut f, id);
    }
    f.link("test:parent", "test:child", "fork", None).unwrap();
    f.link("test:child", "test:leaf", "resume", None).unwrap();
    f.derive().unwrap();
    assert_eq!(
        read(&f, "test:leaf"),
        SessionLineage {
            root: None,
            depth: None,
            root_pending: true,
            issue: Some(LineageIssue::MissingParent {
                child: SessionId::parse("test:child").unwrap(),
                parent: SessionId::parse("test:parent").unwrap()
            })
        }
    );
    let parent = session(&mut f, "test:parent");
    // Current reads resolve before the background derive step catches up.
    assert_eq!(read(&f, "test:leaf"), rooted("test:parent", 2));
    f.derive().unwrap();
    let conn = db(&f);
    conn.execute("UPDATE session SET has_transcript=0 WHERE pk=?1", [parent])
        .unwrap();
    // As the writer does when a session turns to or from a stub.
    mark_due(&conn, "test:parent").unwrap();
    // A stub parent ends the walk at the child, exactly as a missing one does.
    assert_eq!(
        read(&f, "test:leaf"),
        SessionLineage {
            root: None,
            depth: None,
            root_pending: true,
            issue: Some(LineageIssue::MissingParent {
                child: SessionId::parse("test:child").unwrap(),
                parent: SessionId::parse("test:parent").unwrap()
            })
        }
    );
    conn.execute("UPDATE session SET has_transcript=1 WHERE pk=?1", [parent])
        .unwrap();
    // As the writer does when a session turns to or from a stub.
    mark_due(&conn, "test:parent").unwrap();
    assert_eq!(read(&f, "test:leaf"), rooted("test:parent", 2));
    conn.execute("DELETE FROM session_link WHERE parent_id='test:parent'", [])
        .unwrap();
    // As the writer does with a link it removes.
    mark_due(&conn, "test:child").unwrap();
    assert_eq!(read(&f, "test:leaf"), rooted("test:child", 1));
    f.derive().unwrap();
    assert_eq!(read(&f, "test:leaf"), rooted("test:child", 1));
}

#[test]
fn depth_cycle_reported() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    for id in ["test:a", "test:b", "test:tail"] {
        session(&mut f, id);
    }
    f.link("test:a", "test:b", "spawn", None).unwrap();
    f.link("test:b", "test:a", "resume", None).unwrap();
    f.link("test:b", "test:tail", "fork", None).unwrap();
    let value = read(&f, "test:tail");
    assert_eq!(value.root, None);
    assert_eq!(value.depth, None);
    assert!(!value.root_pending);
    // The walk climbs tail, b, a and stops where it meets b again.
    assert_eq!(
        value.issue,
        Some(LineageIssue::Cycle {
            session: SessionId::parse("test:b").unwrap()
        })
    );
    f.derive().unwrap();
    assert_eq!(read(&f, "test:tail"), value);
}

#[test]
fn conflicting_parents_choose_neither_and_keep_one_structural_parent() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    for id in ["test:a", "test:b", "test:child", "test:leaf"] {
        session(&mut f, id);
    }
    f.link("test:a", "test:child", "fork", None).unwrap();
    f.link("test:child", "test:leaf", "resume", None).unwrap();
    let conn = db(&f);
    conn.execute(
        "UPDATE link SET branch_point='test:one' WHERE child_id='test:child'",
        [],
    )
    .unwrap();
    conn.execute("INSERT INTO session_link(parent_id,child_id,kind,evidence,input_pk,branch_point) SELECT parent_id,child_id,kind,evidence,input_pk,'test:two' FROM session_link WHERE child_id='test:child'",[]).unwrap();
    mark_due(&conn, "test:child").unwrap();
    assert_eq!(read(&f, "test:leaf"), rooted("test:a", 2));
    assert_eq!(
        conn.query_row(
            "SELECT count(*) FROM session_link WHERE child_id='test:child'",
            [],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        2
    );
    f.derive().unwrap();
    f.link("test:b", "test:child", "spawn", None).unwrap();
    let value = read(&f, "test:leaf");
    assert_eq!(value.root, None);
    assert_eq!(value.depth, None);
    assert_eq!(
        value.issue,
        Some(LineageIssue::ConflictingParents {
            child: SessionId::parse("test:child").unwrap(),
            parents: vec![
                SessionId::parse("test:a").unwrap(),
                SessionId::parse("test:b").unwrap()
            ]
        })
    );
    conn.execute("DELETE FROM session_link WHERE parent_id='test:b'", [])
        .unwrap();
    mark_due(&conn, "test:child").unwrap();
    assert_eq!(read(&f, "test:leaf"), rooted("test:a", 2));
}

#[test]
fn lineage_has_no_silent_depth_limit_and_unrelated_components_are_not_dirtied() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    let unrelated = session(&mut f, "test:unrelated");
    let conn = db(&f);
    conn.execute_batch("BEGIN;
      WITH RECURSIVE n(value) AS (SELECT 0 UNION ALL SELECT value+1 FROM n WHERE value<1005)
      INSERT INTO session(id,origin,native_id,kind,changed) SELECT 'test:'||value,'test',cast(value AS TEXT),'main',1 FROM n;
      INSERT INTO session_link(parent_id,child_id,kind,evidence,input_pk)
      SELECT 'test:'||(cast(native_id AS INTEGER)-1),id,'continuation','fixture',(SELECT pk FROM input_ref LIMIT 1)
      FROM session WHERE origin='test' AND native_id<>'unrelated' AND cast(native_id AS INTEGER)>0;
      COMMIT;").unwrap();
    mark_due(&conn, "test:0").unwrap();
    assert_eq!(read(&f, "test:1005"), rooted("test:0", 1005));
    let conn = db(&f);
    conn.execute("DELETE FROM summary_due WHERE session_pk=?1", [unrelated])
        .unwrap();
    conn.execute("DELETE FROM session_link WHERE child_id='test:1'", [])
        .unwrap();
    mark_due(&conn, "test:1").unwrap();
    assert_eq!(
        conn.query_row(
            "SELECT count(*) FROM summary_due WHERE session_pk=?1",
            params![unrelated],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        0
    );
    assert_eq!(read(&f, "test:1005"), rooted("test:1", 1004));
}

#[test]
fn family_entries_keep_known_parents_and_cycles_without_a_root() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    for id in [
        "test:root",
        "test:cycle",
        "test:other",
        "test:fork",
        "test:tail",
        "test:reference",
    ] {
        session(&mut f, id);
    }
    for (parent, child, kind) in [
        ("test:root", "test:fork", "fork"),
        ("test:cycle", "test:fork", "spawn"),
        ("test:cycle", "test:other", "continuation"),
        ("test:other", "test:cycle", "resume"),
        ("test:fork", "test:tail", "resume"),
    ] {
        f.link(parent, child, kind, None).unwrap();
    }
    f.link("test:reference", "test:tail", "reference", Some("mention"))
        .unwrap();
    f.derive().unwrap();
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let tail = SessionId::parse("test:tail").unwrap();
    assert_eq!(reader.lineage_root(&tail).unwrap().unwrap().root, None);
    assert_eq!(
        reader
            .lineage_starting_sessions(&tail)
            .unwrap()
            .into_iter()
            .map(|id| id.to_string())
            .collect::<std::collections::BTreeSet<_>>(),
        ["test:root".to_owned(), "test:cycle".to_owned()]
            .into_iter()
            .collect()
    );
    assert_eq!(
        reader
            .lineage_starting_sessions(&SessionId::parse("test:root").unwrap())
            .unwrap(),
        vec![SessionId::parse("test:root").unwrap()]
    );
}

/// The statements of one relation resolve to its relations
/// (docs/sessions.md, session links): statements whose coordinates agree
/// are one link, a coordinate recorded with several values gives a link for
/// each, and a partial statement another completes is not a link. The
/// reader reads them back as rows, with their inherited prefixes.
#[test]
fn link_statements_resolve_to_relations() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    for id in ["test:p", "test:q", "test:a", "test:b", "test:c", "test:d"] {
        session(&mut f, id);
    }
    type Coordinates = (Option<&'static str>, Option<&'static str>);
    // A parent, a child, a kind and how; the statements of that relation
    // with whether each copies the prefix; the relations they give.
    type Case<'a> = (
        &'a str,
        &'a str,
        &'a str,
        Option<&'a str>,
        &'a [(Option<&'a str>, Option<&'a str>, bool)],
        &'a [Coordinates],
    );
    let cases: &[Case<'_>] = &[
        (
            "test:p",
            "test:a",
            "spawn",
            None,
            &[(None, None, false)],
            &[(None, None)],
        ),
        (
            "test:p",
            "test:a",
            "fork",
            None,
            &[
                (Some("m1"), None, false),
                (None, Some("c1"), true),
                (None, None, false),
            ],
            &[(Some("m1"), Some("c1"))],
        ),
        (
            "test:p",
            "test:b",
            "fork",
            None,
            &[
                (Some("m1"), None, false),
                (Some("m2"), None, false),
                (None, Some("c1"), false),
            ],
            &[(Some("m1"), Some("c1")), (Some("m2"), Some("c1"))],
        ),
        (
            "test:p",
            "test:c",
            "spawn",
            None,
            &[
                (None, Some("c2"), false),
                (None, Some("c1"), false),
                (Some("m1"), None, false),
            ],
            &[(Some("m1"), Some("c1")), (Some("m1"), Some("c2"))],
        ),
        (
            "test:p",
            "test:d",
            "spawn",
            None,
            &[
                (Some("m1"), Some("c1"), false),
                (Some("m2"), Some("c2"), false),
                (Some("m1"), None, false),
                (None, Some("c2"), false),
                (None, None, false),
                (Some("m3"), None, false),
                (None, Some("c3"), false),
            ],
            &[
                (None, Some("c3")),
                (Some("m1"), Some("c1")),
                (Some("m2"), Some("c2")),
                (Some("m3"), None),
            ],
        ),
        (
            "test:p",
            "test:d",
            "reference",
            Some("mention"),
            &[(None, None, false)],
            &[(None, None)],
        ),
        (
            "test:p",
            "test:d",
            "reference",
            Some("briefing"),
            &[(None, Some("c9"), false)],
            &[(None, Some("c9"))],
        ),
        (
            "test:q",
            "test:d",
            "resume",
            None,
            &[(None, None, false)],
            &[(None, None)],
        ),
        (
            "test:q",
            "test:b",
            "continuation",
            None,
            &[(None, None, false)],
            &[(None, None)],
        ),
    ];
    let conn = db(&f);
    let mut expected_links: Vec<SessionLink> = Vec::new();
    for (parent, child, kind, how, statements, expected) in cases {
        let stated: Vec<Statement> = statements
            .iter()
            .enumerate()
            .map(|(n, (point, call, copies))| Statement {
                branch_point: point.map(str::to_owned),
                spawning_call: call.map(str::to_owned),
                copies_prefix: *copies,
                evidence: format!("field.{n}"),
                source_chain: n as i64,
                input_pk: 1,
            })
            .collect();
        let resolved = relations(&stated);
        let coordinates: Vec<Coordinates> = resolved
            .iter()
            .map(|r| (r.branch_point.as_deref(), r.spawning_call.as_deref()))
            .collect();
        assert_eq!(coordinates, *expected, "{parent} {child} {kind} {how:?}");
        let copies = statements.iter().any(|(_, _, copies)| *copies);
        for relation in &resolved {
            assert_eq!(relation.evidence, "field.0");
            assert_eq!(relation.copies_prefix, copies);
            conn.execute(
                "INSERT INTO session_link(parent_id, child_id, kind, reason, copies_prefix, branch_point,
                     spawning_call, evidence, source_chain, input_pk)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, (SELECT min(pk) FROM input_ref))",
                params![
                    parent,
                    child,
                    kind,
                    how,
                    relation.copies_prefix,
                    relation.branch_point,
                    relation.spawning_call,
                    relation.evidence,
                    relation.source_chain
                ],
            )
            .unwrap();
            expected_links.push(SessionLink {
                parent: (*parent).into(),
                child: (*child).into(),
                kind: (*kind).into(),
                reason: how.map(str::to_owned),
                copies_prefix: copies || (*parent, *child) == ("test:q", "test:b"),
                branch_point: relation.branch_point.clone(),
                spawning_call: relation.spawning_call.clone(),
            });
        }
    }
    conn.execute(
        "INSERT INTO shared_prefix(input_pk, chain, parent_id, child_id, committed, explicit_members)
         VALUES ((SELECT min(pk) FROM input_ref), 1, 'test:q', 'test:b', 1, 0)",
        [],
    )
    .unwrap();
    // A copied prefix no link records.
    conn.execute(
        "INSERT INTO shared_prefix(input_pk, chain, parent_id, child_id, committed, explicit_members)
         VALUES ((SELECT min(pk) FROM input_ref), 1, 'test:c', 'test:a', 1, 0)",
        [],
    )
    .unwrap();
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let sorted = |mut links: Vec<SessionLink>| {
        links.sort_by(|a, b| {
            (
                &a.parent,
                &a.child,
                &a.kind,
                &a.reason,
                &a.branch_point,
                &a.spawning_call,
            )
                .cmp(&(
                    &b.parent,
                    &b.child,
                    &b.kind,
                    &b.reason,
                    &b.branch_point,
                    &b.spawning_call,
                ))
        });
        links
    };
    let some = |text: &str| Some(text.to_owned());
    let expected_prefixes: &[(&str, Vec<(String, Option<String>, bool)>)] = &[
        ("test:p", vec![]),
        ("test:q", vec![]),
        (
            "test:a",
            vec![
                ("test:c".into(), None, true),
                ("test:p".into(), some("m1"), true),
            ],
        ),
        (
            "test:b",
            vec![
                ("test:p".into(), some("m1"), false),
                ("test:p".into(), some("m2"), false),
                ("test:q".into(), None, true),
            ],
        ),
        ("test:c", vec![]),
        ("test:d", vec![("test:q".into(), None, false)]),
    ];
    for (id, prefixes) in expected_prefixes {
        let session = SessionId::parse(id).unwrap();
        let of = |keep: &dyn Fn(&SessionLink) -> bool| {
            sorted(expected_links.iter().filter(|l| keep(l)).cloned().collect())
        };
        assert_eq!(
            sorted(reader.child_links(&session).unwrap()),
            of(&|l| l.parent == *id),
            "children of {id}"
        );
        assert_eq!(
            sorted(reader.parent_links(&session).unwrap()),
            of(&|l| l.child == *id),
            "parents of {id}"
        );
        assert_eq!(
            &inherited_prefixes(&conn, id).unwrap(),
            prefixes,
            "prefixes of {id}"
        );
        let mut viewed: Vec<(String, Option<String>, bool)> = conn
            .prepare("SELECT parent_id, branch_point, copies_prefix FROM session_prefix WHERE child_id = ?1")
            .unwrap()
            .query_map([id], |row| <(_, _, _)>::try_from(row))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        viewed.sort();
        assert_eq!(&viewed, prefixes, "prefix view of {id}");
    }
    assert_eq!(expected_links.len(), 14);
}
