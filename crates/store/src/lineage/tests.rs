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
        db(&f).execute("INSERT INTO prefix_copy(input_pk,chain,parent_id,child_id,explicit_members) SELECT pk,0,'test:root','test:copy',0 FROM input_ref LIMIT 1",[]).unwrap();
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
    conn.execute("UPDATE session SET stub=1 WHERE pk=?1", [parent])
        .unwrap();
    assert!(read(&f, "test:leaf").root_pending);
    conn.execute("UPDATE session SET stub=0 WHERE pk=?1", [parent])
        .unwrap();
    assert_eq!(read(&f, "test:leaf"), rooted("test:parent", 2));
    conn.execute("DELETE FROM link WHERE parent_id='test:parent'", [])
        .unwrap();
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
    assert!(matches!(value.issue, Some(LineageIssue::Cycle { .. })));
    f.derive().unwrap();
    assert_eq!(read(&f, "test:tail"), value);
}

#[test]
fn conflicting_parents_choose_neither_while_anchor_alternatives_keep_one_structural_parent() {
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
    conn.execute("INSERT INTO link(parent_id,child_id,child_pk,kind,evidence,input_pk,branch_point) SELECT parent_id,child_id,child_pk,kind,evidence,input_pk,'test:two' FROM link WHERE child_id='test:child'",[]).unwrap();
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
    conn.execute("DELETE FROM link WHERE parent_id='test:b'", [])
        .unwrap();
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
      INSERT INTO link(parent_id,child_id,child_pk,kind,evidence,input_pk)
      SELECT 'test:'||(cast(native_id AS INTEGER)-1),id,pk,'continuation','fixture',(SELECT pk FROM input_ref LIMIT 1)
      FROM session WHERE origin='test' AND native_id<>'unrelated' AND cast(native_id AS INTEGER)>0;
      COMMIT;").unwrap();
    assert_eq!(read(&f, "test:1005"), rooted("test:0", 1005));
    let conn = db(&f);
    conn.execute("DELETE FROM derive_due WHERE session_pk=?1", [unrelated])
        .unwrap();
    conn.execute("DELETE FROM link WHERE child_id='test:1'", [])
        .unwrap();
    assert_eq!(
        conn.query_row(
            "SELECT count(*) FROM derive_due WHERE session_pk=?1",
            params![unrelated],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        0
    );
    assert_eq!(read(&f, "test:1005"), rooted("test:1", 1004));
}

#[test]
fn family_entries_keep_all_frontiers_and_cycle_components_without_claiming_a_root() {
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
