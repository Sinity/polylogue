use crate::{
    ReadPool,
    fixture::{Fixture, SessionFacts},
};
use polylogue_model::SessionId;

/// A link row whose stored kind this build does not know makes the export
/// fail as a store failure, not as something wrong with the request.
#[test]
fn export_over_a_corrupt_stored_link_kind_is_a_store_failure() {
    let root = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(root.path()).unwrap();
    for id in ["test:parent", "test:child"] {
        fixture
            .session(SessionFacts {
                id,
                ..Default::default()
            })
            .unwrap();
    }
    fixture
        .link("test:parent", "test:child", "fork", None)
        .unwrap();
    let child = SessionId::parse("test:child").unwrap();
    let export = |fixture: &Fixture| {
        let pool = ReadPool::open(&fixture.archive).unwrap();
        pool.reader().unwrap().session_export(&child)
    };
    assert_eq!(export(&fixture).unwrap().unwrap().links.len(), 1);
    fixture
        .link("test:parent", "test:child", "unheard-of", None)
        .unwrap();
    let error = export(&fixture).unwrap_err();
    assert!(
        matches!(error, crate::StoreError::StoredValue { column: 0, .. }),
        "{error:?}"
    );
    assert_eq!(error.code(), "store");
}

// The original query is the independent result and negative work control.
const ALL_PREFIXES: &str = r#"SELECT DISTINCT p.child_id,ir.id,s.chain,ps.reading_version,s.seq,s.record_position,p.parent_id FROM message m JOIN session owner ON owner.pk=m.session_pk JOIN session_prefix p ON p.parent_id=owner.id AND p.copies_prefix=1 AND p.branch_point IS NULL JOIN message_source s ON s.message_pk=m.pk JOIN input_ref ir ON ir.pk=s.input_pk JOIN parse_state ps ON ps.input_pk=s.input_pk AND ps.chain=s.chain WHERE m.id=?1 AND m.duplicate_of IS NULL AND EXISTS(SELECT 1 FROM prefix_copy x WHERE x.input_pk=s.input_pk AND x.chain=s.chain AND x.child_id=p.child_id AND x.parent_id=p.parent_id AND x.explicit_members=0) AND NOT EXISTS(SELECT 1 FROM prefix_copy x WHERE x.input_pk=s.input_pk AND x.chain=s.chain AND x.child_id=p.child_id AND x.parent_id=p.parent_id AND x.explicit_members=1)"#;

fn copied_archive() -> (
    crate::fixture::writer::WriterArchive,
    polylogue_model::MessageId,
) {
    use crate::fixture::writer::{WriterArchive, message};
    use polylogue_model::{Block, Role};
    let mut archive = WriterArchive::new();
    let first = message(
        "first",
        "owner",
        Role::User,
        1,
        vec![Block::Text {
            text: "Synthetic copied prompt".into(),
        }],
    );
    let last = message(
        "later",
        "owner",
        Role::Assistant,
        2,
        vec![Block::Text {
            text: "Synthetic later reply".into(),
        }],
    );
    archive.write("owner", vec![first.clone(), last]);
    archive.write("copy", vec![first.clone()]);
    (archive, first.id)
}

fn old_copies(
    conn: &rusqlite::Connection,
    id: &polylogue_model::MessageId,
) -> Vec<polylogue_model::bundle::CopiedFor> {
    use polylogue_model::{PrefixSource, bundle::CopiedFor};
    conn.prepare(ALL_PREFIXES)
        .unwrap()
        .query_map([id.to_string()], |r| {
            Ok(CopiedFor {
                parent: SessionId::parse(&r.get::<_, String>(6)?).unwrap(),
                child: SessionId::parse(&r.get::<_, String>(0)?).unwrap(),
                source: PrefixSource {
                    input: r.get(1)?,
                    chain: r.get(2)?,
                    reading_version: r.get(3)?,
                },
                position: [r.get(4)?, r.get(5)?],
            })
        })
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap()
}

#[test]
fn copied_message_evidence_keeps_explicit_sources_and_anchored_prefix_boundaries() {
    use crate::fixture::writer::{message, sid};
    use polylogue_model::{Block, Link, LinkKind, PrefixSource, Role, bundle::CopiedFor};
    let (mut archive, id) = copied_archive();
    archive.links.push(Link {
        kind: LinkKind::Fork,
        copies_prefix: true,
        how: None,
        parent: sid("owner"),
        child: sid("anchored"),
        at: Some(id.clone()),
        call_id: None,
        evidence: "synthetic branch".into(),
    });
    archive.write(
        "anchored",
        vec![message(
            "first",
            "owner",
            Role::User,
            1,
            vec![Block::Text {
                text: "Synthetic copied prompt".into(),
            }],
        )],
    );
    let index = crate::archive::open_write(&archive.archive.index_db(), 64).unwrap();
    let reads = ReadPool::open(&archive.archive).unwrap();
    {
        let reader = reads.reader().unwrap();
        let copies = reader.copied_for(&id).unwrap();
        assert_eq!(copies, old_copies(&index, &id));
        assert_eq!(
            copies.len(),
            1,
            "the anchored fork is not inferred as an unbounded copy"
        );
        assert_eq!(copies[0].parent, sid("owner"));
        assert_eq!(copies[0].child, sid("copy"));
        assert_eq!(copies[0].source.input, "input-copy");
        assert_eq!(copies[0].source.reading_version, "synthetic=1");
    }
    index
        .execute(
            "UPDATE message SET duplicate_of='claude-code:owner-later' WHERE id=?1",
            [id.to_string()],
        )
        .unwrap();
    {
        let reader = reads.reader().unwrap();
        assert!(reader.copied_for(&id).unwrap().is_empty());
        assert!(old_copies(&index, &id).is_empty());
    }
    index
        .execute(
            "UPDATE message SET duplicate_of=NULL WHERE id=?1",
            [id.to_string()],
        )
        .unwrap();
    index
        .execute(
            "UPDATE prefix_copy SET explicit_members=1 WHERE child_id=?1",
            [sid("copy").to_string()],
        )
        .unwrap();
    index.execute("INSERT INTO prefix_member(input_pk,chain,parent_id,child_id,message_id,source_input,source_chain,source_version,sequence,position)
        SELECT input_pk,chain,parent_id,child_id,?1,'original-copy',17,'synthetic=2','18446744073709551615','42' FROM prefix_copy WHERE child_id=?2",
        rusqlite::params![id.to_string(),sid("copy").to_string()]).unwrap();
    let expected = CopiedFor {
        parent: sid("owner"),
        child: sid("copy"),
        source: PrefixSource {
            input: "original-copy".into(),
            chain: 17,
            reading_version: "synthetic=2".into(),
        },
        position: [u64::MAX, 42],
    };
    {
        let reader = reads.reader().unwrap();
        assert_eq!(reader.copied_for(&id).unwrap(), vec![expected.clone()]);
        assert!(
            old_copies(&index, &id).is_empty(),
            "explicit membership suppresses inferred expansion"
        );
        assert!(
            reader
                .copied_for(&polylogue_model::MessageId::parse("claude-code:owner-later").unwrap())
                .unwrap()
                .is_empty()
        );
    }
    // Explicit retained evidence is independent of whether its message still
    // has a selected row. Do not canonicalize an absent/duplicate owner into it.
    let absent = polylogue_model::MessageId::parse("claude-code:absent").unwrap();
    index
        .execute(
            "UPDATE prefix_member SET message_id=?1",
            [absent.to_string()],
        )
        .unwrap();
    let reader = reads.reader().unwrap();
    assert_eq!(reader.copied_for(&absent).unwrap(), vec![expected]);
    assert!(reader.copied_for(&id).unwrap().is_empty());
}

#[test]
fn copied_message_evidence_does_not_walk_unrelated_prefix_edges() {
    use std::sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    };
    let (archive, id) = copied_archive();
    let index = crate::archive::open_write(&archive.archive.index_db(), 64).unwrap();
    let reads = ReadPool::open(&archive.archive).unwrap();
    let measure = || {
        let reader = reads.reader().unwrap();
        let conn = reader.conn().unwrap();
        conn.pragma_update(None, "cache_size", -64).unwrap();
        conn.flush_prepared_statement_cache();
        let steps = Arc::new(AtomicU64::new(0));
        let count = Arc::clone(&steps);
        conn.progress_handler(
            1,
            Some(move || {
                count.fetch_add(1, Ordering::Relaxed);
                false
            }),
        )
        .unwrap();
        let result = reader.copied_for(&id).unwrap();
        conn.progress_handler(0, None::<fn() -> bool>).unwrap();
        let mut old = conn.prepare(ALL_PREFIXES).unwrap();
        let previous = old
            .query_map([id.to_string()], |r| r.get::<_, String>(0))
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap();
        assert_eq!(previous, vec!["claude-code:copy"]);
        (
            result,
            steps.load(Ordering::Relaxed),
            old.get_status(rusqlite::StatementStatus::VmStep),
        )
    };
    let (expected, before, old_before) = measure();
    index.execute("WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n WHERE x<2000)
        INSERT INTO link(parent_id,child_id,kind,copies_prefix,evidence,input_pk)
        SELECT 'claude-code:other-parent-'||x,'claude-code:other-child-'||x,'fork',1,'synthetic',(SELECT min(pk) FROM input_ref) FROM n",[]).unwrap();
    index
        .execute(
            "INSERT INTO prefix_copy(input_pk,chain,parent_id,child_id,explicit_members)
        SELECT input_pk,0,parent_id,child_id,0 FROM link WHERE evidence='synthetic'",
            [],
        )
        .unwrap();
    let (actual, after, old_after) = measure();
    assert_eq!(actual, expected);
    assert!(
        after <= before + 500,
        "selected copy evidence grew from {before} to {after} VM steps"
    );
    assert!(
        old_after > old_before + 100_000,
        "negative control did not expose global prefix work: {old_before} -> {old_after}"
    );
}
