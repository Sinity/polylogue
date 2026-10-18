use super::*;
mod construction;
mod statements;
fn fixture() -> (tempfile::TempDir, crate::Writer, Owner, i64) {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("archive");
    let writer = crate::Writer::open(&root).unwrap();
    writer
        ._index
        .execute("INSERT INTO input_ref(pk,id) VALUES(1,'input')", [])
        .unwrap();
    writer._index.execute("INSERT INTO session(pk,id,origin,native_id,kind,changed) VALUES(1,'opaque-session','test','s','main',0)",[]).unwrap();
    writer
        ._index
        .execute(
            "INSERT INTO message(pk,id,session_pk,seq) VALUES(1,'opaque-message',1,0)",
            [],
        )
        .unwrap();
    writer._index.execute("INSERT INTO message_body(message_pk,session_pk,basis_digest,contributor_revision,removal_epoch,layout,role,written_by,input_pk,parser_version) VALUES(1,1,zeroblob(32),0,0,'flat','assistant','model',1,1)",[]).unwrap();
    let body = writer._index.last_insert_rowid();
    (
        dir,
        writer,
        Owner {
            message: 1,
            session: 1,
        },
        body,
    )
}
fn leaf(tx: &Transaction<'_>, ordinal: u64) -> Edge {
    tx.execute("INSERT INTO block(message_pk,session_pk,group_ordinal,local_ordinal,slot_kind,kind) VALUES(1,1,0,?1,'ordinary','text')",[i64::try_from(ordinal).unwrap()]).unwrap();
    Edge {
        link: Link::Block {
            pk: tx.last_insert_rowid(),
            call: None,
        },
        counts: Counts {
            ordinary: 1,
            ..Counts::default()
        },
    }
}
#[test]
fn next_reads_at_most_31_nodes_and_includes_zero_count_leaves() {
    let (_dir, mut writer, owner, body) = fixture();
    let file = writer.root.join("index.db");
    let keys = [0, 9, 10, 255, 256, u64::MAX];
    let mut root = None;
    let mut expected = Vec::new();
    for (ordinal, key) in keys.into_iter().enumerate() {
        let tx = writer._index.transaction().unwrap();
        let mut edge = leaf(&tx, ordinal as u64);
        if key == 10 {
            edge.counts = Counts::default();
        }
        let (new, _) = install(&tx, &file, owner, body, root, key, edge);
        tx.commit().unwrap();
        root = Some(new);
        expected.push((key, edge));
    }
    let mut after = None;
    for item in expected {
        let mut work = Work::default();
        assert_eq!(
            next(
                &writer._index,
                &file,
                owner,
                Kind::Blocks,
                root,
                after,
                &mut work
            )
            .unwrap(),
            Some(item)
        );
        assert!(work.nodes_read <= 31);
        assert!(work.edges_read <= 31 * 16);
        assert_eq!(work.nodes_written, 0);
        after = Some(item.0);
    }
    let mut work = Work::default();
    assert!(
        next(
            &writer._index,
            &file,
            owner,
            Kind::Blocks,
            root,
            Some(u64::MAX),
            &mut work
        )
        .unwrap()
        .is_none()
    );
    assert_eq!(work.nodes_read, 0);
    for (after, expected) in [
        (Some(8), 9),
        (Some(10), 255),
        (Some(254), 255),
        (Some(257), u64::MAX),
    ] {
        assert_eq!(
            next(
                &writer._index,
                &file,
                owner,
                Kind::Blocks,
                root,
                after,
                &mut Work::default()
            )
            .unwrap()
            .unwrap()
            .0,
            expected
        );
    }
    assert!(
        next(
            &writer._index,
            &file,
            owner,
            Kind::Fields,
            root,
            None,
            &mut Work::default()
        )
        .is_err()
    );
    // A missing table is a SQLite error, not an empty result.
    writer
        ._index
        .execute("ALTER TABLE message_body_edge RENAME TO hidden_edges", [])
        .unwrap();
    assert!(matches!(
        next(
            &writer._index,
            &file,
            owner,
            Kind::Blocks,
            root,
            None,
            &mut Work::default()
        ),
        Err(Error::Sqlite { .. })
    ));
}
/// Collect unreferenced nodes one per transaction until none remain.
fn drain(writer: &mut crate::Writer, file: &Path) -> usize {
    let mut collected = 0;
    loop {
        let pk: Option<i64> = writer
            ._index
            .query_row(
                "SELECT pk FROM message_body_node WHERE incoming=0 ORDER BY pk LIMIT 1",
                [],
                |r| r.get(0),
            )
            .optional()
            .unwrap();
        let Some(pk) = pk else { return collected };
        let tx = writer._index.transaction().unwrap();
        assert!(collect(&tx, file, pk).unwrap());
        tx.commit().unwrap();
        collected += 1;
    }
}
fn install(
    tx: &Transaction<'_>,
    file: &Path,
    owner: Owner,
    body: i64,
    root: Option<i64>,
    key: u64,
    leaf: Edge,
) -> (i64, Work) {
    let mut work = Work::default();
    let (new, _) = replace(
        tx,
        file,
        owner,
        Kind::Blocks,
        root,
        key,
        Some(leaf),
        &mut work,
    )
    .unwrap();
    acquire(tx, file, owner, new).unwrap();
    // Hold the new root through a build-frontier row, as production does.
    tx.execute("INSERT INTO message_build(message_pk,session_pk,body_pk,basis_digest,contributor_revision,removal_epoch,phase) VALUES(1,1,?1,zeroblob(32),0,0,'layout') ON CONFLICT(message_pk) DO NOTHING",[body]).unwrap();
    let build: i64 = tx
        .query_row("SELECT pk FROM message_build WHERE message_pk=1", [], |r| {
            r.get(0)
        })
        .unwrap();
    tx.execute("INSERT INTO message_build_frontier(build_pk,message_pk,session_pk,depth,node_pk,next_slot) VALUES(?1,1,1,0,?2,0) ON CONFLICT(build_pk,depth) DO UPDATE SET node_pk=excluded.node_pk",params![build,new]).unwrap();
    if let Some(root) = root {
        release(tx, file, owner, root).unwrap();
    }
    (new, work)
}
#[test]
fn replace_copies_one_path_and_old_snapshot_survives_collect() {
    let (_dir, mut writer, owner, body) = fixture();
    let file = writer.root.join("index.db");
    let mut root = None;
    for i in 0..128u64 {
        let tx = writer._index.transaction().unwrap();
        let item = leaf(&tx, i);
        let (next, work) = install(&tx, &file, owner, body, root, i, item);
        assert_eq!(work.nodes_written, 16);
        assert!(work.edges_written <= 256);
        tx.commit().unwrap();
        root = Some(next);
    }
    let old = root.unwrap();
    let reader = writer.read_handle().unwrap();
    reader.index.execute_batch("BEGIN").unwrap();
    let mut work = Work::default();
    let before = lookup(
        &reader.index,
        &file,
        owner,
        Kind::Blocks,
        Some(old),
        0,
        &mut work,
    )
    .unwrap()
    .unwrap();
    assert!(work.nodes_read <= 16 && work.edges_read <= 256);
    let tx = writer._index.transaction().unwrap();
    let item = leaf(&tx, 999);
    let (new, work) = install(&tx, &file, owner, body, root, 0, item);
    assert_eq!(work.nodes_read, 16);
    assert_eq!(work.nodes_written, 16);
    assert!(work.edges_written <= 256);
    assert!(work.references_changed <= 256);
    tx.commit().unwrap();
    let tx = writer._index.transaction().unwrap();
    assert!(collect(&tx, &file, old).unwrap());
    tx.commit().unwrap();
    assert_eq!(
        lookup(
            &reader.index,
            &file,
            owner,
            Kind::Blocks,
            Some(old),
            0,
            &mut Work::default()
        )
        .unwrap(),
        Some(before)
    );
    assert_eq!(
        lookup(
            &writer._index,
            &file,
            owner,
            Kind::Blocks,
            Some(new),
            0,
            &mut Work::default()
        )
        .unwrap(),
        Some(item)
    );
    reader.index.execute_batch("ROLLBACK").unwrap();
    drop(reader);
    drop(writer);
}
#[test]
fn collect_frees_one_node_per_call_and_keeps_shared_live_path() {
    let (_dir, mut writer, owner, body) = fixture();
    let file = writer.root.join("index.db");
    let tx = writer._index.transaction().unwrap();
    let item = leaf(&tx, 0);
    let (old, _) = install(&tx, &file, owner, body, None, 0, item);
    let build: i64 = tx
        .query_row("SELECT pk FROM message_build WHERE message_pk=1", [], |r| {
            r.get(0)
        })
        .unwrap();
    acquire(&tx, &file, owner, old).unwrap();
    tx.execute("INSERT INTO message_build_frontier(build_pk,message_pk,session_pk,depth,node_pk,next_slot) VALUES(?1,1,1,1,?2,0)",params![build,old]).unwrap();
    tx.commit().unwrap();
    let tx = writer._index.transaction().unwrap();
    let item2 = leaf(&tx, 1);
    let (new, _) = install(&tx, &file, owner, body, Some(old), 1, item2);
    tx.commit().unwrap();
    let tx = writer._index.transaction().unwrap();
    tx.execute(
        "DELETE FROM message_build_frontier WHERE build_pk=?1 AND depth=0",
        [build],
    )
    .unwrap();
    release(&tx, &file, owner, new).unwrap();
    tx.commit().unwrap();
    assert_eq!(drain(&mut writer, &file), 16);
    assert_eq!(
        lookup(
            &writer._index,
            &file,
            owner,
            Kind::Blocks,
            Some(old),
            0,
            &mut Work::default()
        )
        .unwrap(),
        Some(item)
    );
    assert!(
        lookup(
            &writer._index,
            &file,
            owner,
            Kind::Blocks,
            Some(old),
            1,
            &mut Work::default()
        )
        .unwrap()
        .is_none()
    );
    let tx = writer._index.transaction().unwrap();
    tx.execute(
        "DELETE FROM message_build_frontier WHERE build_pk=?1 AND depth=1",
        [build],
    )
    .unwrap();
    release(&tx, &file, owner, old).unwrap();
    tx.commit().unwrap();
    assert_eq!(drain(&mut writer, &file), 16);
    assert_eq!(
        writer
            ._index
            .query_row("SELECT count(*) FROM message_body_node", [], |r| r
                .get::<_, i64>(0))
            .unwrap(),
        0
    );
}

#[test]
fn collect_skips_referenced_root_and_rollback_keeps_children() {
    let (_dir, mut writer, owner, body) = fixture();
    let file = writer.root.join("index.db");
    let tx = writer._index.transaction().unwrap();
    let item = leaf(&tx, 0);
    let (root, _) = install(&tx, &file, owner, body, None, 0, item);
    tx.commit().unwrap();
    let tx = writer._index.transaction().unwrap();
    assert!(!collect(&tx, &file, root).unwrap());
    tx.commit().unwrap();
    let tx = writer._index.transaction().unwrap();
    let second = leaf(&tx, 1);
    let (next, _) = install(&tx, &file, owner, body, Some(root), 1, second);
    tx.commit().unwrap();
    let child: i64 = writer
        ._index
        .query_row(
            "SELECT child_node_pk FROM message_body_edge WHERE parent_pk=?1 AND slot=0",
            [root],
            |r| r.get(0),
        )
        .unwrap();
    let refs: i64 = writer
        ._index
        .query_row(
            "SELECT incoming FROM message_body_node WHERE pk=?1",
            [child],
            |r| r.get(0),
        )
        .unwrap();
    let tx = writer._index.transaction().unwrap();
    assert!(collect(&tx, &file, root).unwrap());
    tx.rollback().unwrap();
    assert_eq!(
        writer
            ._index
            .query_row(
                "SELECT incoming FROM message_body_node WHERE pk=?1",
                [child],
                |r| r.get::<_, i64>(0)
            )
            .unwrap(),
        refs
    );
    let tx = writer._index.transaction().unwrap();
    assert!(collect(&tx, &file, root).unwrap());
    tx.commit().unwrap();
    assert!(
        lookup(
            &writer._index,
            &file,
            owner,
            Kind::Blocks,
            Some(next),
            0,
            &mut Work::default()
        )
        .unwrap()
        .is_some()
    );
    assert!(
        lookup(
            &writer._index,
            &file,
            owner,
            Kind::Blocks,
            Some(next),
            1,
            &mut Work::default()
        )
        .unwrap()
        .is_some()
    );
    assert!(
        load(
            &writer._index,
            &file,
            owner,
            Kind::Blocks,
            root,
            &mut Work::default()
        )
        .is_err()
    );
}
#[test]
fn conflict_fact_subtree_is_shared_and_old_snapshot_survives_collect() {
    let (_dir, mut writer, owner, body) = fixture();
    let file = writer.root.join("index.db");
    let tx = writer._index.transaction().unwrap();
    tx.execute(
        "INSERT INTO message_field_slot(message_pk,session_pk,semantic_key) VALUES(1,1,x'01')",
        [],
    )
    .unwrap();
    let slot = tx.last_insert_rowid();
    let mut facts_root = None;
    let mut facts = Vec::new();
    for i in 1..=64i64 {
        tx.execute("INSERT INTO message_candidate(message_pk,session_pk,input_pk,chain,revision,revision_length,evidence_revision,logical_key,parser_version,source_state_generation,source_state_step,source_basis_digest,basis_digest,finish,route) VALUES(1,1,1,1,1,100,1,?1,1,1,0,zeroblob(32),zeroblob(32),1,2)",[blake3::hash(&i.to_be_bytes()).as_bytes().as_slice()]).unwrap();
        let candidate = tx.last_insert_rowid();
        let bytes = i.to_string();
        tx.execute("INSERT INTO body_value(codec,length,hash,state,incoming) VALUES('json',?1,?2,'ready',1)",params![bytes.len() as i64,blake3::hash(bytes.as_bytes()).as_bytes().as_slice()]).unwrap();
        let value = tx.last_insert_rowid();
        tx.execute(
            "INSERT INTO body_value_chunk(value_pk,offset,bytes) VALUES(?1,0,?2)",
            params![value, bytes.as_bytes()],
        )
        .unwrap();
        tx.execute("INSERT INTO message_field_fact(message_pk,session_pk,candidate_pk,slot_pk,value_pk,supplied) VALUES(1,1,?1,?2,?3,1)",params![candidate,slot,value]).unwrap();
        let fact = tx.last_insert_rowid();
        facts.push(fact);
        let (root, _) = replace(
            &tx,
            &file,
            owner,
            Kind::ConflictFacts,
            facts_root,
            fact as u64,
            Some(Edge {
                link: Link::Field(fact),
                counts: Counts::default(),
            }),
            &mut Work::default(),
        )
        .unwrap();
        facts_root = Some(root);
    }
    let conflict_facts = facts_root.unwrap();
    let conflict = Edge {
        link: Link::Conflict {
            slot,
            root: conflict_facts,
            height: 15,
        },
        counts: Counts::default(),
    };
    let (first, _) = replace(
        &tx,
        &file,
        owner,
        Kind::Fields,
        None,
        slot as u64,
        Some(conflict),
        &mut Work::default(),
    )
    .unwrap();
    acquire(&tx, &file, owner, first).unwrap();
    tx.execute(
        "UPDATE message_body SET field_root_pk=?1 WHERE pk=?2",
        params![first, body],
    )
    .unwrap();
    // Changing another field shares the conflict's fact subtree. A build-frontier
    // row holds the new root while the old body still exists.
    tx.execute(
        "INSERT INTO message_field_slot(message_pk,session_pk,semantic_key) VALUES(1,1,x'02')",
        [],
    )
    .unwrap();
    let other_slot = tx.last_insert_rowid();
    let (candidate, value): (i64, i64) = tx
        .query_row(
            "SELECT candidate_pk,value_pk FROM message_field_fact WHERE pk=?1",
            [facts[0]],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    tx.execute("INSERT INTO message_field_fact(message_pk,session_pk,candidate_pk,slot_pk,value_pk,supplied) VALUES(1,1,?1,?2,?3,1)",params![candidate,other_slot,value]).unwrap();
    let other_fact = tx.last_insert_rowid();
    tx.execute(
        "UPDATE body_value SET incoming=incoming+1 WHERE pk=?1",
        [value],
    )
    .unwrap();
    let mut work = Work::default();
    let (second, _) = replace(
        &tx,
        &file,
        owner,
        Kind::Fields,
        Some(first),
        other_slot as u64,
        Some(Edge {
            link: Link::Field(other_fact),
            counts: Counts {
                source_fields: 1,
                ..Counts::default()
            },
        }),
        &mut work,
    )
    .unwrap();
    assert_eq!(work.nodes_written, 16);
    assert!(work.nodes_read <= 16);
    assert!(work.edges_written <= 256);
    acquire(&tx, &file, owner, second).unwrap();
    tx.execute("INSERT INTO message_build(message_pk,session_pk,body_pk,basis_digest,contributor_revision,removal_epoch,phase) VALUES(1,1,?1,zeroblob(32),0,0,'fold')",[body]).unwrap();
    let build = tx.last_insert_rowid();
    tx.execute("INSERT INTO message_build_frontier(build_pk,message_pk,session_pk,depth,node_pk,next_slot) VALUES(?1,1,1,0,?2,0)",params![build,second]).unwrap();
    assert!(
        tx.execute(
            "UPDATE message_body_edge SET child_kind=NULL WHERE conflict_slot_pk IS NOT NULL",
            []
        )
        .is_err()
    );
    assert!(
        tx.execute(
            "UPDATE message_body_edge SET child_height=NULL WHERE conflict_slot_pk IS NOT NULL",
            []
        )
        .is_err()
    );
    tx.commit().unwrap();
    let reader = writer.read_handle().unwrap();
    reader.index.execute_batch("BEGIN").unwrap();
    assert_eq!(
        lookup(
            &reader.index,
            &file,
            owner,
            Kind::Fields,
            Some(first),
            slot as u64,
            &mut Work::default()
        )
        .unwrap(),
        Some(conflict)
    );
    let tx = writer._index.transaction().unwrap();
    tx.execute(
        "UPDATE message_body SET field_root_pk=NULL WHERE pk=?1",
        [body],
    )
    .unwrap();
    release(&tx, &file, owner, first).unwrap();
    tx.commit().unwrap();
    drain(&mut writer, &file);
    assert_eq!(
        lookup(
            &writer._index,
            &file,
            owner,
            Kind::Fields,
            Some(second),
            slot as u64,
            &mut Work::default()
        )
        .unwrap(),
        Some(conflict)
    );
    assert_eq!(
        lookup(
            &writer._index,
            &file,
            owner,
            Kind::ConflictFacts,
            Some(conflict_facts),
            facts[63] as u64,
            &mut Work::default()
        )
        .unwrap()
        .unwrap()
        .link,
        Link::Field(facts[63])
    );
    let tx = writer._index.transaction().unwrap();
    tx.execute(
        "DELETE FROM message_build_frontier WHERE build_pk=?1",
        [build],
    )
    .unwrap();
    release(&tx, &file, owner, second).unwrap();
    tx.commit().unwrap();
    drain(&mut writer, &file);
    assert!(
        load(
            &writer._index,
            &file,
            owner,
            Kind::ConflictFacts,
            conflict_facts,
            &mut Work::default()
        )
        .is_err()
    );
    assert_eq!(
        lookup(
            &reader.index,
            &file,
            owner,
            Kind::Fields,
            Some(first),
            slot as u64,
            &mut Work::default()
        )
        .unwrap(),
        Some(conflict)
    );
    assert_eq!(
        lookup(
            &reader.index,
            &file,
            owner,
            Kind::ConflictFacts,
            Some(conflict_facts),
            facts[0] as u64,
            &mut Work::default()
        )
        .unwrap()
        .unwrap()
        .link,
        Link::Field(facts[0])
    );
    reader.index.execute_batch("ROLLBACK").unwrap();
}
