use super::*;
use crate::archive::{Archive, cache, open_write};
use crate::excise::recovery;
use std::io::Cursor;

fn descriptor(
    conn: &Connection,
    unit: Unit,
    excision: Option<ExcisionPublication>,
) -> (RestoreUnit, Vec<u8>) {
    let batch = read(conn, &unit).unwrap();
    let bytes = batch.contents();
    (
        RestoreUnit {
            unit,
            batch: Some(format!("blake3:{}", blake3::hash(&bytes))),
            bytes: Some(bytes.len() as u64),
            rows: batch.rows,
            blobs: vec![],
            excision,
        },
        bytes,
    )
}
fn install(target: &mut Restoring, id: &str, descriptors: &[(RestoreUnit, Vec<u8>)]) {
    let selected = select(
        &descriptors
            .iter()
            .map(|(u, _)| u.clone())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    target
        .install(id, &selected, |unit| {
            Ok(Some(Cursor::new(
                descriptors
                    .iter()
                    .find(|(u, _)| u.unit == unit.unit)
                    .unwrap()
                    .1
                    .clone(),
            )))
        })
        .unwrap();
}
#[test]
fn restoring_unchanged_excisions_keeps_the_index_but_changed_rows_rebuild_it() {
    let dir = tempfile::tempdir().unwrap();
    let (source, _source_lock) = Archive::open_or_create(&dir.path().join("source")).unwrap();
    let conn = open_write(&source.archive_db(), cache::SETUP).unwrap();
    let mut publication = ExcisionPublication {
        version: 1,
        replacements: vec![],
        retired_batches: vec![],
        digest: String::new(),
    };
    publication.digest = publication.computed_digest().unwrap();
    let metadata = serde_json::json!({"publication":publication}).to_string();
    conn.execute("INSERT INTO excision(id,scope,reason,refused_json,created_at,commit_seq) VALUES ('remove','session:claude-code:gone','original',?1,1,1)",[metadata]).unwrap();
    let id = Published::open(&source.root).unwrap().archive_id().unwrap();
    let excision = Unit {
        commit_seq: 1,
        kind: "excision".into(),
        key: "remove".into(),
    };
    let original = descriptor(&conn, excision.clone(), Some(publication.clone()));
    let (target, _target_lock) = Archive::open_or_create(&dir.path().join("target")).unwrap();
    let mut restoring = Restoring::open(&target).unwrap();
    install(&mut restoring, &id, std::slice::from_ref(&original));
    let canonical = open_write(&target.archive_db(), cache::SETUP).unwrap();
    let before = recovery::current(&canonical).unwrap();
    assert!(before.version > 0);
    let index = open_write(&target.index_db(), cache::SETUP).unwrap();
    index
        .execute(
            "INSERT INTO meta(key,value) VALUES ('test_retained_index','yes')",
            [],
        )
        .unwrap();
    drop(index);
    conn.execute("INSERT INTO note(id,root,kind,value,author,machine,created_at,commit_seq) VALUES ('other','other','note','unrelated','operator','synthetic',1,2)",[]).unwrap();
    let note = descriptor(
        &conn,
        Unit {
            commit_seq: 2,
            kind: "note".into(),
            key: "other".into(),
        },
        None,
    );
    install(&mut restoring, &id, &[original, note.clone()]);
    assert_eq!(recovery::current(&canonical).unwrap(), before);
    let index = open_write(&target.index_db(), cache::SETUP).unwrap();
    assert_eq!(
        index
            .query_row(
                "SELECT value FROM meta WHERE key='test_retained_index'",
                [],
                |r| r.get::<_, String>(0)
            )
            .unwrap(),
        "yes"
    );
    drop(index);
    conn.execute(
        "UPDATE excision SET reason='changed at same id and commit sequence' WHERE id='remove'",
        [],
    )
    .unwrap();
    let changed = descriptor(&conn, excision, Some(publication));
    install(&mut restoring, &id, &[changed, note]);
    assert_ne!(recovery::current(&canonical).unwrap(), before);
    let index = open_write(&target.index_db(), cache::SETUP).unwrap();
    assert_eq!(
        index
            .query_row(
                "SELECT count(*) FROM meta WHERE key='test_retained_index'",
                [],
                |r| r.get::<_, i64>(0)
            )
            .unwrap(),
        0
    );
    assert_eq!(
        recovery::current(&index).unwrap(),
        recovery::current(&canonical).unwrap()
    );
    let reads = crate::ReadPool::open(&target).unwrap();
    assert!(reads.reader().is_ok());
}

#[test]
fn a_restore_policy_conflict_rolls_back_rows_revision_and_checkpoint() {
    let dir = tempfile::tempdir().unwrap();
    let (source, _source_lock) = Archive::open_or_create(&dir.path().join("source")).unwrap();
    let conn = open_write(&source.archive_db(), cache::SETUP).unwrap();
    let mut publication = ExcisionPublication {
        version: 1,
        replacements: vec![],
        retired_batches: vec![],
        digest: String::new(),
    };
    publication.digest = publication.computed_digest().unwrap();
    let metadata = serde_json::json!({"publication":publication}).to_string();
    conn.execute("INSERT INTO excision(id,scope,refused_json,created_at,commit_seq) VALUES ('remove','session:claude-code:gone',?1,1,1)",[metadata]).unwrap();
    let id = Published::open(&source.root).unwrap().archive_id().unwrap();
    let unit = Unit {
        commit_seq: 1,
        kind: "excision".into(),
        key: "remove".into(),
    };
    let original = descriptor(&conn, unit.clone(), Some(publication.clone()));
    let (target, _target_lock) = Archive::open_or_create(&dir.path().join("target")).unwrap();
    let mut restoring = Restoring::open(&target).unwrap();
    install(&mut restoring, &id, std::slice::from_ref(&original));
    let canonical = open_write(&target.archive_db(), cache::SETUP).unwrap();
    let before = recovery::fingerprint(&canonical).unwrap();
    let revision = recovery::current(&canonical).unwrap();
    let checkpoint: String = canonical
        .query_row(
            "SELECT value FROM meta WHERE key='sinex_restored_view'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    conn.execute(
        "UPDATE excision SET scope='session:claude-code:different' WHERE id='remove'",
        [],
    )
    .unwrap();
    conn.execute("INSERT INTO note(id,root,kind,value,author,machine,created_at,commit_seq) VALUES ('must-rollback','must-rollback','note','uncommitted','operator','synthetic',1,2)",[]).unwrap();
    let changed = descriptor(&conn, unit, Some(publication));
    let note = descriptor(
        &conn,
        Unit {
            commit_seq: 2,
            kind: "note".into(),
            key: "must-rollback".into(),
        },
        None,
    );
    let descriptors = [changed, note];
    let selected = select(
        &descriptors
            .iter()
            .map(|(u, _)| u.clone())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let result = restoring.install(&id, &selected, |unit| {
        Ok(Some(Cursor::new(
            descriptors
                .iter()
                .find(|(u, _)| u.unit == unit.unit)
                .unwrap()
                .1
                .clone(),
        )))
    });
    assert!(
        matches!(&result,Err(crate::StoreError::Invalid(message)) if message=="restored view changes an existing excision rule"),
        "{result:?}"
    );
    assert_eq!(
        recovery::fingerprint(&canonical).unwrap(),
        before,
        "an error must preserve the original canonical removal rule"
    );
    assert_eq!(recovery::current(&canonical).unwrap(), revision);
    assert_eq!(
        canonical
            .query_row(
                "SELECT value FROM meta WHERE key='sinex_restored_view'",
                [],
                |r| r.get::<_, String>(0)
            )
            .unwrap(),
        checkpoint
    );
    assert_eq!(
        canonical
            .query_row(
                "SELECT count(*) FROM note WHERE id='must-rollback'",
                [],
                |r| r.get::<_, i64>(0)
            )
            .unwrap(),
        0
    );
    assert!(
        crate::ReadPool::open(&target)
            .unwrap()
            .reader()
            .unwrap()
            .status()
            .is_ok()
    );
    let retry = restoring.install(&id, &selected, |unit| {
        Ok(Some(Cursor::new(
            descriptors
                .iter()
                .find(|(u, _)| u.unit == unit.unit)
                .unwrap()
                .1
                .clone(),
        )))
    });
    assert!(retry.is_err());
    assert_eq!(recovery::fingerprint(&canonical).unwrap(), before);
    conn.execute(
        "UPDATE excision SET scope='session:claude-code:gone' WHERE id='remove'",
        [],
    )
    .unwrap();
    let repaired = descriptor(
        &conn,
        descriptors[0].0.unit.clone(),
        descriptors[0].0.excision.clone(),
    );
    install(&mut restoring, &id, &[repaired, descriptors[1].clone()]);
    assert_eq!(recovery::fingerprint(&canonical).unwrap(), before);
    assert_eq!(recovery::current(&canonical).unwrap(), revision);
    assert_eq!(
        canonical
            .query_row(
                "SELECT count(*) FROM note WHERE id='must-rollback'",
                [],
                |r| r.get::<_, i64>(0)
            )
            .unwrap(),
        1
    );
    assert!(
        crate::ReadPool::open(&target)
            .unwrap()
            .reader()
            .unwrap()
            .status()
            .is_ok()
    );
}
