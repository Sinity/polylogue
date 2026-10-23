use super::*;
use crate::{
    Archive,
    archive::{cache, open_write},
};
use std::io::Cursor;

fn published() -> (
    tempfile::TempDir,
    Archive,
    crate::ArchiveLock,
    RestoreUnit,
    Vec<u8>,
    String,
) {
    let dir = tempfile::tempdir().unwrap();
    let (archive, lock) = Archive::open_or_create(dir.path()).unwrap();
    let c = open_write(&archive.archive_db(), cache::SETUP).unwrap();
    let mut publication = ExcisionPublication {
        version: 1,
        replacements: vec![],
        retired_batches: vec![],
        digest: String::new(),
    };
    publication.digest = publication.computed_digest().unwrap();
    c.execute("INSERT INTO excision(id,scope,refused_json,created_at,commit_seq) VALUES ('remove','text:synthetic',?1,1,1)",[json!({"publication":publication}).to_string()]).unwrap();
    c.execute(
        "INSERT INTO vector_deletion_policy VALUES ('remove',1,?1)",
        [serde_json::to_string(&vec![hex(&[1; 32])]).unwrap()],
    )
    .unwrap();
    let unit = Unit {
        commit_seq: 1,
        kind: "excision".into(),
        key: "remove".into(),
    };
    let batch = read(&c, &unit).unwrap();
    let bytes = batch.contents();
    let descriptor = RestoreUnit {
        unit,
        batch: Some(format!("blake3:{}", blake3::hash(&bytes))),
        bytes: Some(batch.bytes),
        rows: batch.rows,
        blobs: vec![],
        excision: Some(publication),
    };
    let id = c
        .query_row("SELECT value FROM meta WHERE key='archive_id'", [], |r| {
            r.get::<_, String>(0)
        })
        .unwrap();
    (dir, archive, lock, descriptor, bytes, id)
}

#[test]
fn canonical_restore_reconciles_exact_policies_and_preserves_repurchase_on_repeat() {
    let (_dir, source, _lock, descriptor, bytes, id) = published();
    let dest = tempfile::tempdir().unwrap();
    let (archive, _dest_lock) = Archive::open_or_create(dest.path()).unwrap();
    let mut restoring = Restoring::open(&archive).unwrap();
    let selected = select(std::slice::from_ref(&descriptor)).unwrap();
    restoring
        .install(&id, &selected, |_| Ok(Some(Cursor::new(bytes.clone()))))
        .unwrap();
    let vectors = open_write(&archive.embeddings_db(), cache::SETUP).unwrap();
    for h in [1, 2] {
        vectors.execute("INSERT INTO embedded(text_hash,model,dims,vector,embedded_at) VALUES (?1,'test',1,x'0000803f',1)",[vec![h;32]]).unwrap();
    }
    // A historical vector file loses its receipt; restored canonical policy must replay.
    vectors
        .execute("DELETE FROM vector_policy_receipt", [])
        .unwrap();
    drop(vectors);
    restoring
        .install(&id, &selected, |_| Ok(Some(Cursor::new(bytes.clone()))))
        .unwrap();
    let vectors = open_write(&archive.embeddings_db(), cache::SETUP).unwrap();
    assert_eq!(
        vectors
            .query_row("SELECT count(*) FROM embedded", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        1
    );
    vectors.execute("INSERT INTO embedded(text_hash,model,dims,vector,embedded_at) VALUES (?1,'test',1,x'0000803f',2)",[vec![1;32]]).unwrap();
    drop(vectors);
    let c = open_write(&source.archive_db(), cache::SETUP).unwrap();
    c.execute("UPDATE excision SET reason='new descriptive metadata'", [])
        .unwrap();
    let batch = read(&c, &descriptor.unit).unwrap();
    let updated = batch.contents();
    let mut next = descriptor.clone();
    next.batch = Some(format!("blake3:{}", blake3::hash(&updated)));
    next.bytes = Some(batch.bytes);
    next.rows = batch.rows;
    let selected = select(&[next]).unwrap();
    restoring
        .install(&id, &selected, |_| Ok(Some(Cursor::new(updated.clone()))))
        .unwrap();
    assert_eq!(
        open_write(&archive.embeddings_db(), cache::SETUP)
            .unwrap()
            .query_row("SELECT count(*) FROM embedded", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        2
    );
}

#[test]
fn a_new_owned_policy_refuses_an_old_target_and_cannot_be_dropped_by_restore() {
    let (_dir, _source, _lock, descriptor, bytes, id) = published();
    let selected = select(std::slice::from_ref(&descriptor)).unwrap();
    let mut old = Connection::open_in_memory().unwrap();
    crate::schema::create_archive(&old).unwrap();
    for migration in crate::schema::ARCHIVE_MIGRATIONS.iter().take(4) {
        old.execute_batch(migration).unwrap();
    }
    old.pragma_update(None, "user_version", 5).unwrap();
    assert!(
        restore::install(&mut old, &selected, None, |_| Ok(Some(Cursor::new(
            bytes.clone()
        ))))
        .is_err()
    );
    assert_eq!(
        old.query_row("SELECT count(*) FROM excision", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        0
    );
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let mut restoring = Restoring::open(&archive).unwrap();
    restoring
        .install(&id, &selected, |_| Ok(Some(Cursor::new(bytes.clone()))))
        .unwrap();
    let mut legacy = Vec::new();
    let mut count = 0;
    for line in bytes.split(|b| *b == b'\n').filter(|l| !l.is_empty()) {
        let value: Value = serde_json::from_slice(line).unwrap();
        if value["table"] == "vector_deletion_policy" {
            continue;
        }
        legacy.extend_from_slice(line);
        legacy.push(b'\n');
        count += 1;
    }
    let mut omitted = descriptor;
    omitted.rows = count;
    omitted.bytes = Some(legacy.len() as u64);
    omitted.batch = Some(format!("blake3:{}", blake3::hash(&legacy)));
    let selected = select(&[omitted]).unwrap();
    assert!(
        restoring
            .install(&id, &selected, |_| Ok(Some(Cursor::new(legacy.clone()))))
            .is_err()
    );
    assert_eq!(
        open_write(&archive.archive_db(), cache::SETUP)
            .unwrap()
            .query_row("SELECT count(*) FROM vector_deletion_policy", [], |r| r
                .get::<_, i64>(0))
            .unwrap(),
        1
    );
}

#[test]
fn valid_publication_metadata_changes_do_not_replay_an_immutable_policy() {
    let (_dir, archive, _lock, _descriptor, _bytes, _id) = published();
    let mut c = open_write(&archive.archive_db(), cache::SETUP).unwrap();
    archive.reconcile_vectors(&mut c).unwrap();
    open_write(&archive.embeddings_db(),cache::SETUP).unwrap().execute("INSERT INTO embedded(text_hash,model,dims,vector,embedded_at) VALUES (?1,'repurchased',1,x'0000803f',2)",[vec![1;32]]).unwrap();
    let mut publication = ExcisionPublication {
        version: 1,
        replacements: vec![],
        retired_batches: vec![format!("blake3:{}", hex(&[3; 32]))],
        digest: String::new(),
    };
    publication.digest = publication.computed_digest().unwrap();
    c.execute(
        "UPDATE excision SET refused_json=?1",
        [json!({"publication":publication}).to_string()],
    )
    .unwrap();
    crate::excise::recovery::acknowledge(
        &c,
        &open_write(&archive.index_db(), cache::SETUP).unwrap(),
    )
    .unwrap();
    archive.reconcile_vectors(&mut c).unwrap();
    assert_eq!(
        open_write(&archive.embeddings_db(), cache::SETUP)
            .unwrap()
            .query_row("SELECT count(*) FROM embedded", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        1
    );
}

#[cfg(feature = "faults")]
mod interrupted;
