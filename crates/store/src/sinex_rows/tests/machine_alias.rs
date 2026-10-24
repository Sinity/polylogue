use super::*;
use crate::archive::{Archive, cache, open_write};
use crate::writer::{Batch as WriteBatch, InFlight, InputKey, Observation};
use crate::{Change, Writer};
use std::io::Cursor;

fn observe(writer: &Writer, host: &str, at: i64, change: Change) {
    writer
        .send(WriteBatch {
            input: InputKey {
                id: "input".into(),
                format: "synthetic-text".into(),
                route_rank: 0,
            },
            change,
            observation: Some(Observation {
                machine: "machine".into(),
                host: host.into(),
                path: "synthetic".into(),
                facts: None,
                seen_at_ms: at,
                import: None,
            }),
            continues: false,
            piece: None,
            blobs: vec![],
            length: 0,
            parse: vec![],
            release: InFlight::default(),
        })
        .unwrap();
    assert!(writer.flush().unwrap().errors.is_empty());
}

#[test]
fn machine_aliases_leave_old_observation_bytes_exact_after_restore_and_restart() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, lock) = Archive::open_or_create(dir.path()).unwrap();
    let writer = Writer::start(&archive).unwrap();
    let mut original = None;
    for (host, at, change) in [
        ("host-a", 900, Change::First),
        ("host-b", 1, Change::Unchanged),
    ] {
        observe(&writer, host, at, change);
        let published = Published::open(&archive.root).unwrap();
        let inventory = published.inventory().unwrap();
        if let Some((unit, bytes, hash)) = &original {
            let actual = published.read(unit).unwrap().contents();
            assert_eq!(&actual, bytes);
            assert_eq!(&blake3::hash(&actual), hash);
        } else {
            let unit = (1..=inventory.len())
                .map(|p| inventory.unit(p).unwrap())
                .find(|u| u.kind == "observation")
                .unwrap();
            let bytes = published.read(&unit).unwrap().contents();
            let hash = blake3::hash(&bytes);
            original = Some((unit, bytes, hash));
        }
    }
    drop(writer);
    drop(lock);
    drop(archive);
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let published = Published::open(&archive.root).unwrap();
    let (units, payloads) = super::append_facts::descriptors(&published);
    let target_dir = tempfile::tempdir().unwrap();
    let (target, target_lock) = Archive::open_or_create(target_dir.path()).unwrap();
    let mut restore = Restoring::open(&target).unwrap();
    let selected = select(&units).unwrap();
    restore
        .install(&published.archive_id().unwrap(), &selected, |u| {
            Ok(Some(Cursor::new(
                payloads[units.iter().position(|x| x == u).unwrap()].clone(),
            )))
        })
        .unwrap();
    drop(restore);
    drop(target_lock);
    let (target, _lock) = Archive::open_or_create(target_dir.path()).unwrap();
    let restored = Published::open(&target.root).unwrap();
    let (unit, bytes, hash) = original.unwrap();
    let actual = restored.read(&unit).unwrap().contents();
    assert_eq!(actual, bytes);
    assert_eq!(blake3::hash(&actual), hash);
    for (unit, bytes) in units.iter().zip(&payloads) {
        assert_eq!(restored.read(&unit.unit).unwrap().contents(), *bytes);
    }
    assert!(
        payloads
            .iter()
            .any(|bytes| String::from_utf8_lossy(bytes).contains("host-b"))
    );
}

#[test]
fn legacy_alias_migration_keeps_current_batches_and_excludes_a_future_alias() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, lock) = Archive::open_or_create(dir.path()).unwrap();
    let writer = Writer::start(&archive).unwrap();
    observe(&writer, "host-a", 900, Change::First);
    observe(&writer, "host-b", 1, Change::Unchanged);
    drop(writer);
    let db = open_write(&archive.archive_db(), cache::SETUP).unwrap();
    // Archive8 exposed every current alias to every observation. Capture that
    // closure, then remove the new local-only column to exercise its migration.
    db.execute("UPDATE machine SET first_commit_seq=0", [])
        .unwrap();
    let source = Published::open(&archive.root).unwrap();
    let before = super::append_facts::descriptors(&source);
    drop(source);
    db.execute_batch("ALTER TABLE machine DROP COLUMN first_commit_seq; PRAGMA user_version=8;")
        .unwrap();
    drop(db);
    drop(lock);
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let source = Published::open(&archive.root).unwrap();
    assert_eq!(super::append_facts::descriptors(&source), before);
    drop(source);
    let writer = Writer::start(&archive).unwrap();
    observe(&writer, "host-c", 0, Change::Unchanged);
    drop(writer);
    let source = Published::open(&archive.root).unwrap();
    for (unit, bytes) in before.0.iter().zip(before.1.iter()) {
        assert_eq!(source.read(&unit.unit).unwrap().contents(), *bytes);
    }
    let db = open_write(&archive.archive_db(), cache::SETUP).unwrap();
    assert_eq!(
        db.query_row("SELECT COUNT(*) FROM machine", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        3
    );
    assert_eq!(
        db.query_row(
            "SELECT COUNT(*) FROM machine WHERE first_commit_seq=0",
            [],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        2
    );
}

#[test]
fn verified_older_machine_closure_refuses_a_migrated_conflicting_target_atomically() {
    let dir = tempfile::tempdir().unwrap();
    let (source, _lock) = Archive::open_or_create(&dir.path().join("source")).unwrap();
    let writer = Writer::start(&source).unwrap();
    observe(&writer, "host-a", 900, Change::First);
    observe(&writer, "host-b", 1, Change::Unchanged);
    drop(writer);
    let published = Published::open(&source.root).unwrap();
    let id = published.archive_id().unwrap();
    let (units, payloads) = super::append_facts::descriptors(&published);
    let selected = select(&units).unwrap();
    let (target, lock) = Archive::open_or_create(&dir.path().join("target")).unwrap();
    let install = |restore: &mut Restoring| {
        restore.install(&id, &selected, |u| {
            Ok(Some(Cursor::new(
                payloads[units.iter().position(|x| x == u).unwrap()].clone(),
            )))
        })
    };
    let mut restore = Restoring::open(&target).unwrap();
    install(&mut restore).unwrap();
    drop(restore);
    let mut db = open_write(&target.archive_db(), cache::SETUP).unwrap();
    db.execute_batch("ALTER TABLE machine DROP COLUMN first_commit_seq; PRAGMA user_version=8;")
        .unwrap();
    crate::schema::migrate_archive(&mut db).unwrap();
    // Migration preserves the current alias set. The original verified older
    // batch now proves that the target's baseline closure already diverged.
    let before = super::append_facts::descriptors(&Published::open(&target.root).unwrap());
    let checkpoint: String = db
        .query_row(
            "SELECT value FROM meta WHERE key='sinex_restored_view'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    let counter: String = db
        .query_row(
            "SELECT value FROM meta WHERE key='write_counter'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    let mut restore = Restoring::open(&target).unwrap();
    assert!(
        matches!(install(&mut restore),Err(crate::StoreError::Invalid(message)) if message.contains("machine dependencies conflict"))
    );
    assert_eq!(
        super::append_facts::descriptors(&Published::open(&target.root).unwrap()),
        before
    );
    assert_eq!(
        db.query_row(
            "SELECT value FROM meta WHERE key='sinex_restored_view'",
            [],
            |r| r.get::<_, String>(0)
        )
        .unwrap(),
        checkpoint
    );
    assert_eq!(
        db.query_row(
            "SELECT value FROM meta WHERE key='write_counter'",
            [],
            |r| r.get::<_, String>(0)
        )
        .unwrap(),
        counter
    );
    // A changed view forces row installation before the same conflict, and
    // must roll that work back as well as preserve the checkpoint.
    let mut changed = units.clone();
    let mut changed_payloads = payloads.clone();
    let at = changed
        .iter()
        .position(|u| u.unit.kind == "observation")
        .unwrap();
    let mut rows: Vec<serde_json::Value> = changed_payloads[at]
        .split(|b| *b == b'\n')
        .filter(|l| !l.is_empty())
        .map(|l| serde_json::from_slice(l).unwrap())
        .collect();
    for row in &mut rows {
        if row["table"] == "observation" {
            row["row"]["path"] = "changed-uncommitted-path".into();
        }
    }
    let mut bytes = Vec::new();
    for row in rows {
        serde_json::to_writer(&mut bytes, &row).unwrap();
        bytes.push(b'\n');
    }
    changed[at].batch = Some(format!("blake3:{}", blake3::hash(&bytes)));
    changed[at].bytes = Some(bytes.len() as u64);
    changed_payloads[at] = bytes;
    let changed_selection = select(&changed).unwrap();
    assert!(
        matches!(restore.install(&id,&changed_selection,|u|Ok(Some(Cursor::new(changed_payloads[changed.iter().position(|x|x==u).unwrap()].clone())))),Err(crate::StoreError::Invalid(message)) if message.contains("machine dependencies conflict"))
    );
    assert_eq!(
        super::append_facts::descriptors(&Published::open(&target.root).unwrap()),
        before
    );
    assert_eq!(
        db.query_row(
            "SELECT value FROM meta WHERE key='sinex_restored_view'",
            [],
            |r| r.get::<_, String>(0)
        )
        .unwrap(),
        checkpoint
    );
    drop(lock);
}

mod history;
