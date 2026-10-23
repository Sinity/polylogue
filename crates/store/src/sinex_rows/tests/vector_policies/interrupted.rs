use super::*;
use crate::{
    StoreError,
    excise::recovery,
    fault::{self, Action},
    fixture::{Fixture, MessageFacts, SessionFacts},
};
use std::{collections::BTreeMap, path::Path, process::Command};

#[derive(serde::Serialize, serde::Deserialize)]
struct View {
    id: String,
    descriptors: Vec<RestoreUnit>,
    payloads: BTreeMap<String, Vec<u8>>,
    retired: String,
}

impl View {
    fn selected(&self) -> Selection {
        select(&self.descriptors).unwrap()
    }

    fn install(&self, target: &mut Restoring) -> crate::Result<u64> {
        target.install(&self.id, &self.selected(), |unit| {
            let name = unit.batch.as_ref().unwrap();
            assert_ne!(name, &self.retired, "retired before-image was requested");
            Ok(Some(Cursor::new(self.payloads[name].clone())))
        })
    }
}

fn view() -> View {
    let (_dir, source, _lock, mut excision, _, id) = published();
    let conn = open_write(&source.archive_db(), cache::SETUP).unwrap();
    conn.execute("UPDATE excision SET commit_seq=2", [])
        .unwrap();
    conn.execute("INSERT INTO note(id,root,kind,value,author,machine,created_at,commit_seq) VALUES ('selected','selected','note','obsolete synthetic text','operator','synthetic',1,1)", []).unwrap();
    let unit = Unit {
        commit_seq: 1,
        kind: "note".into(),
        key: "selected".into(),
    };
    let old = read(&conn, &unit).unwrap();
    let retired = format!("blake3:{}", blake3::hash(&old.contents()));
    let original = RestoreUnit {
        unit: unit.clone(),
        batch: Some(retired.clone()),
        bytes: Some(old.bytes),
        rows: old.rows,
        blobs: vec![],
        excision: None,
    };
    conn.execute("UPDATE note SET value='current exact value'", [])
        .unwrap();
    let replacement = read(&conn, &unit).unwrap();
    let replacement_bytes = replacement.contents();
    let name = format!("blake3:{}", blake3::hash(&replacement_bytes));
    let mut publication = excision.excision.take().unwrap();
    publication.replacements.push(Replacement {
        unit,
        batch: Some(name.clone()),
        bytes: replacement.bytes,
        rows: replacement.rows,
        blobs: vec![],
    });
    publication.retired_batches.push(retired.clone());
    publication.digest = publication.computed_digest().unwrap();
    conn.execute(
        "UPDATE excision SET refused_json=?1",
        [json!({"publication":publication}).to_string()],
    )
    .unwrap();
    excision.unit.commit_seq = 2;
    let batch = read(&conn, &excision.unit).unwrap();
    let bytes = batch.contents();
    let excision_name = format!("blake3:{}", blake3::hash(&bytes));
    excision.batch = Some(excision_name.clone());
    excision.bytes = Some(batch.bytes);
    excision.rows = batch.rows;
    excision.excision = Some(publication);
    View {
        id,
        descriptors: vec![original, excision],
        payloads: BTreeMap::from([(name, replacement_bytes), (excision_name, bytes)]),
        retired,
    }
}

fn target(root: &Path, id: &str) {
    let mut fixture = Fixture::new(root).unwrap();
    let session = fixture
        .session(SessionFacts {
            id: "claude-code:old",
            ..Default::default()
        })
        .unwrap();
    fixture
        .message(
            session,
            MessageFacts {
                id: "claude-code:old-message",
                role: "user",
                written_by: "human",
                text: Some("obsolete synthetic text"),
                ..Default::default()
            },
        )
        .unwrap();
    open_write(&fixture.archive.archive_db(), cache::SETUP)
        .unwrap()
        .execute("UPDATE meta SET value=?1 WHERE key='archive_id'", [id])
        .unwrap();
    let vectors = open_write(&fixture.archive.embeddings_db(), cache::SETUP).unwrap();
    vectors
        .execute(
            "UPDATE vector_meta SET value=?1 WHERE key='archive_id'",
            [id],
        )
        .unwrap();
    for hash in [1, 2] {
        vectors.execute("INSERT INTO embedded(text_hash,model,dims,vector,embedded_at) VALUES (?1,'test',1,x'0000803f',1)", [vec![hash; 32]]).unwrap();
        vectors.execute("INSERT INTO embed_failure(text_hash,model,error,failed_at) VALUES (?1,'test','old failure',1)", [vec![hash; 32]]).unwrap();
    }
}

fn checkpoint(conn: &Connection) -> String {
    conn.query_row(
        "SELECT value FROM meta WHERE key='sinex_restored_view'",
        [],
        |r| r.get(0),
    )
    .unwrap()
}

fn note(conn: &Connection) -> String {
    conn.query_row("SELECT value FROM note WHERE id='selected'", [], |r| {
        r.get(0)
    })
    .unwrap()
}

fn hashes(archive: &Archive) -> Vec<Vec<u8>> {
    let c = open_write(&archive.embeddings_db(), cache::SETUP).unwrap();
    c.prepare("SELECT text_hash FROM embedded ORDER BY text_hash")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap()
}

#[test]
fn post_commit_restore_failures_recover_without_replaying_old_payloads() {
    exercise(false);
}

#[test]
fn a_stopped_restore_recovers_without_replaying_old_payloads() {
    if let Some(root) = std::env::var_os("POLYLOGUE_RESTORE_STOP_FIXTURE") {
        let root = std::path::PathBuf::from(root);
        let view: View =
            serde_json::from_slice(&std::fs::read(root.join("view.json")).unwrap()).unwrap();
        let (archive, _lock) = Archive::open_or_create(&root.join("target")).unwrap();
        let mut restoring = Restoring::open(&archive).unwrap();
        let point = std::env::var("POLYLOGUE_RESTORE_STOP_POINT").unwrap();
        // Exit at the injected boundary without unwinding or dropping connections.
        let expected = format!("injected panic at {point} (");
        std::panic::set_hook(Box::new(move |info| {
            let message = info
                .payload()
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| info.payload().downcast_ref::<&str>().copied());
            eprintln!("{info}");
            std::process::exit(
                if message.is_some_and(|message| message.starts_with(&expected)) {
                    86
                } else {
                    88
                },
            );
        }));
        fault::arm(&point, "", Action::Panic, 0, 1);
        view.install(&mut restoring).unwrap();
        std::process::exit(87);
    }
    exercise(true);
}

fn exercise(stop: bool) {
    for point in [
        "restore.after_canonical_commit",
        "excision.after_vector_cleanup",
        "restore.after_vector_reconciliation",
        "index.after_retirement",
        "index.after_creation",
    ] {
        fault::clear();
        let dir = tempfile::tempdir().unwrap();
        let view = view();
        let path = dir.path().join("target");
        target(&path, &view.id);
        let archive = Archive::at(&path);
        let pool = crate::ReadPool::open(&archive).unwrap();
        let reserved = pool.reader().unwrap();
        if stop {
            std::fs::write(
                dir.path().join("view.json"),
                serde_json::to_vec(&view).unwrap(),
            )
            .unwrap();
            let status = Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "sinex_rows::tests::vector_policies::interrupted::a_stopped_restore_recovers_without_replaying_old_payloads", "--nocapture"])
                .env("POLYLOGUE_RESTORE_STOP_FIXTURE", dir.path())
                .env("POLYLOGUE_RESTORE_STOP_POINT", point)
                .status()
                .unwrap();
            assert_eq!(status.code(), Some(86), "{point}");
        } else {
            let (archive, _lock) = Archive::open_or_create(&path).unwrap();
            let mut restoring = Restoring::open(&archive).unwrap();
            fault::arm(point, "", Action::Fail, 0, 1);
            let error = view.install(&mut restoring).unwrap_err();
            assert!(error.to_string().contains(point), "{point}: {error}");
        }
        fault::clear();
        let canonical = crate::archive::open_read(&archive.archive_db()).unwrap();
        let committed_checkpoint = checkpoint(&canonical);
        assert_eq!(
            committed_checkpoint,
            blake3::hash(&serde_json::to_vec(&(Some(view.id.as_str()), &view.selected())).unwrap())
                .to_hex()
                .to_string()
        );
        let revision = recovery::current(&canonical).unwrap();
        let rules = recovery::fingerprint(&canonical).unwrap();
        assert!(revision.version > 0);
        assert_eq!(note(&canonical), "current exact value");
        assert!(reserved.status().is_err(), "reserved reader at {point}");
        drop(reserved);
        drop(pool);
        let pool = crate::ReadPool::open(&archive).unwrap();
        match pool.reader() {
            Ok(reader) => assert!(reader.status().is_err(), "new reader at {point}"),
            Err(_) => assert!(!archive.index_db().exists(), "existing index at {point}"),
        }
        drop(pool);
        let vectors = crate::vectors::Vectors::for_reader(&archive, 128, false, revision.clone());
        if point == "restore.after_canonical_commit" {
            assert!(matches!(vectors, Err(StoreError::ExcisionPending)));
            assert_eq!(hashes(&archive), [vec![1; 32], vec![2; 32]]);
        } else {
            assert_eq!(
                vectors.unwrap().done("test").unwrap(),
                std::collections::HashSet::from([vec![2; 32]])
            );
            assert_eq!(hashes(&archive), [vec![2; 32]]);
        }
        drop(canonical);
        let (reopened, _lock) = Archive::open_or_create(&path).unwrap();
        let c = open_write(&reopened.archive_db(), cache::SETUP).unwrap();
        assert_eq!(checkpoint(&c), committed_checkpoint);
        assert_eq!(recovery::current(&c).unwrap(), revision);
        assert_eq!(recovery::fingerprint(&c).unwrap(), rules);
        let pool = crate::ReadPool::open(&reopened).unwrap();
        let reader = pool.reader().unwrap();
        assert_eq!(reader.status().unwrap().messages, 0);
        assert_eq!(
            reader.vectors().unwrap().done("test").unwrap(),
            std::collections::HashSet::from([vec![2; 32]])
        );
        drop(reader);
        drop(pool);
        // A completed receipt permits a later purchase of the same hash.
        open_write(&reopened.embeddings_db(), cache::SETUP).unwrap().execute("INSERT INTO embedded(text_hash,model,dims,vector,embedded_at) VALUES (?1,'test',1,x'0000803f',2)", [vec![1; 32]]).unwrap();
        c.execute(
            "UPDATE note SET value='later local exact value' WHERE id='selected'",
            [],
        )
        .unwrap();
        assert_eq!(
            view.install(&mut Restoring::open(&reopened).unwrap())
                .unwrap(),
            0
        );
        assert_eq!(checkpoint(&c), committed_checkpoint);
        assert_eq!(recovery::current(&c).unwrap(), revision);
        assert_eq!(recovery::fingerprint(&c).unwrap(), rules);
        assert_eq!(hashes(&reopened), [vec![1; 32], vec![2; 32]]);
        assert_eq!(note(&c), "later local exact value");
    }
}
