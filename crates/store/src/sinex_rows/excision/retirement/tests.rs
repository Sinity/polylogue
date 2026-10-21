use super::*;
use crate::archive::{cache, open_write};
use crate::sinex_rows::{ExcisionPublication, Unit};
use crate::{Archive, StoreError};
use std::collections::{BTreeMap, BTreeSet};

fn replacement(key: &str, batch: Option<u8>) -> Replacement {
    Replacement {
        unit: Unit {
            commit_seq: 1,
            kind: "note".into(),
            key: key.into(),
        },
        batch: batch.map(hash),
        bytes: u64::from(batch.is_some()),
        rows: u64::from(batch.is_some()),
        blobs: vec![],
    }
}
fn hash(value: u8) -> String {
    format!("blake3:{}", format!("{value:02x}").repeat(32))
}
fn source() -> Connection {
    let db = Connection::open_in_memory().unwrap();
    db.execute_batch(
        "CREATE TABLE excision(commit_seq INTEGER,id TEXT,refused_json TEXT);
        CREATE TEMP TABLE excision_units(commit_seq INTEGER,kind TEXT,key TEXT,
            PRIMARY KEY(commit_seq,kind,key));",
    )
    .unwrap();
    db
}
fn history(db: &Connection, seq: i64, id: &str, replacements: Vec<Replacement>) {
    let mut publication = ExcisionPublication {
        version: 1,
        replacements,
        retired_batches: vec![],
        digest: String::new(),
    };
    publication.digest = publication.computed_digest().unwrap();
    db.execute(
        "INSERT INTO excision VALUES (?1,?2,?3)",
        params![
            seq,
            id,
            serde_json::json!({"publication":publication}).to_string()
        ],
    )
    .unwrap();
}
fn changed(db: &Connection, key: &str) {
    db.execute("INSERT INTO excision_units VALUES (1,'note',?1)", [key])
        .unwrap();
}

#[test]
fn shared_batches_tombstones_and_owner_order_determine_retirement() {
    let db = source();
    // Insertion order differs from owner order, including a same-sequence tie.
    history(&db, 4, "z", vec![replacement("unaffected", Some(1))]);
    history(
        &db,
        2,
        "a",
        vec![
            replacement("changed", Some(1)),
            replacement("gone", Some(2)),
        ],
    );
    history(&db, 4, "a", vec![replacement("unaffected", Some(3))]);
    history(&db, 3, "a", vec![replacement("gone", Some(4))]);
    changed(&db, "changed");
    changed(&db, "gone");
    let now = [replacement("changed", Some(5)), replacement("gone", None)];
    // Hash 1 remains referenced by an unaffected unit's latest replacement.
    assert_eq!(batches(&db, &now).unwrap(), [hash(2), hash(4)]);
}

#[test]
fn disk_selection_matches_a_sorted_history_with_reused_hashes() {
    let db = source();
    let changed_keys: BTreeSet<_> = (0..32)
        .filter(|n| n % 3 == 0)
        .map(|n| format!("{n:02}"))
        .collect();
    for key in &changed_keys {
        changed(&db, key);
    }
    let mut histories = Vec::new();
    for owner in (2..18).rev() {
        let replacements: Vec<_> = (0..32)
            .filter(|n| (n + owner) % 4 != 0)
            .map(|n| {
                replacement(
                    &format!("{n:02}"),
                    ((n + owner) % 7 != 0).then_some(((n * 7 + owner) % 13) as u8),
                )
            })
            .collect();
        history(&db, owner, "a", replacements.clone());
        histories.push((owner, replacements));
    }
    histories.sort_by_key(|h| h.0);
    let mut selected = BTreeMap::new();
    let mut retired = BTreeSet::new();
    for (_, replacements) in histories {
        for replacement in replacements {
            if changed_keys.contains(&replacement.unit.key)
                && let Some(batch) = &replacement.batch
            {
                retired.insert(batch.clone());
            }
            selected.insert(replacement.unit, replacement.batch);
        }
    }
    let now: Vec<_> = changed_keys
        .iter()
        .map(|k| replacement(k, Some(20)))
        .collect();
    for replacement in &now {
        selected.insert(replacement.unit.clone(), replacement.batch.clone());
    }
    for batch in selected.values().flatten() {
        retired.remove(batch);
    }
    assert!(!retired.is_empty());
    assert_eq!(
        batches(&db, &now).unwrap(),
        retired.into_iter().collect::<Vec<_>>()
    );
}

#[cfg(target_os = "linux")]
#[test]
fn unrelated_history_keys_stay_on_disk_during_retirement() {
    fn peak() -> u64 {
        std::fs::read_to_string("/proc/self/status")
            .unwrap()
            .lines()
            .find_map(|line| {
                line.strip_prefix("VmHWM:")
                    .map(|v| v.split_whitespace().next().unwrap().parse().unwrap())
            })
            .unwrap()
    }
    let dir = tempfile::tempdir().unwrap();
    let mut db = Connection::open(dir.path().join("source.db")).unwrap();
    db.execute_batch(
        "PRAGMA cache_size=-1024;
        CREATE TABLE excision(commit_seq INTEGER,id TEXT,refused_json TEXT);
        CREATE TEMP TABLE excision_units(commit_seq INTEGER,kind TEXT,key TEXT,
            PRIMARY KEY(commit_seq,kind,key));",
    )
    .unwrap();
    let suffix = "x".repeat(4096);
    let tx = db.transaction().unwrap();
    for n in (0..6000).rev() {
        history(
            &tx,
            n + 2,
            &format!("erase-{n}"),
            vec![replacement(&format!("{n}:{suffix}"), Some(1))],
        );
    }
    history(&tx, 7000, "target", vec![replacement("changed", Some(2))]);
    changed(&tx, "changed");
    tx.commit().unwrap();
    let before = peak();
    assert_eq!(
        batches(&db, &[replacement("changed", None)]).unwrap(),
        [hash(2)]
    );
    let growth = peak().saturating_sub(before);
    eprintln!("6,000 historical 4KiB keys: retirement high-water growth {growth}KiB");
    assert!(growth < 16 * 1024, "retirement grew by {growth}KiB");
}

#[test]
fn current_batch_and_source_holds_follow_latest_owner_including_tombstones() {
    let db = source();
    let mut old = replacement("changed", Some(1));
    old.blobs = vec![hash(8)];
    let mut new = replacement("changed", Some(2));
    new.blobs = vec![hash(9)];
    history(&db, 4, "later", vec![new]);
    history(&db, 2, "earlier", vec![old]);
    history(&db, 2, "old", vec![replacement("gone", Some(3))]);
    history(&db, 3, "new", vec![replacement("gone", None)]);
    let current = CurrentBatches::new(&db).unwrap();
    for n in [2, 9] {
        assert!(
            current
                .contains(&polylogue_model::BlobHash([n; 32]))
                .unwrap()
        );
    }
    for n in [1, 3, 8] {
        assert!(
            !current
                .contains(&polylogue_model::BlobHash([n; 32]))
                .unwrap()
        );
    }
}

#[test]
fn catalog_and_current_replacement_references_refuse_cleanup_and_keep_intent() {
    for kind in ["catalog", "batch", "source"] {
        let dir = tempfile::tempdir().unwrap();
        let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
        let store = hashed_blobs::Store::open(archive.root.clone(), vec![], 0o600).unwrap();
        let bytes = b"synthetic retained payload";
        let hash = store.put(bytes.as_slice(), 1).unwrap();
        let conn = open_write(&archive.archive_db(), cache::SETUP).unwrap();
        if kind == "catalog" {
            conn.execute("INSERT INTO blob(hash,size,stored_size,place,encoding,state) VALUES (?1,?2,?3,'file','plain','present')", params![&hash.0[..],bytes.len() as i64,store.stored_size(&hash).unwrap() as i64]).unwrap();
        } else {
            let mut publication = ExcisionPublication {
                version: 1,
                replacements: vec![Replacement {
                    unit: Unit {
                        commit_seq: 1,
                        kind: "note".into(),
                        key: "kept".into(),
                    },
                    batch: Some(format!(
                        "blake3:{}",
                        if kind == "batch" {
                            hash.hex()
                        } else {
                            "01".repeat(32)
                        }
                    )),
                    bytes: bytes.len() as u64,
                    rows: 1,
                    blobs: if kind == "source" {
                        vec![format!("blake3:{}", hash.hex())]
                    } else {
                        vec![]
                    },
                }],
                retired_batches: vec![],
                digest: String::new(),
            };
            publication.digest = publication.computed_digest().unwrap();
            conn.execute("INSERT INTO excision(id,scope,refused_json,created_at,commit_seq) VALUES ('held','session:claude-code:removed',?1,1,2)", [serde_json::json!({"publication": publication}).to_string()]).unwrap();
        }
        conn.execute(
            "INSERT INTO pending_blob_excision VALUES (?1,'pending')",
            [&hash.0[..]],
        )
        .unwrap();
        assert!(
            matches!(crate::excise::blob_cleanup::recover(&archive, &conn), Err(StoreError::ExcisionBlobReferenced(h)) if h.0 == hash.0),
            "{kind}"
        );
        assert!(store.contains(&hash), "{kind}");
        assert!(
            crate::excise::blob_cleanup::pending(&conn).unwrap(),
            "{kind}"
        );
    }
}
