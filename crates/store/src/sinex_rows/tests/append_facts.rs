use super::*;
use crate::archive::{Archive, cache, open_write};
use crate::writer::{Batch as WriteBatch, InFlight, InputKey, Observation};
use crate::{Change, Writer};
use std::io::Cursor;

pub(super) fn descriptors(source: &Published) -> (Vec<RestoreUnit>, Vec<Vec<u8>>) {
    let inventory = source.inventory().unwrap();
    let mut units = Vec::new();
    let mut payloads = Vec::new();
    for position in 1..=inventory.len() {
        let unit = inventory.unit(position).unwrap();
        let batch = source.read(&unit).unwrap();
        let bytes = batch.contents();
        units.push(RestoreUnit {
            unit,
            batch: Some(format!("blake3:{}", blake3::hash(&bytes))),
            bytes: Some(bytes.len() as u64),
            rows: batch.rows,
            blobs: vec![],
            excision: None,
        });
        payloads.push(bytes);
    }
    (units, payloads)
}

#[test]
fn late_and_deleted_observations_have_units_and_restore_without_growing_the_revision() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let writer = Writer::start(&archive).unwrap();
    let mut original = None;
    for (at, change) in [
        (1, Change::First),
        (2, Change::Unchanged),
        (3, Change::Disappeared),
    ] {
        writer
            .send(WriteBatch {
                input: InputKey {
                    id: "input:/opaque".into(),
                    format: "synthetic-text".into(),
                    route_rank: 0,
                },
                change,
                observation: Some(Observation {
                    machine: "machine".into(),
                    host: "host".into(),
                    path: "synthetic".into(),
                    facts: None,
                    seen_at_ms: at,
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
        let source = Published::open(&archive.root).unwrap();
        let inventory = source.inventory().unwrap();
        let revision = (1..=inventory.len())
            .map(|n| inventory.unit(n).unwrap())
            .find(|u| u.kind == "revision")
            .unwrap();
        let bytes = source.read(&revision).unwrap().contents();
        if let Some(original) = &original {
            assert_eq!(&bytes, original);
        } else {
            original = Some(bytes);
        }
        assert_eq!(inventory.len(), at as u64 + 1);
    }
    drop(writer);
    let source = Published::open(&archive.root).unwrap();
    let (units, payloads) = descriptors(&source);
    assert_eq!(
        units
            .iter()
            .filter(|u| u.unit.kind == "observation")
            .count(),
        3
    );
    let selected = select(&units).unwrap();
    let target_dir = tempfile::tempdir().unwrap();
    let (target, _lock) = Archive::open_or_create(target_dir.path()).unwrap();
    let mut target = Restoring::open(&target).unwrap();
    let id = source.archive_id().unwrap();
    let install = |target: &mut Restoring| {
        target.install(&id, &selected, |u| {
            Ok(Some(Cursor::new(
                payloads[units.iter().position(|x| x == u).unwrap()].clone(),
            )))
        })
    };
    for omit_machine in [false, true] {
        let index = units
            .iter()
            .position(|u| u.unit.kind == "observation")
            .unwrap();
        let mut forged_units = units.clone();
        let mut forged_payloads = payloads.clone();
        let mut bytes = Vec::new();
        let mut count = 0;
        for line in payloads[index]
            .split(|b| *b == b'\n')
            .filter(|b| !b.is_empty())
        {
            let mut value: Value = serde_json::from_slice(line).unwrap();
            if omit_machine && value["table"] == "machine" {
                continue;
            }
            if !omit_machine && value["table"] == "observation" {
                value["row"]["revision"] = json!(99);
            }
            serde_json::to_writer(&mut bytes, &value).unwrap();
            bytes.push(b'\n');
            count += 1;
        }
        forged_units[index].batch = Some(format!("blake3:{}", blake3::hash(&bytes)));
        forged_units[index].bytes = Some(bytes.len() as u64);
        forged_units[index].rows = count;
        forged_payloads[index] = bytes;
        let forged = select(&forged_units).unwrap();
        assert!(
            target
                .install(&id, &forged, |u| Ok(Some(Cursor::new(
                    forged_payloads[forged_units.iter().position(|x| x == u).unwrap()].clone()
                ))))
                .is_err()
        );
    }
    assert!(install(&mut target).unwrap() > 0);
    assert_eq!(install(&mut target).unwrap(), 0);
    let db = open_write(&target.archive.archive_db(), cache::SETUP).unwrap();
    assert_eq!(
        db.query_row(
            "SELECT count(*) FROM observation WHERE class='deleted' AND revision IS NULL",
            [],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        1
    );
}

#[test]
fn legacy_parent_closure_is_verified_but_selected_fact_owns_the_restored_row() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let db = open_write(&archive.archive_db(), cache::SETUP).unwrap();
    db.execute_batch("INSERT INTO delivery(id,recipient_kind,sender,event,body,content_hash,refs_json,left_out_json,at,bytes,commit_seq) VALUES ('opaque:[id]','operator','cli','briefing',x'626f6479',zeroblob(32),'[]','[]',1,4,1);
        INSERT INTO delivery_event(delivery,what,at,detail,commit_seq) VALUES ('opaque:[id]','failed',2,'exact detail',0);").unwrap();
    let parent = Unit {
        commit_seq: 1,
        kind: "delivery".into(),
        key: "opaque:[id]".into(),
    };
    let legacy = read(&db, &parent).unwrap();
    let legacy_bytes = legacy.contents();
    // Old immutable batches have no sequence column on the embedded event.
    let mut legacy_rows: Vec<Value> = legacy_bytes
        .split(|b| *b == b'\n')
        .filter(|b| !b.is_empty())
        .map(|b| serde_json::from_slice(b).unwrap())
        .collect();
    legacy_rows
        .iter_mut()
        .filter(|v| v["table"] == "delivery_event")
        .for_each(|v| {
            v["row"].as_object_mut().unwrap().remove("commit_seq");
        });
    let mut bytes = Vec::new();
    for row in legacy_rows {
        serde_json::to_writer(&mut bytes, &row).unwrap();
        bytes.push(b'\n');
    }
    db.execute("UPDATE delivery_event SET commit_seq=2", [])
        .unwrap();
    let source = Published::open(&archive.root).unwrap();
    let (mut units, mut payloads) = descriptors(&source);
    units[0].rows = legacy.rows;
    units[0].batch = Some(format!("blake3:{}", blake3::hash(&bytes)));
    units[0].bytes = Some(bytes.len() as u64);
    payloads[0] = bytes;
    let selected = select(&units).unwrap();
    let target_dir = tempfile::tempdir().unwrap();
    let (target, _lock) = Archive::open_or_create(target_dir.path()).unwrap();
    let mut target = Restoring::open(&target).unwrap();
    target
        .install(&source.archive_id().unwrap(), &selected, |u| {
            Ok(Some(Cursor::new(
                payloads[units.iter().position(|x| x == u).unwrap()].clone(),
            )))
        })
        .unwrap();
    let db = open_write(&target.archive.archive_db(), cache::SETUP).unwrap();
    assert_eq!(
        db.query_row(
            "SELECT commit_seq,what,at,detail FROM delivery_event",
            [],
            |r| Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, i64>(2)?,
                r.get::<_, String>(3)?
            ))
        )
        .unwrap(),
        (2, "failed".into(), 2, "exact detail".into())
    );
}

#[test]
fn excision_redacts_independent_events_and_publishes_excised_observations_before_replay() {
    use crate::excise::{ExciseRequest, Scope};
    use crate::user_writes::{UserRequest, UserWrite};
    use crate::{Delivery, DeliveryOutcome};
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let writer = Writer::start(&archive).unwrap();
    writer
        .user(UserRequest::now(
            "delivery:[opaque]".into(),
            UserWrite::Deliver(Delivery {
                recipient_kind: "operator".into(),
                recipient_id: None,
                sender: "cli".into(),
                event: "advisory".into(),
                cwd: None,
                text: "Synthetic body".into(),
                refs: vec![],
                left_out: vec![],
            }),
        ))
        .unwrap();
    writer
        .user(UserRequest::now(
            "event".into(),
            UserWrite::DeliveryEvent {
                delivery: "delivery:[opaque]".into(),
                what: DeliveryOutcome::Failed,
                detail: Some("synthetic secret evidence".into()),
            },
        ))
        .unwrap();
    writer
        .send(WriteBatch {
            input: InputKey {
                id: "input".into(),
                format: "synthetic-text".into(),
                route_rank: 0,
            },
            change: Change::First,
            observation: Some(Observation {
                machine: "machine".into(),
                host: "host".into(),
                path: "synthetic".into(),
                facts: None,
                seen_at_ms: 1,
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
    let source = Published::open(&archive.root).unwrap();
    let (originals, original_payloads) = descriptors(&source);
    drop(source);
    let store = std::sync::Arc::new(
        hashed_blobs::Store::open(archive.root.clone(), vec![], 0o600).unwrap(),
    );
    writer
        .excise(
            ExciseRequest {
                id: "redact".into(),
                scope: Scope::Text("secret".into()),
                reason: None,
                dry_run: false,
            },
            store.clone(),
        )
        .unwrap();
    writer
        .excise(
            ExciseRequest {
                id: "remove".into(),
                scope: Scope::Input("input".into()),
                reason: None,
                dry_run: false,
            },
            store,
        )
        .unwrap();
    drop(writer);
    let source = Published::open(&archive.root).unwrap();
    let inventory = source.inventory().unwrap();
    let mut confirmations = originals.clone();
    for position in originals.len() as u64 + 1..=inventory.len() {
        let unit = inventory.unit(position).unwrap();
        let batch = source.read(&unit).unwrap();
        let bytes = batch.contents();
        confirmations.push(RestoreUnit {
            excision: source.publication(&unit).unwrap(),
            unit,
            batch: Some(format!("blake3:{}", blake3::hash(&bytes))),
            bytes: Some(bytes.len() as u64),
            rows: batch.rows,
            blobs: vec![],
        });
    }
    let selected = select(&confirmations).unwrap();
    let current = selected_units(&selected);
    assert!(
        current
            .iter()
            .any(|u| u.unit.kind == "revision" && u.batch.is_none()),
        "removed revision needs its own tombstone"
    );
    assert!(current.iter().any(|u| {
        u.unit.kind == "delivery_event"
            && u.batch
                != originals
                    .iter()
                    .find(|o| o.unit.kind == "delivery_event")
                    .unwrap()
                    .batch
    }));
    let target_dir = tempfile::tempdir().unwrap();
    let (target, _lock) = Archive::open_or_create(target_dir.path()).unwrap();
    let mut target = Restoring::open(&target).unwrap();
    let id = source.archive_id().unwrap();
    // Install the original view first, exercising removal as well as fresh replay.
    let old = select(&originals).unwrap();
    target
        .install(&id, &old, |u| {
            Ok(Some(Cursor::new(
                original_payloads[originals.iter().position(|x| x == u).unwrap()].clone(),
            )))
        })
        .unwrap();
    for _ in 0..2 {
        target
            .install(&id, &selected, |u| {
                Ok(u.batch.as_ref().map(|name| {
                    let is_replacement = confirmations.iter().any(|c| {
                        c.excision
                            .as_ref()
                            .is_some_and(|x| x.replacements.iter().any(|r| r.unit == u.unit))
                    });
                    if is_replacement {
                        Cursor::new(file_bytes(source.replacement(name).unwrap()))
                    } else {
                        Cursor::new(source.read(&u.unit).unwrap().contents())
                    }
                }))
            })
            .unwrap();
    }
    let db = open_write(&target.archive.archive_db(), cache::SETUP).unwrap();
    let detail: String = db
        .query_row("SELECT detail FROM delivery_event", [], |r| r.get(0))
        .unwrap();
    assert_eq!(detail, "synthetic        evidence");
    assert_eq!(
        db.query_row(
            "SELECT count(*) FROM observation WHERE class='excised'",
            [],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        1
    );
}

#[test]
fn migration_and_restore_keep_observation_insertion_order_when_ids_and_times_disagree() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let mut db = open_write(&archive.archive_db(), cache::SETUP).unwrap();
    db.execute_batch("ALTER TABLE machine DROP COLUMN first_commit_seq; DROP TABLE extracted_blob; DROP TABLE excision_document; DROP TRIGGER vector_policy_insert; DROP TRIGGER vector_policy_update; DROP TRIGGER vector_policy_delete; DROP TABLE vector_deletion_policy; DELETE FROM meta WHERE key='vector_policy_revision'; DROP INDEX observation_commit_seq; DROP INDEX delivery_event_commit_seq;
        ALTER TABLE observation DROP COLUMN commit_seq; ALTER TABLE delivery_event DROP COLUMN commit_seq;
        PRAGMA user_version=4;
        INSERT INTO input VALUES ('input','synthetic-text');
        INSERT INTO machine(id,host,first_seen) VALUES ('machine','host',1);
        INSERT INTO observation(id,input,machine,path,class,polylogue_version,seen_at) VALUES ('z','input','machine','first','deleted','test',900);
        INSERT INTO observation(id,input,machine,path,class,polylogue_version,seen_at) VALUES ('a','input','machine','last','deleted','test',1);").unwrap();
    let latest = |db: &Connection| {
        db.query_row(
            "SELECT path FROM observation WHERE input='input' ORDER BY rowid DESC LIMIT 1",
            [],
            |r| r.get::<_, String>(0),
        )
        .unwrap()
    };
    assert_eq!(latest(&db), "last");
    crate::schema::migrate_archive(&mut db).unwrap();
    let source = Published::open(&archive.root).unwrap();
    let (units, payloads) = descriptors(&source);
    assert_eq!(
        units
            .iter()
            .map(|u| u.unit.key.as_str())
            .collect::<Vec<_>>(),
        ["z", "a"]
    );
    let selected = select(&units).unwrap();
    let target_dir = tempfile::tempdir().unwrap();
    let (target, _lock) = Archive::open_or_create(target_dir.path()).unwrap();
    let mut target = Restoring::open(&target).unwrap();
    target
        .install(&source.archive_id().unwrap(), &selected, |u| {
            Ok(Some(Cursor::new(
                payloads[units.iter().position(|x| x == u).unwrap()].clone(),
            )))
        })
        .unwrap();
    let restored = open_write(&target.archive.archive_db(), cache::SETUP).unwrap();
    assert_eq!(latest(&restored), latest(&db));
}

#[test]
fn more_than_ten_excised_locations_restore_the_latest_inserted_observation() {
    use crate::excise::{ExciseRequest, Scope};
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let writer = Writer::start(&archive).unwrap();
    for n in 0..12 {
        writer
            .send(WriteBatch {
                input: InputKey {
                    id: "input".into(),
                    format: "synthetic-text".into(),
                    route_rank: 0,
                },
                change: if n == 0 {
                    Change::First
                } else {
                    Change::Unchanged
                },
                observation: Some(Observation {
                    machine: "machine".into(),
                    host: "host".into(),
                    path: format!("location-{n}"),
                    facts: None,
                    seen_at_ms: 12 - n,
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
    let source = Published::open(&archive.root).unwrap();
    let (mut confirmations, _) = descriptors(&source);
    drop(source);
    writer
        .excise(
            ExciseRequest {
                id: "remove".into(),
                scope: Scope::Input("input".into()),
                reason: None,
                dry_run: false,
            },
            std::sync::Arc::new(
                hashed_blobs::Store::open(archive.root.clone(), vec![], 0o600).unwrap(),
            ),
        )
        .unwrap();
    drop(writer);
    let source = Published::open(&archive.root).unwrap();
    let inventory = source.inventory().unwrap();
    for position in confirmations.len() as u64 + 1..=inventory.len() {
        let unit = inventory.unit(position).unwrap();
        let batch = source.read(&unit).unwrap();
        let bytes = batch.contents();
        confirmations.push(RestoreUnit {
            excision: source.publication(&unit).unwrap(),
            unit,
            batch: Some(format!("blake3:{}", blake3::hash(&bytes))),
            bytes: Some(bytes.len() as u64),
            rows: batch.rows,
            blobs: vec![],
        });
    }
    let selected = select(&confirmations).unwrap();
    let target_dir = tempfile::tempdir().unwrap();
    let (target, _lock) = Archive::open_or_create(target_dir.path()).unwrap();
    let mut target = Restoring::open(&target).unwrap();
    target
        .install(&source.archive_id().unwrap(), &selected, |u| {
            Ok(u.batch
                .as_ref()
                .map(|_| Cursor::new(source.read(&u.unit).unwrap().contents())))
        })
        .unwrap();
    let source_db = open_write(&archive.archive_db(), cache::SETUP).unwrap();
    let restored = open_write(&target.archive.archive_db(), cache::SETUP).unwrap();
    let rows = |db: &Connection| {
        db.prepare("SELECT id,path,class FROM observation ORDER BY rowid")
            .unwrap()
            .query_map([], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                ))
            })
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap()
    };
    assert_eq!(
        rows(&source_db)
            .iter()
            .filter(|(_, _, class)| class == "excised")
            .count(),
        12
    );
    assert_eq!(rows(&restored), rows(&source_db));
}

#[test]
fn captured_fact_deletions_restore_as_independent_tombstones() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let mut db = open_write(&archive.archive_db(), cache::SETUP).unwrap();
    db.execute_batch("INSERT INTO input VALUES ('input','synthetic-text');
        INSERT INTO machine(id,host,first_seen) VALUES ('machine','host',1);
        INSERT INTO observation(id,input,machine,path,class,polylogue_version,seen_at,commit_seq) VALUES ('observation','input','machine','synthetic','deleted','test',1,1);
        INSERT INTO delivery(id,recipient_kind,sender,event,body,content_hash,refs_json,left_out_json,at,bytes,commit_seq) VALUES ('opaque:[id]','operator','cli','briefing',x'626f6479',zeroblob(32),'[]','[]',1,4,1);
        INSERT INTO delivery_event(delivery,what,at,detail,commit_seq) VALUES ('opaque:[id]','failed',2,'exact detail',2);").unwrap();
    let source = Published::open(&archive.root).unwrap();
    let (mut confirmations, old_bytes) = descriptors(&source);
    drop(source);
    let tx = db.transaction().unwrap();
    let capture = Capture::start(&tx, "remove").unwrap();
    tx.execute("INSERT INTO excision(id,scope,created_at,commit_seq) VALUES ('remove','text:synthetic',3,3)",[]).unwrap();
    tx.execute_batch("DELETE FROM delivery_event; DELETE FROM observation;")
        .unwrap();
    let publication = capture
        .finish(
            &tx,
            &hashed_blobs::Store::open(archive.root.clone(), vec![], 0o600).unwrap(),
        )
        .unwrap();
    assert_eq!(
        publication
            .replacements
            .iter()
            .filter(|r| r.batch.is_none())
            .map(|r| r.unit.kind.as_str())
            .collect::<Vec<_>>(),
        ["observation", "delivery_event"]
    );
    tx.execute(
        "UPDATE excision SET refused_json=?1",
        [serde_json::json!({"publication":publication}).to_string()],
    )
    .unwrap();
    tx.commit().unwrap();
    let source = Published::open(&archive.root).unwrap();
    let unit = source.inventory().unwrap().unit(4).unwrap();
    let batch = source.read(&unit).unwrap();
    let bytes = batch.contents();
    confirmations.push(RestoreUnit {
        excision: source.publication(&unit).unwrap(),
        unit,
        batch: Some(format!("blake3:{}", blake3::hash(&bytes))),
        bytes: Some(bytes.len() as u64),
        rows: batch.rows,
        blobs: vec![],
    });
    let selected = select(&confirmations).unwrap();
    let target_dir = tempfile::tempdir().unwrap();
    let (target, _lock) = Archive::open_or_create(target_dir.path()).unwrap();
    let mut target = Restoring::open(&target).unwrap();
    let old = select(&confirmations[..3]).unwrap();
    let id = source.archive_id().unwrap();
    target
        .install(&id, &old, |u| {
            Ok(Some(Cursor::new(
                old_bytes[confirmations.iter().position(|x| x == u).unwrap()].clone(),
            )))
        })
        .unwrap();
    target
        .install(&id, &selected, |u| {
            Ok(u.batch
                .as_ref()
                .map(|_| Cursor::new(source.read(&u.unit).unwrap().contents())))
        })
        .unwrap();
    let db = open_write(&target.archive.archive_db(), cache::SETUP).unwrap();
    for table in ["observation", "delivery_event"] {
        assert_eq!(
            db.query_row(&format!("SELECT count(*) FROM {table}"), [], |r| r
                .get::<_, i64>(0))
                .unwrap(),
            0
        );
    }
}
