use super::*;
use crate::{UserRequest, UserWrite};
use polylogue_model::Author;

fn settings(writer: &Writer, prefix: &str, count: usize) {
    for i in 0..count {
        writer
            .user(UserRequest::now(
                format!("{prefix}-{i}"),
                UserWrite::SetSetting {
                    key: "synthetic-order".into(),
                    value: i.to_string(),
                    author: Author::Operator,
                    machine: None,
                },
            ))
            .unwrap();
    }
}
fn view(archive: &Archive) -> (String, Vec<RestoreUnit>, Vec<Vec<u8>>) {
    let source = Published::open(&archive.root).unwrap();
    let (units, payloads) = super::super::append_facts::descriptors(&source);
    (source.archive_id().unwrap(), units, payloads)
}
fn install(
    target: &Archive,
    id: &str,
    units: &[RestoreUnit],
    payloads: &[Vec<u8>],
) -> crate::Result<u64> {
    Restoring::open(target)?.install(id, &select(units)?, |u| {
        Ok(Some(Cursor::new(
            payloads[units.iter().position(|x| x == u).unwrap()].clone(),
        )))
    })
}

#[test]
fn omitted_local_machine_unit_refuses_before_partial_alias_import_and_later_writes_keep_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let (source, _lock) = Archive::open_or_create(&dir.path().join("source")).unwrap();
    let writer = Writer::start(&source).unwrap();
    observe(&writer, "host-a", 1, Change::First);
    drop(writer);
    let (id, seed, seed_bytes) = view(&source);
    let (local, _lock) = Archive::open_or_create(&dir.path().join("local")).unwrap();
    install(&local, &id, &seed, &seed_bytes).unwrap();
    let writer = Writer::start(&local).unwrap();
    settings(&writer, "local", 3);
    observe(&writer, "host-a", 5, Change::Unchanged);
    drop(writer);
    let before = view(&local);
    assert!(
        before
            .1
            .iter()
            .any(|u| u.unit.kind == "observation" && u.unit.commit_seq == 5)
    );
    let writer = Writer::start(&source).unwrap();
    observe(&writer, "host-b", 2, Change::Unchanged);
    settings(&writer, "remote", 7);
    observe(&writer, "host-b", 10, Change::Unchanged);
    drop(writer);
    let (_, all, all_bytes) = view(&source);
    let (partial, partial_bytes): (Vec<_>, Vec<_>) = all
        .iter()
        .cloned()
        .zip(all_bytes.iter().cloned())
        .filter(|(u, _)| u.unit.kind != "observation" || u.unit.commit_seq != 2)
        .unzip();
    assert!(
        matches!(install(&local,&id,&partial,&partial_bytes),Err(crate::StoreError::Invalid(message)) if message.contains("locally retained observation"))
    );
    assert_eq!(view(&local), before);
    let db = open_write(&local.archive_db(), cache::SETUP).unwrap();
    assert_eq!(
        db.query_row(
            "SELECT count(*) FROM machine WHERE host='host-b'",
            [],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        0
    );

    // A unit written after the later import gets the restored counter floor.
    let (after, _lock) = Archive::open_or_create(&dir.path().join("after")).unwrap();
    install(&after, &id, &partial, &partial_bytes).unwrap();
    let db = open_write(&after.archive_db(), cache::SETUP).unwrap();
    assert_eq!(
        db.query_row(
            "SELECT first_commit_seq FROM machine WHERE host='host-b'",
            [],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        10
    );
    let writer = Writer::start(&after).unwrap();
    observe(&writer, "host-b", 11, Change::Unchanged);
    drop(writer);
    let (_, later, later_bytes) = view(&after);
    let (held, held_bytes) = later
        .into_iter()
        .zip(later_bytes)
        .find(|(u, _)| u.unit.kind == "observation" && u.unit.commit_seq == 11)
        .unwrap();
    let mut complete: Vec<_> = all.into_iter().zip(all_bytes).collect();
    complete.push((held.clone(), held_bytes.clone()));
    complete.sort_by(|(a, _), (b, _)| a.unit.cmp(&b.unit));
    let (complete, complete_bytes): (Vec<_>, Vec<_>) = complete.into_iter().unzip();
    install(&after, &id, &complete, &complete_bytes).unwrap();
    assert_eq!(
        db.query_row(
            "SELECT first_commit_seq FROM machine WHERE host='host-b'",
            [],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        2
    );
    assert_eq!(
        Published::open(&after.root)
            .unwrap()
            .read(&held.unit)
            .unwrap()
            .contents(),
        held_bytes
    );
}

#[test]
fn history_state_binds_all_candidate_alias_fields_and_ignores_other_machines() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let writer = Writer::start(&archive).unwrap();
    observe(&writer, "host-a", 1, Change::First);
    observe(&writer, "host-b", 2, Change::Unchanged);
    observe(&writer, "host-c", 3, Change::Unchanged);
    drop(writer);
    let db = open_write(&archive.archive_db(), cache::SETUP).unwrap();
    db.execute(
        "UPDATE machine SET first_commit_seq=0 WHERE host IN ('host-a','host-b')",
        [],
    )
    .unwrap();
    let state = || {
        Published::open(&archive.root)
            .unwrap()
            .machine_history_state()
            .unwrap()
            .unwrap()
    };
    let mut prior = state();
    db.execute(
        "INSERT INTO machine(id,host,first_seen,first_commit_seq) VALUES ('other','unrelated',4,4)",
        [],
    )
    .unwrap();
    assert_eq!(state(), prior);
    for sql in [
        "UPDATE machine SET first_seen=first_seen+1 WHERE host='host-c'",
        "UPDATE machine SET first_commit_seq=first_commit_seq+1 WHERE host='host-c'",
        "UPDATE machine SET host='renamed-c' WHERE host='host-c'",
        "UPDATE machine SET id='renamed-machine' WHERE id='machine'",
    ] {
        db.execute(sql, []).unwrap();
        let changed = state();
        assert_ne!(changed, prior);
        prior = changed;
    }
}
