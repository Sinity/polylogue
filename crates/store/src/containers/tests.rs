use super::*;
use crate::blobs::Stored;
use crate::staging::PreparedBatch;
use crate::writer::{InFlight, InputKey};
use crate::{Archive, Change, ImportMember, Observation, Piece, Writer};
use polylogue_model::BlobHash;
use std::sync::{Arc, Barrier};

const ORIGINAL: &[u8] = b"synthetic container byte representation";
const MEMBER: &[u8] = b"synthetic independently retained member";

// Store checks custody and canonical relationships; actual ZIP decoding is tested in Engine.
fn stage(
    archive: &Archive,
    body: &[u8],
    acquisition: &str,
    member: &str,
    container: Option<ImportContainer>,
) -> PreparedBatch {
    let prepared = archive.prepare_bytes(&[], body).unwrap();
    let batch = Batch {
        input: InputKey {
            id: prepared.hash().to_string(),
            format: "export-file".into(),
            route_rank: 2,
        },
        change: Change::First,
        observation: Some(Observation {
            machine: "synthetic-machine".into(),
            host: "synthetic-host".into(),
            path: format!("/synthetic/{member}"),
            facts: None,
            seen_at_ms: 1,
            import: Some(ImportMember {
                version: 1,
                acquisition: acquisition.into(),
                member: member.into(),
                format: Some("export-file".into()),
                container,
            }),
        }),
        continues: false,
        piece: Some(Piece {
            hash: prepared.hash(),
            size: prepared.len(),
            stored: Stored::File { stored_size: 0 },
        }),
        blobs: vec![],
        length: prepared.len(),
        parse: vec![],
        release: InFlight::default(),
    };
    PreparedBatch {
        batch,
        files: vec![prepared],
    }
}

fn linked(archive: &Archive, acquisition: &str, duplicates: bool) -> Vec<PreparedBatch> {
    let root = BlobHash::of(ORIGINAL).to_string();
    let mut group = vec![
        stage(archive, ORIGINAL, acquisition, "original.zip", None),
        stage(
            archive,
            MEMBER,
            acquisition,
            "first.dat",
            Some(ImportContainer {
                input: root.clone(),
                member_index: 0,
            }),
        ),
    ];
    if duplicates {
        group.push(stage(
            archive,
            MEMBER,
            acquisition,
            "second.dat",
            Some(ImportContainer {
                input: root,
                member_index: 1,
            }),
        ));
    }
    group
}

fn remove(
    writer: &Writer,
    store: Arc<hashed_blobs::Store>,
    id: &str,
) -> Result<crate::excise::Excised> {
    writer.excise(
        crate::excise::ExciseRequest {
            id: id.into(),
            scope: crate::excise::Scope::Input(BlobHash::of(MEMBER).to_string()),
            reason: None,
            dry_run: false,
        },
        store,
    )
}

#[test]
fn linked_duplicate_bytes_keep_every_observation_and_member_index() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let writer = Writer::start(&archive).unwrap();
    let store = Arc::new(hashed_blobs::Store::open(dir.path(), vec![], 0o600).unwrap());
    for acquisition in ["one", "two"] {
        let outcomes = writer
            .sender()
            .unwrap()
            .store_group(linked(&archive, acquisition, true), Arc::clone(&store))
            .unwrap();
        assert!(outcomes.iter().all(Result::is_ok), "{outcomes:?}");
    }
    let db = crate::archive::open_read(&archive.archive_db()).unwrap();
    let count = |table: &str| {
        db.query_row(&format!("SELECT count(*) FROM {table}"), [], |r| {
            r.get::<_, i64>(0)
        })
        .unwrap()
    };
    assert_eq!(
        (
            count("input"),
            count("revision"),
            count("extent"),
            count("blob")
        ),
        (2, 2, 2, 2)
    );
    assert_eq!(
        (count("observation"), count("observation_container")),
        (6, 4)
    );
    assert_eq!(
        db.query_row(
            "SELECT count(DISTINCT member_index) FROM observation_container",
            [],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        2
    );
    let removed = remove(&writer, store, "linked-member").unwrap();
    assert_eq!(removed.zips_removed, [BlobHash::of(ORIGINAL).to_string()]);
    assert_eq!(count("excision"), 1);
    assert_eq!(count("extent"), 0);
    assert_eq!(count("observation_container"), 0);
}

#[test]
fn prepared_linked_admission_and_member_exclusion_serialize_without_hidden_copies() {
    for concurrent in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
        let writer = Arc::new(Writer::start(&archive).unwrap());
        let store = Arc::new(hashed_blobs::Store::open(dir.path(), vec![], 0o600).unwrap());
        let initial = stage(&archive, MEMBER, "initial", "initial.dat", None);
        writer
            .sender()
            .unwrap()
            .store_group(vec![initial], Arc::clone(&store))
            .unwrap()
            .pop()
            .unwrap()
            .unwrap();
        let group = linked(&archive, "linked", false);
        if !concurrent {
            remove(&writer, Arc::clone(&store), "queued-remove").unwrap();
            let outcomes = writer
                .sender()
                .unwrap()
                .store_group(group, Arc::clone(&store))
                .unwrap();
            assert!(outcomes.iter().all(Result::is_err), "{outcomes:?}");
            assert!(!store.contains(&hashed_blobs::Hash(BlobHash::of(ORIGINAL).0)));
            continue;
        }
        let barrier = Arc::new(Barrier::new(3));
        let admit = {
            let sender = writer.sender().unwrap();
            let store = Arc::clone(&store);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                sender.store_group(group, store).unwrap()
            })
        };
        let exclusion = {
            let writer = Arc::clone(&writer);
            let store = Arc::clone(&store);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                remove(&writer, store, "concurrent-remove")
            })
        };
        barrier.wait();
        let outcomes = admit.join().unwrap();
        exclusion.join().unwrap().unwrap();
        assert!(
            outcomes.iter().all(Result::is_ok) || outcomes.iter().all(Result::is_err),
            "{outcomes:?}"
        );
        assert!(!store.contains(&hashed_blobs::Hash(BlobHash::of(ORIGINAL).0)));
        assert!(!store.contains(&hashed_blobs::Hash(BlobHash::of(MEMBER).0)));
    }
}

#[cfg(feature = "faults")]
mod unfinished;

#[test]
fn a_container_in_one_acquisition_can_be_a_real_member_in_another() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let writer = Writer::start(&archive).unwrap();
    let store = Arc::new(hashed_blobs::Store::open(dir.path(), vec![], 0o600).unwrap());
    let inner = BlobHash::of(ORIGINAL).to_string();
    let outer = b"synthetic outer container representation";
    let outer_id = BlobHash::of(outer).to_string();
    let group = vec![
        stage(&archive, outer, "outer", "outer.zip", None),
        stage(&archive, ORIGINAL, "inner", "inner.zip", None),
        stage(
            &archive,
            ORIGINAL,
            "outer",
            "file-Native.dat",
            Some(ImportContainer {
                input: outer_id,
                member_index: 0,
            }),
        ),
        stage(
            &archive,
            MEMBER,
            "inner",
            "first.dat",
            Some(ImportContainer {
                input: inner,
                member_index: 0,
            }),
        ),
    ];
    let results = writer.sender().unwrap().store_group(group, store).unwrap();
    assert!(results.iter().all(Result::is_ok), "{results:?}");
    let pool = crate::ReadPool::open(&archive).unwrap();
    let (members, _) = pool.reader().unwrap().unindexed_import_members().unwrap();
    let mut names: Vec<_> = members
        .into_iter()
        .map(|member| member.membership.member)
        .collect();
    names.sort();
    assert_eq!(names, ["file-Native.dat", "first.dat"]);
}

mod excision;
