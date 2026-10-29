use super::*;
use crate::staging::PreparedBatch;
use crate::writer::{InFlight, InputKey};
use crate::{Archive, Change, ImportMember, NewBlob, Observation, Writer};
use polylogue_model::BlobHash;
use std::sync::Arc;

const ORIGINAL: &[u8] = b"synthetic container byte representation";
const MEMBER: &[u8] = b"synthetic independently stored member";

// Store checks how ZIPs and members are stored and linked; ZIP decoding is tested in Engine.
fn stage(
    archive: &Archive,
    body: &[u8],
    import_id: &str,
    member: &str,
    zip: Option<InContainer>,
) -> PreparedBatch {
    let prepared = archive.prepare_bytes(body).unwrap();
    let batch = Batch {
        input: InputKey {
            id: prepared.hash().to_string(),
            format: "export-file".into(),
            arrival_rank: 2,
        },
        change: Change::First,
        observation: Some(Observation {
            machine: "synthetic-machine".into(),
            host: "synthetic-host".into(),
            path: format!("/synthetic/{member}"),
            facts: None,
            seen_at_ms: 1,
            container: zip,
            import: Some(ImportMember {
                version: 1,
                import_id: import_id.into(),
                member: member.into(),
                format: Some("export-file".into()),
            }),
        }),
        continues: false,
        new_blob: Some(NewBlob {
            hash: prepared.hash(),
            size: prepared.len(),
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

fn linked(archive: &Archive, import_id: &str, duplicates: bool) -> Vec<PreparedBatch> {
    let root = BlobHash::of(ORIGINAL).to_string();
    let mut group = vec![
        stage(archive, ORIGINAL, import_id, "original.zip", None),
        stage(
            archive,
            MEMBER,
            import_id,
            "first.dat",
            Some(InContainer {
                input: root.clone(),
                member_index: 0,
                kind: "zip".into(),
                path_bytes: b"member".to_vec(),
            }),
        ),
    ];
    if duplicates {
        group.push(stage(
            archive,
            MEMBER,
            import_id,
            "second.dat",
            Some(InContainer {
                input: root,
                member_index: 1,
                kind: "zip".into(),
                path_bytes: b"member".to_vec(),
            }),
        ));
    }
    group
}

#[test]
fn linked_duplicate_bytes_keep_every_observation_and_member_index() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let writer = Writer::start(&archive).unwrap();
    let store = Arc::new(archive.blobs().unwrap());
    for import_id in ["one", "two"] {
        let outcomes = writer
            .sender()
            .unwrap()
            .store_group(linked(&archive, import_id, true), Arc::clone(&store))
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
}

#[cfg(feature = "faults")]
mod durability;

#[test]
fn a_container_in_one_import_can_be_a_real_member_in_another() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let writer = Writer::start(&archive).unwrap();
    let store = Arc::new(archive.blobs().unwrap());
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
            Some(InContainer {
                input: outer_id,
                member_index: 0,
                kind: "zip".into(),
                path_bytes: b"member".to_vec(),
            }),
        ),
        stage(
            &archive,
            MEMBER,
            "inner",
            "first.dat",
            Some(InContainer {
                input: inner,
                member_index: 0,
                kind: "zip".into(),
                path_bytes: b"member".to_vec(),
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

#[test]
fn historical_observation_requires_the_exact_stored_revision_point_and_has_no_new_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let writer = Writer::start(&archive).unwrap();
    let blobs = Arc::new(archive.blobs().unwrap());
    let result = writer
        .sender()
        .unwrap()
        .store_group(
            vec![stage(&archive, ORIGINAL, "one", "source", None)],
            Arc::clone(&blobs),
        )
        .unwrap();
    assert!(result[0].is_ok());
    let mut observed = stage(&archive, ORIGINAL, "two", "copy", None);
    observed.batch.change = Change::Observed {
        revision: 1,
        chain: 1,
    };
    observed.batch.new_blob = None;
    observed.files.clear();
    let result = writer
        .sender()
        .unwrap()
        .store_group(vec![observed], Arc::clone(&blobs))
        .unwrap();
    assert!(result[0].as_ref().unwrap().already_stored);
    let mut wrong = stage(&archive, ORIGINAL, "three", "wrong", None);
    wrong.batch.change = Change::Observed {
        revision: 1,
        chain: 2,
    };
    wrong.batch.new_blob = None;
    wrong.files.clear();
    let result = writer
        .sender()
        .unwrap()
        .store_group(vec![wrong], blobs)
        .unwrap();
    assert!(result[0].is_err());
    let db = crate::archive::open_read(&archive.archive_db()).unwrap();
    assert_eq!(
        db.query_row("SELECT count(*) FROM revision", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        1
    );
    assert_eq!(
        db.query_row("SELECT count(*) FROM observation", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        2
    );
}
