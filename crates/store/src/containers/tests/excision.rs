use super::*;
use polylogue_model::BlobHash;
use crate::archive::{cache, open_read, open_write};
use std::collections::BTreeSet;

fn id(body: &[u8]) -> String {
    BlobHash::of(body).to_string()
}
fn member(
    archive: &Archive,
    body: &[u8],
    acquisition: &str,
    original: &[u8],
    ordinal: u64,
) -> PreparedBatch {
    stage(
        archive,
        body,
        acquisition,
        "member.dat",
        Some(ImportContainer {
            input: id(original),
            member_index: ordinal,
        }),
    )
}
fn admit(writer: &Writer, store: &Arc<hashed_blobs::Store>, group: Vec<PreparedBatch>) {
    let results = writer
        .sender()
        .unwrap()
        .store_group(group, Arc::clone(store))
        .unwrap();
    assert!(results.iter().all(Result::is_ok), "{results:?}");
}

#[test]
fn shared_selected_input_retires_each_containing_acquisition_and_keeps_unrelated_units() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let writer = Writer::start(&archive).unwrap();
    let store = Arc::new(hashed_blobs::Store::open(&archive.root, vec![], 0o600).unwrap());
    let sibling = b"surviving shared member";
    let other = b"second original with selected member";
    let unrelated = b"unrelated original without selected member";
    let mut group = vec![];
    for (acquisition, original, selected) in [
        ("one", ORIGINAL, true),
        ("two", other.as_slice(), true),
        ("unrelated", unrelated.as_slice(), false),
    ] {
        group.push(stage(&archive, original, acquisition, "original.zip", None));
        if selected {
            group.push(member(&archive, MEMBER, acquisition, original, 0));
        }
        group.push(member(&archive, sibling, acquisition, original, 1));
    }
    admit(&writer, &store, group);
    writer.user(crate::user_writes::UserRequest::now("unrelated-note".into(), crate::user_writes::UserWrite::AddNote { note: serde_json::from_value(serde_json::json!({"kind":"note","value":"unrelated authored note","author":"operator","targets":["repo:synthetic"]})).unwrap() })).unwrap();
    let removed = remove(&writer, Arc::clone(&store), "shared-member").unwrap();
    assert_eq!(
        removed
            .containers_retired
            .into_iter()
            .collect::<BTreeSet<_>>(),
        [id(ORIGINAL), id(other)].into_iter().collect()
    );
    assert_eq!(
        removed.inputs_removed.into_iter().collect::<BTreeSet<_>>(),
        [id(ORIGINAL), id(other), id(MEMBER)].into_iter().collect()
    );
    for body in [ORIGINAL, other.as_slice(), MEMBER] {
        assert!(!store.contains(&hashed_blobs::Hash(BlobHash::of(body).0)));
    }
    for body in [sibling.as_slice(), unrelated.as_slice()] {
        assert!(store.contains(&hashed_blobs::Hash(BlobHash::of(body).0)));
    }
    let db = open_read(&archive.archive_db()).unwrap();
    assert_eq!(
        db.query_row("SELECT count(*) FROM observation_container", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        1
    );
    assert_eq!(
        db.query_row(
            "SELECT container_input FROM observation_container",
            [],
            |r| r.get::<_, String>(0)
        )
        .unwrap(),
        id(unrelated)
    );
}

#[test]
fn nested_container_closure_is_iterative_beyond_sixty_four_levels() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let writer = Writer::start(&archive).unwrap();
    let store = Arc::new(hashed_blobs::Store::open(&archive.root, vec![], 0o600).unwrap());
    let bodies: Vec<_> = (0..70)
        .map(|i| format!("synthetic containing representation {i}").into_bytes())
        .collect();
    let mut group = vec![];
    for (i, original) in bodies.iter().enumerate() {
        let acquisition = format!("nested-{i}");
        group.push(stage(
            &archive,
            original,
            &acquisition,
            "original.zip",
            None,
        ));
        group.push(member(
            &archive,
            bodies.get(i + 1).map(Vec::as_slice).unwrap_or(MEMBER),
            &acquisition,
            original,
            0,
        ));
    }
    admit(&writer, &store, group);
    let removed = remove(&writer, store, "nested-member").unwrap();
    assert_eq!(removed.containers_retired.len(), 70);
    assert_eq!(removed.inputs_removed.len(), 71);
    assert_eq!(
        open_read(&archive.archive_db())
            .unwrap()
            .query_row("SELECT count(*) FROM extent", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        0
    );
}

#[test]
fn malformed_or_missing_container_basis_refuses_before_mutation() {
    for cycle in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
        let writer = Writer::start(&archive).unwrap();
        let store = Arc::new(hashed_blobs::Store::open(&archive.root, vec![], 0o600).unwrap());
        admit(&writer, &store, linked(&archive, "bad", false));
        let db = open_write(&archive.archive_db(), cache::SETUP).unwrap();
        if cycle {
            // Forge a two-node cycle with otherwise valid immutable source facts.
            db.execute(
                "INSERT INTO observation_container SELECT id,?1,1 FROM observation WHERE input=?2",
                rusqlite::params![id(MEMBER), id(ORIGINAL)],
            )
            .unwrap();
        } else {
            db.execute_batch("DELETE FROM extent WHERE input IN (SELECT container_input FROM observation_container)").unwrap();
        }
        let before: i64 = db
            .query_row("SELECT count(*) FROM extent", [], |r| r.get(0))
            .unwrap();
        assert!(remove(&writer, Arc::clone(&store), "malformed").is_err());
        assert_eq!(
            db.query_row("SELECT count(*) FROM excision", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            0
        );
        assert_eq!(
            db.query_row("SELECT count(*) FROM extent", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            before
        );
        assert!(store.contains(&hashed_blobs::Hash(BlobHash::of(MEMBER).0)));
    }
}

#[test]
#[cfg(feature = "faults")]
fn committed_container_retirement_replays_pending_private_cleanup_after_restart() {
    crate::fault::clear();
    let dir = tempfile::tempdir().unwrap();
    let (archive, lock) = Archive::open_or_create(dir.path()).unwrap();
    let writer = Writer::start(&archive).unwrap();
    let store = Arc::new(hashed_blobs::Store::open(&archive.root, vec![], 0o600).unwrap());
    admit(&writer, &store, linked(&archive, "interrupted", false));
    crate::fault::arm(
        "excision.before_blob_delete",
        &id(ORIGINAL),
        crate::fault::Action::Fail,
        0,
        1,
    );
    assert!(remove(&writer, Arc::clone(&store), "interrupted-retirement").is_err());
    crate::fault::clear();
    drop(writer);
    drop(lock);
    drop(archive);
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    assert!(!store.contains(&hashed_blobs::Hash(BlobHash::of(ORIGINAL).0)));
    assert!(!store.contains(&hashed_blobs::Hash(BlobHash::of(MEMBER).0)));
    let db = open_read(&archive.archive_db()).unwrap();
    assert_eq!(
        db.query_row("SELECT count(*) FROM observation_container", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        0
    );
    assert_eq!(
        db.query_row("SELECT count(*) FROM blob WHERE state = 'deleting'", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        0
    );
}

#[test]
fn original_bytes_shared_with_unlinked_source_refuse_before_removal() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let writer = Writer::start(&archive).unwrap();
    let store = Arc::new(hashed_blobs::Store::open(&archive.root, vec![], 0o600).unwrap());
    admit(&writer, &store, linked(&archive, "shared-original", false));
    let mut watched = stage(&archive, ORIGINAL, "watched", "watched.zip", None);
    watched.batch.input.id = "independent-watched-input".into();
    watched.batch.input.format = "synthetic-text".into();
    watched.batch.input.route_rank = 0;
    watched.batch.change = Change::First;
    watched.batch.observation.as_mut().unwrap().import = None;
    admit(&writer, &store, vec![watched]);
    let db = open_read(&archive.archive_db()).unwrap();
    let before: i64 = db
        .query_row("SELECT count(*) FROM extent", [], |r| r.get(0))
        .unwrap();
    assert!(matches!(
        remove(&writer, Arc::clone(&store), "shared-original-refusal"),
        Err(StoreError::ExcisionFormat { .. })
    ));
    assert_eq!(
        db.query_row("SELECT count(*) FROM excision", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        0
    );
    assert_eq!(
        db.query_row("SELECT count(*) FROM extent", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        before
    );
    assert!(store.contains(&hashed_blobs::Hash(BlobHash::of(ORIGINAL).0)));
}

#[test]
fn original_bytes_referenced_by_attachment_refuse_before_archive_or_blob_mutation() {
    use crate::fixture::writer::{WriterWorld, message};
    use polylogue_model::{Attachment, AttachmentDirection, Block, Role};
    let mut world = WriterWorld::new();
    let store = Arc::new(hashed_blobs::Store::open(&world.archive.root, vec![], 0o600).unwrap());
    admit(
        &world.writer,
        &store,
        linked(&world.archive, "attachment-owner", false),
    );
    world.write(
        "unrelated",
        vec![message(
            "file",
            "unrelated",
            Role::User,
            1,
            vec![Block::File(Attachment {
                name: Some("original.zip".into()),
                media_type: Some("application/zip".into()),
                size: Some(ORIGINAL.len() as u64),
                blob: Some(BlobHash::of(ORIGINAL)),
                url: None,
                source_id: None,
                direction: AttachmentDirection::Input,
            })],
        )],
    );
    let db = open_read(&world.archive.archive_db()).unwrap();
    let before: String = db
        .query_row(
            "SELECT value FROM meta WHERE key='write_counter'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert!(matches!(
        remove(&world.writer, Arc::clone(&store), "attachment-refusal"),
        Err(StoreError::ExcisionFormat { .. })
    ));
    assert_eq!(
        db.query_row(
            "SELECT value FROM meta WHERE key='write_counter'",
            [],
            |r| r.get::<_, String>(0)
        )
        .unwrap(),
        before
    );
    assert_eq!(
        db.query_row("SELECT count(*) FROM excision", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        0
    );
    assert_eq!(
        db.query_row("SELECT count(*) FROM observation_container", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        1
    );
    let mut bytes = store
        .read_plain(&hashed_blobs::Hash(BlobHash::of(ORIGINAL).0), None, || {
            Ok(())
        })
        .unwrap();
    let mut actual = vec![];
    std::io::Read::read_to_end(&mut bytes, &mut actual).unwrap();
    assert_eq!(actual, ORIGINAL);
}

#[test]
fn dry_run_reports_originals_without_changing_linked_bytes_or_policy() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let writer = Writer::start(&archive).unwrap();
    let store = Arc::new(hashed_blobs::Store::open(&archive.root, vec![], 0o600).unwrap());
    admit(&writer, &store, linked(&archive, "dry-run", false));
    let result = writer
        .excise(
            crate::excise::ExciseRequest {
                id: "dry-run".into(),
                scope: crate::excise::Scope::Input(id(MEMBER)),
                reason: None,
                dry_run: true,
            },
            Arc::clone(&store),
        )
        .unwrap();
    assert!(result.dry_run);
    assert_eq!(result.containers_retired, [id(ORIGINAL)]);
    assert_eq!(
        result.inputs_removed.into_iter().collect::<BTreeSet<_>>(),
        [id(ORIGINAL), id(MEMBER)].into_iter().collect()
    );
    let db = open_read(&archive.archive_db()).unwrap();
    assert_eq!(
        db.query_row("SELECT count(*) FROM excision", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        0
    );
    assert_eq!(
        db.query_row("SELECT count(*) FROM observation_container", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        1
    );
    for body in [ORIGINAL, MEMBER] {
        assert!(store.contains(&hashed_blobs::Hash(BlobHash::of(body).0)));
    }
}
