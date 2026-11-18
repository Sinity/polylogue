use super::*;
use crate::primary::{AppliedSnapshot, Confirmed, ConfirmedChange, Product, ReplayControl};
use std::path::Path;

fn input(id: &str, position: i64) -> Confirmed {
    Confirmed {
        sinex_context: None,
        position,
        event: format!("event-{id}"),
        output_index: 0,
        change: ConfirmedChange::Live(Box::new(Product::Input {
            rows: Box::new(crate::input_rows::InputRows {
                acquisition: None,
                input: id.into(),
                format: "synthetic".into(),
                revision: None,
                extents: Vec::new(),
                observation: None,
            }),
            material: None,
        })),
    }
}

fn begin(writer: &Writer) {
    writer
        .apply_confirmed(vec![read_control(ReplayControl::SnapshotStart {
            scope: "scope".into(),
            origin: 0,
            stream_instance: "stream".into(),
            reason: None,
        })])
        .unwrap();
}

fn original(root: &Path) -> Material {
    let bytes = b"confirmed original";
    let prepared = hashed_blobs::Preparation::new(root.join("scratch"))
        .unwrap()
        .plain(&bytes[..], 1)
        .unwrap();
    let mut segment = hashed_blobs::SegmentWriter::open(
        root.join("producer"),
        "segments/original.hb",
        0,
        &[],
        0o600,
    )
    .unwrap();
    let group = segment
        .append_group([Ok(prepared)], |_| Ok(None))
        .unwrap()
        .wait()
        .unwrap();
    segment.acknowledge(&group).unwrap();
    Material {
        id: "original".into(),
        hash: BlobHash::of(bytes),
        length: bytes.len() as u64,
        extents: vec![group.extents[0].encode()],
    }
}

fn snapshot(writer: &Writer, position: i64, ids: &[&str], attachment: Option<&Material>) {
    let mut page: Vec<_> = ids.iter().map(|id| input(id, position)).collect();
    if let Some(material) = attachment {
        page.push(Confirmed {
            sinex_context: None,
            position,
            event: "blob-event".into(),
            output_index: 0,
            change: ConfirmedChange::Live(Box::new(Product::Blob(material.clone()))),
        });
    }
    page.push(read_control(ReplayControl::SnapshotComplete {
        snapshot: AppliedSnapshot {
            position,
            scope: "scope".into(),
            stream_instance: "stream".into(),
            last: None,
        },
    }));
    writer.apply_confirmed(page).unwrap();
}

fn latest(writer: &Writer, position: i64) {
    writer
        .apply_confirmed(vec![read_control(ReplayControl::LatestPosition {
            position,
        })])
        .unwrap();
}

fn setup(root: &Path, attachment: Option<&Material>) -> Arc<OfflineProducer> {
    let producer = Arc::new(OfflineProducer {
        root: root.join("producer"),
    });
    std::fs::create_dir_all(&producer.root).unwrap();
    std::fs::write(producer.root.join("ready"), []).unwrap();
    std::fs::write(
        producer.root.join("pending"),
        b"unacknowledged authored bytes",
    )
    .unwrap();
    {
        let (archive, _lock) = Archive::open_primary(&root.join("cache"), &producer.root).unwrap();
        let writer = Writer::start_on_sinex(&archive, producer.clone()).unwrap();
        begin(&writer);
        snapshot(&writer, 4, &["retained"], attachment);
        latest(&writer, 4);
    }
    std::fs::write(root.join("cache/rebuild-index"), []).unwrap();
    producer
}

fn ids(reader: &crate::Reader<'_>) -> Vec<String> {
    let mut q = reader
        .conn()
        .unwrap()
        .prepare("SELECT id FROM source_input ORDER BY id")
        .unwrap();
    q.query_map([], |row| row.get(0))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap()
}

fn serving(archive: &Archive) -> crate::ReadPool {
    crate::ReadPool::serving(archive, 3, 8 << 20).unwrap()
}

/// The private read has an older end mark than the last committed page.
/// Even a checkpoint at this exact boundary cannot copy every WAL frame.
fn assert_pinned_wal(archive: &Archive) {
    let index =
        crate::archive::open_runtime_writer(&archive.index_db(), crate::archive::cache::SETUP)
            .unwrap();
    let (busy, frames, copied): (i64, i64, i64) = index
        .query_row("PRAGMA main.wal_checkpoint(PASSIVE)", [], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })
        .unwrap();
    assert_eq!(busy, 0);
    assert!(
        copied >= 0 && copied < frames,
        "{copied} of {frames} WAL frames copied"
    );
}

#[test]
fn rebuild_swaps_only_when_caught_up() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    let attachment = original(root);
    let producer = setup(root, Some(&attachment));
    let cache = root.join("cache");
    let (archive, lock) = Archive::open_primary(&cache, &producer.root).unwrap();
    let reads = serving(&archive);
    assert!(matches!(
        reads.reader(),
        Err(crate::StoreError::IndexBuilding(_))
    ));
    let writer = Writer::start_on_sinex(&archive, producer.clone()).unwrap();
    begin(&writer);
    writer.synchronize_primary().unwrap();
    let old = reads.reader().unwrap();
    assert_eq!(ids(&old), ["retained"]);
    assert_eq!(old.confirmed_checkpoint().unwrap(), Some(4));
    let hash = hashed_blobs::Hash(attachment.hash.0);
    let bytes = |reader: &crate::Reader<'_>| {
        reader
            .blob_reader()
            .unwrap()
            .read_range(&hash, 0, attachment.length)
            .unwrap()
    };
    // The candidate has no attachment catalog row yet.
    assert_eq!(bytes(&old), b"confirmed original");
    snapshot(&writer, 5, &["retained", "new"], Some(&attachment));
    let pipeline = crate::ReadPool::open(&archive).unwrap();
    let private = pipeline.reader().unwrap();
    assert_eq!(private.confirmed_checkpoint().unwrap(), Some(5));
    latest(&writer, 7);
    assert!(writer.flush().unwrap().errors.is_empty());
    assert_eq!(
        reads.reader().unwrap().confirmed_checkpoint().unwrap(),
        Some(4)
    );
    assert!(!cache.join("index.db").exists());

    writer
        .apply_confirmed(vec![read_control(ReplayControl::Completion {
            position: 7,
            delivery: None,
        })])
        .unwrap();
    // A delivery can follow the sampled head before its closing marker arrives.
    writer.apply_confirmed(vec![input("tail", 8)]).unwrap();
    assert!(writer.flush().unwrap().errors.is_empty());
    assert_eq!(ids(&reads.reader().unwrap()), ["retained"]);
    latest(&writer, 8);
    writer
        .apply_confirmed(vec![read_control(ReplayControl::Completion {
            position: 8,
            delivery: None,
        })])
        .unwrap();
    assert_pinned_wal(&archive);
    assert!(writer.flush().unwrap().errors.is_empty());
    assert_eq!(private.confirmed_checkpoint().unwrap(), Some(5));
    assert!(
        std::fs::symlink_metadata(cache.join("index.db"))
            .unwrap()
            .is_symlink()
    );
    let current = reads.reader().unwrap();
    assert_eq!(ids(&current), ["new", "retained", "tail"]);
    assert_eq!(bytes(&current), b"confirmed original");
    assert_eq!(bytes(&old), b"confirmed original");
    assert_eq!(current.confirmed_checkpoint().unwrap(), Some(8));
    assert_eq!(ids(&old), ["retained"]);
    assert_eq!(old.confirmed_checkpoint().unwrap(), Some(4));
    drop(current);
    drop(old);
    drop(private);
    drop(pipeline);
    // The follower still observes the writer's physical file after publication.
    latest(&writer, 9);
    writer
        .apply_confirmed(vec![read_control(ReplayControl::Completion {
            position: 9,
            delivery: None,
        })])
        .unwrap();
    assert_eq!(writer.synchronize_primary().unwrap(), 9);
    drop(writer);
    drop(reads);
    drop(lock);
    drop(archive);
    let (archive, _lock) = Archive::open_primary(&cache, &producer.root).unwrap();
    assert!(
        !std::fs::symlink_metadata(cache.join("index.db"))
            .unwrap()
            .is_symlink()
    );
    assert!(!cache.join("index-build").exists());
    let reads = serving(&archive);
    assert_eq!(ids(&reads.reader().unwrap()), ["new", "retained", "tail"]);
    assert_eq!(
        reads.reader().unwrap().confirmed_checkpoint().unwrap(),
        Some(9)
    );
    assert_eq!(
        std::fs::read(producer.root.join("pending")).unwrap(),
        b"unacknowledged authored bytes"
    );
}

#[test]
fn narrowing_withdraws_the_previous_primary_index_across_restart() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    let producer = setup(root, None);
    let cache = root.join("cache");
    {
        let (archive, _lock) = Archive::open_primary(&cache, &producer.root).unwrap();
        let reads = serving(&archive);
        let writer = Writer::start_on_sinex(&archive, producer.clone()).unwrap();
        begin(&writer);
        writer.synchronize_primary().unwrap();
        let old = reads.reader().unwrap();
        assert_eq!(ids(&old), ["retained"]);
        writer
            .apply_confirmed(vec![read_control(ReplayControl::Reset {
                scope: "narrowed".into(),
            })])
            .unwrap();
        assert!(matches!(old.check(), Err(crate::StoreError::ScopeChanged)));
        assert!(matches!(
            reads.reader(),
            Err(crate::StoreError::IndexBuilding(_))
        ));
    }
    let (archive, _lock) = Archive::open_primary(&cache, &producer.root).unwrap();
    let reads = serving(&archive);
    let writer = Writer::start_on_sinex(&archive, producer).unwrap();
    // A later successful response for the original scope cannot revive the
    // old index after its withdrawal. Only the rebuilt rows can be published.
    begin(&writer);
    writer.synchronize_primary().unwrap();
    assert!(matches!(
        reads.reader(),
        Err(crate::StoreError::IndexBuilding(_))
    ));
}

#[cfg(feature = "faults")]
#[test]
fn primary_rebuild_crash_child() {
    let Some(root) = std::env::var_os("POLYLOGUE_PRIVATE_BUILD_ROOT") else {
        return;
    };
    let point = std::env::var("POLYLOGUE_PRIVATE_BUILD_POINT").unwrap();
    let root = Path::new(&root);
    let producer = setup(root, None);
    let cache = root.join("cache");
    crate::fault::arm(&point, "", crate::fault::Action::StopProcess, 0, 1);
    {
        let (archive, _lock) = Archive::open_primary(&cache, &producer.root).unwrap();
        let writer = Writer::start_on_sinex(&archive, producer.clone()).unwrap();
        begin(&writer);
        let pipeline = crate::ReadPool::open(&archive).unwrap();
        let private = pipeline.reader().unwrap();
        assert_eq!(private.confirmed_checkpoint().unwrap(), None);
        snapshot(&writer, 7, &["retained", "new"], None);
        latest(&writer, 7);
        writer.synchronize_primary().unwrap();
        assert_pinned_wal(&archive);
        assert!(writer.flush().unwrap().errors.is_empty());
        assert_eq!(private.confirmed_checkpoint().unwrap(), None);
    }
    let _fold = Archive::open_primary(&cache, &producer.root).unwrap();
    panic!("did not reach {point}");
}

#[cfg(feature = "faults")]
#[test]
fn primary_rebuild_recovers_on_each_side_of_publication() {
    for point in [
        "primary.build.after_set_aside",
        "primary.build.after_sync",
        "primary.build.after_install",
        "primary.build.after_fold",
    ] {
        let root = crate::writer::tests::private_build::crash_test_at(
            "writer::primary_capture::tests::rebuild::primary_rebuild_crash_child",
            point,
        );
        let cache = root.path().join("cache");
        let producer = Arc::new(OfflineProducer {
            root: root.path().join("producer"),
        });
        let (archive, _lock) = Archive::open_primary(&cache, &producer.root).unwrap();
        let reads = serving(&archive);
        if point == "primary.build.after_sync" || point == "primary.build.after_set_aside" {
            assert!(matches!(
                reads.reader(),
                Err(crate::StoreError::IndexBuilding(_))
            ));
            let writer = Writer::start_on_sinex(&archive, producer.clone()).unwrap();
            if point == "primary.build.after_set_aside" {
                begin(&writer);
            }
            writer.synchronize_primary().unwrap();
            assert_eq!(ids(&reads.reader().unwrap()), ["retained"]);
            if point == "primary.build.after_set_aside" {
                snapshot(&writer, 7, &["retained", "new"], None);
                latest(&writer, 7);
            }
            assert!(writer.flush().unwrap().errors.is_empty(), "{point}");
        }
        let current = reads.reader().unwrap();
        assert_eq!(
            ids(&current),
            ["new", "retained"],
            "{point} served position {:?}. {}",
            current.confirmed_checkpoint().unwrap(),
            recovery_state(&archive),
        );
        assert_eq!(
            reads.reader().unwrap().confirmed_checkpoint().unwrap(),
            Some(7),
            "{point}"
        );
        assert_eq!(
            std::fs::read(producer.root.join("pending")).unwrap(),
            b"unacknowledged authored bytes"
        );
    }
}

#[cfg(feature = "faults")]
fn recovery_state(archive: &Archive) -> String {
    let index = crate::archive::open_read(&archive.index_db()).unwrap();
    let state = crate::primary::confirmed::replay_state(&index).unwrap();
    let parsed = crate::archive::index_rebuild::built(&index).unwrap();
    let derived = crate::derive::caught_up(&index).unwrap();
    let (pending, complete, search): (i64, i64, i64) = index
        .query_row(
            "SELECT (SELECT count(*) FROM primary_pending),
                (SELECT count(*) FROM primary_complete),
                (SELECT count(*) FROM search_pending)",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .unwrap();
    format!(
        "Candidate {state:?}, parser ready {parsed}, derivations ready {derived}, pending deliveries {pending}, completions {complete}, pending search {search}, build {:?}",
        archive.index_build()
    )
}
