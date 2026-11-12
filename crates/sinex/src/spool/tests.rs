use super::*;
use std::process::Command;

trait CaptureFixture {
    fn prepare_capture(&self, frame: Ingress, bytes: &[u8], capture: &Capture) -> Result<Prepared>;
}
impl CaptureFixture for Spool {
    fn prepare_capture(&self, frame: Ingress, bytes: &[u8], capture: &Capture) -> Result<Prepared> {
        self.prepare_encoded_capture(frame, vec![crate::tests::body(&self.root, bytes)], capture)
    }
}

pub(super) fn boundary(name: &str) {
    if std::env::var("POLYLOGUE_SPOOL_CRASH").as_deref() == Ok(name) {
        std::process::exit(77);
    }
}

fn events(identity: &str, value: &str) -> Ingress {
    serde_json::from_value(serde_json::json!({
        "kind":"events", "frame":identity,
        "events":[{"kind":{"name":"fixture.value","version":1},"payload":{"text":value},"reading_order":0,"class":"open"}],
        "failures":[]
    })).unwrap()
}

#[test]
fn producer_backup_cut_keeps_original_frames_after_live_confirmed_removal() {
    let directory = tempfile::tempdir().unwrap();
    let spool = Spool::open(directory.path(), "host").unwrap();
    let first = spool.prepare(events("before-core-cut", "first")).unwrap();
    let acknowledgment: Acknowledgment = serde_json::from_value(serde_json::json!({
        "number":1,"frame":first.identity,"material":null,"admitted":[],
        "content_hash":null,"result":"admitted","counts":crate::tests::zero_counts(),"commit_seq":10
    }))
    .unwrap();
    spool.acknowledge(acknowledgment.clone()).unwrap();
    let pending = spool
        .prepare(events("after-core-cut", "retained original"))
        .unwrap();
    let encoded = fs::read(
        spool
            .operation(&pending.identity)
            .unwrap()
            .join("frame.zst"),
    )
    .unwrap();
    let pinned = spool
        .pin_backup(&AtomicBool::new(false))
        .unwrap()
        .finish(&AtomicBool::new(false))
        .unwrap();

    // Ordinary capture and confirmed removal continue after pinning. Neither can
    // change the bounded producer artifact or its original retry identities.
    spool
        .prepare(events("after-producer-cut", "later"))
        .unwrap();
    spool
        .remove_confirmed(
            &crate::tests::applied(10),
            &crate::tests::every_event_applied,
            "fixture-producer",
            &|_, _| Ok(false),
        )
        .unwrap();
    assert!(spool.get(&first.identity).unwrap().is_none());
    let restored = Spool::open(pinned.path(), "host").unwrap();
    assert_eq!(
        restored.acknowledgment(&first.identity).unwrap(),
        Some(acknowledgment)
    );
    assert!(restored.get("after-producer-cut").unwrap().is_none());
    let retry = restored.get(&pending.identity).unwrap().unwrap();
    assert!(retry.phase.is_pending());
    assert_eq!(retry.identity, pending.identity);
    assert_eq!(
        fs::read(
            restored
                .operation(&retry.identity)
                .unwrap()
                .join("frame.zst")
        )
        .unwrap(),
        encoded
    );
}

#[test]
fn producer_backup_keeps_group_material_encoding_after_confirmed_removal() {
    let directory = tempfile::tempdir().unwrap();
    let spool = Spool::open(directory.path(), "host").unwrap();
    let bytes = b"original provider bytes";
    spool
        .prepare_material(material("raw", bytes), bytes.as_slice())
        .unwrap();
    spool.acknowledge(ack("raw", true, None)).unwrap();
    spool
        .prepare_group(events("group", "facts"), &["raw".into()])
        .unwrap();
    spool.acknowledge(ack("group", false, Some(10))).unwrap();
    let encoded = fs::read(spool.operation("raw").unwrap().join("body.zst")).unwrap();
    let pinned = spool
        .pin_backup(&AtomicBool::new(false))
        .unwrap()
        .finish(&AtomicBool::new(false))
        .unwrap();
    spool
        .remove_confirmed(
            &crate::tests::applied(10),
            &crate::tests::every_event_applied,
            "fixture-producer",
            &|_, _| Ok(false),
        )
        .unwrap();
    assert!(spool.get("raw").unwrap().is_none());
    assert!(spool.get("group").unwrap().is_none());
    let restored = Spool::open(pinned.path(), "host").unwrap();
    assert_eq!(
        fs::read(restored.operation("raw").unwrap().join("body.zst")).unwrap(),
        encoded
    );
    let mut decoded = Vec::new();
    restored
        .body("raw")
        .unwrap()
        .read_to_end(&mut decoded)
        .unwrap();
    assert_eq!(decoded, bytes);
    assert_eq!(
        restored.capture_group("raw").unwrap().unwrap().identity,
        "group"
    );
}

#[test]
fn producer_backup_cancelled_before_pin_keeps_live_queue() {
    let directory = tempfile::tempdir().unwrap();
    let spool = Spool::open(directory.path(), "host").unwrap();
    spool.prepare(events("original", "retained")).unwrap();
    assert!(spool.pin_backup(&AtomicBool::new(true)).is_err());
    assert!(spool.get("original").unwrap().is_some());
    let parent = spool.root.parent().unwrap();
    assert!(!fs::read_dir(parent).unwrap().any(|entry| {
        entry
            .unwrap()
            .file_name()
            .to_str()
            .is_some_and(|name| name.starts_with(".backup-cut-"))
    }));
}

#[test]
fn named_prefix_lookup_reads_only_matching_operations_and_survives_restart() {
    let directory = tempfile::tempdir().unwrap();
    let spool = Spool::open(directory.path(), "host").unwrap();
    let one = format!("query:0123456789ab{}", "1".repeat(52));
    let two = format!("query:0123456789ab{}", "2".repeat(52));
    for name in [&one, &two, "bulk:unrelated"] {
        spool.prepare(events(name, "frozen object")).unwrap();
    }
    let unrelated = spool.operation("bulk:unrelated").unwrap().join("frame.zst");
    let original = fs::read(&unrelated).unwrap();
    fs::write(&unrelated, b"not a queue record").unwrap();
    let mut found = Vec::new();
    spool
        .visit_names("query:0123456789ab", &mut |identity| {
            found.push(identity.to_owned());
            Ok(())
        })
        .unwrap();
    found.sort();
    assert_eq!(found, [one.clone(), two.clone()]);
    assert!(spool.decode_frame("bulk:unrelated").is_err());
    fs::write(unrelated, original).unwrap();
    drop(spool);
    let spool = Spool::open(directory.path(), "host").unwrap();
    let mut restored = Vec::new();
    spool
        .visit_names("query:0123456789ab", &mut |identity| {
            restored.push(identity.to_owned());
            Ok(())
        })
        .unwrap();
    restored.sort();
    assert_eq!(restored, found);
}

#[test]
fn named_lookup_accepts_unbounded_logical_names() {
    let directory = tempfile::tempdir().unwrap();
    let spool = Spool::open(directory.path(), "host").unwrap();
    let prefix = format!("query:{}", "x".repeat(8192));
    let first = format!("{prefix}:one");
    let second = format!("{prefix}:two");
    spool.prepare(events(&first, "first")).unwrap();
    spool.prepare(events(&second, "second")).unwrap();
    let mut found = Vec::new();
    spool
        .visit_names(&first, &mut |identity| {
            found.push(identity.to_owned());
            Ok(())
        })
        .unwrap();
    assert_eq!(found, [first]);
}

#[test]
fn a_late_acknowledgment_removes_at_the_already_applied_barrier() {
    let directory = tempfile::tempdir().unwrap();
    let spool = Spool::open(directory.path(), "host").unwrap();
    spool.prepare(events("late", "value")).unwrap();
    spool
        .remove_confirmed(
            &crate::tests::applied(10),
            &crate::tests::every_event_applied,
            "fixture-producer",
            &|_, _| Ok(false),
        )
        .unwrap();
    assert!(spool.get("late").unwrap().is_some());
    spool.acknowledge(ack("late", false, Some(8))).unwrap();
    spool
        .remove_confirmed(
            &crate::tests::applied(10),
            &crate::tests::every_event_applied,
            "fixture-producer",
            &|_, _| Ok(false),
        )
        .unwrap();
    assert!(spool.get("late").unwrap().is_none());
}

fn material(identity: &str, bytes: &[u8]) -> Ingress {
    material_with(
        identity,
        format!("blake3:{}", blake3::hash(bytes).to_hex()),
        bytes.len() as u64,
    )
}

fn material_with(identity: &str, blob: String, length: u64) -> Ingress {
    Ingress::Material {
        frame: identity.into(),
        material: Box::new(sinex_protocol::MaterialFrame {
            source: crate::source_reference(),
            id: crate::material_id("polylogue-test", identity),
            input: sinex_model::InputRef {
                kind: "fixture".into(),
                locator: identity.into(),
                host: "fixture-host".into(),
            },
            blob,
            length,
            chunks: Vec::new(),
            parser: "fixture".into(),
            parser_version: 1,
            part_index: 0,
            previous_part: None,
            manifest: sinex_model::Manifest::default(),
        }),
    }
}

fn counts(acknowledgment: &mut Acknowledgment) -> &mut sinex_model::AdmissionCounts {
    match &mut acknowledgment.result {
        FrameResult::Admitted { counts, .. } => counts,
        other => panic!("not an admission: {other:?}"),
    }
}

fn ack(identity: &str, material: bool, position: Option<i64>) -> Acknowledgment {
    serde_json::from_value(serde_json::json!({
        "number":1, "frame":identity, "material":material.then(||crate::material_id("polylogue-test", identity)),"material_extents":[],
        "admitted":[],"content_hash":null,"result":"admitted","counts":crate::tests::zero_counts(),"commit_seq":position
    })).unwrap()
}

fn all(spool: &Spool, size: usize) -> Vec<Prepared> {
    let mut entries = Vec::new();
    let mut after = None;
    loop {
        let page = spool.pending_after(after, size).unwrap();
        if page.is_empty() {
            break;
        }
        after = page.last().map(|entry| entry.order);
        entries.extend(page);
    }
    entries
}

#[test]
fn retries_keep_exact_bytes_and_later_equal_values_are_separate_operations() {
    let directory = tempfile::tempdir().unwrap();
    let spool = Spool::open(directory.path(), "host-a").unwrap();
    let first = events("write-a", "A");
    spool.prepare(first.clone()).unwrap();
    spool.prepare(events("write-b", "B")).unwrap();
    spool.prepare(events("write-c", "A")).unwrap();
    assert!(spool.prepare(events("write-a", "changed")).is_err());
    drop(spool);
    let spool = Spool::open(directory.path(), "host-a").unwrap();
    assert_eq!(
        all(&spool, 1)
            .iter()
            .map(|entry| entry.identity.as_str())
            .collect::<Vec<_>>(),
        ["write-a", "write-b", "write-c"]
    );
    assert_eq!(
        serde_json::to_value(spool.decode_frame("write-a").unwrap()).unwrap(),
        serde_json::to_value(first).unwrap()
    );
    assert_eq!(spool.prepare(events("write-a", "A")).unwrap().order, 1);
    assert_eq!(spool.counts().unwrap().1, 3);
    assert!(all(&Spool::open(directory.path(), "host-b").unwrap(), 1).is_empty());
}

#[test]
fn acknowledgment_is_not_the_confirmed_application_barrier() {
    let directory = tempfile::tempdir().unwrap();
    let spool = Spool::open(directory.path(), "host-a").unwrap();
    assert!(spool.is_known_empty());
    spool.prepare(events("write", "original")).unwrap();
    assert!(!spool.is_known_empty());
    spool.acknowledge(ack("write", false, Some(18))).unwrap();
    drop(spool);
    let spool = Spool::open(directory.path(), "host-a").unwrap();
    assert!(
        !spool.is_known_empty(),
        "restart retains unpublished/unapplied ownership"
    );
    let page = spool.pending_after(None, 1).unwrap();
    assert!(page[0].phase.is_acknowledged());
    spool
        .remove_confirmed(
            &crate::tests::applied(17),
            &crate::tests::every_event_applied,
            "fixture-producer",
            &|_, _| Ok(false),
        )
        .unwrap();
    assert!(spool.get("write").unwrap().is_some());
    assert!(!spool.is_known_empty());
    spool
        .remove_confirmed(
            &crate::tests::applied(18),
            &crate::tests::every_event_applied,
            "fixture-producer",
            &|_, _| Ok(false),
        )
        .unwrap();
    assert!(spool.get("write").unwrap().is_none());
    assert!(spool.is_known_empty());
}

#[test]
fn one_group_owns_every_capture_until_its_confirmed_position_is_applied() {
    let directory = tempfile::tempdir().unwrap();
    let spool = Spool::open(directory.path(), "host-a").unwrap();
    for identity in ["zip", "member"] {
        spool
            .prepare_material(material(identity, b"bytes"), b"bytes".as_slice())
            .unwrap();
        spool.acknowledge(ack(identity, true, None)).unwrap();
    }
    let members = vec!["member".into(), "zip".into()];
    spool
        .prepare_group(events("canonical", "complete"), &members)
        .unwrap();
    assert!(spool.group_for(&["zip".into()]).unwrap().is_none());
    assert_eq!(
        spool.group_for(&members).unwrap().unwrap().identity,
        "canonical"
    );
    assert!(
        spool
            .prepare_group(events("conflicting", "other"), &["zip".into()])
            .is_err()
    );
    assert!(
        spool
            .prepare_group(events("canonical", "changed"), &members)
            .is_err()
    );
    assert_eq!(
        spool.capture_group("member").unwrap().unwrap().identity,
        "canonical"
    );
    spool
        .remove_confirmed(
            &crate::tests::applied(999),
            &crate::tests::every_event_applied,
            "fixture-producer",
            &|_, _| Ok(false),
        )
        .unwrap();
    assert!(spool.get("zip").unwrap().is_some());
    spool
        .acknowledge(ack("canonical", false, Some(21)))
        .unwrap();
    spool
        .remove_confirmed(
            &crate::tests::applied(20),
            &crate::tests::every_event_applied,
            "fixture-producer",
            &|_, _| Ok(false),
        )
        .unwrap();
    assert_eq!(spool.counts().unwrap().0, 2);
    spool
        .remove_confirmed(
            &crate::tests::applied(21),
            &crate::tests::every_event_applied,
            "fixture-producer",
            &|_, _| Ok(false),
        )
        .unwrap();
    assert!(all(&spool, 2).is_empty());
    assert!(spool.get("member").unwrap().is_none());
}

#[test]
fn malformed_acknowledgments_do_not_advance_validated_queue_position() {
    let directory = tempfile::tempdir().unwrap();
    let spool = Spool::open(directory.path(), "host").unwrap();
    spool
        .prepare_material(material("raw", b"body"), b"body".as_slice())
        .unwrap();
    assert!(spool.acknowledge(ack("raw", false, None)).is_err());
    spool
        .prepare_group(events("group", "canonical"), &["raw".into()])
        .unwrap();
    assert!(spool.acknowledge(ack("group", false, None)).is_err());
    assert!(spool.acknowledge(ack("group", false, Some(-1))).is_err());
    spool.acknowledge(ack("group", false, Some(12))).unwrap();
    let acknowledgment = spool
        .operation("group")
        .unwrap()
        .join("ack/acknowledgment.json");
    let original = fs::read(&acknowledgment).unwrap();
    let mut damaged: serde_json::Value = serde_json::from_slice(&original).unwrap();
    damaged["value"]["phase"]["commit_seq"] = 0.into();
    fs::write(&acknowledgment, serde_json::to_vec(&damaged).unwrap()).unwrap();
    assert!(spool.acknowledgment("group").is_err());
    spool
        .remove_confirmed(
            &crate::tests::applied(1),
            &crate::tests::every_event_applied,
            "fixture-producer",
            &|_, _| Ok(false),
        )
        .unwrap();
    assert!(spool.get("raw").unwrap().is_some());
    assert_eq!(
        spool.get("group").unwrap().unwrap().phase,
        Phase::Acknowledged {
            commit_seq: Some(12)
        }
    );
    fs::write(&acknowledgment, original).unwrap();
    assert!(spool.acknowledgment("group").unwrap().is_some());
    spool
        .remove_confirmed(
            &crate::tests::applied(12),
            &crate::tests::every_event_applied,
            "fixture-producer",
            &|_, _| Ok(false),
        )
        .unwrap();
    assert!(spool.get("raw").unwrap().is_none());
}

#[test]
fn material_bodies_are_verified_including_empty_files_and_fifo_rejection() {
    let directory = tempfile::tempdir().unwrap();
    let spool = Spool::open(directory.path(), "host").unwrap();
    assert!(spool.prepare(material("empty", b"")).is_err());
    spool
        .prepare_material(material("empty", b""), std::io::empty())
        .unwrap();
    assert_eq!(
        spool
            .body("empty")
            .unwrap()
            .read_to_end(&mut Vec::new())
            .unwrap() as u64,
        0
    );
    assert!(
        spool
            .prepare_material(material("wrong", b"right"), b"wrong".as_slice())
            .is_err()
    );
    assert!(spool.get("wrong").unwrap().is_none());
    let path = spool.operation("empty").unwrap().join("body.zst");
    fs::remove_file(&path).unwrap();
    let status = Command::new("mkfifo").arg(&path).status().unwrap();
    assert!(status.success());
    assert!(spool.body("empty").is_err());
}

#[test]
fn pending_metadata_stays_small_with_large_immutable_frames_and_streamed_bodies() {
    let directory = tempfile::tempdir().unwrap();
    let spool = Spool::open(directory.path(), "host").unwrap();
    spool
        .prepare(events("large-frame", &"x".repeat(4 << 20)))
        .unwrap();
    let length = 24 << 20;
    let mut digest = blake3::Hasher::new();
    let block = vec![7u8; 64 << 10];
    for _ in 0..length / block.len() {
        digest.update(&block);
    }
    let blob = format!("blake3:{}", digest.finalize().to_hex());
    spool
        .prepare_material(
            material_with("large-body", blob, length as u64),
            std::io::repeat(7).take(length as u64),
        )
        .unwrap();
    let page = spool.pending_after(None, 2).unwrap();
    let records: Vec<_> = page.iter().map(|entry| &entry.record).collect();
    assert!(serde_json::to_vec(&records).unwrap().len() < 4096);
    assert!(
        spool
            .queue
            .get("large-frame")
            .unwrap()
            .unwrap()
            .metadata
            .len()
            < 4096
    );
    let physical = spool.counts().unwrap().2;
    let (body_path, descriptor) = spool.encoded_body("large-body").unwrap().remove(0);
    assert_eq!(body_path.metadata().unwrap().len(), descriptor.stored_len);
    assert!(physical > descriptor.stored_len && physical < (28 << 20));
    assert_eq!(
        spool
            .body("large-body")
            .unwrap()
            .read_to_end(&mut Vec::new())
            .unwrap() as u64,
        length as u64
    );
}

#[test]
fn crash_worker() {
    let Ok(root) = std::env::var("POLYLOGUE_SPOOL_CRASH_ROOT") else {
        return;
    };
    let spool = Spool::open(Path::new(&root), "host").unwrap();
    let stage = std::env::var("POLYLOGUE_SPOOL_CRASH").unwrap();
    if stage.starts_with("acquisition_") {
        spool
            .prepare_acquisition("acquisition", "two-raws", 2, |staging| {
                for id in ["first", "second"] {
                    staging.prepare_material(
                        material(id, b"source bytes"),
                        b"source bytes".as_slice(),
                    )?;
                }
                Ok(())
            })
            .unwrap();
    } else if stage == "capture_synced" {
        let bytes = b"original source";
        let capture = parsed_capture(bytes, "synthetic answer");
        spool
            .prepare_capture(
                capture_frame("captured", bytes, &capture),
                bytes.as_slice(),
                &capture,
            )
            .unwrap();
    } else if stage.starts_with("group_") {
        for identity in ["zip", "member"] {
            spool
                .prepare_material(material(identity, b"bytes"), b"bytes".as_slice())
                .unwrap();
        }
        spool
            .prepare_group(
                events("canonical", "all members"),
                &["zip".into(), "member".into()],
            )
            .unwrap();
        spool.acknowledge(ack("canonical", false, Some(9))).unwrap();
        spool
            .remove_confirmed(
                &crate::tests::applied(9),
                &crate::tests::every_event_applied,
                "fixture-producer",
                &|_, _| Ok(false),
            )
            .unwrap();
    } else if stage == "ack_synced"
        || stage == "ack_published"
        || stage == "ack_files_published"
        || stage == "operation_removed"
    {
        spool.prepare(events("write", "original")).unwrap();
        spool.acknowledge(ack("write", false, Some(9))).unwrap();
        spool
            .remove_confirmed(
                &crate::tests::applied(9),
                &crate::tests::every_event_applied,
                "fixture-producer",
                &|_, _| Ok(false),
            )
            .unwrap();
    } else {
        spool
            .prepare_material(material("raw", b"source bytes"), b"source bytes".as_slice())
            .unwrap();
    }
    panic!("fault boundary did not terminate the process");
}

#[test]
fn process_death_publishes_all_acquisition_members_together() {
    for stage in ["acquisition_synced", "acquisition_published"] {
        let directory = tempfile::tempdir().unwrap();
        let status = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "spool::tests::crash_worker", "--nocapture"])
            .env("POLYLOGUE_SPOOL_CRASH_ROOT", directory.path())
            .env("POLYLOGUE_SPOOL_CRASH", stage)
            .status()
            .unwrap();
        assert_eq!(status.code(), Some(77));
        let spool = Spool::open(directory.path(), "host").unwrap();
        let entries = all(&spool, 1);
        assert_eq!(
            entries.len(),
            if stage == "acquisition_published" {
                2
            } else {
                0
            }
        );
        if !entries.is_empty() {
            assert_eq!(
                entries
                    .iter()
                    .map(|entry| entry.identity.as_str())
                    .collect::<Vec<_>>(),
                ["first", "second"]
            );
            let mut raw = Vec::new();
            spool.body("second").unwrap().read_to_end(&mut raw).unwrap();
            assert_eq!(raw, b"source bytes");
        }
    }
}

#[test]
fn process_death_at_durability_boundaries_exposes_only_complete_operations() {
    for stage in [
        "body_synced",
        "operation_synced",
        "operation_published",
        "ack_synced",
        "ack_published",
        "operation_removed",
    ] {
        let directory = tempfile::tempdir().unwrap();
        let status = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "spool::tests::crash_worker", "--nocapture"])
            .env("POLYLOGUE_SPOOL_CRASH_ROOT", directory.path())
            .env("POLYLOGUE_SPOOL_CRASH", stage)
            .status()
            .unwrap();
        assert_eq!(status.code(), Some(77), "{stage}");
        let spool = Spool::open(directory.path(), "host").unwrap();
        match stage {
            "body_synced" | "operation_synced" => assert!(all(&spool, 1).is_empty()),
            "operation_published" => {
                let prepared = spool.get("raw").unwrap().unwrap();
                assert!(prepared.body().is_some());
                assert_eq!(
                    spool
                        .body("raw")
                        .unwrap()
                        .read_to_end(&mut Vec::new())
                        .unwrap() as u64,
                    12
                );
            }
            "ack_synced" => {
                assert!(spool.get("write").unwrap().unwrap().phase.is_pending());
                assert!(spool.acknowledgment("write").unwrap().is_none());
                spool.acknowledge(ack("write", false, Some(9))).unwrap();
                assert!(spool.get("write").unwrap().unwrap().phase.is_acknowledged());
            }
            "ack_published" => {
                assert!(spool.get("write").unwrap().unwrap().phase.is_acknowledged());
                spool
                    .remove_confirmed(
                        &crate::tests::applied(8),
                        &crate::tests::every_event_applied,
                        "fixture-producer",
                        &|_, _| Ok(false),
                    )
                    .unwrap();
                assert!(spool.get("write").unwrap().is_some());
            }
            "operation_removed" => assert!(all(&spool, 1).is_empty()),
            _ => unreachable!(),
        }
    }
}

#[test]
fn process_death_never_exposes_a_partial_capture_group() {
    for stage in [
        "group_synced",
        "group_published",
        "group_members_published",
        "group_member_removed",
    ] {
        let directory = tempfile::tempdir().unwrap();
        let status = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "spool::tests::crash_worker", "--nocapture"])
            .env("POLYLOGUE_SPOOL_CRASH_ROOT", directory.path())
            .env("POLYLOGUE_SPOOL_CRASH", stage)
            .status()
            .unwrap();
        assert_eq!(status.code(), Some(77), "{stage}");
        let spool = Spool::open(directory.path(), "host").unwrap();
        let members = vec!["zip".into(), "member".into()];
        if stage == "group_synced" {
            assert!(spool.group_for(&members).unwrap().is_none());
            assert!(spool.capture_group("zip").unwrap().is_none());
            assert!(spool.capture_group("member").unwrap().is_none());
        } else if stage != "group_member_removed" {
            assert_eq!(
                spool.group_for(&members).unwrap().unwrap().identity,
                "canonical"
            );
            assert_eq!(
                spool.capture_group("zip").unwrap().unwrap().identity,
                "canonical"
            );
            assert_eq!(
                spool.capture_group("member").unwrap().unwrap().identity,
                "canonical"
            );
        }
        if stage == "group_member_removed" {
            spool
                .remove_confirmed(
                    &crate::tests::applied(9),
                    &crate::tests::every_event_applied,
                    "fixture-producer",
                    &|_, _| Ok(false),
                )
                .unwrap();
            assert!(all(&spool, 1).is_empty());
        }
    }
}

#[test]
fn request_and_blob_visitors_can_use_small_metadata_until_application() {
    let directory = tempfile::tempdir().unwrap();
    let spool = Spool::open(directory.path(), "host").unwrap();
    let request_body = b"{\"request\":\"immutable complete preparation\"}";
    let mut request = material("request", request_body);
    if let Ingress::Material { material, .. } = &mut request {
        material.parser = "polylogue-request".into();
    }
    spool
        .prepare_material(request, request_body.as_slice())
        .unwrap();
    let bytes = b"blob";
    let mut blob = material("blob", bytes);
    if let Ingress::Material { material, .. } = &mut blob {
        material.manifest.details.insert(
            "polylogue_capture".into(),
            serde_json::to_value(Capture::Blob {
                hash: polylogue_model::BlobHash::parse(blake3::hash(bytes).to_hex().as_ref())
                    .unwrap(),
                length: bytes.len() as u64,
            })
            .unwrap(),
        );
    }
    spool.prepare_material(blob, bytes.as_slice()).unwrap();
    spool.acknowledge(ack("request", true, None)).unwrap();
    spool.acknowledge(ack("blob", true, None)).unwrap();
    let page = spool.pending_after(None, 10).unwrap();
    assert_eq!(page.iter().filter(|entry| entry.is_request()).count(), 1);
    assert_eq!(
        page.iter()
            .filter(|entry| matches!(entry.capture(), Some(Capture::Blob { .. })))
            .count(),
        1
    );
    assert!(page.iter().all(|entry| entry.phase.is_acknowledged()));
    spool
        .remove_confirmed(
            &crate::tests::applied(999),
            &crate::tests::every_event_applied,
            "fixture-producer",
            &|_, _| Ok(false),
        )
        .unwrap();
    assert_eq!(spool.pending_after(None, 10).unwrap().len(), 2);
    let mut body = String::new();
    spool
        .body("request")
        .unwrap()
        .read_to_string(&mut body)
        .unwrap();
    assert_eq!(body, std::str::from_utf8(request_body).unwrap());
}

#[test]
fn replay_capture_group_retains_its_material_until_applied() {
    let directory = tempfile::tempdir().unwrap();
    let spool = Spool::open(directory.path(), "host").unwrap();
    spool
        .prepare_material(material("raw", b"bytes"), b"bytes".as_slice())
        .unwrap();
    let replay = Ingress::ReplaySequence {
        frame: "replacement".into(),
        sequence: Box::new(sinex_protocol::ProducerReplay {
            source: crate::source_reference(),
            operation: sinex_model::OperationId(uuid::Uuid::nil()),
            root: crate::material_id("polylogue-test", "raw"),
            materials: vec![],
            event_types: None,
            from: None,
            until: None,
            declarations: vec![],
            parser_build: "fixture".into(),
            parser_version: "fixture".into(),
            expected_live_hash: "fixture".into(),
            batches: vec![],
        }),
    };
    let prepared = spool
        .prepare_group(replay.clone(), &["raw".into()])
        .unwrap();
    assert!(matches!(
        prepared.payload,
        Payload::ReplaySequence { group: Some(_), .. }
    ));
    assert_eq!(spool.decode_frame("replacement").unwrap(), replay);
    let mut replayed = ack("replacement", false, Some(21));
    replayed.result = FrameResult::Replayed {
        operation: sinex_model::OperationId(uuid::Uuid::nil()),
        admitted: vec![],
        commit_seq: Some(21),
    };
    spool.acknowledge(replayed).unwrap();
    spool
        .remove_confirmed(
            &crate::tests::applied(20),
            &crate::tests::every_event_applied,
            "fixture-producer",
            &|_, _| Ok(false),
        )
        .unwrap();
    assert!(spool.get("raw").unwrap().is_some());
    spool
        .remove_confirmed(
            &crate::tests::applied(21),
            &crate::tests::every_event_applied,
            "fixture-producer",
            &|_, _| Ok(false),
        )
        .unwrap();
    assert!(all(&spool, 1).is_empty());
}

fn parsed_capture(bytes: &[u8], text: &str) -> Capture {
    let mut output = polylogue_model::Output::default();
    output.messages.push(serde_json::from_value(serde_json::json!({
        "offset":0,"end":bytes.len(),"message":{"id":"codex:message","session":"codex:session","role":"assistant","attrs":{},"blocks":[{"type":"text","text":text}]}
    })).unwrap());
    let mut step = polylogue_store::ParseStep::new(
        output,
        polylogue_model::RecordCounts::default(),
        0..bytes.len() as u64,
        true,
    );
    step.prepare();
    let parsed = vec![polylogue_store::primary::CapturedReading::from_step(&step, None).unwrap()];
    Capture::Input(serde_json::from_value(serde_json::json!({
        "input":{"origin":"codex","path":"/fixture/session.jsonl","host":"fixture-host"},
        "rows":{"input":"input-x","format":"codex","revision":{"number":1,"produced":"file","chain":1,"length":bytes.len()},"extents":[{"chain":1,"offset":0,"length":bytes.len(),"blob_offset":0,"blob":polylogue_model::BlobHash::of(bytes)}]},
        "index":0,"previous":null,"hash":polylogue_model::BlobHash::of(bytes),"length":bytes.len(),"group":null,"route":"capture","parsed":parsed
    })).unwrap())
}

fn capture_frame(identity: &str, bytes: &[u8], capture: &Capture) -> Ingress {
    let mut frame = material(identity, bytes);
    if let Ingress::Material { material, .. } = &mut frame {
        material.manifest.details.insert(
            "polylogue_capture".into(),
            serde_json::to_value(capture_header(capture)).unwrap(),
        );
    }
    frame
}

#[test]
fn complete_parser_facts_are_immutable_sidecars_and_queue_headers_stay_small() {
    let directory = tempfile::tempdir().unwrap();
    let spool = Spool::open(directory.path(), "host").unwrap();
    let bytes = b"original source";
    let capture = parsed_capture(bytes, &"synthetic answer ".repeat(100_000));
    let frame = capture_frame("captured", bytes, &capture);
    let mut invalid = capture_frame("invalid", bytes, &capture);
    if let Ingress::Material { material, .. } = &mut invalid {
        material.manifest.details.insert(
            "polylogue_capture".into(),
            serde_json::to_value(&capture).unwrap(),
        );
    }
    assert!(
        spool
            .prepare_capture(invalid, bytes.as_slice(), &capture)
            .is_err()
    );
    let prepared = spool
        .prepare_capture(frame.clone(), bytes.as_slice(), &capture)
        .unwrap();
    assert!(serde_json::to_vec(&prepared.record).unwrap().len() < 4096);
    assert!(prepared.captured().unwrap().length > 1_000_000);
    assert_eq!(prepared.capture(), Some(&capture_header(&capture)));
    assert_eq!(spool.capture("captured").unwrap(), capture);
    spool
        .prepare_capture(frame.clone(), bytes.as_slice(), &capture)
        .unwrap();
    let changed = parsed_capture(bytes, "changed facts");
    assert!(
        spool
            .prepare_capture(frame.clone(), bytes.as_slice(), &changed)
            .is_err()
    );
    assert!(
        spool
            .prepare_capture(frame.clone(), b"different source".as_slice(), &capture)
            .is_err()
    );
    drop(spool);
    let spool = Spool::open(directory.path(), "host").unwrap();
    assert_eq!(spool.capture("captured").unwrap(), capture);
    spool.acknowledge(ack("captured", true, None)).unwrap();
    assert_eq!(spool.pending_after(None, 1).unwrap().len(), 1);
    assert_eq!(spool.capture("captured").unwrap(), capture);
    let path = spool.operation("captured").unwrap().join("captured.zst");
    let mut content = fs::read(&path).unwrap();
    content[0] ^= 1;
    fs::write(path, content).unwrap();
    assert!(spool.capture("captured").is_err());
}

#[test]
fn capture_sidecar_and_body_publish_together_after_process_death() {
    let directory = tempfile::tempdir().unwrap();
    let status = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "spool::tests::crash_worker", "--nocapture"])
        .env("POLYLOGUE_SPOOL_CRASH_ROOT", directory.path())
        .env("POLYLOGUE_SPOOL_CRASH", "capture_synced")
        .status()
        .unwrap();
    assert_eq!(status.code(), Some(77));
    let spool = Spool::open(directory.path(), "host").unwrap();
    assert!(spool.pending_after(None, 1).unwrap().is_empty());
}

#[test]
fn request_restore_metadata_remains_discoverable_after_acknowledgment_and_restart() {
    let directory = tempfile::tempdir().unwrap();
    let spool = Spool::open(directory.path(), "host").unwrap();
    for (identity, parser, expected) in [
        ("live", "polylogue-request", true),
        ("restore", "polylogue-request-restore", true),
        ("observation", "polylogue-observation", false),
    ] {
        let mut frame = material(identity, b"[]");
        if let Ingress::Material { material, .. } = &mut frame {
            material.parser = parser.into();
        }
        assert_eq!(
            spool
                .prepare_material(frame, b"[]".as_slice())
                .unwrap()
                .is_request(),
            expected
        );
        spool.acknowledge(ack(identity, true, None)).unwrap();
    }
    drop(spool);
    let spool = Spool::open(directory.path(), "host").unwrap();
    let requests: Vec<_> = spool
        .pending_after(None, 10)
        .unwrap()
        .into_iter()
        .filter(|entry| entry.is_request())
        .map(|entry| entry.identity.clone())
        .collect();
    assert_eq!(requests, ["live", "restore"]);
}

#[test]
fn encoded_capture_adoption_retains_exact_codec_bytes_across_retry_and_restart() {
    let directory = tempfile::tempdir().unwrap();
    let source = tempfile::tempdir().unwrap();
    let bytes = b"synthetic encoded acquisition ".repeat(300);
    let capture = parsed_capture(&bytes, "synthetic parser output");
    let frame = capture_frame("encoded", &bytes, &capture);
    let preparation = Preparation::new(source.path())
        .unwrap()
        .with_frame_size(113)
        .unwrap();
    let encoded = preparation.plain(bytes.as_slice(), 1).unwrap();
    let expected = encoded.descriptor();
    let spool = Spool::open(directory.path(), "host").unwrap();
    spool
        .prepare_encoded_capture(frame.clone(), vec![prepared_body(encoded, None)], &capture)
        .unwrap();
    let (path, actual) = spool.encoded_body("encoded").unwrap().remove(0);
    assert_eq!(actual, expected);
    assert_eq!(
        hashed_blobs::Hash::of(&fs::read(&path).unwrap()),
        expected.encoded_hash
    );
    assert_eq!(
        spool.prepared_body("encoded").unwrap()[0].descriptor(),
        expected
    );
    let retry = Preparation::new(source.path())
        .unwrap()
        .plain(bytes.as_slice(), 3)
        .unwrap();
    assert_ne!(
        retry.descriptor(),
        expected,
        "fixture distinguishes recompression"
    );
    spool
        .prepare_encoded_capture(frame, vec![prepared_body(retry, None)], &capture)
        .unwrap();
    assert_eq!(spool.encoded_body("encoded").unwrap().remove(0).1, expected);
    assert_eq!(spool.capture("encoded").unwrap(), capture);
    drop(spool);
    let spool = Spool::open(directory.path(), "host").unwrap();
    assert_eq!(spool.encoded_body("encoded").unwrap().remove(0).1, expected);
    let mut decoded = Vec::new();
    spool
        .body("encoded")
        .unwrap()
        .read_to_end(&mut decoded)
        .unwrap();
    assert_eq!(decoded, bytes);
}

#[test]
fn corrupted_encoded_capture_is_rejected_before_publication() {
    let directory = tempfile::tempdir().unwrap();
    let source = tempfile::tempdir().unwrap();
    let bytes = b"synthetic source";
    let encoded = Preparation::new(source.path())
        .unwrap()
        .plain(bytes.as_slice(), 3)
        .unwrap();
    let descriptor = encoded.descriptor();
    let file = encoded.into_file();
    let mut corrupt = file.try_clone().unwrap();
    let encoded =
        hashed_blobs::Prepared::from_encoded(file, descriptor, None, ReadLimits::default())
            .unwrap();
    corrupt.rewind().unwrap();
    corrupt.write_all(&[0]).unwrap();
    let spool = Spool::open(directory.path(), "host").unwrap();
    assert!(
        spool
            .prepare_encoded_material(material("invalid", bytes), prepared_body(encoded, None))
            .is_err()
    );
    assert!(spool.get("invalid").unwrap().is_none());
    assert!(spool.pending_after(None, 1).unwrap().is_empty());
}

fn prepared_body(
    prepared: hashed_blobs::Prepared,
    path: Option<PathBuf>,
) -> polylogue_store::primary::PreparedBody {
    polylogue_store::primary::PreparedBody {
        descriptor: prepared.descriptor(),
        file: prepared.into_file(),
        path,
    }
}

#[test]
fn prepared_path_links_only_the_pinned_inode_and_survives_original_path_drop() {
    use std::os::unix::fs::MetadataExt;
    let directory = tempfile::tempdir().unwrap();
    let source = tempfile::NamedTempFile::new_in(directory.path()).unwrap();
    let bytes = b"synthetic exact encoded source";
    let encoded = Preparation::new(directory.path())
        .unwrap()
        .plain_into(source.as_file().try_clone().unwrap(), bytes.as_slice(), 3)
        .unwrap();
    let inode = source.as_file().metadata().unwrap().ino();
    let spool = Spool::open(directory.path(), "host").unwrap();
    spool
        .prepare_encoded_material(
            material("linked", bytes),
            prepared_body(encoded, Some(source.path().into())),
        )
        .unwrap();
    let (path, _) = spool.encoded_body("linked").unwrap().remove(0);
    assert_eq!(fs::metadata(path).unwrap().ino(), inode);
    drop(source);
    let mut decoded = Vec::new();
    spool
        .body("linked")
        .unwrap()
        .read_to_end(&mut decoded)
        .unwrap();
    assert_eq!(decoded, bytes);
}

#[test]
fn changed_or_symlinked_prepared_path_falls_back_to_the_verified_pinned_file() {
    use std::os::unix::fs::{MetadataExt, symlink};
    let directory = tempfile::tempdir().unwrap();
    let wrong = directory.path().join("different-inode");
    fs::write(&wrong, b"not the pinned data").unwrap();
    let alias = directory.path().join("symlink");
    symlink(&wrong, &alias).unwrap();
    let spool = Spool::open(directory.path(), "host").unwrap();
    for (identity, path) in [("different", wrong.clone()), ("symlinked", alias)] {
        let bytes = b"the pinned source";
        let encoded = Preparation::new(directory.path())
            .unwrap()
            .plain(bytes.as_slice(), 3)
            .unwrap();
        spool
            .prepare_encoded_material(
                material(identity, bytes),
                prepared_body(encoded, Some(path)),
            )
            .unwrap();
        let (path, _) = spool.encoded_body(identity).unwrap().remove(0);
        assert_ne!(
            fs::metadata(path).unwrap().ino(),
            fs::metadata(&wrong).unwrap().ino()
        );
        let mut decoded = Vec::new();
        spool
            .body(identity)
            .unwrap()
            .read_to_end(&mut decoded)
            .unwrap();
        assert_eq!(decoded, bytes);
    }
    assert_eq!(fs::read(wrong).unwrap(), b"not the pinned data");
}

#[test]
fn every_payload_kind_is_encoded_and_counts_never_decode_it() {
    let directory = tempfile::tempdir().unwrap();
    let spool = Spool::open(directory.path(), "host").unwrap();
    let bytes = b"synthetic capture";
    let capture = parsed_capture(bytes, "synthetic parser output");
    spool
        .prepare_capture(
            capture_frame("raw", bytes, &capture),
            bytes.as_slice(),
            &capture,
        )
        .unwrap();
    spool
        .prepare_group(events("group", "canonical facts"), &["raw".into()])
        .unwrap();
    spool.acknowledge(ack("group", false, Some(19))).unwrap();
    for (identity, relative) in [
        ("raw", "frame.zst"),
        ("raw", "body.zst"),
        ("raw", "captured.zst"),
        ("group", "frame.zst"),
        ("group", "ack/payload.zst"),
    ] {
        let data = fs::read(spool.operation(identity).unwrap().join(relative)).unwrap();
        assert_eq!(&data[..4], &[0x28, 0xb5, 0x2f, 0xfd], "{relative}");
    }
    let before = spool.counts().unwrap();
    let path = spool.operation("raw").unwrap().join("frame.zst");
    let mut data = fs::read(&path).unwrap();
    data[0] ^= 1;
    fs::write(path, data).unwrap();
    assert_eq!(spool.counts().unwrap(), before);
    assert!(spool.decode_frame("raw").is_err());
}

#[test]
fn plaintext_predecessor_operations_require_an_explicit_preserved_transition() {
    let directory = tempfile::tempdir().unwrap();
    let spool = Spool::open(directory.path(), "host").unwrap();
    let path = spool.operation("old").unwrap();
    fs::create_dir(&path).unwrap();
    fs::write(path.join("frame.json"), b"{synthetic old pending bytes}").unwrap();
    drop(spool);
    assert!(Spool::open(directory.path(), "host").is_err());
    assert!(path.join("frame.json").exists());
}

#[test]
fn acknowledgment_body_conflict_is_terminal_evidence_and_keeps_offered_bytes_after_restart() {
    let root = tempfile::tempdir().unwrap();
    let spool = Spool::open(root.path(), "host").unwrap();
    let bytes = b"actual newly offered provider bytes";
    let frame = material("same-frame", bytes);
    let prepared = spool.prepare_material(frame, bytes.as_slice()).unwrap();
    let prior_frame = material("same-frame", b"previous different provider bytes");
    let mut acknowledgment = ack("same-frame", false, Some(9));
    acknowledgment.content_hash =
        Some(sinex_transport::producer::content_hash(&prior_frame).unwrap());
    let Err(Error::AcknowledgmentConflict {
        frame,
        offered_hash,
        acknowledgment: conflict,
    }) = crate::client::verified_acknowledgment(&prepared, acknowledgment.clone())
    else {
        panic!("different recorded contents are an acknowledgment conflict")
    };
    spool
        .record_conflict(&frame, &offered_hash, &conflict)
        .unwrap();
    let terminal = spool.get("same-frame").unwrap().unwrap();
    assert!(matches!(terminal.phase, Phase::Conflicted { .. }));
    assert!(spool.acknowledgment("same-frame").unwrap().is_none());
    assert_eq!(
        spool.conflict_acknowledgment("same-frame").unwrap(),
        Some(acknowledgment.clone())
    );
    drop(spool);
    let spool = Spool::open(root.path(), "host").unwrap();
    spool
        .remove_confirmed(
            &crate::tests::applied(i64::MAX),
            &crate::tests::every_event_applied,
            "fixture-producer",
            &|_, _| Ok(false),
        )
        .unwrap();
    let retained = spool.get("same-frame").unwrap().unwrap();
    assert!(matches!(retained.phase, Phase::Conflicted { .. }));
    let mut observed = Vec::new();
    spool
        .body("same-frame")
        .unwrap()
        .read_to_end(&mut observed)
        .unwrap();
    assert_eq!(observed, bytes);
    spool
        .record_conflict("same-frame", &prepared.frame.blob, &acknowledgment)
        .unwrap();
    assert_eq!(
        spool.conflict_acknowledgment("same-frame").unwrap(),
        Some(acknowledgment)
    );
}

#[test]
fn conflict_evidence_for_another_frame_cannot_change_an_operation_phase() {
    let root = tempfile::tempdir().unwrap();
    let spool = Spool::open(root.path(), "host").unwrap();
    let prepared = spool.prepare(events("owned", "frozen bytes")).unwrap();
    let acknowledgment = ack("foreign", false, Some(2));
    assert!(
        spool
            .record_conflict("owned", &prepared.frame.blob, &acknowledgment)
            .is_err()
    );
    let unchanged = spool.get("owned").unwrap().unwrap();
    assert!(unchanged.phase.is_pending());
    assert!(spool.conflict_acknowledgment("owned").unwrap().is_none());
}

#[test]
fn a_finished_operation_refuses_a_different_outcome_through_the_spool() {
    let root = tempfile::tempdir().unwrap();
    let spool = Spool::open(root.path(), "host").unwrap();
    let prepared = spool.prepare(events("write", "frozen")).unwrap();
    let foreign = ack("write", false, Some(2));
    spool
        .record_conflict("write", &prepared.frame.blob, &{
            let mut acknowledgment = foreign.clone();
            acknowledgment.content_hash = Some("blake3:other".into());
            acknowledgment
        })
        .unwrap();
    assert!(spool.acknowledge(foreign).is_err());
    assert!(matches!(
        spool.get("write").unwrap().unwrap().phase,
        Phase::Conflicted { .. }
    ));

    let accepted = spool.prepare(events("accepted", "frozen")).unwrap();
    spool.acknowledge(ack("accepted", false, Some(5))).unwrap();
    let acknowledgment = ack("accepted", false, Some(6));
    assert!(
        spool
            .record_conflict("accepted", &accepted.frame.blob, &acknowledgment)
            .is_err()
    );
    assert!(spool.acknowledge(ack("accepted", false, Some(6))).is_err());
    assert_eq!(
        spool.get("accepted").unwrap().unwrap().phase,
        Phase::Acknowledged {
            commit_seq: Some(5)
        }
    );
}

#[test]
fn complete_no_position_outcomes_preserve_acknowledgment_and_application_identity() {
    let directory = tempfile::tempdir().unwrap();
    let spool = Spool::open(directory.path(), "host").unwrap();
    let mut repeat = events("coverage-repeat", "owner-known event");
    if let Ingress::Events { coverage, .. } = &mut repeat {
        *coverage = Some(sinex_protocol::CoverageProof {
            stamp: "owner-signed-coverage".into(),
            position: 42,
            scope: "scope".into(),
        });
    }
    spool.prepare(repeat.clone()).unwrap();
    let mut acknowledgment = crate::tests::no_event_acknowledgment(&repeat);
    counts(&mut acknowledgment).repeats = 1;
    spool.acknowledge(acknowledgment.clone()).unwrap();
    assert_eq!(
        spool.acknowledgment(repeat.frame()).unwrap(),
        Some(acknowledgment)
    );
    assert_eq!(
        spool.get(repeat.frame()).unwrap().unwrap().phase,
        Phase::Acknowledged { commit_seq: None }
    );
    // An accepted metadata operation can settle without claiming any index
    // position. Raw materials remain governed by the group barrier below.
    spool
        .remove_confirmed(
            &ReplayState::default(),
            &crate::tests::every_event_applied,
            "fixture-producer",
            &|_, _| Ok(false),
        )
        .unwrap();
    assert!(spool.get(repeat.frame()).unwrap().is_none());

    let empty = Ingress::RegisterKinds {
        frame: "empty-kinds".into(),
        declarations: vec![],
    };
    spool.prepare(empty.clone()).unwrap();
    spool
        .acknowledge(crate::tests::no_event_acknowledgment(&empty))
        .unwrap();
    spool
        .remove_confirmed(
            &ReplayState::default(),
            &crate::tests::every_event_applied,
            "fixture-producer",
            &|_, _| Ok(false),
        )
        .unwrap();
    assert!(spool.get(empty.frame()).unwrap().is_none());
}

#[test]
fn incomplete_or_new_admission_without_position_cannot_settle_or_remove_raw_group() {
    let directory = tempfile::tempdir().unwrap();
    let spool = Spool::open(directory.path(), "host").unwrap();
    let bytes = b"retained provider bytes";
    spool
        .prepare_material(material("raw", bytes), bytes.as_slice())
        .unwrap();
    spool.acknowledge(ack("raw", true, None)).unwrap();
    let frame = events("no-position", "known record");
    spool.prepare_group(frame.clone(), &["raw".into()]).unwrap();
    let complete = crate::tests::no_event_acknowledgment(&frame);
    let new_events: [fn(&mut sinex_model::AdmissionCounts); 2] =
        [|counts| counts.admitted = 1, |counts| counts.replaced = 1];
    for count in new_events {
        let mut acknowledgment = complete.clone();
        count(counts(&mut acknowledgment));
        assert!(spool.acknowledge(acknowledgment).is_err());
    }
    let mut acknowledgment = complete.clone();
    acknowledgment.content_hash = Some("different original bytes".into());
    assert!(spool.acknowledge(acknowledgment).is_err());
    let mut acknowledgment = complete.clone();
    if let FrameResult::Admitted { admitted, .. } = &mut acknowledgment.result {
        admitted.push(sinex_model::EventId(uuid::Uuid::from_bytes([1; 16])));
    }
    assert!(spool.acknowledge(acknowledgment).is_err());
    let mut refused = complete;
    refused.result = FrameResult::Refused {
        kind: sinex_model::FrameRefusal::Content,
        reason: "owner refused data".into(),
    };
    spool.acknowledge(refused).unwrap();
    spool
        .remove_confirmed(
            &crate::tests::applied(42),
            &crate::tests::every_event_applied,
            "fixture-producer",
            &|_, _| Ok(false),
        )
        .unwrap();
    assert!(spool.get("raw").unwrap().is_some());
    assert!(
        spool
            .get(frame.frame())
            .unwrap()
            .unwrap()
            .phase
            .is_refused()
    );
}

#[test]
fn new_no_position_group_at_same_completed_head_is_removed() {
    let directory = tempfile::tempdir().unwrap();
    let spool = Spool::open(directory.path(), "host").unwrap();
    let state = crate::tests::applied(42);
    spool
        .remove_confirmed(
            &state,
            &crate::tests::every_event_applied,
            "fixture-producer",
            &|_, _| Ok(false),
        )
        .unwrap();
    spool
        .prepare_material(material("raw", b"bytes"), b"bytes".as_slice())
        .unwrap();
    spool.acknowledge(ack("raw", true, None)).unwrap();
    let frame = events("repeat", "known record");
    spool.prepare_group(frame.clone(), &["raw".into()]).unwrap();
    let mut acknowledgment = crate::tests::no_event_acknowledgment(&frame);
    counts(&mut acknowledgment).repeats = 1;
    spool.acknowledge(acknowledgment).unwrap();
    spool
        .remove_confirmed(
            &state,
            &crate::tests::every_event_applied,
            "fixture-producer",
            &|_, _| Ok(false),
        )
        .unwrap();
    assert!(
        spool.get("raw").unwrap().is_none(),
        "new same-head ACK invalidates the memo"
    );
    assert!(spool.get(frame.frame()).unwrap().is_none());
    spool
        .prepare_material(material("raw-next", b"next"), b"next".as_slice())
        .unwrap();
    spool.acknowledge(ack("raw-next", true, None)).unwrap();
    let frame = events("repeat-next", "known next record");
    spool
        .prepare_group(frame.clone(), &["raw-next".into()])
        .unwrap();
    let mut acknowledgment = crate::tests::no_event_acknowledgment(&frame);
    counts(&mut acknowledgment).repeats = 1;
    spool.acknowledge(acknowledgment).unwrap();
    let mut behind = state.clone();
    behind.applied_position = Some(41);
    spool
        .remove_confirmed(
            &behind,
            &crate::tests::every_event_applied,
            "fixture-producer",
            &|_, _| Ok(false),
        )
        .unwrap();
    assert!(spool.get("raw-next").unwrap().is_some());
    let mut stale = state.clone();
    stale.head = None;
    spool
        .remove_confirmed(
            &stale,
            &crate::tests::every_event_applied,
            "fixture-producer",
            &|_, _| Ok(false),
        )
        .unwrap();
    assert!(spool.get("raw-next").unwrap().is_some());
    // A fresh verified post-ACK head can have the same numeric position.
    spool
        .remove_confirmed(
            &state,
            &crate::tests::every_event_applied,
            "fixture-producer",
            &|_, _| Ok(false),
        )
        .unwrap();
    assert!(spool.get("raw-next").unwrap().is_none());
    assert!(spool.get(frame.frame()).unwrap().is_none());
}

#[test]
fn archive_only_replay_without_admitted_event_position_waits_for_completed_prefix() {
    let directory = tempfile::tempdir().unwrap();
    let spool = Spool::open(directory.path(), "host").unwrap();
    spool
        .prepare_material(material("raw", b"bytes"), b"bytes".as_slice())
        .unwrap();
    spool.acknowledge(ack("raw", true, None)).unwrap();
    let replay = Ingress::ReplaySequence {
        frame: "archive-only".into(),
        sequence: Box::new(sinex_protocol::ProducerReplay {
            source: crate::source_reference(),
            operation: sinex_model::OperationId(uuid::Uuid::nil()),
            root: crate::material_id("polylogue-test", "raw"),
            materials: vec![],
            event_types: None,
            from: None,
            until: None,
            declarations: vec![],
            parser_build: "fixture".into(),
            parser_version: "fixture".into(),
            expected_live_hash: "fixture".into(),
            batches: vec![],
        }),
    };
    spool
        .prepare_group(replay.clone(), &["raw".into()])
        .unwrap();
    let mut acknowledgment = crate::tests::no_event_acknowledgment(&replay);
    acknowledgment.result = FrameResult::Replayed {
        operation: sinex_model::OperationId(uuid::Uuid::nil()),
        admitted: vec![],
        commit_seq: None,
    };
    spool.acknowledge(acknowledgment.clone()).unwrap();
    let mut state = crate::tests::applied(42);
    state.head = Some(43); // lifecycle commit belongs to the stream, not the event ACK
    spool
        .remove_confirmed(
            &state,
            &crate::tests::every_event_applied,
            "fixture-producer",
            &|_, _| Ok(false),
        )
        .unwrap();
    assert!(spool.get("raw").unwrap().is_some());
    assert_eq!(
        spool.acknowledgment(replay.frame()).unwrap(),
        Some(acknowledgment.clone())
    );
    let pinned = spool
        .pin_backup(&AtomicBool::new(false))
        .unwrap()
        .finish(&AtomicBool::new(false))
        .unwrap();
    state.applied_position = Some(43);
    spool
        .remove_confirmed(
            &state,
            &crate::tests::every_event_applied,
            "fixture-producer",
            &|_, _| Ok(false),
        )
        .unwrap();
    assert!(spool.get("raw").unwrap().is_none());
    let restored = Spool::open(pinned.path(), "host").unwrap();
    assert_eq!(
        restored.acknowledgment(replay.frame()).unwrap(),
        Some(acknowledgment)
    );
    let mut bytes = Vec::new();
    restored
        .body("raw")
        .unwrap()
        .read_to_end(&mut bytes)
        .unwrap();
    assert_eq!(bytes, b"bytes");
}

#[test]
fn source_five_pending_capture_keeps_opaque_context_and_original_material_bytes() {
    let directory = tempfile::tempdir().unwrap();
    let bytes = b"original source bytes";
    let mut capture = parsed_capture(bytes, "synthetic parser facts");
    let Capture::Input(input) = &mut capture else {
        unreachable!()
    };
    let cursor = input.parsed[0].parsed_upto;
    input.parsed[0].checkpoint = Some(polylogue_model::batch::ParserCheckpoint::new(
        input.parsed[0].reading_version.clone(),
        cursor,
        Some(
            serde_json::json!({"_polylogue_context":{"version":1,"base":0,"through":cursor,"entries":[{"key":{"kind":"alias","namespace":"","uuid":"native-later"},"value":{"kind":"alias","target":"codex:message"}}]}}),
        ),
    ));
    let mut frame = capture_frame("old-context", bytes, &capture);
    if let Ingress::Material { material, .. } = &mut frame {
        material.source = sinex_protocol::SourceRef {
            name: "polylogue".into(),
            version: 5,
        };
    }
    let spool = Spool::open(directory.path(), "host").unwrap();
    spool
        .prepare_capture(frame.clone(), bytes.as_slice(), &capture)
        .unwrap();
    let raw_frame = fs::read(spool.operation("old-context").unwrap().join("frame.zst")).unwrap();
    let descriptor = spool.encoded_body("old-context").unwrap().remove(0).1;
    drop(spool);
    let spool = Spool::open(directory.path(), "host").unwrap();
    assert_eq!(spool.capture("old-context").unwrap(), capture);
    assert_eq!(spool.decode_frame("old-context").unwrap(), frame);
    assert_eq!(
        spool.encoded_body("old-context").unwrap().remove(0).1,
        descriptor
    );
    assert_eq!(
        fs::read(spool.operation("old-context").unwrap().join("frame.zst")).unwrap(),
        raw_frame
    );
    spool
        .prepare_capture(frame, bytes.as_slice(), &capture)
        .unwrap();
    assert_eq!(
        fs::read(spool.operation("old-context").unwrap().join("frame.zst")).unwrap(),
        raw_frame
    );
}

fn unpublished_names(root: &Path) -> Vec<String> {
    let mut found = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(directory) = stack.pop() {
        for entry in fs::read_dir(&directory).unwrap() {
            let entry = entry.unwrap();
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with(".preparing-") || name.starts_with(".ack-preparing-") {
                found.push(name);
            } else if entry.file_type().unwrap().is_dir() {
                stack.push(entry.path());
            }
        }
    }
    found
}

#[test]
fn open_removes_unpublished_preparations() {
    // `ack_synced` leaves `.ack-preparing-*` inside a published operation.
    for stage in ["body_synced", "acquisition_synced", "ack_synced"] {
        let directory = tempfile::tempdir().unwrap();
        let status = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "spool::tests::crash_worker", "--nocapture"])
            .env("POLYLOGUE_SPOOL_CRASH_ROOT", directory.path())
            .env("POLYLOGUE_SPOOL_CRASH", stage)
            .status()
            .unwrap();
        assert_eq!(status.code(), Some(77), "{stage}");
        assert!(
            !unpublished_names(directory.path()).is_empty(),
            "{stage}: the crash left no preparation to remove"
        );
        let spool = Spool::open(directory.path(), "host").unwrap();
        assert_eq!(
            unpublished_names(directory.path()),
            Vec::<String>::new(),
            "{stage}"
        );
        if stage == "ack_synced" {
            assert!(spool.get("write").unwrap().unwrap().phase.is_pending());
            assert_eq!(all(&spool, 8).len(), 1);
        } else {
            assert!(all(&spool, 8).is_empty(), "{stage}");
        }
    }
}

#[test]
fn an_operation_whose_events_the_index_lacks_is_not_removed() {
    let directory = tempfile::tempdir().unwrap();
    let spool = Spool::open(directory.path(), "host").unwrap();
    spool.prepare(events("lost-in-stream", "written")).unwrap();
    let acknowledgment: Acknowledgment = serde_json::from_value(serde_json::json!({
        "number":1, "frame":"lost-in-stream", "material":null, "material_extents":[],
        "admitted":["01900000-0000-7000-8000-000000000001"],
        "content_hash":null,"result":"admitted","counts":crate::tests::zero_counts(), "commit_seq":10
    }))
    .unwrap();
    let event = acknowledgment.admitted()[0].to_string();
    spool.acknowledge(acknowledgment).unwrap();

    // The applied position passed 10, but index.db never received the event:
    // the confirmed stream lost that position.
    spool
        .remove_confirmed(
            &crate::tests::applied(20),
            &|_| Ok(false),
            "fixture-producer",
            &|_, _| Ok(false),
        )
        .unwrap();
    assert!(spool.get("lost-in-stream").unwrap().is_some());

    spool
        .remove_confirmed(
            &crate::tests::applied(21),
            &|applied| Ok(applied == event),
            "fixture-producer",
            &|_, _| Ok(false),
        )
        .unwrap();
    assert!(spool.get("lost-in-stream").unwrap().is_none());
}

#[test]
fn last_acknowledgement_time_read_from_spool_after_restart() {
    let directory = tempfile::tempdir().unwrap();
    let before = std::time::SystemTime::now() - std::time::Duration::from_secs(1);
    {
        let spool = Spool::open(directory.path(), "host").unwrap();
        assert_eq!(spool.last_acknowledged().unwrap(), None);
        spool.prepare(events("acknowledged", "value")).unwrap();
        spool
            .acknowledge(ack("acknowledged", false, Some(3)))
            .unwrap();
    }
    // A new process, after the operation has removed, still knows when
    // Sinex last answered.
    let spool = Spool::open(directory.path(), "host").unwrap();
    spool
        .remove_confirmed(
            &crate::tests::applied(10),
            &crate::tests::every_event_applied,
            "fixture-producer",
            &|_, _| Ok(false),
        )
        .unwrap();
    assert!(spool.get("acknowledged").unwrap().is_none());
    let at = spool
        .last_acknowledged()
        .unwrap()
        .expect("a recorded acknowledgement");
    assert!(at >= before, "{at:?}");
    assert!(at <= std::time::SystemTime::now());
}

#[test]
fn confirmed_removal_does_not_read_unrelated_pending_metadata_or_acknowledgments() {
    let directory = tempfile::tempdir().unwrap();
    let spool = Spool::open(directory.path(), "host").unwrap();
    for number in 0..5000 {
        let operation = empty_operation(
            format!("unrelated-{number}"),
            format!("unused-{number}"),
            b"not decoded by removal".to_vec(),
            queue::Kind::Events,
        );
        spool.queue.prepare(&operation, None, &[]).unwrap();
    }
    spool.prepare(events("confirmed", "visible fact")).unwrap();
    let mut acknowledgment = ack("confirmed", false, Some(10));
    if let FrameResult::Admitted {
        admitted, counts, ..
    } = &mut acknowledgment.result
    {
        admitted.push(sinex_model::EventId(uuid::Uuid::nil()));
        counts.admitted = 1;
    }
    spool.acknowledge(acknowledgment).unwrap();
    fs::write(
        spool
            .operation("confirmed")
            .unwrap()
            .join("ack/payload.zst"),
        b"already validated on admission",
    )
    .unwrap();
    let checked = std::cell::Cell::new(0);
    spool
        .remove_confirmed(
            &crate::tests::applied(10),
            &|identity| {
                assert_eq!(identity, uuid::Uuid::nil().to_string());
                checked.set(checked.get() + 1);
                Ok(true)
            },
            "fixture-producer",
            &|_, _| Ok(false),
        )
        .unwrap();
    assert_eq!(checked.get(), 1);
    assert!(spool.get("confirmed").unwrap().is_none());
    assert_eq!(spool.queue.len().unwrap(), 5000);
}

#[test]
fn a_published_acknowledgment_finishes_its_queue_transition_after_a_crash() {
    let directory = tempfile::tempdir().unwrap();
    let status = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "spool::tests::crash_worker", "--nocapture"])
        .env("POLYLOGUE_SPOOL_CRASH_ROOT", directory.path())
        .env("POLYLOGUE_SPOOL_CRASH", "ack_files_published")
        .status()
        .unwrap();
    assert_eq!(status.code(), Some(77));
    let spool = Spool::open(directory.path(), "host").unwrap();
    assert!(spool.get("write").unwrap().unwrap().phase.is_pending());
    let acknowledgment = spool.acknowledgment("write").unwrap().unwrap();
    spool.acknowledge(acknowledgment.clone()).unwrap();
    assert_eq!(
        spool.get("write").unwrap().unwrap().phase,
        Phase::Acknowledged {
            commit_seq: Some(9)
        }
    );
    let bytes = spool.counts().unwrap().2;
    spool.acknowledge(acknowledgment).unwrap();
    assert_eq!(spool.counts().unwrap().2, bytes);
    assert!(spool.ready(&[], 1).unwrap().is_empty());
}

#[test]
fn admitted_source_publication_survives_until_its_local_interpretation_commits() {
    let directory = tempfile::tempdir().unwrap();
    let source = ParsedSource {
        root_material: "00000000-0000-0000-0000-000000000001".into(),
        reading: serde_json::from_value(serde_json::json!({
            "input":{"id":"original","format":"codex","route":"live"},
            "chain":1,"reading_version":"codex=2","source_length":12,
            "order_from":null,"repeats_whole_conversations":false,
            "start":0,"parsed_upto":12,"progress":"at_end"
        }))
        .unwrap(),
    };
    let frame = events("interpreted-original", "native fact");
    let spool = Spool::open(directory.path(), "host").unwrap();
    spool.prepare_parsed(frame.clone(), &source).unwrap();
    assert_eq!(
        spool.publication(frame.frame(), "producer").unwrap(),
        ParsedPublication::Pending {
            identity: frame.frame().into()
        }
    );
    spool
        .acknowledge(crate::tests::no_event_acknowledgment(&frame))
        .unwrap();
    let expected = PublicationReceipt {
        producer: "producer".into(),
        frame: frame.frame().into(),
        number: 1,
        root_material: source.root_material.clone(),
    };
    assert_eq!(
        spool.publication(frame.frame(), "producer").unwrap(),
        ParsedPublication::Admitted(expected.clone())
    );
    drop(spool);
    let spool = Spool::open(directory.path(), "host").unwrap();
    spool
        .remove_confirmed(
            &crate::tests::applied(100),
            &crate::tests::every_event_applied,
            "producer",
            &|queued, receipt| {
                assert_eq!(queued, &source);
                assert_eq!(receipt, &expected);
                Ok(false)
            },
        )
        .unwrap();
    assert!(spool.get(frame.frame()).unwrap().is_some());
    spool
        .remove_confirmed(
            &crate::tests::applied(100),
            &crate::tests::every_event_applied,
            "producer",
            &|queued, receipt| Ok(queued == &source && receipt == &expected),
        )
        .unwrap();
    assert!(spool.get(frame.frame()).unwrap().is_none());
}
