use super::*;
use std::process::Command;

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
            id: crate::material_id(identity),
            input: sinex_model::InputRef {
                kind: "fixture".into(),
                locator: identity.into(),
                host: "fixture-host".into(),
            },
            blob,
            length,
            parser: "fixture".into(),
            parser_version: 1,
            part_index: 0,
            previous_part: None,
            manifest: sinex_model::Manifest::default(),
        }),
    }
}

fn ack(identity: &str, material: bool, position: Option<i64>) -> Acknowledgment {
    serde_json::from_value(serde_json::json!({
        "number":1, "frame":identity, "material":material.then(||crate::material_id(identity)),"material_extents":[],
        "admitted":[],"outcome":{},"refusal":null,"commit_seq":position
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
    spool.prepare(events("write", "original")).unwrap();
    spool.acknowledge(ack("write", false, Some(18))).unwrap();
    drop(spool);
    let spool = Spool::open(directory.path(), "host-a").unwrap();
    let page = spool.pending_after(None, 1).unwrap();
    assert!(page[0].acknowledged);
    spool.retire(17).unwrap();
    assert!(spool.get("write").unwrap().is_some());
    spool.retire(18).unwrap();
    assert!(spool.get("write").unwrap().is_none());
}

#[test]
fn one_group_owns_every_capture_until_its_canonical_position_is_applied() {
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
    spool.retire(999).unwrap();
    assert!(spool.get("zip").unwrap().is_some());
    spool
        .acknowledge(ack("canonical", false, Some(21)))
        .unwrap();
    spool.retire(20).unwrap();
    assert_eq!(spool.counts().unwrap().0, 2);
    spool.retire(21).unwrap();
    assert!(all(&spool, 2).is_empty());
    assert!(spool.get("member").unwrap().is_none());
}

#[test]
fn malformed_receipts_and_group_members_never_authorize_retirement() {
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
    let receipt = spool.operation("group").join("receipt.json");
    let mut damaged: serde_json::Value =
        serde_json::from_reader(File::open(&receipt).unwrap()).unwrap();
    damaged["value"]["commit_seq"] = 0.into();
    fs::write(&receipt, serde_json::to_vec(&damaged).unwrap()).unwrap();
    assert!(spool.retire(1).is_err());
    assert!(spool.get("raw").unwrap().is_some());
    fs::remove_file(&receipt).unwrap();
    assert_eq!(spool.get("group").unwrap().unwrap().commit_seq, Some(12));
    fs::write(
        spool.operation("group").join("captures.jsonl"),
        b"\"other\"\n",
    )
    .unwrap();
    assert!(spool.retire(12).is_err());
    assert!(spool.get("raw").unwrap().is_some());
}

#[test]
fn material_bodies_are_verified_including_empty_files_and_fifo_rejection() {
    let directory = tempfile::tempdir().unwrap();
    let spool = Spool::open(directory.path(), "host").unwrap();
    assert!(spool.prepare(material("empty", b"")).is_err());
    spool
        .prepare_material(material("empty", b""), std::io::empty())
        .unwrap();
    assert_eq!(spool.body("empty").unwrap().metadata().unwrap().len(), 0);
    assert!(
        spool
            .prepare_material(material("wrong", b"right"), b"wrong".as_slice())
            .is_err()
    );
    assert!(spool.get("wrong").unwrap().is_none());
    let path = spool.operation("empty").join("body");
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
    assert!(serde_json::to_vec(&page).unwrap().len() < 4096);
    assert!(
        spool
            .operation("large-frame")
            .join("metadata.json")
            .metadata()
            .unwrap()
            .len()
            < 4096
    );
    assert!(spool.counts().unwrap().2 > (28 << 20));
    assert_eq!(
        spool.body("large-body").unwrap().metadata().unwrap().len(),
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
    if stage == "capture_synced" {
        let bytes = b"original source";
        let capture = parsed_capture(bytes, "synthetic answer");
        spool
            .prepare_capture(
                capture_frame("captured", bytes, &capture),
                bytes.as_slice(),
                &capture,
            )
            .unwrap();
    } else if stage == "discard_member_retired" {
        spool
            .prepare_material(
                material("private", b"private bytes"),
                b"private bytes".as_slice(),
            )
            .unwrap();
        spool
            .prepare_group(events("doomed", "private projection"), &["private".into()])
            .unwrap();
        spool
            .discard_confirmed(&["private".into(), "doomed".into()], 12, 12)
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
        spool.retire(9).unwrap();
    } else if stage == "ack_published" || stage == "operation_retired" {
        spool.prepare(events("write", "original")).unwrap();
        spool.acknowledge(ack("write", false, Some(9))).unwrap();
        spool.retire(9).unwrap();
    } else {
        spool
            .prepare_material(material("raw", b"source bytes"), b"source bytes".as_slice())
            .unwrap();
    }
    panic!("fault boundary did not terminate the process");
}

#[test]
fn process_death_at_durability_boundaries_exposes_only_complete_operations() {
    for stage in [
        "body_synced",
        "operation_synced",
        "operation_published",
        "ack_published",
        "operation_retired",
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
                assert!(prepared.body.is_some());
                assert_eq!(spool.body("raw").unwrap().metadata().unwrap().len(), 12);
            }
            "ack_published" => {
                assert!(spool.get("write").unwrap().unwrap().acknowledged);
                spool.retire(8).unwrap();
                assert!(spool.get("write").unwrap().is_some());
            }
            "operation_retired" => assert!(all(&spool, 1).is_empty()),
            _ => unreachable!(),
        }
    }
}

#[test]
fn process_death_never_exposes_a_partial_capture_group() {
    for stage in [
        "group_synced",
        "group_published",
        "group_indexed_member",
        "group_member_retired",
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
        } else {
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
        if stage == "group_member_retired" {
            spool.retire(9).unwrap();
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
                hash: polylogue_model::BlobHash::parse(&blake3::hash(bytes).to_hex().to_string())
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
    assert_eq!(page.iter().filter(|entry| entry.request).count(), 1);
    assert_eq!(
        page.iter()
            .filter(|entry| matches!(entry.capture, Some(Capture::Blob { .. })))
            .count(),
        1
    );
    assert!(page.iter().all(|entry| entry.acknowledged));
    spool.retire(999).unwrap();
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
fn confirmed_tombstones_discard_only_complete_authorized_groups() {
    let directory = tempfile::tempdir().unwrap();
    let spool = Spool::open(directory.path(), "host").unwrap();
    for identity in ["root", "part"] {
        spool
            .prepare_material(material(identity, b"private"), b"private".as_slice())
            .unwrap();
    }
    spool
        .prepare_group(
            events("canonical", "private products"),
            &["root".into(), "part".into()],
        )
        .unwrap();
    spool
        .prepare(events(
            "independent-note",
            "authored target reference survives",
        ))
        .unwrap();
    let complete = vec!["root".into(), "part".into(), "canonical".into()];
    assert!(spool.discard_confirmed(&complete, 18, 17).is_err());
    assert!(spool.discard_confirmed(&["root".into()], 18, 18).is_err());
    assert!(
        spool
            .discard_confirmed(&["root".into(), "canonical".into()], 18, 18)
            .is_err()
    );
    assert_eq!(all(&spool, 2).len(), 4);
    spool.discard_confirmed(&complete, 18, 18).unwrap();
    spool.discard_confirmed(&complete, 18, 18).unwrap();
    assert_eq!(
        all(&spool, 1)
            .iter()
            .map(|entry| entry.identity.as_str())
            .collect::<Vec<_>>(),
        ["independent-note"]
    );
}

#[test]
fn confirmed_tombstone_outbox_can_retry_after_process_death() {
    let directory = tempfile::tempdir().unwrap();
    let status = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "spool::tests::crash_worker", "--nocapture"])
        .env("POLYLOGUE_SPOOL_CRASH_ROOT", directory.path())
        .env("POLYLOGUE_SPOOL_CRASH", "discard_member_retired")
        .status()
        .unwrap();
    assert_eq!(status.code(), Some(77));
    let spool = Spool::open(directory.path(), "host").unwrap();
    spool
        .discard_confirmed(&["private".into(), "doomed".into()], 12, 12)
        .unwrap();
    assert!(all(&spool, 1).is_empty());
}

#[test]
fn confirmed_tombstones_remove_unpublished_copies_left_after_body_sync() {
    let directory = tempfile::tempdir().unwrap();
    let status = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "spool::tests::crash_worker", "--nocapture"])
        .env("POLYLOGUE_SPOOL_CRASH_ROOT", directory.path())
        .env("POLYLOGUE_SPOOL_CRASH", "body_synced")
        .status()
        .unwrap();
    assert_eq!(status.code(), Some(77));
    let spool = Spool::open(directory.path(), "host").unwrap();
    assert!(spool.get("raw").unwrap().is_none());
    assert!(
        fs::read_dir(spool.root.join("operations"))
            .unwrap()
            .next()
            .is_some()
    );
    assert!(spool.discard_confirmed(&["raw".into()], 14, 13).is_err());
    spool.discard_confirmed(&["raw".into()], 14, 14).unwrap();
    assert!(
        fs::read_dir(spool.root.join("operations"))
            .unwrap()
            .next()
            .is_none()
    );
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
            operation: sinex_model::OperationId(uuid::Uuid::nil()),
            root: crate::material_id("raw"),
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
    assert_eq!(prepared.kind, Kind::ReplaySequence);
    assert_eq!(spool.decode_frame("replacement").unwrap(), replay);
    spool
        .acknowledge(ack("replacement", false, Some(21)))
        .unwrap();
    spool.retire(20).unwrap();
    assert!(spool.get("raw").unwrap().is_some());
    spool.retire(21).unwrap();
    assert!(all(&spool, 1).is_empty());
}

fn parsed_capture(bytes: &[u8], text: &str) -> Capture {
    let mut output = polylogue_model::Output::default();
    output.messages.push(serde_json::from_value(serde_json::json!({
        "offset":0,"end":bytes.len(),"message":{"id":"codex:message","session":"codex:session","role":"assistant","attrs":{},"blocks":[{"type":"text","text":text}]}
    })).unwrap());
    let step = polylogue_store::ParseStep::new(
        output,
        polylogue_model::RecordCounts::default(),
        0..bytes.len() as u64,
        true,
    );
    let parsed = vec![polylogue_store::primary::CapturedReading::from_step(&step).unwrap()];
    Capture::Input(serde_json::from_value(serde_json::json!({
        "input":{"origin":"codex","path":"/fixture/session.jsonl","host":"fixture-host"},
        "rows":{"input":"input-x","format":"codex","revision":{"number":1,"produced":"file","chain":1,"length":bytes.len()},"extent":{"chain":1,"offset":0,"length":bytes.len(),"blob":polylogue_model::BlobHash::of(bytes)}},
        "index":0,"previous":null,"hash":polylogue_model::BlobHash::of(bytes),"length":bytes.len(),"group":null,"arrival_rank":1,"parsed":parsed
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
    assert!(serde_json::to_vec(&prepared).unwrap().len() < 4096);
    assert!(prepared.captured.as_ref().unwrap().length > 1_000_000);
    assert_eq!(prepared.capture, Some(capture_header(&capture)));
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
    let path = spool.operation("captured").join("captured.json");
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
    spool
        .discard_confirmed(&["captured".into()], 10, 10)
        .unwrap();
    assert!(
        fs::read_dir(spool.root.join("operations"))
            .unwrap()
            .next()
            .is_none()
    );
}
