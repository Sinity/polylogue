use super::*;
use polylogue_model::BlobHash;
use serde::Deserialize;

#[derive(Deserialize)]
struct Oracle {
    source_commit: String,
    source_fingerprint: String,
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    name: String,
    material: Ingress,
    body: String,
    events: Ingress,
    dependencies: Vec<String>,
}

fn oracle() -> Oracle {
    crate::spool::decode_json(include_bytes!("oracle.json").as_slice()).unwrap()
}
fn custody(case: &Case) -> (QueuedPublicationCustody, &MaterialFrame) {
    let Ingress::Material { frame, material } = &case.material else {
        panic!("original material")
    };
    (
        QueuedPublicationCustody {
            producer_family: "polylogue".into(),
            producer_identity: "polylogue-test".into(),
            receipt_number: 11,
            frame: frame.clone(),
            frame_content_hash: BlobHash::of(&serde_json::to_vec(&case.material).unwrap()),
            material: material.id.to_string(),
            blob: BlobHash::of(case.body.as_bytes()),
            length: material.length,
            source: material.source.name.clone(),
            source_version: material.source.version,
            source_fingerprint: crate::client::confirmed::source3::source_fingerprint().into(),
            parser: material.parser.clone(),
            parser_version: material.parser_version,
        },
        material,
    )
}

#[test]
fn private_continuation_matches_executed_source3_request_and_restore_encoder_bytes() {
    // Produced by the actual immutable 16708 encoders, not by this successor.
    // The qualification run used synthetic receipts solely to exercise codecs.
    let oracle = oracle();
    assert_eq!(
        oracle.source_commit,
        "16708b50bc860330f39f9869761e2ca5a1e8fad2"
    );
    assert_eq!(
        oracle.source_fingerprint,
        crate::client::confirmed::source3::source_fingerprint()
    );
    let mut matched = 0;
    for case in &oracle.cases {
        let (custody, material) = custody(case);
        if case.name == "attachments" {
            assert!(build(&custody, material, case.body.as_bytes()).is_err());
            continue;
        }
        let actual = build(&custody, material, case.body.as_bytes()).unwrap();
        assert_eq!(
            serde_json::to_vec(&actual).unwrap(),
            serde_json::to_vec(&case.events).unwrap(),
            "{}",
            case.name
        );
        assert_eq!(case.dependencies, [custody.frame]);
        matched += 1;
    }
    assert_eq!(matched, 11);
}

#[test]
fn continuation_stages_original_source3_group_and_exact_retry_reuses_it_after_reopen() {
    let root = tempfile::tempdir().unwrap();
    let case = oracle().cases.remove(0);
    let (custody, material) = custody(&case);
    let expected = build(&custody, material, case.body.as_bytes()).unwrap();
    {
        let client = crate::tests::client(root.path());
        let queue = client.queue().unwrap();
        queue
            .prepare_material(case.material.clone(), case.body.as_bytes())
            .unwrap();
        queue
            .prepare_group(expected.clone(), &case.dependencies)
            .unwrap();
    }
    let client = crate::tests::client(root.path());
    let queue = client.queue().unwrap();
    let replay = queue
        .prepare_group(expected.clone(), &case.dependencies)
        .unwrap();
    assert_eq!(queue.decode_frame(&replay.identity).unwrap(), expected);
    assert_eq!(queue.pending_after(None, 64).unwrap().len(), 2);
    let mut changed = expected;
    if let Ingress::Events { events, .. } = &mut changed {
        events[0].payload["machine"] = "current-host".into();
    }
    assert!(queue.prepare_group(changed, &case.dependencies).is_err());
    let mut retained = Vec::new();
    std::io::BufReader::new(queue.body(&custody.frame).unwrap())
        .read_to_end(&mut retained)
        .unwrap();
    assert_eq!(retained, case.body.as_bytes());
}

struct Calendar;
impl ::signals::Calendar for Calendar {
    fn logical_day(&self, _: i64) -> String {
        "synthetic-day".into()
    }
    fn minute(&self, _: i64) -> u16 {
        0
    }
    fn next_at(&self, after: i64, _: u16) -> i64 {
        after + 1
    }
}

#[test]
fn retained_original_bytes_cannot_bypass_the_existing_derived_signal_publication_guard() {
    let mut case = oracle().cases.remove(0);
    let checkpoint = polylogue_model::signal::Checkpoint {
        owner: "fixture-machine".into(),
        evaluator: ::signals::Evaluator::new(0, vec![], ::signals::Policy::default(), &Calendar)
            .unwrap(),
        rules: vec![],
        rule_sources: Default::default(),
        rule_revisions: Default::default(),
        host: None,
        notifications: None,
        outlet_attempts: Default::default(),
    };
    let mut body: Value = crate::spool::decode_json(case.body.as_bytes()).unwrap();
    body["effect"]["signal_checkpoints"] = serde_json::to_value([checkpoint.clone()]).unwrap();
    body["request"]["write"] = serde_json::to_value(
        polylogue_model::user::UserWrite::SignalCommit(Box::new(polylogue_model::signal::Commit {
            prior: None,
            checkpoint,
            outbox: vec![],
            events: vec![],
        })),
    )
    .unwrap();
    case.body = serde_json::to_string(&body).unwrap();
    if let Ingress::Material { material, .. } = &mut case.material {
        material.blob = format!("blake3:{}", BlobHash::of(case.body.as_bytes()));
        material.length = case.body.len() as u64;
    }
    let (custody, material) = custody(&case);
    assert!(
        matches!(build(&custody,material,case.body.as_bytes()),Err(Error::Refused{reason,..}) if reason.contains("derived admission"))
    );
}
