use super::*;

fn frame(identity: &str, bytes: &str) -> Ingress {
    Ingress::Chunk {
        frame: identity.into(),
        for_frame: "material".into(),
        blob: "blake3:synthetic".into(),
        offset: 0,
        total: bytes.len() as u64,
        data: bytes.into(),
    }
}

#[test]
fn interrupted_sends_keep_original_identity_and_contents_after_restart() {
    let directory = tempfile::tempdir().unwrap();
    let original = frame("first-write", "synthetic bytes");
    let spool = Spool::open(directory.path(), "host-a").unwrap();
    spool.prepare(original.clone()).unwrap();
    drop(spool);
    let spool = Spool::open(directory.path(), "host-a").unwrap();
    let prepared = spool.prepared().unwrap();
    assert_eq!(prepared.len(), 1);
    assert_eq!(
        serde_json::to_value(&prepared[0].frame).unwrap(),
        serde_json::to_value(original).unwrap()
    );
    assert!(prepared[0].acknowledgment.is_none());
    assert!(spool.prepare(frame("first-write", "changed")).is_err());
    spool
        .prepare(frame("later-write", "synthetic bytes"))
        .unwrap();
    assert_eq!(spool.prepared().unwrap().len(), 2);
    assert!(
        Spool::open(directory.path(), "host-b")
            .unwrap()
            .prepared()
            .unwrap()
            .is_empty()
    );
}

#[test]
fn acknowledgment_survives_restart_until_consumer_commits_its_position() {
    let directory = tempfile::tempdir().unwrap();
    let spool = Spool::open(directory.path(), "host-a").unwrap();
    spool.prepare(frame("write", "bytes")).unwrap();
    assert!(spool.settle("write", 99).is_err());
    spool
        .acknowledge(Acknowledgment {
            frame: "write".into(),
            material: None,
            material_extents: vec![],
            admitted: vec![],
            outcome: serde_json::json!({}),
            refusal: None,
            commit_seq: Some(18),
        })
        .unwrap();
    drop(spool);
    let spool = Spool::open(directory.path(), "host-a").unwrap();
    assert!(spool.settle("write", 17).is_err());
    assert_eq!(spool.prepared().unwrap().len(), 1);
    spool.settle("write", 18).unwrap();
    assert!(spool.prepared().unwrap().is_empty());
}
