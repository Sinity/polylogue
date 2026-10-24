use super::*;

#[test]
fn document_policy_round_trips_as_owned_excision_rows() {
    let source = Connection::open_in_memory().unwrap();
    crate::schema::create_archive(&source).unwrap();
    let mut publication = ExcisionPublication {
        version: 1,
        replacements: vec![],
        retired_batches: vec![],
        digest: String::new(),
    };
    publication.digest = publication.computed_digest().unwrap();
    source.execute("INSERT INTO excision(id,scope,refused_json,created_at,commit_seq) VALUES ('remove','document:claude-ai:doc:gone',?1,1,1)",[json!({"sessions":[],"publication":publication}).to_string()]).unwrap();
    source
        .execute(
            "INSERT INTO excision_document VALUES ('remove','claude-ai:doc:gone')",
            [],
        )
        .unwrap();
    let unit = Unit {
        commit_seq: 1,
        kind: "excision".into(),
        key: "remove".into(),
    };
    let batch = read(&source, &unit).unwrap();
    let bytes = batch.contents();
    let row = bytes
        .split(|b| *b == b'\n')
        .filter(|s| !s.is_empty())
        .map(|s| serde_json::from_slice::<Value>(s).unwrap())
        .find(|v| v["table"] == "excision_document")
        .unwrap();
    assert_eq!(row["row"]["document_id"], "claude-ai:doc:gone");
    let mut target = Connection::open_in_memory().unwrap();
    crate::schema::create_archive(&target).unwrap();
    let descriptor = RestoreUnit {
        unit,
        batch: Some(format!("blake3:{}", blake3::hash(&bytes))),
        bytes: Some(bytes.len() as u64),
        rows: batch.rows,
        blobs: vec![],
        excision: Some(publication),
    };
    let selection = select(&[descriptor]).unwrap();
    restore::install_fixture(&mut target, &selection, None, |_| {
        Ok(Some(std::io::Cursor::new(bytes.clone())))
    })
    .unwrap();
    assert_eq!(
        target
            .query_row("SELECT document_id FROM excision_document", [], |r| r
                .get::<_, String>(0))
            .unwrap(),
        "claude-ai:doc:gone"
    );
    let refused = crate::excise::Refused::load(&target).unwrap();
    assert!(
        refused
            .matching_documents(&["claude-ai:doc:gone".into(), "claude-ai:doc:kept".into()])
            .unwrap()
            .contains("claude-ai:doc:gone")
    );
    let before = crate::excise::recovery::current(&target).unwrap();
    target
        .execute(
            "UPDATE excision_document SET document_id='claude-ai:doc:changed'",
            [],
        )
        .unwrap();
    assert_ne!(crate::excise::recovery::current(&target).unwrap(), before);
}
