use super::*;
use polylogue_testkit::ClaudeHome;

fn conversation() -> Vec<u8> {
    conversation_with("asset-session", "asset-message", "file-Native")
}

fn conversation_with(session: &str, message: &str, file: &str) -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!([{
        "id":session, "title":"Attachment example", "mapping": {
            "node": {"id":"node","parent":null,"children":[],"message":{
                "id":message,"author":{"role":"user"},"create_time":1700000000,
                "content":{"content_type":"multimodal_text","parts":[{
                    "content_type":"image_asset_pointer","asset_pointer":format!("file-service://{file}")
                }]},"metadata":{}
            }}
        },"current_node":"node"
    }])).unwrap()
}

fn message_id() -> polylogue_model::MessageId {
    polylogue_model::MessageId::parse("chatgpt:asset-message").unwrap()
}

#[test]
fn source_scoped_assets_survive_provider_removal_and_rebuild_without_changing_messages() {
    let w = ClaudeHome::new();
    let directory = w.dir.path().join("account");
    std::fs::create_dir_all(&directory).unwrap();
    std::fs::write(directory.join("conversations.json"), conversation()).unwrap();
    std::fs::write(directory.join("file-Native.dat"), b"native bytes").unwrap();
    std::fs::write(directory.join("notes-file-Native.dat"), b"decoy bytes").unwrap();
    std::fs::write(directory.join("named.png"), b"basename decoy").unwrap();
    std::fs::write(
        directory.join("conversation_asset_file_names.json"),
        br#"{"file-Native.dat":"named.png"}"#,
    )
    .unwrap();
    let mut previous = None;
    for rebuild in [false, true] {
        if rebuild {
            std::fs::remove_dir_all(&directory).unwrap();
            polylogue_testkit::delete_index(&w.root);
        }
        let (mut engine, _lock) = engine(&w, 1024);
        let errors = if rebuild {
            engine.run_pass().unwrap().errors
        } else {
            engine
                .import(std::slice::from_ref(&directory))
                .unwrap()
                .pass
                .errors
        };
        assert!(errors.is_empty(), "{errors:?}");
        let reader = engine.reads.reader().unwrap();
        let resolution = reader.attachment(&message_id(), 0, 0).unwrap();
        assert_eq!(
            resolution.state,
            polylogue_store::assets::AttachmentState::Available
        );
        assert_eq!(resolution.matches.len(), 1);
        assert_eq!(resolution.matches[0].names, ["named.png"]);
        assert_eq!(
            resolution.matches[0].blob,
            Some(polylogue_model::BlobHash::of(b"native bytes"))
        );
        assert_eq!(
            resolution.source.blob, None,
            "source Attachment must remain unchanged"
        );
        let hash=reader.query("SELECT hex(m.version_hash) AS hash FROM message m WHERE m.id='chatgpt:asset-message'",&[]).unwrap();
        if let Some(previous) = &previous {
            assert_eq!(previous, &hash);
        } else {
            previous = Some(hash);
        }
    }
}

#[test]
fn identical_conversations_in_conflicting_imports_do_not_pick_an_asset() {
    let w = ClaudeHome::new();
    let (mut engine, _lock) = engine(&w, 1024);
    for (directory, bytes) in [("one", b"first".as_slice()), ("two", b"second".as_slice())] {
        let directory = w.dir.path().join(directory);
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join("conversations.json"), conversation()).unwrap();
        std::fs::write(directory.join("file-Native.dat"), bytes).unwrap();
        let report = engine.import(&[directory]).unwrap();
        assert!(report.pass.errors.is_empty(), "{:?}", report.pass.errors);
    }
    let reader = engine.reads.reader().unwrap();
    let resolution = reader.attachment(&message_id(), 0, 0).unwrap();
    assert_eq!(
        resolution.state,
        polylogue_store::assets::AttachmentState::Ambiguous
    );
    assert_eq!(resolution.matches.len(), 2);
    assert_eq!(w.count("SELECT count(*) AS n FROM message"), 1);
}

#[test]
fn directory_import_keeps_attachment_bytes_after_the_source_file_is_removed() {
    let w = ClaudeHome::new();
    let directory = w.dir.path().join("account-export");
    std::fs::create_dir_all(&directory).unwrap();
    let bytes = b"synthetic stored attachment\0\xff";
    let path = directory.join("file-native.dat");
    std::fs::write(&path, bytes).unwrap();
    std::fs::write(directory.join("conversations.json"), conversation()).unwrap();
    let (mut engine, _lock) = engine(&w, 1024);
    let report = engine.import(&[directory]).unwrap();
    assert!(report.pass.errors.is_empty(), "{:?}", report.pass.errors);
    std::fs::remove_file(&path).unwrap();
    let reader = engine.reads.reader().unwrap();
    let id = blake3::hash(bytes).to_hex().to_string();
    let progress = reader.latest_chain(&id).unwrap();
    assert!(
        progress.is_some(),
        "directory attachment bytes were not stored"
    );
    let progress = progress.unwrap();
    assert_eq!(
        reader
            .stored_range(&id, progress.chain, 0, progress.stored, &engine.blobs)
            .unwrap(),
        bytes
    );
}

#[test]
fn duplicate_zip_member_content_keeps_each_member_association() {
    use std::io::Write;
    let w = ClaudeHome::new();
    let path = w.dir.path().join("export.zip");
    let mut zip = zip::ZipWriter::new(std::fs::File::create(&path).unwrap());
    for name in ["file-first.dat", "file-second.dat"] {
        zip.start_file(name, zip::write::SimpleFileOptions::default())
            .unwrap();
        zip.write_all(b"same stored bytes").unwrap();
    }
    zip.start_file(
        "conversations.json",
        zip::write::SimpleFileOptions::default(),
    )
    .unwrap();
    zip.write_all(&conversation()).unwrap();
    zip.finish().unwrap();
    let (mut engine, _lock) = engine(&w, 1024);
    let report = engine.import(&[path]).unwrap();
    assert!(report.pass.errors.is_empty(), "{:?}", report.pass.errors);
    assert_eq!(w.count("SELECT count(*) AS n FROM a.input"), 3);
    assert_eq!(w.count("SELECT count(*) AS n FROM a.observation"), 4);
    assert_eq!(
        w.count("SELECT count(*) AS n FROM a.observation_container"),
        3
    );
    // Each member keeps its own zip index and name, though two share bytes.
    for (index, name) in [
        (0, "file-first.dat"),
        (1, "file-second.dat"),
        (2, "conversations.json"),
    ] {
        assert_eq!(
            w.count(&format!(
                "SELECT count(*) AS n FROM a.observation_container \
                 WHERE member_position = {index} AND path_bytes = CAST('{name}' AS BLOB)"
            )),
            1,
            "{name}"
        );
    }
}

#[test]
fn sharded_export_keeps_an_archive_attachment_without_importing_its_contents() {
    use std::io::Write;

    let w = ClaudeHome::new();
    let export = w.dir.path().join("export.zip");
    let attachment = "file_Native-example.tar.zst";
    let first = conversation_with("asset-session", "asset-message", "file_Native");
    let second = conversation_with("other-session", "other-message", "file_Other");
    let inside = conversation_with("nested-session", "nested-message", "file_Inside");
    let mut tar = tar::Builder::new(Vec::new());
    for (name, bytes) in [
        ("generated/table.csv", b"x,y\n1,2\n".as_slice()),
        ("other/conversations.json", inside.as_slice()),
    ] {
        let mut header = tar::Header::new_gnu();
        header.set_size(bytes.len() as u64);
        header.set_mode(0o600);
        header.set_cksum();
        tar.append_data(&mut header, name, bytes).unwrap();
    }
    let bytes = zstd::stream::encode_all(tar.into_inner().unwrap().as_slice(), 1).unwrap();
    let mut zip = zip::ZipWriter::new(std::fs::File::create(&export).unwrap());
    for (name, body) in [
        (attachment, bytes.as_slice()),
        ("conversations-000.json", first.as_slice()),
        ("conversations-001.json", second.as_slice()),
    ] {
        zip.start_file(name, zip::write::SimpleFileOptions::default())
            .unwrap();
        zip.write_all(body).unwrap();
    }
    zip.finish().unwrap();

    let (mut engine, _lock) = engine(&w, 1024);
    let report = engine.import(std::slice::from_ref(&export)).unwrap();
    assert!(report.refused.is_empty(), "{:?}", report.refused);
    assert!(report.pass.errors.is_empty(), "{:?}", report.pass.errors);
    assert_eq!(report.members.len(), 4);
    assert_eq!(w.count("SELECT count(*) AS n FROM message"), 2);
    let retry = engine.import(&[export]).unwrap();
    assert!(retry.refused.is_empty(), "{:?}", retry.refused);
    assert!(retry.imported.is_empty());
    assert_eq!(retry.already_stored.len(), 4);
    assert_eq!(w.count("SELECT count(*) AS n FROM message"), 2);

    let reader = engine.reads.reader().unwrap();
    let resolution = reader.attachment(&message_id(), 0, 0).unwrap();
    assert_eq!(
        resolution.state,
        polylogue_store::assets::AttachmentState::Available
    );
    assert_eq!(resolution.matches.len(), 1);
    let hash = resolution.matches[0].blob.unwrap();
    let input = hash.to_string();
    let progress = reader.latest_chain(&input).unwrap().unwrap();
    assert_eq!(
        reader
            .stored_range(&input, progress.chain, 0, progress.stored, &engine.blobs)
            .unwrap(),
        bytes
    );
    drop(reader);

    let path = w.dir.path().join(attachment);
    std::fs::write(&path, bytes).unwrap();
    let explicit = engine.import(&[path]).unwrap();
    assert_eq!(explicit.refused.len(), 1);
    assert!(explicit.refused[0].path.ends_with("generated/table.csv"));
    assert!(
        explicit.pass.errors.is_empty(),
        "{:?}",
        explicit.pass.errors
    );
    assert_eq!(w.count("SELECT count(*) AS n FROM message"), 3);
}

#[test]
fn sidecar_parsing_is_scoped_to_its_declared_role_in_each_import() {
    let w = ClaudeHome::new();
    let (mut engine, _lock) = engine(&w, 1024);
    let names = br#"{"file-Native.dat":"native-name.png"}"#;
    for (directory, sidecar, bytes) in [
        (
            "one",
            "conversation_asset_file_names.json",
            b"first".as_slice(),
        ),
        ("two", "unparsed.json", b"second".as_slice()),
        ("three", "library_files.json", b"third".as_slice()),
    ] {
        let directory = w.dir.path().join(directory);
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join("conversations.json"), conversation()).unwrap();
        std::fs::write(directory.join("file-Native.dat"), bytes).unwrap();
        std::fs::write(directory.join(sidecar), names).unwrap();
        let report = engine.import(&[directory]).unwrap();
        if sidecar == "library_files.json" {
            assert!(
                report
                    .pass
                    .errors
                    .iter()
                    .any(|error| error.to_string().contains("asset sidecar"))
            );
        } else {
            assert!(report.pass.errors.is_empty(), "{:?}", report.pass.errors);
        }
    }
    let reader = engine.reads.reader().unwrap();
    let resolution = reader.attachment(&message_id(), 0, 0).unwrap();
    assert_eq!(resolution.matches.len(), 3);
    let first_blob = Some(polylogue_model::BlobHash::of(b"first"));
    let first = resolution
        .matches
        .iter()
        .find(|asset| asset.blob == first_blob)
        .expect("the first import's asset");
    assert_eq!(first.names, ["native-name.png"]);
    for asset in resolution.matches.iter().filter(|a| a.blob != first_blob) {
        assert!(
            asset.names.is_empty(),
            "a filename role from another import must not be borrowed"
        );
    }
}
