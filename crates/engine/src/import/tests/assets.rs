use super::*;
use polylogue_testkit::ClaudeHome;

fn conversation() -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!([{
        "id":"asset-session", "title":"Attachment example", "mapping": {
            "node": {"id":"node","parent":null,"children":[],"message":{
                "id":"asset-message","author":{"role":"user"},"create_time":1700000000,
                "content":{"content_type":"multimodal_text","parts":[{
                    "content_type":"image_asset_pointer","asset_pointer":"file-service://file-Native"
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
            engine.converge().unwrap().errors
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
        let hash=reader.query("SELECT hex(ms.version_hash) AS hash FROM message m JOIN message_source ms ON ms.message_pk=m.pk AND ms.preferred=1 WHERE m.id='chatgpt:asset-message'",&[]).unwrap();
        if let Some(previous) = &previous {
            assert_eq!(previous, &hash);
        } else {
            previous = Some(hash);
        }
    }
}

#[test]
fn identical_conversations_in_conflicting_acquisitions_do_not_pick_an_asset() {
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
fn directory_import_keeps_attachment_bytes_after_provider_removal() {
    let w = ClaudeHome::new();
    let directory = w.dir.path().join("account-export");
    std::fs::create_dir_all(&directory).unwrap();
    let bytes = b"synthetic stored attachment\0\xff";
    let path = directory.join("file-native.dat");
    std::fs::write(&path, bytes).unwrap();
    let (mut engine, _lock) = engine(&w, 1024);
    let report = engine.import(&[directory]).unwrap();
    assert!(report.pass.errors.is_empty(), "{:?}", report.pass.errors);
    std::fs::remove_file(&path).unwrap();
    let reader = engine.reads.reader().unwrap();
    let id = blake3::hash(bytes).to_hex().to_string();
    let progress = reader.input_progress(&id).unwrap();
    assert!(
        progress.is_some(),
        "directory attachment bytes were not stored"
    );
    let progress = progress.unwrap();
    assert_eq!(
        reader
            .stored_range(&id, progress.chain, 0, progress.length, &engine.blobs)
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
    zip.finish().unwrap();
    let (mut engine, _lock) = engine(&w, 1024);
    let report = engine.import(&[path]).unwrap();
    assert!(report.pass.errors.is_empty(), "{:?}", report.pass.errors);
    assert_eq!(w.count("SELECT count(*) AS n FROM a.input"), 2);
    assert_eq!(w.count("SELECT count(*) AS n FROM a.observation"), 3);
    assert_eq!(
        w.count("SELECT count(*) AS n FROM a.observation_container"),
        2
    );
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
                    .any(|error| error.contains("asset sidecar"))
            );
        } else {
            assert!(report.pass.errors.is_empty(), "{:?}", report.pass.errors);
        }
    }
    let reader = engine.reads.reader().unwrap();
    let resolution = reader.attachment(&message_id(), 0, 0).unwrap();
    assert_eq!(resolution.matches.len(), 3);
    for asset in resolution.matches {
        if asset.blob == Some(polylogue_model::BlobHash::of(b"first")) {
            assert_eq!(asset.names, ["native-name.png"]);
        } else {
            assert!(
                asset.names.is_empty(),
                "a filename role from another acquisition must not be borrowed"
            );
        }
    }
}

#[test]
fn excised_asset_bytes_stay_unavailable_after_rebuild_and_reimport() {
    let w = ClaudeHome::new();
    let directory = w.dir.path().join("account");
    std::fs::create_dir_all(&directory).unwrap();
    std::fs::write(directory.join("conversations.json"), conversation()).unwrap();
    std::fs::write(
        directory.join("file-Native.dat"),
        b"remove these asset bytes",
    )
    .unwrap();
    {
        let (mut engine, _lock) = engine(&w, 1024);
        let report = engine.import(std::slice::from_ref(&directory)).unwrap();
        assert!(report.pass.errors.is_empty(), "{:?}", report.pass.errors);
        let input = blake3::hash(b"remove these asset bytes")
            .to_hex()
            .to_string();
        engine
            .excise(ExciseRequest {
                id: "remove-asset".into(),
                scope: ExciseScope::Input(input),
                reason: None,
                dry_run: false,
            })
            .unwrap();
        let resolution = engine
            .reads
            .reader()
            .unwrap()
            .attachment(&message_id(), 0, 0)
            .unwrap();
        assert_eq!(
            resolution.state,
            polylogue_store::assets::AttachmentState::Unavailable
        );
        assert!(resolution.matches.is_empty());
    }
    polylogue_testkit::delete_index(&w.root);
    let (mut engine, _lock) = engine(&w, 1024);
    let pass = engine.converge().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    for reimport in [false, true] {
        if reimport {
            let _ = engine.import(std::slice::from_ref(&directory)).unwrap();
        }
        let reader = engine.reads.reader().unwrap();
        let resolution = reader.attachment(&message_id(), 0, 0).unwrap();
        assert_eq!(
            resolution.state,
            polylogue_store::assets::AttachmentState::Unavailable
        );
        assert!(resolution.matches.is_empty());
    }
}

#[test]
fn text_excision_refuses_opaque_assets_without_changing_their_stored_bytes() {
    let w = ClaudeHome::new();
    let source = w.dir.path().join("asset-source");
    std::fs::create_dir(&source).unwrap();
    std::fs::write(source.join("conversations.json"), conversation()).unwrap();
    let secret = "synthetic-sensitive-asset-content";
    let original = format!("before {secret} after");
    std::fs::write(source.join("file-Native.dat"), original.as_bytes()).unwrap();
    let mut current = None;
    for rebuild in [false, true] {
        if rebuild {
            std::fs::remove_dir_all(&source).unwrap();
            polylogue_testkit::delete_index(&w.root);
        }
        let (mut engine, _lock) = engine(&w, 1024);
        if !rebuild {
            assert!(
                engine
                    .import(std::slice::from_ref(&source))
                    .unwrap()
                    .pass
                    .errors
                    .is_empty()
            );
            let before = engine
                .reads
                .reader()
                .unwrap()
                .attachment(&message_id(), 0, 0)
                .unwrap();
            let error = engine
                .excise(ExciseRequest {
                    id: "remove-asset-text".into(),
                    scope: ExciseScope::Text(secret.into()),
                    reason: None,
                    dry_run: false,
                })
                .unwrap_err();
            assert!(
                matches!(
                    error,
                    EngineError::Store(polylogue_store::StoreError::ExcisionFormat { .. })
                ),
                "{error}"
            );
            let reader = engine.reads.reader().unwrap();
            assert_eq!(
                reader
                    .attachment_range(&before.matches[0], &[], 0..before.matches[0].size)
                    .unwrap(),
                original.as_bytes()
            );
        } else {
            assert!(engine.converge().unwrap().errors.is_empty());
        }
        let reader = engine.reads.reader().unwrap();
        let resolution = reader.attachment(&message_id(), 0, 0).unwrap();
        assert_eq!(
            resolution.state,
            polylogue_store::assets::AttachmentState::Available
        );
        let asset = &resolution.matches[0];
        let bytes = reader.attachment_range(asset, &[], 0..asset.size).unwrap();
        assert_eq!(bytes, original.as_bytes());
        assert_eq!(
            asset.blob,
            Some(polylogue_model::BlobHash::of(original.as_bytes()))
        );
        if let Some((binding, expected)) = &current {
            assert_eq!(&resolution.binding, binding);
            assert_eq!(&bytes, expected);
        } else {
            current = Some((resolution.binding, bytes));
        }
    }
}
