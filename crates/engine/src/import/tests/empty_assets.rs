use super::*;
use polylogue_model::{BlobHash, MessageId};
use polylogue_store::Batch;
use polylogue_store::assets::{AssetMatch, AttachmentState, ObservedMember};
use polylogue_store::primary::{AcquisitionRoute, RawRange};
use polylogue_testkit::ClaudeHome;
use std::io::{Read, Seek, SeekFrom, Write};

const MEMBER: &str = "file-Empty.dat";

fn message() -> MessageId {
    MessageId::parse("chatgpt:empty-message").unwrap()
}

fn conversation() -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!([{
        "id":"empty-session", "current_node":"node", "mapping":{
            "node":{"id":"node","parent":null,"children":[],"message":{
                "id":"empty-message","author":{"role":"user"},
                "content":{"content_type":"multimodal_text","parts":[{
                    "content_type":"image_asset_pointer",
                    "asset_pointer":"file-service://file-Empty"
                }]}
            }}
        }
    }]))
    .unwrap()
}

fn package(home: &ClaudeHome, zipped: bool, empty: bool) -> PathBuf {
    if zipped {
        let path = home.dir.path().join("account.zip");
        let mut zip = zip::ZipWriter::new(std::fs::File::create(&path).unwrap());
        zip.start_file(
            "conversations.json",
            zip::write::SimpleFileOptions::default(),
        )
        .unwrap();
        zip.write_all(&conversation()).unwrap();
        if empty {
            zip.start_file(MEMBER, zip::write::SimpleFileOptions::default())
                .unwrap();
        }
        zip.finish().unwrap();
        path
    } else {
        let path = home.dir.path().join("account");
        std::fs::create_dir_all(&path).unwrap();
        std::fs::write(path.join("conversations.json"), conversation()).unwrap();
        if empty {
            std::fs::write(path.join(MEMBER), b"").unwrap();
        }
        path
    }
}

fn import(engine: &mut Engine, path: &Path) {
    let report = engine.import(&[path.to_path_buf()]).unwrap();
    assert!(report.refused.is_empty(), "{report:?}");
    assert!(report.pass.errors.is_empty(), "{report:?}");
}

fn observed(reader: &polylogue_store::Reader<'_>, member: &str) -> ObservedMember {
    let rows = reader.query(
        "SELECT id,input,details_json FROM source_observation WHERE json_extract(details_json,'$.import.member')=?1",
        &[polylogue_store::SqlValue::Text(member.into())],
    ).unwrap();
    assert_eq!(rows.len(), 1);
    let details: serde_json::Value =
        serde_json::from_str(rows[0]["details_json"].as_str().unwrap()).unwrap();
    ObservedMember {
        observation: rows[0]["id"].as_str().unwrap().into(),
        input: rows[0]["input"].as_str().unwrap().into(),
        membership: serde_json::from_value(details["import"].clone()).unwrap(),
    }
}

fn available(engine: &Engine) -> (AssetMatch, ObservedMember) {
    let reader = engine.reads.reader().unwrap();
    let selected = reader.attachment(&message(), 0, 0).unwrap();
    assert_eq!(selected.state, AttachmentState::Available);
    assert_eq!(selected.matches.len(), 1);
    assert_eq!(selected.source.blob, None);
    let asset = selected.matches.into_iter().next().unwrap();
    assert_eq!(asset.size, 0);
    assert_eq!(asset.blob, Some(BlobHash::of(b"")));
    assert_eq!(reader.attachment_hash(&asset).unwrap(), BlobHash::of(b""));
    assert!(reader.attachment_range(&asset, 0..0).unwrap().is_empty());
    assert!(reader.attachment_range(&asset, 0..1).is_err());
    let member = observed(&reader, MEMBER);
    let witness = reader
        .validate_attachment_member(&member, BlobHash::of(b""), 0)
        .unwrap();
    assert_eq!(witness.identity().hash, BlobHash::of(b""));
    assert_eq!(witness.identity().size, 0);
    (asset, member)
}

#[test]
fn real_directory_and_zip_empty_members_survive_provider_removal_reopen_and_rebuild() {
    for zipped in [false, true] {
        let home = ClaudeHome::new();
        let provider = package(&home, zipped, true);
        let expected = {
            let (mut engine, _lock) = engine(&home, 1024);
            import(&mut engine, &provider);
            available(&engine).0
        };
        if zipped {
            std::fs::remove_file(&provider).unwrap();
        } else {
            std::fs::remove_dir_all(&provider).unwrap();
        }
        for rebuild in [false, true] {
            if rebuild {
                polylogue_testkit::delete_index(&home.root);
            }
            let (mut engine, _lock) = engine(&home, 1024);
            let pass = engine.run_pass().unwrap();
            assert!(pass.errors.is_empty(), "{pass:?}");
            assert_eq!(available(&engine).0, expected);
            assert_eq!(
                home.count("SELECT count(*) AS n FROM a.extent WHERE length=0"),
                0
            );
        }
    }
}

#[test]
fn missing_or_corrupt_empty_frames_refuse_hash_range_and_receiving_witness() {
    for missing in [false, true] {
        let home = ClaudeHome::new();
        let provider = package(&home, false, true);
        let (mut engine, _lock) = engine(&home, 1024);
        import(&mut engine, &provider);
        let (asset, member) = available(&engine);
        let extent = engine
            .blobs
            .extent(&hashed_blobs::Hash(BlobHash::of(b"").0))
            .unwrap();
        assert!(extent.len > 0);
        let path = engine.blobs.root().join(&extent.segment);
        let held = path.with_extension("held-empty");
        let mut original = [0];
        let offset = extent.offset + extent.len - 1;
        if missing {
            std::fs::rename(&path, &held).unwrap();
        } else {
            let mut file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(&path)
                .unwrap();
            file.seek(SeekFrom::Start(offset)).unwrap();
            file.read_exact(&mut original).unwrap();
            file.seek(SeekFrom::Start(offset)).unwrap();
            file.write_all(&[original[0] ^ 1]).unwrap();
            file.sync_all().unwrap();
        }
        let rejected = {
            let reader = engine.reads.reader().unwrap();
            [
                reader.attachment_hash(&asset).is_err(),
                reader.attachment_range(&asset, 0..0).is_err(),
                reader
                    .validate_attachment_member(&member, BlobHash::of(b""), 0)
                    .is_err(),
            ]
        };
        // Restore this synthetic physical fault before assertions or teardown.
        if missing {
            std::fs::rename(&held, &path).unwrap();
        } else {
            let mut file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
            file.seek(SeekFrom::Start(offset)).unwrap();
            file.write_all(&original).unwrap();
            file.sync_all().unwrap();
        }
        assert!(rejected.into_iter().all(|rejected| rejected));
        assert_eq!(available(&engine).0, asset);
    }
}

fn unrelated_empty_blob(engine: &Engine) {
    let prepared = engine.archive.prepare_bytes(b"".as_slice()).unwrap();
    engine
        .writer
        .store_blob(
            polylogue_store::NewBlob {
                hash: prepared.hash(),
                size: prepared.len(),
            },
            Some(prepared),
        )
        .unwrap();
}

fn unavailable(engine: &Engine) {
    let reader = engine.reads.reader().unwrap();
    let resolution = reader.attachment(&message(), 0, 0).unwrap();
    assert_eq!(resolution.state, AttachmentState::Unavailable);
    assert!(resolution.matches.is_empty());
    assert!(
        reader
            .asset_source(&BlobHash::of(b"").to_string())
            .unwrap()
            .is_none()
    );
}

#[test]
fn unrelated_empty_blob_does_not_supply_a_missing_import_member() {
    let home = ClaudeHome::new();
    let provider = package(&home, false, false);
    let (mut engine, _lock) = engine(&home, 1024);
    import(&mut engine, &provider);
    unrelated_empty_blob(&engine);
    unavailable(&engine);
}

#[test]
fn empty_member_namespaces_must_match_the_actual_receiving_observation() {
    let home = ClaudeHome::new();
    let provider = package(&home, false, true);
    let (mut engine, _lock) = engine(&home, 1024);
    import(&mut engine, &provider);
    let (_, member) = available(&engine);
    let reader = engine.reads.reader().unwrap();
    for change_namespace in [false, true] {
        let mut forged = member.clone();
        if change_namespace {
            forged.membership.import_id = "unrelated-namespace".into();
        } else {
            forged.membership.member = "different-empty.dat".into();
        }
        assert!(
            reader
                .validate_attachment_member(&forged, BlobHash::of(b""), 0)
                .is_err()
        );
    }
}

#[test]
fn metadata_only_and_mismatched_receipts_cannot_borrow_an_unrelated_empty_body() {
    for invalid in ["absent", "machine", "path", "range", "hash"] {
        let home = ClaudeHome::new();
        let provider = package(&home, false, false);
        let (mut engine, _lock) = engine(&home, 1024);
        import(&mut engine, &provider);
        unrelated_empty_blob(&engine);
        let namespace = observed(&engine.reads.reader().unwrap(), "conversations.json")
            .membership
            .import_id;
        let path = home.dir.path().join(MEMBER);
        std::fs::write(&path, b"").unwrap();
        let prepared = engine
            .archive
            .prepare_bytes(std::fs::File::open(&path).unwrap())
            .unwrap();
        let mut observation = engine.observation(&path, None);
        observation.import = Some(polylogue_store::ImportMember {
            version: 1,
            import_id: namespace,
            member: MEMBER.into(),
            authorship: None,
            format: Some("export-file".into()),
        });
        let mut receipt = engine
            .acquisition_context(&path, "export-file", AcquisitionRoute::ExplicitImport)
            .finish(RawRange { start: 0, end: 0 }, prepared.hash());
        match invalid {
            "machine" => receipt.source.machine = "another-machine".into(),
            "path" => receipt.source.path_bytes = b"/synthetic/another-empty".to_vec(),
            "range" => receipt.range = RawRange { start: 1, end: 1 },
            "hash" => receipt.hash = BlobHash::of(b"different"),
            _ => {}
        }
        let result = engine.writer.store_bytes(
            prepared,
            Batch {
                required_disposition: None,
                input: InputKey {
                    id: BlobHash::of(b"").to_string(),
                    format: "export-file".into(),
                    route: polylogue_model::Route::Export,
                },
                change: Change::First,
                observation: Some(observation),
                continues: false,
                new_blob: None,
                blobs: vec![],
                length: 0,
                parse: vec![],
                acquisition: (invalid != "absent").then_some(receipt),
                basis: None,
                release: Default::default(),
            },
        );
        if matches!(invalid, "machine" | "path") {
            assert!(result.is_err(), "mismatched {invalid} receipt was accepted");
        } else if result.is_ok() {
            engine.refresh_import_assets(&mut Pass::default()).unwrap();
            let reader = engine.reads.reader().unwrap();
            let member = observed(&reader, MEMBER);
            assert!(
                reader
                    .validate_attachment_member(&member, BlobHash::of(b""), 0)
                    .is_err()
            );
        }
        unavailable(&engine);
    }
}

#[test]
fn saved_empty_selection_refuses_a_real_nonempty_source_rewrite() {
    let home = ClaudeHome::new();
    let provider = package(&home, false, true);
    let (mut engine, _lock) = engine(&home, 1024);
    import(&mut engine, &provider);
    let (asset, _) = available(&engine);
    let path = home.dir.path().join("actual-rewrite.dat");
    std::fs::write(&path, b"replacement bytes").unwrap();
    let prepared = engine
        .archive
        .prepare_bytes(std::fs::File::open(&path).unwrap())
        .unwrap();
    engine
        .writer
        .store_bytes(
            prepared,
            Batch {
                required_disposition: None,
                input: InputKey {
                    id: asset.input.clone(),
                    format: "export-file".into(),
                    route: polylogue_model::Route::Live,
                },
                change: Change::Rewritten,
                observation: Some(engine.observation(&path, None)),
                continues: false,
                new_blob: None,
                blobs: vec![],
                length: b"replacement bytes".len() as u64,
                parse: vec![],
                acquisition: None,
                basis: None,
                release: Default::default(),
            },
        )
        .unwrap();
    let reader = engine.reads.reader().unwrap();
    assert_ne!(
        reader.input_progress(&asset.input).unwrap().unwrap().chain,
        asset.chain
    );
    assert!(reader.attachment_hash(&asset).is_err());
    assert!(reader.attachment_range(&asset, 0..0).is_err());
}
