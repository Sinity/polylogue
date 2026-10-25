use super::*;
use polylogue_testkit::ClaudeHome;
use std::io::Write;

fn archive(w: &ClaudeHome) -> (PathBuf, Vec<u8>) {
    let path = w.dir.path().join("export.zip");
    let mut zip = zip::ZipWriter::new(std::fs::File::create(&path).unwrap());
    zip.set_comment("synthetic original container metadata");
    for (name, body) in [
        ("file-one.dat", b"first member".as_slice()),
        ("file-two.dat", b"second member".as_slice()),
    ] {
        zip.start_file(
            name,
            zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored),
        )
        .unwrap();
        zip.write_all(body).unwrap();
    }
    zip.finish().unwrap();
    let bytes = std::fs::read(&path).unwrap();
    (path, bytes)
}

#[test]
fn original_zip_bytes_survive_provider_removal_and_rebuild() {
    let w = ClaudeHome::new();
    let (path, original) = archive(&w);
    let id = blake3::hash(&original).to_hex().to_string();
    for rebuild in [false, true] {
        if rebuild {
            std::fs::remove_file(&path).unwrap();
            polylogue_testkit::delete_index(&w.root);
        }
        let (mut engine, _lock) = engine(&w, 1024);
        if rebuild {
            assert!(engine.converge().unwrap().errors.is_empty());
        } else {
            assert!(
                engine
                    .import(std::slice::from_ref(&path))
                    .unwrap()
                    .pass
                    .errors
                    .is_empty()
            );
        }
        let reader = engine.reads.reader().unwrap();
        let stored = reader
            .input_progress(&id)
            .unwrap()
            .expect("original ZIP input must be stored");
        let bytes = reader
            .stored_range(&id, stored.chain, 0, stored.length, &engine.blobs)
            .unwrap();
        assert_eq!(
            bytes, original,
            "regenerating equal members does not preserve original ZIP metadata"
        );
    }
}

#[test]
fn a_refused_zip_member_prevents_committing_the_zip_and_its_siblings() {
    let w = ClaudeHome::new();
    let source = w.dir.path().join("file-one.dat");
    std::fs::write(&source, b"first member").unwrap();
    let (path, original) = archive(&w);
    let (mut engine, _lock) = engine(&w, 1024);
    assert!(engine.import(&[source]).unwrap().pass.errors.is_empty());
    let member = blake3::hash(b"first member").to_hex().to_string();
    engine
        .excise(ExciseRequest {
            id: "refuse-before-zip".into(),
            scope: ExciseScope::Input(member),
            reason: None,
            dry_run: false,
        })
        .unwrap();
    let observations = w.count("SELECT count(*) AS n FROM a.observation");
    let report = engine.import(&[path]).unwrap();
    assert!(
        report
            .pass
            .errors
            .iter()
            .any(|error| error.contains("excluded")),
        "{report:?}"
    );
    assert_eq!(
        w.count("SELECT count(*) AS n FROM a.observation"),
        observations,
        "a refused ZIP stores none of its members"
    );
    assert!(
        !engine
            .blobs
            .contains(&hashed_blobs::Hash(*blake3::hash(&original).as_bytes()))
    );
    assert!(!engine.blobs.contains(&hashed_blobs::Hash(
        *blake3::hash(b"second member").as_bytes()
    )));
}

#[test]
fn unreadable_or_encrypted_zip_population_refuses_before_any_public_install() {
    for encrypted in [false, true] {
        let w = ClaudeHome::new();
        let (path, mut original) = archive(&w);
        if encrypted {
            // The declared traditional-encryption flag requires a password.
            original[6] |= 1;
            let directory = original
                .windows(4)
                .position(|bytes| bytes == b"PK\x01\x02")
                .unwrap();
            original[directory + 8] |= 1;
        } else {
            let start = original
                .windows(b"second member".len())
                .position(|bytes| bytes == b"second member")
                .unwrap();
            original[start] ^= 1; // preserve the declared CRC, making this member unreadable.
        }
        std::fs::write(&path, &original).unwrap();
        let (mut engine, _lock) = engine(&w, 1024);
        assert!(engine.import(&[path]).is_err());
        for table in [
            "input",
            "revision",
            "extent",
            "blob",
            "observation",
            "observation_container",
        ] {
            assert_eq!(
                w.count(&format!("SELECT count(*) AS n FROM a.{table}")),
                0,
                "{table}"
            );
        }
        assert!(
            !engine
                .blobs
                .contains(&hashed_blobs::Hash(*blake3::hash(&original).as_bytes()))
        );
        assert!(!engine.blobs.contains(&hashed_blobs::Hash(
            *blake3::hash(b"first member").as_bytes()
        )));
    }
}

#[test]
fn session_and_document_removal_remove_the_zip_while_an_unrelated_input_still_goes() {
    use std::io::Write;
    let w = ClaudeHome::new();
    let path = w.dir.path().join("scoped.zip");
    let mut output = zip::ZipWriter::new(std::fs::File::create(&path).unwrap());
    for (name, bytes) in [
        (
            "conversation.jsonl",
            polylogue_testkit::claude_user(
                "user",
                "zip-session",
                "synthetic stored conversation",
            )
            .into_bytes(),
        ),
        (
            "memories.json",
            br#"[{"account_uuid":"zip-account","conversations_memory":"supplied memory"}]"#
                .to_vec(),
        ),
    ] {
        output
            .start_file(name, zip::write::SimpleFileOptions::default())
            .unwrap();
        output.write_all(&bytes).unwrap();
    }
    output.finish().unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let (mut e, _lock) = engine(&w, 1024);
    let report = e.import(&[path]).unwrap();
    assert!(report.pass.errors.is_empty(), "{:?}", report.pass.errors);
    let doc = e
        .reads
        .reader()
        .unwrap()
        .query("SELECT id FROM document", &[])
        .unwrap()[0]["id"]
        .as_str()
        .unwrap()
        .to_owned();
    let original_id = blake3::hash(&bytes).to_hex().to_string();
    for (index, scope) in [
        ExciseScope::Session(polylogue_model::SessionId::parse("claude-code:zip-session").unwrap()),
        ExciseScope::Document(polylogue_model::DocumentId::parse(&doc).unwrap()),
    ]
    .into_iter()
    .enumerate()
    {
        let removed = e
            .excise(ExciseRequest {
                id: format!("remove-scope-{index}"),
                scope,
                reason: None,
                dry_run: false,
            })
            .unwrap();
        if index == 0 {
            assert_eq!(
                removed.zips_removed.as_slice(),
                std::slice::from_ref(&original_id)
            );
        }
        assert!(
            e.reads
                .reader()
                .unwrap()
                .input_progress(&original_id)
                .unwrap()
                .is_none()
        );
    }
    let before = w.count("SELECT count(*) AS n FROM a.extent");
    let unrelated = w.dir.path().join("unrelated.dat");
    std::fs::write(&unrelated, b"unrelated stored bytes").unwrap();
    assert!(e.import(&[unrelated]).unwrap().pass.errors.is_empty());
    e.excise(ExciseRequest {
        id: "unrelated-removal".into(),
        scope: ExciseScope::Input(
            blake3::hash(b"unrelated stored bytes")
                .to_hex()
                .to_string(),
        ),
        reason: None,
        dry_run: false,
    })
    .unwrap();
    assert_eq!(w.count("SELECT count(*) AS n FROM a.extent"), before);
}

#[test]
fn original_bytes_and_ordinal_preserve_raw_names_separately_from_decoded_projection() {
    let w = ClaudeHome::new();
    let (path, mut original) = archive(&w);
    original[7] &= !8;
    original[30] = 0xff;
    let central = original
        .windows(4)
        .position(|bytes| bytes == b"PK\x01\x02")
        .unwrap();
    original[central + 9] &= !8;
    original[central + 46] = 0xff;
    std::fs::write(&path, &original).unwrap();
    let (mut e, _lock) = engine(&w, 1024);
    assert!(e.import(&[path]).unwrap().pass.errors.is_empty());
    let reader = e.reads.reader().unwrap();
    let id = blake3::hash(&original).to_hex().to_string();
    let input = reader.input_progress(&id).unwrap().unwrap();
    let stored = reader
        .stored_range(&id, input.chain, 0, input.length, &e.blobs)
        .unwrap();
    assert_eq!(stored, original);
    let mut container = zip::ZipArchive::new(std::io::Cursor::new(stored)).unwrap();
    let member = container.by_index(0).unwrap();
    assert_eq!(member.name_raw()[0], 0xff);
    assert!(std::str::from_utf8(member.name_raw()).is_err());
    let names=reader.query("SELECT json_extract(o.details_json,'$.import.member') AS name FROM a.observation_container c JOIN a.observation o ON o.id=c.observation_id WHERE c.member_index=0",&[]).unwrap();
    assert_eq!(
        names[0]["name"],
        member.name(),
        "the scalar is a decoded display projection, not raw name bytes"
    );
}

#[test]
fn identical_and_changed_containers_keep_acquisitions_while_sharing_member_bytes() {
    let w = ClaudeHome::new();
    let (path, original) = archive(&w);
    let other = w.dir.path().join("second.zip");
    let (mut e, _lock) = engine(&w, 1024);
    assert!(e.import(&[path]).unwrap().pass.errors.is_empty());
    std::fs::write(&other, &original).unwrap();
    assert!(
        e.import(std::slice::from_ref(&other))
            .unwrap()
            .pass
            .errors
            .is_empty()
    );
    let mut changed = original;
    *changed.last_mut().unwrap() ^= 1;
    std::fs::write(&other, changed).unwrap();
    assert!(e.import(&[other]).unwrap().pass.errors.is_empty());
    assert_eq!(w.count("SELECT count(*) AS n FROM a.input"), 4);
    assert_eq!(w.count("SELECT count(*) AS n FROM a.revision"), 4);
    assert_eq!(w.count("SELECT count(*) AS n FROM a.observation"), 9);
    assert_eq!(
        w.count("SELECT count(*) AS n FROM a.observation_container"),
        6
    );
    assert_eq!(
        w.count("SELECT count(DISTINCT container_input) AS n FROM a.observation_container"),
        2
    );
    assert_eq!(w.count("SELECT count(DISTINCT json_extract(o.details_json,'$.import.acquisition')) AS n FROM a.observation_container c JOIN a.observation o ON o.id=c.observation_id"),3);
}

mod excision;
