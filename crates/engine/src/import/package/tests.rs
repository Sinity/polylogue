use super::*;
use std::io::Write;
const CHAT: &str =
    r#"[{"title":"Example","conversation_id":"one","current_node":"root","mapping":{}}]"#;

fn zip(path: &Path, files: &[(&str, &[u8])]) {
    let mut zip = zip::ZipWriter::new(std::fs::File::create(path).unwrap());
    for (name, body) in files {
        zip.start_file(*name, zip::write::SimpleFileOptions::default())
            .unwrap();
        zip.write_all(body).unwrap();
    }
    zip.finish().unwrap();
}

#[test]
fn import_every_structure_keeps_package_assets_and_refuses_unknown() {
    let scratch = tempfile::tempdir().unwrap();
    let source = scratch.path().join("export.zip");
    zip(
        &source,
        &[
            ("conversations.json", CHAT.as_bytes()),
            ("chat.html", b"<html>archive</html>"),
            ("asset.dat", b"asset"),
        ],
    );
    let members = stage(&source, &Archive::at(&scratch.path().join("encoded")), None).unwrap();
    let names: Vec<_> = members.iter().map(|m| m.origin.map(|o| o.name)).collect();
    assert_eq!(
        names,
        [
            Some("export-file"),
            Some("chatgpt"),
            Some("export-file"),
            Some("export-file")
        ]
    );
    let unknown = scratch.path().join("unknown.jsonl");
    std::fs::write(&unknown, "{\"type\":\"user\",\"message\":{}}\n").unwrap();
    assert!(
        stage(
            &unknown,
            &Archive::at(&scratch.path().join("encoded")),
            None
        )
        .unwrap()[0]
            .origin
            .is_none()
    );
    assert_eq!(
        stage(
            &unknown,
            &Archive::at(&scratch.path().join("encoded")),
            Some("claude-code")
        )
        .unwrap()[0]
            .origin
            .unwrap()
            .name,
        "claude-code"
    );
}

#[test]
fn nested_tar_and_compressed_tar_keep_originals_and_decoded_members() {
    let scratch = tempfile::tempdir().unwrap();
    let source = scratch.path().join("nested.zip");
    let mut tar = tar::Builder::new(Vec::new());
    let mut header = tar::Header::new_gnu();
    header.set_size(CHAT.len() as u64);
    header.set_mode(0o600);
    header.set_cksum();
    tar.append_data(&mut header, "conversations.json", CHAT.as_bytes())
        .unwrap();
    let bytes = tar.into_inner().unwrap();
    let zst = zstd::stream::encode_all(bytes.as_slice(), 1).unwrap();
    zip(
        &source,
        &[
            ("bundle/file_example-import.tar.zst", &zst),
            ("account/conversations.json", CHAT.as_bytes()),
        ],
    );
    let members = stage(&source, &Archive::at(&scratch.path().join("encoded")), None).unwrap();
    assert_eq!(members.len(), 4);
    assert_eq!(members[2].origin.unwrap().name, "chatgpt");
    assert_eq!(members[2].parent, Some((1, 0)));
}

#[test]
fn a_container_inside_its_own_bytes_is_refused() {
    let scratch = tempfile::tempdir().unwrap();
    let source = scratch.path().join("export.zip");
    zip(&source, &[("conversations.json", CHAT.as_bytes())]);
    let archive = Archive::at(&scratch.path().join("encoded"));
    let pending = prepare_source(source, &archive).unwrap();
    // As if this container had been reached through a chain of containers
    // that starts with these same bytes.
    let mut seen = Expansion {
        ancestors: vec![pending.member.bytes.hash()],
        ..Expansion::default()
    };
    let error = expand(pending, &archive, None, &mut seen, &mut Vec::new()).unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
}

#[test]
fn a_repeated_container_is_kept_once_and_expanded_once() {
    let scratch = tempfile::tempdir().unwrap();
    let inner = scratch.path().join("inner.zip");
    zip(&inner, &[("conversations.json", CHAT.as_bytes())]);
    let inner = std::fs::read(&inner).unwrap();
    let source = scratch.path().join("outer.zip");
    zip(&source, &[("a/inner.zip", &inner), ("b/inner.zip", &inner)]);
    let members = stage(&source, &Archive::at(&scratch.path().join("encoded")), None).unwrap();
    let containers = members.iter().filter(|member| member.container).count();
    let decoded = members.iter().filter(|member| !member.container).count();
    assert_eq!((containers, decoded), (3, 1));
    assert!(members.iter().all(|member| member.origin.is_some()));
}

#[test]
fn fits_in_staging_refuses_only_sizes_beyond_free_space() {
    let scratch = tempfile::tempdir().unwrap();
    let archive = Archive::at(&scratch.path().join("encoded"));
    assert!(fits_in_staging(&archive, 1).is_ok());
    assert_eq!(
        fits_in_staging(&archive, u64::MAX).unwrap_err().kind(),
        std::io::ErrorKind::StorageFull
    );
}

#[test]
fn fragmented_container_signatures_keep_the_exact_staged_bytes() {
    struct OneByte<'a>(&'a [u8]);
    impl Read for OneByte<'_> {
        fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
            let limit = buffer.len().min(1);
            self.0.read(&mut buffer[..limit])
        }
    }

    let scratch = tempfile::tempdir().unwrap();
    let archive = Archive::at(&scratch.path().join("encoded"));
    for (name, bytes, kind) in [
        ("empty", b"".as_slice(), None),
        ("one", b"P", None),
        ("two", b"PK", None),
        ("three", b"PK\x03", None),
        ("zip", b"PK\x03\x04tail", Some("zip")),
        ("gzip.tar.gz", b"\x1f\x8btail", Some("tar+gzip")),
        ("zstd.tar.zst", b"\x28\xb5\x2f\xfdtail", Some("tar+zstd")),
        ("plain.tar", b"plain", Some("tar")),
    ] {
        let pending = PendingMember::prepare(name.into(), &archive, OneByte(bytes)).unwrap();
        assert_eq!(pending.format.unwrap().map(Container::kind), kind);
        assert_eq!(
            pending.member.bytes.hash(),
            polylogue_model::BlobHash::of(bytes)
        );
        let mut decoded = Vec::new();
        pending
            .member
            .bytes
            .reader()
            .unwrap()
            .read_to_end(&mut decoded)
            .unwrap();
        assert_eq!(decoded, bytes);
    }

    let mut source = OneByte(b"PK\x03\x04");
    let pending =
        PendingMember::prepare("truncated.zip".into(), &archive, source.by_ref().take(3)).unwrap();
    assert_eq!(source.0, b"\x04");
    assert_eq!(pending.member.bytes.len(), 3);
    assert_eq!(
        pending.format.err().unwrap().kind(),
        std::io::ErrorKind::InvalidData
    );
}

#[test]
fn a_zip_member_larger_than_its_declared_size_is_refused() {
    let scratch = tempfile::tempdir().unwrap();
    let source = scratch.path().join("export.zip");
    zip(&source, &[("conversations.json", CHAT.as_bytes())]);

    let mut bytes = std::fs::read(&source).unwrap();
    let central = bytes
        .windows(4)
        .position(|prefix| prefix == b"PK\x01\x02")
        .unwrap();
    bytes[22..26].copy_from_slice(&2u32.to_le_bytes());
    bytes[central + 24..central + 28].copy_from_slice(&2u32.to_le_bytes());
    std::fs::write(&source, bytes).unwrap();

    let archive = Archive::at(&scratch.path().join("encoded"));
    let Err(error) = stage(&source, &archive, None) else {
        panic!("member larger than its declaration was accepted");
    };
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    assert!(error.to_string().contains("different from its declared 2"));
}

#[test]
fn a_corrupt_member_refuses_the_whole_container() {
    let scratch = tempfile::tempdir().unwrap();
    let source = scratch.path().join("broken.zip");
    zip(
        &source,
        &[
            ("conversations.json", CHAT.as_bytes()),
            ("broken.tar.zst", b"not a compressed tar"),
        ],
    );
    assert!(stage(&source, &Archive::at(&scratch.path().join("encoded")), None).is_err());
}

#[test]
fn takeout_routes_only_declared_products() {
    let scratch = tempfile::tempdir().unwrap();
    let source = scratch.path().join("takeout.zip");
    zip(
        &source,
        &[
            ("Takeout/NotebookLM/data.bin", b"bytes"),
            ("Takeout/Mail/message.json", CHAT.as_bytes()),
        ],
    );
    let members = stage(&source, &Archive::at(&scratch.path().join("encoded")), None).unwrap();
    assert_eq!(members[0].origin.unwrap().name, "export-file");
    assert_eq!(members[1].origin.unwrap().name, "export-file");
    assert!(members[2].origin.is_none());
}

#[test]
fn structure_checks_field_types_and_reads_past_long_values() {
    let scratch = tempfile::tempdir().unwrap();
    let source = scratch.path().join("one.json");
    std::fs::write(
        &source,
        br#"{"mapping":"not a tree","current_node":"one","conversation_id":"one"}"#,
    )
    .unwrap();
    assert!(
        stage(&source, &Archive::at(&scratch.path().join("encoded")), None).unwrap()[0]
            .origin
            .is_none()
    );
    std::fs::write(
        &source,
        format!(
            r#"{{"title":"{}","mapping":{{}},"current_node":null,"conversation_id":"one"}}"#,
            "a".repeat(128 * 1024)
        ),
    )
    .unwrap();
    assert_eq!(
        stage(&source, &Archive::at(&scratch.path().join("encoded")), None).unwrap()[0]
            .origin
            .unwrap()
            .name,
        "chatgpt"
    );
    std::fs::write(&source, br#"{"uuid":"one","chat_messages":[]}"#).unwrap();
    assert_eq!(
        stage(&source, &Archive::at(&scratch.path().join("encoded")), None).unwrap()[0]
            .origin
            .unwrap()
            .name,
        "claude-ai"
    );
}

#[test]
fn sibling_packages_route_only_their_declared_origin() {
    let scratch = tempfile::tempdir().unwrap();
    let source = scratch.path().join("mixed.zip");
    zip(
        &source,
        &[
            ("chatgpt/conversations.json", CHAT.as_bytes()),
            (
                "chatgpt/foreign.json",
                br#"{"uuid":"foreign","chat_messages":[]}"#,
            ),
            (
                "claude/conversations.json",
                br#"[{"uuid":"claude","chat_messages":[]}]"#,
            ),
        ],
    );
    let members = stage(&source, &Archive::at(&scratch.path().join("encoded")), None).unwrap();
    assert_eq!(members[1].origin.unwrap().name, "chatgpt");
    assert_eq!(members[2].origin.unwrap().name, "export-file");
    assert_eq!(members[3].origin.unwrap().name, "claude-ai");
}

#[test]
fn aider_requires_its_declared_pair_and_codex_cloud_keeps_its_namespace() {
    let scratch = tempfile::tempdir().unwrap();
    let source = scratch.path().join("pair.zip");
    zip(
        &source,
        &[(
            ".aider.chat.history.md",
            b"# aider chat started at 2026-01-01 10:00:00\n",
        )],
    );
    assert!(
        stage(&source, &Archive::at(&scratch.path().join("encoded")), None)
            .unwrap()
            .iter()
            .all(|member| member.origin.is_none())
    );
    zip(&source,&[(".aider.llm.history",b"TO LLM 2026-01-01T10:00:00\n-------\nUSER hello\nLLM RESPONSE 2026-01-01T10:00:01\nASSISTANT hi\n"),(".aider.chat.history.md",b"# aider chat started at 2026-01-01 10:00:00\n")]);
    let members = stage(&source, &Archive::at(&scratch.path().join("encoded")), None).unwrap();
    assert_eq!(members[1].origin.unwrap().name, "aider");
    assert_eq!(members[2].origin.unwrap().name, "aider");
    let cloud = scratch.path().join("codex.json");
    std::fs::write(&cloud,br#"[{"id":"task_e_one","turns":[{"id":"task_e_one~usertrn_e_one","role":"user","input_items":[]}]}]"#).unwrap();
    assert_eq!(
        stage(&cloud, &Archive::at(&scratch.path().join("encoded")), None).unwrap()[0]
            .origin
            .unwrap()
            .name,
        "codex-cloud"
    );
}

#[cfg(unix)]
#[test]
fn non_utf8_container_names_keep_distinct_original_bytes() {
    use std::os::unix::ffi::OsStringExt;
    let scratch = tempfile::tempdir().unwrap();
    let source = scratch.path().join("names.tar");
    let mut tar = tar::Builder::new(std::fs::File::create(&source).unwrap());
    for (name, bytes) in [
        (b"conversations.json".to_vec(), CHAT.as_bytes()),
        (vec![b'a', 0xff], b"one".as_slice()),
        (vec![b'a', 0xfe], b"two".as_slice()),
    ] {
        let mut header = tar::Header::new_gnu();
        header.set_size(bytes.len() as u64);
        header.set_mode(0o600);
        header.set_cksum();
        tar.append_data(
            &mut header,
            PathBuf::from(std::ffi::OsString::from_vec(name)),
            bytes,
        )
        .unwrap();
    }
    tar.finish().unwrap();
    drop(tar);
    let members = stage(&source, &Archive::at(&scratch.path().join("encoded")), None).unwrap();
    assert_eq!(
        members[2].member_path_bytes.as_deref(),
        Some([b'a', 0xff].as_slice())
    );
    assert_eq!(
        members[3].member_path_bytes.as_deref(),
        Some([b'a', 0xfe].as_slice())
    );
    assert_ne!(members[2].path, members[3].path);
}

#[test]
fn member_names_are_never_filesystem_paths() {
    let root = tempfile::tempdir().unwrap();
    let scratch = root.path().join("scratch");
    std::fs::create_dir(&scratch).unwrap();
    let marker = root.path().join("asset.bin");
    std::fs::write(&marker, b"untouched").unwrap();
    let source = root.path().join("names.zip");
    zip(
        &source,
        &[
            ("../conversations.json", CHAT.as_bytes()),
            ("../asset.bin", b"archive asset"),
            ("/absolute/conversations.json", CHAT.as_bytes()),
        ],
    );
    let members = stage(&source, &Archive::at(&scratch), None).unwrap();
    assert_eq!(members.len(), 4);
    assert_eq!(
        members[1].member_path_bytes.as_deref(),
        Some(b"../conversations.json".as_slice())
    );
    assert_eq!(
        members[2].member_path_bytes.as_deref(),
        Some(b"../asset.bin".as_slice())
    );
    assert_eq!(
        members[3].member_path_bytes.as_deref(),
        Some(b"/absolute/conversations.json".as_slice())
    );
    assert_eq!(std::fs::read(&marker).unwrap(), b"untouched");
    assert!(!root.path().join("conversations.json").exists());
    assert!(!Path::new("/absolute/conversations.json").exists());
    let source = root.path().join("names.tar");
    let mut tar = tar::Builder::new(std::fs::File::create(&source).unwrap());
    for name in ["../conversations.json", "/absolute/conversations.json"] {
        let mut header = tar::Header::new_gnu();
        header.as_mut_bytes()[..name.len()].copy_from_slice(name.as_bytes());
        header.set_size(CHAT.len() as u64);
        header.set_mode(0o600);
        header.set_cksum();
        tar.append(&header, CHAT.as_bytes()).unwrap();
    }
    tar.finish().unwrap();
    drop(tar);
    let members = stage(&source, &Archive::at(&scratch), None).unwrap();
    assert_eq!(members.len(), 3);
    assert_eq!(
        members[1].member_path_bytes.as_deref(),
        Some(b"../conversations.json".as_slice())
    );
    assert_eq!(
        members[2].member_path_bytes.as_deref(),
        Some(b"/absolute/conversations.json".as_slice())
    );
    assert_eq!(std::fs::read(&marker).unwrap(), b"untouched");
    assert!(!root.path().join("conversations.json").exists());
    assert!(!Path::new("/absolute/conversations.json").exists());
}

#[test]
fn package_preflight_releases_refused_members() {
    use std::io::Read;
    let root = tempfile::tempdir().unwrap();
    let archive = Archive::at(&root.path().join("archive"));
    let source = root.path().join("export.zip");
    let asset = vec![b'q'; 4 * 1024 * 1024];
    zip(
        &source,
        &[
            ("conversations.json", CHAT.as_bytes()),
            ("asset.bin", &asset),
        ],
    );
    let members = stage(&source, &archive, None).unwrap();
    assert_eq!(members.len(), 3);
    let mut restored = Vec::new();
    members[2]
        .bytes
        .reader()
        .unwrap()
        .read_to_end(&mut restored)
        .unwrap();
    assert_eq!(restored, asset);
    // Every named private file is an encoded preparation. No expanded asset or
    // tar stream is written as a second plaintext scratch representation.
    let parts = archive.staging_dir();
    let files: Vec<_> = std::fs::read_dir(&parts).unwrap().collect();
    assert_eq!(files.len(), 3);
    for file in files {
        let file = std::fs::File::open(file.unwrap().path()).unwrap();
        let prepared = hashed_blobs::Prepared::inspect(file, None, Default::default()).unwrap();
        assert!(prepared.stored_len() < 64 * 1024);
    }
    drop(members);
    assert_eq!(std::fs::read_dir(&parts).unwrap().count(), 0);
    let broken = root.path().join("broken.zip");
    std::fs::write(&broken, b"PK\x03\x04broken synthetic archive").unwrap();
    assert!(stage(&broken, &archive, None).is_err());
    assert_eq!(std::fs::read_dir(&parts).unwrap().count(), 0);
}

/// Run only by `package_import_respects_a_bounded_descriptor_window`.
#[test]
#[ignore = "child process of package_import_respects_a_bounded_descriptor_window"]
#[allow(unsafe_code)]
fn descriptor_window_child() {
    if std::env::var_os("POLYLOGUE_PACKAGE_FD_CHILD").is_none() {
        return;
    }
    // This resource limit applies only to the synthetic child process.
    let limit = libc::rlimit {
        rlim_cur: 128,
        rlim_max: 128,
    };
    assert_eq!(unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &limit) }, 0);
    let home = polylogue_testkit::ClaudeHome::new();
    let source = home.dir.path().join("many.zip");
    let mut zip = zip::ZipWriter::new(std::fs::File::create(&source).unwrap());
    zip.start_file(
        "conversations.json",
        zip::write::SimpleFileOptions::default(),
    )
    .unwrap();
    zip.write_all(CHAT.as_bytes()).unwrap();
    for index in 0..512 {
        zip.start_file(
            format!("asset-{index}.txt"),
            zip::write::SimpleFileOptions::default(),
        )
        .unwrap();
        write!(zip, "synthetic asset {index}").unwrap();
    }
    zip.finish().unwrap();
    let (mut engine, _lock) = crate::import::tests::engine(&home, 1024);
    let report = engine.import(&[source]).unwrap();
    assert!(report.refused.is_empty(), "{:?}", report.refused);
    assert!(report.pass.errors.is_empty(), "{:?}", report.pass.errors);
    assert_eq!(report.imported.len(), 514);
}

#[test]
fn package_import_respects_a_bounded_descriptor_window() {
    let result = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "import::package::tests::descriptor_window_child",
            "--ignored",
            "--nocapture",
        ])
        .env("POLYLOGUE_PACKAGE_FD_CHILD", "1")
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "descriptor child failed: {}\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    // A renamed child would match no test and still exit successfully.
    assert!(
        String::from_utf8_lossy(&result.stdout).contains("1 passed"),
        "descriptor child did not run: {}",
        String::from_utf8_lossy(&result.stdout)
    );
}

mod declarations;
mod projects;
