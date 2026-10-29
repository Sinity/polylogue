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
    let members = stage(&source, scratch.path(), None).unwrap();
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
        stage(&unknown, scratch.path(), None).unwrap()[0]
            .origin
            .is_none()
    );
    assert_eq!(
        stage(&unknown, scratch.path(), Some("claude-code")).unwrap()[0]
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
    zip(&source, &[("export.tar.zst", &zst)]);
    let members = stage(&source, scratch.path(), None).unwrap();
    assert_eq!(members.len(), 3);
    assert_eq!(members[2].origin.unwrap().name, "chatgpt");
    assert_eq!(members[2].parent, Some((1, 0)));
}

#[test]
fn corrupt_member_refuses_container_before_staging_completes() {
    let scratch = tempfile::tempdir().unwrap();
    let source = scratch.path().join("broken.zip");
    zip(
        &source,
        &[
            ("conversations.json", CHAT.as_bytes()),
            ("broken.tar.zst", b"not a compressed tar"),
        ],
    );
    assert!(stage(&source, scratch.path(), None).is_err());
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
    let members = stage(&source, scratch.path(), None).unwrap();
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
        stage(&source, scratch.path(), None).unwrap()[0]
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
        stage(&source, scratch.path(), None).unwrap()[0]
            .origin
            .unwrap()
            .name,
        "chatgpt"
    );
    std::fs::write(&source, br#"{"uuid":"one","chat_messages":[]}"#).unwrap();
    assert_eq!(
        stage(&source, scratch.path(), None).unwrap()[0]
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
    let members = stage(&source, scratch.path(), None).unwrap();
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
        stage(&source, scratch.path(), None)
            .unwrap()
            .iter()
            .all(|member| member.origin.is_none())
    );
    zip(&source,&[(".aider.llm.history",b"TO LLM 2026-01-01T10:00:00\n-------\nUSER hello\nLLM RESPONSE 2026-01-01T10:00:01\nASSISTANT hi\n"),(".aider.chat.history.md",b"# aider chat started at 2026-01-01 10:00:00\n")]);
    let members = stage(&source, scratch.path(), None).unwrap();
    assert_eq!(members[1].origin.unwrap().name, "aider");
    assert_eq!(members[2].origin.unwrap().name, "aider");
    let cloud = scratch.path().join("codex.json");
    std::fs::write(&cloud,br#"[{"id":"task_e_one","turns":[{"id":"task_e_one~usertrn_e_one","role":"user","input_items":[]}]}]"#).unwrap();
    assert_eq!(
        stage(&cloud, scratch.path(), None).unwrap()[0]
            .origin
            .unwrap()
            .name,
        "codex-cloud"
    );
}
