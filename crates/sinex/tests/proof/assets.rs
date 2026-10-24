//! Export attachments: one kept, one removed by excision before Sinex holds
//! the archive. Both outcomes survive a restore.
use super::*;
use polylogue_model::{BlobHash, MessageId};
use polylogue_store::assets::AttachmentState;

const KEPT: &[u8] = b"synthetic kept asset\0\xff";
const REMOVED: &[u8] = b"synthetic removed asset\0\xfe";
const MESSAGE: &str = "chatgpt:native-asset-message";

pub(super) fn seed(engine: &mut Engine, dir: &Path) {
    let source = dir.join("chatgpt-assets");
    fs::create_dir(&source).unwrap();
    let conversation = serde_json::json!([{"id":"native-asset-session","title":"Synthetic kept assets","mapping":{"n":{"id":"n","parent":null,"children":[],"message":{"id":"native-asset-message","author":{"role":"user"},"content":{"content_type":"multimodal_text","parts":[{"content_type":"image_asset_pointer","asset_pointer":"file-service://file-Native"},{"content_type":"image_asset_pointer","asset_pointer":"file-service://file-Removed"}]},"metadata":{}}}},"current_node":"n"}]);
    fs::write(
        source.join("conversations.json"),
        serde_json::to_vec(&conversation).unwrap(),
    )
    .unwrap();
    fs::write(source.join("file-Native.dat"), KEPT).unwrap();
    fs::write(source.join("file-Removed.dat"), REMOVED).unwrap();
    fs::write(
        source.join("conversation_asset_file_names.json"),
        br#"{"file-Native.dat":"native \u03bb\ud83d\ude42.dat"}"#,
    )
    .unwrap();
    let imported = engine.import(std::slice::from_ref(&source)).unwrap();
    assert!(
        imported.pass.errors.is_empty(),
        "{:?}",
        imported.pass.errors
    );
    fs::remove_dir_all(source).unwrap();
}

pub(super) fn remove(root: &Path, home: &Path) {
    converge(root, home, vec![], |engine| {
        engine
            .excise(polylogue_store::excise::ExciseRequest {
                id: "remove-native-asset".into(),
                scope: polylogue_store::excise::Scope::Input(BlobHash::of(REMOVED).to_string()),
                reason: None,
                dry_run: false,
            })
            .unwrap();
    });
}

pub(super) fn verify(root: &Path, roots: &[PathBuf]) -> serde_json::Value {
    let pool = ReadPool::open(&Archive::at(root)).unwrap();
    let reader = pool.reader().unwrap();
    let message = MessageId::parse(MESSAGE).unwrap();
    let kept = reader.attachment(&message, 0, 0).unwrap();
    assert_eq!(kept.state, AttachmentState::Available);
    assert_eq!(kept.candidates.len(), 1);
    let candidate = &kept.candidates[0];
    assert_eq!(candidate.names, ["native λ🙂.dat"]);
    assert_eq!(candidate.blob, Some(BlobHash::of(KEPT)));
    assert_eq!(
        reader
            .attachment_range(candidate, roots, 0..candidate.size)
            .unwrap(),
        KEPT
    );
    assert_eq!(
        kept.source.blob, None,
        "derived resolution cannot rewrite source Attachment"
    );
    let removed = reader.attachment(&message, 1, 0).unwrap();
    assert_eq!(removed.state, AttachmentState::Unavailable);
    assert!(removed.candidates.is_empty());
    serde_json::to_value(kept).unwrap()
}

pub(super) fn refuse_reimport(root: &Path, home: &Path, roots: Vec<PathBuf>, dir: &Path) {
    let path = dir.join("file-Removed.dat");
    fs::write(&path, REMOVED).unwrap();
    converge(root, home, roots, |engine| {
        let report = engine.import(std::slice::from_ref(&path)).unwrap();
        assert!(
            !report.pass.errors.is_empty(),
            "restored input refusal was lost"
        );
    });
    fs::remove_file(path).unwrap();
}
