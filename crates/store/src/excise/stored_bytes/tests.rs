use super::*;
use crate::archive::{cache, open_write};
use crate::encoding::{PreparedMessage, pack_json};

#[test]
fn inspection_finds_dictionary_metadata_and_refuses_damaged_frames() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = crate::Archive::open_or_create(dir.path()).unwrap();
    let message: polylogue_model::Message = serde_json::from_value(serde_json::json!({
        "id":"codex:sample-42", "session":"codex:sample-session",
        "parent":"codex:sample-parent", "role":"assistant", "model":"metadata-only-marker",
        "prompt_kind":"model", "blocks":[], "usage":{}, "stop_reason":"end_turn"
    }))
    .unwrap();
    let prepared = PreparedMessage::new(&message).unwrap();
    assert!(zstd::zstd_safe::get_dict_id_from_frame(&prepared.attrs).is_some());
    assert!(
        !prepared
            .attrs
            .windows(b"metadata-only-marker".len())
            .any(|w| w == b"metadata-only-marker")
    );
    let db = open_write(&archive.index_db(), cache::SETUP).unwrap();
    for bytes in [prepared.attrs.clone(), pack_json(&message).unwrap()] {
        db.execute(
            "INSERT OR REPLACE INTO message_version VALUES (?1,?2)",
            params![&prepared.hash[..], bytes],
        )
        .unwrap();
        let columns = index_columns_containing(&archive, "metadata-only-marker").unwrap();
        assert!(
            columns.contains(&"message_version.attrs".to_owned()),
            "{columns:?}"
        );
        assert!(
            index_columns_containing(&archive, "absent marker")
                .unwrap()
                .is_empty()
        );
    }
    let mut opaque = [0xff_u8; 32];
    opaque[..b"binary-marker".len()].copy_from_slice(b"binary-marker");
    db.execute("UPDATE message_version SET hash=?1", [&opaque[..]])
        .unwrap();
    assert!(
        index_columns_containing(&archive, "binary-marker")
            .unwrap()
            .contains(&"message_version.hash".to_owned())
    );
    db.execute(
        "UPDATE message_version SET attrs=?1",
        [&prepared.attrs[..prepared.attrs.len() - 1]],
    )
    .unwrap();
    assert!(matches!(
        index_columns_containing(&archive, "metadata-only-marker"),
        Err(StoreError::Io(_))
    ));
}
