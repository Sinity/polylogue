use super::*;
use crate::archive::{cache, open_write};
use crate::encoding::pack_json;

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
    let row = crate::encoding::RowPacker::new()
        .unwrap()
        .pack_json(&message)
        .unwrap();
    assert!(zstd::zstd_safe::get_dict_id_from_frame(&row).is_some());
    assert!(
        !row.windows(b"metadata-only-marker".len())
            .any(|w| w == b"metadata-only-marker")
    );
    let db = open_write(&archive.index_db(), cache::SETUP).unwrap();
    db.execute("INSERT INTO input_ref(pk,id) VALUES (1,'input')", [])
        .unwrap();
    for bytes in [row.clone(), pack_json(&message).unwrap()] {
        db.execute(
            "INSERT OR REPLACE INTO event(pk,id,seq,kind,input_pk,data) VALUES (1,'event',0,'synthetic',1,?1)",
            [bytes],
        )
        .unwrap();
        let columns = index_columns_containing(&archive, "metadata-only-marker").unwrap();
        assert!(columns.contains(&"event.data".to_owned()), "{columns:?}");
        assert!(
            index_columns_containing(&archive, "absent marker")
                .unwrap()
                .is_empty()
        );
    }
    db.execute("UPDATE event SET id='binary-marker'", [])
        .unwrap();
    assert!(
        index_columns_containing(&archive, "binary-marker")
            .unwrap()
            .contains(&"event.id".to_owned())
    );
    db.execute("UPDATE event SET data=?1", [&row[..row.len() - 1]])
        .unwrap();
    assert!(matches!(
        index_columns_containing(&archive, "metadata-only-marker"),
        Err(StoreError::Io(_))
    ));
}
