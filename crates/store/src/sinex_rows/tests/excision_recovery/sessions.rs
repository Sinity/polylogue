use super::*;
use crate::StoreError;

#[test]
fn invalid_removed_session_ids_refuse_restore_before_rows_and_checkpoint_commit() {
    let dir = tempfile::tempdir().unwrap();
    let (source, _source_lock) = Archive::open_or_create(&dir.path().join("source")).unwrap();
    let conn = open_write(&source.archive_db(), cache::SETUP).unwrap();
    let mut publication = ExcisionPublication {
        version: 1,
        replacements: vec![],
        retired_batches: vec![],
        digest: String::new(),
    };
    publication.digest = publication.computed_digest().unwrap();
    conn.execute("INSERT INTO excision(id,scope,refused_json,created_at,commit_seq) VALUES ('remove','input:source','{}',1,1)", []).unwrap();
    let id = Published::open(&source.root).unwrap().archive_id().unwrap();
    for (n, sessions) in [
        None,
        Some(json!(null)),
        Some(json!([1])),
        Some(json!([""])),
        Some(json!("codex:wrong")),
    ]
    .into_iter()
    .enumerate()
    {
        let missing = sessions.is_none();
        let mut metadata = json!({"publication":publication});
        if let Some(sessions) = sessions {
            metadata["sessions"] = sessions;
        }
        conn.execute(
            "UPDATE excision SET refused_json=?1",
            [metadata.to_string()],
        )
        .unwrap();
        let (unit, bytes) = descriptor(
            &conn,
            Unit {
                commit_seq: 1,
                kind: "excision".into(),
                key: "remove".into(),
            },
            Some(publication.clone()),
        );
        let selected = select(std::slice::from_ref(&unit)).unwrap();
        let (target, _target_lock) =
            Archive::open_or_create(&dir.path().join(format!("target-{n}"))).unwrap();
        let mut restoring = Restoring::open(&target).unwrap();
        let error = restoring
            .install(&id, &selected, |_| Ok(Some(Cursor::new(bytes.clone()))))
            .unwrap_err();
        if missing {
            assert!(
                matches!(error, StoreError::Invalid(message) if message.contains("no retained session selection"))
            );
        } else {
            assert!(matches!(error, StoreError::Json(_)), "{error}");
        }
        let db = open_write(&target.archive_db(), cache::SETUP).unwrap();
        assert_eq!(
            db.query_row("SELECT count(*) FROM excision", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            0
        );
        assert_eq!(
            db.query_row(
                "SELECT count(*) FROM meta WHERE key='sinex_restored_view'",
                [],
                |r| r.get::<_, i64>(0)
            )
            .unwrap(),
            0
        );
    }
}
