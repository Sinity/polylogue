use super::*;
fn reparse(f: &mut Fixture) {
    let old = f.head.clone();
    f.head = f
        .writer
        .begin_parser(&parser_state::ParserInput {
            input_pk: old.token.input_pk,
            chain: old.token.chain,
            parser_version: old.parser_version,
            digest: [8; 32],
            acquired_upto: old.acquired_upto,
            companions_json: None,
            excisions_seen: 0,
        })
        .unwrap();
    assert!(f.head.token.generation > old.token.generation);
    assert_eq!(f.head.parser_version, old.parser_version);
    // Drain the old parser state's cells.
    loop {
        let path = f.index_path();
        let result = parser_state::collect_retired(
            &mut f.writer._index,
            &path,
            &f.head.token,
            parser_state::Budget {
                cells: 1,
                encoded_bytes: 4096,
            },
        )
        .unwrap();
        if !result.remaining {
            break;
        }
    }
    assert_eq!(f.scalar("SELECT count(*) FROM parser_cell"), 0);
}
#[test]
fn reparse_publish_survives_cell_collection_and_late_failure() {
    let mut f = Fixture::new();
    let old_build = f.prepare().unwrap().build;
    let done = f.finish(&old_build);
    let old = f
        .writer
        .publish_source_projection(&done, None, b"old", 4096)
        .unwrap();
    f.writer
        .select_compatible_source_message_for_test(&old, 4096)
        .unwrap();
    reparse(&mut f);
    let new = f.prepare().unwrap().build;
    let done = f.finish(&new);
    let mut f = f.reopen();
    f.exec("CREATE TRIGGER refuse_reparse_delete BEFORE DELETE ON message_build BEGIN SELECT RAISE(ABORT,'late reparse failure'); END;");
    assert!(matches!(
        f.writer
            .publish_source_projection(&done, None, b"new", 4096),
        Err(Error::Sqlite { .. })
    ));
    assert_eq!(
        f.scalar(&format!(
            "SELECT eligible FROM message_candidate WHERE pk={}",
            old_build.candidate
        )),
        1
    );
    assert_eq!(
        f.scalar(&format!(
            "SELECT ready FROM message_body WHERE pk={}",
            new.body
        )),
        0
    );
    assert!(
        f.writer
            .current_source_projection(&f.head.token, &[3; 32], 4096)
            .unwrap()
            .is_none()
    );
    assert_eq!(
        &f.writer
            .read_handle()
            .unwrap()
            .selected_message(&f.message)
            .unwrap()
            .unwrap()
            .body_id(),
        old.body()
    );
    f.exec("DROP TRIGGER refuse_reparse_delete");
    let replacement = f
        .writer
        .publish_source_projection(&done, None, b"new", 4096)
        .unwrap();
    assert_eq!(
        f.scalar(&format!(
            "SELECT eligible FROM message_candidate WHERE pk={}",
            old_build.candidate
        )),
        0
    );
    assert_eq!(
        &f.writer
            .read_handle()
            .unwrap()
            .selected_message(&f.message)
            .unwrap()
            .unwrap()
            .body_id(),
        old.body()
    );
    f.writer
        .select_compatible_source_message_for_test(&replacement, 4096)
        .unwrap();
    let f = f.reopen();
    assert_eq!(
        f.writer
            .current_source_projection(&f.head.token, &[3; 32], 4096)
            .unwrap()
            .unwrap()
            .body(),
        replacement.body()
    );
}
#[test]
fn reparse_rejects_missing_or_inconsistent_old_source() {
    for damage in [
        "missing same generation",
        "old generation",
        "old step",
        "old source basis",
        "old preparation basis",
        "future generation",
        "new source basis",
    ] {
        let mut f = Fixture::new();
        let old_build = f.prepare().unwrap().build;
        let done = f.finish(&old_build);
        let old = f
            .writer
            .publish_source_projection(&done, None, b"old", 4096)
            .unwrap();
        f.writer
            .select_compatible_source_message_for_test(&old, 4096)
            .unwrap();
        if damage == "missing same generation" {
            f.advance_parser();
            f.writer
                ._index
                .execute(
                    "DELETE FROM parser_cell WHERE namespace='message.source.v1'",
                    [],
                )
                .unwrap();
        } else {
            reparse(&mut f);
        }
        let new = f.prepare().unwrap().build;
        let done = f.finish(&new);
        let (pk, column, value) = match damage {
            "old generation" => (
                old_build.candidate,
                "source_state_generation",
                "source_state_generation+10".to_string(),
            ),
            "old step" => (
                old_build.candidate,
                "source_state_step",
                "source_state_step+1".to_string(),
            ),
            "old source basis" => (
                old_build.candidate,
                "source_basis_digest",
                "zeroblob(32)".to_string(),
            ),
            "old preparation basis" => (
                old_build.candidate,
                "basis_digest",
                "zeroblob(32)".to_string(),
            ),
            "new source basis" => (
                new.candidate,
                "source_basis_digest",
                "zeroblob(32)".to_string(),
            ),
            "future generation" => {
                let mut token = old.token().clone();
                token.generation = f.head.token.generation + 1;
                let digest = preparation_digest(&token, 1, &[3; 32], 1);
                f.writer._index.execute("UPDATE message_candidate SET source_state_generation=?2,basis_digest=?3 WHERE pk=?1",params![old_build.candidate,token.generation,digest.as_slice()]).unwrap();
                (
                    old_build.candidate,
                    "source_state_step",
                    token.step.to_string(),
                )
            }
            _ => (
                old_build.candidate,
                "source_state_step",
                old.token().step.to_string(),
            ),
        };
        f.writer
            ._index
            .execute(
                &format!("UPDATE message_candidate SET {column}={value} WHERE pk=?1"),
                [pk],
            )
            .unwrap();
        assert!(
            matches!(
                f.writer
                    .publish_source_projection(&done, None, b"new", 4096),
                Err(Error::StaleMessage)
            ),
            "{damage}"
        );
        assert_eq!(
            f.scalar(&format!(
                "SELECT eligible FROM message_candidate WHERE pk={}",
                old_build.candidate
            )),
            1,
            "{damage}"
        );
        assert_eq!(
            f.scalar(&format!(
                "SELECT ready FROM message_body WHERE pk={}",
                new.body
            )),
            0,
            "{damage}"
        );
        assert_eq!(
            &f.writer
                .read_handle()
                .unwrap()
                .selected_message(&f.message)
                .unwrap()
                .unwrap()
                .body_id(),
            old.body()
        );
        assert!(
            f.writer
                .current_source_projection(&f.head.token, &[3; 32], 4096)
                .unwrap()
                .is_none()
        );
    }
}
#[test]
fn reparse_leaves_other_logical_keys_and_revisions_alone() {
    for other_revision in [false, true] {
        let mut f = Fixture::new();
        let old_build = f.prepare().unwrap().build;
        let done = f.finish(&old_build);
        f.writer
            .publish_source_projection(&done, None, b"old", 4096)
            .unwrap();
        reparse(&mut f);
        let preparation = f.writer.prepare_message(
            Identity {
                session: &f.session,
                origin: &f.origin,
                native_session: "session:with:colons",
                native_session_encoding: None,
                session_kind: SessionKind::Main,
                message: &f.message,
                sequence: 0,
                role: Role::Assistant,
                written_by: WrittenBy::Model,
            },
            Source {
                token: &f.head.token,
                revision: if other_revision { 2 } else { 1 },
                evidence_revision: 1,
                logical_key: if other_revision { [3; 32] } else { [4; 32] },
                route: RouteKind::Native,
            },
            4096,
        );
        if other_revision {
            // The input has only revision 1, so revision 2 is refused before
            // any candidate row is written.
            assert!(matches!(preparation, Err(Error::StaleMessage)));
            assert_eq!(
                f.scalar("SELECT count(*) FROM message_candidate WHERE eligible=1"),
                1
            );
            assert_eq!(f.scalar("SELECT count(*) FROM message_build"), 0);
            continue;
        }
        let new = preparation.unwrap().build;
        let done = f.finish(&new);
        let replacement = f
            .writer
            .publish_source_projection(&done, None, b"independent", 4096)
            .unwrap();
        assert_eq!(
            f.scalar("SELECT count(*) FROM message_candidate WHERE eligible=1"),
            2
        );
        assert_eq!(
            f.writer
                .select_compatible_source_message_for_test(&replacement, 4096)
                .unwrap(),
            SourceSelection::Contested
        );
    }
}
