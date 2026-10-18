use super::*;
#[test]
fn finished_projection_stays_unready_and_can_be_retired() {
    let mut f = Fixture::new();
    let build = f.prepare().unwrap().build;
    f.exec("CREATE TRIGGER refuse_certificate BEFORE UPDATE OF facts_ready ON message_candidate WHEN NEW.facts_ready=1 BEGIN SELECT RAISE(ABORT,'synthetic certificate failure'); END;");
    let step = step(0, b"done");
    let hash = polylogue_model::BlobHash::of(b"synthetic-evidence");
    assert!(matches!(
        f.writer.finish_message_projection(
            &build,
            &step,
            hash,
            polylogue_model::merge::FinishState::Unknown,
            4096
        ),
        Err(Error::Sqlite { .. })
    ));
    assert_eq!(f.scalar("SELECT ready FROM message_body"), 0);
    assert_eq!(f.sequence(&build), 0);
    f.exec("DROP TRIGGER refuse_certificate");
    f.writer
        .finish_message_projection(
            &build,
            &step,
            hash,
            polylogue_model::merge::FinishState::Unknown,
            4096,
        )
        .unwrap();
    assert_eq!(f.scalar("SELECT in_progress IS NULL FROM message_body"), 1);
    assert_eq!(f.scalar("SELECT finish FROM message_candidate"), 1);
    assert_eq!(f.scalar("SELECT eligible FROM message_candidate"), 0);
    assert_eq!(f.scalar("SELECT selected_body_pk IS NULL FROM message"), 1);
    assert!(matches!(
        f.writer.advance_message_projection(&build, &step, 4096),
        Err(Error::StaleMessage)
    ));
    let mut f = f.reopen();
    assert_eq!(
        f.prepare().unwrap().projection.unwrap().version_hash(),
        hash
    );
    assert_eq!(f.scalar("SELECT ready FROM message_body"), 0);
    assert_eq!(f.scalar("SELECT ready FROM message_candidate"), 0);
    f.writer.retire_message_preparation(&build).unwrap();
    f.retire_until_complete(&build);
    assert_eq!(f.scalar("SELECT count(*) FROM message_build"), 0);
}
#[test]
fn hidden_message_or_new_parser_blocks_writes_but_not_retirement() {
    let mut f = Fixture::new();
    let build = f.prepare().unwrap().build;
    let value = f
        .writer
        .begin_message_value(&build, values::Codec::Utf8, 10)
        .unwrap();
    f.writer
        ._index
        .execute("UPDATE message SET hidden=1,removal_epoch=1", [])
        .unwrap();
    assert!(matches!(
        f.writer.begin_message_value(&build, values::Codec::Utf8, 0),
        Err(Error::StaleMessage)
    ));
    assert!(matches!(f.prepare(), Err(Error::StaleMessage)));
    f.writer
        .begin_parser(&parser_state::Basis {
            input_pk: f.head.token.input_pk,
            chain: 1,
            parser_version: 4,
            digest: [8; 32],
            acquired_upto: f.head.acquired_upto,
            companions_json: None,
            excisions_seen: 0,
        })
        .unwrap();
    // Retirement of a stale preparation only drops its lease.
    f.writer.retire_message_preparation(&build).unwrap();
    assert_eq!(
        f.writer.value_head(&value.id).unwrap().state,
        values::State::Retiring
    );
    let mut other = Fixture::new();
    assert!(matches!(
        other.writer.retire_message_preparation(&build),
        Err(Error::StaleMessage)
    ));
}
#[test]
fn failed_or_conflicting_prepare_leaves_no_rows() {
    let mut f = Fixture::new();
    f.exec("CREATE TRIGGER refuse_build BEFORE INSERT ON message_build BEGIN SELECT RAISE(ABORT,'synthetic reservation failure'); END;");
    assert!(matches!(f.prepare(), Err(Error::Sqlite { .. })));
    assert_eq!(f.scalar("SELECT count(*) FROM message"), 0);
    assert_eq!(f.scalar("SELECT count(*) FROM message_body"), 0);
    assert_eq!(f.scalar("SELECT count(*) FROM message_candidate"), 0);
    f.exec("DROP TRIGGER refuse_build");
    let build = f.prepare().unwrap().build;
    let error = f.writer.prepare_message(
        Identity {
            session: &f.session,
            origin: &f.origin,
            native_session: "session:with:colons",
            native_session_encoding: None,
            session_kind: SessionKind::Main,
            message: &f.message,
            sequence: 0,
            role: Role::User,
            written_by: WrittenBy::Model,
        },
        Source {
            predecessor: &f.head.token,
            revision: 1,
            evidence_revision: 1,
            logical_key: [3; 32],
            route: RouteKind::Native,
        },
        4096,
    );
    assert!(matches!(error, Err(Error::Invariant(_))));
    assert_eq!(f.prepare().unwrap().build, build);
    assert_eq!(f.scalar("SELECT count(*) FROM message_build"), 1);
}
#[test]
fn source_growth_keeps_preparation_but_new_parser_blocks_seal() {
    let mut f = Fixture::new();
    let length = f.head.acquired_upto;
    f.head = f
        .writer
        .begin_parser(&parser_state::Basis {
            input_pk: f.head.token.input_pk,
            chain: 1,
            parser_version: 3,
            digest: [11; 32],
            acquired_upto: 0,
            companions_json: None,
            excisions_seen: 0,
        })
        .unwrap();
    let build = f.prepare().unwrap().build;
    let grown = f
        .writer
        .begin_parser(&parser_state::Basis {
            input_pk: f.head.token.input_pk,
            chain: 1,
            parser_version: 3,
            digest: [11; 32],
            acquired_upto: length,
            companions_json: None,
            excisions_seen: 0,
        })
        .unwrap();
    assert_eq!(grown.token, build.source);
    let value = f
        .writer
        .begin_message_value(&build, values::Codec::Utf8, 0)
        .unwrap();
    let proof = f.writer.validate_value(&value.id).unwrap();
    f.writer
        .begin_parser(&parser_state::Basis {
            input_pk: f.head.token.input_pk,
            chain: 1,
            parser_version: 4,
            digest: [12; 32],
            acquired_upto: length,
            companions_json: None,
            excisions_seen: 0,
        })
        .unwrap();
    assert!(matches!(
        f.writer.seal_message_value(&build, &proof),
        Err(Error::StaleMessage)
    ));
    assert_eq!(
        f.writer.value_head(&value.id).unwrap().state,
        values::State::Building
    );
    assert_eq!(
        f.scalar("SELECT count(*) FROM message_body WHERE ready=1"),
        0
    );
    f.writer.retire_message_preparation(&build).unwrap();
    assert_eq!(
        f.writer.value_head(&value.id).unwrap().state,
        values::State::Retiring
    );
}
