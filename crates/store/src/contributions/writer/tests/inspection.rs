use super::*;
use crate::contributions::ProjectionStep;
fn inspect_with(
    f: &mut Fixture,
    build: &Build,
    change: impl FnOnce(&mut Identity<'_>, &mut Source<'_>),
    allowance: usize,
) -> Result<Preparation> {
    let mut identity = Identity {
        session: &f.session,
        origin: &f.origin,
        native_session: "session:with:colons",
        native_session_encoding: None,
        session_kind: SessionKind::Main,
        message: &f.message,
        sequence: 0,
        role: Role::Assistant,
        written_by: WrittenBy::Model,
    };
    let mut source = Source {
        predecessor: &f.head.token,
        revision: 1,
        evidence_revision: 1,
        logical_key: [3; 32],
        route: RouteKind::Native,
    };
    change(&mut identity, &mut source);
    f.writer
        .inspect_message_preparation(build, identity, source, allowance)
}
fn inspect(f: &mut Fixture, build: &Build) -> Result<Preparation> {
    inspect_with(f, build, |_, _| {}, 4096)
}
fn wal(f: &Fixture) -> Vec<Vec<u8>> {
    ["index.db-wal", "archive.db-wal"]
        .iter()
        .map(|name| std::fs::read(f.root.join(name)).unwrap_or_default())
        .collect()
}
#[test]
fn exact_preparation_inspection_works_with_sql_writes_denied_and_does_not_reserve_or_write_wal() {
    let mut f = Fixture::new();
    let build = f.prepare().unwrap().build;
    let writes = (
        f.writer._index.total_changes(),
        f.writer.archive.total_changes(),
    );
    let seq = f.scalar("SELECT seq FROM sqlite_sequence WHERE name='message'");
    let before = wal(&f);
    f.writer
        ._index
        .execute_batch("PRAGMA query_only=ON")
        .unwrap();
    f.writer
        .archive
        .execute_batch("PRAGMA query_only=ON")
        .unwrap();
    for _ in 0..20 {
        let p = inspect(&mut f, &build).unwrap();
        assert_eq!(p.build, build);
        assert!(
            p.active_value.is_none() && p.active_origin_value.is_none() && p.projection.is_none()
        );
    }
    assert_eq!(
        writes,
        (
            f.writer._index.total_changes(),
            f.writer.archive.total_changes()
        )
    );
    assert_eq!(
        seq,
        f.scalar("SELECT seq FROM sqlite_sequence WHERE name='message'")
    );
    assert_eq!(before, wal(&f));
    // The oracle distinguishes the existing allocation/reservation path.
    assert!(f.prepare().is_err());
    f.writer
        ._index
        .execute_batch("PRAGMA query_only=OFF")
        .unwrap();
    f.writer
        .archive
        .execute_batch("PRAGMA query_only=OFF")
        .unwrap();
    assert_eq!(f.scalar("SELECT count(*) FROM message"), 1);
}
#[test]
fn exact_preparation_inspection_resumes_both_partial_producers_without_mutation() {
    let mut f = Fixture::new();
    let build = f.prepare().unwrap().build;
    let value = f
        .writer
        .begin_message_value(&build, values::Codec::Json, 8)
        .unwrap();
    f.writer.append_value_chunk(&value.id, 0, b"nu", 2).unwrap();
    let origin = f.writer.begin_message_origin_value(&build, 4).unwrap();
    f.writer
        .append_value_chunk(&origin.id, 0, b"nu", 2)
        .unwrap();
    let mut f = f.reopen();
    let before = wal(&f);
    let p = inspect(&mut f, &build).unwrap();
    let v = p.active_value.unwrap();
    let o = p.active_origin_value.unwrap();
    assert_eq!(v.id, value.id);
    assert_eq!(o.id, origin.id);
    assert_eq!((v.appended, o.appended), (2, 2));
    assert_eq!(v.state, values::State::Building);
    assert_eq!(before, wal(&f));
}
#[test]
fn exact_preparation_inspection_keeps_completed_rank_and_cursor_after_reopen() {
    let mut f = Fixture::new();
    let build = f.prepare().unwrap().build;
    let cert = f
        .writer
        .finish_message_projection(
            &build,
            &ProjectionStep {
                expected_sequence: 0,
                next_cursor: b"rank",
            },
            polylogue_model::BlobHash::of(b"synthetic evidence"),
            polylogue_model::merge::FinishState::Unknown,
            4096,
        )
        .unwrap();
    let mut f = f.reopen();
    let before = wal(&f);
    let p = inspect(&mut f, &build).unwrap();
    assert_eq!(p.projection.unwrap().version_hash(), cert.version_hash());
    let cursor = f.writer.message_projection(&build, 4096).unwrap();
    assert_eq!(cursor.sequence, 1);
    assert_eq!(cursor.cursor.as_deref(), Some(b"rank".as_slice()));
    assert_eq!(before, wal(&f));
    f.writer.retire_message_preparation(&build).unwrap();
    let count = f.scalar("SELECT count(*) FROM message_body");
    assert!(inspect(&mut f, &build).is_err());
    assert_eq!(count, f.scalar("SELECT count(*) FROM message_body"));
}
#[test]
fn exact_preparation_inspection_refuses_hidden_removal_contributor_and_source_changes() {
    for sql in [
        "UPDATE message SET hidden=1",
        "UPDATE message SET removal_epoch=removal_epoch+1",
        "UPDATE message SET contributor_revision=contributor_revision+1",
    ] {
        let mut f = Fixture::new();
        let build = f.prepare().unwrap().build;
        f.writer._index.execute_batch(sql).unwrap();
        let before = wal(&f);
        assert!(matches!(
            inspect(&mut f, &build),
            Err(Error::MessagePredecessorChanged)
        ));
        assert_eq!(before, wal(&f));
    }
    let mut f = Fixture::new();
    let build = f.prepare().unwrap().build;
    f.writer
        .begin_parser(&parser_state::Basis {
            input_pk: f.head.token.input_pk,
            chain: f.head.token.chain,
            parser_version: 3,
            digest: [8; 32],
            acquired_upto: f.head.acquired_upto,
            companions_json: None,
            excisions_seen: 0,
        })
        .unwrap();
    assert!(matches!(
        inspect(&mut f, &build),
        Err(Error::MessagePredecessorChanged)
    ));
}
#[test]
fn exact_preparation_inspection_keeps_identity_route_revision_and_admission_guards() {
    let mut f = Fixture::new();
    let build = f.prepare().unwrap().build;
    assert!(
        inspect_with(
            &mut f,
            &build,
            |id, _| id.native_session = "different-native",
            4096
        )
        .is_err()
    );
    assert!(inspect_with(&mut f, &build, |id, _| id.role = Role::User, 4096).is_err());
    assert!(
        inspect_with(
            &mut f,
            &build,
            |_, source| source.route = RouteKind::PageDom,
            4096
        )
        .is_err()
    );
    assert!(
        inspect_with(
            &mut f,
            &build,
            |_, source| source.logical_key = [4; 32],
            4096
        )
        .is_err()
    );
    assert!(inspect_with(&mut f, &build, |_, source| source.revision = 2, 4096).is_err());
    assert!(matches!(
        inspect_with(&mut f, &build, |_, _| {}, 1),
        Err(Error::ParserAdmission { .. })
    ));
    assert_eq!(f.scalar("SELECT count(*) FROM message"), 1);
    assert!(inspect(&mut f, &build).is_ok());
}
