use super::*;

#[test]
fn same_preparation_resumes_and_a_new_parser_drains_the_old_one() {
    let mut f = Fixture::new();
    let build = f.prepare().unwrap().build;
    let partial = f
        .writer
        .begin_message_value(&build, values::Codec::Utf8, 20)
        .unwrap();
    f.writer
        .append_value_chunk(&partial.id, 0, b"part", 4)
        .unwrap();
    assert!(
        !f.writer
            .retire_superseded_message_turn(&f.message, &source(&f.head.token), 4096)
            .unwrap()
    );
    assert_eq!(f.prepare().unwrap().build, build);
    let old = f.head.token.clone();
    f.head = f
        .writer
        .begin_parser(&parser_state::Basis {
            input_pk: old.input_pk,
            chain: old.chain,
            parser_version: 4,
            digest: [9; 32],
            acquired_upto: f.head.acquired_upto,
            companions_json: None,
            excisions_seen: 0,
        })
        .unwrap();
    assert!(matches!(
        f.writer
            .retire_superseded_message_turn(&f.message, &source(&old), 4096),
        Err(Error::StaleMessage)
    ));
    let mut invalid = source(&f.head.token);
    invalid.revision = 999;
    assert!(matches!(
        f.writer
            .retire_superseded_message_turn(&f.message, &invalid, 4096),
        Err(Error::StaleMessage)
    ));
    assert_eq!(
        f.scalar("SELECT count(*) FROM message_build WHERE phase='project'"),
        1
    );
    assert!(
        f.writer
            .retire_superseded_message_turn(&f.message, &source(&f.head.token), 4096)
            .unwrap()
    );
    assert_eq!(
        f.scalar("SELECT count(*) FROM message_build WHERE phase='retiring'"),
        1
    );
    let mut f = f.reopen();
    let mut turns = 0;
    while f
        .writer
        .retire_superseded_message_turn(&f.message, &source(&f.head.token), 4096)
        .unwrap()
    {
        turns += 1;
        assert!(turns < 64);
    }
    assert!(turns > 0);
    assert_eq!(f.scalar("SELECT count(*) FROM message_build"), 0);
    assert_eq!(f.scalar("SELECT count(*) FROM message_body"), 0);
    assert_eq!(f.scalar("SELECT count(*) FROM message_candidate"), 0);
    assert_eq!(
        f.scalar("SELECT count(*) FROM body_value WHERE state='retiring' AND incoming=0"),
        1
    );
    let next = f.prepare().unwrap().build;
    assert_ne!(next.token(), build.token());
    assert!(f.writer.message_projection(&build, 4096).is_err());
}

#[test]
fn superseded_turn_checks_limits_and_ignores_missing_message() {
    let mut f = Fixture::new();
    let build = f.prepare().unwrap().build;
    assert!(matches!(
        f.writer
            .retire_superseded_message_turn(&f.message, &source(&f.head.token), 0),
        Err(Error::ParserAdmission { .. })
    ));
    let missing = MessageId::new(f.session.clone(), "missing").unwrap();
    assert!(
        !f.writer
            .retire_superseded_message_turn(&missing, &source(&f.head.token), 4096)
            .unwrap()
    );
    assert_eq!(f.prepare().unwrap().build, build);
    assert_eq!(
        f.scalar("SELECT count(*) FROM message_build WHERE phase='project'"),
        1
    );
}
