use super::*;

#[test]
fn late_directory_edge_failure_rolls_back_ready_nodes_values_roots_and_cursor() {
    let mut f = Fixture::new();
    let build = f.prepare().unwrap().build;
    // Eight levels and their child acquisitions succeed before the next edge
    // fails. Ready nodes inserted in this transaction must not survive failure.
    f.writer._index.execute_batch("CREATE TRIGGER refuse_late_directory_edge BEFORE INSERT ON message_body_edge WHEN NEW.parent_height=8 BEGIN SELECT RAISE(ABORT,'synthetic late directory edge'); END;").unwrap();
    assert!(matches!(
        f.writer.write_small_field(
            &build,
            &slot("one"),
            None,
            true,
            false,
            &payload(),
            &step(0),
            65536
        ),
        Err(Error::Sqlite { .. })
    ));
    f.writer
        ._index
        .execute_batch("DROP TRIGGER refuse_late_directory_edge")
        .unwrap();
    let mut f = f.reopen();
    for table in [
        "body_value",
        "body_value_chunk",
        "message_field_slot",
        "message_field_fact",
        "message_body_node",
        "message_body_edge",
    ] {
        assert_eq!(
            f.scalar(&format!("SELECT count(*) FROM {table}")),
            0,
            "{table}"
        );
    }
    assert_eq!(
        f.writer.message_projection(&build, 65536).unwrap().sequence,
        0
    );
    assert_eq!(
        f.scalar("SELECT field_root_pk IS NULL FROM message_body"),
        1
    );
    assert_eq!(
        f.scalar(
            "SELECT active_value_pk IS NULL AND active_origin_value_pk IS NULL FROM message_build"
        ),
        1
    );
    // Ordinary retry starts from the same Build and creates one complete path.
    f.writer
        .write_small_field(
            &build,
            &slot("one"),
            None,
            true,
            false,
            &payload(),
            &step(0),
            65536,
        )
        .unwrap();
    assert_eq!(f.scalar("SELECT count(*) FROM message_body_node"), 16);
    assert_eq!(
        f.scalar("SELECT count(*) FROM message_body_node WHERE state!='ready'"),
        0
    );
    assert_eq!(
        f.writer.message_projection(&build, 65536).unwrap().sequence,
        1
    );
}
