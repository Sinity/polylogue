use super::*;

fn messages(duration: u64) -> (Message, Message) {
    let mut call = text_message("c", "s", Role::Assistant, "", Some("tool_use"));
    call.blocks = vec![Block::ToolCall {
        run_by_script: false,
        call_id: Some("t1".into()),
        tool: Some("Bash".into()),
        action: polylogue_model::ToolAction::Shell,
        input: Json::Null,
        command: Some("printf x".into()),
        path: None,
    }];
    let mut result = text_message("r", "s", Role::User, "", None);
    result.blocks = vec![Block::ToolResult {
        reported_duration_ms: Some(duration),
        call_id: Some("t1".into()),
        outcome: ToolOutcome::Ok,
        output: Some("synthetic output".into()),
        data: None,
        attachments: vec![],
    }];
    (call, result)
}

#[test]
fn a_duration_outside_sql_integer_range_remains_canonical_in_both_pairing_orders() {
    for duration in [27, u64::MAX] {
        for result_first in [false, true] {
            let archive = test_archive();
            let (call, result) = messages(duration);
            let (first, second) = if result_first {
                (result, call)
            } else {
                (call, result)
            };
            archive
                .store_reading(
                    batch("duration", 0, 0, vec![row(0, first)], &["s"]),
                    Some(&SOURCE_RECORD),
                )
                .unwrap();
            archive
                .store_reading(
                    batch("duration", 0, 100, vec![row(100, second)], &["s"]),
                    Some(&SOURCE_RECORD),
                )
                .unwrap();
            archive.writer.derive().unwrap();
            let pool = ReadPool::open(&archive.archive).unwrap();
            let reader = pool.reader().unwrap();
            let rows = reader.query("SELECT reported_duration_ms AS duration,result_block_pk IS NOT NULL AS paired FROM tool_call WHERE call_id='t1'",&[]).unwrap();
            assert_eq!(
                rows[0]["duration"],
                i64::try_from(duration).ok().map_or(Json::Null, Json::from)
            );
            assert_eq!(rows[0]["paired"], 1);
            let stored = reader.messages(&sid("s"), false).unwrap();
            let actual = stored
                .iter()
                .flat_map(|message| &message.blocks)
                .find_map(|block| match block {
                    Block::ToolResult {
                        reported_duration_ms,
                        ..
                    } => Some(*reported_duration_ms),
                    _ => None,
                })
                .unwrap();
            assert_eq!(actual, Some(duration));
        }
    }
}

#[test]
fn an_oversized_background_duration_does_not_wrap_or_rewrite_the_canonical_result() {
    let archive = test_archive();
    let (call, result) = messages(27);
    archive
        .store_reading(
            batch("duration", 0, 0, vec![row(0, call), row(1, result)], &["s"]),
            Some(&SOURCE_RECORD),
        )
        .unwrap();
    let mut update = batch("duration", 0, 100, vec![], &["s"]);
    update.parse[0].counts.records = 1;
    update.parse[0].counts.add("notice", &RecordOutcome::Used);
    update.parse[0]
        .output
        .background
        .push(polylogue_model::batch::BackgroundUpdate {
            result_at: None,
            result_at_evidence: Default::default(),
            reported_duration_ms: Some(u64::MAX),
            session: sid("s"),
            task_id: "native-task".into(),
            call_id: Some("t1".into()),
            outcome: ToolOutcome::Ok,
            output_file: None,
        });
    archive.store_reading(update, Some(&SOURCE_RECORD)).unwrap();
    archive.writer.derive().unwrap();
    let pool = ReadPool::open(&archive.archive).unwrap();
    let reader = pool.reader().unwrap();
    assert!(
        reader
            .query(
                "SELECT reported_duration_ms AS duration FROM tool_call WHERE call_id='t1'",
                &[]
            )
            .unwrap()[0]["duration"]
            .is_null()
    );
    let stored = reader.messages(&sid("s"), false).unwrap();
    assert!(
        stored
            .iter()
            .flat_map(|message| &message.blocks)
            .any(|block| matches!(
                block,
                Block::ToolResult {
                    reported_duration_ms: Some(27),
                    ..
                }
            ))
    );
}
