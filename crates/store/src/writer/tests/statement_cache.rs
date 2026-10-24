use super::*;
use rusqlite::hooks::{AuthAction, AuthContext, Authorization};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

fn preparations(messages: usize, tools: bool) -> u64 {
    let dir = tempfile::tempdir().unwrap();
    let mut conn = open_write(&dir.path().join("index.db"), 1).unwrap();
    crate::schema::create_index(&conn).unwrap();
    let tx = conn.transaction().unwrap();
    let observed = Arc::new(AtomicU64::new(0));
    let count = observed.clone();
    tx.authorizer(Some(move |context: AuthContext<'_>| {
        if matches!(
            context.action,
            AuthAction::Read {
                table_name: "message",
                column_name: "id"
            }
        ) {
            count.fetch_add(1, Ordering::Relaxed);
        }
        Authorization::Allow
    }))
    .unwrap();
    let rows = (0..messages)
        .map(|n| {
            let mut message = text_message(
                &format!("message-{n}"),
                "session1",
                Role::User,
                "Synthetic body",
                None,
            );
            if tools && n % 3 == 1 {
                message.role = Role::Assistant;
                message.prompt_kind = None;
                message.blocks.push(Block::ToolCall {
                    call_id: Some(format!("call-{}", n / 3)),
                    tool: Some("Bash".into()),
                    action: polylogue_model::ToolAction::Shell,
                    input: Json::Null,
                    command: Some("printf x".into()),
                    path: None,
                });
            } else if tools && n % 3 == 2 {
                message.prompt_kind = None;
                message.blocks.push(Block::ToolResult {
                    call_id: Some(format!("call-{}", n / 3)),
                    outcome: ToolOutcome::Ok,
                    output: Some("Synthetic tool output".into()),
                    data: None,
                    attachments: vec![],
                });
            }
            row(n as u64, message)
        })
        .collect();
    let batch = batch("input1", 0, 0, rows, &["session1"]);
    assert_eq!(
        write_rows(&tx, &batch, &batch.parse[0], 1, 1, true).unwrap(),
        messages as u64
    );
    tx.authorizer(None::<fn(AuthContext<'_>) -> Authorization>)
        .unwrap();
    assert_eq!(
        tx.query_row("SELECT COUNT(*) FROM message", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        messages as i64
    );
    if tools {
        assert_eq!(
            tx.query_row("SELECT COUNT(*) FROM tool_call", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            ((messages + 1) / 3) as i64
        );
        assert_eq!(
            tx.query_row(
                "SELECT COUNT(*) FROM tool_call WHERE result_block_pk IS NOT NULL",
                [],
                |r| r.get::<_, i64>(0)
            )
            .unwrap(),
            (messages / 3) as i64
        );
    }
    observed.load(Ordering::Relaxed)
}

#[test]
fn repeated_messages_reuse_the_hot_writer_statements() {
    for tools in [false, true] {
        let small = preparations(32, tools);
        let large = preparations(128, tools);
        println!("tools={tools}: message SQL preparations:32={small},128={large}");
        assert!(
            large <= small + 8,
            "tools={tools}: message ingestion reparsed the hot SQL loop: {small}->{large}"
        );
    }
}
