//! Tool call and tool result blocks and the lifetime of their tool_call rows.
use super::*;
use crate::contributions::{ToolAction, blocks, retirement};
use polylogue_model::{ToolOutcome, UnknownReason};

fn call() -> BlockKind<'static> {
    BlockKind::ToolCall {
        operation_id: "opaque/proved-operation",
        at: Some(123),
        action: Some(ToolAction::Shell),
    }
}
fn tool() -> Slot {
    Slot::Block {
        block: BlockSlot::Item(0),
        field: BlockField::Tool,
    }
}
#[test]
fn tool_call_fields_and_command_survive_reopen_and_install() {
    let mut f = Fixture::new();
    let build = f.prepare().unwrap().build;
    let block = f
        .writer
        .begin_block(
            &build,
            0,
            0,
            call(),
            presence(true, false),
            &step(0, b"next"),
            4096,
        )
        .unwrap();
    assert_eq!(
        f.scalar("SELECT count(*) FROM message_body_edge WHERE tool_call_pk IS NOT NULL"),
        0
    );
    assert_eq!(f.scalar("SELECT outcome='unknown' AND outcome_reason='no_result' AND result_block_pk IS NULL FROM tool_call"),1);
    let tool_value = f
        .json(&build, ValuePurpose::Normalized, b"\"exec_command\"")
        .id
        .pk();
    f.writer
        .write_block_field(&build, &tool(), None, true, &step(1, b"next"), 4096)
        .unwrap();
    assert_eq!(
        f.scalar("SELECT incoming FROM body_value WHERE codec='json'"),
        2
    );
    assert_eq!(
        f.scalar(
            "SELECT t.tool_value_pk=f.value_pk FROM tool_call t CROSS JOIN message_field_fact f"
        ),
        1
    );
    let command = f.value(&build, b"printf 'synthetic'\n").ready.id.pk();
    f.writer
        .write_tool_command(&build, &step(2, b"next"), 4096)
        .unwrap();
    f = f.reopen();
    assert_eq!(
        f.writer
            .install_block(&build, &step(3, b"next"), 4096)
            .unwrap(),
        block
    );
    assert_eq!(f.scalar("SELECT count(*) FROM message_body_edge e JOIN tool_call t ON t.pk=e.tool_call_pk AND t.call_block_pk=e.block_pk"),1);
    assert_eq!(
        f.scalar("SELECT active_block_pk IS NULL AND active_value_pk IS NULL FROM message_build"),
        1
    );
    assert_eq!(f.scalar("SELECT count(*) FROM tool_call_trigram"), 0);
    assert_eq!(
        f.scalar("SELECT count(*) FROM message WHERE selected_body_pk IS NOT NULL"),
        0
    );
    assert_eq!(
        f.scalar(&format!(
            "SELECT tool_value_pk={tool_value} AND command_value_pk={command} FROM tool_call"
        )),
        1
    );
}
#[test]
fn failed_cursor_rolls_back_each_tool_call_write() {
    let mut f = Fixture::new();
    let build = f.prepare().unwrap().build;
    f.exec("CREATE TRIGGER refuse_tool_cursor BEFORE INSERT ON parser_cell WHEN NEW.namespace='message.projection.v1' BEGIN SELECT RAISE(ABORT,'synthetic cursor refusal'); END;");
    assert!(matches!(
        f.writer.begin_block(
            &build,
            0,
            0,
            call(),
            presence(true, false),
            &step(0, b"next"),
            4096
        ),
        Err(Error::Sqlite { .. })
    ));
    assert_eq!(f.scalar("SELECT count(*) FROM tool_call"), 0);
    assert_eq!(f.scalar("SELECT count(*) FROM block"), 0);
    f.exec("DROP TRIGGER refuse_tool_cursor;");
    f.writer
        .begin_block(
            &build,
            0,
            0,
            call(),
            presence(true, false),
            &step(0, b"next"),
            4096,
        )
        .unwrap();
    f.json(&build, ValuePurpose::Normalized, b"\"tool\"")
        .id
        .pk();
    f.exec("CREATE TRIGGER refuse_tool_cursor BEFORE INSERT ON parser_cell WHEN NEW.namespace='message.projection.v1' BEGIN SELECT RAISE(ABORT,'synthetic cursor refusal'); END;");
    assert!(matches!(
        f.writer
            .write_block_field(&build, &tool(), None, true, &step(1, b"next"), 4096),
        Err(Error::Sqlite { .. })
    ));
    assert_eq!(f.scalar("SELECT incoming FROM body_value"), 1);
    assert_eq!(f.scalar("SELECT tool_value_pk IS NULL FROM tool_call"), 1);
    assert_eq!(f.scalar("SELECT count(*) FROM message_field_fact"), 0);
    f.exec("DROP TRIGGER refuse_tool_cursor;");
    f.writer
        .write_block_field(&build, &tool(), None, true, &step(1, b"next"), 4096)
        .unwrap();
    f.value(&build, b"command").ready.id.pk();
    f.exec("CREATE TRIGGER refuse_tool_cursor BEFORE INSERT ON parser_cell WHEN NEW.namespace='message.projection.v1' BEGIN SELECT RAISE(ABORT,'synthetic cursor refusal'); END;");
    assert!(matches!(
        f.writer.write_tool_command(&build, &step(2, b"next"), 4096),
        Err(Error::Sqlite { .. })
    ));
    assert_eq!(
        f.scalar("SELECT command_value_pk IS NULL FROM tool_call"),
        1
    );
    assert_eq!(
        f.scalar("SELECT active_value_pk IS NOT NULL FROM message_build"),
        1
    );
    f.exec("DROP TRIGGER refuse_tool_cursor;");
    f.writer
        .write_tool_command(&build, &step(2, b"next"), 4096)
        .unwrap();
    assert!(matches!(
        f.writer.install_block(&build, &step(2, b"next"), 4096),
        Err(Error::StaleMessage)
    ));
    assert_eq!(
        f.scalar("SELECT active_block_pk IS NOT NULL FROM message_build"),
        1
    );
    assert_eq!(
        f.scalar("SELECT block_root_pk IS NULL FROM message_body"),
        1
    );
}
#[test]
fn tool_result_stores_outcome_and_tool_call_needs_operation_id() {
    let mut f = Fixture::new();
    let build = f.prepare().unwrap().build;
    let outcome = ToolOutcome::Unknown {
        reason: UnknownReason::OutputNotRecorded,
    };
    let block = f
        .writer
        .write_block(
            &build,
            0,
            0,
            BlockKind::ToolResult {
                outcome: &outcome,
                exit_code: Some(0),
                duration_ms: Some(0),
            },
            presence(true, false),
            &step(0, b"next"),
            4096,
        )
        .unwrap();
    assert_eq!(f.scalar("SELECT result_outcome='unknown' AND result_reason='output_not_recorded' AND result_exit_code=0 AND result_duration_ms=0 FROM block"),1);
    assert_eq!(f.scalar("SELECT count(*) FROM tool_call"), 0);
    assert!(
        f.writer
            .begin_block(
                &build,
                0,
                1,
                BlockKind::ToolCall {
                    operation_id: "",
                    at: None,
                    action: None
                },
                presence(true, false),
                &step(1, b"next"),
                4096
            )
            .is_err()
    );
    assert_eq!(f.scalar("SELECT count(*) FROM block"), 1);
    f.json(&build, ValuePurpose::Normalized, b"null").id.pk();
    assert!(matches!(
        f.writer.begin_block(
            &build,
            0,
            1,
            call(),
            presence(true, false),
            &step(1, b"next"),
            4096
        ),
        Err(Error::MessagePreparationPending)
    ));
    assert_eq!(
        f.scalar("SELECT active_value_pk IS NOT NULL FROM message_build"),
        1
    );
    assert!(block > 0);
}
#[test]
fn retirement_releases_each_tool_call_value_lease() {
    let mut f = Fixture::new();
    let build = f.prepare().unwrap().build;
    f.writer
        .begin_block(
            &build,
            0,
            0,
            call(),
            presence(true, false),
            &step(0, b"next"),
            4096,
        )
        .unwrap();
    f.json(&build, ValuePurpose::Normalized, b"\"tool\"")
        .id
        .pk();
    f.writer
        .write_block_field(&build, &tool(), None, true, &step(1, b"next"), 4096)
        .unwrap();
    let command = f.value(&build, b"command").ready.id.pk();
    f.writer
        .write_tool_command(&build, &step(2, b"next"), 4096)
        .unwrap();
    // Two more columns point at the command value, each with its own lease.
    f.writer
        ._index
        .execute(
            "UPDATE body_value SET incoming=incoming+2 WHERE pk=?1",
            [command],
        )
        .unwrap();
    f.writer
        ._index
        .execute(
            "UPDATE tool_call SET command_norm_value_pk=?1,path_value_pk=?1",
            [command],
        )
        .unwrap();
    f.writer.retire_message_preparation(&build).unwrap();
    assert_eq!(f.scalar("SELECT count(*) FROM block"), 0);
    assert_eq!(f.scalar("SELECT count(*) FROM tool_call"), 0);
    assert_eq!(
        f.scalar("SELECT incoming=0 AND state='retiring' FROM body_value WHERE codec='utf8'"),
        1
    );
    assert_eq!(
        f.scalar("SELECT incoming=1 AND state='ready' FROM body_value WHERE codec='json'"),
        1
    );
    f.retire_until_complete(&build);
    assert_eq!(
        f.scalar("SELECT count(*) FROM body_value WHERE incoming<>0"),
        0
    );
}
#[test]
fn tool_call_is_kept_until_its_last_reference_is_released() {
    let mut f = Fixture::new();
    let build = f.prepare().unwrap().build;
    let outcome = ToolOutcome::Ok;
    let result = f
        .writer
        .write_block(
            &build,
            0,
            0,
            BlockKind::ToolResult {
                outcome: &outcome,
                exit_code: Some(0),
                duration_ms: None,
            },
            presence(true, false),
            &step(0, b"next"),
            4096,
        )
        .unwrap();
    f.writer
        .write_block(
            &build,
            0,
            1,
            call(),
            presence(true, false),
            &step(1, b"next"),
            4096,
        )
        .unwrap();
    // Pair the call with the result directly; pairing is not under test.
    f.writer
        ._index
        .execute("UPDATE tool_call SET result_block_pk=?1", [result])
        .unwrap();
    let root = f.scalar("SELECT block_root_pk FROM message_body");
    let path = f.index_path();
    let owner = d::Owner {
        message: build.message,
        session: build.session,
    };
    let tx = f.writer._index.transaction().unwrap();
    d::acquire(&tx, &path, owner, root).unwrap();
    tx.commit().unwrap();
    f.writer.retire_message_preparation(&build).unwrap();
    assert_eq!(
        f.writer.retire_message_turn(&build).unwrap(),
        Retirement::Root
    );
    assert_eq!(f.scalar("SELECT count(*) FROM tool_call"), 1);
    assert_eq!(f.scalar("SELECT count(*) FROM block"), 2);
    let tx = f.writer._index.transaction().unwrap();
    d::release(&tx, &path, owner, root).unwrap();
    tx.commit().unwrap();
    f.collect_nodes();
    assert_eq!(f.scalar("SELECT count(*) FROM tool_call"), 0);
    assert_eq!(f.scalar("SELECT count(*) FROM block"), 0);
}
#[test]
fn external_reference_keeps_tool_call_and_duplicate_row_is_an_error() {
    let mut f = Fixture::new();
    let build = f.prepare().unwrap().build;
    let block = f
        .writer
        .begin_block(
            &build,
            0,
            0,
            call(),
            presence(true, false),
            &step(0, b"next"),
            4096,
        )
        .unwrap();
    // Replace the unique index so a corrupt duplicate row can be inserted.
    f.exec("DROP INDEX tool_call_block; CREATE INDEX tool_call_block ON tool_call(call_block_pk); INSERT INTO tool_call(session_pk,message_pk,operation_id,call_block_pk,outcome,outcome_reason) SELECT session_pk,message_pk,'duplicate',call_block_pk,'unknown','no_result' FROM tool_call;");
    assert!(matches!(
        f.writer.install_block(&build, &step(1, b"next"), 4096),
        Err(Error::Invariant(_))
    ));
    f.writer
        ._index
        .execute("DELETE FROM tool_call WHERE operation_id='duplicate'", [])
        .unwrap();
    f.writer
        .install_block(&build, &step(1, b"next"), 4096)
        .unwrap();
    f.exec("INSERT INTO repo_command_last(repo_root,command_norm,tool_call_pk,at,outcome) SELECT '/synthetic','command',pk,0,'unknown' FROM tool_call;");
    f.writer.retire_message_preparation(&build).unwrap();
    f.writer.retire_message_turn(&build).unwrap();
    f.collect_nodes();
    assert_eq!(f.scalar("SELECT count(*) FROM tool_call"), 1);
    assert_eq!(f.scalar("SELECT count(*) FROM block"), 1);
    f.writer
        ._index
        .execute("DELETE FROM repo_command_last", [])
        .unwrap();
    let path = f.index_path();
    let tx = f.writer._index.transaction().unwrap();
    blocks::release_unreferenced_block(&tx, &path, block).unwrap();
    tx.commit().unwrap();
    assert_eq!(f.scalar("SELECT count(*) FROM tool_call"), 0);
    assert_eq!(f.scalar("SELECT count(*) FROM block"), 0);
}
#[test]
fn tool_call_limits_are_checked_first_and_tool_field_needs_json() {
    let mut f = Fixture::new();
    let build = f.prepare().unwrap().build;
    let oversized = "opaque".repeat(1000);
    assert!(matches!(
        f.writer.begin_block(
            &build,
            0,
            0,
            BlockKind::ToolCall {
                operation_id: &oversized,
                at: None,
                action: None
            },
            presence(true, false),
            &step(0, b"next"),
            64
        ),
        Err(Error::ParserAdmission { .. })
    ));
    assert_eq!(f.scalar("SELECT count(*) FROM block"), 0);
    f.writer
        .begin_block(
            &build,
            0,
            0,
            call(),
            presence(true, false),
            &step(0, b"next"),
            4096,
        )
        .unwrap();
    f.value(&build, b"not JSON").ready.id.pk();
    assert!(matches!(
        f.writer
            .write_block_field(&build, &tool(), None, true, &step(1, b"next"), 4096),
        Err(Error::ValueNotReady)
    ));
    assert_eq!(f.scalar("SELECT incoming FROM body_value"), 1);
    assert_eq!(f.scalar("SELECT tool_value_pk IS NULL FROM tool_call"), 1);
    assert_eq!(
        f.scalar("SELECT active_value_pk IS NOT NULL FROM message_build"),
        1
    );
}
#[test]
fn attached_tool_item_version_keeps_tool_call_and_block() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("archive");
    let mut writer = Writer::open(&root).unwrap();
    writer._index.execute_batch("INSERT INTO session(pk,id,origin,native_id,kind,changed) VALUES(1,'synthetic:session','synthetic','session','main',0);
        INSERT INTO message(pk,id,session_pk,seq) VALUES(1,'synthetic:message',1,0);
        INSERT INTO block(pk,message_pk,session_pk,group_ordinal,local_ordinal,slot_kind,kind) VALUES(1,1,1,0,0,'ordinary','tool_call');
        INSERT INTO tool_item(pk,id,session_pk,provider_namespace,identity_schema,identity_kind) VALUES(1,'opaque-operation',1,'synthetic','synthetic-v1','provisional');
        INSERT INTO tool_item_version(pk,tool_item_pk,session_pk,evidence_revision,kind,agent_class,correspondence,call_present,result_present,output_present) VALUES(1,1,1,0,'synthetic','unknown','unresolved',1,0,0);
        INSERT INTO tool_call(pk,session_pk,message_pk,operation_id,tool_item_pk,tool_item_version_pk,call_block_pk,outcome,outcome_reason) VALUES(1,1,1,'opaque-operation',1,1,1,'unknown','no_result');").unwrap();
    let path = root.join("index.db");
    let tx = writer._index.transaction().unwrap();
    assert!(retirement::tool_call_referenced(&tx, &path, 1).unwrap());
    blocks::release_unreferenced_block(&tx, &path, 1).unwrap();
    assert_eq!(
        tx.query_row("SELECT count(*) FROM tool_call", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        1
    );
    assert_eq!(
        tx.query_row("SELECT count(*) FROM block", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        1
    );
    // Detached from the tool item, the call is released; the version stays.
    tx.execute(
        "UPDATE tool_call SET tool_item_pk=NULL,tool_item_version_pk=NULL WHERE pk=1",
        [],
    )
    .unwrap();
    assert!(!retirement::tool_call_referenced(&tx, &path, 1).unwrap());
    blocks::release_unreferenced_block(&tx, &path, 1).unwrap();
    assert_eq!(
        tx.query_row("SELECT count(*) FROM tool_call", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        0
    );
    assert_eq!(
        tx.query_row("SELECT count(*) FROM block", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        0
    );
    assert_eq!(
        tx.query_row("SELECT count(*) FROM tool_item_version", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        1
    );
    tx.commit().unwrap();
}
