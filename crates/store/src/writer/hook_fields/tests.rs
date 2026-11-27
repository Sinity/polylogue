use crate::fixture::writer::message;
use crate::writer::ParsePart;
use polylogue_model::{Block, MessageRow, Output, ParsedHookField, RecordCounts, Role};

fn step() -> ParsePart {
    step_with_second_block(Block::Text {
        text: "actual context".into(),
    })
}

fn step_with_second_block(second: Block) -> ParsePart {
    let message = message(
        "hook",
        "session",
        Role::System,
        0,
        vec![
            Block::Text {
                text: "hook diagnostic".into(),
            },
            second,
        ],
    );
    let mut output = Output {
        messages: vec![MessageRow {
            response: None,
            page_text: false,
            id_kind: polylogue_model::IdKind::Native,
            offset: 10,
            end: 80,
            message,
        }],
        ..Output::default()
    };
    output.hook_fields.insert(
        0,
        vec![ParsedHookField {
            block: 1,
            start: 10,
            end: 80,
            pointer: "/hookAdditionalContext/3".into(),
        }],
    );
    let mut step = ParsePart::new(output, RecordCounts::default(), 0..100, true);
    step.prepare();
    step
}

#[test]
fn stated_hook_fields_bind_record_and_text_block() {
    let step = step();
    step.prepared.validate(&step.output).unwrap();
    let before = step.prepared.messages[0].hash;
    let complete = step.complete_output().unwrap();
    assert_eq!(
        complete.messages[0].message.content_hash().unwrap().0,
        before
    );
    assert_eq!(complete.hook_fields, step.output.hook_fields);
    for case in 0..6 {
        let mut invalid = step.clone();
        let claim = &mut invalid.output.hook_fields.get_mut(&0).unwrap()[0];
        match case {
            0 => claim.start = 11,
            1 => claim.end = 79,
            2 => claim.block = 2,
            3 => claim.pointer = "/hookAdditionalContext/03".into(),
            4 => claim.pointer = "/content/0/text".into(),
            // A claim for a message index the step does not have.
            _ => {
                invalid.output.hook_fields.insert(1, Vec::new());
            }
        }
        let error = invalid
            .prepared
            .validate(&invalid.output)
            .expect_err("an invalid stated field is refused")
            .to_string();
        let cause = [
            "stated hook field span differs from its source record",
            "stated hook field span differs from its source record",
            "stated hook field names a block its message did not select",
            "stated hook field pointer is not a hook context index",
            "stated hook field pointer is not a hook context index",
            "stated hook field names a message the step does not hold",
        ][case];
        assert_eq!(error, cause, "case {case}");
    }
}

#[test]
fn a_stated_hook_field_must_name_a_text_block() {
    let step = step_with_second_block(Block::Thinking {
        text: Some("reasoning".into()),
        signature: None,
        redacted: false,
    });
    let error = step
        .prepared
        .validate(&step.output)
        .expect_err("a claim on a non-text block is refused")
        .to_string();
    assert_eq!(error, "stated hook field names a block that is not text");
}

#[test]
fn sparse_hook_fields_keep_original_ordinals() {
    let mut original = message(
        "sparse-hook",
        "session",
        Role::System,
        0,
        (0..4)
            .map(|index| Block::Text {
                text: format!("source block {index}"),
            })
            .collect(),
    );
    let source_version = original.content_hash().unwrap();
    original.blocks = vec![original.blocks[3].clone()];
    let mut output = Output {
        messages: vec![MessageRow {
            response: None,
            page_text: false,
            id_kind: polylogue_model::IdKind::Native,
            offset: 10,
            end: 80,
            message: original,
        }],
        ..Output::default()
    };
    output.block_selections.insert(
        0,
        polylogue_model::bundle::ShownBlocks {
            source_version,
            block_count: 4,
            block_ordinals: vec![3],
        },
    );
    output.hook_fields.insert(
        0,
        vec![ParsedHookField {
            block: 3,
            start: 10,
            end: 80,
            pointer: "/hookAdditionalContext/2".into(),
        }],
    );
    let mut reading = ParsePart::new(output, RecordCounts::default(), 0..100, true);
    reading.prepare();
    reading.prepared.validate(&reading.output).unwrap();
    assert_eq!(reading.prepared.messages[0].hash, source_version.0);
    reading.output.hook_fields.get_mut(&0).unwrap()[0].block = 0;
    assert_eq!(
        reading
            .prepared
            .validate(&reading.output)
            .unwrap_err()
            .to_string(),
        "stated hook field names a block its message did not select"
    );
}
