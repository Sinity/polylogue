use crate::fixture::writer::message;
use crate::writer::ParseStep;
use polylogue_model::{Block, HookFieldClaim, MessageRow, Output, RecordCounts, Role};

fn step() -> ParseStep {
    let message = message(
        "hook",
        "session",
        Role::System,
        0,
        vec![
            Block::Text {
                text: "hook diagnostic".into(),
            },
            Block::Text {
                text: "actual context".into(),
            },
        ],
    );
    let mut output = Output {
        messages: vec![MessageRow {
            id_kind: polylogue_model::IdKind::Native,
            offset: 10,
            end: 80,
            message,
        }],
        ..Output::default()
    };
    output.native_hook_fields.insert(
        0,
        vec![HookFieldClaim {
            block: 1,
            start: 10,
            end: 80,
            pointer: "/hookAdditionalContext/3".into(),
        }],
    );
    let mut step = ParseStep::new(output, RecordCounts::default(), 0..100, true);
    step.prepare();
    step
}

#[test]
fn field_claims_bind_exact_record_and_original_selected_text_block() {
    let step = step();
    step.prepared.validate(&step.output).unwrap();
    let before = step.prepared.messages[0].hash;
    let complete = step.complete_output().unwrap();
    assert_eq!(
        complete.messages[0].message.content_hash().unwrap().0,
        before
    );
    assert_eq!(complete.native_hook_fields, step.output.native_hook_fields);
    for case in 0..6 {
        let mut invalid = step.clone();
        let claim = &mut invalid.output.native_hook_fields.get_mut(&0).unwrap()[0];
        match case {
            0 => claim.start = 11,
            1 => claim.end = 79,
            2 => claim.block = 2,
            3 => claim.pointer = "/hookAdditionalContext/03".into(),
            4 => claim.pointer = "/content/0/text".into(),
            _ => {
                invalid.output.native_hook_fields.insert(1, Vec::new());
            }
        }
        assert!(
            invalid.prepared.validate(&invalid.output).is_err(),
            "case {case}"
        );
    }
}

#[test]
fn sparse_field_claims_keep_original_ordinals_instead_of_selected_positions() {
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
            id_kind: polylogue_model::IdKind::Native,
            offset: 10,
            end: 80,
            message: original,
        }],
        ..Output::default()
    };
    output.message_projections.insert(
        0,
        polylogue_model::bundle::MessageProjection {
            source_version,
            block_count: 4,
            block_ordinals: vec![3],
        },
    );
    output.native_hook_fields.insert(
        0,
        vec![HookFieldClaim {
            block: 3,
            start: 10,
            end: 80,
            pointer: "/hookAdditionalContext/2".into(),
        }],
    );
    let mut reading = ParseStep::new(output, RecordCounts::default(), 0..100, true);
    reading.prepare();
    reading.prepared.validate(&reading.output).unwrap();
    assert_eq!(reading.prepared.messages[0].hash, source_version.0);
    reading.output.native_hook_fields.get_mut(&0).unwrap()[0].block = 0;
    assert!(reading.prepared.validate(&reading.output).is_err());
}
