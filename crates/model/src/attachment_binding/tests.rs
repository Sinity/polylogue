use super::*;
use crate::{
    Block, Message,
    bundle::{BundleError, MessageRecord},
};
use serde_json::json;

fn record() -> MessageRecord {
    let message: Message = serde_json::from_value(json!({
        "id":"codex:binding", "session":"codex:session", "role":"assistant",
        "blocks":[{"type":"text","text":"first"},{"type":"file","source_id":"opaque:source:id","direction":"unknown"}]
    })).unwrap();
    let source_version = message.content_hash().unwrap();
    serde_json::from_value(json!({
        "ref":"message:codex:binding", "commit":1, "order":0,
        "source_version":source_version, "block_count":2, "block_ordinals":[0,1],
        "message":message,
        "bindings":[{"source_version":source_version,"block":1,"attachment":0,"source_id":"opaque:source:id","candidates":[{
            "hash":BlobHash::of(b"actual candidate"),"size":16,"sources":[{
                "input":"historical-input","chain":3,"source_version":"historical-version",
                "members":["assets/original"],"names":["supplied name"],"media_types":["application/octet-stream"],"declared_sizes":[9]
            }]
        }]}]
    })).unwrap()
}

#[test]
fn attachment_claims_preserve_message_identity_and_original_sparse_positions() {
    let original = record();
    let message_hash = original.message.content_hash().unwrap();
    assert!(original.validate().is_ok());
    let mut sparse = original.clone();
    sparse.message.blocks.remove(0);
    sparse.block_ordinals.remove(0);
    assert!(sparse.validate().is_ok());
    let restored: MessageRecord =
        serde_json::from_slice(&serde_json::to_vec(&sparse).unwrap()).unwrap();
    assert_eq!(restored, sparse);
    assert_eq!(restored.bindings[0].block, 1);
    assert_eq!(
        restored.bindings[0].candidates[0].sources[0].declared_sizes,
        [9]
    );
    let mut without = original.clone();
    without.bindings.clear();
    assert_eq!(without.message.content_hash().unwrap(), message_hash);
    assert_eq!(original.message, without.message);
    assert!(
        serde_json::to_value(without)
            .unwrap()
            .get("bindings")
            .is_none()
    );
}

#[test]
fn binding_slots_refuse_other_versions_blocks_indexes_ids_and_blob_overrides() {
    for invalid in [
        "version",
        "block",
        "index",
        "id",
        "duplicate",
        "blob",
        "input",
        "chain",
        "source_version",
    ] {
        let mut value = record();
        match invalid {
            "version" => value.bindings[0].source_version = BlobHash::of(b"other source"),
            "block" => value.bindings[0].block = 0,
            "index" => value.bindings[0].attachment = 1,
            "id" => value.bindings[0].source_id = "another:id".into(),
            "duplicate" => value.bindings.push(value.bindings[0].clone()),
            "blob" => {
                let Block::File(attachment) = &mut value.message.blocks[1] else {
                    unreachable!()
                };
                attachment.blob = Some(BlobHash::of(b"other bytes"));
            }
            "input" => value.bindings[0].candidates[0].sources[0].input.clear(),
            "chain" => value.bindings[0].candidates[0].sources[0].chain = 0,
            "source_version" => value.bindings[0].candidates[0].sources[0]
                .source_version
                .clear(),
            _ => unreachable!(),
        }
        assert_eq!(
            value.validate(),
            Err(BundleError::InvalidAttachmentBinding),
            "{invalid}"
        );
    }
}

#[test]
fn unknown_and_conflicting_candidates_keep_every_claim_without_a_local_availability_flag() {
    let mut value = record();
    let mut other = value.bindings[0].candidates[0].clone();
    other.sources[0].input = "another-historical-input".into();
    value.bindings[0].candidates.push(other);
    assert!(value.validate().is_ok());
    let restored: MessageRecord =
        serde_json::from_slice(&serde_json::to_vec(&value).unwrap()).unwrap();
    assert_eq!(restored.bindings[0].candidates.len(), 2);
    assert_ne!(
        restored.bindings[0].candidates[0].sources,
        restored.bindings[0].candidates[1].sources
    );
    let mut raw = serde_json::to_value(&value).unwrap();
    raw["bindings"][0]["available"] = json!(true);
    assert!(serde_json::from_value::<MessageRecord>(raw).is_err());
    value.bindings[0].candidates[0].sources.clear();
    assert!(value.validate().is_ok());
    value.bindings[0].candidates.clear();
    assert!(value.validate().is_ok());
}

#[test]
fn tool_result_binding_uses_its_attachment_index_inside_the_original_block() {
    let mut value = record();
    let Block::File(original) = value.message.blocks[1].clone() else {
        unreachable!()
    };
    let mut another = original.clone();
    another.source_id = Some("different:attachment".into());
    value.message.blocks[1] = Block::ToolResult {
        reported_duration_ms: None,
        call_id: Some("call".into()),
        outcome: crate::ToolOutcome::Ok,
        output: None,
        data: None,
        attachments: vec![another, original],
    };
    value.source_version = value.message.content_hash().unwrap();
    value.bindings[0].source_version = value.source_version;
    value.bindings[0].attachment = 1;
    assert!(value.validate().is_ok());
    value.bindings[0].attachment = 0;
    assert_eq!(value.validate(), Err(BundleError::InvalidAttachmentBinding));
}
