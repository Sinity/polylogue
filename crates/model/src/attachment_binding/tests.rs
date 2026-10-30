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
        "bindings":[{"source_version":source_version,
            "namespace":{"input":"message-source","chain":2,"import_id":"original-import"},
            "exported_from":{"input":"current-message-source","chain":4,"import_id":"current-import"},
            "prior_exports":[{"input":"supplied-prior-source","chain":1,"import_id":"supplied-prior-import"}],"block":1,"attachment":0,"source_id":"opaque:source:id","candidates":[{
            "hash":BlobHash::of(b"actual candidate"),"size":16,"sources":[{
                "namespace":{"input":"original-message-source","chain":7,"import_id":"historical-import"},
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
        "namespace_input",
        "namespace_chain",
        "namespace_import",
        "source_namespace_input",
        "source_namespace_chain",
        "source_namespace_import",
        "export_input",
        "export_chain",
        "export_import",
        "prior_input",
        "prior_chain",
        "prior_import",
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
            "namespace_input" => value.bindings[0].namespace.input.clear(),
            "namespace_chain" => value.bindings[0].namespace.chain = 0,
            "namespace_import" => value.bindings[0].namespace.import_id.clear(),
            "source_namespace_input" => value.bindings[0].candidates[0].sources[0]
                .namespace
                .as_mut()
                .unwrap()
                .input
                .clear(),
            "source_namespace_chain" => {
                value.bindings[0].candidates[0].sources[0]
                    .namespace
                    .as_mut()
                    .unwrap()
                    .chain = 0
            }
            "source_namespace_import" => value.bindings[0].candidates[0].sources[0]
                .namespace
                .as_mut()
                .unwrap()
                .import_id
                .clear(),
            "export_input" => value.bindings[0].exported_from.input.clear(),
            "export_chain" => value.bindings[0].exported_from.chain = 0,
            "export_import" => value.bindings[0].exported_from.import_id.clear(),
            "prior_input" => value.bindings[0].prior_exports[0].input.clear(),
            "prior_chain" => value.bindings[0].prior_exports[0].chain = 0,
            "prior_import" => value.bindings[0].prior_exports[0].import_id.clear(),
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

#[test]
fn qualified_namespaces_preserve_conflicting_claims_for_one_source_slot() {
    let original = record();
    let hash = original.message.content_hash().unwrap();
    for difference in ["import", "input", "chain"] {
        let mut value = original.clone();
        let mut another = value.bindings[0].clone();
        match difference {
            "import" => another.namespace.import_id = "another-import".into(),
            "input" => another.namespace.input = "another-source".into(),
            "chain" => another.namespace.chain += 1,
            _ => unreachable!(),
        }
        another.candidates[0].hash = BlobHash::of(b"conflicting bytes");
        another.candidates[0].size = 17;
        value.bindings.push(another);
        assert!(value.validate().is_ok(), "{difference}");
        let restored: MessageRecord =
            serde_json::from_slice(&serde_json::to_vec(&value).unwrap()).unwrap();
        assert_eq!(restored, value);
        assert_eq!(restored.bindings.len(), 2);
        assert_ne!(
            restored.bindings[0].namespace,
            restored.bindings[1].namespace
        );
        assert_ne!(
            restored.bindings[0].candidates[0].hash,
            restored.bindings[1].candidates[0].hash
        );
        assert_eq!(restored.message.content_hash().unwrap(), hash);
        assert_eq!(restored.message, original.message);
    }
}

#[test]
fn unknown_candidates_keep_their_required_namespace_and_unknown_original_sources() {
    let mut value = record();
    let namespace = value.bindings[0].namespace.clone();
    value.bindings[0].candidates[0].sources[0].namespace = None;
    assert!(value.validate().is_ok());
    let restored: MessageRecord =
        serde_json::from_slice(&serde_json::to_vec(&value).unwrap()).unwrap();
    assert_eq!(restored.bindings[0].namespace, namespace);
    assert_eq!(
        restored.bindings[0].candidates[0].sources[0].namespace,
        None
    );
    value.bindings[0].candidates.clear();
    let mut another = value.bindings[0].clone();
    another.namespace.import_id = "another-unknown-import".into();
    value.bindings.push(another);
    assert!(value.validate().is_ok());
    let restored: MessageRecord =
        serde_json::from_slice(&serde_json::to_vec(&value).unwrap()).unwrap();
    assert_eq!(restored, value);
    assert!(
        restored
            .bindings
            .iter()
            .all(|binding| binding.candidates.is_empty())
    );
    assert_ne!(
        restored.bindings[0].namespace,
        restored.bindings[1].namespace
    );
    let mut raw = serde_json::to_value(&value).unwrap();
    raw["bindings"][0]
        .as_object_mut()
        .unwrap()
        .remove("namespace");
    assert!(serde_json::from_value::<MessageRecord>(raw).is_err());
}

#[test]
fn complete_claim_identity_preserves_conflicts_and_supplied_export_contexts() {
    let original = record();
    for difference in ["bytes", "export", "prior"] {
        let mut value = original.clone();
        let mut another = value.bindings[0].clone();
        match difference {
            "bytes" => another.candidates[0].hash = BlobHash::of(b"conflicting claim"),
            "export" => another.exported_from.import_id = "another-current-import".into(),
            "prior" => another.prior_exports[0].import_id = "another-supplied-prior".into(),
            _ => unreachable!(),
        }
        value.bindings.push(another);
        assert!(value.validate().is_ok(), "{difference}");
        let restored: MessageRecord =
            serde_json::from_slice(&serde_json::to_vec(&value).unwrap()).unwrap();
        assert_eq!(restored, value);
        assert_eq!(restored.bindings.len(), 2);
        assert_eq!(
            restored.bindings[0].namespace,
            restored.bindings[1].namespace
        );
        assert_eq!(
            restored.bindings[0].prior_exports,
            original.bindings[0].prior_exports
        );
        let duplicate = value.bindings[1].clone();
        value.bindings.push(duplicate);
        assert_eq!(value.validate(), Err(BundleError::InvalidAttachmentBinding));
    }
    let mut unknown = original;
    unknown.bindings[0].candidates.clear();
    let mut another = unknown.bindings[0].clone();
    another.exported_from.import_id = "another-unknown-export".into();
    unknown.bindings.push(another);
    assert!(unknown.validate().is_ok());
    let restored: MessageRecord =
        serde_json::from_slice(&serde_json::to_vec(&unknown).unwrap()).unwrap();
    assert_eq!(restored, unknown);
    assert_eq!(restored.bindings[0].prior_exports.len(), 1);
    assert_ne!(
        restored.bindings[0].namespace,
        restored.bindings[0].exported_from
    );
}
