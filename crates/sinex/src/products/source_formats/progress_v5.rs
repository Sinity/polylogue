//! Exact stored parse-progress source shape, independent of the successor enum.
use polylogue_model::RecordCounts;
use polylogue_store::primary::Reading;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

/// Complete parser and authored facts; rebuilding the cache requires no parsing.
#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(tag = "product", rename_all = "snake_case")]
#[schemars(rename = "Product")]
enum Product {
    ParseProgress {
        reading: Reading,
        counts: RecordCounts,
        reset: bool,
        comparison: Option<polylogue_store::tool_databases::Comparison>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        checkpoint: Option<polylogue_model::batch::ParserCheckpoint>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        semantic_event_ids: Option<std::collections::BTreeMap<usize, String>>,
    },
}

fn build_declaration() -> sinex_model::KindDeclaration {
    use sinex_model::kinds::{FieldHint, KeyRule, Normalization, RemovedRule, Search, TimeRule};
    use sinex_model::{Class, KindDeclaration, KindRef, ReplaceRule, TimeQuality};
    let mut facts = serde_json::to_value(schemars::schema_for!(Product)).expect("stored schema");
    let definitions = facts
        .as_object_mut()
        .expect("schema object")
        .remove("$defs")
        .unwrap_or_else(|| json!({}));
    let definitions = super::super::reachable_definitions(&facts, &definitions);
    KindDeclaration {
        kind: KindRef {
            name: "polylogue.parse_progress".into(),
            version: 5,
        },
        declaring_producer: "polylogue".into(),
        payload_schema: Some(json!({
            "$schema":"https://json-schema.org/draft/2020-12/schema", "type":"object",
            "properties":{
                "record_key":{"type":"string","minLength":1}, "record":facts,
                "text":{"type":"string"},"query":{"type":"string"},"rows":{},
                "refs":{"type":"array","items":{"type":"string"}},
                "at":{"type":["string","null"]},"started":{"type":["string","null"]}
            }, "required":["record_key","record"],"$defs":definitions
        })),
        normalization: Normalization::Json,
        key_rule: Some(KeyRule::Fields {
            fields: vec!["record_key".into()],
        }),
        replace_rule: Some(ReplaceRule::ReplaceWhenNewer),
        time_rule: Some(TimeRule::FirstPresent {
            fields: vec!["at".into(), "started".into()],
            quality: TimeQuality::Intrinsic,
        }),
        field_hints: ["record", "text", "query", "rows", "refs"]
            .into_iter()
            .map(|field| FieldHint {
                field: format!("/{field}"),
                class: Class::Guarded,
            })
            .collect(),
        removed_rule: Some(RemovedRule::Never),
        keep_known: vec![],
        owned_blobs: vec![],
        search: Search::None,
    }
}

pub(super) fn project(
    mut payload: serde_json::Value,
) -> crate::Result<polylogue_store::primary::Product> {
    let source: Product =
        serde_json::from_value(payload.get("record").cloned().ok_or_else(|| {
            crate::Error::Invalid("stored parse progress has no complete record".into())
        })?)?;
    // The version-5 source never stated a stopped attempt. Unknown extension
    // fields do not acquire successor semantics when its source is projected.
    payload["record"] = serde_json::to_value(source)?;
    super::super::decode("polylogue.parse_progress", payload)
}

static DECLARATION: std::sync::LazyLock<(sinex_model::KindDeclaration, String)> =
    std::sync::LazyLock::new(|| {
        let declaration = build_declaration();
        let hash = declaration.definition_hash();
        (declaration, hash)
    });
pub(super) fn declaration() -> &'static sinex_model::KindDeclaration {
    &DECLARATION.0
}
pub(super) fn definition_hash() -> &'static str {
    &DECLARATION.1
}
