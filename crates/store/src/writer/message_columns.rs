//! Pure selected-message columns, prepared with the resolved value before it
//! reaches the writer. Row identities, source coordinates and commit state stay
//! with the writing transaction.

use polylogue_model::{Json, Message, WrittenBy};

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct MessageColumns {
    pub author: WrittenBy,
    pub role: String,
    pub prompt_kind: Option<String>,
    pub words: i64,
    pub attrs: Option<String>,
    pub provider_cost_nusd: Option<i64>,
    pub cost_nusd: Option<i64>,
}

impl MessageColumns {
    pub fn new(message: &Message) -> Self {
        let attrs = {
            let mut attrs = serde_json::Map::new();
            if !message.attrs.is_empty() {
                attrs.insert("attrs".into(), Json::Object(message.attrs.clone()));
            }
            if let Some(cost) = &message.reported_cost_usd {
                attrs.insert("reported_cost_usd".into(), cost.clone().into());
            }
            if !message.production.is_empty() {
                attrs.insert(
                    "production".into(),
                    serde_json::to_value(&message.production).unwrap_or_default(),
                );
            }
            (!attrs.is_empty()).then(|| Json::Object(attrs).to_string())
        };
        Self {
            author: WrittenBy::of(message),
            role: super::rows::name(&message.role),
            prompt_kind: message.prompt_kind.map(|kind| super::rows::name(&kind)),
            words: message
                .blocks
                .iter()
                .filter_map(polylogue_model::Block::text)
                .map(|text| text.split_whitespace().count() as i64)
                .sum(),
            attrs,
            // Provider amounts keep their original decimal spelling.
            provider_cost_nusd: message
                .reported_cost_usd
                .as_ref()
                .and_then(|amount| polylogue_model::money::nusd_from_usd(&amount.to_string())),
            cost_nusd: polylogue_model::Catalog::builtin()
                .cost_nusd(message.model.as_deref(), &message.usage),
        }
    }
}
