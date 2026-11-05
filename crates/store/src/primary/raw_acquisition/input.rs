//! Scheduling fidelity does not identify a physical acquisition stream.
use crate::InputKey;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub(super) struct Identity {
    id: String,
    format: String,
}

pub(super) fn serialize<S: Serializer>(input: &InputKey, serializer: S) -> Result<S::Ok, S::Error> {
    Identity {
        id: input.id.clone(),
        format: input.format.clone(),
    }
    .serialize(serializer)
}

pub(super) fn deserialize<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<InputKey, D::Error> {
    let Identity { id, format } = Identity::deserialize(deserializer)?;
    Ok(InputKey {
        id,
        format,
        arrival_rank: 0,
    })
}
