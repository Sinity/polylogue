//! One lookup payload also supplies the final checkpoint's response identity.
use super::*;
use polylogue_model::{BlobHash, MessageId, MessageRow};
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;

#[derive(Serialize)]
struct Response<'a> {
    kind: &'static str,
    row: ResponseRow<'a>,
}

#[derive(Serialize)]
struct ResponseRow<'a> {
    #[serde(flatten)]
    row: &'a MessageRow,
    message_id: &'a MessageId,
}

pub(super) fn encode(value: &Value) -> Result<Vec<u8>> {
    let Value::Response { row } = value else {
        return crate::encoding::pack_json(value);
    };
    // The nested message's bytes are its existing canonical version identity.
    // Keep its small ID in this same payload so checkpointing can borrow the
    // JSON without decoding the message. Overwrites compute no unused hashes.
    let bytes = canonical_json::canonical_json(&Response {
        kind: "response",
        row: ResponseRow {
            row,
            message_id: &row.message.id,
        },
    })
    .map_err(crate::encoding::identity_encoding)?;
    #[cfg(test)]
    work(|w| {
        w.response_encodes += 1;
        w.encoded_response_bytes += bytes.len() as u64;
    });
    Ok(crate::encoding::pack_bytes(&bytes)?)
}

fn decode_json(bytes: &[u8]) -> Result<Value> {
    let value = crate::encoding::read_json(bytes)?;
    #[cfg(test)]
    if matches!(value, Value::Response { .. }) {
        work(|w| w.response_decodes += 1);
    }
    Ok(value)
}

pub(super) fn decode(bytes: &[u8]) -> Result<Value> {
    decode_json(&crate::encoding::unpack_bytes(bytes)?)
}

#[derive(Deserialize)]
struct CheckpointValue<'a> {
    kind: &'a str,
    #[serde(borrow)]
    row: Option<CheckpointRow<'a>>,
}

#[derive(Deserialize)]
struct CheckpointRow<'a> {
    message_id: MessageId,
    offset: u64,
    end: u64,
    #[serde(borrow)]
    message: &'a RawValue,
}

pub(super) fn reference(key: &Key, packed: &[u8]) -> Result<Reference> {
    let bytes = crate::encoding::unpack_bytes(packed)?;
    // Alias and state values keep their single ordinary decode. A mismatched
    // value still reaches the existing key/value-kind validation on application.
    if matches!(key, Key::Response { .. }) {
        #[cfg(test)]
        work(|w| w.response_views += 1);
        let mut reader = serde_json::Deserializer::from_slice(&bytes);
        reader.disable_recursion_limit();
        let value = CheckpointValue::deserialize(&mut reader)?;
        reader.end()?;
        if value.kind == "response" {
            let row = value.row.or_invalid("parser scratch response has no row")?;
            let message = row.message.get().as_bytes();
            #[cfg(test)]
            work(|w| {
                w.response_hashes += 1;
                w.hashed_response_bytes += message.len() as u64;
            });
            return Ok(Reference::Response {
                message: row.message_id,
                version: BlobHash::of(message),
                offset: row.offset,
                end: row.end,
            });
        }
    }
    Ok(match decode_json(&bytes)? {
        Value::Alias { target } => Reference::Alias { target },
        Value::State { value } => Reference::State { value },
        Value::Response { row } => Reference::Response {
            version: row.message.content_hash()?,
            message: row.message.id,
            offset: row.offset,
            end: row.end,
        },
    })
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Work {
    pub response_encodes: u64,
    pub encoded_response_bytes: u64,
    pub response_decodes: u64,
    pub response_views: u64,
    pub response_hashes: u64,
    pub hashed_response_bytes: u64,
}
#[cfg(test)]
thread_local! {
    static WORK: std::cell::Cell<Work> = const { std::cell::Cell::new(Work {
        response_encodes: 0, encoded_response_bytes: 0, response_decodes: 0,
        response_views: 0, response_hashes: 0, hashed_response_bytes: 0,
    }) };
}
#[cfg(test)]
fn work(update: impl FnOnce(&mut Work)) {
    WORK.with(|work| {
        let mut value = work.get();
        update(&mut value);
        work.set(value);
    });
}
#[cfg(test)]
pub(super) fn take_work() -> Work {
    WORK.with(|work| work.replace(Work::default()))
}
