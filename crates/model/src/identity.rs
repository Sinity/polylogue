use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;
use std::collections::BTreeMap;
use std::fmt;

#[derive(Debug, thiserror::Error)]
#[error(
    "origin must start with an ASCII letter or digit and contain only letters, digits, dot, underscore or hyphen"
)]
pub struct InvalidOrigin;

/// An open product name, independent of parser registration.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct Origin(String);
impl Origin {
    pub fn new(value: impl Into<String>) -> Result<Self, InvalidOrigin> {
        let value = value.into();
        if !value
            .as_bytes()
            .first()
            .is_some_and(u8::is_ascii_alphanumeric)
            || !value
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || b"-_.".contains(&c))
        {
            return Err(InvalidOrigin);
        }
        Ok(Self(value))
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl TryFrom<String> for Origin {
    type Error = InvalidOrigin;
    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}
impl From<Origin> for String {
    fn from(value: Origin) -> Self {
        value.0
    }
}
impl fmt::Display for Origin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum IdError {
    #[error("identity must not be empty")]
    Empty,
}

/// An opaque stored ID. Its components are supplied separately, never recovered by splitting it.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct SessionId(String);
impl SessionId {
    pub fn new(origin: &Origin, native_id: &str) -> Result<Self, IdError> {
        if native_id.is_empty() {
            return Err(IdError::Empty);
        }
        Ok(Self(format!("{origin}:{native_id}")))
    }
    pub fn from_stored(value: impl Into<String>) -> Result<Self, IdError> {
        let value = value.into();
        if value.is_empty() {
            return Err(IdError::Empty);
        }
        Ok(Self(value))
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl TryFrom<String> for SessionId {
    type Error = IdError;
    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::from_stored(value)
    }
}
impl From<SessionId> for String {
    fn from(value: SessionId) -> Self {
        value.0
    }
}

/// The separately supplied session relation survives IDs containing arbitrary colons.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "MessageIdWire", into = "MessageIdWire")]
pub struct MessageId {
    id: String,
    session: SessionId,
}
#[derive(Serialize, Deserialize)]
struct MessageIdWire {
    id: String,
    session: SessionId,
}
impl MessageId {
    pub fn new(session: SessionId, native_id: &str) -> Result<Self, IdError> {
        if native_id.is_empty() {
            return Err(IdError::Empty);
        }
        Ok(Self {
            id: format!("{}:{native_id}", session.as_str()),
            session,
        })
    }
    pub fn from_stored(id: impl Into<String>, session: SessionId) -> Result<Self, IdError> {
        let id = id.into();
        if id.is_empty() {
            return Err(IdError::Empty);
        }
        Ok(Self { id, session })
    }
    pub fn as_str(&self) -> &str {
        &self.id
    }
    pub fn session(&self) -> &SessionId {
        &self.session
    }
}
impl TryFrom<MessageIdWire> for MessageId {
    type Error = IdError;
    fn try_from(value: MessageIdWire) -> Result<Self, Self::Error> {
        Self::from_stored(value.id, value.session)
    }
}
impl From<MessageId> for MessageIdWire {
    fn from(value: MessageId) -> Self {
        Self {
            id: value.id,
            session: value.session,
        }
    }
}

/// The provider's block position, supplied independently of the opaque stored ID.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlockSlot {
    Item(u32),
    Reasoning(u32),
    Attachment(String),
}

/// A block keeps its message relation and slot without decoding its ID text.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "BlockIdWire", into = "BlockIdWire")]
pub struct BlockId {
    id: String,
    message: MessageId,
    slot: BlockSlot,
}
#[derive(Serialize, Deserialize)]
struct BlockIdWire {
    id: String,
    message: MessageId,
    slot: BlockSlot,
}
impl BlockId {
    pub fn new(message: MessageId, index: u32) -> Self {
        Self {
            id: format!("{}:{index}", message.as_str()),
            message,
            slot: BlockSlot::Item(index),
        }
    }
    pub fn reasoning(message: MessageId, index: u32) -> Self {
        Self {
            id: format!("{}:r{index}", message.as_str()),
            message,
            slot: BlockSlot::Reasoning(index),
        }
    }
    pub fn attachment(message: MessageId, native: &str) -> Result<Self, IdError> {
        if native.is_empty() {
            return Err(IdError::Empty);
        }
        Ok(Self {
            id: format!("{}:a{native}", message.as_str()),
            message,
            slot: BlockSlot::Attachment(native.to_owned()),
        })
    }
    pub fn from_stored(
        id: impl Into<String>,
        message: MessageId,
        slot: BlockSlot,
    ) -> Result<Self, IdError> {
        let id = id.into();
        if id.is_empty() || matches!(&slot, BlockSlot::Attachment(native) if native.is_empty()) {
            return Err(IdError::Empty);
        }
        Ok(Self { id, message, slot })
    }
    pub fn as_str(&self) -> &str {
        &self.id
    }
    pub fn message(&self) -> &MessageId {
        &self.message
    }
    pub fn slot(&self) -> &BlockSlot {
        &self.slot
    }
    pub fn index(&self) -> Option<u32> {
        match self.slot {
            BlockSlot::Item(n) | BlockSlot::Reasoning(n) => Some(n),
            BlockSlot::Attachment(_) => None,
        }
    }
}
impl TryFrom<BlockIdWire> for BlockId {
    type Error = IdError;
    fn try_from(value: BlockIdWire) -> Result<Self, Self::Error> {
        Self::from_stored(value.id, value.message, value.slot)
    }
}
impl From<BlockId> for BlockIdWire {
    fn from(value: BlockId) -> Self {
        Self {
            id: value.id,
            message: value.message,
            slot: value.slot,
        }
    }
}

#[derive(Debug, thiserror::Error)]
#[error("blob hash must be exactly 64 lowercase hexadecimal characters")]
pub struct InvalidBlobHash;

/// BLAKE3 of the exact bytes, with no text decoding or normalization.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct BlobHash([u8; 32]);
impl BlobHash {
    pub fn of(bytes: &[u8]) -> Self {
        Self(*blake3::hash(bytes).as_bytes())
    }
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }
    pub fn bytes(&self) -> &[u8; 32] {
        &self.0
    }
}
impl fmt::Display for BlobHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}
impl From<BlobHash> for String {
    fn from(value: BlobHash) -> Self {
        value.to_string()
    }
}
impl TryFrom<String> for BlobHash {
    type Error = InvalidBlobHash;
    fn try_from(value: String) -> Result<Self, Self::Error> {
        if value.len() != 64
            || !value
                .bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
        {
            return Err(InvalidBlobHash);
        }
        let mut bytes = [0; 32];
        for (slot, pair) in bytes.iter_mut().zip(value.as_bytes().chunks_exact(2)) {
            let nibble = |c| if c <= b'9' { c - b'0' } else { c - b'a' + 10 };
            *slot = nibble(pair[0]) * 16 + nibble(pair[1]);
        }
        Ok(Self(bytes))
    }
}

/// Canonicalize a raw provider record, omitting only adapter-declared top-level volatile fields.
/// RawValue preserves exponent spelling, which arbitrary_precision alone normalizes.
pub fn canonical_record(raw: &[u8], volatile: &[&str]) -> Result<Vec<u8>, serde_json::Error> {
    fn encode(raw: &RawValue, omit: &[&str], out: &mut String) -> Result<(), serde_json::Error> {
        let text = raw.get();
        match text.as_bytes()[0] {
            b'{' => {
                let fields: BTreeMap<String, &RawValue> = serde_json::from_str(text)?;
                out.push('{');
                let mut separator = "";
                for (key, value) in fields {
                    if omit.contains(&key.as_str()) {
                        continue;
                    }
                    out.push_str(separator);
                    out.push_str(&serde_json::to_string(&key)?);
                    out.push(':');
                    encode(value, &[], out)?;
                    separator = ",";
                }
                out.push('}');
            }
            b'[' => {
                let items: Vec<&RawValue> = serde_json::from_str(text)?;
                out.push('[');
                for (index, item) in items.into_iter().enumerate() {
                    if index != 0 {
                        out.push(',');
                    }
                    encode(item, &[], out)?;
                }
                out.push(']');
            }
            b'"' => out.push_str(&serde_json::to_string(&serde_json::from_str::<String>(
                text,
            )?)?),
            _ => out.push_str(text),
        }
        Ok(())
    }
    let raw: &RawValue = serde_json::from_slice(raw)?;
    let mut output = String::new();
    encode(raw, volatile, &mut output)?;
    Ok(output.into_bytes())
}

/// The caller counts preceding records with this same hash in this input revision.
pub fn fallback_hash(raw: &[u8], volatile: &[&str]) -> Result<String, serde_json::Error> {
    Ok(BlobHash::of(&canonical_record(raw, volatile)?).to_string()[..12].to_owned())
}
pub fn provisional_message_id(
    session: SessionId,
    raw: &[u8],
    volatile: &[&str],
    earlier_identical: u64,
) -> Result<MessageId, serde_json::Error> {
    let hash = fallback_hash(raw, volatile)?;
    Ok(MessageId {
        id: format!("{}:~{hash}-{earlier_identical}", session.as_str()),
        session,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn origins_are_open_and_validate_on_wire() {
        for name in ["future-product-7", "Future_Product", "7", "product.v2"] {
            let origin = Origin::new(name).unwrap();
            assert_eq!(origin.as_str(), name);
            assert_eq!(
                serde_json::from_str::<Origin>(&format!("{name:?}")).unwrap(),
                origin
            );
        }
        for invalid in [
            "",
            ".",
            "..",
            ".hidden",
            "-codex",
            "_codex",
            "codex:other",
            "two words",
            "codex/cli",
            "codex\n",
            "é",
        ] {
            assert!(Origin::new(invalid).is_err());
            assert!(serde_json::from_str::<Origin>(&format!("{invalid:?}")).is_err());
        }
    }
    #[test]
    fn native_colons_never_change_message_session_relation() {
        let session = SessionId::new(&Origin::new("future").unwrap(), "one:two/agent").unwrap();
        let message = MessageId::new(session.clone(), "three:four").unwrap();
        assert_eq!(message.as_str(), "future:one:two/agent:three:four");
        assert_eq!(message.session(), &session);
        let decoded: MessageId =
            serde_json::from_slice(&serde_json::to_vec(&message).unwrap()).unwrap();
        assert_eq!(decoded, message);
    }
    #[test]
    fn empty_ids_are_rejected_by_constructors_and_serde() {
        let origin = Origin::new("future").unwrap();
        assert_eq!(SessionId::new(&origin, ""), Err(IdError::Empty));
        assert_eq!(SessionId::from_stored(""), Err(IdError::Empty));
        let session = SessionId::new(&origin, "native:session").unwrap();
        assert_eq!(MessageId::new(session.clone(), ""), Err(IdError::Empty));
        assert_eq!(MessageId::from_stored("", session), Err(IdError::Empty));
        assert!(serde_json::from_str::<SessionId>(r#""""#).is_err());
        assert!(serde_json::from_str::<SessionId>("null").is_err());
        for wire in [
            r#"{"id":"","session":"future:native:session"}"#,
            r#"{"id":"native:message","session":""}"#,
            r#"{"session":"future:native:session"}"#,
        ] {
            assert!(serde_json::from_str::<MessageId>(wire).is_err());
        }
        // No trimming, colon splitting or grammar is imposed on nonempty native IDs.
        let opaque = MessageId::new(SessionId::new(&origin, ": ").unwrap(), " :").unwrap();
        let decoded: MessageId =
            serde_json::from_slice(&serde_json::to_vec(&opaque).unwrap()).unwrap();
        assert_eq!(decoded, opaque);
        assert_eq!(opaque.session().as_str(), "future:: ");
    }
    #[test]
    fn hash_is_exact_bytes_and_checked_on_wire() {
        assert_ne!(BlobHash::of(b"a\r\n"), BlobHash::of(b"a\n"));
        assert_ne!(
            BlobHash::of("é".as_bytes()),
            BlobHash::of("e\u{301}".as_bytes())
        );
        let hash = BlobHash::of(b"\0\xff");
        assert_eq!(hash.bytes(), blake3::hash(b"\0\xff").as_bytes());
        assert_eq!(
            serde_json::from_str::<BlobHash>(&serde_json::to_string(&hash).unwrap()).unwrap(),
            hash
        );
        assert!(serde_json::from_str::<BlobHash>("\"bad\"").is_err());
    }
    #[test]
    fn canonical_keys_and_raw_numbers_distinguish_provider_records() {
        let a = br#"{ "z": [1.00, 1e+02, -0], "a": {"b": true,"a":null}, "volatile": 9 }"#;
        let canonical = canonical_record(a, &["volatile"]).unwrap();
        assert_eq!(
            String::from_utf8(canonical).unwrap(),
            r#"{"a":{"a":null,"b":true},"z":[1.00,1e+02,-0]}"#
        );
        for token in ["1E2", "1e2", "1e+2", "-0", "1.00", "184467440737095516160"] {
            let raw = format!("{{\"n\":{token}}}");
            assert_eq!(
                canonical_record(raw.as_bytes(), &[]).unwrap(),
                raw.as_bytes()
            );
        }
        assert_eq!(
            fallback_hash(br#"{"b":2,"a":1}"#, &[]).unwrap(),
            fallback_hash(br#"{ "a":1, "b":2 }"#, &[]).unwrap()
        );
        for other in [
            br#"{"n":1.0}"#.as_slice(),
            br#"{"n":1e0}"#,
            br#"{"n":1E0}"#,
            br#"{"n":1e+0}"#,
            br#"{"n":"1"}"#,
            br#"{"n":null}"#,
            br#"{}"#,
        ] {
            assert_ne!(
                fallback_hash(br#"{"n":1}"#, &[]).unwrap(),
                fallback_hash(other, &[]).unwrap()
            );
        }
    }
    #[test]
    fn duplicate_counter_is_local_to_hash_not_record_position() {
        let session = SessionId::from_stored("codex:thread:with:colons").unwrap();
        let raw = br#"{"text":"ok"}"#;
        let first = provisional_message_id(session.clone(), raw, &[], 0).unwrap();
        let second = provisional_message_id(session, raw, &[], 1).unwrap();
        assert_ne!(first, second);
        assert!(second.as_str().ends_with("-1"));
    }
}
