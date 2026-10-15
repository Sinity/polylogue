use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;
use std::collections::BTreeMap;
use std::fmt;

#[derive(Debug, thiserror::Error)]
#[error("origin must be a nonempty plain ASCII name without namespace separators")]
pub struct InvalidOrigin;

/// An open product name, independent of parser registration.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct Origin(String);
impl Origin {
    pub fn new(value: impl Into<String>) -> Result<Self, InvalidOrigin> {
        let value = value.into();
        if value.is_empty()
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

/// An opaque stored ID. Its components are supplied separately, never recovered by splitting it.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SessionId(String);
impl SessionId {
    pub fn new(origin: &Origin, native_id: &str) -> Self {
        Self(format!("{origin}:{native_id}"))
    }
    pub fn from_stored(value: impl Into<String>) -> Self {
        Self(value.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// The separately supplied session relation survives IDs containing arbitrary colons.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MessageId {
    id: String,
    session: SessionId,
}
impl MessageId {
    pub fn new(session: SessionId, native_id: &str) -> Self {
        Self {
            id: format!("{}:{native_id}", session.as_str()),
            session,
        }
    }
    pub fn from_stored(id: impl Into<String>, session: SessionId) -> Self {
        Self {
            id: id.into(),
            session,
        }
    }
    pub fn as_str(&self) -> &str {
        &self.id
    }
    pub fn session(&self) -> &SessionId {
        &self.session
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
    Ok(MessageId::new(
        session,
        &format!("~{hash}-{earlier_identical}"),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn origins_are_open_and_validate_on_wire() {
        for name in ["future-product-7", "Future_Product", "7", "product.v2"] {
            assert!(Origin::new(name).is_ok());
        }
        for invalid in ["", "codex:other", "two words", "codex/cli", "codex\n", "é"] {
            assert!(Origin::new(invalid).is_err());
            assert!(serde_json::from_str::<Origin>(&format!("{invalid:?}")).is_err());
        }
    }
    #[test]
    fn native_colons_never_change_message_session_relation() {
        let session = SessionId::new(&Origin::new("future").unwrap(), "one:two/agent");
        let message = MessageId::new(session.clone(), "three:four");
        assert_eq!(message.as_str(), "future:one:two/agent:three:four");
        assert_eq!(message.session(), &session);
        let decoded: MessageId =
            serde_json::from_slice(&serde_json::to_vec(&message).unwrap()).unwrap();
        assert_eq!(decoded, message);
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
        let session = SessionId::from_stored("codex:thread:with:colons");
        let raw = br#"{"text":"ok"}"#;
        let first = provisional_message_id(session.clone(), raw, &[], 0).unwrap();
        let second = provisional_message_id(session, raw, &[], 1).unwrap();
        assert_ne!(first, second);
        assert!(second.as_str().ends_with("-1"));
    }
}
