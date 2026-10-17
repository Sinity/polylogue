use crate::{NativeId, RetainedJson};
use serde::{Deserialize, Serialize};
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
        Self::from_native(origin, &NativeId::unicode(native_id))
    }
    pub fn from_native(origin: &Origin, native: &NativeId) -> Result<Self, IdError> {
        if native.as_str().is_empty() {
            return Err(IdError::Empty);
        }
        let id = if ordinary_session(native) {
            format!("{origin}:{}", native.as_str())
        } else {
            extended(&frame(
                1,
                &[origin.as_str().as_bytes(), &native_bytes(native)],
            ))
        };
        Ok(Self(id))
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
        Self::from_native(session, &NativeId::unicode(native_id))
    }
    pub fn from_native(session: SessionId, native: &NativeId) -> Result<Self, IdError> {
        if native.as_str().is_empty() {
            return Err(IdError::Empty);
        }
        let id = if !session.as_str().starts_with("~i1:") && ordinary_local(native) {
            format!("{}:{}", session.as_str(), native.as_str())
        } else {
            extended(&frame(
                3,
                &[&session_binary(&session), &native_bytes(native)],
            ))
        };
        Ok(Self { id, session })
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
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
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
    /// Resolve only this codec's supported exact block references. Extended
    /// reasoning and attachment forms are outside the admitted address scope.
    pub fn address(reference: &str) -> Option<(std::borrow::Cow<'_, str>, BlockSlot)> {
        if reference.starts_with("~i1:") {
            let bytes = decoded_frame(reference)?;
            let fields = frame_fields(&bytes, 5, 2)?;
            let message = message_from_binary(fields[0])?;
            let index = u32::try_from(u64::from_le_bytes(fields[1].try_into().ok()?)).ok()?;
            let block = Self::new(message.clone(), index);
            if block.as_str() != reference {
                return None;
            }
            return Some((
                std::borrow::Cow::Owned(message.as_str().to_owned()),
                BlockSlot::Item(index),
            ));
        }
        let (message, tail) = reference.rsplit_once(':').filter(|(m, _)| !m.is_empty())?;
        let (reasoning, number) = tail.strip_prefix('r').map_or((false, tail), |n| (true, n));
        let index: u32 = number.parse().ok()?;
        let canonical = if reasoning {
            format!("r{index}")
        } else {
            index.to_string()
        };
        if tail != canonical {
            return None;
        }
        Some((
            std::borrow::Cow::Borrowed(message),
            if reasoning {
                BlockSlot::Reasoning(index)
            } else {
                BlockSlot::Item(index)
            },
        ))
    }
    pub fn new(message: MessageId, index: u32) -> Self {
        Self {
            id: if message.as_str().starts_with("~i1:") {
                extended(&frame(
                    5,
                    &[&message_binary(&message), &u64::from(index).to_le_bytes()],
                ))
            } else {
                format!("{}:{index}", message.as_str())
            },
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

/// Immutable imports are named by exact bytes and canonical interpretation
/// context. Provider validation/admission precedes this pure framing primitive.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InputId(String);
impl InputId {
    pub fn imported(original: &BlobHash, format: &str, context: &[&str]) -> Self {
        fn field(h: &mut blake3::Hasher, value: &str) {
            h.update(&(value.len() as u64).to_le_bytes());
            h.update(value.as_bytes());
        }
        let mut h = blake3::Hasher::new();
        h.update(b"polylogue-import-v1\0");
        h.update(original.bytes());
        field(&mut h, format);
        h.update(&(context.len() as u64).to_le_bytes());
        for value in context {
            field(&mut h, value);
        }
        Self(h.finalize().to_hex().to_string())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl fmt::Display for InputId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Canonicalize a raw provider record, omitting only adapter-declared top-level volatile fields.
/// RawValue preserves exponent spelling, which arbitrary_precision alone normalizes.
pub fn canonical_record(raw: &[u8], volatile: &[&str]) -> Result<Vec<u8>, serde_json::Error> {
    let mut value =
        RetainedJson::parse(raw).map_err(<serde_json::Error as serde::de::Error>::custom)?;
    if let Some(fields) = value.as_object_mut() {
        for field in volatile {
            fields.remove(&NativeId::unicode(field));
        }
    }
    Ok(value.canonical())
}

/// The caller counts preceding records with this same hash in this input revision.
pub fn fallback_hash(raw: &[u8], volatile: &[&str]) -> Result<String, serde_json::Error> {
    Ok(BlobHash::of(&canonical_record(raw, volatile)?).to_string())
}
pub fn provisional_message_id(
    session: SessionId,
    raw: &[u8],
    volatile: &[&str],
    earlier_identical: u64,
) -> Result<MessageId, serde_json::Error> {
    let hash = fallback_hash(raw, volatile)?;
    Ok(MessageId {
        id: extended(&frame(
            4,
            &[
                &session_binary(&session),
                hash.as_bytes(),
                &earlier_identical.to_le_bytes(),
            ],
        )),
        session,
    })
}

// The identity codec alone owns framing. Provider strings are never split to
// discover relations; stored Message/Block relations remain explicit.
fn ordinary_session(native: &NativeId) -> bool {
    ordinary_local(native) && !native.as_str().contains(':')
}
fn ordinary_local(native: &NativeId) -> bool {
    native.encoding().is_none() && !native.as_str().starts_with('~')
}
pub(crate) fn native_bytes(native: &NativeId) -> Vec<u8> {
    let mut out = Vec::new();
    for item in char::decode_utf16(native.utf16()) {
        match item {
            Ok(c) => {
                let mut b = [0; 4];
                out.extend_from_slice(c.encode_utf8(&mut b).as_bytes());
            }
            Err(e) => {
                let n = u32::from(e.unpaired_surrogate());
                out.extend_from_slice(&[
                    (0xe0 | (n >> 12)) as u8,
                    (0x80 | ((n >> 6) & 63)) as u8,
                    (0x80 | (n & 63)) as u8,
                ]);
            }
        }
    }
    out
}
fn frame(tag: u8, fields: &[&[u8]]) -> Vec<u8> {
    let mut out = vec![1, tag];
    out.extend_from_slice(&(fields.len() as u64).to_le_bytes());
    for field in fields {
        out.extend_from_slice(&(field.len() as u64).to_le_bytes());
        out.extend_from_slice(field);
    }
    out
}
fn extended(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut out = String::from("~i1:");
    for byte in bytes {
        write!(&mut out, "{byte:02x}").expect("String formatting");
    }
    out
}
fn decoded_frame(value: &str) -> Option<Vec<u8>> {
    let hex = value.strip_prefix("~i1:")?;
    if hex.len() % 2 != 0 {
        return None;
    }
    hex.as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let text = std::str::from_utf8(pair).ok()?;
            u8::from_str_radix(text, 16).ok()
        })
        .collect()
}
// A fixed-shape decoder never allocates according to an untrusted field count.
fn frame_fields(bytes: &[u8], tag: u8, count: usize) -> Option<Vec<&[u8]>> {
    if bytes.get(..2)? != [1, tag] {
        return None;
    }
    let supplied = u64::from_le_bytes(bytes.get(2..10)?.try_into().ok()?);
    if supplied != count as u64 {
        return None;
    }
    let mut at = 10usize;
    let mut fields = Vec::with_capacity(count);
    for _ in 0..count {
        let end = at.checked_add(8)?;
        let length =
            usize::try_from(u64::from_le_bytes(bytes.get(at..end)?.try_into().ok()?)).ok()?;
        at = end;
        let end = at.checked_add(length)?;
        fields.push(bytes.get(at..end)?);
        at = end;
    }
    (at == bytes.len()).then_some(fields)
}
fn session_from_binary(bytes: &[u8]) -> Option<SessionId> {
    let fields = frame_fields(bytes, 1, 2)?;
    let origin = Origin::new(std::str::from_utf8(fields[0]).ok()?).ok()?;
    let native = NativeId::from_wtf8(fields[1]).ok()?;
    let session = SessionId::from_native(&origin, &native).ok()?;
    (session_binary(&session) == bytes).then_some(session)
}
fn message_from_binary(bytes: &[u8]) -> Option<MessageId> {
    match *bytes.get(1)? {
        3 => {
            let fields = frame_fields(bytes, 3, 2)?;
            let session = session_from_binary(fields[0])?;
            let native = NativeId::from_wtf8(fields[1]).ok()?;
            let message = MessageId::from_native(session, &native).ok()?;
            (message_binary(&message) == bytes).then_some(message)
        }
        4 => {
            let fields = frame_fields(bytes, 4, 3)?;
            let session = session_from_binary(fields[0])?;
            if fields[1].len() != 64
                || !fields[1]
                    .iter()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(b))
                || fields[2].len() != 8
            {
                return None;
            }
            Some(MessageId {
                id: extended(bytes),
                session,
            })
        }
        _ => None,
    }
}
fn session_binary(session: &SessionId) -> Vec<u8> {
    if let Some(bytes) = decoded_frame(session.as_str()) {
        return bytes;
    }
    // Only ordinary canonical references are generated by the constructors.
    let (origin, native) = session
        .as_str()
        .split_once(':')
        .unwrap_or(("", session.as_str()));
    frame(1, &[origin.as_bytes(), native.as_bytes()])
}
fn message_binary(message: &MessageId) -> Vec<u8> {
    if let Some(bytes) = decoded_frame(message.as_str()) {
        return bytes;
    }
    let local = message
        .as_str()
        .strip_prefix(message.session().as_str())
        .and_then(|s| s.strip_prefix(':'))
        .unwrap_or(message.as_str());
    frame(3, &[&session_binary(message.session()), local.as_bytes()])
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
        assert!(message.as_str().starts_with("~i1:"));
        assert_ne!(message.as_str(), "future:one:two/agent:three:four");
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
        assert!(opaque.session().as_str().starts_with("~i1:"));
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
        assert!(second.as_str().starts_with("~i1:"));
        assert_eq!(fallback_hash(raw, &[]).unwrap().len(), 64);
    }
}

#[cfg(test)]
mod import_identity_tests {
    use super::*;
    #[test]
    fn import_identity_has_independent_exact_framing_and_semantic_scope() {
        let raw = BlobHash::of(b"synthetic");
        let mut wire = b"polylogue-import-v1\0".to_vec();
        wire.extend_from_slice(raw.bytes());
        wire.extend_from_slice(&2u64.to_le_bytes());
        wire.extend_from_slice("é".as_bytes());
        wire.extend_from_slice(&2u64.to_le_bytes());
        wire.extend_from_slice(&1u64.to_le_bytes());
        wire.extend_from_slice(b"a");
        wire.extend_from_slice(&0u64.to_le_bytes());
        assert_eq!(
            InputId::imported(&raw, "é", &["a", ""]).to_string(),
            BlobHash::of(&wire).to_string()
        );
        assert_ne!(
            InputId::imported(&raw, "codex-rollout", &[]).as_str(),
            raw.to_string()
        );
        assert_ne!(
            InputId::imported(&raw, "claude-code-jsonl", &["claude-main", "a"]),
            InputId::imported(&raw, "claude-code-jsonl", &["claude-main", "b"])
        );
        assert_ne!(
            InputId::imported(&raw, "f", &["ab", "c"]),
            InputId::imported(&raw, "f", &["a", "bc"])
        );
        assert_ne!(
            InputId::imported(&raw, "f", &[]),
            InputId::imported(&raw, "f", &[""])
        );
    }
}
