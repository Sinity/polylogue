//! Provider JSON kept exactly: number spelling, lone UTF-16 surrogates and key
//! order survive. Reading a value as text is a separate, explicit step.
use crate::NativeId;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum RawJsonError {
    #[error("{context}: {message}")]
    Malformed {
        context: &'static str,
        message: &'static str,
    },
    #[error("{kind} is nested deeper than {limit}")]
    Capacity { kind: &'static str, limit: usize },
}
fn malformed(context: &'static str, _field: &str, message: &'static str) -> RawJsonError {
    RawJsonError::Malformed { context, message }
}
/// A JSON value kept exactly as the provider wrote it. Strings and object keys
/// keep their decoded UTF-16 units.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RawJson {
    Null,
    Bool(bool),
    Number(RawNumber),
    String(RawString),
    Array(Vec<RawJson>),
    Object(BTreeMap<NativeId, RawJson>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawNumber(String);
impl RawNumber {
    pub fn as_str(&self) -> &str {
        &self.0
    }
    pub fn is_i64(&self) -> bool {
        self.as_i64().is_some()
    }
    pub fn as_i64(&self) -> Option<i64> {
        self.0.parse().ok()
    }
}

#[derive(Clone, Debug)]
pub struct RawString {
    native: NativeId,
    text: String,
    repair: Option<(TextRepairs, u64, u64, u64)>,
    source: Option<StringSource>,
}
#[derive(Clone, Debug)]
struct StringSource {
    repairs: TextRepairs,
    record: u64,
    // Sparse boundaries map decoded UTF-8 offsets back to original JSON bytes.
    positions: Vec<(usize, u64)>,
}
impl StringSource {
    fn original(&self, decoded: usize) -> u64 {
        let &(at, raw) = self
            .positions
            .iter()
            .rev()
            .find(|(at, _)| *at <= decoded)
            .unwrap_or(&(0, self.record));
        raw + (decoded - at) as u64
    }
}
impl PartialEq for RawString {
    fn eq(&self, other: &Self) -> bool {
        self.native == other.native
    }
}
impl Eq for RawString {}
impl RawString {
    fn new(native: NativeId, repair: Option<(TextRepairs, u64, u64)>) -> Self {
        let units = native.utf16();
        let count = char::decode_utf16(units.iter().copied())
            .filter(Result::is_err)
            .count() as u64;
        Self {
            text: String::from_utf16_lossy(&units),
            native,
            repair: repair
                .filter(|_| count > 0)
                .map(|(log, record, at)| (log, record, at, count)),
            source: None,
        }
    }
    /// Decode a string field that holds JSON. Source positions map through the
    /// escapes; text repairs are counted only when the result is read as text.
    pub fn json(&self) -> Result<RawJson, RawJsonError> {
        self.decode_json(self.text.as_bytes(), 0)
    }
    /// Decode a JSON value that starts `skip` bytes into the string. Returns the
    /// value and its end, relative to `skip`; any trailing text is left alone.
    pub fn json_prefix(&self, skip: usize) -> Result<(RawJson, usize), RawJsonError> {
        let body = self
            .text
            .get(skip..)
            .ok_or_else(|| malformed("record", "", "invalid embedded JSON prefix"))?;
        let mut stream =
            serde_json::Deserializer::from_str(body).into_iter::<&serde_json::value::RawValue>();
        let raw = stream
            .next()
            .transpose()
            .map_err(|_| malformed("record", "", "invalid embedded JSON"))?
            .ok_or_else(|| malformed("record", "", "missing embedded JSON"))?;
        let start = raw.get().as_ptr() as usize - self.text.as_ptr() as usize;
        Ok((
            self.decode_json(raw.get().as_bytes(), start)?,
            stream.byte_offset(),
        ))
    }
    fn decode_json(&self, raw: &[u8], start: usize) -> Result<RawJson, RawJsonError> {
        if self.native.encoding().is_some() {
            return Err(malformed(
                "record",
                "",
                "embedded JSON string contains an unpaired code unit",
            ));
        }
        let mut value = RawJson::parse_with_repairs(raw, start as u64, &TextRepairs::default())?;
        fn rebase(value: &mut RawJson, parent: &StringSource) {
            match value {
                RawJson::String(s) => {
                    if let Some((_, _, at, count)) = &s.repair {
                        s.repair = Some((
                            parent.repairs.clone(),
                            parent.record,
                            parent.original(*at as usize),
                            *count,
                        ));
                    }
                    if let Some(source) = &mut s.source {
                        source.repairs = parent.repairs.clone();
                        source.record = parent.record;
                        for (_, raw) in &mut source.positions {
                            *raw = parent.original(*raw as usize);
                        }
                    }
                }
                RawJson::Array(a) => a.iter_mut().for_each(|v| rebase(v, parent)),
                RawJson::Object(o) => o.values_mut().for_each(|v| rebase(v, parent)),
                _ => {}
            }
        }
        if let Some(source) = &self.source {
            rebase(&mut value, source);
        } else {
            value = value.into_opaque();
        }
        Ok(value)
    }
    pub fn as_str(&self) -> &str {
        self.text()
    }
    pub fn native(&self) -> &NativeId {
        &self.native
    }
    pub fn text(&self) -> &str {
        if let Some((log, record, at, count)) = &self.repair {
            log.0
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert((*record, *at), *count);
        }
        &self.text
    }
}
impl From<NativeId> for RawString {
    fn from(s: NativeId) -> Self {
        Self::new(s, None)
    }
}
impl From<String> for RawString {
    fn from(s: String) -> Self {
        Self::new(NativeId::unicode(&s), None)
    }
}
impl From<&str> for RawString {
    fn from(s: &str) -> Self {
        Self::new(NativeId::unicode(s), None)
    }
}

/// Counts of repaired lone surrogates, separate from record counts. A token is
/// counted once even if it is read more than once.
#[derive(Clone, Debug, Default)]
pub struct TextRepairs(Arc<Mutex<BTreeMap<(u64, u64), u64>>>);
impl TextRepairs {
    pub fn entries(&self) -> Vec<TextRepair> {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .map(|(&(record, byte_offset), &count)| TextRepair {
                record,
                byte_offset,
                count,
            })
            .collect()
    }
    pub fn count(&self) -> u64 {
        self.entries().iter().map(|r| r.count).sum()
    }
}
impl PartialEq for TextRepairs {
    fn eq(&self, other: &Self) -> bool {
        self.entries() == other.entries()
    }
}
impl Eq for TextRepairs {}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextRepair {
    pub record: u64,
    pub byte_offset: u64,
    pub count: u64,
}

/// Maximum nesting depth. Deeper valid input is refused; supporting it needs a
/// non-recursive decoder.
pub const JSON_DEPTH_LIMIT: usize = 64;
fn admit_depth(raw: &[u8]) -> Result<(), RawJsonError> {
    let (mut depth, mut quoted, mut escaped) = (0usize, false, false);
    for &byte in raw {
        if quoted {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                quoted = false;
            }
        } else {
            match byte {
                b'"' => quoted = true,
                b'[' | b'{' => {
                    depth += 1;
                    if depth > JSON_DEPTH_LIMIT {
                        return Err(RawJsonError::Capacity {
                            kind: "JSON value",
                            limit: JSON_DEPTH_LIMIT,
                        });
                    }
                }
                b']' | b'}' => depth = depth.saturating_sub(1),
                _ => {}
            }
        }
    }
    Ok(())
}
impl RawJson {
    pub fn parse(raw: &[u8]) -> Result<Self, RawJsonError> {
        Self::parse_with_repairs(raw, 0, &TextRepairs::default())
    }
    pub fn parse_with_repairs(
        raw: &[u8],
        start: u64,
        repairs: &TextRepairs,
    ) -> Result<Self, RawJsonError> {
        let text =
            std::str::from_utf8(raw).map_err(|_| malformed("record", "", "invalid raw UTF-8"))?;
        admit_depth(raw)?;
        let _: &serde_json::value::RawValue =
            serde_json::from_slice(raw).map_err(|_| malformed("record", "", "invalid raw JSON"))?;
        Parser {
            text,
            at: 0,
            start,
            repairs: repairs.clone(),
        }
        .value()
    }
    pub fn get(&self, key: &str) -> Option<&Self> {
        self.as_object()?.get(&NativeId::unicode(key))
    }
    pub fn get_mut(&mut self, key: &str) -> Option<&mut Self> {
        self.as_object_mut()?.get_mut(&NativeId::unicode(key))
    }
    pub fn pointer(&self, pointer: &str) -> Option<&Self> {
        if pointer.is_empty() {
            return Some(self);
        }
        if !pointer.starts_with('/') {
            return None;
        }
        let mut node = self;
        for token in pointer[1..].split('/') {
            let token = token.replace("~1", "/").replace("~0", "~");
            node = match node {
                Self::Object(_) => node.get(&token)?,
                Self::Array(a) => a.get(token.parse::<usize>().ok()?)?,
                _ => return None,
            };
        }
        Some(node)
    }
    pub fn is_object(&self) -> bool {
        matches!(self, Self::Object(_))
    }
    pub fn is_array(&self) -> bool {
        matches!(self, Self::Array(_))
    }
    pub fn pointer_mut(&mut self, pointer: &str) -> Option<&mut Self> {
        if pointer.is_empty() {
            return Some(self);
        }
        if !pointer.starts_with('/') {
            return None;
        }
        let mut node = self;
        for token in pointer[1..].split('/') {
            let token = token.replace("~1", "/").replace("~0", "~");
            node = match node {
                Self::Object(_) => node.get_mut(&token)?,
                Self::Array(a) => a.get_mut(token.parse::<usize>().ok()?)?,
                _ => return None,
            };
        }
        Some(node)
    }
    pub fn as_array(&self) -> Option<&Vec<Self>> {
        if let Self::Array(a) = self {
            Some(a)
        } else {
            None
        }
    }
    pub fn as_array_mut(&mut self) -> Option<&mut Vec<Self>> {
        if let Self::Array(a) = self {
            Some(a)
        } else {
            None
        }
    }
    pub fn as_object(&self) -> Option<&BTreeMap<NativeId, Self>> {
        if let Self::Object(o) = self {
            Some(o)
        } else {
            None
        }
    }
    pub fn as_object_mut(&mut self) -> Option<&mut BTreeMap<NativeId, Self>> {
        if let Self::Object(o) = self {
            Some(o)
        } else {
            None
        }
    }
    pub fn native(&self) -> Option<&NativeId> {
        if let Self::String(s) = self {
            Some(s.native())
        } else {
            None
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        self.text()
    }
    pub fn as_native(&self) -> Option<&NativeId> {
        self.native()
    }
    pub fn text(&self) -> Option<&str> {
        if let Self::String(s) = self {
            Some(s.text())
        } else {
            None
        }
    }
    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }
    pub fn is_string(&self) -> bool {
        matches!(self, Self::String(_))
    }
    pub fn as_bool(&self) -> Option<bool> {
        if let Self::Bool(v) = self {
            Some(*v)
        } else {
            None
        }
    }
    pub fn as_i64(&self) -> Option<i64> {
        if let Self::Number(v) = self {
            v.0.parse().ok()
        } else {
            None
        }
    }
    pub fn as_u64(&self) -> Option<u64> {
        if let Self::Number(v) = self {
            v.0.parse().ok()
        } else {
            None
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        if let Self::Number(v) = self {
            v.0.parse::<f64>().ok().filter(|n| n.is_finite())
        } else {
            None
        }
    }
    /// An opaque payload is never read as text, so it has no repair log. Dropping
    /// the shared log also keeps a stored opaque field from carrying a whole
    /// parser step's diagnostics.
    pub fn opaque(&self) -> Self {
        self.clone().into_opaque()
    }
    pub fn into_opaque(mut self) -> Self {
        fn detach(value: &mut RawJson) {
            match value {
                RawJson::String(s) => {
                    s.repair = None;
                    s.source = None;
                }
                RawJson::Array(a) => a.iter_mut().for_each(detach),
                RawJson::Object(o) => o.values_mut().for_each(detach),
                _ => {}
            }
        }
        detach(&mut self);
        self
    }
    pub fn canonical(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        self.write(&mut bytes);
        bytes
    }
    pub fn remove(&mut self, path: &str) -> Result<(), RawJsonError> {
        if !path.starts_with('/') {
            return Err(malformed(
                "fallback",
                path,
                "volatile field must be a JSON pointer",
            ));
        }
        let parts = path[1..]
            .split('/')
            .map(|s| s.replace("~1", "/").replace("~0", "~"))
            .collect::<Vec<_>>();
        let mut node = self;
        for (i, part) in parts.iter().enumerate() {
            match node {
                Self::Object(m) => {
                    let k = NativeId::unicode(part);
                    if i + 1 == parts.len() {
                        m.remove(&k);
                        return Ok(());
                    }
                    let Some(next) = m.get_mut(&k) else {
                        return Ok(());
                    };
                    node = next;
                }
                _ => return Ok(()),
            }
        }
        Ok(())
    }
    fn write(&self, out: &mut Vec<u8>) {
        match self {
            Self::Null => out.extend(b"null"),
            Self::Bool(true) => out.extend(b"true"),
            Self::Bool(false) => out.extend(b"false"),
            Self::Number(n) => out.extend(n.0.as_bytes()),
            Self::String(s) => json_string(s.native(), out),
            Self::Array(a) => {
                out.push(b'[');
                for (i, n) in a.iter().enumerate() {
                    if i > 0 {
                        out.push(b',');
                    }
                    n.write(out);
                }
                out.push(b']');
            }
            Self::Object(m) => {
                out.push(b'{');
                for (i, (key, n)) in m.iter().enumerate() {
                    if i > 0 {
                        out.push(b',');
                    }
                    json_string(key, out);
                    out.push(b':');
                    n.write(out);
                }
                out.push(b'}');
            }
        }
    }
}
impl From<serde_json::Value> for RawJson {
    fn from(v: serde_json::Value) -> Self {
        use serde_json::Value as V;
        match v {
            V::Null => Self::Null,
            V::Bool(v) => Self::Bool(v),
            V::Number(n) => Self::Number(RawNumber(n.to_string())),
            V::String(s) => Self::String(s.into()),
            V::Array(a) => Self::Array(a.into_iter().map(Self::from).collect()),
            V::Object(o) => Self::Object(
                o.into_iter()
                    .map(|(k, v)| (NativeId::unicode(&k), Self::from(v)))
                    .collect(),
            ),
        }
    }
}
impl From<&RawJson> for RawJson {
    fn from(v: &RawJson) -> Self {
        v.clone()
    }
}
impl From<&str> for RawJson {
    fn from(v: &str) -> Self {
        Self::String(v.into())
    }
}
impl From<String> for RawJson {
    fn from(v: String) -> Self {
        Self::String(v.into())
    }
}
impl From<&String> for RawJson {
    fn from(v: &String) -> Self {
        Self::String(v.as_str().into())
    }
}
impl From<NativeId> for RawJson {
    fn from(v: NativeId) -> Self {
        Self::String(v.into())
    }
}
impl From<&NativeId> for RawJson {
    fn from(v: &NativeId) -> Self {
        Self::String(v.clone().into())
    }
}
impl From<bool> for RawJson {
    fn from(v: bool) -> Self {
        Self::Bool(v)
    }
}
impl<T: Into<RawJson>> From<Option<T>> for RawJson {
    fn from(v: Option<T>) -> Self {
        v.map(Into::into).unwrap_or(Self::Null)
    }
}
macro_rules! number_from { ($($t:ty),*) => { $(impl From<$t> for RawJson {
    fn from(v: $t) -> Self { Self::Number(RawNumber(v.to_string())) }
})* }; }
number_from!(i32, i64, u32, u64, usize);
impl Serialize for RawJson {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let text = String::from_utf8(self.canonical()).map_err(serde::ser::Error::custom)?;
        let raw =
            serde_json::value::RawValue::from_string(text).map_err(serde::ser::Error::custom)?;
        raw.serialize(serializer)
    }
}
impl<'de> Deserialize<'de> for RawJson {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let raw = Box::<serde_json::value::RawValue>::deserialize(d)?;
        Self::parse(raw.get().as_bytes())
            .map(Self::into_opaque)
            .map_err(serde::de::Error::custom)
    }
}
impl std::fmt::Display for RawJson {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&String::from_utf8_lossy(&self.canonical()))
    }
}
impl PartialEq<serde_json::Value> for RawJson {
    fn eq(&self, other: &serde_json::Value) -> bool {
        self == &Self::from(other.clone())
    }
}
impl std::ops::Index<&str> for RawJson {
    type Output = Self;
    fn index(&self, key: &str) -> &Self {
        self.get(key).unwrap_or(&Self::Null)
    }
}
impl std::ops::Index<usize> for RawJson {
    type Output = Self;
    fn index(&self, key: usize) -> &Self {
        self.as_array()
            .and_then(|a| a.get(key))
            .unwrap_or(&Self::Null)
    }
}
fn json_string(s: &NativeId, out: &mut Vec<u8>) {
    out.push(b'"');
    for c in char::decode_utf16(s.utf16()) {
        match c {
            Ok(c) => {
                let mut b = [0; 4];
                let t = c.encode_utf8(&mut b);
                let encoded = serde_json::to_string(t)
                    .unwrap_or_else(|_| unreachable!("a Unicode scalar serializes"));
                out.extend_from_slice(&encoded.as_bytes()[1..encoded.len() - 1]);
            }
            Err(e) => {
                out.extend_from_slice(format!("\\u{:04x}", e.unpaired_surrogate()).as_bytes())
            }
        }
    }
    out.push(b'"');
}
struct Parser<'a> {
    text: &'a str,
    at: usize,
    start: u64,
    repairs: TextRepairs,
}
impl Parser<'_> {
    fn ws(&mut self) {
        while self
            .text
            .as_bytes()
            .get(self.at)
            .is_some_and(u8::is_ascii_whitespace)
        {
            self.at += 1;
        }
    }
    fn value(&mut self) -> Result<RawJson, RawJsonError> {
        self.ws();
        let b = *self
            .text
            .as_bytes()
            .get(self.at)
            .ok_or_else(|| malformed("fallback", "", "missing JSON value"))?;
        match b {
            b'"' => {
                let (native, first, positions) = self.string()?;
                let mut string = RawString::new(
                    native,
                    first.map(|at| (self.repairs.clone(), self.start, at)),
                );
                string.source = Some(StringSource {
                    repairs: self.repairs.clone(),
                    record: self.start,
                    positions,
                });
                Ok(RawJson::String(string))
            }
            b'[' => {
                self.at += 1;
                self.ws();
                let mut a = Vec::new();
                if self.text.as_bytes().get(self.at) == Some(&b']') {
                    self.at += 1;
                    return Ok(RawJson::Array(a));
                }
                loop {
                    a.push(self.value()?);
                    self.ws();
                    let b = self.text.as_bytes()[self.at];
                    self.at += 1;
                    if b == b']' {
                        break;
                    }
                }
                Ok(RawJson::Array(a))
            }
            b'{' => {
                self.at += 1;
                self.ws();
                let mut m = BTreeMap::new();
                if self.text.as_bytes().get(self.at) == Some(&b'}') {
                    self.at += 1;
                    return Ok(RawJson::Object(m));
                }
                loop {
                    self.ws();
                    let key = self.string()?.0;
                    self.ws();
                    self.at += 1;
                    let value = self.value()?;
                    if m.insert(key, value).is_some() {
                        return Err(malformed("fallback", "", "duplicate decoded object key"));
                    }
                    self.ws();
                    let b = self.text.as_bytes()[self.at];
                    self.at += 1;
                    if b == b'}' {
                        break;
                    }
                }
                Ok(RawJson::Object(m))
            }
            _ => {
                let start = self.at;
                while self
                    .text
                    .as_bytes()
                    .get(self.at)
                    .is_some_and(|b| !b.is_ascii_whitespace() && !b",]}".contains(b))
                {
                    self.at += 1;
                }
                Ok(match &self.text[start..self.at] {
                    "null" => RawJson::Null,
                    "true" => RawJson::Bool(true),
                    "false" => RawJson::Bool(false),
                    number => RawJson::Number(RawNumber(number.to_owned())),
                })
            }
        }
    }
    fn string(&mut self) -> Result<(NativeId, Option<u64>, Vec<(usize, u64)>), RawJsonError> {
        self.at += 1;
        let mut units = Vec::new();
        let mut surrogates = Vec::new();
        let mut unit_positions = Vec::new();
        loop {
            let char_at = self.start + self.at as u64;
            let c = self.text[self.at..]
                .chars()
                .next()
                .ok_or_else(|| malformed("fallback", "", "unclosed string"))?;
            self.at += c.len_utf8();
            match c {
                '"' => {
                    let first = surrogates.into_iter().find_map(|(i, at)| {
                        let u = units[i];
                        let paired = if (0xd800..=0xdbff).contains(&u) {
                            units
                                .get(i + 1)
                                .is_some_and(|n| (0xdc00..=0xdfff).contains(n))
                        } else {
                            i > 0 && (0xd800..=0xdbff).contains(&units[i - 1])
                        };
                        (!paired).then_some(at)
                    });
                    let mut positions = Vec::new();
                    let (mut decoded, mut unit) = (0, 0);
                    for c in char::decode_utf16(units.iter().copied()) {
                        let raw = unit_positions[unit];
                        if positions
                            .last()
                            .is_none_or(|&(at, offset)| offset + (decoded - at) as u64 != raw)
                        {
                            positions.push((decoded, raw));
                        }
                        match c {
                            Ok(c) => {
                                decoded += c.len_utf8();
                                unit += c.len_utf16();
                            }
                            Err(_) => {
                                decoded += 3;
                                unit += 1;
                            }
                        }
                    }
                    if positions
                        .last()
                        .is_none_or(|&(at, offset)| offset + (decoded - at) as u64 != char_at)
                    {
                        positions.push((decoded, char_at));
                    }
                    return Ok((NativeId::from_utf16(&units), first, positions));
                }
                '\\' => {
                    let escape = self.text.as_bytes()[self.at];
                    self.at += 1;
                    let unit = match escape {
                        b'"' => 34,
                        b'\\' => 92,
                        b'/' => 47,
                        b'b' => 8,
                        b'f' => 12,
                        b'n' => 10,
                        b'r' => 13,
                        b't' => 9,
                        b'u' => {
                            let end = self.at + 4;
                            let n = u16::from_str_radix(&self.text[self.at..end], 16)
                                .map_err(|_| malformed("fallback", "", "invalid Unicode escape"))?;
                            self.at = end;
                            n
                        }
                        _ => return Err(malformed("fallback", "", "invalid escape")),
                    };
                    if (0xd800..=0xdfff).contains(&unit) {
                        surrogates.push((units.len(), char_at));
                    }
                    unit_positions.push(char_at);
                    units.push(unit);
                }
                _ => {
                    let mut b = [0; 2];
                    let encoded = c.encode_utf16(&mut b);
                    unit_positions.extend(std::iter::repeat_n(char_at, encoded.len()));
                    units.extend_from_slice(encoded);
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "raw_json_tests.rs"]
mod tests;
