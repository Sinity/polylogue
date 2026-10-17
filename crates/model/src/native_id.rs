//! Exact native UTF-16 identity values with explicit transport encoding.
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("native identity: {message}")]
pub struct NativeIdError {
    pub message: &'static str,
}
fn malformed(_kind: &str, _value: &str, message: &'static str) -> NativeIdError {
    NativeIdError { message }
}
fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut s, b| {
            let _ = write!(s, "{b:02x}");
            s
        })
}
fn unhex(s: &str) -> Result<Vec<u8>, NativeIdError> {
    if s.len() % 2 != 0
        || !s
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(malformed(
            "framed",
            s,
            "expected lowercase hexadecimal pairs",
        ));
    }
    s.as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let digit = |b: u8| if b <= b'9' { b - b'0' } else { b - b'a' + 10 };
            Ok(digit(pair[0]) * 16 + digit(pair[1]))
        })
        .collect()
}
/// Exact decoded native string. Canonical WTF-8 includes lone surrogates,
/// while valid surrogate pairs have the ordinary scalar UTF-8 encoding.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NativeId {
    bytes: Vec<u8>,
    text: String,
}
impl NativeId {
    pub fn from_utf16(units: &[u16]) -> Self {
        let mut bytes = Vec::new();
        for c in char::decode_utf16(units.iter().copied()) {
            match c {
                Ok(c) => {
                    let mut b = [0; 4];
                    bytes.extend_from_slice(c.encode_utf8(&mut b).as_bytes());
                }
                Err(e) => {
                    let n = u32::from(e.unpaired_surrogate());
                    bytes.extend_from_slice(&[
                        (0xe0 | (n >> 12)) as u8,
                        (0x80 | ((n >> 6) & 63)) as u8,
                        (0x80 | (n & 63)) as u8,
                    ]);
                }
            }
        }
        let text =
            String::from_utf8(bytes.clone()).unwrap_or_else(|_| format!("~u:{}", hex(&bytes)));
        Self { bytes, text }
    }
    pub fn unicode(s: &str) -> Self {
        Self::from_utf16(&s.encode_utf16().collect::<Vec<_>>())
    }
    pub fn as_str(&self) -> &str {
        &self.text
    }
    /// Strict scalar Unicode, without escaped transport spelling.
    pub fn unicode_str(&self) -> Option<&str> {
        std::str::from_utf8(&self.bytes).ok()
    }
    pub fn utf16(&self) -> Vec<u16> {
        let mut out = Vec::new();
        let mut i = 0;
        while i < self.bytes.len() {
            let b = self.bytes[i];
            let (n, len) = if b < 0x80 {
                (u32::from(b), 1)
            } else if b < 0xe0 {
                (u32::from(b & 31), 2)
            } else if b < 0xf0 {
                (u32::from(b & 15), 3)
            } else {
                (u32::from(b & 7), 4)
            };
            let mut n = n;
            for &b in &self.bytes[i + 1..i + len] {
                n = (n << 6) | u32::from(b & 63);
            }
            if n <= 0xffff {
                out.push(n as u16);
            } else {
                let n = n - 0x10000;
                out.push((0xd800 + (n >> 10)) as u16);
                out.push((0xdc00 + (n & 1023)) as u16);
            }
            i += len;
        }
        out
    }
    pub fn from_wtf8(bytes: &[u8]) -> Result<Self, NativeIdError> {
        let mut units = Vec::new();
        let mut i = 0;
        while i < bytes.len() {
            let b = bytes[i];
            let (mut n, len, min) = match b {
                0..=0x7f => (u32::from(b), 1, 0),
                0xc2..=0xdf => (u32::from(b & 31), 2, 0x80),
                0xe0..=0xef => (u32::from(b & 15), 3, 0x800),
                0xf0..=0xf4 => (u32::from(b & 7), 4, 0x10000),
                _ => return Err(malformed("native", "", "invalid WTF-8")),
            };
            let tail = bytes
                .get(i + 1..i + len)
                .ok_or_else(|| malformed("native", "", "truncated WTF-8"))?;
            for &b in tail {
                if b & 0xc0 != 0x80 {
                    return Err(malformed("native", "", "invalid WTF-8 continuation"));
                }
                n = (n << 6) | u32::from(b & 63);
            }
            if n < min || n > 0x10ffff {
                return Err(malformed("native", "", "noncanonical WTF-8 scalar"));
            }
            if n <= 0xffff {
                units.push(n as u16);
            } else {
                let n = n - 0x10000;
                units.push((0xd800 + (n >> 10)) as u16);
                units.push((0xdc00 + (n & 1023)) as u16);
            }
            i += len;
        }
        let value = Self::from_utf16(&units);
        if value.bytes != bytes {
            return Err(malformed("native", "", "noncanonical surrogate pair"));
        }
        Ok(value)
    }
    /// Native fields use escaped transport spelling only when needed. Literal
    /// tilde strings cannot impersonate a surrogate escape.
    pub fn wire(&self) -> String {
        self.text.clone()
    }
    pub fn encoding(&self) -> Option<&'static str> {
        std::str::from_utf8(&self.bytes).is_err().then_some("wtf8")
    }
    pub fn from_wire(value: &str, encoding: Option<&str>) -> Result<Self, NativeIdError> {
        match encoding {
            None => Ok(Self::unicode(value)),
            Some("wtf8") => {
                let raw = value
                    .strip_prefix("~u:")
                    .ok_or_else(|| malformed("native", value, "expected explicit WTF-8 value"))?;
                let out = Self::from_wtf8(&unhex(raw)?)?;
                if out.encoding() != Some("wtf8") {
                    return Err(malformed(
                        "native",
                        value,
                        "WTF-8 discriminant requires a lone surrogate",
                    ));
                }
                Ok(out)
            }
            Some(_) => Err(malformed("native", value, "unknown native ID encoding")),
        }
    }
}
impl From<&str> for NativeId {
    fn from(s: &str) -> Self {
        Self::unicode(s)
    }
}
impl From<String> for NativeId {
    fn from(s: String) -> Self {
        Self::unicode(&s)
    }
}
impl From<&NativeId> for NativeId {
    fn from(s: &NativeId) -> Self {
        s.clone()
    }
}
impl std::fmt::Display for NativeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}
impl Serialize for NativeId {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self.encoding() {
            None => s.serialize_str(self.as_str()),
            Some(encoding) => {
                #[derive(Serialize)]
                struct Encoded<'a> {
                    value: &'a str,
                    encoding: &'a str,
                }
                Encoded {
                    value: self.as_str(),
                    encoding,
                }
                .serialize(s)
            }
        }
    }
}
impl<'de> Deserialize<'de> for NativeId {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Encoded {
            value: String,
            encoding: String,
        }
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Wire {
            Text(String),
            Encoded(Encoded),
        }
        match Wire::deserialize(d)? {
            Wire::Text(s) => Ok(Self::unicode(&s)),
            Wire::Encoded(e) => {
                Self::from_wire(&e.value, Some(&e.encoding)).map_err(serde::de::Error::custom)
            }
        }
    }
}
#[cfg(test)]
#[path = "native_id_tests.rs"]
mod tests;
