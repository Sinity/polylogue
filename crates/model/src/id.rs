use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum IdError {
    #[error("origin {0:?} must be lowercase ASCII letters, digits and '-'")]
    BadOrigin(String),
    #[error("native id must not be empty")]
    EmptyNative,
    #[error("id {0:?} has no origin prefix")]
    NoOrigin(String),
    #[error("blob hash {0:?} is not 64 hex digits")]
    BadHash(String),
}

/// The format family a session came from: `claude-code`, `codex`,
/// `chatgpt`. It never contains a colon, so `origin:native` splits at the
/// first colon.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Origin(String);

impl Origin {
    pub fn new(name: &str) -> Result<Self, IdError> {
        let ok = !name.is_empty()
            && name
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-');
        if ok {
            Ok(Self(name.to_owned()))
        } else {
            Err(IdError::BadOrigin(name.to_owned()))
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Origin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// An identifier as the source wrote it. Opaque; may contain colons.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct NativeId(String);

impl NativeId {
    pub fn new(id: &str) -> Result<Self, IdError> {
        if id.is_empty() {
            Err(IdError::EmptyNative)
        } else {
            Ok(Self(id.to_owned()))
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for NativeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

macro_rules! scoped_id {
    ($(#[$doc:meta])* $name:ident) => {
        $(#[$doc])*
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name {
            origin: Origin,
            native: NativeId,
        }

        impl $name {
            pub fn new(origin: Origin, native: NativeId) -> Self {
                Self { origin, native }
            }

            pub fn origin(&self) -> &Origin {
                &self.origin
            }

            pub fn native(&self) -> &NativeId {
                &self.native
            }

            /// Parses `origin:native`, splitting at the first colon.
            pub fn parse(text: &str) -> Result<Self, IdError> {
                let (origin, native) = text
                    .split_once(':')
                    .ok_or_else(|| IdError::NoOrigin(text.to_owned()))?;
                Ok(Self::new(Origin::new(origin)?, NativeId::new(native)?))
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}:{}", self.origin, self.native)
            }
        }

        impl Serialize for $name {
            fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                s.collect_str(self)
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                let text = String::deserialize(d)?;
                Self::parse(&text).map_err(serde::de::Error::custom)
            }
        }
    };
}

scoped_id!(
    /// `origin:native_id`. Stored session IDs never change.
    SessionId
);

scoped_id!(
    /// A message's identity within its origin. A message copied into another
    /// session's file keeps this ID, so it is stored once, in the session
    /// where it originated.
    MessageId
);

/// BLAKE3 of a blob's bytes.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlobHash(pub [u8; 32]);

impl BlobHash {
    pub fn of(bytes: &[u8]) -> Self {
        Self(*blake3::hash(bytes).as_bytes())
    }

    pub fn parse(hex: &str) -> Result<Self, IdError> {
        let bad = || IdError::BadHash(hex.to_owned());
        if hex.len() != 64 {
            return Err(bad());
        }
        let mut out = [0u8; 32];
        for (i, pair) in hex.as_bytes().chunks(2).enumerate() {
            let text = std::str::from_utf8(pair).map_err(|_| bad())?;
            out[i] = u8::from_str_radix(text, 16).map_err(|_| bad())?;
        }
        Ok(Self(out))
    }
}

impl fmt::Display for BlobHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.iter().try_for_each(|b| write!(f, "{b:02x}"))
    }
}

impl fmt::Debug for BlobHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BlobHash({self})")
    }
}

impl Serialize for BlobHash {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for BlobHash {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let text = String::deserialize(d)?;
        Self::parse(&text).map_err(serde::de::Error::custom)
    }
}

impl Serialize for Origin {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for Origin {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let text = String::deserialize(d)?;
        Self::new(&text).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests;
