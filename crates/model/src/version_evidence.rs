//! The version hash of a message: a framed BLAKE3 over the raw bytes the
//! provider adapter supplies, in its order. Hash outside writer transactions;
//! after a restart, hashing starts again from the first unit.
use crate::BlobHash;

const PREFIX: &[u8] = b"\0polylogue-message-version\0v1\0";

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("version hash byte allowance used up")]
    TooLarge,
    #[error("evidence length or allowance overflow")]
    Overflow,
    #[error("operation does not match the evidence phase")]
    Phase,
    #[error("chunk exceeds declared evidence length")]
    Length,
    #[error("declared evidence unit count exhausted")]
    Units,
    #[error("evidence is incomplete")]
    Incomplete,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Ready,
    Tag { body: bool, left: u64, payload: u64 },
    Json { left: u64 },
    Body,
}
/// Incremental hasher with a byte allowance that counts framing too. Nothing
/// is buffered between calls.
pub struct Framer {
    hash: blake3::Hasher,
    state: State,
    remaining: u64,
    allowance: u64,
    used: u64,
}
impl Framer {
    /// Bytes `compound` hashes before the first unit.
    pub const fn compound_minimum_bytes() -> u64 {
        PREFIX.len() as u64 + 8
    }

    /// A message that is one whole record with no separate body: no prefix and
    /// no tag, just the record bytes.
    pub fn single_whole_record(payload_length: u64, allowance: u64) -> Self {
        Self {
            hash: blake3::Hasher::new(),
            state: if payload_length == 0 {
                State::Ready
            } else {
                State::Json {
                    left: payload_length,
                }
            },
            remaining: u64::from(payload_length != 0),
            allowance,
            used: 0,
        }
    }
    /// A message made of `units` tagged parts. Zero units is allowed.
    pub fn compound(units: u64, allowance: u64) -> Result<Self, Error> {
        let mut value = Self {
            hash: blake3::Hasher::new(),
            state: State::Ready,
            remaining: units,
            allowance,
            used: 0,
        };
        value.admit(
            u64::try_from(PREFIX.len())
                .map_err(|_| Error::Overflow)?
                .checked_add(8)
                .ok_or(Error::Overflow)?,
        )?;
        value.hash.update(PREFIX);
        value.hash.update(&units.to_be_bytes());
        Ok(value)
    }
    pub fn used_bytes(&self) -> u64 {
        self.used
    }
    /// Raise the allowance. A call refused with TooLarge changed nothing, so it
    /// can be retried after a grant.
    pub fn grant(&mut self, bytes: u64) -> Result<(), Error> {
        self.allowance = self.allowance.checked_add(bytes).ok_or(Error::Overflow)?;
        Ok(())
    }
    fn admit(&mut self, bytes: u64) -> Result<(), Error> {
        let used = self.used.checked_add(bytes).ok_or(Error::Overflow)?;
        if used > self.allowance {
            return Err(Error::TooLarge);
        }
        self.used = used;
        Ok(())
    }
    pub fn begin_json(&mut self, tag_length: u64, payload_length: u64) -> Result<(), Error> {
        self.begin(false, tag_length, payload_length)
    }
    /// A body unit's payload is always 40 bytes.
    pub fn begin_body(&mut self, tag_length: u64) -> Result<(), Error> {
        self.begin(true, tag_length, 40)
    }
    fn begin(&mut self, body: bool, tag_length: u64, payload: u64) -> Result<(), Error> {
        if self.state != State::Ready {
            return Err(Error::Phase);
        }
        if self.remaining == 0 {
            return Err(Error::Units);
        }
        self.admit(if tag_length == 0 { 17 } else { 9 })?;
        self.hash.update(&[u8::from(body)]);
        self.hash.update(&tag_length.to_be_bytes());
        if tag_length == 0 {
            self.hash.update(&payload.to_be_bytes());
            self.open_payload(body, payload);
        } else {
            self.state = State::Tag {
                body,
                left: tag_length,
                payload,
            };
        }
        Ok(())
    }
    /// Chunks end at UTF-8 character boundaries.
    pub fn tag(&mut self, chunk: &str) -> Result<(), Error> {
        let State::Tag {
            body,
            left,
            payload,
        } = self.state
        else {
            return Err(Error::Phase);
        };
        let size = u64::try_from(chunk.len()).map_err(|_| Error::Overflow)?;
        if size > left {
            return Err(Error::Length);
        }
        let cost = size
            .checked_add(if size == left { 8 } else { 0 })
            .ok_or(Error::Overflow)?;
        self.admit(cost)?;
        self.hash.update(chunk.as_bytes());
        if size == left {
            self.hash.update(&payload.to_be_bytes());
            self.open_payload(body, payload);
        } else {
            self.state = State::Tag {
                body,
                left: left - size,
                payload,
            };
        }
        Ok(())
    }
    fn open_payload(&mut self, body: bool, size: u64) {
        if body {
            self.state = State::Body;
        } else if size == 0 {
            self.complete();
        } else {
            self.state = State::Json { left: size };
        }
    }
    pub fn json(&mut self, chunk: &[u8]) -> Result<(), Error> {
        let State::Json { left } = self.state else {
            return Err(Error::Phase);
        };
        let size = u64::try_from(chunk.len()).map_err(|_| Error::Overflow)?;
        if size > left {
            return Err(Error::Length);
        }
        self.admit(size)?;
        self.hash.update(chunk);
        if size == left {
            self.complete();
        } else {
            self.state = State::Json { left: left - size };
        }
        Ok(())
    }
    pub fn body(&mut self, exact_length: u64, digest: BlobHash) -> Result<(), Error> {
        if self.state != State::Body {
            return Err(Error::Phase);
        }
        self.admit(40)?;
        self.hash.update(&exact_length.to_be_bytes());
        self.hash.update(digest.bytes());
        self.complete();
        Ok(())
    }
    fn complete(&mut self) {
        // Only reached for a started, unfinished unit.
        self.remaining -= 1;
        self.state = State::Ready;
    }
    pub fn finish(&self) -> Result<BlobHash, Error> {
        if self.state != State::Ready || self.remaining != 0 {
            return Err(Error::Incomplete);
        }
        Ok(BlobHash::from_bytes(*self.hash.finalize().as_bytes()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Expected bytes built by hand, deliberately not with the framing code
    // under test.
    fn oracle(units: &[(u8, &str, &[u8])]) -> BlobHash {
        let mut bytes = b"\0polylogue-message-version\0v1\0".to_vec();
        bytes.extend_from_slice(&(units.len() as u64).to_be_bytes());
        for (kind, tag, payload) in units {
            bytes.push(*kind);
            bytes.extend_from_slice(&(tag.len() as u64).to_be_bytes());
            bytes.extend_from_slice(tag.as_bytes());
            bytes.extend_from_slice(&(payload.len() as u64).to_be_bytes());
            bytes.extend_from_slice(payload);
        }
        BlobHash::of(&bytes)
    }
    fn jsons(units: &[(&str, &[u8])]) -> BlobHash {
        let mut f = Framer::compound(units.len() as u64, u64::MAX).unwrap();
        for (tag, payload) in units {
            f.begin_json(tag.len() as u64, payload.len() as u64)
                .unwrap();
            if !tag.is_empty() {
                f.tag(tag).unwrap();
            }
            if !payload.is_empty() {
                f.json(payload).unwrap();
            }
        }
        f.finish().unwrap()
    }
    #[test]
    fn singleton_is_raw_hash_and_empty_is_not_absent() {
        for raw in [b"".as_slice(), b"{}", b"null", b"\"\"", b"1.0", b"1e0"] {
            for cut in 0..=raw.len() {
                let mut f = Framer::single_whole_record(raw.len() as u64, u64::MAX);
                if cut != 0 {
                    f.json(&raw[..cut]).unwrap();
                }
                if cut != raw.len() {
                    f.json(&raw[cut..]).unwrap();
                }
                assert_eq!(f.finish().unwrap(), BlobHash::of(raw));
            }
        }
        assert_ne!(jsons(&[("record/0", b"{}")]), BlobHash::of(b"{}"));
        assert_ne!(jsons(&[("", b"")]), jsons(&[]));
        assert_ne!(jsons(&[("r", b"1.0")]), jsons(&[("r", b"1e0")]));
    }
    #[test]
    fn compound_matches_independent_oracle_at_every_cut() {
        let tag = "record/0";
        let raw = b"{\"n\":9007199254740993,\"x\":false}";
        let expected = oracle(&[(0, tag, raw)]);
        for tag_cut in 0..=tag.len() {
            for payload_cut in 0..=raw.len() {
                let mut f = Framer::compound(1, u64::MAX).unwrap();
                f.begin_json(tag.len() as u64, raw.len() as u64).unwrap();
                if tag_cut != 0 {
                    f.tag(&tag[..tag_cut]).unwrap();
                }
                if tag_cut != tag.len() {
                    f.tag(&tag[tag_cut..]).unwrap();
                }
                if payload_cut != 0 {
                    f.json(&raw[..payload_cut]).unwrap();
                }
                if payload_cut != raw.len() {
                    f.json(&raw[payload_cut..]).unwrap();
                }
                assert_eq!(f.finish().unwrap(), expected);
            }
        }
        let mut f = Framer::compound(1, u64::MAX).unwrap();
        f.begin_json(3, 0).unwrap();
        f.tag("é").unwrap();
        f.tag("x").unwrap();
        assert_eq!(f.finish().unwrap(), oracle(&[(0, "éx", b"")]));
    }
    #[test]
    fn framing_distinguishes_boundaries_tags_kinds_and_order() {
        assert_ne!(jsons(&[("a", b"bc")]), jsons(&[("ab", b"c")]));
        assert_ne!(
            jsons(&[("a", b"b"), ("c", b"d")]),
            jsons(&[("a", b"bc"), ("", b"d")])
        );
        assert_ne!(
            jsons(&[("a", b"b"), ("c", b"d")]),
            jsons(&[("c", b"d"), ("a", b"b")])
        );
        let digest = BlobHash::of(b"exact body");
        let mut payload = 10_u64.to_be_bytes().to_vec();
        payload.extend_from_slice(digest.bytes());
        let mut f = Framer::compound(2, u64::MAX).unwrap();
        f.begin_json(0, 2).unwrap();
        f.json(b"{}").unwrap();
        f.begin_body(4).unwrap();
        f.tag("body").unwrap();
        f.body(10, digest).unwrap();
        assert_eq!(
            f.finish().unwrap(),
            oracle(&[(0, "", b"{}"), (1, "body", &payload)])
        );
        assert_ne!(
            f.finish().unwrap(),
            jsons(&[("", b"{}"), ("body", &payload)])
        );
        let mut changed = Framer::compound(1, u64::MAX).unwrap();
        changed.begin_body(0).unwrap();
        changed.body(11, digest).unwrap();
        assert_ne!(changed.finish().unwrap(), oracle(&[(1, "", &payload)]));
    }
    #[test]
    fn omissions_lengths_counts_and_unavailable_required_body_refuse_finish() {
        let mut f = Framer::compound(2, u64::MAX).unwrap();
        assert_eq!(f.finish(), Err(Error::Incomplete));
        f.begin_json(1, 2).unwrap();
        assert_eq!(f.json(b"{}"), Err(Error::Phase));
        assert_eq!(f.tag("xx"), Err(Error::Length));
        f.tag("r").unwrap();
        assert_eq!(f.json(b"too long"), Err(Error::Length));
        f.json(b"{").unwrap();
        assert_eq!(f.finish(), Err(Error::Incomplete));
        f.json(b"}").unwrap();
        f.begin_body(0).unwrap();
        assert_eq!(f.finish(), Err(Error::Incomplete));
        assert_eq!(f.json(b"missing"), Err(Error::Phase));
        f.body(0, BlobHash::of(b"")).unwrap();
        assert!(f.finish().is_ok());
        assert_eq!(f.begin_json(0, 0), Err(Error::Units));
        assert_eq!(f.body(0, BlobHash::of(b"")), Err(Error::Phase));
    }
    #[test]
    fn refused_writes_change_nothing_for_framing_tags_payload_and_body() {
        let prefix = PREFIX.len() as u64 + 8;
        assert!(matches!(
            Framer::compound(1, prefix - 1),
            Err(Error::TooLarge)
        ));
        let mut f = Framer::compound(2, prefix).unwrap();
        assert_eq!(f.begin_json(1, 2), Err(Error::TooLarge));
        assert_eq!(f.used_bytes(), prefix);
        f.grant(9).unwrap();
        f.begin_json(1, 2).unwrap();
        assert_eq!(f.tag("r"), Err(Error::TooLarge));
        assert_eq!(f.used_bytes(), prefix + 9);
        f.grant(9).unwrap();
        f.tag("r").unwrap();
        assert_eq!(f.json(b"{}"), Err(Error::TooLarge));
        f.grant(2).unwrap();
        f.json(b"{}").unwrap();
        f.grant(17).unwrap();
        f.begin_body(0).unwrap();
        assert_eq!(f.body(4, BlobHash::of(b"body")), Err(Error::TooLarge));
        assert_eq!(f.finish(), Err(Error::Incomplete));
        f.grant(40).unwrap();
        f.body(4, BlobHash::of(b"body")).unwrap();
        let mut payload = 4_u64.to_be_bytes().to_vec();
        payload.extend_from_slice(BlobHash::of(b"body").bytes());
        assert_eq!(
            f.finish().unwrap(),
            oracle(&[(0, "r", b"{}"), (1, "", &payload)])
        );
        let used = f.used_bytes();
        assert_eq!(f.grant(u64::MAX), Err(Error::Overflow));
        assert_eq!(f.used_bytes(), used);
        assert!(f.finish().is_ok());
    }
}
