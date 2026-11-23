//! The browser capture log: each capture is a frame whose length and digest
//! let readers skip earlier captures without decoding them. A frame holds
//! the submitted JSON bytes unchanged.

pub const MAGIC: &[u8] = b"POLYLOGUE-CAPTURES/1\n";
/// Zero-padded decimal digits of the frame length.
const LENGTH_DIGITS: usize = 20;
const DIGEST_END: usize = LENGTH_DIGITS + 1 + blake3::OUT_LEN * 2;
/// The length, a space, the hex digest and a newline.
pub const HEADER_LEN: usize = DIGEST_END + 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Header {
    pub length: u64,
    pub digest: String,
}

impl Header {
    pub fn of(bytes: &[u8]) -> Self {
        Self {
            length: bytes.len() as u64,
            digest: blake3::hash(bytes).to_hex().to_string(),
        }
    }

    pub fn encode(&self) -> Vec<u8> {
        format!("{:0LENGTH_DIGITS$} {}\n", self.length, self.digest).into_bytes()
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() != HEADER_LEN
            || bytes[LENGTH_DIGITS] != b' '
            || bytes[DIGEST_END] != b'\n'
            || !bytes[..LENGTH_DIGITS].iter().all(u8::is_ascii_digit)
            || !bytes[LENGTH_DIGITS + 1..DIGEST_END]
                .iter()
                .all(u8::is_ascii_hexdigit)
        {
            return Err("invalid capture frame header".into());
        }
        let length = std::str::from_utf8(&bytes[..LENGTH_DIGITS])
            .map_err(|e| e.to_string())?
            .parse()
            .map_err(|e: std::num::ParseIntError| e.to_string())?;
        let digest = String::from_utf8(bytes[LENGTH_DIGITS + 1..DIGEST_END].to_vec())
            .map_err(|e| e.to_string())?;
        Ok(Self { length, digest })
    }

    pub fn verifies(&self, bytes: &[u8]) -> bool {
        self.matches(bytes.len() as u64, blake3::hash(bytes))
    }

    /// Checks the length and digest of bytes supplied in chunks.
    pub fn verifier(&self) -> Verifier<'_> {
        Verifier {
            header: self,
            length: Some(0),
            hash: blake3::Hasher::new(),
        }
    }

    fn matches(&self, length: u64, digest: blake3::Hash) -> bool {
        length == self.length && digest.to_hex().as_str() == self.digest
    }
}

pub struct Verifier<'a> {
    header: &'a Header,
    length: Option<u64>,
    hash: blake3::Hasher,
}

impl Verifier<'_> {
    pub fn update(&mut self, bytes: &[u8]) {
        self.length = self
            .length
            .and_then(|length| length.checked_add(bytes.len() as u64));
        self.hash.update(bytes);
    }

    /// True only when the complete frame was supplied and its digest matches.
    pub fn verifies(self) -> bool {
        self.length
            .is_some_and(|length| self.header.matches(length, self.hash.finalize()))
    }
}

#[cfg(test)]
mod tests;
