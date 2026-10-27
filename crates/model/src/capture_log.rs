//! The browser capture log: each capture is a frame whose length and digest
//! let readers skip earlier captures without decoding them. A frame holds
//! the exact submitted JSON bytes.

pub const MAGIC: &[u8] = b"POLYLOGUE-CAPTURES/1\n";
pub const HEADER_LEN: usize = 86;

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
        format!("{:020} {}\n", self.length, self.digest).into_bytes()
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() != HEADER_LEN
            || bytes[20] != b' '
            || bytes[85] != b'\n'
            || !bytes[..20].iter().all(u8::is_ascii_digit)
            || !bytes[21..85].iter().all(u8::is_ascii_hexdigit)
        {
            return Err("invalid capture frame header".into());
        }
        let length = std::str::from_utf8(&bytes[..20])
            .map_err(|e| e.to_string())?
            .parse()
            .map_err(|e: std::num::ParseIntError| e.to_string())?;
        let digest = String::from_utf8(bytes[21..85].to_vec()).map_err(|e| e.to_string())?;
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
