//! The browser capture log: each capture is a frame whose length and digest
//! let readers skip earlier captures without decoding them. A frame holds
//! the exact submitted JSON bytes.

use std::io::{self, Read};

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

    /// Reads at most this frame's length and hashes bytes as they are consumed.
    pub fn verifier<R: Read>(&self, reader: R) -> Verifier<'_, R> {
        Verifier {
            header: self,
            reader: reader.take(self.length),
            hash: blake3::Hasher::new(),
        }
    }

    fn matches(&self, length: u64, digest: blake3::Hash) -> bool {
        length == self.length && digest.to_hex().as_str() == self.digest
    }
}

pub struct Verifier<'a, R> {
    header: &'a Header,
    reader: io::Take<R>,
    hash: blake3::Hasher,
}

impl<R> Verifier<'_, R> {
    /// True only when the complete frame was consumed and its digest matches.
    pub fn verifies(self) -> bool {
        self.header.matches(
            self.header.length - self.reader.limit(),
            self.hash.finalize(),
        )
    }
}

impl<R: Read> Read for Verifier<'_, R> {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        let length = self.reader.read(bytes)?;
        self.hash.update(&bytes[..length]);
        Ok(length)
    }
}

#[cfg(test)]
mod tests;
