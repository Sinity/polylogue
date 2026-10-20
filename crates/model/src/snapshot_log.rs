//! Framed snapshots: a length and digest let readers skip old payloads without
//! decoding them. Complete frame bodies are the exact submitted JSON bytes.

pub const MAGIC: &[u8] = b"POLYLOGUE-SNAPSHOTS/1\n";
pub const HEADER_LEN: usize = 151;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Header {
    pub length: u64,
    pub digest: String,
    /// Identity of the observed bytes, retained when payload text is excised.
    pub observed_digest: String,
}

impl Header {
    pub fn of(bytes: &[u8]) -> Self {
        Self {
            length: bytes.len() as u64,
            digest: blake3::hash(bytes).to_hex().to_string(),
            observed_digest: blake3::hash(bytes).to_hex().to_string(),
        }
    }

    pub fn encode(&self) -> Vec<u8> {
        format!(
            "{:020} {} {}\n",
            self.length, self.digest, self.observed_digest
        )
        .into_bytes()
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() != HEADER_LEN
            || bytes[20] != b' '
            || bytes[85] != b' '
            || bytes[150] != b'\n'
            || !bytes[86..150].iter().all(u8::is_ascii_hexdigit)
            || !bytes[..20].iter().all(u8::is_ascii_digit)
            || !bytes[21..85].iter().all(u8::is_ascii_hexdigit)
        {
            return Err("invalid snapshot frame header".into());
        }
        let length = std::str::from_utf8(&bytes[..20])
            .map_err(|e| e.to_string())?
            .parse()
            .map_err(|e: std::num::ParseIntError| e.to_string())?;
        let digest = String::from_utf8(bytes[21..85].to_vec()).map_err(|e| e.to_string())?;
        let observed_digest =
            String::from_utf8(bytes[86..150].to_vec()).map_err(|e| e.to_string())?;
        Ok(Self {
            length,
            digest,
            observed_digest,
        })
    }

    pub fn verifies(&self, bytes: &[u8]) -> bool {
        bytes.len() as u64 == self.length && blake3::hash(bytes).to_hex().as_str() == self.digest
    }
}
