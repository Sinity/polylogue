//! The identity of a stored reading, independent of the input's route.
use super::*;
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ExportReadingKey {
    pub input: InputKey,
    pub chain: i64,
    pub parse_version: String,
    pub length: u64,
}
impl ExportReadingKey {
    pub fn validate(&self) -> crate::Result<()> {
        if self.input.id.is_empty()
            || self.input.format.is_empty()
            || self.chain < 1
            || self.parse_version.is_empty()
            || i64::try_from(self.length).is_err()
        {
            return Err(crate::StoreError::internal(
                "export reading key names no stored source",
            ));
        }
        Ok(())
    }
    pub fn identity(&self) -> BlobHash {
        let mut hash = blake3::Hasher::new();
        hash.update(b"polylogue.export-reading-key.v1\0");
        hash.update(
            &serde_json::to_vec(&self.parts()).expect("export reading identity serializes"),
        );
        BlobHash(*hash.finalize().as_bytes())
    }
    pub fn matches_reading(&self, reading: &Reading) -> bool {
        self.input.id == reading.input.id
            && self.input.format == reading.input.format
            && self.chain == reading.chain
            && self.parse_version == reading.parse_version
            && reading.source_length == Some(self.length)
    }
    fn parts(&self) -> (&str, &str, i64, &str, u64) {
        (
            &self.input.id,
            &self.input.format,
            self.chain,
            &self.parse_version,
            self.length,
        )
    }
}
impl PartialEq for ExportReadingKey {
    fn eq(&self, other: &Self) -> bool {
        self.parts() == other.parts()
    }
}
impl Eq for ExportReadingKey {}
impl Hash for ExportReadingKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.parts().hash(state);
    }
}
