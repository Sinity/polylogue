use crate::parse::ExtractedBlob;
use base64::Engine as _;
use polylogue_model::{AttachmentFailure, BlobHash};

pub(crate) fn decode_bytes(data: Option<&str>) -> Result<Vec<u8>, AttachmentFailure> {
    let data = data.ok_or(AttachmentFailure::InvalidBase64)?;
    base64::engine::general_purpose::STANDARD
        .decode(data)
        .map_err(|_| AttachmentFailure::InvalidBase64)
}

/// AI Studio also writes line-wrapped, unpadded and URL-safe base64.
pub(crate) fn decode_wrapped(data: Option<&str>) -> Result<Vec<u8>, AttachmentFailure> {
    use base64::engine::general_purpose::{STANDARD, STANDARD_NO_PAD, URL_SAFE, URL_SAFE_NO_PAD};
    let data = data.ok_or(AttachmentFailure::InvalidBase64)?;
    let data: String = data.chars().filter(|c| !c.is_ascii_whitespace()).collect();
    STANDARD
        .decode(&data)
        .or_else(|_| STANDARD_NO_PAD.decode(&data))
        .or_else(|_| URL_SAFE.decode(&data))
        .or_else(|_| URL_SAFE_NO_PAD.decode(&data))
        .map_err(|_| AttachmentFailure::InvalidBase64)
}

/// A claimed inline value becomes a blob only after its bytes decode.
pub(crate) fn decode(
    data: Option<&str>,
    blobs: &mut Vec<ExtractedBlob>,
) -> Result<(BlobHash, u64), AttachmentFailure> {
    Ok(keep(decode_bytes(data)?, blobs))
}

pub(crate) fn keep(bytes: Vec<u8>, blobs: &mut Vec<ExtractedBlob>) -> (BlobHash, u64) {
    let hash = BlobHash::of(&bytes);
    let size = bytes.len() as u64;
    blobs.push(ExtractedBlob { hash, size, bytes });
    (hash, size)
}
