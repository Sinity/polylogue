//! Authoritative encoded extents in Sinex material receipts.
use crate::Result;
pub(crate) fn decode_extents(encoded: &[Vec<u8>]) -> Result<Vec<hashed_blobs::Extent>> {
    encoded
        .iter()
        .map(|bytes| hashed_blobs::Extent::decode(bytes).map_err(crate::Error::from))
        .collect()
}
pub(crate) fn parser_version() -> u32 {
    3
}

pub(crate) fn validate(material: &polylogue_store::primary::Material) -> crate::Result<()> {
    let [encoded] = material.extents.as_slice() else {
        return Err(crate::Error::Invalid(
            "material receipt needs one committed extent".into(),
        ));
    };
    let extent = hashed_blobs::Extent::decode(encoded)?;
    if extent.len == 0 || extent.decoded_len != material.length || extent.hash.0 != material.hash.0
    {
        return Err(crate::Error::Invalid(
            "material receipt differs from its bytes".into(),
        ));
    }
    Ok(())
}
