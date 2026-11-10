//! Authoritative encoded extents in Sinex material receipts.
pub(crate) fn validate(material: &polylogue_store::primary::Material) -> crate::Result<()> {
    material
        .descriptors()
        .map_err(|_| crate::Error::Invalid("material receipt differs from its bytes".into()))?;
    Ok(())
}
