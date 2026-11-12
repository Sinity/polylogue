//! Authoritative encoded extents in Sinex material acknowledgments.
pub(crate) fn validate(material: &polylogue_store::primary::Material) -> crate::Result<()> {
    material.descriptors().map_err(|_| {
        crate::Error::Invalid("material acknowledgment differs from its bytes".into())
    })?;
    Ok(())
}
