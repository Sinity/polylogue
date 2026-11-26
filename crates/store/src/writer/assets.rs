//! Export assets: their metadata and attachment bindings.
use super::*;

impl Inner {
    pub(super) fn install_assets(
        &mut self,
        staged: crate::export_items::ExportItems,
    ) -> Result<()> {
        crate::assets::install(&mut self.index, &self.archive, staged)
    }
}
