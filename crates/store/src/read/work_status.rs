//! Canonical physical-component routing status from this reader's pinned view.
use super::*;
impl Reader {
    pub fn work_component_status(
        &self,
        sessions: &[SessionId],
    ) -> Result<Option<polylogue_model::work_products::ComponentStatus>> {
        crate::primary::derived::component_status(self.conn()?, sessions)
    }
}
