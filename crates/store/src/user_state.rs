//! Lookup of an original authored request. Recorded records stream through
//! [`crate::Reader::visit_authored`] for export and copy.
use crate::ConnExt as _;
use crate::{Reader, Result};
use polylogue_model::user::UserRequest;

impl Reader<'_> {
    /// One committed request, including its resolved rows, targets or revision.
    pub fn user_request(&self, id: &str) -> Result<Option<UserRequest>> {
        let packed: Option<Vec<u8>> = self.conn()?.scalar_optional(
            "SELECT request FROM source_client_request WHERE id = ?1",
            [id],
        )?;
        packed
            .map(|packed| {
                Ok(serde_json::from_str(&crate::encoding::unpack_text(
                    &packed,
                )?)?)
            })
            .transpose()
    }
}
