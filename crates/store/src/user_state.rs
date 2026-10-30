//! Lookup of an original authored request. Frozen records stream through
//! [`crate::Reader::visit_authored`] for export and copy.
use crate::{Reader, Result, StoreError};
use polylogue_model::user::UserRequest;

impl Reader<'_> {
    /// One committed request, including its resolved rows, targets or revision.
    pub fn user_request(&self, id: &str) -> Result<Option<UserRequest>> {
        use rusqlite::OptionalExtension;
        let packed: Option<Option<Vec<u8>>> = self
            .conn()?
            .query_row(
                "SELECT request FROM source_client_request WHERE id = ?1",
                [id],
                |row| row.get(0),
            )
            .optional()?;
        packed
            .map(|packed| {
                let packed = packed.ok_or_else(|| StoreError::Conflict(id.into()))?;
                Ok(serde_json::from_str(&crate::encoding::unpack_text(
                    &packed,
                )?)?)
            })
            .transpose()
    }
}
