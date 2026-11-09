//! Validation of a kept result's stored body.
use crate::{Result, StoreError};

pub(crate) fn validate_body(id: &str, body: &str) -> Result<()> {
    polylogue_model::kept::body_metadata(body)
        .map(drop)
        .map_err(|error| {
            StoreError::Invalid(format!("kept result {id:?} field rows_json: {error}"))
        })
}
