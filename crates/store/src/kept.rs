//! Validation of a kept result's stored body.
use crate::Result;

pub(crate) fn validate_body(id: &str, body: &str) -> Result<()> {
    polylogue_model::kept::body_metadata(body)
        .map(drop)
        .map_err(|error| {
            polylogue_model::Reason::BadKeptResult {
                id: id.into(),
                field: "rows_json".into(),
                problem: error.to_string(),
            }
            .into()
        })
}
