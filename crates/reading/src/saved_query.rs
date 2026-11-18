//! Named-query metadata comes from the same pinned reader as its rows.
use super::*;
use polylogue_model::{Reference, RevisionReference};

pub(super) fn header(
    reader: &Reader<'_>,
    target: &str,
    selection: &PreparedSelection<'_>,
) -> Result<Option<SavedQueryHeader>, Error> {
    let Ok(Reference::Query { key, .. }) = Reference::parse(target) else {
        return Ok(None);
    };
    let Some(definition) = selection
        .definitions()
        .iter()
        .find(|definition| definition.name == key)
    else {
        return Ok(None);
    };
    let name = key.rsplit_once('@').map_or(key.as_str(), |(name, _)| name);
    let Some(query) = reader.saved_query_record(name)? else {
        // Shipped definitions have no authored history.
        return Ok(None);
    };
    let selected_revision = Reference::Assertion(RevisionReference::exact(
        definition.root.clone(),
        definition.revision.clone(),
    ))
    .to_string();
    if !query
        .revisions
        .iter()
        .any(|revision| revision.reference == selected_revision)
    {
        // A withdrawn authored override may have fallen back to a shipped definition.
        return Ok(None);
    }
    let resave = reader
        .dependent_saved_queries(&query.dependents)?
        .into_iter()
        .map(|(name, definition)| {
            let mut step = serde_json::json!({"verb":"save","name":name});
            if let Some(denominator) = definition.denominator {
                step["denominator"] = serde_json::Value::String(denominator);
            }
            ReadyCall {
                request: command_surface::RunnableRequest::new(
                    "rows",
                    serde_json::json!({"selection":definition.text,"steps":[step]}),
                    crate::operation::new_run_id().to_string(),
                ),
            }
        })
        .collect();
    Ok(Some(SavedQueryHeader {
        query,
        selected_revision,
        resave,
    }))
}
