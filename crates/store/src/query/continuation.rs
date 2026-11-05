//! Fixed-size commitments to replayable execution inputs.
use crate::selection::SelectionError;
use query_language::{cursor::CursorContext, time::ResolvedTime};

/// The full resolution vector belongs to the execution record. A continuation
/// commits to it and recomputes it from the frozen question and signed clock.
pub fn resolved_times_digest(times: &[ResolvedTime]) -> Result<String, SelectionError> {
    let mut digest = blake3::Hasher::new();
    digest.update(b"polylogue.query.resolved-times\0");
    serde_json::to_writer(&mut digest, times).map_err(|error| SelectionError::Invalid {
        kind: "internal",
        message: format!("resolved query times cannot be encoded: {error}"),
    })?;
    Ok(digest.finalize().to_hex().to_string())
}

pub fn compact_resolved_times(context: &mut CursorContext) -> Result<(), SelectionError> {
    if context.resolved_times.is_empty() && context.versions.contains_key("resolved_times") {
        return Ok(());
    }
    context.versions.insert(
        "resolved_times".into(),
        resolved_times_digest(&context.resolved_times)?,
    );
    context.resolved_times.clear();
    Ok(())
}

/// Call after restoring the authenticated clock and recompiling the same frozen
/// semantic object. Current-clock resolution is not a replay comparison.
pub fn check_resolved_times(
    context: &CursorContext,
    times: &[ResolvedTime],
) -> Result<(), SelectionError> {
    if !context.resolved_times.is_empty()
        || context.versions.get("resolved_times") != Some(&resolved_times_digest(times)?)
    {
        return Err(SelectionError::Invalid {
            kind: "stale_cursor",
            message: "the resolved query times changed".into(),
        });
    }
    Ok(())
}
