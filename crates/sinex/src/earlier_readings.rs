//! Readings stored before a parser reading carried one progress value.
//!
//! Earlier builds wrote a reading's progress as two flags, `at_end` and
//! `no_parser_yet`, in confirmed Sinex events and in capture sidecars of the
//! producer spool. Those records stay readable: the two flags map onto the
//! one value the current reading holds. The kind declarations that verify
//! such events are the frozen Source3 ones (`client::confirmed::source3`).
use crate::{Error, Result};
use polylogue_model::batch::StepProgress;
use serde_json::{Map, Value};

/// The progress value a reading stored as two flags stands for.
pub(crate) fn progress(at_end: bool, no_parser_yet: bool) -> StepProgress {
    if no_parser_yet {
        StepProgress::NoParserYet { at_end }
    } else {
        StepProgress::parsed(at_end)
    }
}

/// Rewrites one stored reading from its two flags into its progress value.
/// A reading that already holds a progress value is left as it is.
pub(crate) fn rewrite(reading: &mut Map<String, Value>) -> Result<()> {
    if reading.contains_key("progress") {
        return Ok(());
    }
    let mut flag = |name: &str| match reading.remove(name) {
        Some(Value::Bool(value)) => Ok(value),
        _ => Err(Error::Invalid(format!(
            "stored reading has no `progress` and no boolean `{name}`"
        ))),
    };
    let at_end = flag("at_end")?;
    let no_parser_yet = flag("no_parser_yet")?;
    reading.insert(
        "progress".into(),
        serde_json::to_value(progress(at_end, no_parser_yet))?,
    );
    Ok(())
}

/// Rewrites the reading of a confirmed event payload (`record.reading`).
pub(crate) fn rewrite_event(payload: &mut Value) -> Result<()> {
    let reading = payload
        .get_mut("record")
        .and_then(|record| record.get_mut("reading"))
        .and_then(Value::as_object_mut)
        .ok_or_else(|| Error::Invalid("stored event has no reading".into()))?;
    rewrite(reading)
}

/// Rewrites every parsed reading of a stored capture (`parsed[*]`).
pub(crate) fn rewrite_capture(capture: &mut Value) -> Result<()> {
    let Some(parsed) = capture.get_mut("parsed").and_then(Value::as_array_mut) else {
        return Ok(());
    };
    for reading in parsed {
        let reading = reading
            .as_object_mut()
            .ok_or_else(|| Error::Invalid("stored capture reading is not an object".into()))?;
        rewrite(reading)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
