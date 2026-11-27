//! The chain's next state from its stored state and one step. A step that
//! does not start where the stored parse ends is refused before anything is
//! written; only a parser's failure of the chain's bytes makes it failed.
use super::*;
use polylogue_model::chain::{ChainState, Outcomes};

/// The state after `step`, which starts at the stored state's end (or at
/// the chain's first byte, replacing it) and reads through `length` stored
/// bytes when it reaches their end.
pub(super) fn next(previous: &ChainState, step: &ParsePart, length: u64) -> Result<ChainState> {
    let counts = if step.reset {
        Outcomes::default()
    } else {
        previous.counts()
    }
    .plus(&step.counts)
    .ok_or_else(|| StoreError::invalid("record accounting exceeds SQLite's integer range"))?;
    let complete_export = step.delivery.as_ref().is_some_and(|delivery| {
        matches!(
            delivery.source,
            polylogue_model::batch::ExportDeliverySource::Complete { .. }
        )
    });
    // A failure stands at the stored length it was met at.
    let seen_to = if step.at_end || complete_export || step.failure.is_some() {
        length
    } else {
        step.parsed_upto
    };
    if seen_to < step.parsed_upto {
        return Err(StoreError::internal("a step parsed past the bytes it saw"));
    }
    Ok(match &step.failure {
        Some(failure) => ChainState::Failed {
            version: step.parse_version.clone(),
            parsed_to: step.parsed_upto,
            seen_to,
            failure: failure.clone(),
            counts,
        },
        None => ChainState::Parsed {
            version: step.parse_version.clone(),
            parsed_to: step.parsed_upto,
            seen_to,
            resume: step.resume.clone(),
            counts,
        },
    })
}

/// Refuses a step that continues the chain's parse from anywhere but where
/// the stored parse ends: it was planned from an earlier state, and the
/// chain stays due from the stored one. A step from the first byte replaces
/// the parse.
pub(super) fn check_start(
    input: &str,
    chain: i64,
    previous: &ChainState,
    step: &ParsePart,
) -> Result<()> {
    if step.reset || step.delivery.is_some() {
        return Ok(());
    }
    let parsed_to = previous.parsed_to();
    let same_version = previous
        .version()
        .is_none_or(|version| version == step.parse_version);
    if step.start == parsed_to && same_version {
        return Ok(());
    }
    Err(StoreError::StalePart {
        input: input.to_owned(),
        chain,
        start: step.start,
        version: step.parse_version.clone(),
        parsed_to,
        stored_version: previous.version().map(str::to_owned),
    })
}
