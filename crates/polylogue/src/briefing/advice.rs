//! Advice just before a tool runs. Only one thing is said, and only when it
//! is certain: the command about to run is a verification (a test, build or lint)
//! that failed here more than once since it last passed. Anything less
//! says nothing, so an agent learns to read every hint.

use super::{FAILING_WINDOW_MS, Prepared, day, scope_dirs};
use polylogue_model::{DeliveryEvent, Timestamp};
use polylogue_store::{Reader, StoreError};
use std::path::Path;

/// Failures since the last pass before a verification earns a hint.
const FAILURES_FOR_ADVICE: u64 = 2;

pub(super) fn advice(
    reader: &Reader<'_>,
    cwd: &Path,
    command: &str,
    session: Option<&polylogue_model::SessionId>,
    now: Timestamp,
) -> Result<Prepared, StoreError> {
    let mut prepared = Prepared {
        event: DeliveryEvent::Advice.as_str(),
        text: String::new(),
        refs: Vec::new(),
        left_out: Vec::new(),
        search_summary: None,
        empty: true,
        next: None,
        advice_id: None,
    };
    let Some(verification) = polylogue_store::derive::verification_title(command) else {
        return Ok(prepared);
    };
    let dirs = scope_dirs(cwd);
    let Some(failures) = reader
        .failing_commands(&dirs, now.unix_ms() - FAILING_WINDOW_MS)?
        .into_iter()
        .find(|f| f.command == verification)
    else {
        return Ok(prepared);
    };
    if failures.times < FAILURES_FOR_ADVICE {
        return Ok(prepared);
    }
    let state = serde_json::to_vec(&(
        session,
        &dirs,
        &verification,
        &failures.state,
        failures.times,
    ))
    .map_err(StoreError::from)?;
    let delivery_id = format!("advice:{}", blake3::hash(&state));
    if reader.delivery(&delivery_id)?.is_some() {
        return Ok(prepared);
    }
    let sessions = match failures.sessions {
        1 => "in one session".to_owned(),
        n => format!("across {n} sessions"),
    };
    let exit = failures
        .exit_code
        .map(|code| format!(", exit {code}"))
        .unwrap_or_default();
    prepared.text = format!(
        "Polylogue (delivery {delivery_id}): `{verification}` failed {} times here since {} {sessions}, with no pass since. \
         The last failure is in {}{exit}; `polylogue read {}` shows it.",
        failures.times,
        day(failures.first),
        failures.session,
        failures.session,
    );
    prepared.advice_id = Some(delivery_id);
    prepared.empty = false;
    Ok(prepared)
}

#[cfg(test)]
mod tests;
