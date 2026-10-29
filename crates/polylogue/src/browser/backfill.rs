//! Backfill jobs as the operator starts them and the extension runs them.
//! The daemon's writer stores them ([`polylogue_store::backfill_jobs`]);
//! this module checks requests and shapes the answers.

use super::actions::Refused;
use polylogue_model::user::NewBackfillJob;
use polylogue_store::backfill_jobs::{BackfillJob, CLAIM_MS};
use serde_json::{Value, json};

/// The providers whose conversation lists the extension can page through,
/// by the extension's names for them.
pub fn providers() -> impl Iterator<Item = &'static str> {
    polylogue_sources::origin::ORIGINS
        .iter()
        .filter_map(|o| o.web.as_ref())
        .filter(|w| w.backfill)
        .filter_map(|w| w.extension_names.first().copied())
}

/// Checks a start request: a known provider, and `since` as a date or a
/// timestamp.
pub fn new_job(provider: &str, since: Option<&str>) -> Result<NewBackfillJob, Refused> {
    if !providers().any(|p| p == provider) {
        return Err(Refused(400, "unsupported_provider"));
    }
    let since = since
        .map(|text| {
            let at = text
                .parse::<jiff::Timestamp>()
                .or_else(|_| {
                    text.parse::<jiff::civil::Date>()
                        .and_then(|d| d.to_zoned(jiff::tz::TimeZone::UTC))
                        .map(|z| z.timestamp())
                })
                .map_err(|_| Refused(400, "since_is_a_date_or_time"))?;
            Ok(at.as_millisecond())
        })
        .transpose()?;
    Ok(NewBackfillJob {
        id: uuid::Uuid::now_v7().to_string(),
        provider: provider.to_owned(),
        since,
    })
}

fn iso(ms: i64) -> Value {
    jiff::Timestamp::from_millisecond(ms).map_or(Value::Null, |t| json!(t.to_string()))
}

/// One job as the CLI and the extension read it, at `now`.
pub fn view(job: &BackfillJob, now: i64) -> Value {
    let (status, holder) = job.status(now);
    let (counts, position) = job.progress();
    let last = job.events.last();
    json!({
        "id": job.id,
        "provider": job.provider,
        "since": job.since.map(iso),
        "status": status,
        "listed": counts.listed,
        "captured": counts.captured,
        "unchanged": counts.unchanged,
        "failed": counts.failed,
        "position": position,
        "detail": last.and_then(|e| e.detail.clone()),
        "created_at": iso(job.created_at),
        "updated_at": iso(last.map_or(job.created_at, |e| e.at)),
        "claimed_by": holder,
        "claim_expires_at": holder.and(last).map(|e| iso(e.at + CLAIM_MS)),
    })
}

#[cfg(test)]
mod tests;
