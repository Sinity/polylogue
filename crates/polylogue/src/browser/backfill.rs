//! Backfill jobs as the operator starts them and the extension runs them.
//! The daemon's writer stores them ([`polylogue_store::backfill_jobs`]);
//! this module checks requests and shapes the answers.

use super::actions::{Refused, iso, providers_where};
use polylogue_model::user::NewBackfillJob;
use polylogue_model::user::{BACKFILL_CLAIM_MS, BackfillJob, BackfillStatus};
use serde::{Deserialize, Serialize};

/// The providers whose conversation lists the extension can page through,
/// by the extension's names for them.
pub fn providers() -> impl Iterator<Item = &'static str> {
    providers_where(|web| web.backfill)
}

/// Checks a start request: a known provider, and `since` as a date or a
/// timestamp.
pub fn new_job(provider: &str, since: Option<&str>) -> Result<NewBackfillJob, Refused> {
    if !providers().any(|p| p == provider) {
        return Err(Refused(polylogue_model::Code::UnsupportedProvider));
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
                .map_err(|_| Refused(polylogue_model::Code::SinceIsADateOrTime))?;
            Ok(at.as_millisecond())
        })
        .transpose()?;
    Ok(NewBackfillJob {
        id: uuid::Uuid::now_v7().to_string(),
        provider: provider.to_owned(),
        since,
    })
}

/// One backfill job as the CLI and the extension read it.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JobSummary {
    pub id: String,
    pub provider: String,
    /// Only conversations changed on or after this time.
    pub since: Option<String>,
    #[schemars(with = "String")]
    pub status: BackfillStatus,
    /// Conversations the provider listed.
    pub listed: u64,
    /// Conversations captured and stored.
    pub captured: u64,
    /// Conversations already in the archive as the provider has them.
    pub unchanged: u64,
    /// Conversations that could not be captured.
    pub failed: u64,
    /// Where the extension continues the provider's list.
    pub position: Option<String>,
    /// The latest report's detail.
    pub detail: Option<String>,
    pub created_at: Option<String>,
    pub updated_at: Option<String>,
    /// The extension instance whose claim has not expired.
    pub claimed_by: Option<String>,
    pub claim_expires_at: Option<String>,
}

/// One job as the CLI and the extension read it, at `now`.
pub fn view(job: &BackfillJob, now: i64) -> JobSummary {
    let (status, holder) = job.status(now);
    let (counts, position) = job.progress();
    let last = job.events.last();
    JobSummary {
        id: job.id.clone(),
        provider: job.provider.clone(),
        since: job.since.and_then(iso),
        status,
        listed: counts.listed,
        captured: counts.captured,
        unchanged: counts.unchanged,
        failed: counts.failed,
        position: position.map(str::to_owned),
        detail: last.and_then(|e| e.detail.clone()),
        created_at: iso(job.created_at),
        updated_at: iso(last.map_or(job.created_at, |e| e.at)),
        claimed_by: holder.map(str::to_owned),
        claim_expires_at: holder.and(last).and_then(|e| iso(e.at + BACKFILL_CLAIM_MS)),
    }
}

#[cfg(test)]
mod tests;
