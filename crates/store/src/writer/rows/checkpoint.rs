//! Checkpoint evidence is changed by admitted parser state, never by a retry.
use super::*;
use polylogue_model::batch::{ParseProgress, StepProgress};

#[derive(PartialEq)]
struct State {
    cursor: u64,
    version: String,
    progress: ParseProgress,
    counts: [i64; 4],
    checkpoint: Option<Vec<u8>>,
}

pub(super) struct Stored {
    state: State,
    commit: Option<i64>,
}
impl Stored {
    pub(super) fn cursor(&self) -> u64 {
        self.state.cursor
    }
    pub(super) fn version(&self) -> &str {
        &self.state.version
    }
}

pub(super) fn read(tx: &Transaction<'_>, input: i64, chain: i64) -> Result<Option<Stored>> {
    let row = tx.query_row("SELECT parsed_upto,reading_version,status,at_length,records,used,skipped,unreadable,commit_seq,checkpoint,reason FROM parse_state WHERE input_pk=?1 AND chain=?2",
        params![input,chain],|r|Ok((r.get::<_,i64>(0)? as u64,r.get::<_,String>(1)?,r.get::<_,String>(2)?,r.get::<_,Option<i64>>(3)?,[r.get::<_,i64>(4)?,r.get(5)?,r.get(6)?,r.get(7)?],r.get::<_,Option<i64>>(8)?,r.get::<_,Option<Vec<u8>>>(9)?,r.get::<_,Option<String>>(10)?))).optional()?;
    let Some((cursor, version, status, at_length, counts, commit, checkpoint, reason)) = row else {
        return Ok(None);
    };
    Ok(Some(Stored {
        state: State {
            cursor,
            version,
            progress: crate::read::stored_parse_progress(&status, at_length, reason)?,
            counts,
            checkpoint,
        },
        commit,
    }))
}

pub(super) struct Update {
    state: State,
    commit: Option<i64>,
}
impl Update {
    pub(super) fn new(
        batch: &Batch,
        step: &ParseStep,
        previous: Option<&Stored>,
        counter: i64,
        confirmed: bool,
        receipt: Option<&crate::wire_shapes::ReadingReceipt>,
    ) -> Result<Self> {
        let counts = [
            step.counts.records,
            step.counts.used.values().sum(),
            step.counts.skipped.values().sum(),
            step.counts.unreadable.values().sum(),
        ];
        let mut totals = [0; 4];
        for (index, value) in counts.into_iter().enumerate() {
            let value = i64::try_from(value).map_err(|_| {
                StoreError::Invalid("record accounting exceeds SQLite's integer range".into())
            })?;
            let prior = if step.reset {
                0
            } else {
                previous.map_or(0, |prior| prior.state.counts[index])
            };
            totals[index] = prior.checked_add(value).ok_or_else(|| {
                StoreError::Invalid("record accounting exceeds SQLite's integer range".into())
            })?;
        }
        let progress = if step.source_prefix_missing {
            ParseProgress::Partial
        } else if matches!(
            step.progress,
            StepProgress::Refused { .. } | StepProgress::Failed { .. }
        ) {
            ParseProgress::after(step.progress.clone(), step.parsed_upto, batch.length)
        } else if let Some(delivery) = &step.delivery {
            if matches!(
                delivery.source,
                polylogue_model::batch::ExportDeliverySource::Complete { .. }
            ) {
                ParseProgress::Done
            } else {
                ParseProgress::Parsing
            }
        } else {
            ParseProgress::after(step.progress.clone(), step.parsed_upto, batch.length)
        };
        if let Some(checkpoint) = &step.checkpoint
            && !checkpoint.matches(&step.reading_version, step.parsed_upto)
        {
            return Err(StoreError::Invalid(
                "parser checkpoint differs from its reading cursor".into(),
            ));
        }
        let state = State {
            cursor: step.parsed_upto,
            version: step.reading_version.clone(),
            progress,
            counts: totals,
            checkpoint: match (&step.checkpoint, &step.progress, previous) {
                (Some(checkpoint), _, _) => Some(crate::encoding::pack_json(checkpoint)?),
                // A refusal or failure leaves the cursor where it was, so the
                // parser state saved there still resumes it.
                (None, StepProgress::Refused { .. } | StepProgress::Failed { .. }, Some(prior))
                    if prior.state.cursor == step.parsed_upto
                        && prior.state.version == step.reading_version =>
                {
                    prior.state.checkpoint.clone()
                }
                (None, _, _) => None,
            },
        };
        let commit = if confirmed {
            receipt
                .or(step.receipt.as_ref())
                .map(|r| r.sequence)
                .filter(|n| *n > 0)
        } else if previous.is_some_and(|prior| prior.state == state) {
            previous.and_then(|prior| prior.commit)
        } else {
            Some(counter).filter(|n| *n > 0)
        };
        Ok(Self { state, commit })
    }

    pub(super) fn write(&self, tx: &Transaction<'_>, input: i64, chain: i64) -> Result<()> {
        let at_length = self
            .state
            .progress
            .at_length()
            .map(i64::try_from)
            .transpose()
            .map_err(|_| {
                StoreError::Invalid("a stored length exceeds SQLite's integer range".into())
            })?;
        tx.prepare_cached("INSERT INTO parse_state(input_pk,chain,status,parsed_upto,records,used,skipped,unreadable,at_length,reading_version,commit_seq,checkpoint,reason)
            VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13)
            ON CONFLICT(input_pk,chain) DO UPDATE SET status=excluded.status,parsed_upto=excluded.parsed_upto,
            records=excluded.records,used=excluded.used,skipped=excluded.skipped,unreadable=excluded.unreadable,
            at_length=excluded.at_length,reading_version=excluded.reading_version,commit_seq=excluded.commit_seq,checkpoint=excluded.checkpoint,reason=excluded.reason")?
            .execute(params![input,chain,self.state.progress.status(),self.state.cursor as i64,self.state.counts[0],self.state.counts[1],self.state.counts[2],self.state.counts[3],at_length,self.state.version,self.commit,self.state.checkpoint,self.state.progress.reason()])?;
        Ok(())
    }
}
