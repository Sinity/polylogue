//! Checkpoint evidence is changed by admitted parser state, never by a retry.
use super::*;

#[derive(PartialEq)]
struct State {
    cursor: u64,
    version: String,
    status: String,
    detail: Option<String>,
    counts: [i64; 4],
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
    Ok(tx.query_row("SELECT parsed_upto,reading_version,status,detail,records,used,skipped,unreadable,commit_seq FROM parse_state WHERE input_pk=?1 AND chain=?2",
        params![input,chain],|r|Ok(Stored{state:State {cursor:r.get::<_,i64>(0)? as u64,version:r.get(1)?,status:r.get(2)?,detail:r.get(3)?,counts:[r.get(4)?,r.get(5)?,r.get(6)?,r.get(7)?]},commit:r.get(8)?})).optional()?)
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
        let status = if step.source_prefix_missing {
            "partial"
        } else if step.at_end && step.parsed_upto < batch.length {
            "waiting"
        } else if step.no_parser_yet {
            "no_parser_yet"
        } else if step.parsed_upto >= batch.length {
            "done"
        } else if step.at_end {
            "waiting"
        } else {
            "parsing"
        };
        let state = State {
            cursor: step.parsed_upto,
            version: step.reading_version.clone(),
            status: status.into(),
            detail: (status == "waiting").then(|| batch.length.to_string()),
            counts: totals,
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
        tx.prepare_cached("INSERT INTO parse_state(input_pk,chain,status,parsed_upto,records,used,skipped,unreadable,detail,reading_version,commit_seq)
            VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)
            ON CONFLICT(input_pk,chain) DO UPDATE SET status=excluded.status,parsed_upto=excluded.parsed_upto,
            records=excluded.records,used=excluded.used,skipped=excluded.skipped,unreadable=excluded.unreadable,
            detail=excluded.detail,reading_version=excluded.reading_version,commit_seq=excluded.commit_seq")?
            .execute(params![input,chain,self.state.status,self.state.cursor as i64,self.state.counts[0],self.state.counts[1],self.state.counts[2],self.state.counts[3],self.state.detail,self.state.version,self.commit])?;
        Ok(())
    }
}
