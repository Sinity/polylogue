//! A chain's state as `parse_state` stores it: one row for a chain with a
//! parse, none for one without. Every reader reads the row back as the
//! model's [`ChainState`] through [`state`], and the index writer writes it
//! through [`write`], so the columns are interpreted in one place.

use crate::row::{Rows as _, row};
use crate::{Result, StoreError};
use polylogue_model::chain::{ChainState, Outcomes, ParserState};
use polylogue_model::{ChainFailure, ChainFailureKind};
use rusqlite::{Connection, params};

polylogue_model::stored_names! {
    /// The stored word of a chain that has a row.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) enum StateWord {
        Parsed => "parsed",
        Failed => "failed",
    }
}

row! {
    /// One chain's row of `parse_state`.
    pub(crate) struct ParseStateRow {
        pub input_pk: i64,
        pub chain: i64,
        pub state: StateWord,
        pub parser_version: String,
        pub parsed_to: u64,
        pub waiting: u64,
        pub used: u64,
        pub skipped: u64,
        pub unreadable: u64,
        pub resume: Option<Vec<u8>>,
        pub failure_kind: Option<ChainFailureKind>,
        pub failure_detail: Option<String>,
    }
    statements {
        pub(crate) const OF_CHAIN = "SELECT input_pk, chain, state, parser_version, parsed_to,
            waiting, used, skipped, unreadable, resume, failure_kind, failure_detail
            FROM parse_state WHERE input_pk = ?1 AND chain = ?2";
        pub(crate) const OF_INPUT_CHAIN = "SELECT p.input_pk, p.chain, p.state, p.parser_version,
            p.parsed_to, p.waiting, p.used, p.skipped, p.unreadable, p.resume, p.failure_kind,
            p.failure_detail
            FROM parse_state p JOIN input_ref r ON r.pk = p.input_pk WHERE r.id = ?1 AND p.chain = ?2";
        pub(crate) const UPSERT = "INSERT INTO parse_state(input_pk, chain, state, parser_version,
            parsed_to, waiting, used, skipped, unreadable, resume, failure_kind, failure_detail)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
            ON CONFLICT(input_pk, chain) DO UPDATE SET state = excluded.state,
            parser_version = excluded.parser_version, parsed_to = excluded.parsed_to,
            waiting = excluded.waiting, used = excluded.used, skipped = excluded.skipped,
            unreadable = excluded.unreadable, resume = excluded.resume,
            failure_kind = excluded.failure_kind, failure_detail = excluded.failure_detail";
    }
}

/// The columns of a `parse_state` row aliased `p`, in [`ParseStateRow`]'s
/// order, with the resume state read only as present or absent: what
/// planning needs, without decoding every chain's parser state.
pub(crate) const PLANNED_COLUMNS: &str = "p.input_pk, p.chain, p.state, p.parser_version,
    p.parsed_to, p.waiting, p.used, p.skipped, p.unreadable, p.resume IS NOT NULL,
    p.failure_kind, p.failure_detail";

impl ParseStateRow {
    /// The state of a joined `parse_state` row whose [`PLANNED_COLUMNS`]
    /// start at column `first` of `row`; a chain with no row reads as
    /// unparsed. A kept resume state reads as an empty one: the caller
    /// reads it from [`state`] when it goes on from it.
    pub(crate) fn planned(row: &rusqlite::Row<'_>, first: usize) -> Result<ChainState> {
        use crate::row::read;
        if read::<Option<i64>>(row, first, "input_pk")?.is_none() {
            return Ok(ChainState::Unparsed);
        }
        let kept: bool = read(row, first + 9, "resume")?;
        Self {
            input_pk: read(row, first, "input_pk")?,
            chain: read(row, first + 1, "chain")?,
            state: read(row, first + 2, "state")?,
            parser_version: read(row, first + 3, "parser_version")?,
            parsed_to: read(row, first + 4, "parsed_to")?,
            waiting: read(row, first + 5, "waiting")?,
            used: read(row, first + 6, "used")?,
            skipped: read(row, first + 7, "skipped")?,
            unreadable: read(row, first + 8, "unreadable")?,
            resume: None,
            failure_kind: read(row, first + 10, "failure_kind")?,
            failure_detail: read(row, first + 11, "failure_detail")?,
        }
        .state()
        .map(|mut state| {
            if let ChainState::Parsed { resume, .. } = &mut state
                && kept
            {
                *resume = Some(ParserState::default());
            }
            state
        })
    }

    /// The state this row stores.
    pub(crate) fn state(self) -> Result<ChainState> {
        let counts = Outcomes {
            used: self.used,
            skipped: self.skipped,
            unreadable: self.unreadable,
        };
        let seen_to = self
            .parsed_to
            .checked_add(self.waiting)
            .ok_or_else(|| StoreError::stored(5, "a stored parse length overflows"))?;
        Ok(match (self.state, self.failure_kind, self.failure_detail) {
            (StateWord::Parsed, None, None) => ChainState::Parsed {
                version: self.parser_version,
                parsed_to: self.parsed_to,
                seen_to,
                resume: self
                    .resume
                    .as_deref()
                    .map(crate::encoding::unpack_json::<ParserState>)
                    .transpose()?,
                counts,
            },
            (StateWord::Failed, Some(kind), Some(detail)) if self.resume.is_none() => {
                ChainState::Failed {
                    version: self.parser_version,
                    parsed_to: self.parsed_to,
                    seen_to,
                    failure: ChainFailure { kind, detail },
                    counts,
                }
            }
            _ => {
                return Err(StoreError::stored(
                    2,
                    "a parse state's failure columns disagree with its state",
                ));
            }
        })
    }

    /// The row that stores `state` for chain `chain` of input `input_pk`;
    /// `None` for a chain without a parse.
    pub(crate) fn of(input_pk: i64, chain: i64, state: &ChainState) -> Result<Option<Self>> {
        let (word, version, parsed_to, seen_to, counts, resume, failure) = match state {
            ChainState::Unparsed => return Ok(None),
            ChainState::Parsed {
                version,
                parsed_to,
                seen_to,
                resume,
                counts,
            } => (
                StateWord::Parsed,
                version,
                *parsed_to,
                *seen_to,
                counts,
                resume
                    .as_ref()
                    .map(crate::encoding::pack_json)
                    .transpose()?,
                None,
            ),
            ChainState::Failed {
                version,
                parsed_to,
                seen_to,
                failure,
                counts,
            } => (
                StateWord::Failed,
                version,
                *parsed_to,
                *seen_to,
                counts,
                None,
                Some(failure),
            ),
        };
        let waiting = seen_to
            .checked_sub(parsed_to)
            .ok_or_else(|| StoreError::internal("a chain state was seen to before its parse"))?;
        Ok(Some(Self {
            input_pk,
            chain,
            state: word,
            parser_version: version.clone(),
            parsed_to,
            waiting,
            used: counts.used,
            skipped: counts.skipped,
            unreadable: counts.unreadable,
            resume,
            failure_kind: failure.map(|failure| failure.kind),
            failure_detail: failure.map(|failure| failure.detail.clone()),
        }))
    }
}

/// The state of chain `chain` of input `input_pk`.
pub(crate) fn state(conn: &Connection, input_pk: i64, chain: i64) -> Result<ChainState> {
    conn.maybe(&ParseStateRow::OF_CHAIN, params![input_pk, chain])?
        .map_or(Ok(ChainState::Unparsed), ParseStateRow::state)
}

/// The state of chain `chain` of the input `archive.db` names `input`.
pub(crate) fn state_of_input(conn: &Connection, input: &str, chain: i64) -> Result<ChainState> {
    conn.maybe(&ParseStateRow::OF_INPUT_CHAIN, params![input, chain])?
        .map_or(Ok(ChainState::Unparsed), ParseStateRow::state)
}

/// Stores `state` as chain `chain`'s state; an unparsed chain has no row.
pub(crate) fn write(
    conn: &Connection,
    input_pk: i64,
    chain: i64,
    state: &ChainState,
) -> Result<()> {
    match ParseStateRow::of(input_pk, chain, state)? {
        Some(row) => {
            conn.insert(&ParseStateRow::UPSERT, &row)?;
        }
        None => {
            conn.prepare_cached("DELETE FROM parse_state WHERE input_pk = ?1 AND chain = ?2")?
                .execute(params![input_pk, chain])?;
        }
    }
    Ok(())
}

/// The rows this module declares statements for, for the every-statement test.
#[cfg(test)]
pub(crate) fn declared_rows(check: &mut crate::row::Check<'_>) {
    check.row::<ParseStateRow>();
}
