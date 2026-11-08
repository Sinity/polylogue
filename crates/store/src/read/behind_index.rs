//! Revision chains whose index lags their stored bytes: what the loop
//! parses again from the archive, for watched and imported inputs alike.
use super::*;

/// One revision chain the index has not parsed to its stored length.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BehindChain {
    pub input: String,
    /// The chain holds the input's latest revision.
    pub current: bool,
    /// For an imported input, the format and path it was imported as.
    pub imported: Option<ImportedAs>,
    pub progress: ChainProgress,
    pub reading_changed: bool,
    pub stopped_attempt: Option<polylogue_model::batch::StoppedAttempt>,
    pub stopped_at_length: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportedAs {
    pub format: String,
    /// The path the parser reads the bytes as, never a file to open.
    pub path: Option<String>,
}

/// Whether chains whose refusal or failure is recorded at their stored length
/// count as behind. A recorded attempt stays settled until its actual program
/// or parser settings change; missing historical identity remains unrecorded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoppedChains {
    Settled,
    ForAttempt(polylogue_model::batch::StoppedAttempt),
}

/// The existing lag planner and completeness proof read this same row stream.
pub(super) struct ChainState {
    pub plan: BehindChain,
    pub origin: String,
    pub version: Option<String>,
    pub progress: Option<polylogue_model::batch::ParseProgress>,
}

const CHAIN_STATES: &str = r#"WITH inputs AS MATERIALIZED (
                SELECT input, max(number) AS number, max(produced = 'imported') AS imported
                FROM source_revision GROUP BY input
            ), chains AS (
                SELECT input, chain, max(length) AS length
                FROM source_revision GROUP BY input, chain
            ), first_paths AS MATERIALIZED (
                SELECT input, min(rowid) AS observation
                FROM source_observation GROUP BY input
            ), formats AS (
                SELECT o.input, o.path, CASE WHEN json_extract(o.details_json, '$.import_format')='export-file'
                         AND json_extract(o.details_json, '$.import.member')='rows.jsonl'
                         AND replace(o.path,char(92),'/') LIKE '%/archive.db!/rows.jsonl'
                     THEN 'polylogue-archive-rows'
                     ELSE json_extract(o.details_json, '$.import_format') END AS format,
                    row_number() OVER (PARTITION BY o.input
                        ORDER BY (json_extract(o.details_json, '$.import_format') = 'export-file'),
                                 o.rowid DESC) AS choice
                FROM source_observation o WHERE o.class = 'imported' AND json_valid(o.details_json)
                    AND json_type(o.details_json, '$.import_format') = 'text'
            )
            SELECT c.input, c.chain, c.length, l.chain = c.chain,
                s.imported,
                COALESCE(f.format, i.format),
                COALESCE(f.path, o.path),
                COALESCE(p.parsed_upto, 0), p.input_pk IS NOT NULL, COALESCE(p.status = 'waiting' AND p.at_length = c.length, 0), p.reading_version, p.status, p.at_length, p.reason, p.stopped_attempt
            FROM chains c JOIN source_input i ON i.id = c.input
            JOIN inputs s ON s.input = c.input
            JOIN source_revision l ON l.input = c.input AND l.number = s.number
            LEFT JOIN formats f ON f.input = c.input AND f.choice = 1
            LEFT JOIN first_paths fp ON fp.input = c.input
            LEFT JOIN source_observation o ON o.rowid = fp.observation
            LEFT JOIN input_ref ir ON ir.id = c.input
            LEFT JOIN parse_state p ON p.input_pk = ir.pk AND p.chain = c.chain
            WHERE (?2 OR (c.length > 0
              AND (p.input_pk IS NULL
                OR (NOT COALESCE(p.at_length = c.length AND (p.status = 'waiting' OR NOT ?3), 0)
                    AND p.parsed_upto < c.length)
                OR (?3 AND p.status IN ('refused','failed') AND p.at_length=c.length)
                OR EXISTS(SELECT 1 FROM json_each(?1) expected
                    WHERE json_extract(expected.value, '$[0]') = COALESCE(f.format, i.format)
                      AND json_extract(expected.value, '$[1]') IS NOT p.reading_version))))
            ORDER BY 5,(SELECT max(o.seen_at) FROM source_observation o WHERE o.input=c.input) DESC,c.input,c.chain DESC"#;

impl Reader<'_> {
    /// Every chain whose index is behind its stored bytes. Waiting at the exact
    /// recorded source length is current for that reading version; a refusal
    /// or failure recorded there is too, unless `stopped` asks for it again.
    pub fn chains_behind_index(
        &self,
        versions: &[(String, String)],
        stopped: StoppedChains,
    ) -> Result<Vec<BehindChain>> {
        let mut chains = Vec::new();
        self.visit_chains_behind_index(versions, stopped, &mut |chain| {
            chains.push(chain);
            Ok(std::ops::ControlFlow::Continue(()))
        })?;
        Ok(chains)
    }

    /// Streams lagging chains without collecting the source backlog.
    pub fn visit_chains_behind_index(
        &self,
        versions: &[(String, String)],
        stopped: StoppedChains,
        visit: &mut dyn FnMut(BehindChain) -> Result<std::ops::ControlFlow<()>>,
    ) -> Result<()> {
        self.visit_chain_states(versions, false, stopped, &mut |state| visit(state.plan))
    }

    pub(super) fn visit_chain_states(
        &self,
        versions: &[(String, String)],
        all: bool,
        stopped: StoppedChains,
        visit: &mut dyn FnMut(ChainState) -> Result<std::ops::ControlFlow<()>>,
    ) -> Result<()> {
        let token = self.scope_token();
        let mut statement = self.conn()?.prepare(CHAIN_STATES)?;
        let mut rows = statement.query(params![
            serde_json::to_string(versions)?,
            all,
            matches!(stopped, StoppedChains::ForAttempt(_))
        ])?;
        while let Some(row) = rows.next()? {
            token.check()?;
            let imported: bool = row.get(4)?;
            let origin: String = row.get(5)?;
            let version: Option<String> = row.get(10)?;
            let reading_changed = versions
                .iter()
                .find(|(name, _)| name == &origin)
                .is_some_and(|(_, expected)| version.as_ref() != Some(expected));
            let state = ChainState {
                plan: BehindChain {
                    input: row.get(0)?,
                    current: row.get(3)?,
                    imported: if imported {
                        Some(ImportedAs {
                            format: origin.clone(),
                            path: row.get(6)?,
                        })
                    } else {
                        None
                    },
                    reading_changed,
                    stopped_at_length: matches!(
                        row.get::<_, Option<String>>(11)?.as_deref(),
                        Some("failed" | "refused")
                    ) && row.get::<_, Option<u64>>(12)?
                        == Some(row.get::<_, u64>(2)?),
                    stopped_attempt: row
                        .get::<_, Option<Vec<u8>>>(14)?
                        .as_deref()
                        .map(crate::encoding::unpack_json)
                        .transpose()?,
                    progress: ChainProgress {
                        chain: row.get(1)?,
                        length: row.get::<_, i64>(2)? as u64,
                        parsed_upto: row.get::<_, i64>(7)? as u64,
                        parsed: row.get(8)?,
                        waiting: row.get(9)?,
                        arrival_rank: if imported { 2 } else { 0 },
                    },
                },
                origin,
                version,
                progress: row
                    .get::<_, Option<String>>(11)?
                    .map(|status| {
                        crate::read::stored_parse_progress(&status, row.get(12)?, row.get(13)?)
                    })
                    .transpose()?,
            };
            if !all
                && !state.plan.reading_changed
                && state.progress.as_ref().is_some_and(|progress| {
                    progress.stopped() && progress.at_length() == Some(state.plan.progress.length)
                })
                && matches!(stopped, StoppedChains::ForAttempt(attempt) if state.plan.stopped_attempt == Some(attempt))
            {
                continue;
            }
            if visit(state)?.is_break() {
                break;
            }
        }
        token.check()
    }
}

#[cfg(test)]
mod tests;
