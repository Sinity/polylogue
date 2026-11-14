//! Revision chains whose index lags their stored bytes: what the loop
//! parses again from the archive, for watched and imported inputs alike.
use super::*;

/// One revision chain the index has not parsed to its stored length.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BehindChain {
    pub input: String,
    /// The chain holds the input's latest revision.
    pub current: bool,
    /// The selected parser format and recorded path, independent of local discovery.
    pub source: StoredSource,
    pub progress: ChainProgress,
    pub reading_changed: bool,
    /// Native facts from a previous interpretation remain even without a local index.
    pub previous_reading: bool,
    pub stopped_attempt: Option<polylogue_model::batch::StoppedAttempt>,
    pub stopped_at_length: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredSource {
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

/// Which chains of each input participate in planning, before reading parser state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RevisionChains {
    All,
    /// The chain containing the greatest revision number, even if an older chain resumes.
    Newest,
    Older,
}

const CHAINS_BEHIND: &str = r#"first_paths AS MATERIALIZED (
                SELECT input, min(rowid) AS observation
                FROM observations GROUP BY input
            ), formats AS (
                SELECT o.input, o.path, json_extract(o.details_json, '$.import_format') AS format,
                    row_number() OVER (PARTITION BY o.input
                        ORDER BY (json_extract(o.details_json, '$.import_format') = 'export-file'),
                                 o.rowid DESC) AS choice
                FROM observations o WHERE o.class = 'imported' AND json_valid(o.details_json)
                    AND json_type(o.details_json, '$.import_format') = 'text'
            )
            SELECT c.input, c.chain, c.length, l.chain = c.chain,
                s.imported,
                COALESCE(f.format, i.format),
                COALESCE(f.path, o.path),
                COALESCE(p.parsed_upto, 0), p.input_pk IS NOT NULL, COALESCE(p.status = 'waiting' AND p.at_length = c.length, 0), p.reading_version, p.status, p.at_length, p.reason, p.stopped_attempt
            FROM chains c JOIN source_input i ON i.id = c.input
            JOIN inputs s ON s.input = c.input
            JOIN revisions l ON l.input = c.input AND l.number = s.number
            LEFT JOIN formats f ON f.input = c.input AND f.choice = 1
            LEFT JOIN first_paths fp ON fp.input = c.input
            LEFT JOIN observations o ON o.rowid = fp.observation
            LEFT JOIN input_ref ir ON ir.id = c.input
            LEFT JOIN parse_state p ON p.input_pk = ir.pk AND p.chain = c.chain
            WHERE (c.length > 0
              AND (p.input_pk IS NULL
                OR (NOT COALESCE(p.at_length = c.length AND (p.status = 'waiting' OR NOT ?2), 0)
                    AND p.parsed_upto < c.length)
                OR (?2 AND p.status IN ('refused','failed') AND p.at_length=c.length)
                OR EXISTS(SELECT 1 FROM json_each(?1) expected
                    WHERE json_extract(expected.value, '$[0]') = COALESCE(f.format, i.format)
                      AND json_extract(expected.value, '$[1]') IS NOT p.reading_version)))
            ORDER BY 5,(SELECT max(o.seen_at) FROM observations o WHERE o.input=c.input) DESC,c.input,c.chain DESC"#;

fn planning_sql(selected: bool, chains: RevisionChains) -> String {
    // Restrict histories before aggregating them. All-input planning
    // keeps its existing source-view access.
    let (revisions, observations) = if selected {
        (
            "SELECT r.* FROM chosen CROSS JOIN source_revision r ON r.input=chosen.input",
            "SELECT o.rowid,o.* FROM chosen CROSS JOIN source_observation o ON o.input=chosen.input",
        )
    } else {
        (
            "SELECT * FROM source_revision",
            "SELECT rowid,* FROM source_observation",
        )
    };
    let chain_rows = match chains {
        RevisionChains::All => "SELECT input,chain,max(length) AS length FROM revisions GROUP BY input,chain",
        // Retained histories may resume an earlier chain with a shorter revision.
        // Its planned length remains the maximum, not the latest revision length.
        RevisionChains::Newest => "SELECT s.input,l.chain,
            (SELECT max(r.length) FROM revisions r WHERE r.input=s.input AND r.chain=l.chain) AS length
            FROM inputs s JOIN revisions l ON l.input=s.input AND l.number=s.number",
        RevisionChains::Older => "SELECT r.input,r.chain,max(r.length) AS length
            FROM inputs s JOIN revisions l ON l.input=s.input AND l.number=s.number
            CROSS JOIN revisions r ON r.input=s.input WHERE r.chain<>l.chain
            GROUP BY r.input,r.chain",
    };
    format!(
        "WITH chosen AS MATERIALIZED (SELECT DISTINCT value AS input FROM json_each(?3)),
         revisions AS NOT MATERIALIZED ({revisions}),
         observations AS NOT MATERIALIZED ({observations}),
         inputs AS MATERIALIZED (
             SELECT input,max(number) AS number,max(produced='imported') AS imported
             FROM revisions GROUP BY input
         ), chains AS ({chain_rows}), {CHAINS_BEHIND}"
    )
}

impl Reader<'_> {
    /// Every chain whose index is behind its stored bytes. Waiting at the exact
    /// recorded source length is current for that reading version; a refusal
    /// or failure recorded there is too, unless `stopped` asks for it again.
    /// `selected=None` visits all inputs; an empty selection visits none.
    pub fn chains_behind_index(
        &self,
        versions: &[(String, String)],
        stopped: StoppedChains,
        chains: RevisionChains,
        selected: Option<&[String]>,
    ) -> Result<Vec<BehindChain>> {
        let mut found = Vec::new();
        self.visit_chains_behind_index(versions, stopped, chains, selected, &mut |chain| {
            found.push(chain);
            Ok(std::ops::ControlFlow::Continue(()))
        })?;
        Ok(found)
    }

    /// Streams lagging chains without collecting the source backlog.
    pub fn visit_chains_behind_index(
        &self,
        versions: &[(String, String)],
        stopped: StoppedChains,
        chains: RevisionChains,
        selected: Option<&[String]>,
        visit: &mut dyn FnMut(BehindChain) -> Result<std::ops::ControlFlow<()>>,
    ) -> Result<()> {
        let token = self.scope_token();
        crate::fault::check("read.chains_behind_index", "")?;
        let connection = self.conn()?;
        let sql = planning_sql(selected.is_some(), chains);
        let mut statement = connection.prepare(&sql)?;
        let mut rows = statement.query(params![
            serde_json::to_string(versions)?,
            matches!(stopped, StoppedChains::ForAttempt(_)),
            selected.map(serde_json::to_string).transpose()?,
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
            let length = u64::try_from(row.get::<_, i64>(2)?)
                .map_err(|_| StoreError::Invalid("stored chain length is negative".into()))?;
            let parsed_upto = u64::try_from(row.get::<_, i64>(7)?)
                .map_err(|_| StoreError::Invalid("stored parser cursor is negative".into()))?;
            let progress = row
                .get::<_, Option<String>>(11)?
                .map(|status| {
                    crate::read::stored_parse_progress(&status, row.get(12)?, row.get(13)?)
                })
                .transpose()?;
            let stopped_at_length = progress
                .as_ref()
                .is_some_and(|progress| progress.stopped() && progress.at_length() == Some(length));
            let input: String = row.get(0)?;
            let chain_number: i64 = row.get(1)?;
            let previous_reading = if self.pool.archive.is_primary() {
                versions
                    .iter()
                    .find(|(name, _)| name == &origin)
                    .map(|(_, version)| {
                        crate::primary::products::previous_reading_bound(
                            self.conn()?,
                            &input,
                            chain_number,
                            version,
                        )
                    })
                    .transpose()?
                    .flatten()
                    .is_some()
            } else {
                false
            };
            let chain = BehindChain {
                input,
                current: row.get(3)?,
                source: StoredSource {
                    format: origin,
                    path: row.get(6)?,
                },
                reading_changed,
                previous_reading,
                stopped_at_length,
                stopped_attempt: row
                    .get::<_, Option<Vec<u8>>>(14)?
                    .as_deref()
                    .map(crate::encoding::unpack_json)
                    .transpose()?,
                progress: ChainProgress {
                    chain: chain_number,
                    length,
                    parsed_upto,
                    parsed: row.get(8)?,
                    waiting: row.get(9)?,
                    imported,
                },
            };
            if !chain.reading_changed
                && chain.stopped_at_length
                && matches!(stopped, StoppedChains::ForAttempt(attempt) if chain.stopped_attempt == Some(attempt))
            {
                continue;
            }
            if visit(chain)?.is_break() {
                break;
            }
        }
        token.check()
    }
}

#[cfg(test)]
mod tests;
