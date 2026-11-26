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
    pub stopped_attempt: Option<polylogue_model::batch::StoppedAttempt>,
    pub stopped_at_length: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredSource {
    pub format: String,
    /// The path the parser reads the bytes as, never a file to open.
    pub path: Option<String>,
    /// The path's raw bytes from the same observation, when recorded.
    /// This is a parser hint, never permission to reopen a source file.
    pub path_bytes: Option<Vec<u8>>,
}

/// Whether chains whose refusal or failure is recorded at their stored length
/// count as behind. A recorded attempt stays finished until its program or
/// parser settings change; an older record without that identity stays unrecorded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoppedChains {
    Finished,
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

fn chains_behind() -> String {
    use crate::source::{IMPORT_FORMAT, IMPORT_FORMAT_ORDER, STATES_IMPORT_FORMAT};
    format!(
        r#"first_paths AS MATERIALIZED (
                SELECT input, min(id) AS observation
                FROM observations GROUP BY input
            ), formats AS (
                SELECT o.input, o.path_bytes, {IMPORT_FORMAT} AS format,
                    row_number() OVER (PARTITION BY o.input ORDER BY {IMPORT_FORMAT_ORDER}) AS choice
                FROM observations o WHERE {STATES_IMPORT_FORMAT}
            )
            SELECT c.input, c.chain, c.length, l.chain = c.chain,
                s.imported,
                COALESCE(f.format, i.origin) AS format,
                COALESCE(f.path_bytes, o.path_bytes),
                COALESCE(p.parsed_upto, 0), p.input_pk IS NOT NULL, COALESCE(p.status = 'waiting' AND p.at_length = c.length, 0), p.parse_version, p.status, p.at_length, p.reason, p.stopped_attempt,
                COALESCE(f.path_bytes, o.path_bytes)
            FROM chains c JOIN source_input i ON i.id = c.input
            JOIN inputs s ON s.input = c.input
            JOIN revisions l ON l.input = c.input AND l.number = s.number
            LEFT JOIN formats f ON f.input = c.input AND f.choice = 1
            LEFT JOIN first_paths fp ON fp.input = c.input
            LEFT JOIN observations o ON o.id = fp.observation
            LEFT JOIN input_ref ir ON ir.id = c.input
            LEFT JOIN parse_state p ON p.input_pk = ir.pk AND p.chain = c.chain
            WHERE (c.length > 0
              AND (p.input_pk IS NULL
                OR (NOT COALESCE(p.at_length = c.length AND (p.status = 'waiting' OR NOT ?2), 0)
                    AND p.parsed_upto < c.length)
                OR (?2 AND p.status IN ('refused','failed') AND p.at_length=c.length)
                OR EXISTS(SELECT 1 FROM json_each(?1) expected
                    WHERE json_extract(expected.value, '$[0]') = COALESCE(f.format, i.origin)
                      AND json_extract(expected.value, '$[1]') IS NOT p.parse_version)))
            "#
    )
}

fn planning_sql(selected: bool, chains: RevisionChains) -> String {
    format!(
        "{} ORDER BY 5,(SELECT max(o.seen_at) FROM observations o WHERE o.input=c.input) DESC,c.input,c.chain DESC",
        behind_sql(selected, chains)
    )
}

/// The chains behind the index, in no order.
fn behind_sql(selected: bool, chains: RevisionChains) -> String {
    // Restrict histories before aggregating them. All-input planning
    // keeps its existing source-view access.
    let (revisions, observations) = if selected {
        (
            "SELECT r.* FROM chosen CROSS JOIN source_revision r ON r.input=chosen.input",
            "SELECT o.* FROM chosen CROSS JOIN source_observation o ON o.input=chosen.input",
        )
    } else {
        (
            "SELECT * FROM source_revision",
            "SELECT * FROM source_observation",
        )
    };
    // Keep the selected chain lengths once: flattening the current-chain MAX
    // would evaluate it again for each progress predicate and result column.
    let chain_rows = match chains {
        RevisionChains::All => "SELECT input,chain,max(length) AS length FROM revisions GROUP BY input,chain",
        // Kept histories may resume an earlier chain with a shorter revision.
        // Its planned length remains the maximum, not the latest revision length.
        RevisionChains::Newest => "SELECT s.input,l.chain,
            (SELECT max(r.length) FROM revisions r WHERE r.input=s.input AND r.chain=l.chain) AS length
            FROM inputs s JOIN revisions l ON l.input=s.input AND l.number=s.number",
        RevisionChains::Older => "SELECT r.input,r.chain,max(r.length) AS length
            FROM inputs s JOIN revisions l ON l.input=s.input AND l.number=s.number
            CROSS JOIN revisions r ON r.input=s.input WHERE r.chain<>l.chain
            GROUP BY r.input,r.chain",
    };
    let database = crate::source::database_original("c.input");
    let chains_behind = chains_behind();
    format!(
        "WITH chosen AS MATERIALIZED (SELECT DISTINCT value AS input FROM json_each(?3)),
         revisions AS NOT MATERIALIZED ({revisions}),
         observations AS NOT MATERIALIZED ({observations}),
         inputs AS MATERIALIZED (
             SELECT input,max(number) AS number,max(produced='imported') AS imported
             FROM revisions GROUP BY input
         ), chains AS MATERIALIZED ({chain_rows}), {chains_behind}
         AND NOT {database}"
    )
}

impl Reader<'_> {
    /// The upper input key of a bounded catch-up pass, from this read snapshot.
    pub fn last_input_id(&self) -> Result<Option<String>> {
        let token = self.scope_token();
        let last = self
            .conn()?
            .scalar_optional("SELECT id FROM source_input ORDER BY id DESC LIMIT 1", [])?;
        token.check()?;
        Ok(last)
    }

    /// One page of stored input identities, independent of their parsing progress.
    pub fn input_ids_after(&self, after: Option<&str>, limit: usize) -> Result<Vec<String>> {
        let token = self.scope_token();
        let conn = self.conn()?;
        let condition = if after.is_some() { " WHERE id>?1" } else { "" };
        let mut query = conn.prepare_cached(&format!(
            "SELECT id FROM source_input{condition} ORDER BY id LIMIT ?2",
        ))?;
        let ids = query
            .query_map(params![after, limit as u64], |row| row.get(0))?
            .collect::<std::result::Result<Vec<String>, _>>()?;
        token.check()?;
        Ok(ids)
    }

    /// Bytes the newest readable chains hold and their committed parser positions.
    pub fn newest_parse_bytes(&self, versions: &[(String, String)]) -> Result<(u64, u64)> {
        let token = self.scope_token();
        let database = crate::source::database_original("c.input");
        let format = crate::source::parse_format("i.id", "i.origin");
        let progress = self.conn()?.query_row(
            &format!(
                "WITH latest AS (
                 SELECT input,max(number) AS number FROM source_revision GROUP BY input
             ), newest AS (
                 SELECT r.input,r.chain,
                     (SELECT max(s.length) FROM source_revision s
                      WHERE s.input=r.input AND s.chain=r.chain) AS length
                 FROM latest l JOIN source_revision r ON r.input=l.input AND r.number=l.number
             ), formats AS (
                 SELECT i.id,{format} AS format FROM source_input i
             )
             SELECT coalesce(sum(c.length),0),coalesce(sum(
                 CASE WHEN p.parse_version=json_extract(e.value,'$[1]')
                      THEN min(c.length,p.parsed_upto) ELSE 0 END),0)
             FROM newest c JOIN formats f ON f.id=c.input
             JOIN json_each(?1) e ON json_extract(e.value,'$[0]')=f.format
             LEFT JOIN input_ref i ON i.id=c.input
             LEFT JOIN parse_state p ON p.input_pk=i.pk AND p.chain=c.chain
             WHERE NOT {database}"
            ),
            [serde_json::to_string(versions)?],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        token.check()?;
        Ok(progress)
    }

    /// How many chains of the inputs up to `through` with a reading in
    /// `versions` [`Self::chains_behind_index`] yields over all chains, and their stored
    /// bytes, counted in one query. It counts chains whose refusal or failure
    /// is recorded at their length for this attempt, which planning then
    /// leaves out.
    pub fn count_chains_behind_index(
        &self,
        versions: &[(String, String)],
        through: &str,
    ) -> Result<(u64, u64)> {
        let token = self.scope_token();
        let counted = self.conn()?.query_row(
            &format!(
                "SELECT count(*),coalesce(sum(length),0) FROM ({}) behind
                 WHERE input<=?4 AND EXISTS(SELECT 1 FROM json_each(?1) expected
                     WHERE json_extract(expected.value,'$[0]')=behind.format)",
                behind_sql(false, RevisionChains::All)
            ),
            params![
                serde_json::to_string(versions)?,
                true,
                None::<String>,
                through
            ],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        token.check()?;
        Ok(counted)
    }

    /// Every chain whose index is behind its stored bytes. Waiting at the
    /// recorded source length is current for that parse version; a refusal
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
        let selected_json = selected.map(serde_json::to_string).transpose()?;
        #[cfg(feature = "faults")]
        crate::fault::check(
            "read.chains_behind_index",
            &format!(
                "{chains:?}:{}",
                selected_json.as_deref().unwrap_or("all inputs")
            ),
        )?;
        let connection = self.conn()?;
        let sql = planning_sql(selected.is_some(), chains);
        let mut statement = connection.prepare(&sql)?;
        let mut rows = statement.query(params![
            serde_json::to_string(versions)?,
            matches!(stopped, StoppedChains::ForAttempt(_)),
            selected_json,
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
            let length = row.get::<_, u64>(2)?;
            let parsed_upto = row.get::<_, u64>(7)?;
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
            let chain = BehindChain {
                input,
                current: row.get(3)?,
                source: StoredSource {
                    format: origin,
                    path: row
                        .get::<_, Option<Vec<u8>>>(6)?
                        .map(|bytes| String::from_utf8_lossy(&bytes).into_owned()),
                    path_bytes: row.get(15)?,
                },
                reading_changed,
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
