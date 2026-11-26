//! Revision chains whose index lags their stored bytes: what the loop
//! parses again from the archive, for watched and imported inputs alike.
//! One query joins the archive's chains with their `parse_state` rows, and
//! the model's [`due`] judges each; nothing here decides by itself whether
//! a chain is caught up.
use super::*;
use polylogue_model::chain::{ChainFacts, ChainState, ChainStatusName, Due, due, status};
use polylogue_model::freshness::CompletenessReason;

/// One revision chain the index has not parsed to its stored length or
/// under its parser's version.
#[derive(Debug, Clone, PartialEq)]
pub struct BehindChain {
    pub input: String,
    /// The chain holds the input's latest revision.
    pub current: bool,
    /// The selected parser format and recorded path, independent of local discovery.
    pub source: StoredSource,
    pub chain: i64,
    /// The chain's bytes came in through `polylogue import`.
    pub imported: bool,
    /// The chain's stored length.
    pub stored: u64,
    /// Its state; a kept resume state is read as present only
    /// ([`Reader::chain_state`] reads it whole).
    pub state: ChainState,
    /// What the chain needs ([`due`]); never [`Due::No`].
    pub due: Due,
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

/// Which chains of each input participate in planning, before reading parser state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RevisionChains {
    All,
    /// The chain containing the greatest revision number, even if an older chain resumes.
    Newest,
    Older,
}

/// Every chain of the archive by origin and by how it is shown, and the
/// chains due with their stored bytes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChainCensus {
    pub chains: BTreeMap<(String, ChainStatusName), u64>,
    /// Why the chains shown as `parsing` are due, by origin: never parsed,
    /// parsed under another parser version, or stored further.
    pub behind: BTreeMap<String, std::collections::BTreeSet<CompletenessReason>>,
}

impl ChainCensus {
    /// Chains by how they are shown, over every origin.
    pub fn by_status(&self) -> BTreeMap<String, u64> {
        let mut counts = BTreeMap::new();
        for ((_, name), n) in &self.chains {
            *counts.entry(name.as_str().to_owned()).or_default() += n;
        }
        counts
    }
}

/// The facts of a chain of `stored` bytes whose origin's parser version is
/// `parser` (`None`: it has no parser).
fn facts(stored: u64, parser: Option<&str>, imported: bool) -> ChainFacts {
    ChainFacts {
        stored,
        repeated_by: None,
        own_from: 0,
        parser: parser.map(str::to_owned),
        finished: imported,
    }
}

fn parser<'a>(versions: &'a [(String, String)], origin: &str) -> Option<&'a str> {
    versions
        .iter()
        .find(|(name, _)| name == origin)
        .map(|(_, version)| version.as_str())
}

fn chain_rows() -> String {
    use crate::source::{IMPORT_FORMAT, IMPORT_FORMAT_ORDER, STATES_IMPORT_FORMAT};
    let state = crate::parse_state::PLANNED_COLUMNS;
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
                {state}
            FROM chains c JOIN source_input i ON i.id = c.input
            JOIN inputs s ON s.input = c.input
            JOIN revisions l ON l.input = c.input AND l.number = s.number
            LEFT JOIN formats f ON f.input = c.input AND f.choice = 1
            LEFT JOIN first_paths fp ON fp.input = c.input
            LEFT JOIN observations o ON o.id = fp.observation
            LEFT JOIN input_ref ir ON ir.id = c.input
            LEFT JOIN parse_state p ON p.input_pk = ir.pk AND p.chain = c.chain
            WHERE c.length > 0
            "#
    )
}

fn planning_sql(selected: bool, chains: RevisionChains) -> String {
    format!(
        "{} ORDER BY 5,(SELECT max(o.seen_at) FROM observations o WHERE o.input=c.input) DESC,c.input,c.chain DESC",
        behind_sql(selected, chains)
    )
}

/// The chains of the selected inputs with their parse states, in no order.
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
    let chain_lengths = match chains {
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
    let chain_rows = chain_rows();
    format!(
        "WITH chosen AS MATERIALIZED (SELECT DISTINCT value AS input FROM json_each(?1)),
         revisions AS NOT MATERIALIZED ({revisions}),
         observations AS NOT MATERIALIZED ({observations}),
         inputs AS MATERIALIZED (
             SELECT input,max(number) AS number,max(produced='imported') AS imported
             FROM revisions GROUP BY input
         ), chains AS MATERIALIZED ({chain_lengths}), {chain_rows}
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
                 CASE WHEN p.parser_version=json_extract(e.value,'$[1]')
                      THEN min(c.length,p.parsed_to) ELSE 0 END),0)
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

    /// How many chains of the inputs up to `through` with a parser in
    /// `versions` are due, and their stored bytes.
    pub fn count_chains_behind_index(
        &self,
        versions: &[(String, String)],
        through: &str,
    ) -> Result<(u64, u64)> {
        let (mut chains, mut bytes) = (0, 0);
        self.visit_chain_rows(None, RevisionChains::All, &mut |row| {
            if row.input.as_str() <= through
                && parser(versions, &row.source.format).is_some()
                && row.due(versions) != Due::No
            {
                chains += 1;
                bytes += row.stored;
            }
            Ok(std::ops::ControlFlow::Continue(()))
        })?;
        Ok((chains, bytes))
    }

    /// Every chain of the selected inputs that is due. `selected=None`
    /// visits all inputs; an empty selection visits none.
    pub fn chains_behind_index(
        &self,
        versions: &[(String, String)],
        chains: RevisionChains,
        selected: Option<&[String]>,
    ) -> Result<Vec<BehindChain>> {
        let mut found = Vec::new();
        self.visit_chain_rows(selected, chains, &mut |row| {
            let due = row.due(versions);
            if due != Due::No {
                found.push(BehindChain {
                    input: row.input,
                    current: row.current,
                    source: row.source,
                    chain: row.chain,
                    imported: row.imported,
                    stored: row.stored,
                    state: row.state,
                    due,
                });
            }
            Ok(std::ops::ControlFlow::Continue(()))
        })?;
        Ok(found)
    }

    /// Every chain of the archive counted by origin and status, judged
    /// against the origins' parser versions in `contracts`.
    pub fn chain_census(&self, contracts: &[ReadingContract]) -> Result<ChainCensus> {
        let token = self.scope_token();
        let database = crate::source::database_original("c.input");
        let format = crate::source::parse_format("c.input", "i.origin");
        let state = crate::parse_state::PLANNED_COLUMNS;
        let sql = format!(
            "SELECT c.input, c.length,
                CASE WHEN {database} THEN 'export-file' ELSE {format} END, {state}
             FROM (SELECT input, chain, max(length) AS length FROM source_revision
                   GROUP BY input, chain) c
             JOIN source_input i ON i.id = c.input
             LEFT JOIN input_ref ir ON ir.id = c.input
             LEFT JOIN parse_state p ON p.input_pk = ir.pk AND p.chain = c.chain"
        );
        let conn = self.conn()?;
        let mut statement = conn.prepare_cached(&sql)?;
        let mut rows = statement.query([])?;
        let mut census = ChainCensus::default();
        while let Some(row) = rows.next()? {
            token.check()?;
            let stored: u64 = row.get(1)?;
            let origin: String = row.get(2)?;
            let state = crate::parse_state::ParseStateRow::planned(row, 3)?;
            let parser = contracts
                .iter()
                .find(|c| c.origin() == origin)
                .and_then(|contract| match contract {
                    ReadingContract::Expected { version, .. } => Some(version.as_str()),
                    _ => None,
                });
            let facts = facts(stored, parser, false);
            let shown = status(&state, &facts);
            if shown.name() == ChainStatusName::Parsing {
                let why = match state.version() {
                    None => CompletenessReason::MissingCheckpoint,
                    Some(version) if parser != Some(version) => {
                        CompletenessReason::ParseVersionChanged
                    }
                    Some(_) => CompletenessReason::ParserBehind,
                };
                census.behind.entry(origin.clone()).or_default().insert(why);
            }
            *census.chains.entry((origin, shown.name())).or_default() += 1;
        }
        token.check()?;
        Ok(census)
    }

    /// Hands each chain of the selected inputs, with its parse state, to
    /// `visit`, in planning order.
    fn visit_chain_rows(
        &self,
        selected: Option<&[String]>,
        chains: RevisionChains,
        visit: &mut dyn FnMut(ChainRow) -> Result<std::ops::ControlFlow<()>>,
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
        let mut rows = statement.query(params![selected_json])?;
        while let Some(row) = rows.next()? {
            token.check()?;
            let path_bytes: Option<Vec<u8>> = row.get(6)?;
            let chain = ChainRow {
                input: row.get(0)?,
                chain: row.get(1)?,
                stored: row.get(2)?,
                current: row.get(3)?,
                imported: row.get(4)?,
                source: StoredSource {
                    format: row.get(5)?,
                    path: path_bytes
                        .as_deref()
                        .map(|bytes| String::from_utf8_lossy(bytes).into_owned()),
                    path_bytes,
                },
                state: crate::parse_state::ParseStateRow::planned(row, 7)?,
            };
            if visit(chain)?.is_break() {
                break;
            }
        }
        token.check()
    }
}

/// One chain as the planning query reads it.
struct ChainRow {
    input: String,
    chain: i64,
    stored: u64,
    current: bool,
    imported: bool,
    source: StoredSource,
    state: ChainState,
}

impl ChainRow {
    fn due(&self, versions: &[(String, String)]) -> Due {
        due(
            &self.state,
            &facts(
                self.stored,
                parser(versions, &self.source.format),
                self.imported,
            ),
        )
    }
}

#[cfg(test)]
mod tests;
