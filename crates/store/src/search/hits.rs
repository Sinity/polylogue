//! Choose a page's source blocks together. Small scopes probe the index by
//! text identity; larger scopes start from index hits and walk ancestors once.
use super::*;
use rusqlite::params;

const SMALL_SCOPE: u64 = 256;
const SCOPE_ADVANTAGE: u64 = 64;

impl Reader<'_> {
    pub(super) fn snippet_hits(
        &self,
        queries: &[SnippetQuery<'_>],
        scopes: &[Within],
    ) -> Result<Vec<Option<Hit>>> {
        let mut hits = (0..scopes.len()).map(|_| None).collect::<Vec<_>>();
        for (query, terms) in queries.iter().enumerate() {
            let Some(fts) = terms.fts() else { continue };
            let pending = scopes
                .iter()
                .enumerate()
                .filter(|(i, _)| hits[*i].is_none())
                .map(|(i, scope)| match scope {
                    Within::Session(pk) => (i, *pk, true),
                    Within::Message(pk) => (i, *pk, false),
                })
                .collect::<Vec<_>>();
            if pending.is_empty() {
                break;
            }
            let (table, _, _) = parts(terms.index);
            let wanted = serde_json::to_string(&pending).expect("snippet scope serializes");
            let scope = scope_ctes(terms.index, false);
            let probe = format!("WITH RECURSIVE {scope} SELECT DISTINCT block FROM eligible");
            let driver = self.search_driver(
                &probe,
                &[
                    crate::SqlValue::Text(fts.clone()),
                    crate::SqlValue::Text(wanted.clone()),
                ],
                terms.index,
                &fts,
                false,
            )?;
            let sql = if driver == SearchDriver::Scope {
                let scope = scope_ctes(terms.index, true);
                format!(
                    "WITH RECURSIVE {scope}, \
                    texts AS MATERIALIZED (SELECT DISTINCT text_id FROM eligible), \
                    matches AS MATERIALIZED (SELECT t.text_id,bm25({table}) AS rank \
                      FROM texts t CROSS JOIN {table} ON {table}.rowid=t.text_id WHERE {table} MATCH ?1), \
                    candidates AS (SELECT e.*,h.rank FROM matches h JOIN eligible e USING(text_id)), {}",
                    winner()
                )
            } else {
                index_sql(terms.index)
            };
            let conn = self.conn()?;
            let mut statement = conn.prepare_cached(&sql)?;
            let rows = statement.query_map(params![fts, wanted], |r| {
                Ok((
                    r.get::<_, usize>(0)?,
                    Hit {
                        query,
                        block: r.get(1)?,
                        session_pk: r.get(2)?,
                        session: r.get(3)?,
                        message: r.get(4)?,
                    },
                ))
            })?;
            for row in rows {
                let (ordinal, hit) = row?;
                hits[ordinal] = Some(hit);
            }
        }
        Ok(hits)
    }

    pub(crate) fn search_driver(
        &self,
        scope: &str,
        params: &[crate::SqlValue],
        index: SnippetIndex,
        fts: &str,
        recent: bool,
    ) -> Result<SearchDriver> {
        let conn = self.conn()?;
        let (table, kinds, writer) = parts(index);
        let mut indexed = conn.prepare_cached(&format!(
            "SELECT b.pk FROM {table} CROSS JOIN block b INDEXED BY block_text ON b.text_id={table}.rowid \
             CROSS JOIN message m ON m.pk=b.message_pk WHERE {table} MATCH ?1 AND b.kind IN ({kinds}){writer}"
        ))?;
        let mut index_rows = indexed.query([fts])?;
        let mut occurrences = 0u64;
        // A rare index leg is cheaper to read directly than to establish a
        // scope's size, especially across owners with no eligible blocks.
        while occurrences <= SMALL_SCOPE {
            if index_rows.next()?.is_none() {
                return Ok(SearchDriver::Index);
            }
            occurrences += 1;
        }
        if recent {
            return Ok(SearchDriver::Recency);
        }
        let mut scoped = conn.prepare_cached(scope)?;
        for (i, value) in params.iter().take(scoped.parameter_count()).enumerate() {
            scoped.raw_bind_parameter(i + 1, value)?;
        }
        let mut scope_rows = scoped.raw_query();
        let mut blocks = 0u64;
        while blocks <= SMALL_SCOPE {
            if scope_rows.next()?.is_none() {
                return Ok(SearchDriver::Scope);
            }
            blocks += 1;
        }
        // Probe the two sources together, without retaining their prefixes.
        // A shared indexed text counts once for every block it expands to.
        loop {
            while occurrences <= blocks.saturating_mul(SCOPE_ADVANTAGE) {
                if index_rows.next()?.is_none() {
                    return Ok(SearchDriver::Index);
                }
                occurrences += 1;
            }
            if scope_rows.next()?.is_none() {
                return Ok(SearchDriver::Scope);
            }
            blocks += 1;
        }
    }
}

fn parts(index: SnippetIndex) -> (&'static str, &'static str, &'static str) {
    match index {
        SnippetIndex::Prose => (
            "text_fts",
            "'text'",
            " AND (m.written_by IN ('human','model','unknown') OR m.prompt_kind IS 'compaction_summary')",
        ),
        SnippetIndex::Output => ("output_fts", "'tool_result','tool_call'", ""),
    }
}

const WANTED: &str = "wanted AS MATERIALIZED (SELECT json_extract(value,'$[0]') AS ordinal,json_extract(value,'$[1]') AS pk,json_extract(value,'$[2]') AS is_session FROM json_each(?2))";

fn scope_ctes(index: SnippetIndex, materialize: bool) -> String {
    let (_, kinds, writer) = parts(index);
    let storage = if materialize { "" } else { "NOT " };
    format!(
        "{WANTED}, \
      tree(ordinal,pk) AS (SELECT ordinal,pk FROM wanted WHERE is_session=1 \
        UNION SELECT t.ordinal,l.child_pk FROM tree t CROSS JOIN session p ON p.pk=t.pk \
        CROSS JOIN session_link l ON l.parent_id=p.id AND l.kind='spawn' WHERE l.child_pk IS NOT NULL), \
      eligible AS {storage}MATERIALIZED (\
        SELECT w.ordinal,b.pk AS block,b.text_id,m.session_pk,s.id AS session,m.id AS message,m.at \
        FROM wanted w CROSS JOIN message m ON m.pk=w.pk CROSS JOIN block b ON b.message_pk=m.pk \
        CROSS JOIN session s ON s.pk=m.session_pk WHERE w.is_session=0 AND b.text_id IS NOT NULL AND b.kind IN ({kinds}){writer} \
        UNION ALL \
        SELECT t.ordinal,b.pk,b.text_id,m.session_pk,s.id,m.id,m.at \
        FROM tree t CROSS JOIN message m ON m.session_pk=t.pk CROSS JOIN block b ON b.message_pk=m.pk \
        CROSS JOIN session s ON s.pk=m.session_pk WHERE m.duplicate_of IS NULL AND coalesce(m.on_active_path,1)=1 \
          AND b.text_id IS NOT NULL AND b.kind IN ({kinds}){writer})"
    )
}

fn winner() -> String {
    format!(
        "ranked AS ({}) SELECT ordinal,block,session_pk,session,message FROM ranked",
        super::best_source("candidates", "ordinal")
    )
}

fn index_sql(index: SnippetIndex) -> String {
    let (table, kinds, writer) = parts(index);
    format!(
        "WITH RECURSIVE {WANTED}, \
      texts AS MATERIALIZED (SELECT rowid AS text_id,bm25({table}) AS rank FROM {table} WHERE {table} MATCH ?1), \
      matched AS MATERIALIZED (SELECT b.pk AS block,m.pk AS message_pk,m.session_pk,s.id AS session,m.id AS message,m.at,h.rank,m.duplicate_of,coalesce(m.on_active_path,1) AS active \
        FROM texts h CROSS JOIN block b INDEXED BY block_text ON b.text_id=h.text_id \
        CROSS JOIN message m ON m.pk=b.message_pk CROSS JOIN session s ON s.pk=m.session_pk WHERE b.kind IN ({kinds}){writer}), \
      ancestors(block,session_pk) AS (SELECT block,session_pk FROM matched WHERE duplicate_of IS NULL AND active=1 \
        AND EXISTS(SELECT 1 FROM wanted WHERE is_session=1) \
        UNION SELECT c.block,p.pk FROM ancestors c CROSS JOIN session_link l ON l.child_pk=c.session_pk AND l.kind='spawn' CROSS JOIN session p ON p.id=l.parent_id), \
      candidates AS (SELECT w.ordinal,m.block,m.session_pk,m.session,m.message,m.at,m.rank FROM wanted w JOIN matched m ON m.message_pk=w.pk WHERE w.is_session=0 \
        UNION ALL SELECT w.ordinal,m.block,m.session_pk,m.session,m.message,m.at,m.rank FROM wanted w JOIN ancestors a ON a.session_pk=w.pk JOIN matched m ON m.block=a.block WHERE w.is_session=1), {}",
        winner()
    )
}

#[cfg(test)]
mod tests;
