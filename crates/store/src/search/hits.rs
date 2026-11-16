//! Choose a page's source blocks together. Small scopes probe the index by
//! text identity; larger scopes start from index hits and walk ancestors once.
use super::*;
use rusqlite::params;

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
            let (table, _, _) = index_parts(terms.index);
            let wanted = serde_json::to_string(&pending).expect("snippet scope serializes");
            let scope = scope_ctes(terms.index);
            let hit_budget = HIT_BUDGET.to_string();
            let scope_budget = SCOPE_BUDGET.to_string();
            let plan = plan_ctes(
                "s",
                terms.index,
                "?1",
                "eligible",
                Budgets {
                    hits: &hit_budget,
                    scope: &scope_budget,
                },
                None,
            )
            .join(", ");
            let indexed = index_sql(terms.index);
            let sql = format!(
                "WITH RECURSIVE {scope}, {plan}, \
                 scope_texts AS MATERIALIZED (SELECT DISTINCT text_id FROM s_scope_sample \
                   WHERE (SELECT plan FROM s_plan)='scope'), \
                 scope_matches AS MATERIALIZED (SELECT t.text_id,bm25({table}) AS rank \
                   FROM scope_texts t CROSS JOIN {table} ON {table}.rowid=t.text_id WHERE {table} MATCH ?1), \
                 {indexed}, \
                 candidates AS (SELECT e.ordinal,e.block,e.session_pk,e.session,e.message,e.at,h.rank \
                   FROM scope_matches h JOIN s_scope_sample e USING(text_id) \
                   UNION ALL SELECT * FROM index_candidates), {}",
                winner()
            );
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
}

const WANTED: &str = "wanted AS MATERIALIZED (SELECT json_extract(value,'$[0]') AS ordinal,json_extract(value,'$[1]') AS pk,json_extract(value,'$[2]') AS is_session FROM json_each(?2))";

fn scope_ctes(index: SnippetIndex) -> String {
    let (_, kinds, writer) = index_parts(index);
    format!(
        "{WANTED}, \
      tree(ordinal,pk) AS (SELECT ordinal,pk FROM wanted WHERE is_session=1 \
        UNION SELECT t.ordinal,l.child_pk FROM tree t CROSS JOIN session p ON p.pk=t.pk \
        CROSS JOIN session_link l ON l.parent_id=p.id AND l.kind='spawn' WHERE l.child_pk IS NOT NULL), \
      eligible AS NOT MATERIALIZED (\
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
    let (table, kinds, writer) = index_parts(index);
    format!(
        "texts AS MATERIALIZED (SELECT rowid AS text_id,bm25({table}) AS rank FROM {table} WHERE (SELECT plan FROM s_plan)<>'scope' AND {table} MATCH ?1), \
      matched AS MATERIALIZED (SELECT b.pk AS block,m.pk AS message_pk,m.session_pk,s.id AS session,m.id AS message,m.at,h.rank,m.duplicate_of,coalesce(m.on_active_path,1) AS active \
        FROM texts h CROSS JOIN block b INDEXED BY block_text ON b.text_id=h.text_id \
        CROSS JOIN message m ON m.pk=b.message_pk CROSS JOIN session s ON s.pk=m.session_pk WHERE b.kind IN ({kinds}){writer}), \
      ancestors(block,session_pk) AS (SELECT block,session_pk FROM matched WHERE duplicate_of IS NULL AND active=1 \
        AND EXISTS(SELECT 1 FROM wanted WHERE is_session=1) \
        UNION SELECT c.block,p.pk FROM ancestors c CROSS JOIN session_link l ON l.child_pk=c.session_pk AND l.kind='spawn' CROSS JOIN session p ON p.id=l.parent_id), \
      index_candidates AS (SELECT w.ordinal,m.block,m.session_pk,m.session,m.message,m.at,m.rank FROM wanted w JOIN matched m ON m.message_pk=w.pk WHERE w.is_session=0 \
        UNION ALL SELECT w.ordinal,m.block,m.session_pk,m.session,m.message,m.at,m.rank FROM wanted w JOIN ancestors a ON a.session_pk=w.pk JOIN matched m ON m.block=a.block WHERE w.is_session=1)"
    )
}

#[cfg(test)]
mod tests;
