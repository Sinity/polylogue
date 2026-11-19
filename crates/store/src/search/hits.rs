//! Choose the best source from explicitly supplied page owners.
//! Eligible blocks are materialized once and each distinct text is checked once.
use super::*;
use crate::query::fields::STRUCTURAL_KINDS;
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
            let (table, _, _) = parts(terms.index);
            let wanted = serde_json::to_string(&pending).expect("snippet scope serializes");
            let scope = scope_ctes(terms.index);
            let sql = format!(
                "WITH RECURSIVE {scope}, \
                texts AS MATERIALIZED (SELECT DISTINCT text_id FROM eligible), \
                matches AS MATERIALIZED (SELECT t.text_id,bm25({table}) AS rank \
                  FROM texts t CROSS JOIN {table} ON {table}.rowid=t.text_id WHERE {table} MATCH ?1), \
                candidates AS (SELECT e.*,h.rank FROM matches h JOIN eligible e USING(text_id)), {}",
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

fn scope_ctes(index: SnippetIndex) -> String {
    let (_, kinds, writer) = parts(index);
    format!(
        "{WANTED}, \
      tree(ordinal,pk) AS (SELECT ordinal,pk FROM wanted WHERE is_session=1 \
        UNION SELECT t.ordinal,l.child_pk FROM tree t CROSS JOIN session p ON p.pk=t.pk \
        CROSS JOIN session_link l ON l.parent_id=p.id AND l.kind IN ({STRUCTURAL_KINDS}) WHERE l.child_pk IS NOT NULL), \
      eligible AS MATERIALIZED (\
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

#[cfg(test)]
mod tests;
