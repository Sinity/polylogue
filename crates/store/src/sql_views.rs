//! The views `polylogue sql` offers outside programs. The tables are
//! internal and change with `INDEX_VERSION`; these views keep their names
//! and columns. A change that would alter a view's columns adds the next
//! version beside it (`v2_sessions`), so a program written against one
//! version keeps working. They are temporary views of each read connection,
//! because a view stored in `index.db` cannot read the attached
//! `archive.db`, where notes live.

use rusqlite::Connection;

/// Each view: its name and its query. Times are Unix milliseconds, costs
/// dollars; an unknown value is NULL, never 0.
pub const VIEWS: &[(&str, &str)] = &[
    (
        "v1_sessions",
        "SELECT s.id, s.origin, s.kind, coalesce(s.title, s.derived_title) AS title, \
         s.cwd, s.repo, s.branch, s.harness, s.started_at, s.last_activity_at \
         FROM main.session s WHERE s.stub = 0",
    ),
    (
        "v1_messages",
        "SELECT m.id, s.id AS session, m.role, m.written_by, m.model, m.at, \
         (SELECT group_concat(pl_text(t.body), char(10)) FROM \
            (SELECT b.text_id FROM main.block b WHERE b.message_pk = m.pk AND b.kind = 'text' ORDER BY b.idx) bt \
            JOIN main.text t ON t.id = bt.text_id) AS text, \
         m.input_tokens, m.output_tokens, m.cache_read_tokens, m.cache_write_tokens, \
         m.cost_nusd / 1e9 AS cost_usd, m.duplicate_of, m.on_active_path \
         FROM main.message m JOIN main.session s ON s.pk = m.session_pk",
    ),
    (
        "v1_tool_calls",
        "SELECT s.id AS session, c.call_id, c.at, c.tool, c.action, c.command, c.path, \
         c.outcome, c.exit_code \
         FROM main.tool_call c JOIN main.session s ON s.pk = c.session_pk",
    ),
    (
        "v1_links",
        "SELECT l.parent_id AS parent, l.child_id AS child, l.kind, l.how \
         FROM main.link l",
    ),
    (
        "v1_assertions",
        "SELECT a.id,a.root,a.kind,a.key,a.value,a.author,a.via,a.machine,a.created_at,a.valid_until, \
         (SELECT group_concat(ref,' ') FROM (SELECT ref FROM source_link WHERE record=a.id AND role='about' ORDER BY ref)) AS about, \
         rs.current,rs.withdrawn,(SELECT count(*) FROM source_revision_state c WHERE c.root=a.root AND c.current=1)>1 AS conflict \
         FROM source_assertion a JOIN source_revision_state rs ON rs.id=a.id",
    ),
];

/// Creates the views on a read connection with `archive.db` attached as `a`.
pub fn create(conn: &Connection) -> rusqlite::Result<()> {
    for (name, query) in VIEWS {
        conn.execute_batch(&format!("CREATE TEMP VIEW {name} AS {query}"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
