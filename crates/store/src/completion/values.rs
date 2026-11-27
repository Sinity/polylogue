//! What completion offers from the archive, read on demand by the daemon:
//! the values of completable fields with their counts, sessions by ID
//! prefix or title, kept results, and saved query and setting names.
use crate::{Reader, Result};
use rusqlite::params;

/// A value of a field and how many sessions have it; for `model` and
/// `tool`, how many messages name the model or calls use the tool; for
/// `tag`, how many current tags carry it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValueCount {
    pub value: String,
    pub count: u64,
}

/// A session as completion describes it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Session {
    pub id: String,
    pub native_id: String,
    /// The shortest prefix of `id` no other session's ID starts with, never
    /// shorter than the origin and its colon.
    pub prefix: String,
    pub title: String,
    pub origin: String,
    pub repo: Option<String>,
    pub last_activity: Option<i64>,
}

/// A kept result as completion names it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeptName {
    pub id: String,
    pub name: Option<String>,
    pub unit: Option<String>,
    pub row_count: Option<u64>,
}

/// The session columns each field's values come from.
const SESSION_FIELDS: [&str; 5] = ["origin", "repo", "cwd", "branch", "harness"];

impl Reader<'_> {
    /// The values of `field` that start with `prefix`, most common first.
    /// A working directory also counts for each of its parent directories.
    /// A field with no stored values has none.
    pub fn completion_values(&self, field: &str, prefix: &str) -> Result<Vec<ValueCount>> {
        let conn = self.conn()?;
        let upper = prefix_end(prefix);
        let mut counts = std::collections::BTreeMap::<String, u64>::new();
        if let Some(column) = SESSION_FIELDS.iter().find(|column| **column == field) {
            // A directory's parents start with the prefix only when the
            // directory does, so one range of the column holds them all.
            let mut statement = conn.prepare(&format!(
                "SELECT {column}, count(*) FROM session WHERE {column} >= ?1 AND (?2 IS NULL OR {column} < ?2)
                 AND {column} <> '' GROUP BY {column}"
            ))?;
            let mut rows = statement.query(params![prefix, upper])?;
            while let Some(row) = rows.next()? {
                self.check()?;
                let value: String = row.get(0)?;
                let n = row.get::<_, i64>(1)? as u64;
                if *column == "cwd" {
                    for (at, _) in value.match_indices('/').skip(1) {
                        *counts.entry(value[..at].to_owned()).or_default() += n;
                    }
                }
                *counts.entry(value).or_default() += n;
            }
        } else if field == "tag" {
            let current = crate::annotations::sql::current_annotation("n");
            let mut statement = conn.prepare(&format!(
                "SELECT value, count(*) FROM (
                   SELECT n.key AS value FROM source_annotation n WHERE n.kind='tag' AND {current}
                   UNION ALL
                   SELECT n.key || ':' || json_extract(n.value, '$.value') FROM source_annotation n
                   WHERE n.kind='tag' AND json_extract(n.value, '$.value') IS NOT NULL AND {current})
                 WHERE value >= ?1 AND (?2 IS NULL OR value < ?2) GROUP BY value"
            ))?;
            let mut rows = statement.query(params![prefix, upper])?;
            while let Some(row) = rows.next()? {
                self.check()?;
                counts.insert(row.get(0)?, row.get::<_, i64>(1)? as u64);
            }
        } else {
            let mut statement = conn.prepare(
                "SELECT value, sum(count) FROM session_value WHERE field = ?1 AND count > 0
                 AND value >= ?2 AND (?3 IS NULL OR value < ?3) GROUP BY value",
            )?;
            let mut rows = statement.query(params![field, prefix, upper])?;
            while let Some(row) = rows.next()? {
                self.check()?;
                counts.insert(row.get(0)?, row.get::<_, i64>(1)? as u64);
            }
        }
        let mut values: Vec<ValueCount> = counts
            .into_iter()
            .filter(|(value, _)| value.starts_with(prefix))
            .map(|(value, count)| ValueCount { value, count })
            .collect();
        values.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.value.cmp(&b.value)));
        Ok(values)
    }

    /// Sessions whose full or native ID starts with `prefix`, most recent
    /// first. When none does, the sessions whose title holds the prefix's
    /// characters in order, ignoring case and accents.
    /// `origins` are the origin names a native ID can follow.
    pub fn completion_sessions(&self, prefix: &str, origins: &[String]) -> Result<Vec<Session>> {
        let conn = self.conn()?;
        let title = crate::session_sql::display_title("s");
        let columns = format!(
            "s.id, s.native_id, coalesce({title}, ''), s.origin, s.repo, s.last_active_at,
             (SELECT max(p.id) FROM session p WHERE p.id < s.id),
             (SELECT min(n.id) FROM session n WHERE n.id > s.id)"
        );
        let mut statement = conn.prepare(&format!(
            "SELECT {columns} FROM session s WHERE s.id >= ?1 AND (?2 IS NULL OR s.id < ?2)"
        ))?;
        let mut sessions = Vec::new();
        // The native ID follows the origin's colon, so each origin's range
        // of the ID index holds the native IDs with this prefix.
        let mut ranges = vec![prefix.to_owned()];
        if !prefix.contains(':') {
            ranges.extend(origins.iter().map(|origin| format!("{origin}:{prefix}")));
        }
        for start in &ranges {
            let mut rows = statement.query(params![start, prefix_end(start)])?;
            while let Some(row) = rows.next()? {
                self.check()?;
                sessions.push(session(row)?);
            }
        }
        sessions.sort_by(|a, b| a.id.cmp(&b.id));
        sessions.dedup_by(|a, b| a.id == b.id);
        let key = polylogue_model::text::key(prefix);
        // A prefix of only combining marks has an empty key, which would
        // match every title.
        if sessions.is_empty() && !key.is_empty() {
            let mut statement = conn.prepare(&format!("SELECT {columns} FROM session s"))?;
            let mut rows = statement.query([])?;
            while let Some(row) = rows.next()? {
                self.check()?;
                let found = session(row)?;
                if in_order(&key, &polylogue_model::text::key(&found.title)) {
                    sessions.push(found);
                }
            }
        }
        sessions.sort_by(|a, b| {
            (
                a.last_activity.is_none(),
                std::cmp::Reverse(a.last_activity),
                &a.id,
            )
                .cmp(&(
                    b.last_activity.is_none(),
                    std::cmp::Reverse(b.last_activity),
                    &b.id,
                ))
        });
        Ok(sessions)
    }

    /// Kept results whose name or ID starts with `prefix`, by name.
    pub fn completion_kept(&self, prefix: &str) -> Result<Vec<KeptName>> {
        let conn = self.conn()?;
        let mut statement = conn.prepare(
            "SELECT k.id, k.name,
               CASE WHEN json_valid(k.rows_json) AND json_type(k.rows_json,'$.unit')='text' THEN json_extract(k.rows_json,'$.unit') END,
               CASE WHEN json_valid(k.rows_json) AND json_type(k.rows_json,'$.rows')='array' THEN json_array_length(k.rows_json,'$.rows') END
             FROM source_saved_rows k
             WHERE (k.name >= ?1 AND (?2 IS NULL OR k.name < ?2)) OR (k.id >= ?1 AND (?2 IS NULL OR k.id < ?2))
             ORDER BY coalesce(k.name, k.id), k.id",
        )?;
        let mut rows = statement.query(params![prefix, prefix_end(prefix)])?;
        let mut kept = Vec::new();
        while let Some(row) = rows.next()? {
            self.check()?;
            kept.push(KeptName {
                id: row.get(0)?,
                name: row.get(1)?,
                unit: row.get(2)?,
                row_count: row.get::<_, Option<i64>>(3)?.map(|count| count as u64),
            });
        }
        Ok(kept)
    }

    /// The names of current saved queries that start with `prefix`, in order.
    pub fn completion_saved_names(&self, prefix: &str) -> Result<Vec<String>> {
        let current = crate::annotations::sql::current_annotation("n");
        let names = self
            .conn()?
            .prepare(&format!(
                "SELECT DISTINCT n.key FROM source_annotation n WHERE n.kind='saved_query' AND {current}
                 AND n.key >= ?1 AND (?2 IS NULL OR n.key < ?2) ORDER BY n.key"
            ))?
            .query_map(params![prefix, prefix_end(prefix)], |row| row.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        self.check()?;
        Ok(names)
    }

    /// The setting keys any annotation has set, in order.
    pub fn completion_setting_keys(&self) -> Result<Vec<String>> {
        let keys = self
            .conn()?
            .prepare("SELECT DISTINCT key FROM source_annotation WHERE kind='setting' AND key IS NOT NULL ORDER BY key")?
            .query_map([], |row| row.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        self.check()?;
        Ok(keys)
    }
}

fn session(row: &rusqlite::Row<'_>) -> rusqlite::Result<Session> {
    let id: String = row.get(0)?;
    let before: Option<String> = row.get(6)?;
    let after: Option<String> = row.get(7)?;
    let longest = [before, after]
        .iter()
        .flatten()
        .map(|neighbor| common_prefix(&id, neighbor))
        .max()
        .unwrap_or(0);
    let at = longest.max(id.find(':').map_or(0, |at| at + 1));
    let end = id
        .char_indices()
        .find(|(index, _)| *index >= at)
        .map_or(id.len(), |(index, c)| index + c.len_utf8());
    Ok(Session {
        prefix: id[..end].to_owned(),
        id,
        native_id: row.get(1)?,
        title: row.get(2)?,
        origin: row.get(3)?,
        repo: row.get(4)?,
        last_activity: row.get(5)?,
    })
}

fn common_prefix(a: &str, b: &str) -> usize {
    a.chars()
        .zip(b.chars())
        .take_while(|(a, b)| a == b)
        .map(|(c, _)| c.len_utf8())
        .sum()
}

/// The smallest string above every string that starts with `prefix`; `None`
/// means no upper bound (the empty prefix, or a prefix of only the greatest
/// character).
fn prefix_end(prefix: &str) -> Option<String> {
    let mut value = prefix.to_owned();
    while let Some(c) = value.pop() {
        let mut next = u32::from(c) + 1;
        if (0xd800..=0xdfff).contains(&next) {
            next = 0xe000;
        }
        if let Some(c) = char::from_u32(next) {
            value.push(c);
            return Some(value);
        }
    }
    None
}

fn in_order(needle: &str, haystack: &str) -> bool {
    let mut rest = haystack.chars();
    needle.chars().all(|c| rest.any(|h| h == c))
}

#[cfg(test)]
mod tests;
