//! One indexed ancestry walk for derived products, current reads and queries.
use crate::{Result, query::fields::STRUCTURAL_KINDS};
use polylogue_model::{
    SessionId,
    read::{LineageIssue, SessionLineage},
};
use rusqlite::{Connection, OptionalExtension};

/// The visible incoming relations, including their stored presentation order.
/// Only the affected child's indexed edge set is read while writing.
pub(crate) fn relation_identity(conn: &Connection, child: &str) -> Result<String> {
    Ok(conn.prepare_cached(
        "SELECT json_group_array(json_array(pk,parent_id,child_id,kind,how,branch_point,spawning_call,copies_prefix))
         FROM (SELECT * FROM session_link_identity WHERE child_id=?1 ORDER BY pk)",
    )?.query_row([child], |row| row.get(0))?)
}

/// How many sessions' links reads have looked up, for tests of how much of
/// a links tree a page reads.
#[cfg(any(test, feature = "fixture"))]
static LINK_READS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[cfg(any(test, feature = "fixture"))]
pub fn link_reads() -> u64 {
    LINK_READS.load(std::sync::atomic::Ordering::Relaxed)
}

/// One stored link between two sessions, as the reading needs it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionLink {
    pub parent: String,
    pub child: String,
    /// A documented provider relationship, or `reference`.
    pub kind: String,
    pub how: Option<String>,
    pub copies_prefix: bool,
    pub branch_point: Option<String>,
    pub spawning_call: Option<String>,
}

impl SessionLink {
    /// A reference link never joins ancestry.
    pub fn is_reference(&self) -> bool {
        self.kind == polylogue_model::LinkKind::Reference.as_str()
    }
}

/// The links whose `column` (the parent or the child) is `session`, as
/// the `session_link_identity` view states them. The view decides each
/// statement with subqueries over all the statements of its parent and
/// child, which for one session with thousands of statements is their
/// square; this reads the statements once and decides each pair in one
/// pass, with the view as its oracle.
fn links_where(conn: &Connection, column: &str, session: &str) -> Result<Vec<SessionLink>> {
    #[cfg(any(test, feature = "fixture"))]
    LINK_READS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut copied: std::collections::HashSet<(String, String)> = Default::default();
    for pair in conn
        .prepare_cached(&format!(
            "SELECT DISTINCT parent_id, child_id FROM prefix_copy WHERE {column} = ?1"
        ))?
        .query_map([session], |row| <(String, String)>::try_from(row))?
    {
        copied.insert(pair?);
    }
    // The statements of each relation: a parent, a child, a kind and
    // how, in that order.
    type Relation = (String, String, String, Option<String>);
    type Statement = (Option<String>, Option<String>, bool);
    let mut relations: std::collections::BTreeMap<Relation, Vec<Statement>> = Default::default();
    for row in conn
        .prepare_cached(&format!(
            "SELECT parent_id, child_id, kind, how, branch_point, spawning_call, copies_prefix
             FROM link WHERE {column} = ?1"
        ))?
        .query_map([session], |row| {
            Ok((
                (row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?),
                (row.get(4)?, row.get(5)?, row.get(6)?),
            ))
        })?
    {
        let (relation, statement): (Relation, Statement) = row?;
        relations.entry(relation).or_default().push(statement);
    }
    let mut links = Vec::new();
    for ((parent, child, kind, how), statements) in relations {
        let copies_prefix = statements.iter().any(|(_, _, copies)| *copies)
            || copied.contains(&(parent.clone(), child.clone()));
        let known = |of: fn(&Statement) -> &Option<String>| {
            statements
                .iter()
                .filter_map(|statement| of(statement).clone())
                .collect::<std::collections::BTreeSet<String>>()
        };
        let branch_points = known(|statement| &statement.0);
        let calls = known(|statement| &statement.1);
        // Coordinates that agree are one link; a coordinate recorded
        // with several values gives one link for each.
        let coordinates: Vec<(Option<String>, Option<String>)> =
            match (branch_points.len() > 1, calls.len() > 1) {
                (false, false) => {
                    vec![(branch_points.first().cloned(), calls.first().cloned())]
                }
                (true, false) => branch_points
                    .iter()
                    .map(|point| (Some(point.clone()), calls.first().cloned()))
                    .collect(),
                (false, true) => calls
                    .iter()
                    .map(|call| (branch_points.first().cloned(), Some(call.clone())))
                    .collect(),
                // Both differ: every statement stands, except one that
                // another statement completes.
                (true, true) => {
                    let with_call: std::collections::HashSet<&str> = statements
                        .iter()
                        .filter(|(_, call, _)| call.is_some())
                        .filter_map(|(point, _, _)| point.as_deref())
                        .collect();
                    let with_point: std::collections::HashSet<&str> = statements
                        .iter()
                        .filter(|(point, _, _)| point.is_some())
                        .filter_map(|(_, call, _)| call.as_deref())
                        .collect();
                    let mut kept: Vec<_> = statements
                        .iter()
                        .filter(|(point, call, _)| match (point, call) {
                            (Some(_), Some(_)) => true,
                            (Some(point), None) => !with_call.contains(point.as_str()),
                            (None, Some(call)) => !with_point.contains(call.as_str()),
                            (None, None) => false,
                        })
                        .map(|(point, call, _)| (point.clone(), call.clone()))
                        .collect();
                    kept.sort();
                    kept
                }
            };
        links.extend(
            coordinates
                .into_iter()
                .map(|(branch_point, spawning_call)| SessionLink {
                    parent: parent.clone(),
                    child: child.clone(),
                    kind: kind.clone(),
                    how: how.clone(),
                    copies_prefix,
                    branch_point,
                    spawning_call,
                }),
        );
    }
    Ok(links)
}

/// The prefixes a session inherits, as the `session_prefix` view states
/// them, in its order: each parent with a branch point and whether the
/// child's file copies the prefix. A fork, continuation or resume link
/// gives one, and so does any link or recorded copy that says the prefix is
/// copied. Read from the session's links in one pass ([`links_where`]).
pub(crate) fn inherited_prefixes(
    conn: &Connection,
    child: &str,
) -> Result<Vec<(String, Option<String>, bool)>> {
    let mut prefixes: std::collections::BTreeSet<(String, Option<String>, bool)> =
        links_where(conn, "child_id", child)?
            .into_iter()
            .filter(|link| {
                !link.is_reference()
                    && (matches!(link.kind.as_str(), "fork" | "continuation" | "resume")
                        || link.copies_prefix)
            })
            .map(|link| (link.parent, link.branch_point, link.copies_prefix))
            .collect();
    let linked: std::collections::HashSet<String> =
        prefixes.iter().map(|prefix| prefix.0.clone()).collect();
    for parent in conn
        .prepare_cached("SELECT DISTINCT parent_id FROM prefix_copy WHERE child_id = ?1")?
        .query_map([child], |row| row.get::<_, String>(0))?
    {
        let parent = parent?;
        if !linked.contains(&parent) {
            prefixes.insert((parent, None, true));
        }
    }
    Ok(prefixes.into_iter().collect())
}

/// The walk advances only when all recorded structural edges name one parent.
/// Different branch points on that parent remain recorded alternatives, but do not
/// change structural depth. No path is selected between different parents.
fn walk(session: &str) -> String {
    format!(
        "WITH RECURSIVE ancestry(pk,id,depth,path,cycle) AS (
      SELECT pk,id,0,printf('/%d/',pk),0 FROM session WHERE pk={session}
      UNION ALL
      SELECT p.pk,p.id,a.depth+1,a.path||printf('%d/',p.pk),
        instr(a.path,printf('/%d/',p.pk))>0
      FROM ancestry a JOIN session p ON p.id=(
        SELECT min(l.parent_id) FROM link l
        WHERE l.child_id=a.id AND l.kind IN ({STRUCTURAL_KINDS})
        HAVING count(DISTINCT l.parent_id)=1)
      WHERE a.cycle=0 AND p.stub=0
    ), terminal AS (SELECT * FROM ancestry ORDER BY depth DESC LIMIT 1),
    observed AS (SELECT t.*, (SELECT json_group_array(parent_id) FROM (
      SELECT DISTINCT l.parent_id FROM link l
      WHERE l.child_id=t.id AND l.kind IN ({STRUCTURAL_KINDS})
      ORDER BY l.parent_id)) AS parents FROM terminal t)"
    )
}

fn observed_lineage_sql(session: &str) -> String {
    format!(
        "{} SELECT CASE WHEN cycle=0 AND parents='[]' THEN id END AS root,
      CASE WHEN cycle=0 AND parents='[]' THEN depth END AS depth,
      cycle=0 AND json_array_length(parents)=1 AS root_pending,
      CASE WHEN cycle<>0 THEN 'cycle' WHEN json_array_length(parents)>1 THEN 'conflict'
        WHEN parents<>'[]' THEN 'missing' END AS problem,
      id AS stopped_at, parents FROM observed",
        walk(session)
    )
}

/// Cached products are used only while their structural causes are unchanged.
/// A query during ingestion uses the very same walk, without exposing stale roots.
pub(crate) fn lineage_column_sql(session: &str, column: &str) -> String {
    let valid = format!("v.session_pk={session} AND EXISTS(
      SELECT 1 FROM derive_state ds WHERE ds.session_pk=v.session_pk AND ds.name='lineage' AND pl_derived_current(ds.name,ds.version))
      AND NOT EXISTS(SELECT 1 FROM derive_due dd WHERE dd.session_pk=v.session_pk AND dd.lineage=1)");
    format!(
        "(CASE WHEN EXISTS(SELECT 1 FROM session_lineage v WHERE {valid})
      THEN (SELECT v.{column} FROM session_lineage v WHERE v.session_pk={session})
      ELSE (SELECT {column} FROM ({})) END)",
        observed_lineage_sql(session)
    )
}

pub(crate) fn compute_lineage(conn: &Connection, session: i64) -> Result<Option<SessionLineage>> {
    let sql = observed_lineage_sql("?1");
    let row = conn
        .prepare_cached(&sql)?
        .query_row([session], |r| {
            <(
                Option<String>,
                Option<u64>,
                bool,
                Option<String>,
                String,
                String,
            )>::try_from(r)
        })
        .optional()?;
    let Some((root, depth, root_pending, problem, at, parents)) = row else {
        return Ok(None);
    };
    let parse = |id: &str| SessionId::parse(id).map_err(crate::StoreError::stored_data);
    let child = parse(&at)?;
    let parents: Vec<String> = serde_json::from_str(&parents)?;
    let issue = match problem.as_deref() {
        Some("cycle") => Some(LineageIssue::Cycle { session: child }),
        Some("conflict") => Some(LineageIssue::ConflictingParents {
            child,
            parents: parents.iter().map(|p| parse(p)).collect::<Result<_>>()?,
        }),
        Some("missing") => Some(LineageIssue::MissingParent {
            child,
            parent: parse(&parents[0])?,
        }),
        _ => None,
    };
    Ok(Some(SessionLineage {
        root: root.as_deref().map(parse).transpose()?,
        depth,
        root_pending,
        issue,
    }))
}

impl crate::read::Reader<'_> {
    /// Links cursors bind the relationship graph, independently of body appends.
    pub fn lineage_version(&self) -> Result<polylogue_model::BlobHash> {
        let revision: String = self
            .conn()?
            .prepare_cached(
                "SELECT coalesce((SELECT value FROM meta WHERE key='relationship_revision'),'0')",
            )?
            .query_row([], |row| row.get(0))?;
        Ok(polylogue_model::BlobHash::of(
            format!("relationships:{revision}").as_bytes(),
        ))
    }

    /// Links from `id` to its children, ordered by their coordinates.
    pub fn child_links(&self, id: &SessionId) -> Result<Vec<SessionLink>> {
        links_where(self.conn()?, "parent_id", &id.to_string())
    }

    /// Links from `id`'s parents to it, ordered by their coordinates.
    pub fn parent_links(&self, id: &SessionId) -> Result<Vec<SessionLink>> {
        links_where(self.conn()?, "child_id", &id.to_string())
    }

    /// Current structural root and depth in this pinned read. Missing sessions
    /// return None; unresolved ancestry is an explicit typed issue.
    pub fn lineage_root(&self, session: &SessionId) -> Result<Option<SessionLineage>> {
        let Some(pk) = self.session_pk(session)? else {
            return Ok(None);
        };
        let cached = self.conn()?.prepare_cached("SELECT v.root,v.depth,v.root_pending,v.issue FROM session_lineage v
          JOIN derive_state ds ON ds.session_pk=v.session_pk AND ds.name='lineage' AND pl_derived_current(ds.name,ds.version)
          WHERE v.session_pk=?1 AND NOT EXISTS(SELECT 1 FROM derive_due dd WHERE dd.session_pk=v.session_pk AND dd.lineage=1)")?
          .query_row([pk], |r| <(Option<String>, Option<u64>, bool, Option<String>)>::try_from(r)).optional()?;
        match cached {
            Some((root, depth, root_pending, issue)) => Ok(Some(SessionLineage {
                root: root
                    .map(|id| SessionId::parse(&id).map_err(crate::StoreError::stored_data))
                    .transpose()?,
                depth,
                root_pending,
                issue: issue.map(|s| serde_json::from_str(&s)).transpose()?,
            })),
            None => compute_lineage(self.conn()?, pk),
        }
    }
    /// Graph entry points for inspecting a session family. A known root is
    /// the only entry of a tree. Otherwise keep every known parent and
    /// a session from each cycle; these are display positions, never asserted roots.
    pub fn lineage_starting_sessions(&self, seed: &SessionId) -> Result<Vec<SessionId>> {
        if let Some(root) = self.lineage_root(seed)?.and_then(|lineage| lineage.root) {
            return Ok(vec![root]);
        }
        let mut states = std::collections::HashMap::new();
        let mut starts = Vec::new();
        let mut entered = std::collections::HashSet::new();
        let mut stack = vec![(seed.clone(), false)];
        while let Some((session, leaving)) = stack.pop() {
            self.check()?;
            if leaving {
                states.insert(session, 2);
                continue;
            }
            match states.get(&session) {
                Some(1) => {
                    if entered.insert(session.clone()) {
                        starts.push(session);
                    }
                    continue;
                }
                Some(_) => continue,
                None => {}
            }
            states.insert(session.clone(), 1);
            let mut unique = std::collections::HashSet::new();
            let parents = self
                .parent_links(&session)?
                .into_iter()
                .filter(|link| !link.is_reference())
                .map(|link| SessionId::parse(&link.parent).map_err(crate::StoreError::stored_data))
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .filter(|id| unique.insert(id.clone()))
                .collect::<Vec<_>>();
            if parents.is_empty() && entered.insert(session.clone()) {
                starts.push(session.clone());
            }
            stack.push((session, true));
            stack.extend(parents.into_iter().rev().map(|parent| (parent, false)));
        }
        Ok(starts)
    }
}

#[cfg(test)]
mod tests;
