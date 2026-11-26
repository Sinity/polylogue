//! One indexed ancestry walk for derived products, current reads and queries.
use crate::row::{Json, Rows as _, SessionPk, Statement, row};
use crate::{Result, query::fields::STRUCTURAL_KINDS};
use polylogue_model::{
    SessionId,
    read::{LineageIssue, SessionLineage},
};
use rusqlite::Connection;

/// Marks the lineage summary due for session `id` and every session below it
/// through lineage links: what a link written, removed or retyped, or a
/// session added, removed or turned to or from a stub, can change. The code
/// that makes the change calls it in the change's transaction.
pub(crate) fn mark_due(conn: &Connection, id: &str) -> Result<()> {
    conn.prepare_cached(
        "INSERT INTO summary_due(session_pk,from_order,lineage)
         WITH RECURSIVE affected(id) AS (
           SELECT ?1
           UNION SELECT l.child_id FROM session_link l JOIN affected a ON l.parent_id=a.id
             WHERE l.kind IN ('spawn','fork','continuation','resume')
         ) SELECT s.pk,X'FF',1 FROM affected a JOIN session s ON s.id=a.id WHERE true
         ON CONFLICT(session_pk) DO UPDATE SET lineage=1,marked=marked+1",
    )?
    .execute([id])?;
    Ok(())
}

/// Whether a link of `kind` places sessions in a lineage.
pub(crate) fn is_lineage(kind: &str) -> bool {
    matches!(kind, "spawn" | "fork" | "continuation" | "resume")
}

/// The visible incoming relations, including their stored presentation order.
/// Only the affected child's indexed edge set is read while writing.
pub(crate) fn relation_identity(conn: &Connection, child: &str) -> Result<String> {
    conn.value(
        "SELECT json_group_array(json_array(pk,parent_id,child_id,kind,reason,branch_point,spawning_call,copies_prefix))
         FROM (SELECT * FROM session_link WHERE child_id=?1 ORDER BY pk)",
        [child],
    )?
    .ok_or_else(|| crate::StoreError::internal("an aggregate read no row"))
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
    pub reason: Option<String>,
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

row! {
    /// A parent and child a recorded prefix copy names.
    struct CopiedPair {
        parent_id: String,
        child_id: String,
    }
    statements {
        const OF_PARENT = "SELECT DISTINCT parent_id, child_id FROM shared_prefix WHERE parent_id = ?1";
        const OF_CHILD = "SELECT DISTINCT parent_id, child_id FROM shared_prefix WHERE child_id = ?1";
    }
}

row! {
    /// One relation between two sessions, as stored.
    struct LinkRow {
        parent_id: String,
        child_id: String,
        kind: String,
        reason: Option<String>,
        branch_point: Option<String>,
        spawning_call: Option<String>,
        copies_prefix: bool,
    }
    statements {
        const OF_PARENT = "SELECT parent_id, child_id, kind, reason, branch_point, spawning_call, copies_prefix
            FROM session_link INDEXED BY session_link_parent WHERE parent_id = ?1 ORDER BY pk";
        const OF_CHILD = "SELECT parent_id, child_id, kind, reason, branch_point, spawning_call, copies_prefix
            FROM session_link INDEXED BY session_link_child WHERE child_id = ?1 ORDER BY pk";
    }
}

/// Which end of its links a session is.
#[derive(Clone, Copy)]
enum End {
    Parent,
    Child,
}

/// The links whose `end` (the parent or the child) is `session`, one row
/// each; a recorded prefix copy of a pair also says its prefix is copied.
fn links_where(conn: &Connection, end: End, session: &str) -> Result<Vec<SessionLink>> {
    #[cfg(any(test, feature = "fixture"))]
    LINK_READS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let (copies, stored) = match end {
        End::Parent => (&CopiedPair::OF_PARENT, &LinkRow::OF_PARENT),
        End::Child => (&CopiedPair::OF_CHILD, &LinkRow::OF_CHILD),
    };
    let copied: std::collections::HashSet<(String, String)> = conn
        .all(copies, [session])?
        .into_iter()
        .map(|pair| (pair.parent_id, pair.child_id))
        .collect();
    Ok(conn
        .all(stored, [session])?
        .into_iter()
        .map(|link| SessionLink {
            copies_prefix: link.copies_prefix
                || copied.contains(&(link.parent_id.clone(), link.child_id.clone())),
            parent: link.parent_id,
            child: link.child_id,
            kind: link.kind,
            reason: link.reason,
            branch_point: link.branch_point,
            spawning_call: link.spawning_call,
        })
        .collect())
}

/// One shown statement of a link, as assembly resolves it.
pub(crate) struct Statement {
    pub branch_point: Option<String>,
    pub spawning_call: Option<String>,
    pub copies_prefix: bool,
    pub evidence: String,
    pub source_chain: i64,
    pub input_pk: i64,
}

/// One relation the statements of a (parent, child, kind, how) give.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Relation {
    pub branch_point: Option<String>,
    pub spawning_call: Option<String>,
    pub copies_prefix: bool,
    pub evidence: String,
    pub source_chain: i64,
    pub input_pk: i64,
}

/// The relations of one (parent, child, kind, how) from all its
/// statements, in `pk` order: coordinates that agree are one relation; a
/// coordinate recorded with several values gives one relation for each;
/// when both differ, every statement stands except one that another
/// statement completes. Each relation takes its input from the first
/// statement it agrees with, its evidence from the least of them all, and
/// copies the prefix when any statement says so.
pub(crate) fn relations(statements: &[Statement]) -> Vec<Relation> {
    use std::collections::{BTreeSet, HashSet};
    let Some(evidence) = statements.iter().map(|s| s.evidence.as_str()).min() else {
        return Vec::new();
    };
    let copies_prefix = statements.iter().any(|s| s.copies_prefix);
    let known = |of: fn(&Statement) -> &Option<String>| {
        statements
            .iter()
            .filter_map(|statement| of(statement).clone())
            .collect::<BTreeSet<String>>()
    };
    let branch_points = known(|statement| &statement.branch_point);
    let calls = known(|statement| &statement.spawning_call);
    let coordinates: Vec<(Option<String>, Option<String>)> =
        match (branch_points.len() > 1, calls.len() > 1) {
            (false, false) => vec![(branch_points.first().cloned(), calls.first().cloned())],
            (true, false) => branch_points
                .iter()
                .map(|point| (Some(point.clone()), calls.first().cloned()))
                .collect(),
            (false, true) => calls
                .iter()
                .map(|call| (branch_points.first().cloned(), Some(call.clone())))
                .collect(),
            (true, true) => {
                let with_call: HashSet<&str> = statements
                    .iter()
                    .filter(|s| s.spawning_call.is_some())
                    .filter_map(|s| s.branch_point.as_deref())
                    .collect();
                let with_point: HashSet<&str> = statements
                    .iter()
                    .filter(|s| s.branch_point.is_some())
                    .filter_map(|s| s.spawning_call.as_deref())
                    .collect();
                let kept: BTreeSet<(Option<String>, Option<String>)> = statements
                    .iter()
                    .filter(|s| match (&s.branch_point, &s.spawning_call) {
                        (Some(_), Some(_)) => true,
                        (Some(point), None) => !with_call.contains(point.as_str()),
                        (None, Some(call)) => !with_point.contains(call.as_str()),
                        (None, None) => false,
                    })
                    .map(|s| (s.branch_point.clone(), s.spawning_call.clone()))
                    .collect();
                kept.into_iter().collect()
            }
        };
    coordinates
        .into_iter()
        .map(|(branch_point, spawning_call)| {
            let agrees = |value: &Option<String>, coordinate: &Option<String>| {
                value.is_none() || value == coordinate
            };
            let first = statements
                .iter()
                .find(|s| {
                    agrees(&s.branch_point, &branch_point)
                        && agrees(&s.spawning_call, &spawning_call)
                })
                .unwrap_or(&statements[0]);
            Relation {
                branch_point,
                spawning_call,
                copies_prefix,
                evidence: evidence.to_owned(),
                source_chain: first.source_chain,
                input_pk: first.input_pk,
            }
        })
        .collect()
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
        links_where(conn, End::Child, child)?
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
    let copied: Vec<String> = conn.values(
        "SELECT DISTINCT parent_id FROM shared_prefix WHERE child_id = ?1",
        [child],
    )?;
    for parent in copied {
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
        SELECT min(l.parent_id) FROM session_link l
        WHERE l.child_id=a.id AND l.kind IN ({STRUCTURAL_KINDS})
        HAVING count(DISTINCT l.parent_id)=1)
      WHERE a.cycle=0 AND p.has_transcript=1
    ), terminal AS (SELECT * FROM ancestry ORDER BY depth DESC LIMIT 1),
    observed AS (SELECT t.*, (SELECT json_group_array(parent_id) FROM (
      SELECT DISTINCT l.parent_id FROM session_link l
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
      SELECT 1 FROM summary_state ds WHERE ds.session_pk=v.session_pk AND ds.name='lineage' AND pl_summary_current(ds.name,ds.version))
      AND NOT EXISTS(SELECT 1 FROM summary_due dd WHERE dd.session_pk=v.session_pk AND dd.lineage=1)");
    format!(
        "(CASE WHEN EXISTS(SELECT 1 FROM session_lineage v WHERE {valid})
      THEN (SELECT v.{column} FROM session_lineage v WHERE v.session_pk={session})
      ELSE (SELECT {column} FROM ({})) END)",
        observed_lineage_sql(session)
    )
}

row! {
    /// Where a session's ancestry walk stopped: the root and depth when it
    /// reached one, else the problem met and the parents recorded there.
    struct ObservedLineage {
        root: Option<String>,
        depth: Option<u64>,
        root_pending: bool,
        problem: Option<String>,
        stopped_at: String,
        parents: Json<Vec<String>>,
    }
}

row! {
    /// A session's lineage as the lineage derivation stored it.
    struct StoredLineage {
        root: Option<SessionId>,
        depth: Option<u64>,
        root_pending: bool,
        issue: Option<Json<LineageIssue>>,
    }
    statements {
        const CURRENT = "SELECT v.root, v.depth, v.root_pending, v.issue FROM session_lineage v
            JOIN summary_state ds ON ds.session_pk=v.session_pk AND ds.name='lineage' AND pl_summary_current(ds.name,ds.version)
            WHERE v.session_pk=?1 AND NOT EXISTS(SELECT 1 FROM summary_due dd WHERE dd.session_pk=v.session_pk AND dd.lineage=1)";
    }
}

pub(crate) fn compute_lineage(
    conn: &Connection,
    session: SessionPk,
) -> Result<Option<SessionLineage>> {
    let Some(ObservedLineage {
        root,
        depth,
        root_pending,
        problem,
        stopped_at,
        parents: Json(parents),
    }) = conn.maybe(&Statement::built(observed_lineage_sql("?1")), [session])?
    else {
        return Ok(None);
    };
    let parse = |id: &str| SessionId::parse(id).map_err(crate::StoreError::stored_data);
    let child = parse(&stopped_at)?;
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
            .value(
                "SELECT coalesce((SELECT value FROM meta WHERE key='relationship_revision'),'0')",
                [],
            )?
            .unwrap_or_default();
        Ok(polylogue_model::BlobHash::of(
            format!("relationships:{revision}").as_bytes(),
        ))
    }

    /// Links from `id` to its children, ordered by their coordinates.
    pub fn child_links(&self, id: &SessionId) -> Result<Vec<SessionLink>> {
        links_where(self.conn()?, End::Parent, &id.to_string())
    }

    /// Links from `id`'s parents to it, ordered by their coordinates.
    pub fn parent_links(&self, id: &SessionId) -> Result<Vec<SessionLink>> {
        links_where(self.conn()?, End::Child, &id.to_string())
    }

    /// Current structural root and depth in this pinned read. Missing sessions
    /// return None; unresolved ancestry is an explicit typed issue.
    pub fn lineage_root(&self, session: &SessionId) -> Result<Option<SessionLineage>> {
        let Some(pk) = self.session_pk(session)? else {
            return Ok(None);
        };
        match self.conn()?.maybe(&StoredLineage::CURRENT, [pk])? {
            Some(stored) => Ok(Some(SessionLineage {
                root: stored.root,
                depth: stored.depth,
                root_pending: stored.root_pending,
                issue: stored.issue.map(|Json(issue)| issue),
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

/// The rows this module declares statements for, for the every-statement test.
#[cfg(test)]
pub(crate) fn declared_rows(check: &mut crate::row::Check<'_>) {
    check.row::<CopiedPair>();
    check.row::<LinkStatement>();
    check.row::<StoredLineage>();
}

#[cfg(test)]
mod tests;
