//! One indexed ancestry walk for derived products, current reads and queries.
use crate::{Result, StoreError};
use polylogue_model::{
    SessionId,
    read::{LineageIssue, SessionLineage},
};
use rusqlite::{Connection, OptionalExtension};

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

/// Which way a lineage walk goes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lineage {
    Ancestors,
    Descendants,
}

/// The walk advances only when all recorded structural edges name one parent.
/// Different anchors to that parent remain recorded alternatives, but do not
/// change structural depth. No path is selected between different parents.
fn walk(session: &str) -> String {
    format!(
        "WITH RECURSIVE ancestry(pk,id,depth,path,cycle) AS (
      SELECT pk,id,0,printf('/%d/',pk),0 FROM session WHERE pk={session}
      UNION ALL
      SELECT p.pk,p.id,a.depth+1,a.path||printf('%d/',p.pk),
        instr(a.path,printf('/%d/',p.pk))>0
      FROM ancestry a JOIN session p ON p.id=(
        SELECT min(l.parent_id) FROM session_link_identity l
        WHERE l.child_id=a.id AND l.kind IN ('spawn','fork','continuation','resume')
        HAVING count(DISTINCT l.parent_id)=1)
      WHERE a.cycle=0 AND p.stub=0
    ), terminal AS (SELECT * FROM ancestry ORDER BY depth DESC LIMIT 1),
    observed AS (SELECT t.*, (SELECT json_group_array(parent_id) FROM (
      SELECT DISTINCT l.parent_id FROM session_link_identity l
      WHERE l.child_id=t.id AND l.kind IN ('spawn','fork','continuation','resume')
      ORDER BY l.parent_id)) AS parents FROM terminal t)"
    )
}

fn projection(session: &str) -> String {
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
pub(crate) fn value_sql(session: &str, column: &str) -> String {
    let valid = format!("v.session_pk={session} AND EXISTS(
      SELECT 1 FROM derive_state ds WHERE ds.session_pk=v.session_pk AND ds.name='lineage' AND pl_derived_current(ds.name,ds.version))
      AND NOT EXISTS(SELECT 1 FROM derive_due dd WHERE dd.session_pk=v.session_pk AND dd.lineage=1)");
    format!(
        "(CASE WHEN EXISTS(SELECT 1 FROM session_lineage v WHERE {valid})
      THEN (SELECT v.{column} FROM session_lineage v WHERE v.session_pk={session})
      ELSE (SELECT {column} FROM ({})) END)",
        projection(session)
    )
}

pub(crate) fn compute(conn: &Connection, session: i64) -> Result<Option<SessionLineage>> {
    let sql = projection("?1");
    let row = conn
        .prepare_cached(&sql)?
        .query_row([session], |r| {
            Ok((
                r.get::<_, Option<String>>(0)?,
                r.get::<_, Option<u64>>(1)?,
                r.get::<_, bool>(2)?,
                r.get::<_, Option<String>>(3)?,
                r.get::<_, String>(4)?,
                r.get::<_, String>(5)?,
            ))
        })
        .optional()?;
    let Some((root, depth, root_pending, problem, at, parents)) = row else {
        return Ok(None);
    };
    let parse = |id: &str| SessionId::parse(id).map_err(|e| StoreError::Invalid(e.to_string()));
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
    fn links_where(&self, column: &str, session: &str) -> Result<Vec<SessionLink>> {
        #[cfg(any(test, feature = "fixture"))]
        LINK_READS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(self
            .conn()?
            .prepare_cached(&format!(
                "SELECT parent_id, child_id, kind, how, branch_point, spawning_call, copies_prefix FROM session_link
                 WHERE {column} = ?1 ORDER BY pk"
            ))?
            .query_map([session], |row| {
                Ok(SessionLink {
                    parent: row.get(0)?,
                    child: row.get(1)?,
                    kind: row.get(2)?,
                    how: row.get(3)?,
                    copies_prefix: row.get(6)?,
                    branch_point: row.get(4)?,
                    spawning_call: row.get(5)?,
                })
            })?
            .collect::<rusqlite::Result<_>>()?)
    }

    /// Links from `id` to its children, in the order they were stored.
    pub fn child_links(&self, id: &SessionId) -> Result<Vec<SessionLink>> {
        self.links_where("parent_id", &id.to_string())
    }

    /// Links from `id`'s parents to it, in the order they were stored.
    pub fn parent_links(&self, id: &SessionId) -> Result<Vec<SessionLink>> {
        self.links_where("child_id", &id.to_string())
    }

    /// Every session reached from `id` through links other than
    /// `reference`, transitively, each once; a cycle ends where it closes.
    pub fn lineage(&self, id: &SessionId, direction: Lineage) -> Result<Vec<String>> {
        let mut seen = std::collections::HashSet::from([id.to_string()]);
        let mut queue = std::collections::VecDeque::from([id.to_string()]);
        let mut out = Vec::new();
        while let Some(current) = queue.pop_front() {
            self.check()?;
            let links = match direction {
                Lineage::Ancestors => self.links_where("child_id", &current)?,
                Lineage::Descendants => self.links_where("parent_id", &current)?,
            };
            for link in links.into_iter().filter(|link| link.kind != "reference") {
                let next = match direction {
                    Lineage::Ancestors => link.parent,
                    Lineage::Descendants => link.child,
                };
                if seen.insert(next.clone()) {
                    out.push(next.clone());
                    queue.push_back(next);
                }
            }
        }
        Ok(out)
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
          .query_row([pk], |r| Ok((r.get::<_,Option<String>>(0)?,r.get::<_,Option<u64>>(1)?,r.get::<_,bool>(2)?,r.get::<_,Option<String>>(3)?))).optional()?;
        match cached {
            Some((root, depth, root_pending, issue)) => Ok(Some(SessionLineage {
                root: root
                    .map(|id| SessionId::parse(&id).map_err(|e| StoreError::Invalid(e.to_string())))
                    .transpose()?,
                depth,
                root_pending,
                issue: issue.map(|s| serde_json::from_str(&s)).transpose()?,
            })),
            None => compute(self.conn()?, pk),
        }
    }
    /// Graph entry points for inspecting a session family. A proven root is
    /// the only entry of a tree. Otherwise retain every parent frontier and
    /// cycle witness; these are display positions, never asserted roots.
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
                .filter(|link| link.kind != "reference")
                .map(|link| {
                    SessionId::parse(&link.parent).map_err(|e| StoreError::Invalid(e.to_string()))
                })
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
