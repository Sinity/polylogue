//! Logical membership is a directed source fact, never a property inferred
//! from a session name, timestamp, command, or a preferred physical ID.
use crate::{Link, SessionId};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

pub const VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ConfirmedLink {
    pub event: String,
    pub facts: Link,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct LogicalWork {
    pub version: u32,
    pub root: Option<SessionId>,
    pub physical: Vec<SessionId>,
    pub links: Vec<ConfirmedLink>,
    pub unavailable: Vec<String>,
}

impl LogicalWork {
    pub fn identity(&self) -> String {
        blake3::hash(&serde_json::to_vec(self).expect("membership contains typed source facts"))
            .to_hex()
            .to_string()
    }
}

/// All four Link kinds are explicit membership links. Reference relationships
/// have a different domain type and cannot enter this reducer.
pub fn derive(
    sessions: impl IntoIterator<Item = SessionId>,
    links: impl IntoIterator<Item = ConfirmedLink>,
) -> Vec<LogicalWork> {
    let mut adjacent = BTreeMap::<SessionId, BTreeSet<SessionId>>::new();
    let mut parents = BTreeMap::<SessionId, BTreeSet<SessionId>>::new();
    let mut children = BTreeMap::<SessionId, BTreeSet<SessionId>>::new();
    let mut evidence = BTreeMap::<SessionId, BTreeSet<usize>>::new();
    for session in sessions {
        adjacent.entry(session).or_default();
    }
    let mut links: Vec<_> = links.into_iter().collect();
    links.sort_by_cached_key(|link| {
        (
            link.event.clone(),
            serde_json::to_vec(&link.facts).expect("typed link"),
        )
    });
    links.dedup();
    let conflicting_events: BTreeSet<_> = links
        .windows(2)
        .filter(|pair| pair[0].event == pair[1].event)
        .map(|pair| pair[0].event.as_str())
        .collect();
    for (index, link) in links.iter().enumerate() {
        let Link { parent, child, .. } = &link.facts;
        adjacent
            .entry(parent.clone())
            .or_default()
            .insert(child.clone());
        adjacent
            .entry(child.clone())
            .or_default()
            .insert(parent.clone());
        parents
            .entry(child.clone())
            .or_default()
            .insert(parent.clone());
        children
            .entry(parent.clone())
            .or_default()
            .insert(child.clone());
        evidence.entry(parent.clone()).or_default().insert(index);
        evidence.entry(child.clone()).or_default().insert(index);
    }
    let mut visited = BTreeSet::new();
    let mut out = Vec::new();
    for start in adjacent.keys() {
        if !visited.insert(start.clone()) {
            continue;
        }
        let mut queue = VecDeque::from([start.clone()]);
        let mut physical = BTreeSet::new();
        let mut contributing = BTreeSet::new();
        while let Some(session) = queue.pop_front() {
            physical.insert(session.clone());
            contributing.extend(evidence.get(&session).into_iter().flatten().copied());
            for next in &adjacent[&session] {
                if visited.insert(next.clone()) {
                    queue.push_back(next.clone());
                }
            }
        }
        let mut unavailable = Vec::new();
        let mut degrees = BTreeMap::new();
        for session in &physical {
            let degree = parents.get(session).map_or(0, BTreeSet::len);
            if degree > 1 {
                unavailable.push("logical_membership_multiple_parents".into());
            }
            degrees.insert(session.clone(), degree);
        }
        let roots: Vec<_> = degrees
            .iter()
            .filter(|(_, degree)| **degree == 0)
            .map(|(id, _)| id.clone())
            .collect();
        let mut ready: VecDeque<_> = roots.iter().cloned().collect();
        let mut processed = 0;
        while let Some(session) = ready.pop_front() {
            processed += 1;
            for child in children.get(&session).into_iter().flatten() {
                let degree = degrees
                    .get_mut(child)
                    .expect("component contains every child");
                *degree -= 1;
                if *degree == 0 {
                    ready.push_back(child.clone());
                }
            }
        }
        if processed != physical.len() {
            unavailable.push("logical_membership_cycle".into());
        }
        if roots.len() != 1 {
            unavailable.push("logical_membership_root_unavailable".into());
        }
        let mut component_links = Vec::new();
        let mut events = BTreeSet::new();
        for index in contributing {
            let link = links[index].clone();
            if link.event.is_empty() {
                unavailable.push("logical_membership_evidence_unavailable".into());
            }
            if conflicting_events.contains(link.event.as_str()) {
                unavailable.push("logical_membership_event_conflict".into());
            }
            if !events.insert(link.event.clone()) {
                unavailable.push("logical_membership_event_conflict".into());
            }
            component_links.push(link);
        }
        unavailable.sort();
        unavailable.dedup();
        out.push(LogicalWork {
            version: VERSION,
            root: unavailable.is_empty().then(|| roots[0].clone()),
            physical: physical.into_iter().collect(),
            links: component_links,
            unavailable,
        });
    }
    out
}

#[cfg(test)]
mod tests;
