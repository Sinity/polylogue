//! The links view: every session linked to the seed, as a tree from the
//! roots of its logical session, paged from a place in its depth-first walk.
//!
//! The walk shows each node once, at the first link that reaches it; a later
//! link to a node already shown is a cross-link, and `reference` links are
//! cross-links that never join the tree. Only link rows are read. A page
//! starts from a [`LinksPlace`]: the nodes being walked, with how many of
//! each one's tree links were taken, and the nodes before the place that
//! more than one link reaches. That is everything the walk needs to go on
//! or to go back, so a page reads the links of its own nodes and of the
//! nodes above them, never of the nodes before it.
use super::super::{Daemon, Failure};
use super::ReadRequest;
use polylogue_model::{SessionId, read::*};
use polylogue_store::Reader;
use std::collections::BTreeMap;
use std::sync::Arc;

/// A tree link to take: to a root from the walk's top, or from a node.
#[derive(Clone)]
struct TreeLink {
    child: String,
    kind: String,
    how: String,
}

/// A `reference` link from a node: the session it names and how.
type ReferenceLink = (String, String);

struct Frame {
    session: Option<String>,
    links: Vec<TreeLink>,
    taken: usize,
}

/// One step of the walk.
enum Step {
    Node {
        session: String,
        /// The node whose link reached it, with that link's kind and how.
        joined: Option<(String, String, String)>,
        references: Vec<ReferenceLink>,
    },
    /// A tree link to a node already shown.
    Cross { from: String, to: String },
}

struct Walk<'r, 'a> {
    reader: &'r Reader<'a>,
    frames: Vec<Frame>,
    /// Nodes more than one link reaches, shown before the place, with the
    /// parent and slot of the link that showed them.
    shown: BTreeMap<String, (Option<String>, usize)>,
    before: u64,
}

/// A node's tree links and its `reference` links, in stored order.
fn links_of(
    reader: &Reader<'_>,
    session: &str,
) -> Result<(Vec<TreeLink>, Vec<ReferenceLink>), Failure> {
    let Ok(id) = SessionId::parse(session) else {
        return Ok((Vec::new(), Vec::new()));
    };
    let mut tree = Vec::new();
    let mut references = Vec::new();
    for link in reader.child_links(&id)? {
        if link.kind == "reference" {
            references.push((link.child, link.how));
        } else {
            tree.push(TreeLink {
                child: link.child,
                kind: link.kind,
                how: link.how,
            });
        }
    }
    Ok((tree, references))
}

/// How many of a node's parents link to it other than by `reference`.
fn tree_parents(reader: &Reader<'_>, session: &str) -> Result<usize, Failure> {
    Ok(match SessionId::parse(session) {
        Ok(id) => reader
            .parent_links(&id)?
            .iter()
            .filter(|link| link.kind != "reference")
            .count(),
        Err(_) => 0,
    })
}

/// The roots of the seed's logical session: the seed and its ancestors that
/// no tree link reaches, or the seed alone when every one has a parent (a
/// cycle).
fn roots(reader: &Reader<'_>, seed: &SessionId) -> Result<Vec<String>, Failure> {
    let mut family: Vec<String> = vec![seed.to_string()];
    family.extend(reader.lineage(seed, polylogue_store::items::Lineage::Ancestors)?);
    let mut roots = Vec::new();
    for member in family {
        if tree_parents(reader, &member)? == 0 {
            roots.push(member);
        }
    }
    if roots.is_empty() {
        roots.push(seed.to_string());
    }
    Ok(roots)
}

impl<'r, 'a> Walk<'r, 'a> {
    fn start(reader: &'r Reader<'a>, roots: Vec<String>) -> Self {
        let links = roots
            .into_iter()
            .map(|child| TreeLink {
                child,
                kind: String::new(),
                how: String::new(),
            })
            .collect();
        Self {
            reader,
            frames: vec![Frame {
                session: None,
                links,
                taken: 0,
            }],
            shown: BTreeMap::new(),
            before: 0,
        }
    }

    /// The walk at `place`, reading the links of the nodes it holds. A place
    /// whose links changed since it was issued is stale.
    fn resume(
        reader: &'r Reader<'a>,
        roots: Vec<String>,
        place: &LinksPlace,
    ) -> Result<Self, Failure> {
        let stale = || {
            Failure::new(
                "stale_cursor",
                "the links changed since the cursor was issued",
            )
        };
        let mut walk = Self::start(reader, roots);
        let mut frames = place.frames.iter();
        let top = frames
            .next()
            .filter(|top| top.session.is_none())
            .ok_or_else(stale)?;
        walk.frames[0].taken = top.taken as usize;
        for frame in frames {
            let session = frame.session.clone().ok_or_else(stale)?;
            let above = walk.frames.last().expect("the walk's top");
            let reached = above
                .taken
                .checked_sub(1)
                .and_then(|slot| above.links.get(slot))
                .is_some_and(|link| link.child == session);
            if !reached {
                return Err(stale());
            }
            let (links, _) = links_of(reader, &session)?;
            walk.frames.push(Frame {
                session: Some(session),
                links,
                taken: frame.taken as usize,
            });
        }
        if walk
            .frames
            .iter()
            .any(|frame| frame.taken > frame.links.len())
        {
            return Err(stale());
        }
        walk.shown = place
            .shown
            .iter()
            .map(|shown| {
                (
                    shown.session.clone(),
                    (shown.parent.clone(), shown.slot as usize),
                )
            })
            .collect();
        walk.before = place.before;
        Ok(walk)
    }

    fn place(&self) -> LinksPlace {
        LinksPlace {
            before: self.before,
            frames: self
                .frames
                .iter()
                .map(|frame| LinksFrame {
                    session: frame.session.clone(),
                    taken: frame.taken as u32,
                })
                .collect(),
            shown: self
                .shown
                .iter()
                .map(|(session, (parent, slot))| LinksShown {
                    session: session.clone(),
                    parent: parent.clone(),
                    slot: *slot as u32,
                })
                .collect(),
        }
    }

    /// The next step, or `None` at the end of the walk.
    fn forward(&mut self) -> Result<Option<Step>, Failure> {
        loop {
            let Some(top) = self.frames.last_mut() else {
                return Ok(None);
            };
            if top.taken == top.links.len() {
                if self.frames.len() == 1 {
                    return Ok(None);
                }
                self.frames.pop();
                continue;
            }
            let slot = top.taken;
            let link = top.links[slot].clone();
            top.taken += 1;
            let parent = top.session.clone();
            if self.shown.contains_key(&link.child) {
                return Ok(Some(Step::Cross {
                    from: parent.unwrap_or_default(),
                    to: link.child,
                }));
            }
            let (links, references) = links_of(self.reader, &link.child)?;
            // A root counts as reached once by the walk itself.
            let reaching = tree_parents(self.reader, &link.child)? + usize::from(parent.is_none());
            if reaching > 1 {
                self.shown
                    .insert(link.child.clone(), (parent.clone(), slot));
            }
            self.frames.push(Frame {
                session: Some(link.child.clone()),
                links,
                taken: 0,
            });
            self.before += 1;
            return Ok(Some(Step::Node {
                session: link.child,
                joined: parent.map(|parent| (parent, link.kind, link.how)),
                references,
            }));
        }
    }

    /// The step before the place, or `None` at the start of the walk.
    fn backward(&mut self) -> Result<Option<Step>, Failure> {
        loop {
            let depth = self.frames.len();
            let top = self.frames.last_mut().expect("the walk's top");
            if top.taken == 0 {
                if depth == 1 {
                    return Ok(None);
                }
                let frame = self.frames.pop().expect("a node's frame");
                let session = frame.session.expect("a node's session");
                let above = self.frames.last_mut().expect("the walk's top");
                above.taken -= 1;
                let link = above.links[above.taken].clone();
                let parent = above.session.clone();
                self.shown.remove(&session);
                self.before -= 1;
                let (_, references) = links_of(self.reader, &session)?;
                return Ok(Some(Step::Node {
                    session,
                    joined: parent.map(|parent| (parent, link.kind, link.how)),
                    references,
                }));
            }
            let slot = top.taken - 1;
            let link = top.links[slot].clone();
            let parent = top.session.clone();
            // A node reached more than once is shown at the link recorded
            // with it; any other link to it is a cross-link.
            let showed = self
                .shown
                .get(&link.child)
                .is_none_or(|(by, at)| *by == parent && *at == slot);
            if !showed {
                top.taken = slot;
                return Ok(Some(Step::Cross {
                    from: parent.unwrap_or_default(),
                    to: link.child,
                }));
            }
            // Its subtree ends just before here: go back from its end.
            let (links, _) = links_of(self.reader, &link.child)?;
            let taken = links.len();
            self.frames.push(Frame {
                session: Some(link.child),
                links,
                taken,
            });
        }
    }
}

/// The items of one step.
fn items(reader: &Reader<'_>, seed: &SessionId, step: &Step) -> Result<Vec<ReadItem>, Failure> {
    let edge = |reference: &str, from: &str, to: &str, relation: &str, how, cross| ReadItem {
        source: Source::record(reference),
        kind: ItemKind::Card {
            card: Box::new(Card::LinkEdge {
                from: format!("session:{from}"),
                to: format!("session:{to}"),
                relation: relation.into(),
                how,
                cross_link: cross,
            }),
        },
    };
    let (session, joined, references) = match step {
        Step::Cross { from, to } => {
            let reference = format!("session:{from}");
            return Ok(vec![edge(&reference, from, to, "tree", None, true)]);
        }
        Step::Node {
            session,
            joined,
            references,
        } => (session, joined, references),
    };
    let reference = format!("session:{session}");
    let id = SessionId::parse(session).ok();
    let stored = match &id {
        Some(id) => reader.session(id)?.is_some(),
        None => false,
    };
    fn missing<T>() -> ValueState<T> {
        ValueState::Unavailable {
            reason: "not in the archive".into(),
        }
    }
    let (title, own) = match &id {
        Some(id) if stored => (
            reader
                .display_title(id)?
                .map(ValueState::known)
                .unwrap_or_default(),
            reader
                .own_message_count(id)?
                .map_or(ValueState::Unknown { reason: None }, ValueState::known),
        ),
        _ => (missing(), missing()),
    };
    let mut out = vec![ReadItem {
        source: Source::record(&reference),
        kind: ItemKind::Card {
            card: Box::new(Card::LinkNode {
                reference: reference.clone(),
                title,
                parents: joined
                    .iter()
                    .map(|(parent, _, _)| format!("session:{parent}"))
                    .collect(),
                own_messages: own,
                inherited_messages: ValueState::Unknown { reason: None },
                own_cost: ValueState::Unknown { reason: None },
                tree_cost: ValueState::Unknown { reason: None },
                seed: *session == seed.to_string(),
                open: None,
            }),
        },
    }];
    // The tree link that reached the node, then its `reference` links.
    if let Some((parent, kind, how)) = joined {
        out.push(edge(
            &reference,
            parent,
            session,
            kind,
            Some(how.clone()),
            false,
        ));
    }
    for (to, how) in references {
        out.push(edge(
            &reference,
            session,
            to,
            "reference",
            Some(how.clone()),
            true,
        ));
    }
    Ok(out)
}

/// The encoded bytes and estimated tokens of a step's items.
fn encoded_size(items: &[ReadItem]) -> (usize, u64) {
    items
        .iter()
        .map(|item| {
            serde_json::to_string(item)
                .map_or((0, 0), |text| (text.len(), estimate_tokens(&text) as u64))
        })
        .fold((0, 0), |(bytes, tokens), (b, t)| (bytes + b, tokens + t))
}

pub(super) fn page(
    reader: &Reader<'_>,
    d: &Daemon,
    request: &ReadRequest,
    operation: &Arc<crate::operation::Context>,
    scope: super::session::Scope,
) -> Result<ReadPage, Failure> {
    let ReadRequest {
        target,
        options,
        format,
        budget,
        cursor,
        ..
    } = request;
    let format =
        Format::parse(format).ok_or_else(|| Failure::new("bad_request", "unknown read format"))?;
    let mut limits = PageBudget::default();
    if let Some(tokens) = budget {
        limits.tokens = std::num::NonZeroU64::new(*tokens as u64)
            .ok_or_else(|| Failure::new("bad_request", "read budget must be positive"))?;
    }
    let digest = options_digest(&(options, format));
    let key = super::cursor_seal(reader)?;
    let seed = &scope.session;
    let roots = roots(reader, seed)?;
    let (mut walk, direction) = match cursor.as_deref() {
        None => (Walk::start(reader, roots), ReadDirection::After),
        Some(text) => {
            let opened = PageCursor::open(text, &key).map_err(super::cursor_failure)?;
            opened
                .check(target, &digest)
                .map_err(super::cursor_failure)?;
            let PagePosition::Links { place } = &opened.position else {
                return Err(Failure::new(
                    "invalid_query",
                    "the cursor names no place in a links walk",
                ));
            };
            if opened.direction == ReadDirection::Past {
                return Err(Failure::new(
                    "invalid_query",
                    "a links reading continues forward or backward only",
                ));
            }
            (Walk::resume(reader, roots, place)?, opened.direction)
        }
    };
    let header = super::session::header(reader, seed, target, options)?;
    let mut used = serde_json::to_vec(&header).map_or(0, |bytes| bytes.len());
    let mut tokens = 0;
    // The steps of the page in walk order, and the places it starts and ends.
    let mut steps: Vec<Vec<ReadItem>> = Vec::new();
    let mut nodes = 0;
    let mut ended = false;
    let anchor = walk.place();
    let mut reached = anchor.clone();
    loop {
        let step = match direction {
            ReadDirection::Before => walk.backward()?,
            _ => walk.forward()?,
        };
        let Some(step) = step else {
            ended = true;
            break;
        };
        let node = matches!(step, Step::Node { .. });
        let step_items = items(reader, seed, &step)?;
        let (size, cost) = encoded_size(&step_items);
        if node
            && nodes > 0
            && (used + size > limits.bytes.get()
                || tokens + cost > limits.tokens.get()
                || nodes >= limits.items.get())
        {
            break;
        }
        used += size;
        tokens += cost;
        nodes += usize::from(node);
        steps.push(step_items);
        reached = walk.place();
    }
    let (start, end, done) = match direction {
        ReadDirection::Before => {
            steps.reverse();
            (reached, anchor, false)
        }
        _ => (anchor, reached, ended),
    };
    let items: Vec<ReadItem> = steps.into_iter().flatten().collect();
    let call = |place: &LinksPlace, direction: ReadDirection| -> Result<ReadyCall, Failure> {
        let next = PageCursor {
            target: target.clone(),
            options: digest.clone(),
            variants: request.options.variants.clone(),
            versions: PageCursor::version(&[]),
            position: PagePosition::Links {
                place: place.clone(),
            },
            direction,
        };
        super::next_call(d, request, next.seal(&key), super::deadline(operation))
    };
    let footer = ReadFooter {
        estimator: Some(TOKEN_ESTIMATOR.into()),
        next: (!done)
            .then(|| call(&end, ReadDirection::After))
            .transpose()?,
        previous: (start.before > 0)
            .then(|| call(&start, ReadDirection::Before))
            .transpose()?,
        position: Some(start.before),
        shown: items.len() as u64,
        // The total is known once the walk has reached its end.
        count: if done {
            ValueState::known(end.before)
        } else {
            ValueState::Unknown {
                reason: Some("counted when the walk reaches its end".into()),
            }
        },
        ..ReadFooter::default()
    };
    let snapshot = ReadPageSnapshot {
        query: target.clone(),
        ran_at: operation.ran_at,
        caught_up: super::caught_up(reader)?,
        read_at: reader.change_position()?,
        currency: d.currency(reader)?,
    };
    Ok(ReadPage {
        format,
        snapshot,
        records: vec![],
        document: ReadDocument {
            header,
            items,
            footer,
        },
    })
}

#[cfg(test)]
mod tests;
