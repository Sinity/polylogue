use crate::test_daemon;
use command_surface::RunnableRequest;
use polylogue_model::read::{
    Card, ItemKind, ReadDelivery, ReadItem, ReadOptions, ReadPage, ValueState, View,
};

const CHILDREN: usize = 40;
const GRANDCHILDREN: usize = 30;

/// A root continued in `CHILDREN` sessions, the sixth of which is stored and
/// continued in `GRANDCHILDREN` more; the others are not in the archive.
fn family(d: &test_daemon::TestDaemon) {
    let dir = d.home.join(".claude/projects/-p");
    let continued = |from: &str, to: &str| {
        format!(
            "{}\n",
            serde_json::json!({"type": "continued-in", "sessionId": from, "continuedInSessionId": to})
        )
    };
    let mut root = polylogue_testkit::claude_user("r1", "root", "root question");
    for n in 0..CHILDREN {
        root.push_str(&continued("root", &format!("c{n}")));
    }
    let mut child = polylogue_testkit::claude_user("c1", "c5", "child question");
    for n in 0..GRANDCHILDREN {
        child.push_str(&continued("c5", &format!("g{n}")));
    }
    std::fs::write(dir.join("c5.jsonl"), child).unwrap();
    std::fs::write(dir.join("root.jsonl"), root).unwrap();
    d.wait_for(|status| status["store"]["messages"] == 3);
}

fn read(d: &test_daemon::TestDaemon, budget: usize, cursor: Option<&RunnableRequest>) -> ReadPage {
    read_target(d, "session:claude-code:root", budget, cursor)
}
fn read_target(
    d: &test_daemon::TestDaemon,
    target: &str,
    budget: usize,
    cursor: Option<&RunnableRequest>,
) -> ReadPage {
    let call = match cursor {
        Some(resume) => {
            let mut resume = resume.clone();
            resume.arguments["page_tokens"] = serde_json::json!(budget);
            test_daemon::Call::from_request(&resume)
        }
        None => test_daemon::read(crate::daemon::service::read::ReadRequest {
            target: target.into(),
            options: ReadOptions {
                view: View::Links,
                ..ReadOptions::default()
            },
            delivery: ReadDelivery::Page,
            format: "json".into(),
            budget: Some(budget),
            with_blobs: false,
            cursor: None,
        }),
    };
    let response = call.answer(&d.daemon);
    let Some(value) = response.value().cloned() else {
        panic!("{response:?}")
    };
    serde_json::from_value(value).unwrap()
}

fn resume(call: Option<&polylogue_model::read::ReadyCall>) -> Option<RunnableRequest> {
    call.map(|call| serde_json::from_value(call.http.arguments.clone()).unwrap())
}

fn nodes(items: &[ReadItem]) -> Vec<String> {
    items
        .iter()
        .filter_map(|item| match &item.kind {
            ItemKind::Card { card } => match card.as_ref() {
                Card::LinkNode { reference, .. } => Some(reference.clone()),
                _ => None,
            },
            _ => None,
        })
        .collect()
}

#[test]
fn a_middle_links_page_does_not_visit_the_prefix() {
    let d = test_daemon::start("placeholder");
    family(&d);
    // Pages as large as a page may be (50 nodes) and small ones tile the
    // same walk, which ends with its count known.
    let tile = |budget| {
        let mut pages = vec![(None, read(&d, budget, None))];
        while let Some(next) = resume(pages.last().unwrap().1.document.footer.next.as_ref()) {
            let page = read(&d, budget, Some(&next));
            pages.push((Some(next), page));
        }
        pages
    };
    let flat = |pages: &[(Option<RunnableRequest>, ReadPage)]| -> Vec<ReadItem> {
        pages
            .iter()
            .flat_map(|(_, page)| page.document.items.clone())
            .collect()
    };
    let large = tile(1_000_000);
    let total = 1 + CHILDREN + GRANDCHILDREN;
    let whole = flat(&large);
    let order = nodes(&whole);
    assert_eq!(order.len(), total);
    assert_eq!(
        large.last().unwrap().1.document.footer.count,
        ValueState::known(total as u64)
    );
    // Depth first: c5's continuations follow it, before its next sibling.
    assert_eq!(order[0], "session:claude-code:root");
    let c5 = order
        .iter()
        .position(|node| node == "session:claude-code:c5")
        .unwrap();
    assert!(
        order[c5 + 1..=c5 + GRANDCHILDREN]
            .iter()
            .all(|node| node.starts_with("session:claude-code:g")),
        "{order:?}"
    );
    assert!(order[c5 + GRANDCHILDREN + 1].starts_with("session:claude-code:c"));
    let pages = tile(200);
    assert!(pages.len() > 6, "{} pages", pages.len());
    let tiled = flat(&pages);
    assert_eq!(tiled, whole);
    let mut before = 0;
    for (_, page) in &pages {
        assert_eq!(page.document.footer.position, Some(before));
        before += nodes(&page.document.items).len() as u64;
    }
    // A page in the walk of the grandchildren, after more than half of the
    // nodes, reads the links of its own nodes and of those above them.
    let (at, (anchor, expected)) = pages
        .iter()
        .enumerate()
        .find(|(_, (_, page))| page.document.footer.position > Some(total as u64 / 2))
        .unwrap();
    let shown = nodes(&expected.document.items).len() as u64;
    let reads = polylogue_store::lineage::link_reads();
    let again = read(&d, 200, anchor.as_ref());
    let visited = polylogue_store::lineage::link_reads() - reads;
    assert_eq!(again.document.items, expected.document.items);
    assert!(
        visited <= 2 * shown + 8,
        "{visited} link reads for a page of {shown} nodes at {:?}",
        expected.document.footer.position
    );
    // Going back from it reads what comes just before it.
    let previous = resume(again.document.footer.previous.as_ref()).expect("a previous page");
    let back = read(&d, 200, Some(&previous));
    let start: usize = pages[..at]
        .iter()
        .map(|(_, page)| page.document.items.len())
        .sum();
    let back_items = &back.document.items;
    assert!(!back_items.is_empty());
    assert_eq!(back_items[..], tiled[start - back_items.len()..start]);
    let back_nodes = nodes(back_items).len() as u64;
    assert_eq!(
        back.document.footer.position,
        expected
            .document
            .footer
            .position
            .map(|position| position - back_nodes)
    );
}

fn continued(parent: &str, child: &str) -> String {
    format!(
        "{}\n",
        serde_json::json!({"type":"continued-in","sessionId":parent,"continuedInSessionId":child})
    )
}

#[test]
fn cyclic_and_conflicting_families_tile_both_directions_without_a_guessed_root() {
    use polylogue_model::read::LineageIssue;
    struct Case {
        edges: &'static [(&'static str, &'static str)],
        seed: &'static str,
        cycle_issue: bool,
    }
    let cases = [
        Case {
            edges: &[("a", "b"), ("b", "a"), ("b", "tail")],
            seed: "tail",
            cycle_issue: true,
        },
        Case {
            edges: &[
                ("r", "fork"),
                ("c", "d"),
                ("d", "c"),
                ("c", "fork"),
                ("fork", "tail"),
            ],
            seed: "tail",
            cycle_issue: false,
        },
        Case {
            edges: &[("r", "c"), ("c", "d"), ("d", "c")],
            seed: "d",
            cycle_issue: false,
        },
    ];
    for Case {
        edges,
        seed,
        cycle_issue,
    } in cases
    {
        let d = test_daemon::start("placeholder");
        let ids = edges
            .iter()
            .flat_map(|(p, c)| [*p, *c])
            .collect::<std::collections::BTreeSet<_>>();
        for id in &ids {
            let mut source = polylogue_testkit::claude_user(&format!("question-{id}"), id, id);
            for (parent, child) in edges {
                if parent == id {
                    source.push_str(&continued(parent, child));
                }
            }
            std::fs::write(
                d.home
                    .join(".claude/projects/-p")
                    .join(format!("{id}.jsonl")),
                source,
            )
            .unwrap();
        }
        d.wait_for(|status| status["store"]["messages"] == ids.len() + 1);
        let target = format!("session:claude-code:{seed}");
        let whole = read_target(&d, &target, 1_000_000, None);
        let expected = ids
            .iter()
            .map(|id| format!("session:claude-code:{id}"))
            .collect::<std::collections::BTreeSet<_>>();
        let node_refs = nodes(&whole.document.items);
        assert_eq!(node_refs.len(), expected.len());
        assert_eq!(
            node_refs
                .into_iter()
                .collect::<std::collections::BTreeSet<_>>(),
            expected
        );
        let actual = whole
            .document
            .items
            .iter()
            .filter_map(|item| match &item.kind {
                ItemKind::Card { card } => match card.as_ref() {
                    Card::LinkEdge {
                        from, to, relation, ..
                    } => Some((from.clone(), to.clone(), relation.clone())),
                    _ => None,
                },
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(actual.len(), edges.len());
        assert_eq!(
            actual
                .into_iter()
                .collect::<std::collections::BTreeSet<_>>(),
            edges
                .iter()
                .map(|(p, c)| (
                    format!("session:claude-code:{p}"),
                    format!("session:claude-code:{c}"),
                    "continuation".to_owned()
                ))
                .collect()
        );
        let state = whole
            .document
            .items
            .iter()
            .find_map(|item| match &item.kind {
                ItemKind::Card { card } => match card.as_ref() {
                    Card::LinkNode {
                        seed: true,
                        lineage,
                        ..
                    } => lineage.as_deref(),
                    _ => None,
                },
                _ => None,
            })
            .unwrap();
        assert_eq!(state.root, None);
        assert_eq!(state.depth, None);
        if cycle_issue {
            assert!(matches!(state.issue, Some(LineageIssue::Cycle { .. })));
        } else {
            assert!(matches!(
                state.issue,
                Some(LineageIssue::ConflictingParents { .. })
            ));
        }
        let mut pages = vec![read_target(&d, &target, 60, None)];
        while let Some(next) = resume(pages.last().unwrap().document.footer.next.as_ref()) {
            pages.push(read_target(&d, &target, 60, Some(&next)));
        }
        let tiled = pages
            .iter()
            .flat_map(|page| page.document.items.clone())
            .collect::<Vec<_>>();
        assert_eq!(tiled, whole.document.items);
        assert_eq!(
            pages.last().unwrap().document.footer.count,
            ValueState::known(expected.len() as u64)
        );
        assert!(
            pages
                .iter()
                .any(|page| nodes(&page.document.items).is_empty()),
            "cross-edges must fit their own pages"
        );
        let mut offset = pages[0].document.items.len();
        for page in pages.iter().skip(1) {
            let previous = resume(page.document.footer.previous.as_ref()).unwrap();
            let back = read_target(&d, &target, 60, Some(&previous));
            assert!(!back.document.items.is_empty());
            assert_eq!(
                back.document.items,
                tiled[offset - back.document.items.len()..offset]
            );
            offset += page.document.items.len();
        }
    }
}

#[test]
fn alternative_native_anchors_and_spawning_calls_keep_their_exact_edge_facts() {
    let d = test_daemon::start("placeholder");
    let dir = d.home.join(".claude/projects/-p");
    let mut parent = polylogue_testkit::claude_user("first", "root", "first prompt");
    parent.push_str(&polylogue_testkit::claude_user(
        "second",
        "root",
        "second prompt",
    ));
    parent.push_str(&format!("{}\n",serde_json::json!({"type":"progress","sessionId":"root","parentToolUseID":"spawn-call","data":{"type":"agent_progress","agentId":"worker"}})));
    let mut child = polylogue_testkit::claude_user("own", "child", "child prompt");
    for anchor in ["first", "second"] {
        child.push_str(&format!("{}\n",serde_json::json!({"type":"fork-context-ref","sessionId":"child","parentSessionId":"root","parentLastUuid":anchor})));
    }
    std::fs::write(dir.join("root.jsonl"), parent).unwrap();
    std::fs::write(dir.join("child.jsonl"), child).unwrap();
    d.wait_for(|status| status["store"]["messages"] == 4);
    let page = read_target(&d, "session:claude-code:child", 1_000_000, None);
    let mut anchors = std::collections::BTreeSet::new();
    let mut cross = 0;
    let mut calls = Vec::new();
    for item in &page.document.items {
        if let ItemKind::Card { card } = &item.kind {
            match card.as_ref() {
                Card::LinkNode {
                    reference,
                    parents,
                    lineage,
                    ..
                } if reference == "session:claude-code:child" => {
                    assert_eq!(parents, &["session:claude-code:root"]);
                    let lineage = lineage.as_deref().unwrap();
                    assert_eq!(
                        lineage.root.as_ref().unwrap().to_string(),
                        "claude-code:root"
                    );
                    assert_eq!(lineage.depth, Some(1));
                    assert!(lineage.issue.is_none());
                }
                Card::LinkEdge {
                    from,
                    to,
                    relation,
                    branch_point,
                    spawning_call,
                    copies_prefix,
                    cross_link,
                    ..
                } => {
                    assert_eq!(from, "session:claude-code:root");
                    assert_eq!(relation, "spawn");
                    if to == "session:claude-code:child" {
                        anchors.insert(branch_point.clone().unwrap());
                        assert!(spawning_call.is_none());
                        assert!(*copies_prefix);
                        cross += usize::from(*cross_link);
                    }
                    if to == "session:claude-code:agent-worker" {
                        assert!(branch_point.is_none());
                        assert!(!*copies_prefix);
                        calls.push(spawning_call.clone());
                    }
                }
                _ => {}
            }
        }
    }
    assert_eq!(
        anchors,
        [
            "message:claude-code:first".to_owned(),
            "message:claude-code:second".to_owned()
        ]
        .into_iter()
        .collect()
    );
    assert_eq!(cross, 1);
    assert_eq!(calls, vec![Some("spawn-call".into())]);
}
