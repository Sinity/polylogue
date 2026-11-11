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
    let call = match cursor {
        Some(resume) => {
            let mut resume = resume.clone();
            resume.arguments["page_tokens"] = serde_json::json!(budget);
            test_daemon::Call::from_request(&resume)
        }
        None => test_daemon::read(crate::daemon::service::read::ReadRequest {
            target: "session:claude-code:root".into(),
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
    let reads = polylogue_store::items::link_reads();
    let again = read(&d, 200, anchor.as_ref());
    let visited = polylogue_store::items::link_reads() - reads;
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
