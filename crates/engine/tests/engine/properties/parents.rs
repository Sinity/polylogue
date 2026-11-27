//! The parents home, read after the real engine has stored its files.
use polylogue_model::{SessionId, SessionKind, WrittenBy};
use polylogue_store::query::{Options, compile};
use polylogue_testkit::{Morrow, Set};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::ControlFlow;

const L1: &str = "claude-code:7f3a2c10-0101-4000-8000-000000000101";
const L1_SIDE: &str =
    "claude-code:7f3a2c10-0101-4000-8000-000000000101~side-d1500000-0000-4000-8000-000000000001";
const L4: &str = "claude-code:agent-a7f3a2c1000000104";
const L5: &str = "claude-code:7f3a2c10-0105-4000-8000-000000000105";

fn assert_fact<T: std::fmt::Debug + PartialEq>(id: &str, fact: &str, actual: T, expected: T) {
    assert_eq!(actual, expected, "{fact}: {id}");
}

#[test]
fn parents_set_lineage_matches_morrow_facts() {
    // FACTS.md: the second home holds #1–#4 plus L1–L8, L1-side and 0199.
    // Each row gives the session kind, own messages, messages with parents,
    // and human prompts from the corresponding table row.
    let expected = [
        (
            "claude-code:7f3a2c10-0001-4000-8000-000000000001",
            SessionKind::Main,
            12,
            12,
            3,
        ),
        (
            "claude-code:7f3a2c10-0002-4000-8000-000000000002",
            SessionKind::Main,
            18,
            18,
            3,
        ),
        (
            "claude-code:agent-a7f3a2c1000000003",
            SessionKind::Subagent,
            6,
            10,
            0,
        ),
        (
            "claude-code:7f3a2c10-0004-4000-8000-000000000004",
            SessionKind::Main,
            11,
            29,
            3,
        ),
        (L1, SessionKind::Main, 10, 10, 2),
        (L1_SIDE, SessionKind::Sidechain, 2, 2, 0),
        (
            "claude-code:agent-a7f3a2c1000000102",
            SessionKind::Subagent,
            6,
            6,
            0,
        ),
        (
            "claude-code:agent-a7f3a2c1000000103",
            SessionKind::Subagent,
            4,
            4,
            0,
        ),
        (L4, SessionKind::Subagent, 4, 8, 0),
        (L5, SessionKind::Main, 8, 12, 2),
        (
            "claude-code:agent-a7f3a2c1000000106",
            SessionKind::Subagent,
            4,
            4,
            0,
        ),
        (
            "claude-code:agent-a7f3a2c1000000107",
            SessionKind::Subagent,
            5,
            5,
            0,
        ),
        (
            "claude-code:7f3a2c10-0108-4000-8000-000000000108",
            SessionKind::Main,
            3,
            6,
            0,
        ),
        (
            "claude-code:7f3a2c10-0199-4000-8000-000000000199",
            SessionKind::Main,
            3,
            3,
            2,
        ),
    ];
    let w = Morrow::build(Set::Parents);
    let pool = w.reads();
    let reader = pool.reader().unwrap();
    let options = Options {
        clock: query_language::time::Clock::new(
            "2026-10-08T12:00:00Z".parse().unwrap(),
            query_language::time::DaySetting::new(
                jiff::tz::TimeZone::UTC,
                jiff::civil::time(0, 0, 0, 0),
            )
            .unwrap(),
        ),
        origins: vec!["claude-code".into()],
        page_size: None,
        random_seed: None,
        embeddings: false,
        self_session: None,
        here: None,
        seen_marks_of: None,
        saved: Default::default(),
        fields: [
            "id",
            "messages",
            "cost",
            "tree_cost",
            "subagent_cost",
            "cost_complete",
        ]
        .map(str::to_owned)
        .to_vec(),
        kept: Default::default(),
        provider_catalog: None,
    };
    let program = compile("sessions where physical:true | sort by id", &options).unwrap();
    let rows = reader.query_rows(&program).unwrap();
    let mut expected_ids: Vec<_> = expected.iter().map(|row| row.0).collect();
    expected_ids.sort();
    assert_eq!(
        rows.iter()
            .map(|row| row["id"].as_str().unwrap())
            .collect::<Vec<_>>(),
        expected_ids,
        "parents home sessions"
    );
    let mut links = BTreeSet::new();
    let mut costs = BTreeMap::new();
    let mut own_ids = Vec::new();
    let mut shown_ids = BTreeMap::new();
    for (id, kind, own, shown, human) in expected {
        let sid = SessionId::parse(id).unwrap();
        assert_eq!(
            reader.session(&sid).unwrap().unwrap().kind,
            kind,
            "session kind: {id}"
        );
        let messages = reader.messages(&sid, false).unwrap();
        assert_fact(id, "own messages", messages.len(), own);
        assert_fact(
            id,
            "human prompts",
            messages
                .iter()
                .filter(|m| WrittenBy::of(m) == WrittenBy::Human)
                .count(),
            human,
        );
        own_ids.extend(messages.iter().map(|m| m.id.clone()));
        let selected = reader.select_messages(&sid, true).unwrap();
        let mut ids = Vec::new();
        selected
            .visit(Default::default(), |m| {
                ids.push(m.id);
                ControlFlow::Continue(())
            })
            .unwrap();
        assert_eq!(ids.len(), shown, "messages with parents: {id}");
        shown_ids.insert(id, ids);
        let row = rows.iter().find(|row| row["id"] == id).unwrap();
        assert_fact(
            id,
            "session message count",
            row["messages"].as_u64(),
            Some(own as u64),
        );
        assert_eq!(
            row["cost_complete"].as_i64(),
            Some(1),
            "own cost complete: {id}"
        );
        let amount = row["cost"].as_i64().expect("known own cost");
        costs.insert(id.to_owned(), amount);
        for link in reader.links(&sid).unwrap() {
            links.insert((
                link.parent.to_string(),
                link.child.to_string(),
                link.kind.as_str().to_owned(),
                link.call_id,
                link.reason.map(|reason| reason.as_str().to_owned()),
            ));
        }
    }
    // FACTS.md: provider links table, excluding the Codex pair absent here;
    // copied history creates no provider link, and the briefing is in the hook spool.
    let expected_links: BTreeSet<_> = [
        (1, 2, "spawn", Some("toolu_0203")),
        (4, 5, "spawn", Some("toolu_1102")),
        (4, 6, "spawn", Some("toolu_1101")),
        (6, 7, "spawn", Some("toolu_1201")),
        (4, 9, "fork", None),
        (9, 10, "spawn", Some("toolu_1501")),
        (9, 11, "spawn", Some("toolu_1502")),
    ]
    .map(|(p, c, kind, call)| {
        (
            expected[p].0.to_owned(),
            expected[c].0.to_owned(),
            kind.to_owned(),
            call.map(str::to_owned),
            None,
        )
    })
    .into_iter()
    .collect();
    assert_eq!(links, expected_links, "provider links");
    // FACTS.md: a copied message is stored once; L4 and L5 have four copies,
    // #4 has eighteen, and L8 has three copies belonging to 0199.
    // #3's fork-context-ref names #2's prefix through its fourth message.
    let unique: BTreeSet<_> = own_ids.iter().collect();
    assert_eq!(
        unique.len(),
        own_ids.len(),
        "messages stored once across sessions"
    );
    for (child, parent, copies) in [(2, 1, 4), (3, 1, 18), (8, 6, 4), (9, 4, 4), (12, 13, 3)] {
        assert_fact(
            expected[child].0,
            "copied message identities",
            &shown_ids[expected[child].0][..copies],
            &shown_ids[expected[parent].0][..copies],
        );
    }
    // Morrow's example prices are not loadable into an engine-built archive yet.
    // FACTS.md's links and docs/measures.md define each spawn tree independently.
    for (index, (id, ..)) in expected.iter().enumerate() {
        let members: &[usize] = match index {
            1 => &[1, 2],
            4 => &[4, 5, 6, 7],
            6 => &[6, 7],
            9 => &[9, 10, 11],
            _ => std::slice::from_ref(&index),
        };
        let tree_cost: i64 = members
            .iter()
            .map(|&member| costs[expected[member].0])
            .sum();
        let row = rows.iter().find(|row| row["id"] == *id).unwrap();
        assert_fact(
            id,
            "spawn tree cost",
            row["tree_cost"].as_i64(),
            Some(tree_cost),
        );
        assert_fact(
            id,
            "subagent cost",
            row["subagent_cost"].as_i64(),
            Some(tree_cost - costs[*id]),
        );
    }
}
