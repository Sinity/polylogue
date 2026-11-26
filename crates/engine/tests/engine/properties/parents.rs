//! The parents home, read after the real engine has stored its files.
use polylogue_model::{SessionId, SessionKind, WrittenBy, money::usd_display};
use polylogue_store::query::{Options, compile};
use polylogue_testkit::{Morrow, Set};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::ControlFlow;

#[test]
fn parents_set_lineage_matches_morrow_facts() {
    // FACTS.md: the second home holds #1–#4 plus L1–L8 and L1-side.
    // Each row gives the session kind, own messages, messages with parents,
    // human prompts, own cost and tree cost from the corresponding table row.
    let expected = [
        (
            "claude-code:7f3a2c10-0001-4000-8000-000000000001",
            SessionKind::Main,
            12,
            12,
            3,
            "$1.26",
            None,
        ),
        (
            "claude-code:7f3a2c10-0002-4000-8000-000000000002",
            SessionKind::Main,
            18,
            18,
            3,
            "$2.85",
            None,
        ),
        (
            "claude-code:agent-a7f3a2c1000000003",
            SessionKind::Subagent,
            6,
            6,
            0,
            "$0.16",
            None,
        ),
        (
            "claude-code:7f3a2c10-0004-4000-8000-000000000004",
            SessionKind::Main,
            11,
            29,
            3,
            "$1.59",
            None,
        ),
        (
            "claude-code:7f3a2c10-0101-4000-8000-000000000101",
            SessionKind::Main,
            10,
            10,
            2,
            "$0.76",
            Some("$0.94"),
        ),
        (
            "claude-code:7f3a2c10-0101-4000-8000-000000000101~side-d1500000-0000-4000-8000-000000000001",
            SessionKind::Sidechain,
            2,
            2,
            0,
            "$0.03",
            Some("$0.03"),
        ),
        (
            "claude-code:7f3a2c10-0101-4000-8000-000000000101/a7f3a2c1000000102",
            SessionKind::Subagent,
            6,
            6,
            0,
            "$0.10",
            Some("$0.15"),
        ),
        (
            "claude-code:7f3a2c10-0101-4000-8000-000000000101/a7f3a2c1000000103",
            SessionKind::Subagent,
            4,
            4,
            0,
            "$0.05",
            Some("$0.05"),
        ),
        (
            "claude-code:7f3a2c10-0101-4000-8000-000000000101/a7f3a2c1000000104",
            SessionKind::Subagent,
            4,
            8,
            0,
            "$0.05",
            Some("$0.05"),
        ),
        (
            "claude-code:7f3a2c10-0105-4000-8000-000000000105",
            SessionKind::Main,
            8,
            12,
            2,
            "$0.60",
            Some("$0.66"),
        ),
        (
            "claude-code:7f3a2c10-0105-4000-8000-000000000105/a7f3a2c1000000106",
            SessionKind::Subagent,
            4,
            4,
            0,
            "$0.03",
            Some("$0.03"),
        ),
        (
            "claude-code:7f3a2c10-0105-4000-8000-000000000105/a7f3a2c1000000107",
            SessionKind::Subagent,
            5,
            5,
            0,
            "$0.03",
            Some("$0.03"),
        ),
        (
            "claude-code:7f3a2c10-0108-4000-8000-000000000108",
            SessionKind::Main,
            6,
            6,
            2,
            "$0.36",
            Some("$0.36"),
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
        fields: ["id", "messages", "cost", "tree_cost"]
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
    for (id, kind, own, shown, human, cost, _) in expected {
        let sid = SessionId::parse(id).unwrap();
        assert_eq!(
            reader.session(&sid).unwrap().unwrap().kind,
            kind,
            "session kind: {id}"
        );
        let messages = reader.messages(&sid, false).unwrap();
        assert_eq!(messages.len(), own, "own messages: {id}");
        assert_eq!(
            messages
                .iter()
                .filter(|m| WrittenBy::of(m) == WrittenBy::Human)
                .count(),
            human,
            "human prompts: {id}"
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
        assert_eq!(
            row["messages"].as_u64(),
            Some(own as u64),
            "session message count: {id}"
        );
        let amount = row["cost"].as_i64().unwrap();
        assert_eq!(usd_display(amount), cost, "own cost: {id}");
        costs.insert(id.to_owned(), amount);
        for link in reader.links(&sid).unwrap() {
            links.insert((
                link.parent.to_string(),
                link.child.to_string(),
                link.kind.as_str().to_owned(),
                link.call_id,
                link.how.map(|how| how.as_str().to_owned()),
            ));
        }
    }
    // FACTS.md: provider links table, excluding the Codex pair absent here;
    // copied history creates no provider link, and the briefing is in the hook spool.
    let expected_links = [
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
    // #4 has eighteen, and L8's three copies count as its own.
    let unique: BTreeSet<_> = own_ids.iter().collect();
    assert_eq!(
        unique.len(),
        own_ids.len(),
        "messages stored once across sessions"
    );
    for (child, parent, copies) in [(3, 1, 18), (8, 6, 4), (9, 4, 4)] {
        assert_eq!(
            &shown_ids[expected[child].0][..copies],
            &shown_ids[expected[parent].0][..copies],
            "copied message identities: {}",
            expected[child].0
        );
    }
    // FACTS.md: tree cost includes spawned descendants, never copies or forks.
    for (id, _, _, _, _, _, tree_cost) in expected {
        let Some(tree_cost) = tree_cost else { continue };
        let row = rows.iter().find(|row| row["id"] == id).unwrap();
        let total = row["tree_cost"].as_i64().unwrap();
        assert_eq!(usd_display(total), tree_cost, "tree cost: {id}");
        if id == expected[4].0 {
            // FACTS.md: L1's tree is exactly 0.9364.
            assert_eq!(total, 936_400_000, "L1 exact tree cost");
        }
    }
    // FACTS.md: whole home 7.8715, #1–#4 5.8626, parents set 2.0089.
    assert_eq!(
        costs.values().sum::<i64>(),
        7_871_500_000,
        "whole home cost"
    );
    assert_eq!(
        expected[..4].iter().map(|r| costs[r.0]).sum::<i64>(),
        5_862_600_000,
        "base cost"
    );
    assert_eq!(
        expected[4..].iter().map(|r| costs[r.0]).sum::<i64>(),
        2_008_900_000,
        "parents cost"
    );
}
