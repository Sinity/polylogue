//! Properties of the query language over generated archives. Each archive
//! is written row by row from generated rows, with unknown values in every
//! field that can lack one; generated conditions run through the real
//! compiler and reader, and [`reference`] says what each one means.

use super::*;
use polylogue_store::fixture::{CallFacts, Fixture, MessageFacts, SessionFacts};
use proptest::prelude::*;
use proptest::strategy::Union;
use proptest::test_runner::FileFailurePersistence;
use reference::{
    COMMANDS, COSTS, CallRow, Cond, FIRST_DAY, HOUR, Leaf, MODELS, MessageRow, ORIGINS, OUTCOMES,
    REPO_NAMES, REPOS, Row, Rows, SPEAKERS, SessionRow, THRESHOLDS, TOOLS, Truth, Unit, WORDS,
};
use std::collections::BTreeSet;

mod reference;

fn config(cases: u32) -> ProptestConfig {
    ProptestConfig {
        cases,
        // Seeds of past failures are kept beside this file; cargo runs
        // unit tests from the package directory.
        failure_persistence: Some(Box::new(FileFailurePersistence::Direct(
            "src/daemon/query/tests/generated/regressions.txt",
        ))),
        ..ProptestConfig::default()
    }
}

/// A time on one of the three generated days, six hours apart, so equal
/// times are common; sometimes unknown.
fn time() -> impl Strategy<Value = Option<i64>> {
    prop::option::weighted(0.85, (0i64..12).prop_map(|n| FIRST_DAY + n * 6 * HOUR))
}

fn call_row() -> impl Strategy<Value = CallRow> {
    (
        0..TOOLS.len(),
        0..COMMANDS.len(),
        0..OUTCOMES.len(),
        prop::option::of(prop::sample::select(vec![0i64, 1, 101])),
        time(),
    )
        .prop_map(|(tool, command, outcome, exit_code, at)| CallRow {
            tool: TOOLS[tool],
            command: (TOOLS[tool].1 == "shell").then_some(COMMANDS[command]),
            outcome: OUTCOMES[outcome],
            exit_code,
            at,
        })
}

fn message_row() -> impl Strategy<Value = MessageRow> {
    (
        0..SPEAKERS.len(),
        prop::option::of(0..MODELS.len()),
        prop::option::of(0..COSTS.len()),
        time(),
        prop::option::of(prop::collection::vec(0..WORDS.len(), 1..4)),
        prop_oneof![6 => Just(None), 1 => Just(Some(false)), 1 => Just(Some(true))],
        prop::collection::vec(call_row(), 0..3),
    )
        .prop_map(|(speaker, model, cost, at, words, path, calls)| {
            let model_wrote = SPEAKERS[speaker].1 == "model";
            MessageRow {
                speaker: SPEAKERS[speaker],
                model: model.filter(|_| model_wrote).map(|m| MODELS[m]),
                cost: cost.filter(|_| model_wrote).map(|c| COSTS[c]),
                at,
                text: words.map(|w| w.iter().map(|i| WORDS[*i]).collect::<Vec<_>>().join(" ")),
                path,
                calls: if model_wrote { calls } else { vec![] },
            }
        })
}

fn session_row() -> impl Strategy<Value = SessionRow> {
    (
        0..ORIGINS.len(),
        prop::bool::weighted(0.2),
        prop::option::of(0..REPOS.len()),
        prop::collection::vec(message_row(), 0..5),
    )
        .prop_map(|(origin, subagent, repo, messages)| SessionRow {
            origin: ORIGINS[origin],
            subagent,
            repo: repo.map(|r| REPOS[r]),
            messages,
        })
}

fn rows() -> impl Strategy<Value = Rows> {
    prop::collection::vec(session_row(), 1..7).prop_map(|sessions| Rows { sessions })
}

fn unit() -> impl Strategy<Value = Unit> {
    prop::sample::select(Unit::ALL.to_vec())
}

fn leaf(unit: Unit) -> BoxedStrategy<Leaf> {
    let origin = || prop::sample::select(ORIGINS.to_vec());
    let threshold = || prop::sample::select(THRESHOLDS.to_vec());
    let day = || 1i64..5;
    let choices: Vec<BoxedStrategy<Leaf>> = match unit {
        Unit::Sessions => vec![
            origin().prop_map(Leaf::Origin).boxed(),
            (origin(), origin())
                .prop_map(|(a, b)| Leaf::EitherOrigin(a, b))
                .boxed(),
            any::<bool>().prop_map(Leaf::Subagent).boxed(),
            prop::sample::select(REPO_NAMES.to_vec())
                .prop_map(Leaf::Repo)
                .boxed(),
            Just(Leaf::RepoUnknown).boxed(),
            prop::sample::select(MODELS.to_vec())
                .prop_map(Leaf::Model)
                .boxed(),
            Just(Leaf::ModelUnknown).boxed(),
            threshold().prop_map(Leaf::CostAbove).boxed(),
            threshold().prop_map(Leaf::CostBelow).boxed(),
            Just(Leaf::CostUnknown).boxed(),
            (0usize..5).prop_map(Leaf::MessagesAtLeast).boxed(),
            (0usize..3).prop_map(Leaf::ToolCallsAbove).boxed(),
            (1usize..3).prop_map(Leaf::ErrorsAtLeast).boxed(),
            prop::sample::select(vec!["errors", "tool_calls", "branches"])
                .prop_map(Leaf::Has)
                .boxed(),
            day().prop_map(Leaf::StartedBefore).boxed(),
            day().prop_map(Leaf::StartedFrom).boxed(),
            Just(Leaf::StartedUnknown).boxed(),
            cond(Unit::ToolCalls)
                .prop_map(|c| Leaf::ExistsCall(Box::new(c)))
                .boxed(),
        ],
        Unit::Messages => vec![
            prop::sample::select(vec!["user", "assistant"])
                .prop_map(Leaf::Role)
                .boxed(),
            prop::sample::select(vec!["human", "model", "harness", "unknown"])
                .prop_map(Leaf::WrittenBy)
                .boxed(),
            prop::sample::select(MODELS.to_vec())
                .prop_map(Leaf::MessageModel)
                .boxed(),
            Just(Leaf::MessageModelUnknown).boxed(),
            threshold().prop_map(Leaf::MessageCostAbove).boxed(),
            Just(Leaf::MessageCostUnknown).boxed(),
            day().prop_map(Leaf::AtBefore).boxed(),
            Just(Leaf::AtUnknown).boxed(),
            (0usize..4).prop_map(Leaf::WordsAtLeast).boxed(),
            prop::sample::select(WORDS.to_vec())
                .prop_map(Leaf::Word)
                .boxed(),
            Just(Leaf::Abandoned).boxed(),
            origin().prop_map(Leaf::SessionOrigin).boxed(),
        ],
        Unit::ToolCalls => vec![
            prop::sample::select(TOOLS.iter().map(|t| t.0).collect::<Vec<_>>())
                .prop_map(Leaf::Tool)
                .boxed(),
            prop::sample::select(TOOLS.iter().map(|t| t.1).collect::<Vec<_>>())
                .prop_map(Leaf::Action)
                .boxed(),
            prop::sample::select(OUTCOMES.to_vec())
                .prop_map(Leaf::Outcome)
                .boxed(),
            Just(Leaf::ExitZero).boxed(),
            Just(Leaf::ExitNonzero).boxed(),
            Just(Leaf::ExitUnknown).boxed(),
            // Three characters or more search the trigram index; fewer
            // compare the text.
            prop::sample::select(vec!["cargo te", "git", "go"])
                .prop_map(Leaf::Command)
                .boxed(),
            Just(Leaf::CommandUnknown).boxed(),
            day().prop_map(Leaf::CallAtBefore).boxed(),
            Just(Leaf::CallAtUnknown).boxed(),
            origin().prop_map(Leaf::CallSessionOrigin).boxed(),
        ],
    };
    Union::new(choices).boxed()
}

/// A condition over `unit`: fields joined by AND, OR and NOT, three deep.
fn cond(unit: Unit) -> BoxedStrategy<Cond> {
    leaf(unit)
        .prop_map(Cond::Leaf)
        .prop_recursive(3, 12, 2, |inner| {
            prop_oneof![
                (inner.clone(), inner.clone())
                    .prop_map(|(a, b)| Cond::And(Box::new(a), Box::new(b))),
                (inner.clone(), inner.clone())
                    .prop_map(|(a, b)| Cond::Or(Box::new(a), Box::new(b))),
                inner.prop_map(|a| Cond::Not(Box::new(a))),
            ]
        })
        .boxed()
}

fn unit_and_cond() -> impl Strategy<Value = (Unit, Cond)> {
    unit().prop_flat_map(|u| (Just(u), cond(u)))
}

fn unit_and_conds() -> impl Strategy<Value = (Unit, Cond, Cond)> {
    unit().prop_flat_map(|u| (Just(u), cond(u), cond(u)))
}

/// The generated rows written into an archive, sessions and messages in
/// the order `order` gives; every message keeps its place in session order.
fn write(rows: &Rows, order: &[(usize, usize)]) -> TestArchive {
    let dir = tempfile::tempdir().unwrap();
    let mut fx = Fixture::new(&dir.path().join("archive")).unwrap();
    let mut sessions = std::collections::BTreeMap::new();
    let mut place = std::collections::BTreeMap::new();
    for (i, s) in rows.sessions.iter().enumerate() {
        for j in 0..s.messages.len() {
            let next = place.len() as i64 + 1;
            place.insert((i, j), next);
        }
    }
    let mut session = |fx: &mut Fixture, i: usize| -> i64 {
        *sessions.entry(i).or_insert_with(|| {
            let s = &rows.sessions[i];
            fx.session(SessionFacts {
                id: &reference::session_id(rows, i),
                kind: if s.subagent { "subagent" } else { "main" },
                repo: s.repo,
                started: s.started(),
                last: s.last_active(),
                ..SessionFacts::default()
            })
            .unwrap()
        })
    };
    for &(i, j) in order {
        let pk = session(&mut fx, i);
        let Some(m) = rows.sessions[i].messages.get(j) else {
            continue;
        };
        let id = reference::message_id(i, j);
        let message = fx
            .message(
                pk,
                MessageFacts {
                    id: &id,
                    role: m.speaker.0,
                    written_by: m.speaker.1,
                    prompt_kind: (m.speaker.1 == "human").then_some("prose"),
                    at: m.at,
                    model: m.model,
                    text: m.text.as_deref(),
                    cost_nusd: m.cost,
                    on_active_path: m.path,
                    seq: Some(place[&(i, j)]),
                    ..MessageFacts::default()
                },
            )
            .unwrap();
        for c in &m.calls {
            fx.tool_call(
                pk,
                message,
                CallFacts {
                    tool: c.tool.0,
                    action: c.tool.1,
                    command: c.command,
                    outcome: c.outcome,
                    exit_code: c.exit_code,
                    at: c.at,
                    output: Some("done"),
                    ..CallFacts::default()
                },
            )
            .unwrap();
        }
    }
    TestArchive { _dir: dir, fx }
}

/// Every session, then every message of each, in the order generated.
/// A session without messages appears once, as `(i, 0)`.
fn written_order(rows: &Rows) -> Vec<(usize, usize)> {
    rows.sessions
        .iter()
        .enumerate()
        .flat_map(|(i, s)| (0..s.messages.len().max(1)).map(move |j| (i, j)))
        .collect()
}

fn build(rows: &Rows) -> TestArchive {
    write(rows, &written_order(rows))
}

/// Runs a query for its whole answer.
fn all(w: &TestArchive, text: &str) -> Value {
    let mut request = ask(text);
    request.page_size = Some(1 << 40);
    run_q(w, request).unwrap_or_else(|f| panic!("{text}: {} {}", f.kind, f.message))
}

fn result_rows(v: &Value) -> Vec<Value> {
    v["rows"].as_array().cloned().unwrap_or_default()
}

/// Each result row as [`reference::key`] names it.
fn keys(unit: Unit, v: &Value) -> Vec<String> {
    result_rows(v)
        .iter()
        .map(|r| match unit {
            Unit::Sessions | Unit::Messages => r["id"].as_str().unwrap_or_default().to_owned(),
            Unit::ToolCalls => reference::call_key(
                r["message"].as_str().unwrap_or_default(),
                r["tool"].as_str().unwrap_or_default(),
                r["command"].as_str(),
                r["outcome"].as_str().unwrap_or_default(),
                r["exit_code"].as_i64(),
            ),
        })
        .collect()
}

fn key_set(unit: Unit, v: &Value) -> BTreeSet<String> {
    keys(unit, v).into_iter().collect()
}

fn reference_keys(rows: &Rows, found: &[Row]) -> Vec<String> {
    found.iter().map(|r| reference::key(rows, *r)).collect()
}

fn sorted(mut keys: Vec<String>) -> Vec<String> {
    keys.sort();
    keys
}

/// `UNIT where <every row> <conditions>`: the unit's rows with no default
/// left, so conditions combined after it are compared over one source.
fn over(unit: Unit, conditions: &str) -> String {
    format!("{} where {} {conditions}", unit.name(), unit.every_row())
}

/// Every row of the unit with no default left, by key.
fn every_row(rows: &Rows, unit: Unit) -> Vec<(String, Row)> {
    let mut out = vec![];
    for (i, s) in rows.sessions.iter().enumerate() {
        match unit {
            Unit::Sessions => out.push(Row::Session(i)),
            Unit::Messages => out.extend((0..s.messages.len()).map(|j| Row::Message(i, j))),
            Unit::ToolCalls => {
                for (j, m) in s.messages.iter().enumerate() {
                    if m.path != Some(false) {
                        out.extend((0..m.calls.len()).map(|k| Row::Call(i, j, k)));
                    }
                }
            }
        }
    }
    out.into_iter()
        .map(|row| (reference::key(rows, row), row))
        .collect()
}

proptest! {
    #![proptest_config(config(48))]

    /// Every generated query returns the rows the reference evaluator
    /// gives, in the unit's order (as a set when words rank them), with
    /// the same count.
    #[test]
    fn generated_queries_return_what_the_reference_evaluator_returns(
        rows in rows(),
        (unit, cond) in unit_and_cond(),
    ) {
        let w = build(&rows);
        let text = format!("{} where {}", unit.name(), cond.text());
        let v = all(&w, &text);
        let got = keys(unit, &v);
        let want = reference_keys(&rows, &reference::answer(&rows, unit, &cond));
        if cond.ranks() {
            prop_assert_eq!(sorted(got.clone()), sorted(want.clone()), "{}", text);
        } else {
            prop_assert_eq!(&got, &want, "{}", text);
        }
        prop_assert_eq!(v["count"].as_u64(), Some(want.len() as u64), "{}", text);
    }

    /// Conditions combine as sets: `A B` and `B A` agree, `A B` is the
    /// intersection and `A OR B` the union of `A`'s and `B`'s rows, `NOT
    /// NOT A` and `A A` are `A`, and adding a condition never adds rows.
    #[test]
    fn conditions_combine_as_sets(rows in rows(), (unit, a, b) in unit_and_conds()) {
        let w = build(&rows);
        let (a, b) = (a.text(), b.text());
        let q = |conditions: String| key_set(unit, &all(&w, &over(unit, &conditions)));
        let only_a = q(format!("({a})"));
        let only_b = q(format!("({b})"));
        let both = q(format!("({a}) ({b})"));
        prop_assert_eq!(&both, &q(format!("({b}) ({a})")), "{} / {}", a, b);
        prop_assert_eq!(
            &both,
            &only_a.intersection(&only_b).cloned().collect::<BTreeSet<_>>(),
            "{} AND {}", a, b
        );
        prop_assert_eq!(
            q(format!("(({a}) OR ({b}))")),
            only_a.union(&only_b).cloned().collect::<BTreeSet<_>>(),
            "{} OR {}", a, b
        );
        prop_assert_eq!(&q(format!("NOT (NOT ({a}))")), &only_a, "NOT NOT {}", a);
        prop_assert_eq!(&q(format!("({a}) ({a})")), &only_a, "{} twice", a);
        prop_assert!(both.is_subset(&only_a));
    }

    /// `Q | agg count` counts `Q`'s rows; `sessions where Q | agg sum:cost`
    /// adds each matched session's cost once and counts the unknown ones;
    /// `S | tool_calls where C` returns only tool calls of `S`'s sessions,
    /// exactly those `C` holds for.
    #[test]
    fn aggregates_and_scopes_follow_the_rows(
        rows in rows(),
        (unit, a) in unit_and_cond(),
        s in cond(Unit::Sessions),
        c in cond(Unit::ToolCalls),
    ) {
        let w = build(&rows);
        let query = over(unit, &format!("({})", a.text()));
        let listed = result_rows(&all(&w, &query)).len() as u64;
        let counted = all(&w, &format!("{query} | agg count"));
        prop_assert_eq!(counted["rows"][0]["count"].as_u64(), Some(listed), "{}", query);

        let sessions = over(Unit::Sessions, &format!("({})", s.text()));
        let matched = keys(Unit::Sessions, &all(&w, &sessions));
        let costs: Vec<Option<i64>> = (0..rows.sessions.len())
            .filter(|i| matched.contains(&reference::session_id(&rows, *i)))
            .map(|i| rows.sessions[i].cost())
            .collect();
        let summed = all(&w, &format!("{sessions} | agg sum:cost"));
        let known: i64 = costs.iter().flatten().sum();
        let sum = summed["rows"][0]["sum_cost"].as_f64();
        if costs.iter().any(Option::is_some) {
            let sum = sum.unwrap_or(f64::NAN);
            prop_assert!((sum - known as f64 / 1e9).abs() < 1e-9, "{}: {} {}", sessions, sum, known);
        } else {
            prop_assert_eq!(sum, None, "{}", sessions);
        }
        prop_assert_eq!(
            summed["rows"][0]["unknown_cost"].as_u64(),
            Some(costs.iter().filter(|c| c.is_none()).count() as u64),
            "{}", sessions
        );

        let scoped = format!("{sessions} | {}", over(Unit::ToolCalls, &format!("({})", c.text())));
        let got = sorted(keys(Unit::ToolCalls, &all(&w, &scoped)));
        let want = sorted(
            every_row(&rows, Unit::ToolCalls)
                .into_iter()
                .filter(|(_, row)| {
                    let Row::Call(i, _, _) = row else { unreachable!() };
                    matched.contains(&reference::session_id(&rows, *i))
                        && reference::truth(&rows, &c, *row) == Truth::Yes
                })
                .map(|(key, _)| key)
                .collect(),
        );
        prop_assert_eq!(got, want, "{}", scoped);
    }

    /// `U where A B`, `U where A | where B` and `U where A` filtered by B
    /// row by row give the same rows in the same order. Words joined in one
    /// `where` rank together, so with words on both sides only the rows
    /// are compared.
    #[test]
    fn successive_filters_equal_their_conjunction(
        rows in rows(),
        (unit, a, b) in unit_and_conds(),
    ) {
        let w = build(&rows);
        let (at, bt) = (a.text(), b.text());
        let joined = keys(unit, &all(&w, &over(unit, &format!("({at}) ({bt})"))));
        let staged = keys(unit, &all(&w, &over(unit, &format!("({at}) | where ({bt})"))));
        let by_key: std::collections::BTreeMap<String, Row> = every_row(&rows, unit).into_iter().collect();
        let filtered: Vec<String> = keys(unit, &all(&w, &over(unit, &format!("({at})"))))
            .into_iter()
            .filter(|k| reference::truth(&rows, &b, by_key[k]) == Truth::Yes)
            .collect();
        if a.ranks() && b.ranks() {
            prop_assert_eq!(sorted(joined.clone()), sorted(staged), "{} | where {}", at, bt);
        } else {
            prop_assert_eq!(&joined, &staged, "{} | where {}", at, bt);
        }
        if b.ranks() {
            prop_assert_eq!(sorted(joined), sorted(filtered), "{} then {}", at, bt);
        } else {
            prop_assert_eq!(joined, filtered, "{} then {}", at, bt);
        }
    }

    /// `U where C`, `U where NOT C` and the rows `C` is unknown for
    /// partition `U`; for a field condition, `F:unknown` selects exactly
    /// the rows neither `C` nor `NOT C` returns.
    #[test]
    fn a_condition_its_negation_and_its_unknowns_partition_the_rows(
        rows in rows(),
        (unit, c) in unit_and_cond(),
    ) {
        let w = build(&rows);
        let text = c.text();
        let everything = key_set(unit, &all(&w, &over(unit, "")));
        let yes = key_set(unit, &all(&w, &over(unit, &format!("({text})"))));
        let no = key_set(unit, &all(&w, &over(unit, &format!("NOT ({text})"))));
        prop_assert!(yes.is_disjoint(&no), "{}: both {:?}", text, yes.intersection(&no).collect::<Vec<_>>());
        let neither: BTreeSet<String> = everything.difference(&yes).filter(|k| !no.contains(*k)).cloned().collect();
        let unknown: BTreeSet<String> = every_row(&rows, unit)
            .into_iter()
            .filter(|(_, row)| reference::truth(&rows, &c, *row) == Truth::Unknown)
            .map(|(key, _)| key)
            .collect();
        prop_assert_eq!(&neither, &unknown, "{}", text);
        if let Cond::Leaf(leaf) = &c
            && let Some(form) = leaf.unknown_form()
        {
            let selected = key_set(unit, &all(&w, &over(unit, &form.text())));
            prop_assert_eq!(&selected, &neither, "{} and {}", text, form.text());
        }
    }

    /// Attaching rows never changes which parent rows are returned, their
    /// order or their count; a parent with no matching children carries an
    /// empty list, and the lists hold the children the window selects.
    #[test]
    fn attached_rows_never_change_the_parent_rows(rows in rows(), a in cond(Unit::Sessions)) {
        let w = build(&rows);
        let query = over(Unit::Sessions, &format!("({})", a.text()));
        let plain = all(&w, &query);
        for (attach, field) in [
            ("with messages(id)[role:assistant]", "messages"),
            ("with tool_calls(tool)[last:1]", "tool_calls"),
            ("with messages(id)[role:system]", "messages"),
        ] {
            let attached = all(&w, &format!("{query} {attach}"));
            prop_assert_eq!(keys(Unit::Sessions, &attached), keys(Unit::Sessions, &plain), "{}", attach);
            prop_assert_eq!(&attached["count"], &plain["count"], "{}", attach);
            for row in result_rows(&attached) {
                let id = row["id"].as_str().unwrap_or_default();
                let i = (0..rows.sessions.len())
                    .find(|i| reference::session_id(&rows, *i) == id)
                    .unwrap();
                let s = &rows.sessions[i];
                let children: Vec<Value> = row[field].as_array().cloned().unwrap_or_default();
                let want: Vec<Value> = match attach {
                    "with messages(id)[role:assistant]" => {
                        let mut found: Vec<(Option<i64>, usize)> = s
                            .messages
                            .iter()
                            .enumerate()
                            .filter(|(_, m)| m.path != Some(false) && m.speaker.0 == "assistant")
                            .map(|(j, m)| (m.at, j))
                            .collect();
                        found.sort_by_key(|(at, j)| (at.is_none(), *at, *j));
                        found
                            .into_iter()
                            .map(|(_, j)| json!({"id": reference::message_id(i, j)}))
                            .collect()
                    }
                    "with messages(id)[role:system]" => vec![],
                    _ => {
                        let mut calls: Vec<(Option<i64>, String, usize, &str)> = vec![];
                        for (j, m) in s.messages.iter().enumerate() {
                            if m.path == Some(false) {
                                continue;
                            }
                            for (k, c) in m.calls.iter().enumerate() {
                                calls.push((c.at, reference::message_id(i, j), k, c.tool.0));
                            }
                        }
                        calls.sort_by_key(|(at, m, k, _)| (at.is_none(), *at, m.clone(), *k));
                        calls.last().map(|c| json!({"tool": c.3})).into_iter().collect()
                    }
                };
                prop_assert_eq!(children, want, "{} {}", attach, id);
            }
        }
    }

}

proptest! {
    #![proptest_config(config(12))]

    /// Pages of every size, followed by cursor, concatenate to the unpaged
    /// answer, and every page carries the same count.
    #[test]
    fn pages_of_every_size_concatenate_to_the_whole_answer(
        rows in rows(),
        s in cond(Unit::Sessions),
        m in cond(Unit::Messages),
        c in cond(Unit::ToolCalls),
    ) {
        let w = build(&rows);
        let queries = [
            over(Unit::Sessions, &format!("({})", s.text())),
            over(Unit::Sessions, &format!("({}) | sort by cost", s.text())),
            over(Unit::Sessions, "| sort by started desc"),
            over(Unit::Sessions, "| sort by random(7)"),
            over(Unit::Sessions, "| group by origin | count"),
            over(Unit::Sessions, "with tool_calls(tool)[last:1]"),
            over(Unit::Messages, &format!("({})", m.text())),
            over(Unit::Messages, "parser"),
            over(Unit::Messages, "| sort by model desc"),
            over(Unit::ToolCalls, &format!("({})", c.text())),
            over(Unit::ToolCalls, "| group by outcome | count"),
        ];
        for query in &queries {
            let whole = all(&w, query);
            let want = result_rows(&whole);
            for size in 1..=want.len() as u64 + 1 {
                let mut got = vec![];
                let mut cursor: Option<String> = None;
                loop {
                    let mut request = ask(query);
                    request.page_size = Some(size);
                    request.cursor = cursor.as_deref();
                    let page = run_q(&w, request)
                        .unwrap_or_else(|f| panic!("{query} by {size}: {} {}", f.kind, f.message));
                    prop_assert_eq!(&page["count"], &whole["count"], "{} by {}", query, size);
                    got.extend(result_rows(&page));
                    cursor = page["next"].as_str().map(str::to_owned);
                    if cursor.is_none() {
                        break;
                    }
                }
                prop_assert_eq!(&got, &want, "{} by {}", query, size);
            }
        }
    }
}

proptest! {
    #![proptest_config(config(48))]

    /// A ranking orders rows and never changes which match or how many:
    /// words ranked by relevance against the same rows sorted by time, and
    /// `near:` with vectors for every text, for some, and disabled. A
    /// count after `near:` needs no ranking, so it computes no similarity.
    #[test]
    fn a_ranking_never_changes_which_rows_match(
        rows in rows(),
        m in cond(Unit::Messages),
        s in cond(Unit::Sessions),
        word in prop::sample::select(WORDS.to_vec()),
    ) {
        let w = build(&rows);
        let ranked = all(&w, &over(Unit::Messages, &format!("({}) {word}", m.text())));
        let timed = all(&w, &over(Unit::Messages, &format!("({}) {word} | sort by at", m.text())));
        prop_assert_eq!(key_set(Unit::Messages, &ranked), key_set(Unit::Messages, &timed));
        prop_assert_eq!(&ranked["count"], &timed["count"]);

        let query = over(Unit::Sessions, &format!("({})", s.text()));
        let plain = all(&w, &query);
        let ranked_text = format!("{query} near:\"how dates are parsed\"");
        // Stored vectors stay, so each variant adds to the one before.
        for vectors in ["none", "some texts", "every text"] {
            let similarity = match vectors {
                "every text" => Some(vectors_for(&w, |_| true)),
                "some texts" => Some(vectors_for(&w, |session| session.ends_with('0') || session.ends_with('2'))),
                _ => None,
            };
            let near = |_: &str| -> Result<std::sync::Arc<polylogue_store::vectors::Similarity>, String> {
                Ok(std::sync::Arc::clone(similarity.as_ref().unwrap()))
            };
            let mut request = ask(&ranked_text);
            request.page_size = Some(1 << 40);
            request.embeddings = similarity.is_some();
            if similarity.is_some() {
                request.near = Some(&near);
            }
            let v = run_q(&w, request).unwrap();
            prop_assert_eq!(key_set(Unit::Sessions, &v), key_set(Unit::Sessions, &plain), "{}", vectors);
            prop_assert_eq!(&v["count"], &plain["count"], "{}", vectors);
            if similarity.is_some() {
                prop_assert!(
                    v["unknown"]["near"]
                        .as_u64()
                        .is_some_and(|n| Some(n) <= plain["count"].as_u64()),
                    "{}", v
                );
            }
        }
        let never = |_: &str| -> Result<std::sync::Arc<polylogue_store::vectors::Similarity>, String> {
            panic!("a count computed a similarity")
        };
        let count_text = format!("{ranked_text} | agg count");
        let mut request = ask(&count_text);
        request.embeddings = true;
        request.near = Some(&never);
        let counted = run_q(&w, request).unwrap();
        prop_assert_eq!(counted["rows"][0]["count"].as_u64(), Some(result_rows(&plain).len() as u64));
        prop_assert_eq!(&counted["outcome"], "rows");
    }

    /// The matches of `exists`, `seq(...)` and `| matches` never depend on
    /// the order the rows were written in: the same rows written with
    /// sessions and messages shuffled give the same answers.
    #[test]
    fn matches_do_not_depend_on_the_order_rows_are_read(
        rows in rows(),
        a in cond(Unit::ToolCalls),
        b in cond(Unit::ToolCalls),
        shuffles in prop::collection::vec(any::<u64>(), 3),
    ) {
        let (a, b) = (a.text(), b.text());
        let queries = [
            over(Unit::Sessions, &format!("exists tool_call({a})")),
            over(Unit::Sessions, &format!("exists tool_call({a}) | matches")),
            over(Unit::Sessions, &format!("seq(({a}) -> ({b}))")),
            over(Unit::Sessions, &format!("seq(({a}) -> ({b})) | matches")),
            over(Unit::Sessions, &format!("seq(({a}) -> [next] ({b})) | matches")),
            over(Unit::ToolCalls, &format!("({a})")),
        ];
        let answers = |w: &TestArchive| -> Vec<Value> {
            queries.iter().map(|q| all(w, q)["rows"].clone()).collect()
        };
        let first = answers(&build(&rows));
        for seed in shuffles {
            let mut order = written_order(&rows);
            shuffle(&mut order, seed);
            prop_assert_eq!(&answers(&write(&rows, &order)), &first, "shuffled by {}", seed);
        }
    }

    /// Values in a query stay values: search words, quoted values and IDs
    /// full of SQL and search syntax either run or are refused as a bad
    /// query, never break the statement, never return more rows than the
    /// unit holds, and leave the archive as it was.
    #[test]
    fn query_values_never_become_statements(payload in payload()) {
        let w = morrow();
        let quoted = format!("\"{}\"", payload.replace('\\', "\\\\").replace('"', "\\\""));
        let before = all(&w, "messages where active:* parser")["count"].clone();
        for (unit, text) in [
            ("sessions", quoted.clone()),
            ("sessions", format!("kind:* title:{quoted}")),
            ("sessions", format!("kind:* repo:{quoted}")),
            ("sessions", format!("kind:* origin:{quoted}")),
            ("sessions", format!("kind:* id:{quoted}")),
            ("sessions", format!("kind:* tag:{quoted}")),
            ("messages", format!("messages where active:* {quoted}")),
            ("messages", format!("messages where active:* model:{quoted}")),
            ("tool_calls", format!("tool_calls where action:* command:{quoted}")),
            ("tool_calls", format!("tool_calls where action:* output:{quoted}")),
            ("notes", format!("notes where value:{quoted}")),
        ] {
            let total = match unit {
                "sessions" => all(&w, "kind:*")["count"].as_u64(),
                "messages" => all(&w, "messages where active:*")["count"].as_u64(),
                "tool_calls" => all(&w, "tool_calls where action:*")["count"].as_u64(),
                _ => all(&w, "notes")["count"].as_u64(),
            };
            match run_q(&w, ask(&text)) {
                Ok(v) => prop_assert!(v["count"].as_u64() <= total, "{}: {}", text, v["count"]),
                Err(f) => prop_assert!(
                    !matches!(f.kind.as_str(), "store" | "internal"),
                    "{}: {} {}", text, f.kind, f.message
                ),
            }
        }
        prop_assert_eq!(&all(&w, "messages where active:* parser")["count"], &before);
    }
}

/// SQL and full-text syntax, quotes, control and direction characters,
/// and arbitrary text.
fn payload() -> impl Strategy<Value = String> {
    prop_oneof![
        prop::sample::select(vec![
            "'; DROP TABLE session; --",
            "' OR '1'='1",
            "\" OR \"\"=\"",
            "%' OR '1'='1' --",
            "); DELETE FROM message WHERE ('1'='1",
            "1; SELECT * FROM sqlite_master --",
            "UNION SELECT name FROM sqlite_master",
            "*",
            "**",
            "AND OR NOT",
            "NEAR(foo)",
            "\"hello",
            "foo\"bar",
            "foo*bar",
            "foo:bar",
            "(((",
            ")))",
            "[]",
            "{}",
            "\u{0}",
            "\u{0}\u{1}\u{2}",
            "\u{202e}\u{202d}",
            "%00",
            "_%_",
            "../../etc/passwd",
            "\\",
            "\\\"",
            "\n; DROP TABLE message;",
            "parser*",
            "-parser",
            "^parser",
            "parser NOT fix",
        ])
        .prop_map(str::to_owned),
        "(?s).{0,40}",
    ]
}

/// A seeded shuffle, so a failure replays from its seed.
fn shuffle<T>(items: &mut [T], seed: u64) {
    let mut state = seed | 1;
    for i in (1..items.len()).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let j = (state % (i as u64 + 1)) as usize;
        items.swap(i, j);
    }
}

/// Similarities for `near:` from vectors stored for the prose of the
/// sessions `has_vector` accepts (by session ID); other texts have none.
pub(super) fn vectors_for(
    w: &TestArchive,
    has_vector: impl Fn(&str) -> bool,
) -> std::sync::Arc<polylogue_store::vectors::Similarity> {
    let pool = w.fx.reads().unwrap();
    let reader = pool.reader().unwrap();
    let texts = reader
        .query(
            "SELECT DISTINCT hex(t.hash) AS h, s.id AS session FROM text t \
             JOIN block b ON b.text_id = t.id JOIN session s ON s.pk = b.session_pk \
             WHERE b.kind = 'text'",
            &[],
        )
        .unwrap();
    let mut vectors: Vec<polylogue_store::vectors::Vector> = vec![];
    for (n, row) in texts.iter().enumerate() {
        if !has_vector(row["session"].as_str().unwrap_or_default()) {
            continue;
        }
        let hex = row["h"].as_str().unwrap();
        let text_hash: Vec<u8> = (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect();
        if vectors.iter().any(|v| v.text_hash == text_hash) {
            continue;
        }
        let angle = n as f32;
        vectors.push(polylogue_store::vectors::Vector {
            text_hash,
            chunk: 0,
            values: vec![angle.cos(), angle.sin()],
        });
    }
    let key = "local/test/2/document/unit";
    reader.vectors_writer().unwrap().put(key, &vectors).unwrap();
    let loaded = std::sync::Arc::new(reader.vectors().unwrap().load(key).unwrap());
    std::sync::Arc::new(loaded.similarity(&[1.0, 0.0]).unwrap())
}

#[test]
fn an_error_only_on_an_abandoned_branch_does_not_satisfy_session_has() {
    let rows = Rows {
        sessions: vec![SessionRow {
            origin: "claude-code",
            subagent: false,
            repo: None,
            messages: vec![MessageRow {
                speaker: ("assistant", "model"),
                model: None,
                cost: None,
                at: None,
                text: Some("discarded".into()),
                path: Some(false),
                calls: vec![CallRow {
                    tool: ("Bash", "shell"),
                    command: Some("cargo test"),
                    outcome: "error",
                    exit_code: Some(101),
                    at: None,
                }],
            }],
        }],
    };
    let archive = build(&rows);
    for name in ["errors", "tool_calls"] {
        let condition = Cond::Leaf(Leaf::Has(name));
        assert!(reference::answer(&rows, Unit::Sessions, &condition).is_empty());
        assert_eq!(
            all(&archive, &format!("sessions where {}", condition.text()))["count"],
            0
        );
    }
}
